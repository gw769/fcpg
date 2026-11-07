"""Actual-network CPU stage/resume tests; data fixtures are synthetic, no robot IO."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.v21_data import TaskVocabulary
from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_training import V21Schedule, synthetic_chain
from fcpg.v21_temporal_contract import PROGRESS_TRAINING_POLICY
from scripts import train_v21_from_scratch as fit


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.set_num_threads(previous)


def vocabulary():
    return TaskVocabulary.build(['pick red block'])


def test_default_random_initializers_create_all_modules_before_any_checkpoint_read(monkeypatch):
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('scratch initialization read a checkpoint'))
    vocab = vocabulary()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20000)
        reference = V21ResearchCore(V21Config(len(vocab.words),seed=20000,object_relative=True))
        reference.set_loss_scale([1.,2.,3.,4.,5.,6.])
    actual = fit.initialize_core(vocab,[1.,2.,3.,4.,5.,6.])
    assert actual.config.object_relative and actual.actor.uses_object_relative
    assert any(name.startswith('actor.object_relative.') for name,_ in actual.named_parameters())
    assert set(actual.state_dict()) == set(reference.state_dict())
    for name,value in actual.state_dict().items():
        assert torch.equal(value,reference.state_dict()[name]),name
    assert all(value.dtype == torch.float32 and value.device.type == 'cpu' for value in actual.parameters())


@pytest.mark.parametrize('flag',['--lewm-from','--model-root','--checkpoint','--warm-start'])
def test_external_weight_arguments_are_not_supported(flag):
    with pytest.raises(SystemExit):
        fit.parser().parse_args(['--source-map','source','--body-table','table','--body-manifest','manifest',
            '--output','new',flag,'old'])


def test_cpu_device_configuration_is_backward_compatible_without_cuda_queries(monkeypatch):
    for name in ('is_available','device_count','get_device_properties','set_per_process_memory_fraction','set_device'):
        monkeypatch.setattr(torch.cuda,name,lambda *a,**kw:pytest.fail('CPU entry queried CUDA'))
    assert fit.configure_device()==(torch.device('cpu'),{})
    args=fit.parser().parse_args(['--source-map','source','--body-table','table',
        '--body-manifest','manifest','--output','new'])
    assert args.device=='cpu' and args.gpu_memory_limit_gib==8.


@pytest.mark.parametrize('name',['cuda','cuda:1','mps'])
def test_only_cpu_or_one_logical_cuda_device_is_admitted(name):
    with pytest.raises(ValueError,match='cpu or logical cuda:0'):
        fit.configure_device(name)


@pytest.mark.parametrize('device,microbatch,admitted',[
    ('cpu',32,True),('cpu',33,False),('cuda:0',64,True),('cuda:0',65,False)])
def test_microbatch_limits_preserve_cpu_and_admit_full_effective_cuda_batch(tmp_path,monkeypatch,device,microbatch,admitted):
    class AdmissionReached(Exception):
        pass
    def reached(*args,**kwargs):
        raise AdmissionReached
    monkeypatch.setattr(fit,'configure_device',reached)
    args=['--source-map','source','--body-table','table','--body-manifest','manifest',
        '--output',str(tmp_path/'run'),'--device',device,'--microbatch',str(microbatch)]
    with pytest.raises(AdmissionReached if admitted else ValueError):
        fit.main(args)
    assert fit.V21Schedule().chains_per_batch==64


def validation_fixture():
    core=torch.nn.Module()
    core.register_parameter('trainable',torch.nn.Parameter(torch.tensor([1.,2.])))
    core.register_parameter('frozen',torch.nn.Parameter(torch.tensor([3.,4.]),requires_grad=False))
    core.register_buffer('buffer',torch.tensor([5.,6.]))
    names={'frozen','buffer'}
    frozen={name:value.clone() for name,value in core.state_dict().items() if name in names}
    return core,frozen,names


def test_frozen_and_finite_validation_uses_one_scalar_read_and_all_tensors(monkeypatch):
    core,frozen,names=validation_fixture()
    original=torch.Tensor.item
    scalar_reads=[]
    def item(value,*args,**kwargs):
        scalar_reads.append(value.shape)
        return original(value,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'item',item)
    fit.verify_frozen_model_state(core,frozen,names)
    assert scalar_reads==[torch.Size([])]


@pytest.mark.parametrize('fault,expected',[
    ('frozen','frozen state changed: frozen'),
    ('buffer','frozen state changed: buffer'),
    ('nan','nonfinite model parameter: trainable'),
    ('inf','nonfinite model parameter: trainable'),
    ('shape','frozen state changed: buffer'),
    ('gradient','frozen parameter got gradients: frozen')])
def test_aggregated_validation_rejects_each_original_invariant_with_layer_name(fault,expected):
    core,frozen,names=validation_fixture()
    with torch.no_grad():
        if fault in ('frozen','buffer'):
            getattr(core,fault)[0]+=1
        elif fault in ('nan','inf'):
            core.trainable[-1]=float(fault)
        elif fault=='shape':
            core.buffer=core.buffer[None]
        else:
            core.frozen.grad=torch.ones_like(core.frozen)
    with pytest.raises(ValueError,match=expected):
        fit.verify_frozen_model_state(core,frozen,names)


def test_source_split_rejects_cross_episode_and_train_dev_episode_overlap():
    source=SimpleNamespace(counts=dict(archive=4,dense=2,fish=3),dense_rows=np.array([0,3]),
        archive=SimpleNamespace(archive=SimpleNamespace(split=np.array([0,1,0,1])),
            row_endpoint_index=np.arange(20).reshape(4,5),endpoint_episode_index=np.repeat(np.arange(4),5)),
        fish=SimpleNamespace(starts=np.array([0,61,122]),episodes=np.repeat([0,1,3],61),validate_frames=lambda x:None))
    good=fit.derive_splits(source)
    assert good['train']['archive'].tolist()==[0,2]
    assert good['train']['dense'].tolist()==[0]
    assert good['development']['dense'].tolist()==[3]
    assert good['train']['fish'].tolist()==[0,1]
    source.archive.endpoint_episode_index[1]=9
    with pytest.raises(ValueError,match='crosses episode'):
        fit.derive_splits(source)
    source.archive.endpoint_episode_index[:5]=1
    with pytest.raises(ValueError,match='episodes overlap'):
        fit.derive_splits(source)


@pytest.mark.parametrize('kind,rows',[('archive',[1]),('fish',[2]),('archive',[-1]),('fish',[3]),('archive',[.1])])
def test_dev_or_invalid_original_rows_are_rejected_before_payload_loading(kind,rows):
    source=SimpleNamespace(counts=dict(archive=3,fish=3),
        archive=SimpleNamespace(archive=SimpleNamespace(split=np.array([0,1,0]))),
        fish=SimpleNamespace(starts=np.array([0,1,2]),episodes=np.array([0,1,3])),
        chain=lambda *a,**kw:pytest.fail('unadmitted source row was opened'))
    with pytest.raises(ValueError):
        fit.prepare_request(source,dict(stage='fish',variant=0,groups=[(kind,np.asarray(rows))]))


def test_configuration_and_original_split_are_immutable_without_hashes(tmp_path):
    output=tmp_path/'new'
    config={'recipe':fit.RECIPE,'schedule':asdict(V21Schedule()),'smoke':False}
    splits=dict(train=dict(archive=np.array([0,2]),dense=np.array([2]),fish=np.array([0,1])),
        development=dict(archive=np.array([1]),dense=np.array([1]),fish=np.array([2])))
    fit.configure_output(output,config,splits,resume=False)
    fit.configure_output(output,config,splits,resume=True)
    with pytest.raises(ValueError,match='empty'):
        fit.configure_output(output,config,splits,resume=False)
    with pytest.raises(ValueError,match='configuration'):
        fit.configure_output(output,dict(config,smoke=True),splits,resume=True)
    changed=deepcopy(splits);changed['train']['archive']=np.array([2,0])
    with pytest.raises(ValueError,match='identities changed'):
        fit.configure_output(output,config,changed,resume=True)
    assert 'hash' not in (output/'config.json').read_text()


def test_filtered_sampler_covers_only_train_and_canary_is_two_steps_per_stage():
    populations=dict(archive=np.array([3,10,17,25,49]),dense=np.array([10,49]),fish=np.array([1,4,5]))
    schedule=V21Schedule(archive_passes=2,dense_passes=1,fish_updates=3,chains_per_batch=2,fish_chains_per_batch=4)
    for stage,kind in (('lewm','archive'),('actor','archive'),('dense','dense')):
        requests=list(fit.stage_requests(stage,populations,schedule))
        assert len(requests)==fit.stage_steps(stage,populations,schedule)
        actual=np.concatenate([item['groups'][0][1] for item in requests])
        expected=np.tile(populations[kind],2 if kind=='archive' else 1)
        np.testing.assert_array_equal(np.sort(actual),np.sort(expected))
        assert all(item['groups'][0][0]=='archive' for item in requests)
    for request in fit.stage_requests('fish',populations,schedule):
        assert [kind for kind,_ in request['groups']]==['fish','archive']
        for kind,rows in request['groups']:
            assert len(rows)==2 and np.isin(rows,populations[kind]).all()
    for stage in fit.STAGES:
        assert len(list(fit.stage_requests(stage,populations,schedule,canary=True)))==2


def actual_network_fixture(output):
    vocab=vocabulary()
    source=SimpleNamespace(counts=dict(archive=3,fish=3),
        archive=SimpleNamespace(archive=SimpleNamespace(split=np.array([0,1,0]))),
        fish=SimpleNamespace(starts=np.array([0,1,2]),episodes=np.array([0,1,3])),calls=[])
    def chain(rows,kind,*,images):
        source.calls.append((kind,rows.tolist(),images))
        value=synthetic_chain(vocab,batch=len(rows),images=images)
        return replace(value,object_relative=torch.tensor([1.,.8,.1,-.1,.2,.1,1.,0.]).expand(len(rows),5,8).clone())
    source.chain=chain
    schedule=V21Schedule(archive_passes=1,dense_passes=1,fish_updates=2,chains_per_batch=1,
        fish_chains_per_batch=2,checkpoint_every=1,sigreg_weight=0.)
    core=fit.initialize_core(vocab,[2.]*6)
    trainer=fit.ScratchTrainer(core,vocab,output,schedule,microbatch=1,smoke=True,precision='fp32',
        temporal_feature_policy=PROGRESS_TRAINING_POLICY)
    return trainer,source,dict(archive=np.array([0,2]),dense=np.array([2]),fish=np.array([0,1]))


def test_actual_network_partial_dense_resume_matches_uninterrupted_all_stages(tmp_path):
    direct,source,populations=actual_network_fixture(tmp_path/'direct')
    fit.run_stages(direct,source,populations,canary=True)
    direct_state={name:value.clone() for name,value in direct.core.state_dict().items()}
    interrupted,source,populations=actual_network_fixture(tmp_path/'resumed')
    original=interrupted.step
    calls=0
    def step(batches,optimizer,stage):
        nonlocal calls
        if stage=='dense':
            calls+=1
            if calls==2:raise RuntimeError('fixture interruption before second dense update')
        return original(batches,optimizer,stage)
    interrupted.step=step
    with pytest.raises(RuntimeError,match='fixture interruption'):
        fit.run_stages(interrupted,source,populations,canary=True)
    assert (tmp_path/'resumed/actor/final.pt').is_file()
    assert (tmp_path/'resumed/dense/latest.pt').is_file()
    assert not (tmp_path/'resumed/dense/final.pt').exists()
    resumed,source,populations=actual_network_fixture(tmp_path/'resumed')
    called=[]
    original_run=resumed.run
    def run(stage,*args,**kwargs):
        called.append(stage)
        return original_run(stage,*args,**kwargs)
    resumed.run=run
    fit.run_stages(resumed,source,populations,canary=True)
    assert called==['dense','fish']
    for name,value in resumed.core.state_dict().items():
        assert torch.equal(value,direct_state[name]),name
    for stage in fit.STAGES:
        checkpoint=torch.load(tmp_path/'resumed'/stage/'final.pt',weights_only=False)
        assert checkpoint['smoke'] and checkpoint['complete'] and checkpoint['step']==2
        assert checkpoint['metrics']['last']['frozen_tensors_equal'] is True
        assert checkpoint['metrics']['last']['external_checkpoint_reads']==0


def test_complete_latest_crash_window_is_repaired_before_next_stage(tmp_path):
    trainer,source,populations=actual_network_fixture(tmp_path/'run')
    fit.run_stages(trainer,source,populations,through='lewm',canary=True)
    (tmp_path/'run/lewm/final.pt').unlink()  # Simulate crash between the two final writes.
    resumed,source,populations=actual_network_fixture(tmp_path/'run')
    completed=[]
    def callback(stage):
        assert (tmp_path/'run'/stage/'final.pt').is_file()
        completed.append(stage)
    fit.run_stages(resumed,source,populations,through='actor',canary=True,after_stage=callback)
    assert completed==['lewm','actor'], 'completed-resume stage must still issue its post-save callback'
    assert (tmp_path/'run/lewm/final.pt').is_file()
    assert (tmp_path/'run/actor/final.pt').is_file()
    assert fit.resume_stage(tmp_path/'run',trainer.schedule,canary=True)==1


def test_post_save_development_does_not_change_subsequent_actual_network_training(tmp_path):
    from scripts.evaluate_v21_scratch_development import evaluate_development

    direct,source,populations=actual_network_fixture(tmp_path/'direct')
    fit.run_stages(direct,source,populations,canary=True)
    expected={name:value.clone() for name,value in direct.core.state_dict().items()}
    expected_coverage=deepcopy(direct.core.temporal_feature_coverage)

    evaluated,source,populations=actual_network_fixture(tmp_path/'evaluated')
    source.fish.starts=np.array([0,61,122])
    source.fish.episodes=np.repeat([0,1,3],61)
    source.fish.frame_indices=np.tile(np.arange(61),3)
    development=dict(archive=np.array([1]),dense=np.array([1]),fish=np.array([2]))
    reports=[]
    def callback(stage):
        assert (evaluated.output/stage/'final.pt').is_file()
        report=evaluate_development(evaluated.core,source,development,
            stage=stage,microbatch=1,archive_limit=1)
        assert report['stage']==stage and report['optimizer_steps']==0
        assert report['weights_gradients_rng_preserved'] is True
        reports.append(report)

    # Directly exercise the post-save seam with an isolated two-update fixture;
    # the production canary entry intentionally never installs DEV callbacks.
    fit.run_stages(evaluated,source,populations,canary=True,after_stage=callback)
    assert [report['stage'] for report in reports]==list(fit.STAGES)
    for name,value in evaluated.core.state_dict().items():
        assert torch.equal(value,expected[name]),name
    assert evaluated.core.temporal_feature_coverage==expected_coverage
    for name,value in evaluated.projector.state_dict().items():
        assert torch.equal(value,direct.projector.state_dict()[name]),name
