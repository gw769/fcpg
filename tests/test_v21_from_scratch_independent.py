"""Independent initialization, freeze, checkpoint and source-admission checks.

Only synthetic CPU fixtures are optimized here. These tests do not establish
training quality, execute a robot, or read a previously trained checkpoint.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_temporal_contract import PROGRESS_TRAINING_POLICY
from fcpg.v21_training import (
    V21Schedule, V21Trainer, load_checkpoint, rng_state, save_checkpoint, synthetic_chain,
)


@pytest.fixture(autouse=True)
def cpu_rng_isolation():
    old_threads = torch.get_num_threads()
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    torch.set_num_threads(2)
    random.seed(937); np.random.seed(937); torch.manual_seed(937)
    yield
    random.setstate(python_state); np.random.set_state(numpy_state); torch.set_rng_state(torch_state)
    torch.set_num_threads(old_threads)


def state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def same_tree(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype and np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same_tree(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('temporal_policy', ['legacy', PROGRESS_TRAINING_POLICY])
def test_actual_random_core_freezes_opposite_modules_through_all_four_stages(tmp_path, monkeypatch, temporal_policy):
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('fresh initialization read saved weights'))
    vocabulary = TaskVocabulary.build(['pick red block'])
    core = V21ResearchCore(V21Config(len(vocabulary.words), object_relative=True))
    core.set_loss_scale([2., 3., 4., 5., 6., 7.])
    trainer = V21Trainer(core, vocabulary, tmp_path, microbatch=2, precision='fp32',
                         temporal_feature_policy=temporal_policy)
    original_id = id(core)
    initial = state(core)
    fixture = synthetic_chain(vocabulary, batch=2, images=True)
    protected = deepcopy(fixture)
    assert all(value.dtype == torch.float32 for value in core.parameters())
    assert not any('dino' in name.lower() for name, _ in core.named_parameters())
    frozen_lewm = None
    for stage in ('lewm', 'actor', 'dense', 'fish'):
        before = state(core)
        optimizer = trainer.configure(stage)
        selected = {id(value) for group in optimizer.param_groups for value in group['params']}
        admitted = ([*core.lewm.parameters(), *trainer.projector.parameters()] if stage == 'lewm'
                    else [*core.actor.parameters(), *core.reader.parameters()])
        assert selected == {id(value) for value in admitted}
        batch = replace(fixture, features=None) if stage == 'lewm' else fixture
        result = trainer.step([batch], optimizer, stage)
        assert id(trainer.core) == original_id
        assert np.isfinite([result['loss'], result['gradient_norm']]).all()
        changed = [name for name, value in core.state_dict().items() if not torch.equal(value, before[name])]
        prefixes = ('lewm.',) if stage == 'lewm' else ('reader.', 'actor.')
        assert changed and all(name.startswith(prefixes) for name in changed)
        for prefix in prefixes:
            assert any(name.startswith(prefix) for name in changed)
        for name, value in core.named_parameters():
            if not name.startswith(prefixes):
                assert not value.requires_grad and value.grad is None
        if stage == 'lewm':
            assert all(torch.equal(value, initial[name]) for name, value in core.state_dict().items()
                       if name.startswith(('reader.', 'actor.')))
            frozen_lewm = state(core.lewm)
        else:
            same_tree(state(core.lewm), frozen_lewm)
    for name in ('q', 'commands', 'paths', 'readback_offsets', 'features', 'view_valid', 'endpoint_times'):
        assert torch.equal(getattr(fixture, name), getattr(protected, name)), name


class TinyCore(nn.Module):
    """Small state-compatible fixture for genuine optimizer/checkpoint recovery."""
    def __init__(self, vocabulary_size):
        super().__init__()
        self.config = V21Config(vocabulary_size)
        self.reader = nn.Linear(3, 3)
        self.actor = nn.Linear(3, 3)
        self.lewm = nn.Linear(3, 3)
        self.register_buffer('q_loss_scale', torch.ones(6))


class StochasticTrainer(V21Trainer):
    def step(self, batches, optimizer, stage):
        optimizer.zero_grad(set_to_none=True)
        # Exercise all three restored RNG streams, actual AdamW state, and the
        # LeWM projector. The model-scale freeze check above uses the real loss.
        factor = .2+random.random()+float(np.random.uniform())+float(torch.rand(()))
        parameters = [value for group in optimizer.param_groups for value in group['params']]
        loss = sum((value.reshape(-1)[:3]-factor).square().sum() for value in parameters)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        return dict(loss=float(loss.detach()), gradient_norm=float(norm), chains=1,
                    pair_exposures=4, microbatch=1, precision='fp32')


class InterruptedFixture(Exception):
    pass


def tiny_run(output, vocabulary, schedule, *, interrupt_stage=None, init_seed=937, load_log=None, monkeypatch=None):
    random.seed(init_seed); np.random.seed(init_seed); torch.manual_seed(init_seed)
    trainer = StochasticTrainer(TinyCore(len(vocabulary.words)), vocabulary, output, schedule,
                                microbatch=1, smoke=True, precision='fp32')
    dense_entry = None
    for stage in ('lewm', 'actor', 'dense', 'fish'):
        if stage == 'dense':
            dense_entry = state(trainer.core)
        def prepare(index, current=stage):
            if current == interrupt_stage and index == 1:
                raise InterruptedFixture(current)
            return [None]
        trainer.run(stage, range(2), prepare, 2)
    return trainer, dense_entry


@pytest.mark.parametrize('interrupted_stage', ['lewm', 'dense'])
def test_completed_actor_and_partial_stage_restore_model_optimizer_projector_and_rng(tmp_path, monkeypatch, interrupted_stage):
    import fcpg.v21_training as training
    vocabulary = TaskVocabulary.build(['pick red block'])
    schedule = V21Schedule(archive_passes=1, dense_passes=1, fish_updates=2,
                           chains_per_batch=2, fish_chains_per_batch=2, checkpoint_every=1)
    baseline_dir, resumed_dir = tmp_path/'continuous', tmp_path/'interrupted'
    baseline, baseline_dense_entry = tiny_run(baseline_dir, vocabulary, schedule)
    expected_rng = rng_state('cpu')
    with pytest.raises(InterruptedFixture):
        tiny_run(resumed_dir, vocabulary, schedule, interrupt_stage=interrupted_stage)
    incomplete = torch.load(resumed_dir/interrupted_stage/'latest.pt', map_location='cpu', weights_only=False)
    assert incomplete['step'] == 1 and incomplete['complete'] is False and incomplete['optimizer']['state']
    if interrupted_stage == 'lewm':
        assert incomplete['projector'] is not None
    loaded = []
    original_load = training.load_checkpoint
    def observed_load(path, *args, **options):
        assert path.is_relative_to(resumed_dir)
        assert options['restore_random'] is True and options['optimizer'] is not None
        loaded.append((path.parent.name, path.name))
        return original_load(path, *args, **options)
    monkeypatch.setattr(training, 'load_checkpoint', observed_load)
    resumed, resumed_dense_entry = tiny_run(resumed_dir, vocabulary, schedule, init_seed=19)
    same_tree(state(resumed.core), state(baseline.core))
    same_tree(state(resumed.projector), state(baseline.projector))
    same_tree(rng_state('cpu'), expected_rng)
    same_tree(resumed_dense_entry, baseline_dense_entry)
    if interrupted_stage == 'dense':
        assert loaded == [('lewm', 'final.pt'), ('actor', 'final.pt'), ('dense', 'latest.pt')]
        lewm_final = torch.load(resumed_dir/'lewm/final.pt', map_location='cpu', weights_only=False)
        assert any(not torch.equal(resumed_dense_entry[name], value) for name, value in lewm_final['model'].items()
                   if name.startswith(('actor.', 'reader.')))
    else:
        assert loaded == [('lewm', 'latest.pt')]
    for stage in ('lewm', 'actor', 'dense', 'fish'):
        expected = torch.load(baseline_dir/stage/'final.pt', map_location='cpu', weights_only=False)
        actual = torch.load(resumed_dir/stage/'final.pt', map_location='cpu', weights_only=False)
        same_tree(actual['model'], expected['model'])
        same_tree(actual['optimizer'], expected['optimizer'])
        same_tree(actual['rng'], expected['rng'])
        same_tree(actual['projector'], expected['projector'])
        assert actual['step'] == actual['metrics']['updates'] == 2
        assert actual['complete'] and actual['metrics']['pair_exposures'] == 8


def test_canary_checkpoint_cannot_be_loaded_as_formal(tmp_path):
    vocabulary = TaskVocabulary.build(['pick red block'])
    core = TinyCore(len(vocabulary.words))
    schedule = V21Schedule(fish_updates=2)
    path = tmp_path/'canary/fish/final.pt'
    save_checkpoint(path, core, vocabulary, schedule, stage='fish', step=2,
                    complete=True, smoke=True)
    with pytest.raises(ValueError, match='smoke checkpoint cannot enter a formal run'):
        load_checkpoint(path, TinyCore(len(vocabulary.words)), vocabulary, schedule, stage='fish')


def test_fresh_entry_initialization_is_the_default_random_model_not_a_saved_model(monkeypatch):
    from scripts import train_v21_from_scratch as fresh
    vocabulary = TaskVocabulary.build(['pick red block'])
    scale = np.asarray([2., 3., 4., 5., 6., 7.], np.float32)
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('fresh entry loaded saved weights'))
    actual = fresh.initialize_core(vocabulary, scale)
    torch.manual_seed(20000)
    expected = V21ResearchCore(V21Config(len(vocabulary.words), object_relative=True))
    expected.set_loss_scale(scale)
    same_tree(state(actual), state(expected))
    repeated = fresh.initialize_core(vocabulary, scale)
    same_tree(state(actual), state(repeated))
    different = fresh.initialize_core(vocabulary, scale, seed=20001)
    for name in ('lewm', 'actor', 'reader'):
        assert any(not torch.equal(value, state(getattr(different, name))[key])
                   for key, value in state(getattr(actual, name)).items())
    assert actual.config.object_relative is True
    assert all(value.dtype == torch.float32 and value.device.type == 'cpu' for value in actual.parameters())


class TrainingRowsOnly:
    def __init__(self, values, admitted):
        self.values, self.admitted, self.reads = values, set(admitted.tolist()), []
        self.shape, self.dtype = values.shape, values.dtype

    def __getitem__(self, index):
        rows = index[0] if isinstance(index, tuple) else index
        ids = np.asarray(np.arange(len(self.values))[rows]).reshape(-1)
        assert set(ids.tolist()) <= self.admitted, 'normalization read development rows'
        self.reads.extend(ids.tolist())
        return self.values[index]

    def __array__(self, *args, **kwargs):
        pytest.fail('normalization materialized the complete TRAIN+DEV bank')


def test_loss_scale_reads_only_explicit_original_training_rows():
    from scripts import train_v21_from_scratch as fresh
    train = np.asarray([2, 5, 7], np.int64)
    generator = np.random.default_rng(941)
    commands = generator.normal(0., .025, (9, 4, 5, 6)).astype(np.float32)
    paths = generator.normal(0., .025, (9, 4, 6, 6)).astype(np.float32)
    development = np.setdiff1d(np.arange(9), train)
    commands[development] = np.nan; paths[development] = np.nan
    command_bank, readback_bank = TrainingRowsOnly(commands, train), TrainingRowsOnly(paths, train)
    original = SimpleNamespace(completed_commands=command_bank, measured_state_path=readback_bank,
        rows=9, split=np.where(np.isin(np.arange(9), train), 0, 1).astype(np.uint8))
    source = SimpleNamespace(archive=SimpleNamespace(archive=original, training_rows=np.arange(9), rows=9))
    actual = fresh.training_loss_scale(source, train, chunk_rows=2)
    # The recorded companion stores calibrated positions divided by 100.
    u = (commands[train]*np.float32(100.)).astype(np.float64)
    q = (paths[train, :, 0]*np.float32(100.)).astype(np.float64)
    previous = np.concatenate((q[:, :, None], u[:, :, :-1]), axis=2)
    expected = np.std((u-previous).reshape(-1, 6), axis=0, ddof=0).astype(np.float32)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert sorted(command_bank.reads) == sorted(readback_bank.reads) == train.tolist()


def test_split_derivation_ignores_legacy_all_rows_alias_and_maps_dense_global_ids():
    from scripts import train_v21_from_scratch as fresh
    split = np.asarray([1, 1, 0, 1, 0, 1, 0, 0], np.uint8)
    episodes = np.repeat(np.arange(4), 61)
    def validate_frames(ids):
        assert not np.any(episodes[ids] == 2), 'sealed fish frames accessed'
    source = SimpleNamespace(counts=dict(archive=8, dense=4, fish=3),
        archive=SimpleNamespace(archive=SimpleNamespace(split=split), training_rows=np.arange(8),
            row_endpoint_index=np.arange(40).reshape(8, 5), endpoint_episode_index=np.repeat(np.arange(8), 5)),
        dense_rows=np.asarray([7, 0, 4, 1]),
        fish=SimpleNamespace(starts=np.asarray([0, 61, 183]), episodes=episodes, validate_frames=validate_frames))
    actual = fresh.derive_splits(source)
    np.testing.assert_array_equal(actual['train']['archive'], [2, 4, 6, 7])
    np.testing.assert_array_equal(actual['development']['archive'], [0, 1, 3, 5])
    np.testing.assert_array_equal(actual['train']['dense'], [7, 4])
    np.testing.assert_array_equal(actual['development']['dense'], [0, 1])
    np.testing.assert_array_equal(actual['dense_train_ordinals'], [0, 2])
    np.testing.assert_array_equal(actual['train']['fish'], [0, 1])
    np.testing.assert_array_equal(actual['development']['fish'], [2])
    assert set(episodes[actual['fish_required_endpoints']]) == {0, 1, 3}
    source.archive.endpoint_episode_index[:5] = 2
    with pytest.raises(ValueError, match='TRAIN/DEV episodes overlap'):
        fresh.derive_splits(source)


@dataclass
class InputProbe:
    q: torch.Tensor
    view_valid: torch.Tensor
    features: torch.Tensor | None


def test_every_stage_maps_train_ordinals_before_io_including_mixed_fish_archive():
    from scripts import train_v21_from_scratch as fresh
    populations = dict(archive=np.asarray([4, 8, 12, 16, 20, 24, 28, 32, 36]),
                       dense=np.asarray([12, 28, 36]), fish=np.asarray([1, 3, 5]))
    schedule = V21Schedule(archive_passes=2, dense_passes=3, fish_updates=3,
                           chains_per_batch=4, fish_chains_per_batch=4)
    split, fish_episodes = np.ones(40, np.uint8), np.full(6, 3)
    split[populations['archive']] = 0; fish_episodes[populations['fish']] = [0, 1, 0]
    calls = []
    def chain(rows, kind, *, images):
        ids = np.asarray(rows)
        assert np.isin(ids, populations[kind]).all(), 'ordinal or DEV row reached source.chain'
        calls.append((kind, ids.copy(), images))
        return InputProbe(torch.zeros(len(ids), 5, 6), torch.ones(len(ids), 5, 2, dtype=torch.bool),
            torch.ones(len(ids), 5, 2, 1, 1, dtype=torch.float16) if images else None)
    source = SimpleNamespace(counts=dict(archive=len(split), dense=len(populations['dense']), fish=len(fish_episodes)),
        archive=SimpleNamespace(archive=SimpleNamespace(split=split)),
        fish=SimpleNamespace(starts=np.arange(6), episodes=fish_episodes), chain=chain)
    for stage in ('lewm', 'actor', 'dense', 'fish'):
        requests = list(fresh.stage_requests(stage, populations, schedule))
        assert len(requests) == fresh.stage_steps(stage, populations, schedule)
        for request in requests:
            before = len(calls)
            batches = fresh.prepare_request(source, request)
            assert len(calls)-before == len(request['groups']) == len(batches)
            for (kind, ids), (seen_kind, seen_ids, images), batch in zip(request['groups'], calls[before:], batches):
                assert kind == seen_kind and images == (stage != 'lewm')
                np.testing.assert_array_equal(ids, seen_ids)
                assert np.isin(ids, populations['dense' if stage == 'dense' else kind]).all()
                if images:
                    assert batch.features.dtype == torch.float32
        if stage != 'fish':
            flattened = np.concatenate([request['groups'][0][1] for request in requests])
            bank = populations['dense' if stage == 'dense' else 'archive']
            passes = schedule.dense_passes if stage == 'dense' else schedule.archive_passes
            np.testing.assert_array_equal(np.sort(flattened), np.sort(np.tile(bank, passes)))
            assert all(request['groups'][0][0] == 'archive' for request in requests)
        else:
            assert all([kind for kind, _ in request['groups']] == ['fish', 'archive'] for request in requests)
    before = len(calls)
    for kind, ids in (('archive', np.asarray([0])), ('fish', np.asarray([0]))):
        with pytest.raises(ValueError, match='DEV'):
            fresh.prepare_request(source, dict(stage='fish', variant=0, groups=[(kind, ids)]))
    assert len(calls) == before


def test_original_train_populations_keep_declared_stage_budgets():
    from scripts import train_v21_from_scratch as fresh
    populations = dict(archive=np.arange(272000), dense=np.arange(17000), fish=np.arange(381))
    schedule = V21Schedule()
    assert [fresh.stage_steps(stage, populations, schedule) for stage in fresh.STAGES] == [17000, 17000, 266, 4000]


@pytest.mark.parametrize('dense_updates_before_interrupt', [0, 1])
def test_new_entry_resumes_deepest_actor_or_dense_without_reloading_random_actor(
        tmp_path, monkeypatch, dense_updates_before_interrupt):
    from scripts import train_v21_from_scratch as fresh
    import fcpg.v21_training as training
    vocabulary = TaskVocabulary.build(['pick red block'])
    schedule = V21Schedule(archive_passes=1, dense_passes=1, fish_updates=2,
        chains_per_batch=2, fish_chains_per_batch=2, checkpoint_every=1)
    populations = dict(archive=np.asarray([4, 8, 12, 16]), dense=np.asarray([8, 16]), fish=np.asarray([1, 3]))
    monkeypatch.setattr(V21Trainer, 'step', StochasticTrainer.step)
    interrupted = {'enabled': False, 'dense_calls': 0}
    def prepare(source, request):
        if interrupted['enabled'] and request['stage'] == 'dense':
            if interrupted['dense_calls'] == dense_updates_before_interrupt:
                raise InterruptedFixture('dense')
            interrupted['dense_calls'] += 1
        return [None]
    monkeypatch.setattr(fresh, 'prepare_request', prepare)
    def build(directory, seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        return fresh.ScratchTrainer(TinyCore(len(vocabulary.words)), vocabulary, directory, schedule,
                                    microbatch=1, smoke=True, precision='fp32')
    baseline = build(tmp_path/'baseline', 947)
    fresh.run_stages(baseline, None, populations, canary=True)
    baseline_rng = rng_state('cpu')
    partial = build(tmp_path/'partial', 947)
    interrupted['enabled'] = True
    with pytest.raises(InterruptedFixture):
        fresh.run_stages(partial, None, populations, canary=True)
    interrupted['enabled'] = False
    actual_loads = []
    original = training.load_checkpoint
    def loaded(path, *args, **options):
        actual_loads.append((path.parent.name, path.name))
        return original(path, *args, **options)
    monkeypatch.setattr(training, 'load_checkpoint', loaded)
    resumed = build(tmp_path/'partial', 29)
    fresh.run_stages(resumed, None, populations, canary=True)
    assert actual_loads == ([('actor', 'final.pt')] if dense_updates_before_interrupt == 0 else [('dense', 'latest.pt')])
    same_tree(state(resumed.core), state(baseline.core))
    same_tree(rng_state('cpu'), baseline_rng)
    for stage in ('actor', 'dense', 'fish'):
        a = torch.load(tmp_path/'baseline'/stage/'final.pt', map_location='cpu', weights_only=False)
        b = torch.load(tmp_path/'partial'/stage/'final.pt', map_location='cpu', weights_only=False)
        same_tree(a['model'], b['model']); same_tree(a['optimizer'], b['optimizer']); same_tree(a['rng'], b['rng'])


@pytest.mark.parametrize('canary', [False, True])
def test_main_records_exact_original_split_profile_and_canary_separation(tmp_path, monkeypatch, canary):
    from scripts import train_v21_from_scratch as fresh
    vocabulary = TaskVocabulary.build(['pick red block'])
    original = {'fish_wrist_root': '/fixture/original-wrist'}
    mapping_path = tmp_path/'source-map.json'
    mapping_path.write_text(json.dumps({'source_paths_original': original}))
    source = SimpleNamespace(vocabulary=vocabulary, fish=object())
    splits = dict(train=dict(archive=np.arange(48000, 320000), dense=np.arange(48000, 65000), fish=np.arange(381)),
        development=dict(archive=np.arange(48000), dense=np.arange(3000), fish=np.arange(381, 486)),
        fish_required_endpoints=np.asarray([0, 15, 30]))
    monkeypatch.setattr(fresh, 'resolve_source_paths', lambda *a: ({'fish_wrist_root': '/fixture/wrist'}, {'fixture': True}))
    monkeypatch.setattr(fresh, 'V21Sources', lambda *a, **kw: source)
    monkeypatch.setattr(fresh, 'derive_splits', lambda s: splits)
    monkeypatch.setattr(fresh, 'validate_manifest', lambda *a, **kw: {'fixture': 'body-v2'})
    def scale(s, rows):
        assert s is source
        np.testing.assert_array_equal(rows, splits['train']['archive'])
        return np.asarray([2., 3., 4., 5., 6., 7.], np.float32)
    monkeypatch.setattr(fresh, 'training_loss_scale', scale)
    monkeypatch.setattr(fresh, 'pin_source_map', lambda *a: None)
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('fresh main read a checkpoint'))
    initialization_ids, training_ids = [], []
    actual_init = fresh.initialize_core
    def initialize(*a, **kw):
        core = actual_init(*a, **kw)
        initialization_ids.append(id(core))
        return core
    monkeypatch.setattr(fresh, 'initialize_core', initialize)
    class TrainerProbe:
        def __init__(self, core, vocab, output, schedule, **options):
            self.core, self.output = core, output
            assert options['smoke'] is canary and options['precision'] == 'fp32'
            assert options['temporal_feature_policy'] == PROGRESS_TRAINING_POLICY
    monkeypatch.setattr(fresh, 'ScratchTrainer', TrainerProbe)
    def run(trainer, s, populations, **options):
        assert s is source and populations is splits['train'] and options['canary'] is canary
        training_ids.append(id(trainer.core))
        assert set(dict(trainer.core.named_children())) >= {'reader', 'actor', 'lewm'}
    monkeypatch.setattr(fresh, 'run_stages', run)
    output = tmp_path/'new-run'
    args = ['--source-map', str(mapping_path), '--body-table', str(tmp_path/'body.npy'),
        '--body-manifest', str(tmp_path/'body.json'), '--output', str(output), '--threads', '2', '--microbatch', '2']
    if canary:
        args.append('--two-step-canary')
    assert fresh.main(args) == 0
    assert initialization_ids == training_ids and len(training_ids) == 1
    config = json.loads((output/'config.json').read_text())
    assert config['training_data_contract'] == dict(schema='v21-original-source-split-v1',
        archive_train_rows=272000, archive_development_rows=48000,
        dense_train_rows=17000, dense_development_rows=3000,
        fish_train_episodes=[0, 1], fish_development_episodes=[3], fish_test_episode_used=False)
    assert config['normalization_train_chains'] == 272000 and config['normalization_command_increments'] == 5440000
    assert config['smoke'] is canary and config['two_step_real_data_canary'] is canary
    assert config['external_weights_loaded'] is False and config['external_checkpoint_reads'] == 0
    assert config['visual_backbone'] == 'frozen-dinov2-vits14-real-cache'
    with np.load(output/'training-rows.npz', allow_pickle=False) as saved:
        for role in ('train', 'development'):
            for kind, rows in splits[role].items():
                np.testing.assert_array_equal(saved[role+'_'+kind], rows)
    if canary:
        formal_resume = [arg for arg in args if arg != '--two-step-canary']+['--resume']
        with pytest.raises(ValueError, match='exact scratch run configuration'):
            fresh.main(formal_resume)
