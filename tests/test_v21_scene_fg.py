from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.v21_scene_fg import (FGBatch, pair_loss, fg_batch, spatial_supervision,
    enable_fg_readout, FG_ARCHITECTURE, audit_spatial_gradients, memory_fg_batch)
from fcpg.v21_scene_lewm import V21SceneLeWM, FP32_RESIDUAL_SCENE
from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_native_horizon import core_from_parent


def test_unordered_column_pair_does_not_force_false_fg_identity():
    xy = torch.tensor([[[[.2,.4],[.5,.4]],[[.2,.5],[.6,.5]]]])
    valid = torch.ones(1,2,dtype=torch.bool)
    labels = FGBatch(xy, valid, torch.tensor([[True,False]]))
    p = xy.clone(); p[:,1] = p[:,1].flip(-2)
    loss, report = pair_loss(p, labels)
    assert loss == 0 and report[1]['mean_pixel_error'] == 0
    loss, _ = pair_loss(p, FGBatch(xy, valid, valid))
    assert loss > 0


def test_missing_nan_pseudolabel_never_reaches_loss_or_gradients():
    class Table:
        def gather(self, ids):
            assert ids.tolist() == [7]
            xy = np.zeros((1,2,2,2), np.float32); xy[:,1] = np.nan
            return dict(fg_px=xy, pair_available=np.array([[True,False]]),
                current_image_valid=np.ones((1,2),bool), ordered_identity_available=np.ones((1,2),bool))
    labels = fg_batch(Table(), np.array([[7,-1]]), device='cpu', image_valid=torch.ones(1,2,2,dtype=torch.bool))
    assert labels.valid.tolist() == [[[True,False],[False,False]]]
    p = torch.ones(1,2,2,2,2,requires_grad=True)
    pair_loss(p, labels)[0].backward()
    assert torch.isfinite(p.grad).all() and not p.grad[:,1].any()


class IdentityFGTable:
    def gather(self, ids):
        return dict(fg_px=np.broadcast_to(np.asarray(ids)[:, None, None, None],
                        (len(ids), 2, 2, 2)).copy().astype(np.float32),
            pair_available=np.ones((len(ids), 2), bool),
            current_image_valid=np.ones((len(ids), 2), bool),
            ordered_identity_available=np.ones((len(ids), 2), bool))


def retained_memory_case(available):
    from fcpg.v21_memory import VisualMemory, VisualObservation
    memory = VisualMemory.empty(1)
    times = torch.tensor([[[1., .9], [2., 1.9]]], dtype=torch.float64)
    masks = torch.tensor([available], dtype=torch.bool)
    for i in range(2):
        observation = VisualObservation(torch.full((1,2,256,384), float(i)),
            torch.zeros(1,2,6), times[:,i], masks[:,i])
        memory, _ = memory.write(observation, torch.tensor([float(i+1)], dtype=torch.float64))
    return np.array([[10,20]]), times, masks, memory


def test_missing_column_labels_its_retained_image_not_current_pose():
    ids, times, masks, memory = retained_memory_case([[True,True], [True,False]])
    old = fg_batch(IdentityFGTable(), ids[:, -1], device='cpu', image_valid=memory.valid[:,:,-1])
    # Reproduce the old bug: both heads are asked to read endpoint20, although
    # Column's actual input still contains exposure10.
    assert old.xy[0,1,0,0]*224 == 20
    fixed = memory_fg_batch(IdentityFGTable(), ids, times, masks, memory)
    torch.testing.assert_close(fixed.xy[0,:,0,0]*224, torch.tensor([20.,10.]))
    assert fixed.valid.tolist() == [[True,True]]


def test_fresh_dual_view_labels_are_bit_exact_with_previous_binding():
    ids, times, masks, memory = retained_memory_case([[True,True], [True,True]])
    old = fg_batch(IdentityFGTable(), ids[:,-1], device='cpu', image_valid=memory.valid[:,:,-1])
    fixed = memory_fg_batch(IdentityFGTable(), ids, times, masks, memory)
    for name in ('xy','valid','ordered'):
        assert torch.equal(getattr(old,name), getattr(fixed,name))


def test_cold_camera_is_not_labelled_from_the_other_view():
    ids, times, masks, memory = retained_memory_case([[True,False], [True,False]])
    fixed = memory_fg_batch(IdentityFGTable(), ids, times, masks, memory)
    assert fixed.valid.tolist() == [[True,False]]
    assert not fixed.xy[:,1].any()


def test_retained_exposure_cannot_be_replaced_by_nearest_or_future_label():
    ids, times, masks, memory = retained_memory_case([[True,True], [True,False]])
    times = times.clone(); times[0,0,1] -= .00001
    with pytest.raises(ValueError, match='absent from the factual'):
        memory_fg_batch(IdentityFGTable(), ids, times, masks, memory)


def test_dense40_spatial_loss_updates_current_future_and_predictor_paths():
    torch.set_num_threads(2)
    model = V21SceneLeWM(normalization='layernorm', scene_prediction_mode=FP32_RESIDUAL_SCENE)
    enable_fg_readout(model)
    current = torch.randn(2,2,3,384,requires_grad=True)
    target = torch.randn(2,40,2,384,requires_grad=True)
    predicted = torch.randn(2,40,2,384,requires_grad=True)
    valid = torch.ones(2,40,2,dtype=torch.bool)
    response = SimpleNamespace(observation_scene=current, predicted_scene=predicted,
        observation_valid=torch.ones(2,2,3,dtype=torch.bool))
    future = SimpleNamespace(query_indices=torch.arange(40))
    now = FGBatch(torch.rand(2,2,2,2), valid[:,0], valid[:,0])
    later = FGBatch(torch.rand(2,40,2,2,2), valid, valid)
    loss, report = spatial_supervision(model, response, target, future, now, later)
    audit = audit_spatial_gradients(loss, model, response, target, future, now, later)
    assert audit['query_indices_one_based'] == list(range(1,41))
    assert audit['paths']['future_prediction']['valid_pairs_by_time_and_camera'] == [[2,2]]*40
    loss.backward()
    for x in (current,target,predicted):
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert (target.grad.abs().sum((0,2,3)) > 0).all()
    assert (predicted.grad.abs().sum((0,2,3)) > 0).all()
    assert report['scene_fg_prediction'][0]['pairs'] == 80


def test_cold_camera_has_no_fake_persistence_but_keeps_all_future_supervision():
    torch.set_num_threads(2)
    model = V21SceneLeWM(normalization='layernorm')
    enable_fg_readout(model)
    current = torch.zeros(2,2,3,384,requires_grad=True)
    target = torch.randn(2,40,2,384,requires_grad=True)
    predicted = torch.randn(2,40,2,384,requires_grad=True)
    observed = torch.ones(2,2,3,dtype=torch.bool); observed[:,1] = False
    response = SimpleNamespace(observation_scene=current, predicted_scene=predicted,
                               observation_valid=observed)
    future = SimpleNamespace(query_indices=torch.arange(40))
    valid = torch.ones(2,40,2,dtype=torch.bool)
    now = FGBatch(torch.rand(2,2,2,2), observed[:,:,-1], observed[:,:,-1])
    later = FGBatch(torch.rand(2,40,2,2,2), valid, valid)
    loss, report = spatial_supervision(model, response, target, future, now, later)
    expected_loss = sum(pair_loss(p, labels)[0] for p, labels in (
        (model.fg_readout(current[:,:,-1]), now),
        (model.fg_readout(target), later), (model.fg_readout(predicted), later)))/3
    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
    assert [r['pairs'] for r in report['scene_fg_prediction']] == [80,80]
    for key in ('scene_fg_persistence', 'scene_fg_prediction_same_mask'):
        assert [r['pairs'] for r in report[key]] == [80,0]
        assert report[key][1]['mean_pixel_error'] is None
    loss.backward()
    assert (target.grad[:,:,1].abs().sum((0,2)) > 0).all()
    assert (predicted.grad[:,:,1].abs().sum((0,2)) > 0).all()


def test_spatial_checkpoint_needs_explicit_version_and_exact_head_restore():
    torch.set_num_threads(2)
    core = V21ResearchCore(V21Config(4,prediction_horizon=40))
    core.lewm = V21SceneLeWM(normalization='layernorm',scene_prediction_mode=FP32_RESIDUAL_SCENE)
    before = {k:v.clone() for k,v in core.lewm.state_dict().items()}
    enable_fg_readout(core.lewm)
    for k,v in before.items(): torch.testing.assert_close(v,core.lewm.state_dict()[k],rtol=0,atol=0)
    payload = dict(config=asdict(core.config),model=core.state_dict(),scene_lewm_architecture=core.lewm.architecture,
        scene_normalization='layernorm',scene_prediction_mode=FP32_RESIDUAL_SCENE,scene_fg_architecture=FG_ARCHITECTURE)
    restored = core_from_parent(payload)
    for k,v in payload['model'].items(): torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
    del payload['scene_fg_architecture']
    with pytest.raises(ValueError,match='explicit spatial architecture'): core_from_parent(payload)


def test_spatial_training_cannot_accidentally_run_old_sparse_or_unfrozen_recipe(tmp_path):
    from scripts.train_v21_horizon40 import main
    with pytest.raises(ValueError,match='F/G spatial experiment requires'):
        main(['--parent', str(tmp_path/'none.pt'),'--source-config',str(tmp_path/'none.json'),
            '--output',str(tmp_path/'out'),'--fish-fg-table',str(tmp_path/'labels')])
    assert not (tmp_path/'out').exists()
