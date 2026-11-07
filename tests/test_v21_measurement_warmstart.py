from copy import deepcopy
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.v21_data import TaskVocabulary
from fcpg.v21_training import synthetic_chain
from fcpg.v21_memory import VisualMemory, VisualObservation
from fcpg.v21_scene_fg import memory_fg_batch
from fcpg.v21_scene_lewm import scene_prediction_loss
from scripts.train_v21_horizon40 import (resolve_measurement_dropout, augment_measurements,
    saved_measurement_dropout, canonical_protocol, validate_resource_continuation,
    validate_training_semantics, TRAINING_MODE_POLICY, SIGREG_FEATURE_SOURCE, main)


def test_defaults_preserve_old_scratch_and_warmstart_and_resume_cannot_change_masks():
    assert resolve_measurement_dropout(None) == 'none'
    assert resolve_measurement_dropout(None, from_scratch=True) == 'burst-cold-v1'
    for scratch in (False, True):
        old = dict(schedule=dict(from_scratch=scratch), device={},
            training_mode_policy=TRAINING_MODE_POLICY, sigreg_feature_source=SIGREG_FEATURE_SOURCE)
        mode = resolve_measurement_dropout(None, from_scratch=scratch)
        new = deepcopy(old)
        new['measurement_dropout'] = new['schedule']['measurement_dropout'] = mode
        assert canonical_protocol(new) == canonical_protocol(old)
        validate_training_semantics(old, full_patch_lewm=True, measurement_dropout=mode)
        changed = 'none' if scratch else 'burst-cold-v1'
        with pytest.raises(ValueError, match='measurement dropout changed'):
            validate_training_semantics(old, full_patch_lewm=True, measurement_dropout=changed)
        new['measurement_dropout'] = new['schedule']['measurement_dropout'] = changed
        with pytest.raises(ValueError, match='cannot change'):
            validate_resource_continuation(old, new)
    with pytest.raises(ValueError, match='disagrees'):
        saved_measurement_dropout(dict(measurement_dropout='burst-cold-v1', schedule={}))


def test_augmentation_requires_explicit_frozen_warmstart_before_opening_files(tmp_path):
    with pytest.raises(ValueError, match='frozen Actor/FCPG'):
        main(['--parent', str(tmp_path/'missing.pt'), '--source-config', str(tmp_path/'missing.json'),
              '--output', str(tmp_path/'unused'), '--measurement-dropout', 'burst-cold-v1'])
    assert not (tmp_path/'unused').exists()


@pytest.fixture
def chain():
    torch.set_num_threads(2)
    batch = synthetic_chain(TaskVocabulary.build(['pick red block']), batch=10)
    return replace(batch, capture_times=batch.endpoint_times[:, :, None].expand(-1, -1, 2).clone(),
                   capture_q=batch.q[:, :, None].expand(-1, -1, 2, -1).clone())


def test_only_delivery_masks_change_without_rng_or_source_mutation(chain):
    original_mask = chain.view_valid.clone()
    chain.view_valid[0, 0, 1] = False
    original_mask[0, 0, 1] = False
    rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    for variant in range(10):
        actual = augment_measurements(chain, np.arange(10), protocol='burst-cold-v1', variant=variant)
        assert not (actual.view_valid & ~chain.view_valid).any()
        assert not torch.equal(actual.view_valid, chain.view_valid)
        for field in fields(chain):
            if field.name != 'view_valid':
                assert getattr(actual, field.name) is getattr(chain, field.name)
    assert torch.equal(chain.view_valid, original_mask)
    assert torch.equal(rng, torch.get_rng_state())
    assert np.array_equal(numpy_rng[1], np.random.get_state()[1])


@pytest.mark.parametrize('kwargs', [dict(protocol='none'), dict(images=False), dict(evaluation=True)])
def test_evaluation_and_disabled_inputs_never_get_masked_even_with_grad(chain, kwargs):
    args = dict(protocol='burst-cold-v1', variant=0, **{k: v for k, v in kwargs.items() if k != 'protocol'})
    args['protocol'] = kwargs.get('protocol', args['protocol'])
    assert augment_measurements(chain, np.arange(10), **args) is chain
    with torch.no_grad():
        assert augment_measurements(chain, np.arange(10), protocol='burst-cold-v1', variant=1) is chain


def test_retention_cold_start_and_actual_recovery_bind_their_own_fg(chain):
    class Labels:
        def gather(self, ids):
            return dict(fg_px=np.broadcast_to(ids[:, None, None, None], (len(ids), 2, 2, 2)).astype('f'),
                pair_available=np.ones((len(ids), 2), bool), current_image_valid=np.ones((len(ids), 2), bool),
                ordered_identity_available=np.ones((len(ids), 2), bool))
    masked = augment_measurements(chain, np.arange(10), protocol='burst-cold-v1', variant=0)
    memory = VisualMemory.empty(10)
    ids = np.broadcast_to(np.arange(5)*10+10, (10, 5)).copy()
    for h in range(4):
        memory, _ = memory.write(VisualObservation(masked.features[:, h], masked.capture_q[:, h],
            masked.capture_times[:, h], masked.view_valid[:, h]), masked.endpoint_times[:, h])
        labels = memory_fg_batch(Labels(), ids[:, :h+1], masked.capture_times[:, :h+1],
                                 masked.view_valid[:, :h+1], memory)
        # Case4 keeps the actual H0 Column while Wrist receives fresh exposures.
        torch.testing.assert_close(labels.xy[4, :, 0, 0]*224, torch.tensor([float(10+h*10), 10.]))
        assert labels.valid[6].tolist() == [True, False]  # cold Column
        # Case8 retains both H0 images, then uses actual H3 reacquisition.
        torch.testing.assert_close(labels.xy[8, :, 0, 0]*224,
            torch.full((2,), 40. if h == 3 else 10.))
        assert not (memory.image_times[memory.valid] >
            masked.endpoint_times[:, h, None, None].expand_as(memory.image_times)[memory.valid]).any()


def test_missing_input_view_does_not_remove_any_of_40_actual_future_targets():
    target = torch.ones(2, 40, 2, 8, requires_grad=True)
    predicted = torch.zeros_like(target, requires_grad=True)
    current_valid = torch.ones(2, 2, 3, dtype=torch.bool); current_valid[:, 1] = False
    response = SimpleNamespace(predicted_scene=predicted, hidden=predicted,
        observation_scene=torch.zeros(2, 2, 3, 8), observation_valid=current_valid)
    future = SimpleNamespace(valid=torch.ones(2, 40, 2, dtype=torch.bool),
        query_indices=torch.arange(40), regularization_slots=lambda: torch.arange(4, 40, 5))
    loss, _, report = scene_prediction_loss(None, response, future, encoded_target=target)
    loss.backward()
    assert report['scene_targets'] == 160
    assert (target.grad.abs().sum((0, 3)) > 0).all()
    assert (predicted.grad.abs().sum((0, 3)) > 0).all()
