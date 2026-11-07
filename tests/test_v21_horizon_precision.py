from copy import deepcopy

import pytest
import torch

from fcpg.v21_data import TaskVocabulary
from fcpg.v21_horizon_precision import compute_context
from fcpg.v21_training import rng_state
from scripts.train_v21_horizon40 import (
    initialize_scratch40, restore_training_checkpoint, validate_precision_continuation,
    validate_resource_continuation)


def test_amp_history_no_grad_does_not_detach_later_training_weights(monkeypatch):
    # CPU autocast exercises the same cached-cast bug without reserving a GPU.
    autocast = torch.autocast
    monkeypatch.setattr(torch, 'autocast', lambda device, **kwargs: autocast('cpu', **kwargs))
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 4))
    x = torch.randn(3, 8)
    with compute_context('cuda', 'bf16'):
        with torch.no_grad():
            model(x)
        prediction = model(x)
    assert prediction.dtype == torch.bfloat16
    prediction.float().square().mean().backward()
    assert all(p.dtype == torch.float32 and p.grad is not None
               and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters())


def protocol():
    return dict(initialization='random-all-three-components-native40',
        initial_module_sha256={'reader': 'r', 'actor': 'a', 'lewm': 'l'},
        fish_train_episodes=[0, 1, 3], device={'cuda_visible_devices': 'GPU-1'},
        schedule=dict(batch_size=128, learning_rate=1e-4, feature_bank_gib=48., tf32=True))


def test_precision_continuation_is_explicit_and_legacy_fp32_is_recognized():
    previous = protocol()
    current = deepcopy(previous)
    current['schedule']['precision'] = 'bf16'
    validate_precision_continuation(previous, current)
    with pytest.raises(ValueError, match='cannot change'):
        validate_resource_continuation(previous, current)
    validate_precision_continuation(current, previous)
    with pytest.raises(ValueError, match='only change'):
        validate_precision_continuation(previous, previous)


@pytest.mark.parametrize('section,key,value', [
    ('schedule', 'batch_size', 256), ('schedule', 'learning_rate', 3e-5),
    ('schedule', 'feature_bank_gib', 24.), ('schedule', 'tf32', False),
    ('device', 'cuda_visible_devices', 'GPU-0'), ('initial_module_sha256', 'reader', 'new'),
])
def test_precision_continuation_rejects_unrelated_recipe_changes(section, key, value):
    previous = protocol()
    current = deepcopy(previous)
    current['schedule']['precision'] = 'bf16'
    current[section][key] = value
    with pytest.raises(ValueError, match='only change'):
        validate_precision_continuation(previous, current)


def test_resume_restores_optimizer_rng_and_accumulated_temporal_coverage():
    torch.set_num_threads(2)
    vocab = TaskVocabulary.build(['pick fish'])
    original = initialize_scratch40(vocab, torch.ones(6), 21004, 'cpu')
    optimizer = torch.optim.AdamW(original.actor.parameters(), lr=1e-4)
    sum(p.square().mean() for p in original.actor.parameters()).backward()
    optimizer.step()
    original.temporal_feature_coverage['actor_calls'] = 123
    original.temporal_feature_coverage['rows'] = 15744
    saved = deepcopy(dict(model=original.state_dict(), optimizer=optimizer.state_dict(),
        temporal_training_contract=original.temporal_training_contract,
        temporal_feature_coverage=original.temporal_feature_coverage, rng=rng_state(torch.device('cpu'))))
    expected_rng = torch.rand(5)
    restored = initialize_scratch40(vocab, torch.ones(6), 123, 'cpu')
    new_optimizer = torch.optim.AdamW(restored.actor.parameters(), lr=9e-5)
    restore_training_checkpoint(restored, saved, new_optimizer)
    assert torch.equal(torch.rand(5), expected_rng)
    assert restored.temporal_feature_coverage == original.temporal_feature_coverage
    restored.temporal_feature_coverage['rows'] += 1
    assert saved['temporal_feature_coverage']['rows'] == 15744
    for key, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    assert new_optimizer.param_groups[0]['lr'] == 1e-4
    for index, state in optimizer.state_dict()['state'].items():
        for key, value in state.items():
            assert torch.equal(value, new_optimizer.state_dict()['state'][index][key])
    changed = deepcopy(saved)
    changed['temporal_training_contract']['policy'] = 'different'
    with pytest.raises(ValueError, match='policy differs'):
        restore_training_checkpoint(restored, changed)
