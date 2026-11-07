from dataclasses import asdict
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_horizon_training import frame_grid, horizon_targets, masked_mean
from fcpg.v21_native_horizon import initialize_native40
from fcpg.v21_data import TaskVocabulary
from fcpg.v21_training import save_checkpoint, load_checkpoint, V21Schedule
from scripts.train_v21_horizon40 import ordered_groups, select_fish_training_episodes, validate_resource_continuation


def source(fps=30, length=200):
    q = np.arange(length * 6, dtype=np.float32).reshape(length, 6)
    archive = SimpleNamespace(row_endpoint_index=np.array([[0, 1, 2, 3, 4]]),
        endpoint_episode_index=np.zeros(5, dtype=np.int64),
        endpoint_local_frame_index=np.rint(np.arange(5)*fps*.5).astype(np.int64),
        fps=np.array([fps]))
    return SimpleNamespace(counts={'archive': 1}, archive=archive,
        offsets=np.array([0, length]), raw_q=q, raw_actions=q + .25)


def test_full60hz_write_trace_is_preserved_and_labels_use_actual_frame_ids():
    data = source(60, 400)
    target = horizon_targets(data, [0], 'archive', 3)
    assert target.trace.positions.shape == (1, 240, 6)
    assert target.valid.all()
    np.testing.assert_array_equal(target.trace.positions[0], data.raw_actions[90:330])
    np.testing.assert_array_equal(target.commands[0], data.raw_actions[np.arange(90, 330, 6)])
    np.testing.assert_array_equal(target.readbacks[0], data.raw_q[np.arange(96, 331, 6)])
    assert float(target.readback_offsets[0, -1]) == 4.


def test_source_episode_tail_is_masked_without_reading_the_next_episode():
    data = source(30, 100)
    # A following episode exists in storage but must not enter the first episode.
    data.raw_q = np.concatenate((data.raw_q, np.full((200, 6), 999999., np.float32)))
    data.raw_actions = data.raw_q + .25
    target = horizon_targets(data, [0], 'archive', 3)
    assert int(target.valid.sum()) == 18
    assert target.trace.positions.shape == (1, 54, 6)
    assert torch.equal(target.readbacks[:, 18:], torch.zeros_like(target.readbacks[:, 18:]))
    assert target.readbacks.max() < 1000
    values = torch.ones(1, 40, 6)
    values[:, 18:] = 999999.
    assert masked_mean(values, target.valid) == 1.


def test_15hz_offsets_remain_real_recorded_times():
    frames, times = frame_grid([15.])
    assert np.all(np.diff(frames[0]) > 0)
    np.testing.assert_array_equal(times * 15, frames)
    assert times[0, 1] == 2/15 and times[0, -1] == 4.


def test_short_tail_retains_only_actual_prefix_labels():
    target = horizon_targets(source(60, 103), [0], 'archive', 3)
    assert int(target.valid.sum()) == 2
    assert target.trace.positions.shape == (1, 12, 6)
    with pytest.raises(ValueError, match='no actual future'):
        horizon_targets(source(60, 96), [0], 'archive', 3)


def test_fish_test_episode_is_rejected_before_any_payload_access():
    class NoPayload:
        def __getitem__(self, ids):
            pytest.fail('held-out motion payload was read')
    def validate(ids):
        raise ValueError('fish episode 2 is sealed')
    data = SimpleNamespace(counts={'fish': 1}, fish=SimpleNamespace(starts=np.array([0]),
        validate_frames=validate, q=NoPayload(), actions=NoPayload()))
    with pytest.raises(ValueError, match='sealed'):
        horizon_targets(data, [0], 'fish', 0)


def test_training_sampler_only_draws_original_training_rows():
    splits = dict(train={'archive': np.arange(24), 'dense': np.arange(8), 'fish': np.arange(4)},
                  development={'archive': np.arange(100, 110), 'fish': np.array([100])})
    args = SimpleNamespace(seed=21004, fish_batch_size=4, lewm_fish_updates=4,
        actor_fish_updates=4, canary=False, archive_passes=1, batch_size=4)
    for stage in ('lewm_archive', 'lewm_fish', 'actor_archive', 'actor_dense', 'actor_fish'):
        for h, groups in ordered_groups(stage, splits, args):
            assert h in range(4)
            for kind, rows in groups:
                assert np.isin(rows, splits['train'][kind]).all()


def test_explicit_third_demonstration_moves_to_training_without_validation_overlap():
    splits = dict(train={'archive': np.arange(24), 'fish': np.array([0, 1, 3, 4])},
                  development={'archive': np.arange(100, 110), 'fish': np.array([6, 7])})
    result = select_fish_training_episodes(splits, [0, 1, 3])
    assert result['train']['fish'].tolist() == [0, 1, 3, 4, 6, 7]
    assert len(result['development']['fish']) == 0
    assert np.array_equal(result['development']['archive'], splits['development']['archive'])
    assert splits['development']['fish'].tolist() == [6, 7]
    with pytest.raises(ValueError):
        select_fish_training_episodes(splits, [0, 1, 2, 3])


def test_larger_cache_resume_preserves_data_device_precision_and_optimizer_recipe():
    previous = dict(fish_train_episodes=[0, 1, 3],
        device=dict(cuda_visible_devices='GPU-1', gpu_memory_limit_gib=32., gpu_memory_fraction=.4, tf32_enabled=True),
        schedule=dict(gpu_memory_limit_gib=32., feature_bank_gib=12., batch_size=64, learning_rate=3e-5))
    current = deepcopy(previous)
    current['device'].update(gpu_memory_limit_gib=72., gpu_memory_fraction=.9)
    current['schedule'].update(gpu_memory_limit_gib=72., feature_bank_gib=48.)
    validate_resource_continuation(previous, current)
    for section, key, value in [('device', 'cuda_visible_devices', 'GPU-0'),
                                ('device', 'tf32_enabled', False),
                                ('schedule', 'batch_size', 128),
                                ('schedule', 'learning_rate', 1e-4)]:
        changed = deepcopy(current)
        changed[section][key] = value
        with pytest.raises(ValueError, match='cannot change'):
            validate_resource_continuation(previous, changed)
    current['fish_train_episodes'] = [0, 1]
    with pytest.raises(ValueError, match='cannot change'):
        validate_resource_continuation(previous, current)


def test_native_checkpoint_roundtrip_and_legacy_default_compatibility(tmp_path):
    torch.set_num_threads(2)
    vocab = TaskVocabulary.build(['pick fish'])
    core = V21ResearchCore(V21Config(len(vocab.words)))
    schedule = V21Schedule()
    path = tmp_path/'legacy.pt'
    save_checkpoint(path, core, vocab, schedule, stage='fish', step=1, complete=True, smoke=False)
    payload = torch.load(path, weights_only=False)
    del payload['config']['prediction_horizon']; del payload['config']['execution_horizon']
    torch.save(payload, path)
    load_checkpoint(path, core, vocab, stage='fish')
    initialize_native40(core)
    assert core.config.prediction_horizon == 40 and core.config.execution_horizon == 5
    path = tmp_path/'native.pt'
    save_checkpoint(path, core, vocab, schedule, stage='fish', step=1, complete=True, smoke=False)
    native = V21ResearchCore(V21Config(**asdict(core.config)))
    result = load_checkpoint(path, native, vocab, stage='fish')
    assert result['version'] == '21.0.4'
    assert native.actor.query.shape == native.lewm.query.shape == (1, 40, 192)
    for k, value in core.state_dict().items():
        assert torch.equal(value, native.state_dict()[k])
