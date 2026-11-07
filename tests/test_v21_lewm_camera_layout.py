"""Section 17.59 (4th fix): the LeWM trainer reads the Archive cameras through the camera table, as the Actor trainer
does. Synthetic index and entry-point checks, not a real-data or model-quality result."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.v21_data import V21Sources
from fcpg.v21_encoder import DIM, PATCHES
import fcpg.v21_data as data
from scripts.train_v21_horizon40 import (apply_camera_layout, main, ordered_groups, prepare_training_splits,
                                         validate_resource_continuation)


def source_and_table(tmp_path):
    # 4 canonical episodes of 10 endpoints; chain row r starts at endpoint 5r, so rows 2e and 2e+1 are episode e.
    # TRAIN: episodes 0 and 2, DEV: episodes 1 and 3.
    split = np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64)
    archive = SimpleNamespace(archive=SimpleNamespace(split=split), row_endpoint_index=np.arange(40).reshape(8, 5),
                              endpoint_episode_index=np.repeat(np.arange(4), 10), episodes=[{}] * 4)
    fish = SimpleNamespace(starts=np.array([0, 70, 210]), episodes=np.repeat(np.arange(4), 70))
    fish.validate_frames = lambda ids: None
    valid = np.tile([True, False], (40, 1))                  # every Archive endpoint: only slot 0 recorded
    clock = SimpleNamespace(observations=lambda ids, kind: (valid[ids], np.zeros(valid[ids].shape),
                                                            np.zeros(valid[ids].shape)))
    source = SimpleNamespace(archive=archive, offsets=np.arange(0, 41, 10), dense_rows=np.zeros(0, np.int64),
                             fish=fish, counts=dict(archive=8, dense=0, fish=3), recorded_clock=clock)
    # episode 0 in order, 1 swapped, 2 undecided, 3 decided without a wrist camera
    table = tmp_path / 'layout.npz'
    np.savez(table, order=np.array([[0, 1], [1, 0], [0, 1], [0, 1]]),
             wrist_present=np.array([True, True, False, False]), decided=np.array([True, True, False, True]))
    return source, table


def test_only_decided_episodes_stay_in_both_archive_splits_and_fish_is_untouched(tmp_path):
    source, table = source_and_table(tmp_path)
    splits, _ = prepare_training_splits(source, {}, [0, 1])
    fish = {role: splits[role]['fish'].copy() for role in ('train', 'development')}
    record = apply_camera_layout(source, splits, table)
    assert splits['train']['archive'].tolist() == [0, 1]                # episode 2 is undecided
    assert splits['development']['archive'].tolist() == [2, 3, 6, 7]
    assert all(np.array_equal(splits[role]['fish'], fish[role]) for role in fish)
    assert record['archive_rows']['train'] == dict(rows=4, decided=2, swapped=0, no_wrist=0, undecided=2)
    assert record['archive_rows']['development'] == dict(rows=4, decided=4, swapped=2, no_wrist=2, undecided=0)
    assert len(record['table_sha256']) == 64
    # the recorded clock now gives the views in the model's order: episode 1 swapped, episode 3 has no wrist view
    view = source.recorded_clock.observations(np.array([0, 10, 30]), 'archive')[0]
    assert view.tolist() == [[True, False], [False, True], [False, False]]
    assert source.recorded_clock.observations(np.array([10]), 'fish')[0].tolist() == [[True, False]]
    with pytest.raises(ValueError, match='one camera contract'):
        apply_camera_layout(source, splits, table)


def test_the_archive_tokens_lewm_reads_follow_the_table(tmp_path, monkeypatch):
    source, table = source_and_table(tmp_path)
    splits, _ = prepare_training_splits(source, {}, [0, 1])
    apply_camera_layout(source, splits, table)
    ids = np.array([0, 10, 30])
    tokens = torch.zeros(3, 2, PATCHES, DIM, dtype=torch.float16)
    tokens[:, 0], tokens[:, 1] = 1., 2.
    monkeypatch.setattr(data.V201Sources, '_get_archive_dino', lambda self, ids: (tokens, np.ones((3, 2), bool)))
    got, valid = V21Sources._get_archive_dino(source, ids)
    assert (got[0, 0] == 1).all() and (got[0, 1] == 2).all()             # in order
    assert (got[1, 0] == 2).all() and (got[1, 1] == 1).all()             # swapped
    assert (got[2, 0] == 0).all() and (got[2, 1] == 2).all()             # no wrist camera: slot 0 empty
    assert np.asarray(valid).tolist() == [[True, True], [True, True], [False, True]]


@pytest.mark.parametrize('stage', ['lewm_archive', 'lewm_fish'])
def test_both_lewm_stages_draw_only_admitted_archive_rows(tmp_path, stage):
    source, table = source_and_table(tmp_path)
    splits, _ = prepare_training_splits(source, {}, [0, 1])
    apply_camera_layout(source, splits, table)
    args = SimpleNamespace(seed=21004, lewm_fish_updates=200, actor_fish_updates=200, fish_batch_size=4,
                           canary=False, archive_batch_sampling='iid-v1', lewm_archive_only_diagnostic=False,
                           batch_size=1, lewm_batch_size=1, archive_passes=1, from_scratch=False)
    seen = set()
    for _, groups in ordered_groups(stage, splits, args):
        for kind, rows in groups:
            if kind == 'archive':
                seen.update(rows.tolist())
    assert seen == {0, 1}


def test_the_table_is_refused_outside_lewm_only_runs_before_anything_is_opened(tmp_path, monkeypatch):
    import scripts.train_v21_horizon40 as trainer
    monkeypatch.setattr(trainer, 'configure_device', lambda *a: pytest.fail('GPU touched'))
    with pytest.raises(ValueError, match='LeWM-only'):
        main(['--parent', str(tmp_path / 'not-opened.pt'), '--source-config', str(tmp_path / 'not-read.json'),
              '--output', str(tmp_path / 'run'), '--camera-layout', str(tmp_path / 'layout.npz')])
    assert not (tmp_path / 'run').exists()


def test_runs_without_the_table_stay_resumable_and_a_changed_table_is_refused():
    before = dict(schedule=dict(precision='bf16'), device={})           # written before --camera-layout existed
    validate_resource_continuation(before, dict(schedule=dict(precision='bf16', camera_layout=None), device={}))
    with_table = dict(schedule=dict(precision='bf16', camera_layout='layout.npz'), device={},
                      archive_camera_layout=dict(table_sha256='a'))
    with pytest.raises(ValueError):
        validate_resource_continuation(before, with_table)
    with pytest.raises(ValueError):
        validate_resource_continuation(with_table, dict(with_table, archive_camera_layout=dict(table_sha256='b')))
