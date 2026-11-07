"""17.124: the fish stage learns two episodes of an Archive collection the Archive stage never trained on
(scripts/train_v21_horizon40.py --lewm-fish-from-archive) and is examined on its other episodes."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.train_v21_horizon40 import (canonical_protocol, load_task_pool, main, ordered_groups, parser,
                                         validate_archive_stage_source)
from v21_lewm_helpers import DEFAULTS, finished_archive_stage, protocol_of, stored_run

BASE = ['--parent', 'parent.pt', '--source-config', 'config.json', '--output', 'out']
SPLITS = dict(train=dict(archive=np.arange(0, 100), fish=np.arange(0, 30)),
              development=dict(archive=np.arange(100, 160), fish=np.arange(30, 40)))
TASK = np.array([120, 121, 122])


def fish_stage_args():
    return parser().parse_args(BASE + ['--lewm-only', '--full-patch-lewm', '--lewm-fish-updates', '6',
                                       '--fish-batch-size', '8'])


def same(a, b):
    return len(a) == len(b) and all(h == g and len(x) == len(y) and all(
        k == l and np.array_equal(r, s) for (k, r), (l, s) in zip(x, y)) for (h, x), (g, y) in zip(a, b))


def test_the_task_rows_take_the_fish_half_and_the_archive_half_stays():
    args = fish_stage_args()
    plain = list(ordered_groups('lewm_fish', SPLITS, args))
    pooled = list(ordered_groups('lewm_fish', SPLITS, args, task_rows=TASK))
    assert len(plain) == len(pooled) == 6
    for (h, (fish, archive)), (g, (learn, archive2)) in zip(plain, pooled):
        assert h == g and fish[0] == 'fish' and learn[0] == 'archive' and archive2[0] == 'archive'
        assert np.isin(learn[1], TASK).all() and len(learn[1]) == len(fish[1]) == 4
        assert np.isin(archive2[1], SPLITS['train']['archive']).all() and len(archive2[1]) == 4
    canary = parser().parse_args(BASE + ['--canary'])
    assert same(list(ordered_groups('lewm_archive', SPLITS, canary)),
                list(ordered_groups('lewm_archive', SPLITS, canary, task_rows=TASK)))      # the Archive stage: as before
    diagnostic = SimpleNamespace(**dict(vars(args), lewm_archive_only_diagnostic=True))
    with pytest.raises(ValueError, match='plain iid'):
        list(ordered_groups('lewm_fish', SPLITS, diagnostic, task_rows=TASK))


def archive_of(pool_episodes):
    """Row r -> endpoint r -> episode: training rows 0-99 in episodes 0-9, development rows 100-119 in episode 70,
    the pool's rows 120-123 in pool_episodes, the rest in episode 60."""
    episodes = np.concatenate([np.repeat(np.arange(10), 10), np.full(20, 70), pool_episodes, np.full(36, 60)])
    return SimpleNamespace(row_endpoint_index=np.arange(160)[:, None].repeat(4, 1), endpoint_episode_index=episodes)


def pool_file(tmp_path, **change):
    pool = dict(dict(section='17.124', task='pen2box', collections=['/data/so101/raw_expand/so101_pen2box'],
                     learn_episodes=[50, 51], test_episodes=[52, 53], learn_rows=[120, 121], test_rows=[122, 123],
                     excluded_rows=[120, 121, 122, 123, 124]), **change)
    path = tmp_path/'pool.json'
    path.write_text(json.dumps(pool))
    return path


def test_a_task_pool_must_be_unseen_development_rows_of_other_episodes(tmp_path):
    archive = archive_of([50, 51, 52, 53])
    pool = load_task_pool(pool_file(tmp_path), SPLITS, archive)
    assert pool['learn_rows'].tolist() == [120, 121] and pool['record']['excluded_rows'] == 5
    assert pool['record']['learn_episodes'] == [50, 51] and pool['record']['test_episodes'] == 2
    with pytest.raises(ValueError, match='not in the Archive development split'):
        load_task_pool(pool_file(tmp_path, learn_rows=[99, 121]), SPLITS, archive)
    with pytest.raises(ValueError, match='same episode'):
        load_task_pool(pool_file(tmp_path), SPLITS, archive_of([50, 51, 51, 53]))
    with pytest.raises(ValueError, match='has Archive training rows'):
        load_task_pool(pool_file(tmp_path), SPLITS, archive_of([5, 51, 52, 53]))
    with pytest.raises(ValueError, match='leave the Archive development report'):
        load_task_pool(pool_file(tmp_path, excluded_rows=[120, 121]), SPLITS, archive)
    with pytest.raises(ValueError, match='not a 17.124'):
        load_task_pool(pool_file(tmp_path, section='17.123'), SPLITS, archive)


def test_the_option_is_refused_outside_a_fresh_lewm_only_run_and_older_runs_read_without_it(tmp_path):
    args = ['--parent', str(tmp_path/'missing.pt'), '--source-config', str(tmp_path/'missing.json'),
            '--output', str(tmp_path/'unused'), '--lewm-fish-from-archive', str(tmp_path/'pool.json')]
    with pytest.raises(ValueError, match='lewm-fish-from-archive'):
        main(args)
    assert not (tmp_path/'unused').exists()
    old = dict(schedule=dict(full_patch_lewm=True, scene_latent_lewm=False), sigreg_weight=.02)
    new = dict(schedule=dict(full_patch_lewm=True, scene_latent_lewm=False, lewm_fish_from_archive=None),
               sigreg_weight=.02, lewm_fish_from_archive=None)
    assert canonical_protocol(old) == canonical_protocol(new)


def test_a_task_run_restores_the_same_archive_stage(tmp_path):
    config = stored_run(tmp_path)
    saved = finished_archive_stage(protocol_of(config))
    task_run = protocol_of(config, output='pen2box', lewm_fish_from_archive='task-pools/pen2box.json')
    task_run['lewm_fish_from_archive'] = dict(task='pen2box')
    validate_archive_stage_source(saved, task_run, config, DEFAULTS)
