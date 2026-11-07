"""17.127: a second draw of the LeWM fish stage alone (--fish-stage-seed) on the same restored Archive stage, to replicate
17.125/17.126 before the system LeWM changes. The Archive stage and the examination rows stay those of --seed."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.train_v21_horizon40 import canonical_protocol, ordered_groups, parser, validate_archive_stage_source
from test_v21_fish_start import ARCHIVE_FROM, run_main
from test_v21_task_pool import BASE, SPLITS, fish_stage_args, same
from v21_lewm_helpers import DEFAULTS, finished_archive_stage, protocol_of, stored_run


def with_seed(args, seed, **change):
    return SimpleNamespace(**dict(vars(args), fish_stage_seed=seed, **change))


def test_the_fish_stage_seed_draws_the_fish_stage_again_and_nothing_else():
    args = fish_stage_args()
    plain = list(ordered_groups('lewm_fish', SPLITS, args))
    assert same(plain, list(ordered_groups('lewm_fish', SPLITS, with_seed(args, args.seed))))   # the same seed: as before
    other = list(ordered_groups('lewm_fish', SPLITS, with_seed(args, 42)))
    assert len(other) == len(plain) and not same(other, plain)
    assert all(np.isin(rows, SPLITS['train'][kind]).all() for _, groups in other for kind, rows in groups)
    diagnostic = dict(lewm_archive_only_diagnostic=True)
    replaced = [list(ordered_groups('lewm_fish', SPLITS, with_seed(args, seed, **diagnostic))) for seed in (None, 42)]
    assert not same(*replaced)                        # the Archive rows that replace the fish are drawn again too
    canary = parser().parse_args(BASE + ['--canary'])
    assert same(list(ordered_groups('lewm_archive', SPLITS, canary)),
                list(ordered_groups('lewm_archive', SPLITS, with_seed(canary, 42))))   # the Archive stage: unchanged


def test_the_option_is_for_a_run_that_restores_an_archive_stage(tmp_path):
    for extra in (['--fish-stage-seed', '42'], ARCHIVE_FROM + ['--fish-stage-seed', '-1']):
        with pytest.raises(ValueError, match='fish-stage-seed'):
            run_main(tmp_path, extra)
    assert not (tmp_path/'unused').exists()
    for extra in (ARCHIVE_FROM + ['--fish-stage-seed', '42', '--lewm-archive-only-diagnostic'],
                  ARCHIVE_FROM + ['--fish-stage-seed', '42', '--lewm-fish-start', 'b.pt']):
        with pytest.raises(FileNotFoundError):
            run_main(tmp_path, extra)


def test_older_runs_read_as_drawn_from_their_seed_and_the_archive_stage_check_ignores_it(tmp_path):
    assert parser().parse_args(BASE).fish_stage_seed is None
    old = dict(schedule=dict(full_patch_lewm=True, scene_latent_lewm=False), sigreg_weight=.02)
    assert canonical_protocol(old) == canonical_protocol(dict(old, schedule=dict(old['schedule'], fish_stage_seed=None)))
    config = stored_run(tmp_path)
    saved = finished_archive_stage(protocol_of(config))
    validate_archive_stage_source(saved, protocol_of(config, output='b-archive-s42', fish_stage_seed=42,
                                                     lewm_archive_from='HL/lewm_archive/final.pt',
                                                     lewm_archive_only_diagnostic=True), config, DEFAULTS)
    with pytest.raises(ValueError, match='schedule.seed'):            # --seed itself is still the Archive stage's
        validate_archive_stage_source(saved, protocol_of(config, seed=42), config, DEFAULTS)
