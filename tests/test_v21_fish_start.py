"""17.125: the fish-stage options around a restored Archive stage. The archive-only diagnostic (the fish stage with
Archive rows in place of the fish) fine-tunes the whole LeWM of a full-patch LeWM-only run (--lewm-archive-only-diagnostic);
a later fish stage starts from that LeWM (--lewm-fish-start) and may learn the demonstrations alone
(--lewm-fish-demos-only)."""
import numpy as np
import pytest
import torch

from scripts.train_v21_horizon40 import (canonical_protocol, load_fish_start, main, ordered_groups, parser, sha,
                                         validate_archive_stage_source)
from test_v21_task_pool import BASE, SPLITS, TASK, fish_stage_args, same
from v21_lewm_helpers import DEFAULTS, finished_archive_stage, protocol_of, stored_run

FRESH = ['--lewm-only', '--full-patch-lewm']


def demos_only_args():
    return parser().parse_args(BASE + FRESH + ['--lewm-fish-updates', '6', '--fish-batch-size', '8',
                                               '--lewm-fish-demos-only'])


def test_demos_only_keeps_the_mixed_runs_demonstration_rows_and_nothing_else():
    for task_rows in (None, TASK):
        mixed = list(ordered_groups('lewm_fish', SPLITS, fish_stage_args(), task_rows=task_rows))
        alone = list(ordered_groups('lewm_fish', SPLITS, demos_only_args(), task_rows=task_rows))
        assert len(mixed) == len(alone) == 6
        for (h, groups), (g, kept) in zip(mixed, alone):
            assert h == g and len(groups) == 2 and len(kept) == 1
            assert kept[0][0] == groups[0][0] and np.array_equal(kept[0][1], groups[0][1])
    canary = parser().parse_args(BASE + ['--canary'])
    canary_alone = parser().parse_args(BASE + ['--canary', '--lewm-fish-demos-only'])
    assert same(list(ordered_groups('lewm_archive', SPLITS, canary)),
                list(ordered_groups('lewm_archive', SPLITS, canary_alone)))       # the Archive stage: as before


def run_main(tmp_path, extra):
    return main(['--parent', str(tmp_path/'missing.pt'), '--source-config', str(tmp_path/'missing.json'),
                 '--output', str(tmp_path/'unused')] + extra)


ARCHIVE_FROM = FRESH + ['--lewm-archive-from', 'a.pt']


@pytest.mark.parametrize('extra, message', [
    (FRESH + ['--lewm-archive-only-diagnostic'], 'whole LeWM of a restored Archive stage'),
    (ARCHIVE_FROM + ['--lewm-archive-only-diagnostic', '--lewm-fish-demos-only'], 'without a start'),
    (ARCHIVE_FROM + ['--lewm-archive-only-diagnostic', '--lewm-fish-start', 'b.pt'], 'without a start'),
    (['--lewm-archive-only-diagnostic'], 'scene-parent run'),             # the scene LeWM's diagnostic as before
    (FRESH + ['--lewm-fish-start', 'b.pt'], 'continues a restored Archive stage'),
    (['--lewm-fish-demos-only'], 'demos-only is for a fresh'),
    (FRESH + ['--lewm-fish-demos-only', '--resume'], 'demos-only is for a fresh'),
])
def test_the_options_are_refused_outside_their_runs_before_any_file_opens(tmp_path, extra, message):
    with pytest.raises(ValueError, match=message):
        run_main(tmp_path, extra)
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('extra', [
    ARCHIVE_FROM + ['--lewm-archive-only-diagnostic'],                       # the new base
    ARCHIVE_FROM + ['--lewm-fish-demos-only'],                               # the old base, demos alone
    ARCHIVE_FROM + ['--lewm-fish-start', 'b.pt'],                            # the new base, mixed
    ARCHIVE_FROM + ['--lewm-fish-start', 'b.pt', '--lewm-fish-demos-only'],  # the new base, demos alone
])
def test_the_four_runs_pass_the_option_checks(tmp_path, extra):
    with pytest.raises(FileNotFoundError):           # the first file the run opens: its (missing) source config
        run_main(tmp_path, extra)


def finished_fish_stage(tmp_path, archive, *, protocol_change=None, **change):
    protocol = dict(dict(trained_modules=['lewm'], lewm_fish_from_archive=None,
                         lewm_archive_stage_from=dict(checkpoint=str(archive), sha256=sha(archive)),
                         schedule=dict(output='b-archive', lewm_archive_only_diagnostic=True)),
                    **(protocol_change or {}))
    saved = dict(dict(complete=True, smoke=False, stage='fish', native40_protocol=protocol,
                      model={'lewm.layers.0.ff.0.weight': torch.ones(2, 3), 'actor.head.weight': torch.zeros(1)}),
                 **change)
    path = tmp_path/'start.pt'
    torch.save(saved, path)
    return path


def test_a_start_is_a_whole_lewm_fine_tuned_from_the_same_archive_stage(tmp_path):
    archive = tmp_path/'archive.pt'
    archive.write_bytes(b'the Archive stage this run restores')
    start = load_fish_start(finished_fish_stage(tmp_path, archive), archive)
    assert list(start['lewm']) == ['layers.0.ff.0.weight'] and torch.equal(start['lewm']['layers.0.ff.0.weight'],
                                                                           torch.ones(2, 3))
    assert start['record']['run'] == 'b-archive' and start['record']['archive_only']
    assert not start['record']['demos_only'] and start['record']['sha256'] == sha(tmp_path/'start.pt')
    other = tmp_path/'other.pt'
    other.write_bytes(b'another Archive stage')
    with pytest.raises(ValueError, match='another Archive stage'):
        load_fish_start(finished_fish_stage(tmp_path, archive), other)
    for change in (dict(smoke=True), dict(complete=False), dict(stage='lewm_fish'),
                   dict(protocol_change=dict(trained_modules=['lewm', 'actor']))):
        with pytest.raises(ValueError, match='completed formal LeWM-only'):
            load_fish_start(finished_fish_stage(tmp_path, archive, **change), archive)


def test_older_runs_read_as_a_fish_stage_from_the_archive_stage_with_archive_mixed_in():
    args = parser().parse_args(BASE)
    assert (args.lewm_fish_start, args.lewm_fish_demos_only) == (None, False)
    old = dict(schedule=dict(full_patch_lewm=True, scene_latent_lewm=False), sigreg_weight=.02)
    new = dict(schedule=dict(full_patch_lewm=True, scene_latent_lewm=False, lewm_fish_start=None,
                             lewm_fish_demos_only=False), sigreg_weight=.02, lewm_fish_start=None)
    assert canonical_protocol(old) == canonical_protocol(new)


def test_the_four_runs_restore_the_same_archive_stage(tmp_path):
    config = stored_run(tmp_path)
    saved = finished_archive_stage(protocol_of(config))
    base = protocol_of(config, output='b-archive', lewm_archive_from='HL/lewm_archive/final.pt',
                       lewm_archive_only_diagnostic=True)
    validate_archive_stage_source(saved, base, config, DEFAULTS)
    for demos in (False, True):
        run = protocol_of(config, output='p-b', lewm_archive_from='HL/lewm_archive/final.pt',
                          lewm_fish_start='b-archive/fish/final.pt', lewm_fish_demos_only=demos)
        run.update(lewm_fish_start=dict(run='b-archive'))
        validate_archive_stage_source(saved, run, config, DEFAULTS)
    with pytest.raises(ValueError, match='schedule.fish_learning_rate'):            # the recipe is still checked
        validate_archive_stage_source(saved, dict(base, schedule=dict(base['schedule'], fish_learning_rate=1e-5)),
                                      config, DEFAULTS)
