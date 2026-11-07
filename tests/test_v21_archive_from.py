"""17.123: a fresh LeWM-only full-patch run may restore the finished Archive stage of a run with the same recipe
(--lewm-archive-from) and train only its fish stage."""
import json

import pytest

from scripts.train_v21_horizon40 import canonical_protocol, main, parser, validate_archive_stage_source
from v21_lewm_helpers import DEFAULTS, finished_archive_stage, protocol_of, stored_run

BASE = ['--parent', 'parent.pt', '--source-config', 'config.json', '--output', 'out']


def test_the_option_defaults_to_training_the_archive_stage_and_older_runs_read_as_it():
    assert parser().parse_args(BASE).lewm_archive_from is None
    old = dict(schedule=dict(full_patch_lewm=True, scene_latent_lewm=False), sigreg_weight=.02)
    new = dict(schedule=dict(full_patch_lewm=True, scene_latent_lewm=False, lewm_archive_from=None),
               sigreg_weight=.02, lewm_archive_stage_from=None)
    assert canonical_protocol(old) == canonical_protocol(new)
    restoring = dict(new, schedule=dict(new['schedule'], lewm_archive_from='HL/lewm_archive/final.pt'))
    assert canonical_protocol(old) != canonical_protocol(restoring)


@pytest.mark.parametrize('extra', [
    ['--lewm-archive-from', 'a.pt'],
    ['--lewm-only', '--full-patch-lewm', '--lewm-archive-from', 'a.pt', '--resume'],
])
def test_the_option_is_refused_outside_a_fresh_lewm_only_run_before_any_file_opens(tmp_path, extra):
    args = ['--parent', str(tmp_path/'missing.pt'), '--source-config', str(tmp_path/'missing.json'),
            '--output', str(tmp_path/'unused')] + extra
    with pytest.raises(ValueError, match='fresh --lewm-only --full-patch-lewm'):
        main(args)
    assert not (tmp_path/'unused').exists()


def test_a_restored_archive_stage_must_have_this_runs_recipe(tmp_path):
    config = stored_run(tmp_path)
    saved = finished_archive_stage(protocol_of(config))
    run = protocol_of(config, output='HL-again', lewm_archive_from='HL/lewm_archive/final.pt',
                      gpu_memory_limit_gib=30.)
    run['initial_module_sha256']['lewm'] = 'random-init-of-this-code'
    run['smoke'] = True                       # a canary may restore a formal stage
    record = validate_archive_stage_source(saved, run, config, DEFAULTS)
    assert record['same_source_config'] and record['stage_optimizer_step'] == 22536
    with pytest.raises(ValueError, match='schedule.lewm_fish_updates: 1000 -> 500'):
        validate_archive_stage_source(saved, protocol_of(config, lewm_fish_updates=500), config, DEFAULTS)
    with pytest.raises(ValueError, match='initial_module_sha256.actor'):
        validate_archive_stage_source(saved, dict(run, initial_module_sha256=dict(reader='r', actor='b')),
                                      config, DEFAULTS)
    for change in (dict(complete=False), dict(smoke=True), dict(stage='lewm_fish')):
        with pytest.raises(ValueError, match='completed formal lewm_archive'):
            validate_archive_stage_source(finished_archive_stage(protocol_of(config), **change), run,
                                          config, DEFAULTS)


def test_another_copy_of_the_same_cache_is_the_same_data_and_nothing_else_is(tmp_path):
    config = stored_run(tmp_path)
    saved = finished_archive_stage(protocol_of(config))
    nvme = stored_run(tmp_path, root='cache-b', note='section 17.65: the same blocks on the NVMe')
    record = validate_archive_stage_source(saved, protocol_of(nvme), nvme, DEFAULTS)
    assert not record['same_source_config']
    other = stored_run(tmp_path, root='cache-c', manifest=dict(extraction_id='another', rows=9))
    with pytest.raises(ValueError, match='same cache'):
        validate_archive_stage_source(saved, protocol_of(other), other, DEFAULTS)
    moved = json.loads(nvme.read_text())
    moved['body_table'] = 'another-table.npy'
    changed = tmp_path/'config-changed.json'
    changed.write_text(json.dumps(moved))
    with pytest.raises(ValueError, match='same cache'):
        validate_archive_stage_source(saved, protocol_of(changed), changed, DEFAULTS)
