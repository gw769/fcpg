"""Helpers the full-patch LeWM tests and the trainer's option tests share: a small core, state and targets from the
synthetic chain source, and the stored runs, protocols and finished Archive stages that --lewm-archive-from compares."""
import json

import numpy as np
import torch

from fcpg.v21_core import V21Config, V21ResearchCore
from fcpg.v21_full_patch_lewm import V21FullPatchLeWM
from fcpg.v21_horizon_training import factual_state, horizon_targets
from scripts.train_v21_horizon40 import parser, sha
from test_v21_scratch_features import actual_chain_source


def setup(**lewm):
    """A core (task-reading FCPG), two archive rows' state, their text ids and 40-step targets, the source, and a
    full-patch LeWM built with these options."""
    torch.manual_seed(21004)
    source = actual_chain_source()
    source.offsets = np.array([0, len(source.raw_q)])
    source.archive.endpoint_episode_index = np.zeros(8, dtype=np.int64)
    core = V21ResearchCore(V21Config(len(source.vocabulary.words), prediction_horizon=40, object_relative=True,
                                     reader_task=True)).eval()
    rows = np.array([0, 1])
    chain = source.chain(rows, 'archive', images=True)
    with torch.no_grad():       # the state from the core's own LeWM
        state = factual_state(core, chain, 2, images=True)
    torch.manual_seed(21004)
    core.lewm = V21FullPatchLeWM(increment_scale=core.actor.increment_scale.detach().cpu(), horizon=40, **lewm).eval()
    return core, state, chain.text_ids, horizon_targets(source, rows, 'archive', 2), source


def stored_run(tmp_path, *, root='cache-a', note=None, manifest=None):
    cache = tmp_path/root
    cache.mkdir(exist_ok=True)
    (cache/'manifest.json').write_text(json.dumps(manifest or dict(extraction_id='dinov3-l-int8-20261021', rows=9)))
    config = dict(source_paths=dict(dino_cache_root=str(cache), archive='index'), body_table='table.npy')
    if note:
        config['nvme_store_note'] = note
    path = tmp_path/f'config-{root}.json'
    path.write_text(json.dumps(config))
    return path


def protocol_of(config, **schedule):
    return dict(schedule=dict(dict(full_patch_lewm=True, scene_latent_lewm=False, lewm_only=True,
                                   lewm_fish_updates=1000, fish_learning_rate=3e-5, output='HL',
                                   source_config=str(config)), **schedule),
                sigreg_weight=.02, original_source_config=str(config), source_config_sha256=sha(config),
                device=dict(cuda_visible_devices='0', gpu_memory_limit_gib=24.),
                initial_module_sha256=dict(reader='r', actor='a', lewm='random-init-of-that-code'), smoke=False)


def finished_archive_stage(protocol, **change):
    return dict(dict(complete=True, smoke=False, stage='lewm_archive', stage_optimizer_step=22536,
                     native40_protocol=protocol), **change)


DEFAULTS = {a.dest: a.default for a in parser()._actions}
