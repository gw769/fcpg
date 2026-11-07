#!/usr/bin/env python3
"""Demo 3 and the Archive development rows for LeWMs: the forecast errors (first 5 steps, all 40) on the development
demonstration (fish episode 3) and on a fixed sample of the Archive development rows, the same rows and targets for
every LeWM. A LeWM with memory_pictures=N reads fcpg/v21_picture_memory.py's memory of the last N pictures per camera
of the row (demo 3: its own earlier frames; an Archive row: its recording's earlier pictures) at the stride given by
V21_MEMORY_STRIDE (half-seconds, default 1: then its newest three are the chain's own three) and nothing lost; every
other LeWM reads the chain's memory. A LeWM with the control-setting input is given the planner's setting off [0, 0].
Usage (V21_ENCODER=dinov3_vitl16): python scripts/eval_v21_lewm_demo3.py RUN OUTPUT.json NAME=CHECKPOINT...
RUN is a training run whose config.json and protocol.json name the data, the development rows and the seed.
"""
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fcpg.v21_data import V21Sources  # noqa: E402
from fcpg.v21_horizon_precision import compute_context  # noqa: E402
from fcpg.v21_horizon_training import horizon_errors, horizon_metadata, horizon_targets  # noqa: E402
from fcpg.v21_lewm_inputs import lewm_inputs  # noqa: E402
from fcpg.v21_native_horizon import core_from_parent  # noqa: E402
from fcpg.v21_picture_memory import archive_picture_memory, fish_picture_memory  # noqa: E402
from fcpg.v21_scratch_features import ScratchDinoFeatureBank  # noqa: E402
from scripts.train_v21_from_scratch import stage_chain_to_device  # noqa: E402
from scripts.train_v21_horizon40 import (apply_camera_layout, augment_measurements, development_mode,  # noqa: E402
                                         prepare_training_splits, temporary_eval)


def combine(entries):
    total = sum(e['examples'] for e in entries)
    return dict(examples=total, **{k: sum(e[k] * e['examples'] for e in entries) / total
                                   for k in ('first5_mae', 'future_mae')})


def main(run, output, named):
    run = Path(run)
    config = json.loads((run/'config.json').read_text())
    schedule = json.loads((run/'protocol.json').read_text())['schedule']
    device = torch.device('cuda:0')
    stride = int(os.environ.get('V21_MEMORY_STRIDE', 1))
    if schedule.get('tf32'):
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.backends.mha.set_fastpath_enabled(False)
    precision = schedule.get('precision', 'fp32')
    source = V21Sources(config['source_paths'], device='cpu', object_relative_table=config['body_table'])
    splits, _ = prepare_training_splits(source, config, schedule['fish_train_episodes'])
    if schedule.get('camera_layout'):
        apply_camera_layout(source, splits, Path(schedule['camera_layout']))
    if schedule['fish_train_episodes'] != [0, 1]:
        raise ValueError('demo 3 must be the development demonstration')
    bank = ScratchDinoFeatureBank(source, device, 2 * 1024 ** 3)
    archive = splits['development']['archive']
    archive = np.sort(np.random.default_rng(schedule['seed'] + 800).choice(
        archive, min(len(archive), schedule['development_rows']), replace=False))
    rows_of = dict(fish=splits['development']['fish'], archive=archive)
    report = dict(memory_stride=stride, run=str(run), precision=precision, rows={k: len(v) for k, v in rows_of.items()},
                  checkpoints={})
    for name, path in named:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        core = core_from_parent(payload).to(device).eval()
        reads_setting = getattr(core.lewm, 'controller_input', False)
        memory_pictures = (core.lewm.architecture() if hasattr(core.lewm, 'architecture') else {}).get('memory_pictures', 0)
        result = {}
        with torch.no_grad(), development_mode(core):
            for kind, development in rows_of.items():
                entries = []
                for h in range(4):
                    for begin in range(0, len(development), 32):
                        rows = development[begin:begin + 32]
                        metadata = horizon_metadata(source, rows, kind, h)
                        keep = metadata[-1][:, 0]
                        rows, decision = rows[keep], metadata[0][keep]
                        if not len(rows):
                            continue
                        if kind == 'fish' and (np.asarray(source.fish.episodes[decision]) != 3).any():
                            raise ValueError('a development row is not in demo 3')
                        targets = horizon_targets(source, rows, kind, h).to(device)
                        host = source.chain(rows, kind, images=False)
                        features, checked = bank.load(rows, kind, context_frames=h + 1)
                        if not np.array_equal(checked, host.view_valid[:, :h + 1].numpy()):
                            raise ValueError('cached factual visual prefix identities differ')
                        chain = replace(stage_chain_to_device(host, device), features=features.to(device))
                        chain = augment_measurements(chain, rows, protocol=schedule.get('measurement_dropout', 'none'),
                                                     variant=0, images=True, evaluation=True)
                        with temporary_eval(core.lewm), compute_context(device, precision):
                            state = lewm_inputs(core, chain, h)
                        if memory_pictures:
                            build = archive_picture_memory if kind == 'archive' else fish_picture_memory
                            state = replace(state, memory=build(source, rows, h, memory_pictures, device=device,
                                                                strides=np.full(len(rows), stride)))
                        kwargs = dict(patches=state.memory.features, patch_valid=state.memory.valid,
                                      memory=state.memory, now_time=state.time)
                        if reads_setting:
                            kwargs['controller'] = state.q.new_zeros(len(state.q), 2)
                        with compute_context(device, precision):
                            response = core.lewm(state.q, targets.trace, state.history, targets.readback_offsets,
                                                 **kwargs)
                        entries.append(horizon_errors(response.predicted_q_path[:, 1:], targets.readbacks,
                                                      targets.valid))
                result[kind] = combine(entries)
        report['checkpoints'][name] = dict(checkpoint=str(path), memory_pictures=memory_pictures,
                                           reads_controller=reads_setting, **result)
        print(json.dumps({name: result}), flush=True)
        del core, payload
        torch.cuda.empty_cache()
    Path(output).write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], [item.split('=', 1) for item in sys.argv[3:]])
