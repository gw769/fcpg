#!/usr/bin/env python3
"""The alignment CEM's labels for the Actor's fish stage (fcpg/v21_cem_teacher.py; training overhaul doc section 17.110),
on the outline cue tables (--object-cues, section 17.117) the Actor reads: the state's fish cue is in the 'hold' layout
(against where the demonstrations' jaws held the fish, --hold-reference), so the CEM plans with it moved back to the jaw
midpoint at the state's pose (the wrist calibration, --wrist-reference), as the robot's AlignmentCEMAdapter._jaw_layout;
a shift moves both alike.

Per fish chain of the split and context h, the state is built as the training builds it (horizon targets, the recorded
chain, the cached DINOv3-L pictures, factual_state with the model's own system LeWM forecasting the completed blocks),
in FP32. Where the current wrist token sees the fish and the jaws are open, the CEM of the robot (steer approach on,
the service's geometric wrist response, its table gate at --floor-mm, the LeWM forecast as --alignment-fp32) searches
around the demonstrator's next five commands for the token as it is and moved by --shifts - 1 random shifts. train:
the chains the fish stage draws (demonstrations 0 and 1, --trim-idle-start's moving chains per context); exam: the
moving chains of demonstration 3 (never trained on). Demonstration 2 is never read (derive_splits leaves it out).
No robot I/O, no training.
Usage on .102 (lewm venv, the A100):
  PYTHONPATH=$HOME/so101-c4-step2/analysis/debuglog-redirect:. V21_ENCODER=dinov3_vitl16 python scripts/build_v21_cem_teacher_labels.py \
    --model <packaged robot model> --source-config <config> --camera-layout <layout.npz> --object-cues <dir> \
    --place-table <dir>/fish_place_cue.npy --hold-reference <fish-cues.npz> --wrist-reference <calibration.json> \
    --response <wrist geometric response> --split train --output <new dir>
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcpg.so101_kinematics import SO101MuJoCoKinematics
from fcpg.v21_alignment_cem import (ELITES, IMAGE, ITERATIONS, MAX_OFFSET, MIN_GAIN_PX, OPEN_GRIP, POPULATION, PREFIX,
                                    RAMP, RANKED_CHECKS, SEARCH_JOINTS, calibrated_jaw_midpoint, load_response)
from fcpg.v21_archive_camera_layout import install as install_camera_layout
from fcpg.v21_cem_teacher import (SECTION, SHIFT_LIMIT_PX, TeacherLabels, admission, draw_shifts, forecaster,
                                  remembered, table_gate, teach)
from fcpg.v21_data import V21Sources
from fcpg.v21_horizon_training import factual_state, horizon_metadata, horizon_targets
from fcpg.v21_picture_memory import fish_picture_memory
from fcpg.v21_runtime import load_runtime
from fcpg.v21_sam3_finder import demo_hold_reference
from fcpg.v21_scratch_features import ScratchDinoFeatureBank
from scripts.train_v21_fcpg_actor import FISH_EXAM, FISH_TRAIN, command_onsets, moving_rows, sha256
from scripts.train_v21_from_scratch import derive_splits, stage_chain_to_device

TASK = 'Pick up the red fish toy and place it in the cardboard box'


def one_state(value, index):
    """Row index of a batched state (tensors, arrays and dataclasses of them, as the picture memory), batch kept."""
    from dataclasses import fields, is_dataclass
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return value[index:index + 1] if value.ndim else value
    if is_dataclass(value):
        return replace(value, **{f.name: one_state(getattr(value, f.name), index) for f in fields(value)})
    return value


def jaw_layout(cue, q, hold_px, jaw_midpoint):
    """A 'hold' fish cue [8] moved back to the jaw midpoint at q (AlignmentCEMAdapter._jaw_layout): the same centre."""
    out = np.asarray(cue, np.float64).copy()
    out[2:4] = (out[2:4] * IMAGE + np.asarray(hold_px, np.float64) - np.asarray(jaw_midpoint(q), np.float64)) / IMAGE
    return out


def lewm_state_hash(lewm):
    """The hash scripts/train_v21_fcpg_actor.py takes of its frozen LeWM (same bytes, same order)."""
    return hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes()
                                   for _, v in sorted(lewm.state_dict().items()))).hexdigest()


def spread(rows, limit):
    """All rows, or at most limit of them evenly spaced (a smoke run sees the whole demonstration)."""
    rows = np.asarray(rows)
    if not limit or len(rows) <= limit:
        return rows
    return rows[np.unique(np.linspace(0, len(rows) - 1, limit).round().astype(np.int64))]


def chain_states(core, source, bank, rows_by_context, device, *, batch=16, limit=0, skipped=None):
    """Per fish chain and context h (rows_by_context: {h: chains}), built as the training builds them (horizon targets,
    the recorded chain, the cached pictures, factual_state with the core's own LeWM), in the core's precision:
    yields (row, h, the one-chain state, the demonstrator's next five commands [5, 6] float64, their command offsets
    [1, 5] on the device) where the current wrist token sees the fish, the jaws are open and the five commands are
    recorded. skipped (a Counter) counts the others by reason."""
    skipped = Counter() if skipped is None else skipped
    for h in range(4):
        chains = spread(rows_by_context[h], limit)
        for begin in range(0, len(chains), batch):
            part = chains[begin:begin + batch]
            available = horizon_metadata(source, part, 'fish', h)[-1][:, 0]
            skipped['no_future'] += int((~available).sum())
            part = part[available]
            if not len(part):
                continue
            target = horizon_targets(source, part, 'fish', h)
            host = source.chain(part, 'fish', images=False)
            features, valid = bank.load(part, 'fish', context_frames=h + 1)
            if not np.array_equal(valid, host.view_valid[:, :h + 1].numpy()):
                raise ValueError('cached visual prefix identities differ')
            chain = replace(stage_chain_to_device(host, device), features=features.to(device))
            state = factual_state(core, chain, h, images=True)
            if hasattr(core.lewm, 'memory_pictures'):      # 17.167: the LeWM reads the chain's last N pictures
                state = replace(state, pictures=fish_picture_memory(source, part, h, int(core.lewm.memory_pictures),
                                                                    device=device))
            for i, row in enumerate(part.tolist()):
                single = one_state(state, i)
                cue = getattr(single, 'object_relative', None)
                if not bool(target.valid[i, :5].all()):
                    skipped['short_future'] += 1
                elif cue is None or not float(cue[0, 0]) > .5:
                    skipped['fish_not_seen'] += 1
                elif not float(single.q[0, 5]) >= OPEN_GRIP:
                    skipped['jaws_not_open'] += 1
                else:
                    yield (row, h, single, target.commands[i, :5].double().numpy(),
                           target.command_offsets[i:i + 1, :5].to(device))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('--model', type=Path, required=True, help='packaged model: its reader builds the states, its LeWM forecasts')
    p.add_argument('--source-config', type=Path, required=True)
    p.add_argument('--camera-layout', type=Path, required=True)
    p.add_argument('--object-cues', type=Path, required=True, help='the outline cue tables the Actor reads (17.117)')
    p.add_argument('--place-table', type=Path, default=None, help='its fish_place_cue.npy, for a model reading place cues')
    p.add_argument('--hold-reference', type=Path, required=True,
                   help='the demonstrations\' fish cue report (the robot\'s HOLDREF): where the jaws held the fish')
    p.add_argument('--wrist-reference', type=Path, required=True,
                   help='the wrist calibration the robot\'s object finder projects the jaws with')
    p.add_argument('--response', type=Path, required=True, help='the robot\'s wrist geometric response (--alignment-response)')
    p.add_argument('--split', choices=('train', 'exam'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--floor-mm', type=float, default=-10., help='the robot\'s --table-clearance-floor-mm')
    p.add_argument('--shifts', type=int, default=16, help='token shifts per chain and context, the first is none')
    p.add_argument('--shift-std-px', type=float, default=35.)
    p.add_argument('--seed', type=int, default=21004)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--limit', type=int, default=0, help='smoke: at most this many chains per context, evenly spaced')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f'{args.output} exists; every table gets a new directory')
    started = time.monotonic()
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False          # FP32 as the robot's --alignment-fp32 forecast
    torch.backends.cudnn.allow_tf32 = False

    def forbidden(*a, **kw):
        raise RuntimeError('offline labels prohibit hardware calls')
    runtime = load_runtime(args.model, TASK, device=str(device), physical_check=forbidden, lease_check=forbidden)
    core = runtime.core.eval().requires_grad_(False)
    lower, upper = (getattr(core.actor, 'position_' + name).detach().float().cpu().numpy() for name in ('lower', 'upper'))
    response = load_response(args.response)
    gate = table_gate(args.floor_mm)
    forecast_of = forecaster(core, bf16=False)

    config = json.loads(args.source_config.read_text())
    if bool(getattr(core.config, 'place_relative', False)) != (args.place_table is not None):
        raise SystemExit('give --place-table exactly when the model reads place cues')
    source = V21Sources(config['source_paths'], device='cpu', object_relative_table=args.object_cues / 'fish_object_cue.npy',
                        place_relative_table=args.place_table)
    hold_px = demo_hold_reference(args.hold_reference)
    jaw_midpoint = calibrated_jaw_midpoint(
        args.wrist_reference, SO101MuJoCoKinematics(json.loads(args.response.read_text())['kinematics_model']))
    install_camera_layout(source, args.camera_layout)
    splits = derive_splits(source)
    rows = splits['train']['fish'] if args.split == 'train' else splits['development']['fish']
    episodes = FISH_TRAIN if args.split == 'train' else FISH_EXAM
    if set(np.unique(np.asarray(source.fish.episodes)[np.asarray(source.fish.starts)[rows]]).tolist()) - set(episodes):
        raise SystemExit('the split holds chains of other demonstrations')
    by_context = moving_rows(source, rows, command_onsets(source))
    bank = ScratchDinoFeatureBank(source, device, 4 * 2**30)
    rng = np.random.default_rng(args.seed)

    out = {name: [] for name in ('rows', 'contexts', 'shifts', 'commands', 'usable', 'proposal_cost', 'selected_cost')}
    skipped, reasons, admissions, records = Counter(), Counter(), Counter(), []
    with torch.inference_mode():
        for row, h, single, proposal, offsets in chain_states(core, source, bank, by_context, device, batch=args.batch,
                                                              limit=args.limit, skipped=skipped):
            q = single.q[0].double().cpu().numpy()
            token = jaw_layout(single.object_relative[0].double().cpu().numpy(), q, hold_px, jaw_midpoint)
            forecast = remembered(forecast_of(single, offsets))
            shifts = draw_shifts(rng, args.shifts, args.shift_std_px)
            commands, usable, plans = teach(q, proposal, token, forecast, admission(gate, q), response, shifts,
                                            lower=lower, upper=upper, seed=args.seed * 7919 + row * 4 + h)
            for k, plan in enumerate(plans):
                reasons[('shift' if k else 'none', str(plan['reason']))] += 1
                admissions[('shift' if k else 'none', str(plan['admission']))] += 1
            out['rows'].append(row); out['contexts'].append(h); out['shifts'].append(shifts)
            out['commands'].append(commands); out['usable'].append(usable)
            out['proposal_cost'].append([np.nan if r['proposal_cost_px'] is None else r['proposal_cost_px'] for r in plans])
            out['selected_cost'].append([np.nan if r['selected_cost_px'] is None else r['selected_cost_px'] for r in plans])
            records.append(dict(row=row, h=h, q=q.tolist(), token=token.tolist(), shifts=shifts.tolist(), plans=plans))
            if len(out['rows']) % 50 == 0:
                print(json.dumps(dict(context=h, labelled=len(out['rows']), elapsed_s=round(time.monotonic() - started, 1))),
                      flush=True)
    if not out['rows']:
        raise SystemExit(f'nothing was labelled: {dict(skipped)}')
    arrays = dict(rows=np.asarray(out['rows'], np.int64), contexts=np.asarray(out['contexts'], np.int8),
                  shifts=np.asarray(out['shifts'], np.float32), commands=np.asarray(out['commands'], np.float32),
                  usable=np.asarray(out['usable'], bool), proposal_cost=np.asarray(out['proposal_cost'], np.float32),
                  selected_cost=np.asarray(out['selected_cost'], np.float32))

    def quantiles(values):
        values = values[np.isfinite(values)]
        return [round(float(v), 2) for v in np.percentile(values, (10, 50, 90))] if len(values) else None
    usable = arrays['usable']
    summary = {}
    for name, columns in (('none', slice(0, 1)), ('shifted', slice(1, None))):
        mask = usable[:, columns]
        moved = (np.abs(arrays['commands'][:, columns, :, :5] - arrays['commands'][:, :1, :, :5]).max((-1, -2))
                 if name == 'shifted' else None)
        summary[name] = dict(
            plans=int(mask.size), usable=int(mask.sum()),
            proposal_cost_px_p10_p50_p90=quantiles(arrays['proposal_cost'][:, columns][mask]),
            selected_cost_px_p10_p50_p90=quantiles(arrays['selected_cost'][:, columns][mask]),
            **({} if moved is None else dict(label_change_vs_unshifted_units_p10_p50_p90=quantiles(moved[mask]))))
    manifest = dict(
        split=args.split, episodes=list(episodes), fish_episode_2_read=False, robot_data_used=False, hardware_used=False,
        model=str(args.model), model_fish_sha256=sha256(args.model / 'fish' / 'final.pt'),
        lewm_state_hash=lewm_state_hash(core.lewm), response=str(args.response), response_sha256=sha256(args.response),
        object_cues=str(args.object_cues),
        object_cues_sha256={name: sha256(args.object_cues / name) for name in ('fish_object_cue.npy',)},
        place_table=None if args.place_table is None else str(args.place_table),
        hold_reference=str(args.hold_reference), hold_reference_sha256=sha256(args.hold_reference),
        hold_px=[float(v) for v in hold_px],
        wrist_reference=str(args.wrist_reference), wrist_reference_sha256=sha256(args.wrist_reference),
        token_layout='the state\'s hold cue moved back to the jaw midpoint at its pose (AlignmentCEMAdapter._jaw_layout)',
        source_config=str(args.source_config), source_config_sha256=sha256(args.source_config),
        camera_layout=str(args.camera_layout), camera_layout_sha256=sha256(args.camera_layout),
        chains_per_context=[int(len(spread(by_context[h], args.limit))) for h in range(4)],
        limit=args.limit, smoke=bool(args.limit),
        teacher=dict(planner='fcpg/v21_alignment_cem.AlignmentCEM.plan, commit=False, steer_approach=True',
                     population=POPULATION, elites=ELITES, iterations=ITERATIONS, prefix=PREFIX, ramp=RAMP.tolist(),
                     max_offset=MAX_OFFSET.tolist(), search_joints=list(SEARCH_JOINTS), min_gain_px=MIN_GAIN_PX,
                     ranked_checks=RANKED_CHECKS, target_px=[0., 0.], close_tolerance_px=15.,
                     proposal='the demonstrator\'s next five commands (horizon targets)',
                     forecast='AlignmentCEMAdapter._forecast, FP32 (the robot\'s --alignment-fp32), TF32 off',
                     admission=f'V17.7 chunk audit with the table gate at {args.floor_mm} mm (no planning clearance)',
                     gripper='the demonstrator\'s commands (the close gate is not taught)',
                     label='the selection\'s five arm joints', states='factual_state in FP32 with the model\'s LeWM'),
        shifts=dict(count=args.shifts, std_px=args.shift_std_px, limit_px=SHIFT_LIMIT_PX, seed=args.seed,
                    first='none'),
        labelled=int(len(arrays['rows'])), skipped=dict(skipped),
        reasons={f'{a}:{b}': n for (a, b), n in sorted(reasons.items())},
        admissions={f'{a}:{b}': n for (a, b), n in sorted(admissions.items())},
        summary=summary, elapsed_s=round(time.monotonic() - started, 1), section=SECTION)
    saved = TeacherLabels(arrays, manifest).save(args.output)
    with (args.output / 'records.jsonl').open('w') as stream:
        for record in records:
            stream.write(json.dumps(record) + '\n')
    print(json.dumps(dict(done=str(args.output), labelled=saved['labelled'], skipped=saved['skipped'],
                          summary=summary, elapsed_s=saved['elapsed_s'])), flush=True)


if __name__ == '__main__':
    main()
