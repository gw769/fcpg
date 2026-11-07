#!/usr/bin/env python3
"""The robot exam of a LeWM: its forecasts on robot blocks never trained on (the 535 blocks of 10/31), with the
view-loss exam the system is about: how the forecast holds up when a camera has not seen for the last T seconds.

Every LeWM reads its memory from the per-session picture streams of the exam pack (fcpg/v21_picture_memory.py
robot_memory_index): a picture-memory LeWM its last N pictures per camera, a three-picture LeWM its last three (with
nothing lost these are exactly the snapshot's own three: checked against the pack). View lost for T seconds on a
camera: that camera's pictures younger than T at the block's time are left out, and the memory holds the ones before
(as the robot service's memory would). Variants: own; none (no pictures); lost_wrist_T, lost_external_T, lost_both_T
for T = 0.5, 1, 2, 4, 8, 16 s. A LeWM with the control-setting input gets each block's own setting. Reported: the
joints' MAE (robot units) and the jaw position error (gripper frame, mm, official SO-101 MuJoCo model) on all / carried /
fast / mid / slow blocks, by arm P, by task, and on the blocks whose memory holds both cameras apart from those that
held the wrist only. Read only.
Usage: python scripts/exam_v21_lewm_robot.py --exam EXAM.pt --memory EXAM-MEMORY.pt --arm-p ARM-P.json
       --output OUT.json --lewm NAME=CHECKPOINT ... [--device cuda]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault('V21_ENCODER', 'dinov3_vitl16')
os.environ.setdefault('V21_DINOV2', 'off')
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fcpg.so101_kinematics import SO101MuJoCoKinematics  # noqa: E402
from fcpg.v21_native_horizon import core_from_parent  # noqa: E402
from fcpg.v21_picture_memory import RobotPictureMemory  # noqa: E402
from fcpg.v21_robot_block_pack import RobotBlocks  # noqa: E402

MODEL = ROOT / 'third_party' / 'so_arm100' / 'Simulation' / 'SO101' / 'so101_new_calib.xml'
BATCH = 16
STILL = 5.
LOSSES = (.5, 1., 2., 4., 8., 16.)
WRIST = 0                     # the pack's slot 0 is the wrist camera, slot 1 the external one


def gripper_positions(kinematics, states):
    states = np.asarray(states, np.float64)
    return kinematics.forward_gripper_poses(states.reshape(-1, 6))[:, :3, 3].reshape(*states.shape[:-1], 3)


def variants():
    yield 'own', None
    for t in LOSSES:
        for name, cams in (('wrist', (WRIST,)), ('external', (1 - WRIST,)), ('both', (0, 1))):
            lost = np.zeros(2)
            lost[list(cams)] = t
            yield f'lost_{name}_{t:g}', lost


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--exam', type=Path, required=True, help='the robot exam pack (exam.pt)')
    p.add_argument('--memory', type=Path, required=True, help="the exam pack's picture streams (exam-memory50.pt)")
    p.add_argument('--arm-p', type=Path, required=True, help='the arm P table of the robot sessions')
    p.add_argument('--lewm', nargs='+', required=True, help='NAME=CHECKPOINT')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--blocks', type=int, default=0, help='the first N blocks only (a check of this script; 0 = all)')
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit('give a new output path')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    kinematics = SO101MuJoCoKinematics(MODEL)
    pack = torch.load(args.exam, map_location='cpu', weights_only=False)
    arm_p = json.loads(args.arm_p.read_text())
    blocks = RobotBlocks(pack, device, None, split='exam', arm_p=arm_p)
    stream = RobotPictureMemory(torch.load(args.memory, map_location='cpu', weights_only=False), device)
    holding = blocks.data['holding'].cpu().numpy()
    p32 = blocks.data['arm_p32'].cpu().numpy()
    tasks = np.asarray(pack['tasks'])[blocks.data['task'].cpu().numpy()]
    size = min(blocks.size, args.blocks) if args.blocks else blocks.size
    actual, start, both = [], [], []
    for begin in range(0, size, BATCH):
        ids = np.arange(begin, min(begin + BATCH, size))
        q, trace, history, qt, measured, memory, now, task_ids = blocks.batch(ids)
        actual.append(measured.double().cpu().numpy())
        start.append(q.double().cpu().numpy())
        both.append(memory.valid[:, 1 - WRIST].any(-1).cpu().numpy())      # the block's memory held the external camera
        three = stream.memory(ids, now, 3)              # the snapshot's own three, rebuilt from the stream
        if not (torch.equal(three.valid, memory.valid) and torch.equal(three.image_times, memory.image_times.double())
                and torch.equal(three.features[three.valid], memory.features[memory.valid])
                and torch.equal(three.q_at_image[three.valid], memory.q_at_image[memory.valid].float())):
            raise ValueError('the streams do not give back the snapshots\' own pictures')
    actual, start, both = np.concatenate(actual), np.concatenate(start), np.concatenate(both)
    holding, p32, tasks = holding[:size], p32[:size], tasks[:size]
    actual_xyz = gripper_positions(kinematics, actual)
    travel = np.linalg.norm(actual_xyz[:, -1] - gripper_positions(kinematics, start), axis=-1) * 1000.
    moving = travel >= STILL
    thirds = np.percentile(travel[moving], [100 / 3, 200 / 3])
    groups = dict(all=np.ones(len(travel), bool), carried=holding, not_carried=~holding, P16=~p32, P32=p32,
                  moving_fast=moving & (travel >= thirds[1]),
                  moving_mid=moving & (travel >= thirds[0]) & (travel < thirds[1]), moving_slow=moving & (travel < thirds[0]),
                  still=~moving, both_cameras=both, both_still=both & ~moving, both_moving=both & moving,
                  wrist_only=~both, wrist_only_still=~both & ~moving, wrist_only_moving=~both & moving,
                  **{f'task:{t}': tasks == t for t in sorted(set(tasks))})
    report = dict(exam=str(args.exam), memory=str(args.memory), blocks=int(size),
                  groups={k: int(v.sum()) for k, v in groups.items()}, losses_s=list(LOSSES), lewms={})
    for spec in args.lewm:
        name, path = spec.split('=', 1)
        lewm = core_from_parent(torch.load(path, map_location='cpu', weights_only=False)).lewm.to(device).eval()
        found = lewm.architecture() if hasattr(lewm, 'architecture') else {}
        n = found.get('memory_pictures', 3)
        paths = {}
        for kind, lost in [*variants(), ('none', 'none')]:
            out = []
            for begin in range(0, size, BATCH):
                ids = np.arange(begin, min(begin + BATCH, size))
                q, trace, history, qt, _, _, now, _ = blocks.batch(ids)
                kwargs = dict(controller=blocks.controller(ids)) if found.get('controller_input') else {}
                if kind != 'none':
                    memory = stream.memory(ids, now, n, None if lost is None else np.broadcast_to(lost, (len(ids), 2)))
                    kwargs.update(patches=memory.features, patch_valid=memory.valid, memory=memory, now_time=now)
                out.append(lewm(q, trace, history, qt, **kwargs).predicted_q_path[:, 1:].double().cpu().numpy())
            paths[kind] = np.concatenate(out)
        result = {}
        for kind, path_k in paths.items():
            mm = np.linalg.norm(gripper_positions(kinematics, path_k) - actual_xyz, axis=-1).mean(1) * 1000.
            error = np.abs(path_k - actual).mean((1, 2))
            result[kind] = {g: dict(mae=round(float(error[m].mean()), 4), jaw_mm=round(float(mm[m].mean()), 2))
                            for g, m in groups.items() if m.any()}
        report['lewms'][name] = dict(checkpoint=path, architecture=found, **result)
        print(name, json.dumps({k: (v['all']['mae'], v['all']['jaw_mm']) for k, v in result.items()}), flush=True)
    args.output.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
