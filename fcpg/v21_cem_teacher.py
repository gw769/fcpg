"""The alignment CEM as the Actor's teacher (training overhaul doc section 17.110; James 10/26 「再训练点什么东西吧
能不能把cem这些玩意儿 训练到acotr+lewm里面」).

On the robot the alignment CEM (fcpg/v21_alignment_cem.py) changed 37 of the 40 searched fish blocks of the 10/26
HLwb trials (17.109 trials 10-13): the Actor's own block would have left the fish about 30 px from the jaw midpoint
(median), the CEM's selection about 4 px. Here the same CEM teaches the Actor to make that choice itself. On the fish
demonstrations' chains, with the state built as in training and the system LeWM forecasting every candidate exactly as
the service does (AlignmentCEMAdapter._forecast, FP32 as --alignment-fp32), the geometric wrist response and the
service's table gate, the CEM searches around the DEMONSTRATOR's next five commands: once with the fish token where it
is, and with the token moved by random shifts (the fish elsewhere in the wrist picture; nothing else of the state
changes). Its selection is the label of the first five arm commands (the fish stage supervises those five, --fish-horizon
5). The gripper keeps the demonstrator's commands: the CEM's close gate (15 px) is stricter than the people who
recorded the demonstrations (0 and 1 close with the fish about 13 px above the jaw midpoint, 3 about 29 px off;
cem-distill-20261026/demo_close_token_20261026.py), so it is not taught.

No robot data: the states are the demonstrations' (0 and 1 train, 3 only as an exam, 2 never read); robot records are
read only by the exam (scripts/exam_v21_cem_student.py).
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .v21_alignment_cem import ARM, IMAGE, OPEN_GRIP, AlignmentCEM, AlignmentCEMAdapter

SCHEMA = 'v21-cem-teacher-labels-v1'
SECTION = '17.110'
SHIFT_LIMIT_PX = 100.          # a moved token stays within this many px of where the fish is, per image axis
ARRAYS = ('rows', 'contexts', 'shifts', 'commands', 'usable', 'proposal_cost', 'selected_cost')


def draw_shifts(rng, count, std_px, limit_px=SHIFT_LIMIT_PX):
    """[count, 2] token shifts in px: the first is none, the others normal with std_px per image axis, clipped."""
    if count < 1 or not (np.isfinite(std_px) and std_px > 0) or not (np.isfinite(limit_px) and limit_px > 0):
        raise ValueError('need at least one shift, a positive spread and a positive limit')
    shifts = np.zeros((count, 2))
    shifts[1:] = np.clip(rng.normal(0., std_px, (count - 1, 2)), -limit_px, limit_px)
    return shifts


def table_gate(floor_mm):
    """The service's table gate with --table-clearance-floor-mm floor_mm and no planning clearance
    (scripts/canary_so101_v21_research.clearance_gate): planned paths, read-backs and the physical floor at floor_mm."""
    import scripts.serve_so101_prod_v17_7 as established
    from .so101_kinematics import SO101MuJoCoKinematics
    floor_mm = float(floor_mm)
    return established.V177ClearanceGate(kinematics=SO101MuJoCoKinematics(established.DEFAULT_KINEMATICS_MODEL),
                                         table_z_mm=established.TABLE_Z_MM, planned_clearance_mm=floor_mm,
                                         readback_clearance_mm=floor_mm, physical_floor_mm=floor_mm)


def admission(gate, q):
    """admissible(commands [5, 6]) -> bool at the pose q: the service's physical check
    (fcpg/prod_v20_3_hardware_service.V203HardwareService._physical_check, the V17.7 chunk audit with the table gate)."""
    from scripts.serve_so101_prod_v17_7 import _audit_v177_chunk
    current = np.asarray(q, np.float32)

    def admissible(commands):
        return bool(_audit_v177_chunk(np.asarray(commands, np.float32), current, clearance_gate=gate).get('safe'))
    return admissible


def forecaster(lewm, bf16=False):
    """forecast_of(one-row state, command offsets [1, 5]) -> forecast(commands [n, 5, 6]) -> [n, 6, 6]: the service's
    own call of the system LeWM (AlignmentCEMAdapter._forecast; bf16=False as the robot's --alignment-fp32). lewm: the
    LeWM, or the runtime's core (11/03: a picture-memory LeWM reads the state's pictures and the arm's control setting
    through the core, lewm_pictures / lewm_controller)."""
    adapter = AlignmentCEMAdapter('shadow', None, bf16=bf16)
    core = lewm if hasattr(lewm, 'lewm_pictures') else SimpleNamespace(lewm=lewm)
    adapter.service = SimpleNamespace(owner=SimpleNamespace(runtime=SimpleNamespace(core=core)))
    return adapter._forecast


def remembered(forecast):
    """The same forecast, computed once per distinct command block (every plan of one state starts with the proposal)."""
    seen = {}

    def cached(commands):
        commands = np.asarray(commands)
        key = commands.shape, commands.tobytes()
        if key not in seen:
            seen[key] = np.asarray(forecast(commands)).copy()
        return seen[key].copy()
    return cached


def teach(q, proposal, token, forecast, admissible, response, shifts, *, lower=None, upper=None, seed=0,
          steer_approach=True):
    """The CEM's selection around the proposal for the token moved by each shift. -> (commands [K, 5, 6]: the
    selection's arm joints with the proposal's gripper; usable [K]: the CEM searched the alignment, i.e. not
    'geometry_unavailable'; records [K]). steer_approach as the robot (17.101, STEER=1); commit=False: no deferral
    state carries from one plan to the next."""
    q = np.asarray(q, np.float64)
    proposal = np.asarray(proposal, np.float64)
    token = np.asarray(token, np.float64).reshape(-1)
    if q.shape != (6,) or proposal.shape != (5, 6) or token.shape != (8,):
        raise ValueError('teach needs q [6], a proposal [5, 6] and a token [8]')
    if not token[0] > .5 or not q[5] >= OPEN_GRIP:
        raise ValueError('the CEM teaches the fish approach only: a seen fish token and open jaws')
    shifts = np.asarray(shifts, np.float64)
    commands = np.repeat(proposal[None], len(shifts), 0)
    usable = np.zeros(len(shifts), bool)
    records = []
    for k, shift in enumerate(shifts):
        moved = token.copy()
        moved[2:4] = moved[2:4] + shift / IMAGE
        planner = AlignmentCEM(response, lower=lower, upper=upper, steer_approach=steer_approach, seed=seed + k)
        selected, record = planner.plan(q, proposal, moved, forecast, admissible, commit=False)
        commands[k, :, :ARM] = selected[:, :ARM]
        usable[k] = bool(record.get('searched'))
        records.append({key: record.get(key) for key in (
            'searched', 'reason', 'admission', 'selected_is_actor', 'proposal_cost_px', 'selected_cost_px', 'offset',
            'close_deferred')})
    return commands, usable, records


class TeacherLabels:
    """The labels of one split: per (fish chain row, context h) K shifts, their label commands and usability."""

    def __init__(self, arrays, manifest):
        self.arrays = {name: np.asarray(arrays[name]) for name in ARRAYS}
        self.manifest = dict(manifest)
        rows, contexts = self.arrays['rows'], self.arrays['contexts']
        n, k = self.arrays['usable'].shape
        if (rows.shape != (n,) or contexts.shape != (n,) or self.arrays['shifts'].shape != (n, k, 2)
                or self.arrays['commands'].shape != (n, k, 5, 6) or k < 1):
            raise ValueError('teacher label arrays disagree in shape')
        if not np.isfinite(self.arrays['commands']).all() or not np.isfinite(self.arrays['shifts']).all():
            raise ValueError('nonfinite teacher labels')
        if (self.arrays['shifts'][:, 0] != 0).any():
            raise ValueError('the first shift of every row must be none')
        self.index = {(int(r), int(h)): i for i, (r, h) in enumerate(zip(rows, contexts))}
        if len(self.index) != n:
            raise ValueError('a (row, context) pair is labelled twice')

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez(directory / 'labels.npz', **self.arrays)
        manifest = dict(self.manifest, schema=SCHEMA, section=SECTION,
                        labels_sha256=_sha256(directory / 'labels.npz'))
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
        return manifest

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest.get('schema') != SCHEMA:
            raise ValueError(f'{directory} is not a CEM teacher label table')
        if _sha256(directory / 'labels.npz') != manifest.get('labels_sha256'):
            raise ValueError(f'{directory}/labels.npz is not the file its manifest describes')
        with np.load(directory / 'labels.npz') as stored:
            return cls({name: stored[name] for name in ARRAYS}, manifest)

    def relabel(self, rows, h, targets, chain, generator, unshifted_share):
        """The training rows (fish chain ids, context h) with labels: the first five arm commands become the
        teacher's selection, and the current wrist token (chain.object_relative[:, h]) moves by its shift.
        unshifted_share of them, drawn per row, keep the fish where it is (shift 0). Applied before any mirror, so
        fcpg/v21_mirror.mirror_rows turns label and token together. -> (targets, chain, labelled, shifted)."""
        if not 0. <= unshifted_share <= 1.:
            raise ValueError('the unshifted share is a share')
        if chain.object_relative is None:
            raise ValueError('the teacher moves the wrist token: the chain needs one')
        commands, cue = targets.commands.clone(), chain.object_relative.clone()
        usable, labels, shifts = self.arrays['usable'], self.arrays['commands'], self.arrays['shifts']
        labelled = shifted = 0
        for i, row in enumerate(np.asarray(rows).tolist()):
            draw, pick = generator.random(), generator.random()      # two draws per row, labelled or not
            j = self.index.get((int(row), int(h)))
            if j is None or not bool(targets.valid[i, :5].all()):
                continue
            moved = np.flatnonzero(usable[j, 1:]) + 1
            if len(moved) and draw >= unshifted_share:
                k = int(moved[min(int(pick * len(moved)), len(moved) - 1)])
            elif usable[j, 0]:
                k = 0
            else:
                continue
            commands[i, :5, :ARM] = torch.as_tensor(labels[j, k, :, :ARM], dtype=commands.dtype)
            cue[i, h, 2:4] = cue[i, h, 2:4] + torch.as_tensor(shifts[j, k] / IMAGE, dtype=cue.dtype)
            labelled += 1
            shifted += int(k > 0)
        return replace(targets, commands=commands), replace(chain, object_relative=cue), labelled, shifted


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()
