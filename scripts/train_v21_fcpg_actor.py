#!/usr/bin/env python3
"""FCPG + Actor training, rebuilt (docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md).

The Actor trained by scripts/train_v21_horizon40.py does not use its views: giving every row another recording's
visual inputs changed its Archive error by 2%, and on the held-out demonstration the error even went down.
What this script does differently:
  A. Archive cameras in the model's order, slot 0 wrist, slot 1 external (fcpg/v21_archive_camera_layout.py);
     recordings without a wrist camera have no slot-0 view; undecided recordings are left out.
  B. The Actor does not see how the arm just moved (V21Config.actor_history=False).
  C. FCPG reads for the task: its queries depend on the task words and the current joints (V21Config.reader_task).
  D. Fish stage: demonstrations 0 and 1, demonstration 3 held out as the exam, FCPG frozen, kept steps evaluated.
  E. Every evaluation also reports how much the proposal depends on the views (another row's visual inputs).
  F. Each Archive batch is used once, with a random context length.
Second round (same doc, section 8), one switch each: --gripper-separate (the jaw opening is not an input and is
output as an opening, not a change), --wrist-only (the external camera is not an input), --balanced-collections
(the large Archive collections get equal shares of the training draws), --queries-per-view 32 --actor-width 384
(a larger model: a longer FCPG summary of each view and a wider Actor).
Manufactured data against memorising recordings (section 9): --mirror shows every training row mirrored left-right
with probability one half (fcpg/v21_mirror.py; Archive and fish, so the fish is also on the other side);
--patch-mask hides a random share of each training picture's patches.
Idle start (section 15): the operator holds still for about 2 s before every demonstration; --trim-idle-start never
draws a fish chain whose current frame is before the first commanded motion, and every evaluation also reports the
exam on demonstration 3 from its first commanded motion on (fish_moving). --archive-checkpoint starts the fish stage
from the saved Archive stage of an earlier run, so that a fish-stage change can be compared on the same weights.
Place token (section 17): --place-table gives the Actor where the object goes (the registered drop zone seen in the
wrist picture) against the jaws as a second 8-number input, added to the pose token while seen
(V21Config.place_relative, scripts/build_v21_place_table.py); from --archive-checkpoint its projection is added at
the fish stage (the Archive has no place token), the only module that may be missing from the loaded weights.
--place-learning-rate-scale (section 17.2) lets that new projection learn at a multiple of the fish learning rate; it
starts at zero output, and at 3e-5 its output barely depended on where the place is (17.1, weights measured).
Place goal (section 17.7, the B route): --place-goal-tables gives the Actor the hindsight place goal against its jaws
in the arm's polar frame (fcpg/v21_place_goal.py, scripts/build_v21_place_goal_tables.py) on Archive rows of
recordings that carry something and on the fish demonstrations, added to the pose token while available
(V21Config.place_goal); its projection starts at zero output from its own generator, so every other module starts as
in the same seed's model without it. --place-goal-learning-rate-scale lets it learn at a multiple of each stage's rate.
--fish-goal-dropout (section 17.10) removes the goal from that share of the fish rows of every fish-stage update (a
fresh draw per row, from its own generator): in the demonstrations "no goal" otherwise only follows the release, and
the Q runs learned to stand still without a goal (section 17.9).
Grasp and place cues (section 17.117): an --object-cues directory of scripts/build_v21_halfsecond_cue_tables.py also
holds archive_place_cue.npy; with --place-table pointing at its fish_place_cue.npy the place token (V21Config.place_relative)
reads where the held thing goes on Archive rows too, against the same reference as the grasp cue. --cue-dropout removes
each of the two cues from that share of the rows of every update, both stages, one draw per row and cue from its own
generator (seed + 11), so the Actor also acts on the pictures alone (the Q and T runs learned to lean on one input).
Object cue from outlines (section 17.30): --object-cues fills the Actor's existing wrist object token
(V21Config.object_relative, the eight numbers of fcpg/v21_object_relative.py) with the outline of the thing to grasp:
Archive rows from the S13 outline labels, fish rows from SAM 3 with the task's object word, both against where the jaws
hold it (fcpg/v21_outline_cue.py, scripts/build_v21_outline_cue_tables.py). No colour: the colour body table of the
source config is then not read. Every evaluation also reports the rows with a cue with the cue removed, and the pan with
the cue's dx moved -/+ CUE_SHIFT.
Two cameras (section 17.52; James 10/21: the Actor should see the column camera and judge for itself): without
--wrist-only the external camera (slot 1) is an input again. --mirror-external lets --mirror flip it with the wrist
picture (the mirrored scene with the robot on the other side, as a camera placed on the other side would see it), and
--view-dropout P hides, per training row, the external camera with probability P and the wrist camera with probability
P (only in rows where both cameras have a picture), for the whole chain, from its own generator, so that neither camera
alone is needed.
Held-back wrist pictures (section 17.63; James 10/22 「都做，插队」): the L Actors act on an old wrist picture as if it
were new (17.62: the joints at its exposure hardly move the proposal). --wrist-hold-back P: in a share P of the training
rows with at least one completed block, the newest one to three wrist pictures do not arrive, so that the wrist memory
still holds the picture of one to three blocks earlier with the joints at its exposure, while the target stays what the
demonstrator did now; the external camera is unchanged. --exam-hold-back (with --evaluate-checkpoint) measures saved weights on
demonstration 3 with its newest wrist pictures withheld, with the joints at their exposure, said to be the current
ones, or zeroed, and without a wrist picture (hold-back-exam.json).
Trained further (section 17.72; James 10/23 「你要不要也多連一臉」, read as 「多練一練」, after the NERO sim's M0,
the same Actor trained 4,000 updates longer): --continue-archive starts the Archive stage from the saved Archive stage
of an earlier formal run with the same model, data and training options, with a new optimizer and new generators,
trains --archive-updates more updates and then the fish stage as usual.
Best development weights (section 17.89; James 10/23 「修好就行阿」 after the formal batch-512 run's development error
rose over its last six passes while its training loss kept falling): --archive-select development hands the fish stage
the evaluated Archive weights with the lowest 40-step development error, the last ones included, and
--archive-patience N ends the Archive stage after N evaluations without a lower one (transformers'
load_best_model_at_end with EarlyStoppingCallback, Keras' EarlyStopping with restore_best_weights).
Re-read target (section 17.95; James 10/24 「做那個能定輸贏的實驗，先設計」, then 「可以阿」): with --wrist-hold-back,
--reread-weight W also trains, in the Archive stage, the wrist read of a held-back row (its older pictures and the
joints at their exposure) towards the read the same row gives with its newest pictures (made without gradient):
reading an old picture by how the arm moved since is rewarded directly. 17.63 trained held-back rows with the action
loss alone, and the models learned to trust old pictures less instead. --memory-address temporal builds a new model
whose read addresses keep age, view and patch position but not the joints at each exposure or their displacement to
now (17.95's comparison model; the paper's ablation, trained this way from the start, and its later steps,
--continue-archive and --archive-checkpoint, are given the same switch).
The alignment CEM as teacher (section 17.110; James 10/26 「能不能把cem这些玩意儿 训练到acotr+lewm里面」): --cem-labels
takes a table of scripts/build_v21_cem_teacher_labels.py (the robot's alignment CEM, with the system LeWM forecasting
every candidate, searched around the demonstrator's next five commands on the fish training chains, with the fish token
as it is and moved by random shifts; fcpg/v21_cem_teacher.py). In the fish stage a labelled fish row learns the CEM's
selection as its first five arm commands, with the wrist token moved by the shift the selection was made for;
--cem-unshifted-share of them keep the fish where it is. The gripper, the Archive half and every evaluation keep the
recorded commands.
--object-token-learning-rate-scale (section 17.111) lets the wrist object token's projection learn at a multiple of the
fish rate: an Archive stage without cues never trains it, and at 3e-5 it hardly answers where the token puts the fish.
--world-head (section 17.113; James 10/27 「训练的时候可以actor+lewn合二为一吗」「做吧」): Actor and world model as one
model. A world head on the FCPG reader the Actor reads (fcpg/v21_world_head.py) learns in both stages what one block
of five recorded commands does to the wrist picture (its frozen DINOv3-L patches in fixed principal components, frame
h + 1 read only for this target) and to the joints; its loss reaches the shared reader in the Archive stage.
Carry lift (section 17.120; James 10/27 「搞好阿 重新训练也可以。 狠狠惩罚这种」 after trial 21 carried the fish 3-5 cm
above the table: the robot model asked for little lift once the box was in the wrist picture, which in the two training
demonstrations shows only when the arm is already high and lowering into it). Fish stage, fish rows that hold the fish
(the jaws read below HELD_GRIP and the grasp cue is on, before --cue-dropout): --carry-lift-weight W adds W times the
mean squared shortfall (per CARRY_SCALE_MM, beyond CARRY_MARGIN_MM) of the jaws' height at the Actor's supervised
commands below the demonstrator's (the service's table-clearance kinematics, linearised at the row's pose); ending
higher costs only the action loss, unless --carry-over-margin-mm M: the excess above the demonstrator's beyond M costs
the same way (robot trial 27: HLtp, punished only below, climbed holding the fish to 47 cm and never let go).
--carry-fish-only keeps the penalty to the fish rows: our arm's clearance at the Archive rows' joints is no height of
theirs (with the excess side on, HLtr's first Archive updates had a loss of 150-1,000 against 1.5).
--box-paste-bank B --box-paste-share P: of the held rows whose demonstrator lifts the
jaws more than PASTE_LIFT_MM in the block, a share P see in every wrist exposure the background patches of a box picture
of the bank (demonstration 0 at the box, newest exposure to newest; the jaws and the fish stay theirs), drawn per row from
its own generator (seed + 13); the label stays the demonstrator's lift. The Archive half and every evaluation are unchanged.
--scene-paste-bank S --scene-paste-share P (James 「我要的是通用场景解决方案」): of the same lifting rows, those the box
paste left alone take with share P the background patches of an entry of a scene bank instead (the Archive's wrist
backgrounds of half of its collections: other people's tables, rooms and light; build_scene_bank_20261028.py), from its
own generator (seed + 14), so the lift does not hang on the room either; the box paste's draws stay as they are.
Fish-stage recipe (section 17.120; James 10/28 「真机器算了，你还是训练最强模型把」「也可以参考nero」: the NERO simulation's
Actor runs rise and then fall to a zero learning rate and keep a weight average): --fish-schedule warmup-linear
--fish-warmup N does for the fish stage what --archive-schedule does for the Archive stage, --fish-ema D evaluates and
saves the average of the fish stage's trained weights (the kept steps and the final ones), and --fish-seed S draws the fish
stage (rows, mirror, cue dropout, patch mask, box paste) from S as --archive-checkpoint draws it from --seed, so that one
Archive stage can take fish stages of several seeds (the seed is part of the model config: HLtq's own fish stage and
HLtqa, the same weights and options on another draw, lifted +36 and +54 mm on the 27 hard robot blocks).
Co-training (section 17.121; James 10/28 「继续」 to learning our two demonstrations with the Archive): over three Archive
seeds the lead's recipe lifted +71, +41 and +29 mm, and no fish-stage change made a poor Archive stage good. With
--archive-fish-share S every Archive-stage update also trains fish rows of demonstrations 0 and 1 (the fish stage's rows
at the update's context, its horizon and its augmentations: mirror, patch mask, cue dropout and, with --box-paste-bank,
the box paste), as many beside the --batch-size Archive rows as make them a share S of the rows, every row weighing the
same (Mobile ALOHA co-trains this way at an equal share; the sim-and-real recipe of Maddukuri et al. 2025 finds a small
share of the scarce data better, 10% its default). The co-trained rows draw from their own generators (numpy seed + 21
rows and mirror, + 22 box paste; patch mask seed + 23, cue dropout seed + 24) and give the RNG of the model's dropout back
as they found it, so the Archive rows, their masks and cue drops are those of the same run without them; the Archive
rows of this stage get no carry rows (only fish rows are pasted). The fish stage is unchanged.
The LeWM is the deployed full-patch model, frozen; as before it only rebuilds the factual history. No robot I/O.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, fields, replace
import hashlib
import json
from pathlib import Path
import multiprocessing
import os
import random
import re
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcpg.v21_archive_camera_layout import install as install_camera_layout
from fcpg.v21_cem_teacher import TeacherLabels
from fcpg.v21_world_head import WorldHead, change_loss
from fcpg.regularization import SIGReg
from fcpg.v21_checks import Pending, deferred, fail_if, later
from fcpg.v21_compiled import regions as compiled_regions, start_compile_workers
from fcpg.v21_core import V21Config, V21ResearchCore, CONTRACT, POSITION_UNIT
from fcpg.v21_encoder import DIM, ENCODER, PATCHES
from fcpg.v21_data import V21Sources
from fcpg.v21_dequantize import dequantize, dequantizes_on_device
from fcpg.v21_direct_rows import DirectRows
from fcpg.v21_mirror import flipped_features, mirror_rows
from fcpg.v21_full_patch_lewm import V21FullPatchLeWM, architecture_from_state
from fcpg.v21_horizon_precision import compute_context
from fcpg.v21_horizon_training import horizon_targets, horizon_metadata, factual_state, actor_loss, horizon_errors
from fcpg.v21_models import ARMOutput
from fcpg.v21_scratch_features import ScratchDinoFeatureBank
from fcpg.v21_temporal_contract import (begin_training_temporal_contract, constrain_training_age_columns,
                                        PROGRESS_TRAINING_POLICY, temporal_actor_state)
from fcpg.v21_training import write_json
from scripts.train_v21_from_scratch import configure_device, derive_splits, stage_chain_to_device, training_loss_scale

KIND = 'v21-fcpg-actor-overhaul-v1'
VERSION = '21.0.4'
FISH_TRAIN, FISH_EXAM = (0, 1), (3,)
VISION_SWAP_RULE = ('every row sees the visual inputs (retained DINO features where both rows have the exposure, '
                    'visual update, wrist object cue) of the row half a batch away; pose, history, text, times unchanged')
GOAL_TURN_SHIFT = .4      # rad: the place goal's turn moved by -/+ this for the steering check (section 17.7)
GOAL_TURN_AT = 4          # command index scored (0.5 s)
GOAL_TURN_MOVED = .5      # pan units: a pair counts as moved above this
GOAL_GRASP = 3            # fcpg/v21_place_goal.GRASP: the goal token's grasp-target number (section 17.14)
CUE_SHIFT = .1            # long sides: the object cue's dx moved -/+ this for the steering check (section 17.30)
CUE_NEAR = .05            # long sides: rows whose cue is at least this far from where it is held (the approach)
ONSET_UNITS = 1.
ONSET_RULE = ('first frame at which any commanded joint differs from the demonstration\'s first command by more than '
              f'{ONSET_UNITS:g} position unit')
JAW_OPENING = 1.          # section 17.46: a demonstration gripper command this far above the reading opens the jaws
ARM = 5                   # the five arm joints; the sixth is the gripper
HOLD_BACK_BLOCKS = 3      # section 17.63: at most this many newest wrist pictures held back (15 readbacks, the oldest
                          # picture the Actor may still use: fcpg/v21_temporal_contract.MAX_VISUAL_PROGRESS_SAMPLES)
WRIST_BODY_FIELDS = 24    # section 17.62: a picture's body-state fields at the start of the reader's frame metadata
HELD_GRIP = 12.           # section 17.120: a fish row holds the fish when the jaws read below this and the grasp cue is on
CARRY_MARGIN_MM = 2.      # --carry-lift-weight: a shortfall below the demonstrator's jaw height up to this costs nothing
CARRY_SCALE_MM = 10.      # --carry-lift-weight: the shortfall beyond the margin is squared per this many mm
PASTE_LIFT_MM = 10.       # --box-paste-share: held rows whose demonstrator lifts the jaws more than this in the block
BOX_BANK_SCHEMA = 'v21-box-background-bank-v1'


def clearance_gradient(gate, q, step=.05):
    """section 17.120: the jaws' height above the table (gate.clearance_mm) per unit of each arm joint at q [6]."""
    q = np.asarray(q, np.float64)
    out = np.zeros(ARM)
    for joint in range(ARM):
        up, down = q.copy(), q.copy()
        up[joint] += step
        down[joint] -= step
        out[joint] = (gate.clearance_mm(up) - gate.clearance_mm(down)) / (2 * step)
    return out


def carry_rows(gate, q, cue, commands, *, shut=True):
    """section 17.120: per row of a batch (numpy): q [B, 6] the read-back now, cue [B, 8] the grasp cue now (None: no
    cue), commands [B, >=5, 6] the demonstrator's next commands -> (held [B] bool, gradient [B, 5] mm per unit at q,
    zero where not held, lift [B] mm: the jaws' height at the fifth command over now, linearised, 0 where not held).
    Held: the grasp cue on and, with shut (fish rows: their cue also marks the fish before the grasp), the jaws reading
    below HELD_GRIP; Archive rows (shut=False) have the cue on held frames only (the S13 labels) and grippers of other
    calibrations."""
    q = np.asarray(q, np.float64)
    held = (np.zeros(len(q), bool) if cue is None else np.asarray(cue)[:, 0] > .5) & ((q[:, 5] < HELD_GRIP) | (not shut))
    gradient = np.zeros((len(q), ARM))
    for row in np.flatnonzero(held):
        gradient[row] = clearance_gradient(gate, q[row])
    lift = ((np.asarray(commands, np.float64)[:, 4, :ARM] - q[:, :ARM]) * gradient).sum(-1)
    return held, gradient, lift


def paste_background(features, h, rows, bank, picks, background):
    """section 17.120: features [B, F, 2, P, D] with the wrist exposures h, h - 1, h - 2 (those there are) of each row
    in rows [n] taking the background patches of bank entry picks [n] ([N, 3, len(background), D], newest first)."""
    out = features.clone()
    rows = torch.as_tensor(rows, device=features.device, dtype=torch.long)
    picks = torch.as_tensor(picks, device=features.device, dtype=torch.long)
    background = torch.as_tensor(background, device=features.device, dtype=torch.long)
    for k in range(min(3, h + 1)):
        out[rows[:, None], h - k, 0, background[None, :]] = bank[picks, k].to(device=features.device, dtype=features.dtype)
    return out


def carry_lift_penalty(action, commands, valid, gradient, held, factor=None, over_margin_mm=None):
    """section 17.120: over held rows' valid commands, the mean squared shortfall (per CARRY_SCALE_MM, beyond
    CARRY_MARGIN_MM) of the jaws' height at the Actor's commands below the demonstrator's, linearised by the row's
    gradient; with over_margin_mm (--carry-over-margin-mm) also the excess above the demonstrator's beyond it, the same
    way (HLtp, below only, learned to climb while holding: 47 cm on robot trial 27); each row's cost times factor [B]
    (None: 1; --carry-place-factor for rows that see where the thing goes).
    action, commands [B, n, 6]; valid [B, n] bool; gradient [B, 5] mm per unit; held [B] bool."""
    shortfall = ((commands[..., :ARM] - action[..., :ARM]) * gradient[:, None, :]).sum(-1)
    cost = (torch.relu(shortfall - CARRY_MARGIN_MM) / CARRY_SCALE_MM).square()
    if over_margin_mm is not None:
        cost = cost + (torch.relu(-shortfall - over_margin_mm) / CARRY_SCALE_MM).square()
    if factor is not None:
        cost = cost * factor[:, None].to(cost.dtype)
    weight = (valid & held[:, None]).to(cost.dtype)
    return (cost * weight).sum() / weight.sum().clamp_min(1.)


def teacher_jaw_targets(target, reading, teacher_jaw):
    """Section 17.46 (--fish-jaw-teacher): the demonstration targets with the gripper part from the teacher (the model as
    the fish stage starts, frozen), except where the demonstrator opens the jaws (command at least JAW_OPENING above the
    reading), which stays the demonstration's: the Archive decides how and when to grip, the demonstrations where to let
    go. reading [b]; teacher_jaw [b, n]. -> (targets, opening [b, n] bool)"""
    human = target.commands[..., ARM]
    opening = human >= reading[:, None].to(human) + JAW_OPENING
    jaw = torch.where(opening, human, teacher_jaw.detach().to(human))
    return replace(target, commands=torch.cat((target.commands[..., :ARM], jaw[..., None]), -1)), opening


def jaw_absolute_loss(absolute, target, scale):
    """Section 17.46 (--gripper-absolute-aux): the auxiliary absolute-jaw prediction against the gripper targets, in
    actor_loss's form for one joint (smooth L1 in loss-scale units; the prefix and the whole supervised future)."""
    values = torch.nn.functional.smooth_l1_loss(absolute[..., 0] / scale[ARM], target.commands[..., ARM] / scale[ARM],
                                                reduction='none')
    valid = target.valid.to(values)
    whole = (values * valid).sum() / valid.sum().clamp_min(1)
    prefix = (values[:, :5] * valid[:, :5]).sum() / valid[:, :5].sum().clamp_min(1)
    return (whole + prefix) / 2


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def command_onsets(source):
    """Fish frame index of each demonstration's first commanded motion (ONSET_RULE); never reads episode 2."""
    episodes = np.asarray(source.fish.episodes)
    onsets = {}
    for demo in (*FISH_TRAIN, *FISH_EXAM):
        frames = np.flatnonzero(episodes == demo)
        source.fish.validate_frames(frames)
        commands = np.asarray(source.fish.actions[frames], np.float64)
        moved = np.flatnonzero(np.abs(commands - commands[0]).max(1) > ONSET_UNITS)
        if not len(moved):
            raise ValueError(f'demonstration {demo} never commands a motion')
        onsets[demo] = int(frames[moved[0]])
    return onsets


def moving_rows(source, rows, onsets):
    """Per context h, the fish chains whose current frame (chain start + 15 h) is at or after their demonstration's
    first commanded motion."""
    starts = np.asarray(source.fish.starts)[rows]
    onset = np.asarray([onsets[int(e)] for e in np.asarray(source.fish.episodes)[starts]], np.int64)
    return {h: rows[starts + 15 * h >= onset] for h in range(4)}


def saved_model_config(config):
    """A checkpoint's or protocol's model config with the defaults of switches added after it was saved (section 17.75:
    actor_layers), so that an earlier run compares equal to the same model built now."""
    return None if config is None else {**asdict(V21Config(vocabulary_size=4)), **{k: v for k, v in config.items()
                                                                                   if k != 'vocabulary_size'},
                                        'vocabulary_size': config.get('vocabulary_size')}


def archive_learning_rate_factor(step, total, warmup, schedule):
    """Share of --learning-rate at Archive update `step` (1-based) of `total`: 'constant' keeps 1; 'warmup-linear'
    (section 17.74, BERT's schedule) rises linearly over the first `warmup` updates, then falls linearly to 0 at `total`."""
    if schedule == 'constant':
        return 1.
    if not 0 < warmup < total:
        raise ValueError('--archive-warmup must be at least 1 and fewer than the Archive updates')
    if step <= warmup:
        return step / warmup
    return max(0., (total - step) / (total - warmup))


def fish_stage_generators(seed, mask_generator):
    """Section 17.120: the fish stage's draws from `seed` as --archive-checkpoint makes them from --seed (rows, mirror and
    cue dropout from numpy seed + 4, the patch mask seed + 5, torch seed + 6) and the box and scene pastes' (numpy seed
    + 13 and + 14, as they are made before the stages). Seeds mask_generator and torch; returns the three numpy
    generators."""
    mask_generator.manual_seed(seed + 5)
    torch.manual_seed(seed + 6)
    return np.random.default_rng(seed + 4), np.random.default_rng(seed + 13), np.random.default_rng(seed + 14)


def scene_paste_rows(lifting, box_pasted, draws):
    """Section 17.120 --scene-paste-share: the lifting held rows the box paste left alone that drew the scene paste."""
    return np.asarray(lifting, bool) & ~np.asarray(box_pasted, bool) & np.asarray(draws, bool)


def cotraining_rows(share, batch_size):
    """Section 17.121 --archive-fish-share: the fish rows beside batch_size Archive rows that make them `share` of the
    rows of an Archive-stage update (every row weighing the same)."""
    return int(round(share * batch_size / (1 - share))) if share > 0 else 0


def cotraining_generators(seed, device):
    """Section 17.121: the co-trained fish rows' own draws, apart from every generator of the Archive stage: rows and
    mirror (numpy seed + 21), box paste (numpy seed + 22), patch mask (seed + 23), cue dropout (seed + 24) and the
    model's dropout (torch's global RNG as seed + 25 leaves it, for dropout_stream)."""
    mask, cue = torch.Generator(device=device), torch.Generator(device=device)
    mask.manual_seed(seed + 23)
    cue.manual_seed(seed + 24)
    cuda = torch.device(device).type == 'cuda'
    with torch.random.fork_rng(devices=[device] if cuda else []):
        torch.manual_seed(seed + 25)
        dropout = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device) if cuda else None)
    return dict(rows=np.random.default_rng(seed + 21), paste=np.random.default_rng(seed + 22), mask=mask, cue=cue,
                dropout=dropout)


@contextmanager
def dropout_stream(stream, device):
    """Section 17.121: inside, torch's global RNG (the model's dropout) continues `stream` (the states its rows left
    the last time); outside, it is as it was before, so the Archive rows' dropout draws stay as without them."""
    cuda = torch.device(device).type == 'cuda'
    outer = torch.get_rng_state(), (torch.cuda.get_rng_state(device) if cuda else None)
    torch.set_rng_state(stream['cpu'])
    if cuda:
        torch.cuda.set_rng_state(stream['cuda'], device)
    try:
        yield
    finally:
        stream['cpu'] = torch.get_rng_state()
        if cuda:
            stream['cuda'] = torch.cuda.get_rng_state(device)
        torch.set_rng_state(outer[0])
        if cuda:
            torch.cuda.set_rng_state(outer[1], device)


def planned_archive_updates(rows, window_rows, batch_size, passes, cap=None, pool=1):
    """Archive-stage updates: every pass runs each locality window of window_rows rows in batches of batch_size;
    --archive-updates caps the total. A cap the passes cannot reach is refused (section 17.58: HL2 asked for 6,960
    updates with 2 passes and trained 4,024 without any message). Section 17.82: with a pool of several windows every
    batch but the last is full."""
    if pool > 1:
        total = -(-passes * rows // batch_size)
    else:
        total = passes * sum((min(window_rows, rows - i) + batch_size - 1) // batch_size
                             for i in range(0, rows, window_rows))
    if cap is None:
        return total
    if total < cap:
        raise ValueError(f'--archive-passes {passes} over {rows} Archive rows gives {total} updates, fewer than '
                         f'--archive-updates {cap}: raise --archive-passes')
    return cap


def store_blocks(source, rows):
    """Per Archive row, the feature-store block holding its current picture, as one int64 key (cache, kind, block):
    the half-second DINOv3-L store is compressed in blocks of 64 pictures, so reading one picture reads its block."""
    endpoint = np.asarray(source.archive.row_endpoint_index)[np.asarray(rows, np.int64), 0]
    key = np.full(len(endpoint), -1, np.int64)
    for c, cache in enumerate(source.archive_dino):
        identities = np.asarray(cache.endpoint_index)
        index = np.minimum(np.searchsorted(identities, endpoint), len(identities) - 1)
        hit = (identities[index] == endpoint) & (key < 0)
        tokens = cache.tokens
        if hasattr(tokens, 'source_row'):
            kind = np.asarray(tokens.kind)[index[hit]].astype(np.int64)
            block = np.asarray(tokens.source_row)[index[hit]] // tokens.block_frames
        else:                                            # an uncompressed store: neighbouring rows still share pages
            kind, block = np.zeros(int(hit.sum()), np.int64), index[hit] // 64
        key[hit] = (c * 2 + kind) * (1 << 40) + block
    if (key < 0).any():
        raise ValueError('Archive rows without a stored current picture')
    return key


class ArchivePool:
    """--fish-archive-pool (section 17.59): the fish stage's Archive half comes from a rolling pool of store blocks.

    Drawing each Archive row independently over the whole Archive reads one 12 MB block from the hard disk per row
    (HL2b: 7.6 s per fish update with the A100 idle). Here every block is equally likely to enter the pool, the rows
    of the pool are drawn uniformly, and after every update the `refresh` oldest blocks are replaced: every row stays
    about as likely as before (a row's chance is its block's chance to be in the pool times the update's share of the
    pool's rows), while an update reads about `refresh` new blocks. Rows of one block are 0.5 s steps of one or two
    recordings; an update's rows come from `size` blocks."""

    def __init__(self, rows, keys, size, refresh, rng):
        if not 1 <= refresh <= size:
            raise ValueError('--fish-archive-pool-refresh must be between 1 and --fish-archive-pool')
        order = np.argsort(keys, kind='stable')
        self.rows = np.asarray(rows)[order]
        _, self.start, self.count = np.unique(np.asarray(keys)[order], return_index=True, return_counts=True)
        self.size, self.refresh, self.rng = min(size, len(self.count)), refresh, rng
        self.pool = [int(b) for b in rng.choice(len(self.count), self.size, replace=False)]
        self.blocks_entered = self.size
        self.entered = []

    def draw(self, n):
        members = np.concatenate([self.rows[self.start[b]:self.start[b] + self.count[b]] for b in self.pool])
        drawn = self.rng.choice(members, n, replace=True)
        self.entered = []
        for _ in range(min(self.refresh, len(self.count) - self.size)):
            self.pool.pop(0)
            while True:
                block = int(self.rng.integers(len(self.count)))
                if block not in self.pool:
                    break
            self.pool.append(block)
            self.entered.append(block)
            self.blocks_entered += 1
        return drawn

    def entered_rows(self):
        """The rows of the blocks that entered the pool at the last draw: the next draws may take them."""
        if not self.entered:
            return self.rows[:0]
        return np.concatenate([self.rows[self.start[b]:self.start[b] + self.count[b]] for b in self.entered])


def warm_archive_blocks(source, rows, device_cache=None):
    """Section 17.65: decode the store blocks that hold every picture of these Archive rows into the store's
    decoded-block cache (fcpg/v21_p0_features.py), on a background thread while the GPU trains, so that the next fish
    updates do not wait for the disk. Only a cache is filled: the values and every draw are unchanged. When the feature
    bank's row cache ``device_cache`` multiplies an int8 store's rows on its GPU (fcpg/v21_dequantize.py), the blocks are
    kept as int8 values and scales, the form that path reads."""
    if not len(rows):
        return
    endpoint = np.unique(np.asarray(source.archive.row_endpoint_index[np.asarray(rows, np.int64)]))
    for cache in source.archive_dino:
        if not hasattr(cache.tokens, 'source_row'):      # only the compressed store keeps decoded blocks
            continue
        index = np.minimum(np.searchsorted(cache.endpoint_index, endpoint), len(cache.endpoint_index) - 1)
        hit = np.asarray(cache.endpoint_index)[index] == endpoint
        if hit.any():
            picked = np.asarray(cache.rows_for(endpoint[hit]), np.int64)
            if dequantizes_on_device(device_cache, cache.tokens):
                cache.tokens.quantized(picked)
            else:
                cache.tokens[picked]


def first_commands(target, n):
    """The same factual labels cut to their first n commands (--fish-horizon 5: the fish stage supervises 0.5 s).
    The recorded command trace is kept whole; the Actor loss reads only commands, valid and the command times."""
    if n == target.commands.shape[1]:
        return target
    if not 1 <= n < target.commands.shape[1]:
        raise ValueError('can only cut labels to a shorter, non-empty horizon')
    return replace(target, command_offsets=target.command_offsets[:, :n], readback_offsets=target.readback_offsets[:, :n],
                   commands=target.commands[:, :n], readbacks=target.readbacks[:, :n], valid=target.valid[:, :n])


def vision_swapped(state):
    """The same factual state, but with another row's visual inputs."""
    b = len(state.q)
    other = torch.roll(torch.arange(b, device=state.q.device), b // 2)
    memory = state.memory
    both = memory.valid & memory.valid[other]
    features = torch.where(both[..., None, None], memory.features[other], memory.features)
    relative = None if state.object_relative is None else state.object_relative[other]
    place = {} if getattr(state, 'place_relative', None) is None else dict(place_relative=state.place_relative[other])
    return replace(state, memory=replace(memory, features=features), visual_error=state.visual_error[other],
                   visual_feedback_available=state.visual_feedback_available[other], object_relative=relative, **place)


@torch.no_grad()
def vision_swap_errors(prediction, swapped, target, valid, q):
    b = len(q)
    other = torch.roll(torch.arange(b, device=q.device), b // 2)
    errors = horizon_errors(swapped, target, valid)
    weights = (valid[:, :5] & valid[other, :5]).to(prediction)[..., None]
    count = weights.sum().clamp_min(1)
    change = ((swapped - prediction).abs()[:, :5] * weights).sum((0, 1)) / count
    moves = target - q[:, None]
    recorded = ((moves[other] - moves).abs()[:, :5] * weights).sum((0, 1)) / count
    # section 17.46: the same first-five errors on the five arm joints only (17.43: the jaw part inflates the measure
    # for models that read their jaw from the picture)
    mask = valid[:, :5].to(prediction)[..., None]
    arm = lambda p: float(((p - target).abs()[:, :5, :ARM] * mask).sum() / (mask.sum() * ARM).clamp_min(1))
    return dict(examples=errors['examples'], swap_first5_mae=errors['first5_mae'], swap_future_mae=errors['future_mae'],
                change_first5=float(change.mean()), change_first5_joint=change.tolist(),
                recorded_difference_first5=float(recorded.mean()), recorded_difference_first5_joint=recorded.tolist(),
                first5_mae_arm=arm(prediction), swap_first5_mae_arm=arm(swapped))


def cpu_roles(main=2):
    """Section 17.82: (CPUs of the training thread, CPUs of everything else). The .102 CPU is a hybrid: 8 fast cores
    with two threads each (5.1 GHz) and 16 slow cores (3.9 GHz). The training thread only launches GPU work from
    Python, so its speed is the update's speed: it gets ``main`` fast cores of its own (their second threads stay
    idle); readers, decoders and the batch builder run on the rest. Fast = the highest maximum clock."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
        top = {c: int(Path(f'/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq').read_text()) for c in cpus}
        siblings = {c: Path(f'/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list').read_text().strip()
                    for c in cpus}
    except (OSError, ValueError):
        return None, None
    fast = max(top.values())
    chosen, used = [], set()
    for c in sorted(cpus, key=lambda c: (-top[c], c)):          # the fastest first (5.2 GHz before 5.1 GHz)
        if top[c] >= fast * .97 and siblings[c] not in used and len(chosen) < main:
            chosen.append(c)
            used.add(siblings[c])
    held = {c for c in cpus if siblings[c] in used}
    rest = [c for c in cpus if c not in held]
    return (set(chosen), set(rest)) if len(chosen) == main and len(rest) >= 4 else (None, None)


_CPUS = {}              # section 17.82: 'workers' -> the CPUs readers, decoders and builders run on


def on_worker_cpus():
    """The calling thread or process leaves the training thread's CPUs alone (section 17.82)."""
    if _CPUS.get('workers'):
        os.sched_setaffinity(0, _CPUS['workers'])


def _end_with_the_trainer():
    """Section 17.83: a forked reading or building process ends when the trainer ends, however it ends (Linux
    PR_SET_PDEATHSIG); stopped from outside, the trainer otherwise left them running."""
    try:
        import ctypes
        import signal
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGTERM)       # 1 = PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


_BUILDING = {}          # section 17.81: what the batch-building process runs, set before it is forked


def _build_in_process(*args):
    """prepare_host() in the batch-building process (forked from the trainer; it never touches the GPU)."""
    return _BUILDING['prepare'](*args)


def _start_building_process():
    _end_with_the_trainer()
    torch.set_num_threads(1)        # a forked process must not use the trainer's OpenMP threads
    on_worker_cpus()


class NoForecast(torch.nn.Module):
    """Stands in for the frozen LeWM while training states of a history-free model are built
    (V21Config.actor_history=False). The forecast only becomes the execution-feedback input (q_error); that Actor
    and its FCPG read addresses do not read it, so the proposal is the same (checked once per run) and four
    full-patch forward passes per batch are saved. The real LeWM stays in the core everywhere else. Its answer does
    not depend on the clock (retime_invariant), so a retimed completion keeps it without asking the GPU (17.80)."""
    reads_visual_memory = False
    retime_invariant = True

    def forward(self, q_position, trace, history, query_offsets, **visual):
        path = q_position[:, None].expand(-1, query_offsets.shape[1] + 1, -1).clone()
        return ARMOutput(path, torch.ones_like(path), path.new_zeros(*path.shape[:2], 1))


@contextmanager
def lewm_replaced(core, module):
    real = core.lewm
    core.lewm = module
    try:
        yield
    finally:
        core.lewm = real


@contextmanager
def final_read_only(core, h):
    """While a factual state with h completed blocks is rebuilt, the FCPG reads only in the last observation.

    Every complete() discards the previous visual event and the last observation computes the grounding and the
    visual update afresh, so the reads of the h earlier observations never reach the Actor; inside, they return
    zeros and save up to six of eight reader passes per row. The state the Actor sees is the same (checked once
    per run against fcpg.v21_horizon_training.factual_state)."""
    calls = [0]
    real_observe = core.observe

    def zeros(memory, actual_q, now, history, *, address_mode=None, task_ids=None):
        return actual_q.new_zeros(len(actual_q), 2, core.config.queries_per_view, 192)

    def observe(*args, **kwargs):
        calls[0] += 1
        if calls[0] == h + 1:
            core.reader.__dict__.pop('forward', None)
        return real_observe(*args, **kwargs)

    core.reader.forward = zeros
    core.observe = observe
    try:
        yield
        if calls[0] != h + 1:
            raise ValueError('unexpected number of observations while rebuilding the factual state')
    finally:
        core.reader.__dict__.pop('forward', None)
        core.__dict__.pop('observe', None)


def pool_counts(remaining, batch_size, batches_per_window):
    """Section 17.82: how many rows the next batch takes from each window of the pool, oldest first. A window with no
    more than an even share left gives all of it (it does not linger); otherwise window j (0 = the oldest) is due to
    leave after (j + 1) x batches_per_window batches and gives remaining_j / (j + 1) shares. The counts add up to the
    batch (or to all that is left) and none takes more than its window has left. With one window: the next batch of
    it, as before the pool."""
    remaining = np.asarray(remaining, np.int64)
    total = min(int(batch_size), int(remaining.sum()))
    if len(remaining) == 1:
        return [total]
    small = remaining <= total // len(remaining)
    counts = np.where(small, remaining, 0)
    want = np.where(small, 0., remaining / (np.arange(len(remaining)) + 1.) / batches_per_window)
    if want.sum() > 0:
        want = counts + want / want.sum() * (total - counts.sum())
        counts = np.minimum(np.maximum(np.floor(want).astype(np.int64), counts), remaining)
    while counts.sum() < total:                      # the largest shortfall first, among windows with rows left
        room = counts < remaining
        counts[int(np.argmax(np.where(room, want - counts, -np.inf)))] += 1
    return counts.tolist()


def pool_batches(windows, passes, pool_windows, batch_size, window_rows, readers, rng, mirror, redraw=None):
    """Section 17.82: every draw of the Archive stage in order. A window enters the pool when there is room (its rows in
    a shuffled order) and leaves once its last row is in a batch: every row once per pass (James 「用过就淘汰」). Each
    batch takes from every window in the pool (pool_counts); the context and the mirror draws follow. With one window
    this is the schedule before the pool exactly: per pass the window order, per window its row order, per batch the
    context and the mirror draws. redraw(): new windows at the start of every pass (balanced draws). Yields (pass,
    entering [(key, rows)], leaving [key], following [(key, rows)] to read ahead, rows, context, mirror draws)."""
    pool, queue, epoch = [], deque(), -1
    while True:
        entering = []
        while len(pool) < pool_windows:
            if not queue:
                if epoch + 1 >= passes:
                    break
                epoch += 1
                if redraw is not None:
                    windows = redraw()
                queue.extend((epoch, int(w)) for w in rng.permutation(len(windows)))
            key = queue.popleft()
            pool.append([key, rng.permutation(windows[key[1]]), 0])
            entering.append((key, windows[key[1]]))
        if not pool:
            return
        counts = pool_counts([len(r) - c for _, r, c in pool], batch_size, max(1, window_rows // batch_size))
        rows = np.concatenate([r[c:c + n] for (_, r, c), n in zip(pool, counts)])
        batch_epoch = min(key[0] for (key, _, _), n in zip(pool, counts) if n)
        for slot, n in zip(pool, counts):
            slot[2] += n
        leaving = [key for key, r, c in pool if c == len(r)]
        pool = [slot for slot in pool if slot[2] < len(slot[1])]
        following = [(key, windows[key[1]]) for key in list(queue)[:readers]]
        h = int(rng.integers(4))
        drawn = rng.random(len(rows)) < .5 if mirror else None
        yield batch_epoch, entering, leaving, following, rows, h, drawn


_READING = {}           # section 17.82: the window reader the reading processes use, set before they are forked


def _read_in_process(rows):
    """WindowFeatures._read(rows) in a reading process (forked from the trainer; it never touches the GPU): the int8
    values and scales are written into shared memory and reach the trainer without a copy."""
    return _READING['features']._read(rows, shared=True)


def _start_reading_process(decode_threads):
    _end_with_the_trainer()
    torch.set_num_threads(1)
    on_worker_cpus()
    for cache in _READING['features'].source.archive_dino:
        tokens = cache.tokens
        if hasattr(tokens, '_pool'):    # the store's decoding threads stayed in the trainer: this process makes its own
            tokens._pool, tokens._lock, tokens.decode_threads = None, threading.Lock(), decode_threads


def _reading_process_started(_):
    time.sleep(.5)                      # every reader busy at once, so that all of them are forked now
    return os.getpid()


class WindowFeatures:
    """The exposures of one locality window of Archive chains, read from the frozen DINO cache in one ascending
    (near-sequential) pass and held on the GPU; the window's batches are then drawn from it in shuffled order.
    Same values, validity and camera layout as the per-batch row cache. ``prefetch`` reads the next window on a
    background thread while the current one trains, so the GPU does not wait for the disk. On a GPU an int8 store's
    rows arrive as int8 values and scales and are multiplied there (section 17.65: the float16 conversion was the
    slowest part of a window on the CPU). Section 17.78: on a GPU the background thread also copies the next window
    into page-locked memory, so that load() moves it to the GPU without waiting; load() multiplies it out, checks it
    and lays it out 256 rows at a time (the GPU holds the window and one chunk, not four copies of it). Section 17.79:
    ``readers`` threads read the next windows at the same time, the int8 values are written straight into page-locked
    arrays, and ``prefetch`` takes the windows that follow (as many as there are readers). On a GPU the window after
    the current one is also placed on the GPU, on a side stream, as soon as it is read (James: the GPU memory as the
    store of the next stage), so a window switch only waits for its event. Section 17.82: on a GPU the windows are read
    by ``readers`` separate processes, forked by start_readers() before the Archive stage starts any other thread, so
    reading never holds this process's interpreter lock (at 17.81 the training thread still waited for it about half
    of the time); their int8 values and scales arrive in shared memory. The GPU holds a pool of windows: admit() puts
    one in (placed ahead on the side stream when it was read ahead), release() takes it out once its last row has
    been trained, and features() takes every row from the window that holds it. load() reads one window now (the
    development rows)."""

    def __init__(self, source, device, readers=1):
        self.source, self.device, self.readers = source, device, readers
        self.unique = self.tokens = self.valid = None         # load(): one window
        self.resident = {}                                    # admit(): key -> (unique, tokens, valid), the pool
        self._reader = None                                   # start_readers(): processes on a GPU, else threads
        self._ahead = deque()                                 # (key, rows, future) of the windows being read, in order
        self._placer = ThreadPoolExecutor(1) if device.type == 'cuda' else None
        self._stream = torch.cuda.Stream(device) if device.type == 'cuda' else None
        self._placed = None                                   # (key, future) of the next window placed on the GPU
        self.prefetched = False
        self.waited = 0.

    def start_readers(self, decode_threads=2):
        """Start the readers: on a GPU ``readers`` processes, each decoding with ``decode_threads`` threads of its own,
        all forked now; otherwise threads."""
        if self._reader is not None:
            return
        if self.device.type == 'cuda':
            _READING['features'] = self
            self._reader = ProcessPoolExecutor(self.readers, mp_context=multiprocessing.get_context('fork'),
                                               initializer=_start_reading_process, initargs=(decode_threads,))
            list(self._reader.map(_reading_process_started, range(self.readers)))
        else:
            self._reader = ThreadPoolExecutor(self.readers)

    def _read(self, rows, pinned=False, shared=False):
        """pinned (on a GPU run): the int8 values and scales are written into page-locked arrays; shared (a reading
        process, section 17.82): into shared memory."""
        endpoint = np.asarray(self.source.archive.row_endpoint_index[np.asarray(rows, np.int64)])
        unique = np.unique(endpoint)
        lookback = getattr(self, 'lookback', 0)
        if lookback:
            # 17.166: also the pictures the rows' picture memories reach back to, inside each row's recording
            # (fcpg/v21_picture_memory.py: up to `lookback` endpoints before a row's first one)
            first = endpoint[:, 0]
            episode = np.asarray(self.source.archive.endpoint_episode_index)
            back = (first[:, None] - np.arange(1, lookback + 1)[None]).reshape(-1)
            keep = back >= 0
            keep[keep] = episode[back[keep]] == np.repeat(episode[first], lookback)[keep]
            unique = np.unique(np.concatenate((unique, back[keep])))
        parts, valid, done = [], np.zeros((len(unique), 2), bool), np.zeros(len(unique), bool)
        for cache in self.source.archive_dino:
            index = np.searchsorted(cache.endpoint_index, unique)
            match = (index < len(cache.endpoint_index)) & ~done
            match[match] = cache.endpoint_index[index[match]] == unique[match]
            if match.any():
                cache_rows = cache.rows_for(unique[match])
                if self.device.type == 'cuda' and getattr(cache.tokens, 'int8', False):
                    out = None
                    if pinned or shared:
                        n = len(cache_rows)
                        shapes = (((n, 2, PATCHES, DIM), torch.int8), ((n, 2, PATCHES), torch.float16))
                        out = tuple(torch.empty(shape, dtype=dtype, pin_memory=True) if pinned
                                    else torch.empty(shape, dtype=dtype).share_memory_() for shape, dtype in shapes)
                        cache.tokens.quantized(cache_rows, out=tuple(t.numpy() for t in out))
                    parts.append((np.flatnonzero(match), *(out or cache.tokens.quantized(cache_rows))))
                else:
                    parts.append((np.flatnonzero(match), np.ascontiguousarray(cache.tokens[cache_rows])))
                valid[match] = np.asarray(cache.view_valid[cache_rows], bool)
                done[match] = True
        if not done.all():
            raise ValueError('missing frozen DINO endpoints in the window')
        return unique, parts, valid

    def _place(self, unique, parts, valid, absent=None):
        """The read window as device tensors: int8 values times their scales, the integrity checks, the camera
        layout, 256 rows at a time. Returns unique, tokens, valid and a flag that a check failed."""
        tokens = torch.empty((len(unique), 2, PATCHES, DIM), dtype=torch.float16, device=self.device)
        for where, *value in parts:
            index = torch.as_tensor(where).to(self.device, non_blocking=True)
            value = [torch.as_tensor(v) for v in value]
            for begin in range(0, len(index), 256):
                chunk = slice(begin, begin + 256)
                if len(value) == 2:                          # int8 values and scales: multiplied here
                    values, scales = (v[chunk].to(self.device, non_blocking=True) for v in value)
                    tokens[index[chunk]] = dequantize(values, scales)
                else:
                    tokens[index[chunk]] = value[0][chunk].to(self.device, non_blocking=True)
        # the same integrity contract as the row cache: finite values, invalid views are zero
        absent = torch.as_tensor(~valid if absent is None else absent).to(self.device, non_blocking=True)
        broken = torch.zeros((), dtype=torch.bool, device=self.device)
        layout = getattr(self.source, 'archive_camera_roles', None)
        for begin in range(0, len(unique), 256):
            part = tokens[begin:begin + 256]
            broken |= ~torch.isfinite(part).all() | (part.ne(0) & absent[begin:begin + 256, :, None, None]).any()
            if layout is not None:
                part.copy_(layout.remap(part, unique[begin:begin + 256]))
        return unique, tokens, (valid if layout is None else layout.remap(valid, unique)), broken

    def _place_ahead(self, read):
        """The read window placed on the GPU on the side stream, with the event that marks it done. Section 17.82, as
        torch's DataLoader does with what its worker processes return: the window, in shared memory, is first copied
        into page-locked memory on this thread, so that its copy to the GPU is one fast transfer (from shared memory
        the driver staged and page-faulted it while the training thread waited: about 1-4 s at every window change)."""
        on_worker_cpus()
        unique, parts, valid = read.result()[:3]
        pin = lambda a: a.pin_memory() if isinstance(a, torch.Tensor) and not a.is_pinned() else a
        read = unique, [(where, *(pin(a) for a in rest)) for where, *rest in parts], valid, \
            torch.from_numpy(np.ascontiguousarray(~valid)).pin_memory()
        with torch.cuda.stream(self._stream):
            placed = self._place(*read)
            done = torch.cuda.Event()
            done.record(self._stream)
        return placed, done

    def prefetch(self, windows):
        """Start reading these windows [(key, rows)] (the ones that follow, in order); a window already being read,
        placed or in the pool is not read again. On a GPU, the first of them is placed there as soon as it is read."""
        self.start_readers()
        reading = ({key for key, _, _ in self._ahead} | set(self.resident)
                   | ({self._placed[0]} if self._placed is not None else set()))
        for key, rows in windows:
            if key not in reading:
                read = ((lambda r: self._reader.submit(_read_in_process, r)) if self.device.type == 'cuda'
                        else (lambda r: self._reader.submit(self._read, r)))
                self._ahead.append((key, rows, read(np.array(rows, np.int64))))
                reading.add(key)
        if self._placer is not None and self._placed is None and self._ahead:
            key, _, read = self._ahead.popleft()
            self._placed = (key, self._placer.submit(self._place_ahead, read))

    def admit(self, key, rows):
        """Put this window into the pool, waiting only for its placement when it was read ahead; it stays on the GPU
        until release(key). Returns its bytes there."""
        clock = time.monotonic()
        if self._placed is not None and self._placed[0] == key:
            (unique, tokens, valid, broken), done = self._placed[1].result()
            self._placed, self.prefetched = None, True
            self.waited += time.monotonic() - clock
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(done)
            tokens.record_stream(stream)                     # its memory is not reused while this stream reads it
        else:
            while self._ahead and self._ahead[0][0] != key:
                self._ahead.popleft()                         # read for nothing (cannot happen in the Archive loop)
            ahead = self._ahead.popleft() if self._ahead else None
            self.prefetched = ahead is not None
            read = ahead[2].result() if self.prefetched else self._read(np.asarray(rows, np.int64))
            self.waited += time.monotonic() - clock
            unique, tokens, valid, broken = self._place(*read)
        fail_if(broken, 'frozen DINO data are nonfinite or invalid views are nonzero')     # read later if deferred
        self.resident[key] = (unique, tokens, valid)
        return int(tokens.numel()) * 2

    def release(self, key):
        """Take a window out of the pool once its last row has been trained (section 17.82: used once, then gone)."""
        del self.resident[key]

    def load(self, rows):
        """Read and place one window now (the development rows, section 17.65)."""
        self.unique = self.tokens = self.valid = None
        unique, tokens, valid, broken = self._place(*self._read(np.asarray(rows, np.int64)))
        fail_if(broken, 'frozen DINO data are nonfinite or invalid views are nonzero')
        self.unique, self.tokens, self.valid = unique, tokens, valid
        return int(tokens.numel()) * 2

    def close(self):
        self._ahead.clear()
        self._placed = None
        self.resident.clear()
        if self._reader is not None:
            self._reader.shutdown(wait=True, cancel_futures=True)
        if self._placer is not None:
            self._placer.shutdown(wait=True, cancel_futures=True)

    def copy_to(self, device):
        """The loaded rows with their pictures copied to `device`, for reading only (section 17.65: the development
        rows are held in host memory and copied to the GPU for each evaluation)."""
        held = WindowFeatures.__new__(WindowFeatures)
        held.source, held.device, held.prefetched, held._reader, held._ahead = self.source, torch.device(device), False, None, deque()
        held._placer, held._stream, held._placed, held.resident, held.readers = None, None, None, {}, 1
        held.waited, held.lookback = 0., getattr(self, 'lookback', 0)
        held.unique, held.valid = self.unique, self.valid
        held.tokens = self.tokens.to(held.device, non_blocking=True)
        return held

    def _index(self, index):
        at = torch.from_numpy(np.ascontiguousarray(index, np.int64))
        if self.device.type == 'cuda':                       # section 17.80: from page-locked memory, without waiting
            at = at.pin_memory().to(self.device, non_blocking=True)
        return at

    def features(self, rows, context_frames, *, as_stored=False):
        """as_stored: the float16 values as held (section 17.76: prepare() converts them after the flips and masks).
        Section 17.82: with several windows in the pool, every row from the window that holds all its exposures."""
        endpoint = np.asarray(self.source.archive.row_endpoint_index[np.asarray(rows, np.int64)])[:, :context_frames]
        windows = list(self.resident.values()) or [(self.unique, self.tokens, self.valid)]
        if len(windows) == 1:
            unique, tokens, valid = windows[0]
            index = np.searchsorted(unique, endpoint)
            if index.max(initial=0) >= len(unique) or not np.array_equal(unique[index], endpoint):
                raise ValueError('batch rows are outside the loaded window')
            features = tokens[self._index(index)]
            return (features if as_stored else features.float()), valid[index]
        features, held, todo = None, np.zeros((*endpoint.shape, 2), bool), np.ones(len(endpoint), bool)
        for unique, tokens, valid in windows:
            index = np.searchsorted(unique, endpoint).clip(max=len(unique) - 1)
            here = todo & (unique[index] == endpoint).all(1)
            if not here.any():
                continue
            picked = tokens[self._index(index[here])]
            if features is None:
                features = picked.new_empty((len(endpoint), *picked.shape[1:]))
            features.index_copy_(0, self._index(np.flatnonzero(here)), picked)
            held[here] = valid[index[here]]
            todo &= ~here
        if todo.any():
            raise ValueError('batch rows are outside the windows of the pool')
        return (features if as_stored else features.float()), held

    def pictures(self, ids):
        """17.166: the pictures of these Archive endpoint ids (a batch's picture memories) from the window pool, as
        stored (float16) with their views' validity; every id must be in a resident window (lookback)."""
        ids = np.asarray(ids, np.int64)
        windows = list(self.resident.values()) or [(self.unique, self.tokens, self.valid)]
        out, held, todo = None, np.zeros((len(ids), 2), bool), np.ones(len(ids), bool)
        for unique, tokens, valid in windows:
            index = np.searchsorted(unique, ids).clip(max=len(unique) - 1)
            here = todo & (unique[index] == ids)
            if not here.any():
                continue
            picked = tokens[self._index(index[here])]
            if out is None:
                out = picked.new_empty((len(ids), *picked.shape[1:]))
            out.index_copy_(0, self._index(np.flatnonzero(here)), picked)
            held[here] = valid[index[here]]
            todo &= ~here
        if todo.any():
            raise ValueError('picture memory endpoints are outside the windows of the pool (lookback too short?)')
        return out, held


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-config', type=Path, required=True)
    p.add_argument('--camera-layout', type=Path, required=True, help='layout.npz from the Archive camera audit')
    p.add_argument('--lewm', type=Path, required=True, help='deployed full-patch LeWM checkpoint (frozen)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--gpu-memory-limit-gib', type=float, default=24.)
    p.add_argument('--feature-bank-gib', type=float, default=8.)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--window-rows', type=int, default=2048,
                   help='Archive rows per locality window (one ascending read each; batches shuffled inside)')
    p.add_argument('--window-readers', type=int, default=1,
                   help='section 17.79: threads reading the following Archive windows at the same time')
    p.add_argument('--pool-windows', type=int, default=1,
                   help='section 17.82: Archive windows held on the GPU at once; every batch takes rows from all of them, '
                        'every row once per pass, a window leaves when its last row is used (1: one window at a time, '
                        'the earlier schedule exactly)')
    p.add_argument('--checkpoint-reader', action='store_true',
                   help='section 17.80: the FCPG reader\'s picture projection is recomputed in the backward pass instead '
                        'of kept (the same values and gradients; less GPU memory per row, for larger batches)')
    p.add_argument('--compile', action='store_true',
                   help='section 17.84: a training update runs the FCPG read, the Actor\'s layers, the memory shift and '
                        'the picture checks as torch.compile programs (fcpg/v21_compiled.py), and a wrist-only reader '
                        'skips the other view; evaluations run exactly as before')
    p.add_argument('--fish-batch-size', type=int, default=64, help='half fish, half Archive')
    p.add_argument('--fish-archive-pool', type=int, default=0,
                   help='fish stage: draw the Archive half from a rolling pool of this many feature-store blocks instead '
                        'of the whole Archive (0, the default: independent draws as before); section 17.59')
    p.add_argument('--fish-archive-pool-refresh', type=int, default=2,
                   help='--fish-archive-pool: blocks replaced by new ones after every fish update')
    p.add_argument('--archive-passes', type=int, default=4)
    p.add_argument('--archive-updates', type=int, default=None, help='cap on Archive updates (smoke runs)')
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--archive-schedule', choices=('constant', 'warmup-linear'), default='constant',
                   help='Archive stage learning rate: constant, or (section 17.74, as BERT) a linear rise over '
                        '--archive-warmup updates and a linear fall to zero at the last Archive update')
    p.add_argument('--archive-warmup', type=int, default=0, help='--archive-schedule warmup-linear: updates of the rise')
    p.add_argument('--fish-learning-rate', type=float, default=3e-5)
    p.add_argument('--fish-updates', type=int, default=1000)
    p.add_argument('--fish-schedule', choices=('constant', 'warmup-linear'), default='constant',
                   help='fish stage learning rates: constant, or (section 17.120, the NERO simulation\'s Actor recipe) a '
                        'linear rise over --fish-warmup updates and a linear fall to zero at the last fish update')
    p.add_argument('--fish-warmup', type=int, default=0, help='--fish-schedule warmup-linear: updates of the rise')
    p.add_argument('--fish-ema', type=float, default=0.,
                   help='section 17.120: fish stage only, evaluate and save (the kept steps and the final weights) an '
                        'exponential moving average of the trained weights with this decay, warmed up as --ema (0 = off)')
    p.add_argument('--fish-seed', type=int, default=None,
                   help='section 17.120: the fish stage\'s own draws (rows, mirror, cue dropout, patch mask, box paste) '
                        'from this seed, as --archive-checkpoint draws them from --seed; the model, the splits and the '
                        'Archive stage keep --seed (default: as before)')
    p.add_argument('--fish-train-reader', action='store_true',
                   help='fish stage: the FCPG reader learns too (default: frozen), on both halves of every batch')
    p.add_argument('--keep-steps', type=int, nargs='*', default=[50, 100, 125, 200, 250, 375, 500, 750])
    p.add_argument('--evaluate-every', type=int, default=1000, help='Archive-stage evaluation period (updates)')
    p.add_argument('--development-rows', type=int, default=512)
    p.add_argument('--seed', type=int, default=21004)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16')
    p.add_argument('--smoke', action='store_true', help='tiny run; checkpoints are marked smoke and never load')
    p.add_argument('--timing', action='store_true', help='synchronise the GPU at every timer (exact shares, slower)')
    p.add_argument('--evaluate-checkpoint', type=Path, default=None,
                   help='no training: evaluate these saved weights on Archive development rows, a sample of Archive '
                        'TRAIN rows, fish demonstrations 0/1 and the held-out demonstration 3 (all of it, and from its '
                        'first commanded motion on as the kept fish steps report it: fish_moving)')
    p.add_argument('--exam-hold-back', action='store_true',
                   help='section 17.63, with --evaluate-checkpoint: instead, demonstration 3 with its newest 1..3 wrist '
                        'pictures withheld (hold-back-exam.json)')
    p.add_argument('--gripper-separate', action='store_true',
                   help='V21Config.gripper_separate: the model does not read the jaw opening and outputs the opening itself')
    p.add_argument('--gripper-relative', action='store_true',
                   help='V21Config.gripper_relative (section 17.39/17.43, needs --gripper-separate): the separate jaw output '
                        'is a change from the current reading; the reading is still no input')
    p.add_argument('--gripper-absolute-aux', action='store_true',
                   help='V21Config.gripper_absolute_aux (section 17.46, needs --gripper-relative): an extra head predicts the '
                        'absolute jaw command beside the relative one (loss weight of one joint); never a command')
    p.add_argument('--fish-jaw-teacher', action='store_true',
                   help='section 17.46: in the fish stage the demonstration rows\' gripper targets come from the model as the '
                        'fish stage starts (a frozen copy), except where the demonstrator opens the jaws')
    p.add_argument('--wrist-only', action='store_true',
                   help='V21Config.wrist_only: no external camera; Archive rows without a wrist camera are left out')
    p.add_argument('--balanced-collections', action='store_true',
                   help='Archive draws: collections with at least --large-collection-rows admitted TRAIN rows share '
                        'their draws equally (sampled with replacement, read in ascending order per window)')
    p.add_argument('--large-collection-rows', type=int, default=10000)
    p.add_argument('--mirror', action='store_true',
                   help='training rows (Archive and fish) are mirrored left-right with probability one half; needs '
                        '--wrist-only or --mirror-external')
    p.add_argument('--mirror-external', action='store_true',
                   help='section 17.52, with --mirror and without --wrist-only: the external picture is flipped '
                        'with the wrist picture (the mirrored scene, the robot on the other side)')
    p.add_argument('--view-dropout', type=float, default=0.,
                   help='section 17.52, without --wrist-only: per training row, hide the external camera with this '
                        'probability and the wrist camera with this probability (only in rows where both cameras '
                        'have a picture), for the whole chain')
    p.add_argument('--cue-only-share', type=float, default=0.,
                   help='11/03: per training fish row whose current wrist fish cue is seen, blank both cameras\' pictures '
                        'for the whole chain with this probability (own generator, torch seed + 26; the exposures stay '
                        'delivered, so the cue still reaches the Actor): the Actor has the cue, the joints and the task '
                        'left and must steer by the cue (the release Actor, which sees where the fish is in its pictures, '
                        'never learned to: 0-3 of 267 robot states turned the pan for a moved cue whatever its fish '
                        'stage); evaluations keep the pictures')
    p.add_argument('--archive-cue-only-share', type=float, default=0.,
                   help='11/03: the same for training Archive rows (own generator, torch seed + 27): per Archive row '
                        'whose current wrist grasp cue is seen, both cameras\' pictures blank for the whole chain with '
                        'this probability. The release base turned the pan for a moved cue in 15 of 117 Archive '
                        'development rows (HLcs\'s base: 76): its cue rows are 1.4%% of the Archive (21,410 of '
                        '1,481,603; HLtc\'s 3.9%%) and the pictures show the object as well')
    p.add_argument('--pose-position', action='store_true',
                   help='11/03: a new concatenating model (--concat-inputs) whose FCPG keys also join each patch\'s '
                        'position x its picture\'s joints at capture and their change since (V21Config.pose_position), '
                        'so the joints at capture can move where in a picture it reads, not only weigh whole pictures')
    p.add_argument('--object-cue-on-pose', action='store_true',
                   help='11/03: a new model adds the wrist grasp cue to the pose token while it is seen, as the place '
                        'cue, instead of reading it as a token of its own (V21Config.object_on_pose): in the release '
                        'Actor the place cue steers the pan, the grasp cue does not')
    p.add_argument('--no-task-text', action='store_true',
                   help='James 11/02: neither the FCPG reader nor the Actor reads the task words (V21Config.task_text '
                        '=False); the read queries follow the current joints, and the object cues stay')
    p.add_argument('--concat-inputs', action='store_true',
                   help='James 11/02: the FCPG reader concatenates its inputs instead of adding them, and its '
                        'attention reads them as they are, with no network before it (V21Config.concat_inputs, '
                        'Fig. 1 b); needs --no-task-text')
    p.add_argument('--wrist-hold-back', type=float, default=0.,
                   help='section 17.63: in this share of the training rows (both stages) with at least one completed '
                        'block, the newest 1..3 wrist pictures do not arrive, so the memory holds an older one; 0 changes '
                        'nothing')
    p.add_argument('--reread-weight', type=float, default=0.,
                   help='section 17.95, with --wrist-hold-back: in the Archive stage the wrist read of a held-back row is '
                        'also trained towards the read of the same row with its newest pictures (no gradient through '
                        'that read), with this weight beside the action loss; 0 changes nothing')
    p.add_argument('--memory-address', choices=('readback', 'temporal'), default='readback',
                   help="section 17.95: V21Config.memory_mode; temporal = the read addresses keep age, view and patch "
                        "position, not the joints at each exposure or their displacement to now. With --continue-archive "
                        "or --archive-checkpoint it is the loaded Archive stage's own (the model-switch check refuses "
                        "another)")
    p.add_argument('--mirror-skip-sided-tasks', action='store_true',
                   help='--mirror: Archive episodes whose task text names a side (left, right, clockwise, ...) are '
                        'never mirrored; the mirrored scene would contradict the words (found 10/13 00:20, docs section 12)')
    p.add_argument('--fish-roll-centre', type=float, default=-103.,
                   help='fish robot: twice the wrist roll at which its wrist camera is level (HOME roll about -51.5, '
                        'analysis/wrist_mount_from_demos.py); Archive episodes use twice their median roll')
    p.add_argument('--patch-mask', type=float, default=0.,
                   help='training rows: this share of each picture\'s patches is set to zero, drawn anew every time')
    p.add_argument('--collection-rows', type=int, default=2000,
                   help='--evaluate-checkpoint: collections with at least this many admitted rows are also evaluated alone')
    p.add_argument('--by-camera-order', action='store_true',
                   help='--evaluate-checkpoint: each collection set is also split into episodes whose two cameras were '
                        'recorded in the model\'s order and episodes whose cameras the layout table swapped')
    p.add_argument('--queries-per-view', type=int, choices=(8, 32), default=8,
                   help='V21Config.queries_per_view: FCPG summary tokens per view')
    p.add_argument('--actor-width', type=int, choices=(192, 384), default=192, help='V21Config.actor_width')
    p.add_argument('--actor-layers', choices=('v1', 'v2'), default='v1',
                   help='V21Config.actor_layers (section 17.75): v2 = 4 context layers, command queries attending both '
                        'ways, dropout 0.1')
    p.add_argument('--ema', type=float, default=0.,
                   help='section 17.75: Archive stage only, evaluate, save and hand to the fish stage an exponential '
                        'moving average of the trained weights with this decay (0 = off)')
    p.add_argument('--archive-select', choices=('last', 'development'), default='last',
                   help='section 17.89: the Archive stage hands on its last weights (the default, as before) or, as '
                        'transformers\' load_best_model_at_end, the evaluated weights (the average with --ema) with the '
                        'lowest 40-step error on the development rows')
    p.add_argument('--archive-patience', type=int, default=0,
                   help='section 17.89, with --archive-select development: end the Archive stage once this many '
                        'evaluations in a row found no lower development error (transformers\' EarlyStoppingCallback; '
                        '0 = every planned update)')
    p.add_argument('--random-reads', action='store_true',
                   help='Archive stage: per-batch reads through the row cache instead of one sequential read per window')
    p.add_argument('--trim-idle-start', action='store_true',
                   help='fish stage: never draw a chain of demonstrations 0/1 whose current frame is before the '
                        'operator\'s first commanded motion (section 15)')
    p.add_argument('--fish-horizon', type=int, choices=(5, 40), default=40,
                   help='fish stage, both halves: supervise only the first N commands (5 = the 0.5 s a block executes); '
                        'the model, its Archive stage and the exams keep 40 (docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md '
                        'section 16)')
    p.add_argument('--archive-horizon', type=int, choices=(5, 40), default=40,
                   help='Archive stage: supervise only the first N commands (5 = 0.5 s), as --fish-horizon does for the '
                        'fish stage; the model and the exams keep 40 (section 17.50: a 4 s future invites recognising the '
                        'recording instead of reacting to what is seen)')
    p.add_argument('--archive-checkpoint', type=Path, default=None,
                   help='do not run the Archive stage: load actor_archive/final.pt of an earlier formal run with the '
                        'same model switches, seed and training options, then train the fish stage with new generators')
    p.add_argument('--continue-archive', type=Path, default=None,
                   help='section 17.72: start the Archive stage from actor_archive/final.pt of an earlier formal run with '
                        'the same model switches, seed, data and training options (a new optimizer and new generators), '
                        'train --archive-updates more updates, then the fish stage as usual')
    p.add_argument('--place-table', type=Path, default=None,
                   help='per fish frame where the object goes (the drop zone seen in the wrist) against the jaws '
                        '(scripts/build_v21_place_table.py): the '
                        'Actor reads it as a second input added to the pose token (V21Config.place_relative; docs section 17)')
    p.add_argument('--place-learning-rate-scale', type=float, default=1.,
                   help='fish stage only: the place projection (actor.place_relative.*) learns at this multiple of '
                        '--fish-learning-rate, in its own optimizer group (docs section 17.2); 1 keeps one group as before')
    p.add_argument('--place-goal-tables', type=Path, default=None,
                   help='directory from scripts/build_v21_place_goal_tables.py: the Actor reads the hindsight place goal '
                        'against its jaws (V21Config.place_goal; docs section 17.7)')
    p.add_argument('--place-goal-learning-rate-scale', type=float, default=1.,
                   help='both stages: the goal projection (actor.place_goal.*) learns at this multiple of the stage\'s '
                        'learning rate, in its own optimizer group; 1 keeps one group')
    p.add_argument('--object-cues', type=Path, default=None,
                   help='directory from scripts/build_v21_outline_cue_tables.py: the wrist object token from outlines on '
                        'Archive and fish rows, instead of the colour body table on fish rows only (docs section 17.30)')
    p.add_argument('--fish-token-table', type=Path, default=None,
                   help='directory from scripts/build_v21_fish_token_table.py: the fish rows\' wrist object token from '
                        'SAM 3 outlines with the score field 1, instead of the colour body table (docs section 17.57); '
                        'Archive rows keep no token, as with the body table')
    p.add_argument('--fish-goal-dropout', type=float, default=0.,
                   help='fish stage, fish rows only: remove the place goal from this share of the rows of every update '
                        '(a fresh draw per row from its own generator, seed + 7); the Archive half and every evaluation '
                        'keep their goals; 0 changes nothing (docs section 17.10)')
    p.add_argument('--cue-dropout', type=float, default=0.,
                   help='both stages, every row: remove the grasp cue (object token) and, independently, the place cue '
                        'from this share of the rows of every update (one draw per row and cue from its own generator, '
                        'seed + 11); evaluations keep them; 0 changes nothing (docs section 17.117)')
    p.add_argument('--cem-labels', type=Path, default=None,
                   help='section 17.110: a table of scripts/build_v21_cem_teacher_labels.py --split train. In the fish '
                        'stage the fish rows it labels learn the alignment CEM\'s selection as their first five arm '
                        'commands, with the wrist token moved by the shift the selection was made for '
                        '(fcpg/v21_cem_teacher.py; a draw per row from its own generator, seed + 10); the gripper, the '
                        'Archive half and every evaluation keep the recorded commands')
    p.add_argument('--cem-unshifted-share', type=float, default=.5,
                   help='with --cem-labels: the share of the labelled fish rows that keep the fish where it is')
    p.add_argument('--world-head', action='store_true',
                   help='section 17.113: Actor and world model as one model. A world head on the FCPG reader (shared '
                        'with the Actor) learns, in both stages, what one block of five commands does: the change of '
                        'the wrist picture (its frozen DINOv3-L patches in the top --world-components principal '
                        'components) and the five joint read-backs; its loss reaches the reader. Fresh models only')
    p.add_argument('--world-weight', type=float, default=1.,
                   help='with --world-head: the world loss (picture change over the change itself, plus the read-back '
                        'error, plus SIGReg) is added at this weight to the Actor loss')
    p.add_argument('--world-target', choices=('tokens', 'patches'), default='tokens',
                   help='with --world-head: tokens (the second version): the reader\'s own wrist tokens of the next '
                        'state, read without gradient, SIGReg on the reader\'s wrist tokens; patches (the first '
                        'version): the wrist picture\'s change in fixed principal components')
    p.add_argument('--world-components', type=int, choices=(16, 32, 64, 128), default=64)
    p.add_argument('--world-learning-rate-scale', type=float, default=3.,
                   help='with --world-head: the world head learns at this multiple of each stage\'s rate (3: the '
                        'Archive stage\'s 1e-4 becomes the 3e-4 of the 17.112 step-1 probe)')
    p.add_argument('--world-basis-chains', type=int, default=800,
                   help='with --world-head: training chains whose wrist frames fit the principal components')
    p.add_argument('--world-sigreg-weight', type=float, default=.02)
    p.add_argument('--world-sigreg-projections', type=int, default=8192)
    p.add_argument('--object-token-learning-rate-scale', type=float, default=1.,
                   help='fish stage only: the wrist object token\'s projection (actor.object_relative.*, 8 -> width -> '
                        'width, its last layer zero at the start and never trained by an Archive stage without cues) '
                        'learns at this multiple of --fish-learning-rate, in its own group (section 17.111); 1 keeps it '
                        'with the rest')
    p.add_argument('--carry-lift-weight', type=float, default=0.,
                   help='section 17.120, both stages, every training row that holds something (the grasp cue on; fish '
                        'rows also with the jaws shut): this weight times the mean squared shortfall of the jaws\' '
                        'height at the Actor\'s supervised commands below the demonstrator\'s (per CARRY_SCALE_MM beyond '
                        'CARRY_MARGIN_MM); 0 changes nothing')
    p.add_argument('--carry-over-margin-mm', type=float, default=None,
                   help='section 17.120: with --carry-lift-weight, also the excess of that height above the '
                        'demonstrator\'s beyond this many mm, squared per CARRY_SCALE_MM (robot trial 27: HLtp, punished '
                        'only below, climbed to 47 cm holding the fish); none: below only')
    p.add_argument('--carry-fish-only', action='store_true',
                   help='section 17.120: --carry-lift-weight on the fish rows only: our arm\'s clearance at the Archive '
                        'rows\' joints is no height of theirs (HLtr\'s first Archive updates with --carry-over-margin-mm: '
                        'loss 150-1,000 against 1.5)')
    p.add_argument('--carry-place-factor', type=float, default=1.,
                   help='section 17.120: with --carry-lift-weight, held rows that also see where the thing goes (the '
                        'place cue on, as the box in trial 21) count this many times')
    p.add_argument('--box-paste-bank', type=Path, default=None,
                   help='section 17.120: a box-background bank (lift-diagnosis-20261027/build_box_bank_20261027.py) for '
                        '--box-paste-share')
    p.add_argument('--box-paste-share', type=float, default=0.,
                   help='section 17.120, fish stage: this share of the held fish rows whose demonstrator lifts the jaws '
                        'more than PASTE_LIFT_MM see a box background in every wrist exposure (labels unchanged)')
    p.add_argument('--scene-paste-bank', type=Path, default=None,
                   help='section 17.120: a scene-background bank (lift-diagnosis-20261027/build_scene_bank_20261028.py: '
                        'the Archive wrist backgrounds of half of its collections) for --scene-paste-share')
    p.add_argument('--scene-paste-share', type=float, default=0.,
                   help='section 17.120, fish stage: of the lifting held fish rows the box paste left alone, this share see '
                        'another room\'s background in every wrist exposure (labels unchanged; own generator, seed + 14)')
    p.add_argument('--archive-fish-share', type=float, default=0.,
                   help='section 17.121, Archive stage: every update also trains fish rows of demonstrations 0 and 1 '
                        '(the fish stage\'s rows, horizon and augmentations, the box paste included), as many as make '
                        'them this share of the rows, every row weighing the same (own generators, seed + 21 to + 24)')
    return p


def drop_place_goals(chain, share, generator):
    """--fish-goal-dropout (section 17.10): the chain with the place goal of a random share of its rows removed (all eight
    numbers of every context frame, as in a row that has no goal), one draw per row from `generator` whether or not the
    row has a goal; every other field is the same object. Returns the chain and the number of rows that lost a goal."""
    goal = getattr(chain, 'place_goal', None)
    if share == 0 or goal is None:
        return chain, 0
    drop = torch.rand(len(goal), generator=generator, device=goal.device) < share
    had = (goal[..., 0] > .5).any(-1)
    return replace(chain, place_goal=torch.where(drop[:, None, None], torch.zeros_like(goal), goal)), int((drop & had).sum())


def drop_cues(chain, share, generator):
    """--cue-dropout (section 17.117): the chain with the grasp cue (object_relative) and, by an independent draw, the
    place cue (place_relative) of a random share of its rows removed (all eight numbers of every context frame, as in a
    row that has none); two draws per row from `generator` whether or not the row has the cue. Returns the chain and the
    numbers of rows that lost a grasp and a place cue."""
    if share == 0:
        return chain, 0, 0
    out, lost = {}, []
    for name in ('object_relative', 'place_relative'):
        cue = getattr(chain, name, None)
        if cue is None:
            lost.append(0)
            continue
        drop = (torch.rand(len(cue), generator=generator, device=generator.device) < share).to(cue.device)
        out[name] = torch.where(drop[:, None, None], torch.zeros_like(cue), cue)
        lost.append(int((drop & (cue[..., 0] > .5).any(-1)).sum()))
    return replace(chain, **out), lost[0], lost[1]


def drop_views(chain, share, generator):
    """--view-dropout (section 17.52): one draw per row from `generator`; below `share` the external camera (slot 1) is
    hidden for the whole chain, from `share` to twice it the wrist camera (slot 0), only in rows where both cameras have
    at least one valid exposure in the chain; a hidden camera's exposures become invalid and its features zero.
    Returns the chain and the numbers of rows that lost the external and the wrist camera."""
    if share == 0:
        return chain, 0, 0
    valid = chain.view_valid
    draw = torch.rand(len(valid), generator=generator, device=generator.device).to(valid.device)
    both = valid.any(1).all(1)                                           # [B]: both cameras have a picture
    hide_external = (draw < share) & both
    hide_wrist = (draw >= share) & (draw < 2 * share) & both
    hidden = torch.stack((hide_wrist, hide_external), 1)                 # [B, 2] in slot order
    keep = ~hidden[:, None, :]
    features = None if chain.features is None else chain.features * keep[..., None, None].to(chain.features.dtype)
    return (replace(chain, view_valid=valid & keep, features=features), int(hide_external.sum()),
            int(hide_wrist.sum()))


def cue_only_views(chain, h, share, generator):
    """--cue-only-share (11/03): one draw per row from `generator`; below `share`, in a row whose wrist fish cue at
    context h is seen, both cameras' pictures are blank for the whole chain (features zero, the exposures still
    delivered). The exposures stay valid because the Actor gets the wrist cue only with a current wrist exposure
    (temporal_actor_state): 9afe618 made them invalid, so those rows had neither the pictures nor the cue.
    Returns the chain and the number of rows."""
    if share == 0 or chain.object_relative is None:
        return chain, 0
    if chain.features is None:
        raise ValueError('--cue-only-share blanks the pictures: the chain carries none')
    valid = chain.view_valid
    draw = torch.rand(len(valid), generator=generator, device=generator.device).to(valid.device)
    # only where the current wrist picture arrived (a row --view-dropout left without it has no cue for the Actor)
    hide = (draw < share) & (chain.object_relative[:, h, 0].to(valid.device) > .5) & valid[:, h, 0]
    keep = (~hide[:, None, None]).to(chain.features.device)
    return replace(chain, features=chain.features * keep[..., None, None].to(chain.features.dtype)), int(hide.sum())


def hold_back_wrist(chain, h, share, generator):
    """--wrist-hold-back (section 17.63): two draws per row from `generator` (whether, how far); below `share`, in a chain
    with h >= 1 completed blocks, the newest k wrist pictures (k uniform in 1..min(HOLD_BACK_BLOCKS, h)) do not arrive:
    slot 0 of frames h-k+1..h becomes invalid with zero features, as a camera that delivered nothing. The wrist memory
    then still holds the picture of k blocks earlier with the joints at its exposure, and the wrist object cue is absent
    (it belongs to a current wrist picture). Only rows whose wrist picture k blocks earlier is valid; the external camera
    is unchanged. Returns the chain and the numbers of rows held back by 1, 2 and 3 blocks."""
    counts = [0] * HOLD_BACK_BLOCKS
    if share == 0 or h == 0:
        return chain, counts
    valid = chain.view_valid
    rows = torch.arange(len(valid), device=valid.device)
    draw = torch.rand(len(valid), 2, generator=generator, device=generator.device).to(valid.device)
    most = min(HOLD_BACK_BLOCKS, h)
    k = 1 + (draw[:, 1] * most).long().clamp_max(most - 1)                    # [B] in 1..most
    held = (draw[:, 0] < share) & valid[rows, h - k, 0]
    frame = torch.arange(valid.shape[1], device=valid.device)
    hidden = held[:, None] & (frame[None] > (h - k)[:, None]) & (frame[None] <= h)       # [B, frames]
    view_valid = valid.clone()
    view_valid[..., 0] &= ~hidden
    features = chain.features
    if features is not None:
        keep = torch.ones(features.shape[:3], dtype=torch.bool, device=features.device)
        keep[..., 0] = ~hidden[:, :features.shape[1]].to(features.device)
        features = features * keep[..., None, None].to(features.dtype)
    counts = [int((held & (k == n)).sum()) for n in range(1, HOLD_BACK_BLOCKS + 1)]
    return replace(chain, view_valid=view_valid, features=features), counts


def withhold_wrist(chain, h, k):
    """--exam-hold-back (section 17.63): every row as a row that hold_back_wrist holds back by k blocks (slot 0 of frames
    h-k+1..h invalid, zero features); k = 'all' hides the wrist camera in frames 0..h. The caller picks rows whose wrist
    pictures exist."""
    valid = chain.view_valid
    frame = torch.arange(valid.shape[1], device=valid.device)
    hidden = (frame <= h) & ((frame > h - k) if k != 'all' else True)
    hidden = hidden[None].expand(len(valid), -1)
    view_valid = valid.clone()
    view_valid[..., 0] &= ~hidden
    features = chain.features
    if features is not None:
        keep = torch.ones(features.shape[:3], dtype=torch.bool, device=features.device)
        keep[..., 0] = ~hidden[:, :features.shape[1]].to(features.device)
        features = features * keep[..., None, None].to(features.dtype)
    return replace(chain, view_valid=view_valid, features=features)


def reread_loss(stale, fresh, held):
    """--reread-weight (section 17.95): stale [B,Q,192] is the wrist read the Actor gets when the newest wrist pictures
    were held back (older pictures and the joints at their exposure), fresh [B,Q,192] the read of the same row with
    them, held [B] the rows held back. Mean over the held rows of the per-token mean squared difference after layer
    normalisation (scale-free: shrinking the read cannot lower it); no gradient reaches the fresh read; 0 when no row
    is held. Written without selecting rows (a selection makes the CPU wait for the GPU)."""
    if stale.shape != fresh.shape or stale.ndim != 3 or held.shape != stale.shape[:1] or held.dtype != torch.bool:
        raise ValueError('the re-read target needs [B,Q,D] reads of the same shape and a bool [B] held mask')
    norm = lambda x: torch.nn.functional.layer_norm(x.float(), x.shape[-1:])
    per_row = (norm(stale) - norm(fresh).detach()).square().mean((1, 2))
    weight = held.to(per_row.dtype)
    return (per_row * weight).sum() / weight.sum().clamp_min(1.)


def fish_optimizer(core, parameters, learning_rate, place_scale=1., *, fused=False, goal_scale=1., token_scale=1.,
                   world_scale=1.):
    """The fish-stage AdamW (and, with place_scale 1, the Archive stage's). All scales 1: one group, exactly as before.
    Otherwise the new place projection (actor.place_relative.*, section 17.2), the goal projection
    (actor.place_goal.*, section 17.7), the wrist object token's projection (actor.object_relative.*, section
    17.111) and the world head (world.*, section 17.113) get their own groups at scale x learning_rate; everything
    else unchanged."""
    if place_scale == 1 and goal_scale == 1 and token_scale == 1 and world_scale == 1:
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=.01, fused=fused)
    groups, chosen = [], set()
    for prefix, scale, flag in (('actor.place_relative.', place_scale, '--place-learning-rate-scale'),
                                ('actor.place_goal.', goal_scale, '--place-goal-learning-rate-scale'),
                                ('actor.object_relative.', token_scale, '--object-token-learning-rate-scale'),
                                ('world.', world_scale, '--world-learning-rate-scale')):
        if scale == 1:
            continue
        own = [p for n, p in core.named_parameters() if p.requires_grad and n.startswith(prefix)]
        if not own:
            raise ValueError(f'{flag}: this model has no trainable {prefix[6:-1]} projection')
        chosen |= {id(p) for p in own}
        groups.append(dict(params=own, lr=learning_rate * scale))
    return torch.optim.AdamW([dict(params=[p for p in parameters if id(p) not in chosen]), *groups],
                             lr=learning_rate, weight_decay=.01, fused=fused)


@contextmanager
def evaluating(core):
    modes = {name: module.training for name, module in core.named_modules()}
    core.eval()
    try:
        yield
    finally:
        for name, module in core.named_modules():
            module.training = modes[name]


class WeightAverage:
    """Section 17.75: an exponential moving average of trainable parameters (warmed up as min(decay, (1+n)/(10+n)) after
    n updates). swapped() puts the average into the model for evaluation or saving and restores the trained values;
    keep() leaves the average in the model."""

    def __init__(self, parameters, decay):
        self.parameters, self.decay, self.updates = list(parameters), float(decay), 0
        self.average = [p.detach().clone() for p in self.parameters]

    @torch.no_grad()
    def update(self):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        torch._foreach_lerp_(self.average, [p.detach() for p in self.parameters], 1 - decay)

    @contextmanager
    def swapped(self):
        trained = [p.detach().clone() for p in self.parameters]
        with torch.no_grad():       # in place on the parameters themselves: their versions change, so no cache is stale
            torch._foreach_copy_(self.parameters, self.average)
        try:
            yield
        finally:
            with torch.no_grad():
                torch._foreach_copy_(self.parameters, trained)

    @torch.no_grad()
    def keep(self):
        torch._foreach_copy_(self.parameters, self.average)


class BestWeights:
    """Section 17.89, as transformers (load_best_model_at_end with metric_for_best_model, EarlyStoppingCallback) and
    Keras (EarlyStopping with restore_best_weights) do it: a copy of the evaluated weights with the lowest error so far,
    the update they were taken at, and how many evaluations have passed since without a lower error. offer() takes the
    parameters as they are at an evaluation (the average with --ema); restore() puts the copy back."""

    def __init__(self, parameters):
        self.parameters, self.weights = list(parameters), None
        self.step, self.error, self.since = None, float('inf'), 0

    @torch.no_grad()
    def offer(self, step, error):
        error = float(error)
        if error < self.error:              # not a NaN, and strictly lower: an equal error keeps the earlier weights
            self.weights = [p.detach().clone() for p in self.parameters]
            self.step, self.error, self.since = step, error, 0
        else:
            self.since += 1
        return self.since

    @torch.no_grad()
    def restore(self):
        torch._foreach_copy_(self.parameters, self.weights)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; every run gets a new directory')
    if args.batch_size < 2 or args.fish_batch_size < 4 or args.fish_batch_size % 2 or args.fish_updates < 1:
        raise ValueError('invalid batch or update counts')
    if args.mirror_external and (args.wrist_only or not args.mirror):
        raise ValueError('--mirror-external is for --mirror without --wrist-only (section 17.52)')
    if args.window_rows < args.batch_size:
        raise ValueError('--window-rows must hold at least one batch')
    if args.window_readers < 1:
        raise ValueError('--window-readers must be at least 1')
    if args.pool_windows < 1:
        raise ValueError('--pool-windows must be at least 1')
    if args.fish_archive_pool < 0:
        raise ValueError('--fish-archive-pool is a number of store blocks (0 = off)')
    if args.fish_archive_pool and args.balanced_collections:
        raise ValueError('--fish-archive-pool draws rows uniformly and cannot be combined with --balanced-collections')
    if not 0. <= args.cue_only_share < 1.:
        raise ValueError('--cue-only-share is a share in [0, 1)')
    if args.cue_only_share and args.object_cues is None and args.fish_token_table is None:
        raise ValueError('--cue-only-share needs the fish cue (--object-cues or --fish-token-table) to steer by')
    if not 0. <= args.archive_cue_only_share < 1.:
        raise ValueError('--archive-cue-only-share is a share in [0, 1)')
    if args.archive_cue_only_share and args.object_cues is None:
        raise ValueError('--archive-cue-only-share needs the Archive grasp cues (--object-cues)')
    if not 0. <= args.view_dropout < .5:
        raise ValueError('--view-dropout is a probability in [0, 0.5)')
    if args.view_dropout and args.wrist_only:
        raise ValueError('--view-dropout hides one of two cameras: not with --wrist-only')
    if not (np.isfinite(args.wrist_hold_back) and 0. <= args.wrist_hold_back < 1.):
        raise ValueError('--wrist-hold-back is a share in [0, 1)')
    if not (np.isfinite(args.reread_weight) and args.reread_weight >= 0.):
        raise ValueError('--reread-weight is a nonnegative weight')
    if args.reread_weight and not args.wrist_hold_back:
        raise ValueError('--reread-weight trains held-back rows: it needs --wrist-hold-back')
    if args.reread_weight and (args.archive_checkpoint is not None or args.evaluate_checkpoint is not None):
        raise ValueError('--reread-weight acts in the Archive stage, which --archive-checkpoint / --evaluate-checkpoint skip')
    if args.reread_weight and args.fish_train_reader:
        raise ValueError('--reread-weight is an Archive-stage target: not with --fish-train-reader')
    if args.memory_address != 'readback' and args.evaluate_checkpoint is not None:
        raise ValueError('--memory-address builds a model; an evaluated checkpoint keeps its own')
    if args.exam_hold_back and args.evaluate_checkpoint is None:
        raise ValueError('--exam-hold-back measures saved weights: it needs --evaluate-checkpoint')
    if args.archive_horizon != 40 and (args.archive_checkpoint is not None or args.evaluate_checkpoint is not None):
        raise ValueError('--archive-horizon sets the Archive stage, which --archive-checkpoint / --evaluate-checkpoint skip')
    if not 0. <= args.ema < 1.:
        raise ValueError('--ema is a decay in [0, 1)')
    if args.ema and (args.archive_checkpoint is not None or args.evaluate_checkpoint is not None):
        raise ValueError('--ema averages the Archive stage, which --archive-checkpoint / --evaluate-checkpoint skip')
    if args.archive_patience < 0:
        raise ValueError('--archive-patience is a number of evaluations')
    if args.archive_patience and args.archive_select != 'development':
        raise ValueError('--archive-patience ends the Archive stage on the development error: --archive-select development')
    if args.archive_select != 'last' and (args.archive_checkpoint is not None or args.evaluate_checkpoint is not None):
        raise ValueError('--archive-select chooses within the Archive stage, which --archive-checkpoint / '
                         '--evaluate-checkpoint skip')
    if args.archive_schedule == 'constant' and args.archive_warmup:
        raise ValueError('--archive-warmup belongs to --archive-schedule warmup-linear')
    if args.fish_schedule == 'constant' and args.fish_warmup:
        raise ValueError('--fish-warmup belongs to --fish-schedule warmup-linear')
    if args.fish_schedule != 'constant' and not 0 < args.fish_warmup < args.fish_updates:
        raise ValueError('--fish-warmup must be at least 1 and fewer than --fish-updates')
    if not 0. <= args.fish_ema < 1.:
        raise ValueError('--fish-ema is a decay in [0, 1)')
    if args.fish_seed is not None and args.fish_seed < 0:
        raise ValueError('--fish-seed is a nonnegative seed')
    if ((args.fish_schedule != 'constant' or args.fish_ema or args.fish_seed is not None)
            and args.evaluate_checkpoint is not None):
        raise ValueError('--fish-schedule / --fish-ema / --fish-seed set the fish stage, which --evaluate-checkpoint skips')
    if args.archive_schedule != 'constant' and (args.archive_checkpoint is not None or args.evaluate_checkpoint is not None):
        raise ValueError('--archive-schedule sets the Archive stage, which --archive-checkpoint / --evaluate-checkpoint skip')
    if args.continue_archive is not None and (args.archive_checkpoint is not None or args.evaluate_checkpoint is not None):
        raise ValueError('--continue-archive trains the Archive stage further: not with --archive-checkpoint or '
                         '--evaluate-checkpoint')
    if not (np.isfinite(args.place_learning_rate_scale) and args.place_learning_rate_scale > 0):
        raise ValueError('--place-learning-rate-scale must be a positive number')
    if args.place_learning_rate_scale != 1 and args.place_table is None:
        raise ValueError('--place-learning-rate-scale needs --place-table')
    if not (np.isfinite(args.place_goal_learning_rate_scale) and args.place_goal_learning_rate_scale > 0):
        raise ValueError('--place-goal-learning-rate-scale must be a positive number')
    if args.place_goal_learning_rate_scale != 1 and args.place_goal_tables is None:
        raise ValueError('--place-goal-learning-rate-scale needs --place-goal-tables')
    if not (np.isfinite(args.fish_goal_dropout) and 0 <= args.fish_goal_dropout < 1):
        raise ValueError('--fish-goal-dropout must be a share in [0, 1)')
    if args.fish_goal_dropout and args.place_goal_tables is None:
        raise ValueError('--fish-goal-dropout needs --place-goal-tables')
    if not (np.isfinite(args.cue_dropout) and 0 <= args.cue_dropout < 1):
        raise ValueError('--cue-dropout must be a share in [0, 1)')
    if args.cue_dropout and args.object_cues is None:
        raise ValueError('--cue-dropout needs --object-cues')
    if not (np.isfinite(args.carry_lift_weight) and args.carry_lift_weight >= 0):
        raise ValueError('--carry-lift-weight must be finite and at least 0')
    if not (np.isfinite(args.carry_place_factor) and args.carry_place_factor >= 1):
        raise ValueError('--carry-place-factor must be finite and at least 1')
    if args.carry_place_factor != 1 and not args.carry_lift_weight:
        raise ValueError('--carry-place-factor needs --carry-lift-weight')
    if args.carry_fish_only and not args.carry_lift_weight:
        raise ValueError('--carry-fish-only needs --carry-lift-weight')
    if args.carry_over_margin_mm is not None and not (
            args.carry_lift_weight and np.isfinite(args.carry_over_margin_mm) and args.carry_over_margin_mm >= 0):
        raise ValueError('--carry-over-margin-mm needs --carry-lift-weight and is finite and at least 0')
    if not (np.isfinite(args.box_paste_share) and 0 <= args.box_paste_share <= 1):
        raise ValueError('--box-paste-share is a share in [0, 1]')
    if (args.box_paste_share > 0) != (args.box_paste_bank is not None):
        raise ValueError('--box-paste-share and --box-paste-bank go together')
    if not (np.isfinite(args.scene_paste_share) and 0 <= args.scene_paste_share <= 1):
        raise ValueError('--scene-paste-share is a share in [0, 1]')
    if (args.scene_paste_share > 0) != (args.scene_paste_bank is not None):
        raise ValueError('--scene-paste-share and --scene-paste-bank go together')
    if ((args.carry_lift_weight or args.box_paste_share or args.scene_paste_share) and args.object_cues is None
            and args.fish_token_table is None):
        raise ValueError('section 17.120 finds the held fish rows by the grasp cue: --object-cues or --fish-token-table')
    if (args.carry_lift_weight or args.box_paste_share or args.scene_paste_share) and args.evaluate_checkpoint is not None:
        raise ValueError('--carry-lift-weight, --box-paste-share and --scene-paste-share train the fish stage; '
                         '--evaluate-checkpoint trains nothing')
    if not (np.isfinite(args.archive_fish_share) and 0 <= args.archive_fish_share < 1):
        raise ValueError('--archive-fish-share is a share in [0, 1)')
    if args.archive_fish_share:
        if cotraining_rows(args.archive_fish_share, args.batch_size) < 1:
            raise ValueError('--archive-fish-share gives no fish row beside --batch-size Archive rows')
        if args.archive_checkpoint is not None or args.evaluate_checkpoint is not None:
            raise ValueError('--archive-fish-share trains the Archive stage, which --archive-checkpoint / '
                             '--evaluate-checkpoint skip')
        if (args.carry_lift_weight or args.scene_paste_share or args.fish_goal_dropout or args.view_dropout
                or args.wrist_hold_back or args.reread_weight or args.cem_labels is not None or args.world_head
                or args.fish_jaw_teacher):
            raise ValueError('--archive-fish-share draws the co-trained rows\' mirror, patch mask, cue dropout and box '
                             'paste only: not with --carry-lift-weight, --scene-paste-share, --fish-goal-dropout, '
                             '--view-dropout, --wrist-hold-back, --reread-weight, --cem-labels, --world-head or '
                             '--fish-jaw-teacher')
    if args.object_cues is not None and (args.object_cues / 'archive_place_cue.npy').exists() != (
            args.place_table is not None and args.place_table.resolve().parent == args.object_cues.resolve()):
        raise ValueError('an --object-cues directory with Archive place cues goes with --place-table on its own '
                         'fish_place_cue.npy, and that place table only with its directory (section 17.117)')
    cem_table = None
    if args.cem_labels is not None:
        # section 17.110: labels of the first five commands for a stage that supervises exactly those, on the fish token
        # the labels were made with; the jaw teacher would rewrite the same targets, so the two are not combined
        if args.fish_horizon != 5 or (args.fish_token_table is None and args.object_cues is None) or args.fish_jaw_teacher:
            raise ValueError('--cem-labels needs --fish-horizon 5 and the fish token (--fish-token-table, or the outline '
                             'cues of --object-cues), not --fish-jaw-teacher')
        if args.evaluate_checkpoint is not None:
            raise ValueError('--cem-labels trains the fish stage; --evaluate-checkpoint trains nothing')
        cem_table = TeacherLabels.load(args.cem_labels)
        if cem_table.manifest.get('split') != 'train' or (cem_table.manifest.get('smoke') is not False and not args.smoke):
            raise ValueError('--cem-labels must be a formal table of the train split (a smoke table only in a smoke run)')
        # the labels were made on the very fish token this run reads (the outline cues: 11/03, the release Actor)
        made, have = ((cem_table.manifest.get('fish_token_sha256', {}).get('fish_token.npy'),
                       sha256(args.fish_token_table / 'fish_token.npy')) if args.fish_token_table is not None else
                      (cem_table.manifest.get('object_cues_sha256', {}).get('fish_object_cue.npy'),
                       sha256(args.object_cues / 'fish_object_cue.npy')))
        if made != have:
            raise ValueError('the CEM labels were made with another fish token table')
        if cem_table.manifest.get('source_config_sha256') != sha256(args.source_config):
            raise ValueError('the CEM labels were made from another source config')
    if not (np.isfinite(args.cem_unshifted_share) and 0. <= args.cem_unshifted_share <= 1.):
        raise ValueError('--cem-unshifted-share is a share in [0, 1]')
    if not (np.isfinite(args.object_token_learning_rate_scale) and args.object_token_learning_rate_scale > 0):
        raise ValueError('--object-token-learning-rate-scale must be a positive number')
    if args.world_head and (args.archive_checkpoint is not None or args.continue_archive is not None
                            or args.evaluate_checkpoint is not None):
        raise ValueError('--world-head trains a new model with its world head from the start (section 17.113)')
    if not (np.isfinite(args.world_weight) and args.world_weight >= 0 and np.isfinite(args.world_sigreg_weight)
            and args.world_sigreg_weight >= 0 and args.world_sigreg_projections >= 1 and args.world_basis_chains >= 10):
        raise ValueError('invalid world-head weights, projections or basis chains')
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device, device_record = configure_device(args.device, args.gpu_memory_limit_gib)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_record = dict(device_record, tf32_enabled=True)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    log = (args.output / 'training.jsonl').open('a')

    def status(**values):
        write_json(args.output / 'run-status.json', dict(values, elapsed_s=round(time.monotonic() - started, 1)))

    config = json.loads(args.source_config.read_text())
    cue_manifest = None
    if args.object_cues is not None:
        from fcpg.v21_outline_cue import attach_archive_cues, read_manifest
        cue_manifest = read_manifest(args.object_cues)
    if args.fish_token_table is not None and args.object_cues is not None:
        raise SystemExit('--fish-token-table replaces the colour body table; --object-cues has its own fish cues')
    token_manifest = None
    if args.fish_token_table is not None:
        token_manifest = json.loads((args.fish_token_table / 'manifest.json').read_text())
        if token_manifest.get('schema') != 'v21-fish-token-table-v1':
            raise SystemExit(f'{args.fish_token_table} is not a fish token table')
    fish_table = (args.object_cues / 'fish_object_cue.npy' if args.object_cues is not None else
                  args.fish_token_table / 'fish_token.npy' if args.fish_token_table is not None else config['body_table'])
    source = V21Sources(config['source_paths'], device='cpu',
                        object_relative_table=fish_table,
                        place_relative_table=args.place_table, place_goal_tables=args.place_goal_tables)
    layout = install_camera_layout(source, args.camera_layout)
    if args.object_cues is not None:
        attach_archive_cues(source, layout, args.object_cues)
    splits = derive_splits(source)
    blob = None
    if args.evaluate_checkpoint is not None:
        # the saved model decides its own switches (an earlier checkpoint has no second-round fields: defaults)
        blob = torch.load(args.evaluate_checkpoint, map_location='cpu', weights_only=False)
        model_config = V21Config(**blob['config'])
    else:
        model_config = V21Config(len(source.vocabulary.words), seed=args.seed, object_relative=True,
                                 prediction_horizon=40, execution_horizon=5, actor_history=False, reader_task=True,
                                 gripper_separate=args.gripper_separate, gripper_relative=args.gripper_relative,
                                 gripper_absolute_aux=args.gripper_absolute_aux, wrist_only=args.wrist_only,
                                 queries_per_view=args.queries_per_view, actor_width=args.actor_width,
                                 place_relative=args.place_table is not None,
                                 place_goal=args.place_goal_tables is not None, encoder=ENCODER,
                                 actor_layers=args.actor_layers, memory_mode=args.memory_address,
                                 world_head=args.world_head, world_components=args.world_components,
                                 world_target=args.world_target, task_text=not args.no_task_text,
                                 concat_inputs=args.concat_inputs, object_on_pose=args.object_cue_on_pose,
                                 pose_position=args.pose_position)
    if model_config.place_relative != (args.place_table is not None):
        raise ValueError('a model with the place token needs --place-table, and --place-table needs that model')
    if model_config.place_goal != (args.place_goal_tables is not None):
        raise ValueError('a model with the place goal needs --place-goal-tables, and --place-goal-tables needs that model')
    start_blob, place_added, goal_added = None, False, False
    if args.archive_checkpoint is not None:
        if args.evaluate_checkpoint is not None:
            raise ValueError('--archive-checkpoint trains the fish stage; --evaluate-checkpoint trains nothing')
        start_blob = torch.load(args.archive_checkpoint, map_location='cpu', weights_only=False)
        if (start_blob.get('stage') != 'actor_archive' or start_blob.get('complete') is not True
                or start_blob.get('smoke') is not False):
            raise ValueError('--archive-checkpoint must be the complete Archive stage (actor_archive/final.pt) of a formal run')
        if list(start_blob.get('vocabulary') or []) != list(source.vocabulary.words):
            raise ValueError('the Archive-stage checkpoint is not a model of this vocabulary')
        wanted = asdict(model_config)
        # the place projection is the one module the fish stage may add to an Archive stage that never saw a place token
        place_added = model_config.place_relative and not start_blob['config'].get('place_relative', False)
        goal_added = model_config.place_goal and not start_blob['config'].get('place_goal', False)
        added_switches = ({'place_relative'} if place_added else set()) | ({'place_goal'} if goal_added else set())
        theirs = saved_model_config(start_blob['config'])
        differing = sorted(k for k in (set(wanted) | set(theirs)) - added_switches if wanted.get(k) != theirs.get(k))
        if differing:
            raise ValueError(f'the Archive stage was trained with other model switches or another seed: {differing}')
    continue_blob = None
    if args.continue_archive is not None:
        continue_blob = torch.load(args.continue_archive, map_location='cpu', weights_only=False)
        if (continue_blob.get('stage') != 'actor_archive' or continue_blob.get('complete') is not True
                or continue_blob.get('smoke') is not False):
            raise ValueError('--continue-archive must be the complete Archive stage (actor_archive/final.pt) of a formal run')
        if list(continue_blob.get('vocabulary') or []) != list(source.vocabulary.words):
            raise ValueError('the Archive-stage checkpoint is not a model of this vocabulary')
        wanted = asdict(model_config)
        theirs = saved_model_config(continue_blob['config'])
        differing = sorted(k for k in set(wanted) | set(theirs) if wanted.get(k) != theirs.get(k))
        if differing:
            raise ValueError(f'the Archive stage was trained with other model switches or another seed: {differing}')
    archive_train = layout.admitted_rows(source.archive, splits['train']['archive'], wrist=model_config.wrist_only)
    archive_dev_all = layout.admitted_rows(source.archive, splits['development']['archive'], wrist=model_config.wrist_only)
    collections, train_collection = layout.row_collections(source.archive, archive_train)
    collection_rows = np.bincount(train_collection, minlength=len(collections))
    draw_probability = None
    if args.balanced_collections:
        large = collection_rows >= args.large_collection_rows
        if large.sum() < 2:
            raise ValueError('balancing needs at least two large collections')
        share = collection_rows / collection_rows.sum()
        share[large] = share[large].sum() / large.sum()
        draw_probability = (share / collection_rows.clip(1))[train_collection]
        draw_probability = draw_probability / draw_probability.sum()
    archive_dev = np.sort(np.random.default_rng(args.seed + 800).choice(
        archive_dev_all, min(len(archive_dev_all), args.development_rows), replace=False))
    if args.archive_checkpoint is None and args.evaluate_checkpoint is None:
        # before anything is read or trained: a cap the passes cannot reach stops the run here
        archive_learning_rate_factor(1, planned_archive_updates(len(archive_train), args.window_rows, args.batch_size,
                                                                args.archive_passes, args.archive_updates),
                                     args.archive_warmup, args.archive_schedule)
    fish_train, fish_exam = splits['train']['fish'], splits['development']['fish']
    onsets = command_onsets(source)
    fish_draws = moving_rows(source, fish_train, onsets) if args.trim_idle_start else {h: fish_train for h in range(4)}
    exam_moving = moving_rows(source, fish_exam, onsets)
    if args.smoke:
        archive_dev, fish_exam = archive_dev[:8], fish_exam[:8]
        exam_moving = {h: rows[:8] for h, rows in exam_moving.items()}
    vocabulary = source.vocabulary
    scale = training_loss_scale(source, archive_train)

    if args.mirror and not model_config.wrist_only and not args.mirror_external:
        raise ValueError('a mirrored external picture shows the robot on the other side: --mirror needs --wrist-only '
                         '(or --mirror-external, section 17.52)')
    if not 0. <= args.patch_mask < 1.:
        raise ValueError('--patch-mask is a share in [0, 1)')
    episode_roll_centre = None
    if args.mirror:
        # Archive: twice each episode's median wrist roll (its usual roll taken as level; a wrong value only shifts
        # the mirrored roll by a constant, like another roll calibration)
        archive_offsets = np.asarray(source.offsets, np.int64)
        episode_roll_centre = np.asarray([2. * float(np.median(np.asarray(source.raw_q[a:b, 4], np.float64)))
                                          if b > a else 0. for a, b in zip(archive_offsets[:-1], archive_offsets[1:])],
                                         np.float32)
        if len(episode_roll_centre) != len(source.archive.episodes):
            raise ValueError('episode offsets and Archive episodes disagree')
    episode_sided = np.zeros(len(source.archive.episodes), bool)
    if args.mirror and args.mirror_skip_sided_tasks:
        side = re.compile(r'\b(left|right|leftmost|rightmost|clockwise|counterclockwise|anticlockwise)\b', re.I)
        for e, episode in enumerate(source.archive.episodes):
            tasks = episode['manifest_row'].get('tasks') or []
            episode_sided[e] = bool(side.search(' | '.join(map(str, tasks)) if isinstance(tasks, list) else str(tasks)))
    core = V21ResearchCore(model_config).float()
    core.set_loss_scale(scale)
    core = core.to(device)
    core.reader.checkpoint_projection = args.checkpoint_reader
    begin_training_temporal_contract(core, PROGRESS_TRAINING_POLICY)
    donor = torch.load(args.lewm, map_location='cpu', weights_only=False)
    lewm_state = {k[len('lewm.'):]: v for k, v in donor['model'].items() if k.startswith('lewm.')}
    if not any(key.startswith(('film.', 'concat_inputs')) for key in lewm_state):
        raise ValueError('--lewm must be the deployed full-patch LeWM')
    core.lewm = V21FullPatchLeWM(increment_scale=core.actor.increment_scale.detach().cpu(), horizon=40,
                                 **architecture_from_state(lewm_state)).to(device)
    core.lewm.load_state_dict(lewm_state)
    del donor, lewm_state
    lewm_hash = hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes()
                                        for _, v in sorted(core.lewm.state_dict().items()))).hexdigest()
    if cem_table is not None:
        if cem_table.manifest.get('lewm_state_hash') != lewm_hash:
            raise ValueError('the CEM labels were forecast by another LeWM')
        unknown = set(cem_table.arrays['rows'].tolist()) - set(np.asarray(fish_train).tolist())
        if unknown:
            raise ValueError(f'the CEM labels name {len(unknown)} chains outside the fish training split')
    if not args.random_reads:
        # one ascending read per window, straight into the result, without filling the page cache; the 1.2M store
        # (fcpg/v21_p0_features.py) is not a memory map: it reads and decodes whole lossless blocks of 64 rows
        for cache in source.archive_dino:
            if isinstance(cache.tokens, np.memmap):
                cache.tokens = DirectRows(cache.tokens)
    feature_bank = ScratchDinoFeatureBank(source, device, int(args.feature_bank_gib * 1024**3))
    window_features = (None if args.random_reads or start_blob is not None
                       else WindowFeatures(source, device, readers=args.window_readers))
    world_record, world_sigreg = None, None
    if args.world_head:
        # section 17.113: the fixed target (principal components of training wrist frames) and the motion scales,
        # from randomly drawn training chains, before anything is trained
        basis_rows = np.sort(np.random.default_rng(args.seed + 11).choice(
            archive_train, min(args.world_basis_chains, len(archive_train)), replace=False))
        patches, motions, moves = [], [], []
        for begin in range(0, len(basis_rows), 200):
            part = basis_rows[begin:begin + 200]
            features, valid = feature_bank.load(part, 'archive', context_frames=5)
            wrist = features[:, :, 0][torch.as_tensor(np.asarray(valid)[:, :, 0], device=device)]
            patches.append(wrist.reshape(-1, DIM).float())
            chain = source.chain(part, 'archive', images=False)
            q, commands = chain.q[:, :4].reshape(-1, 6), chain.commands.reshape(-1, 5, 6)
            path = chain.paths[:, :, 1:].reshape(-1, 5, 6)
            fine = torch.isfinite(commands).flatten(1).all(1) & torch.isfinite(path).flatten(1).all(1)
            motions.append(WorldHead.motion_input(q, commands)[fine])
            moves.append((path - q[:, None])[fine])
            del features, wrist
        kept = core.world.fit_basis(torch.cat(patches)) if args.world_target == 'patches' else None
        core.world.set_motion_scale(torch.cat(motions).to(device), torch.cat(moves).to(device))
        del patches
        world_sigreg = SIGReg(knots=17, num_proj=args.world_sigreg_projections).to(device)
        world_record = dict(section='17.113', target_kind=args.world_target, components=args.world_components,
                            basis_chains=int(len(basis_rows)),
                            patch_variance_kept=kept, weight=args.world_weight, sigreg_weight=args.world_sigreg_weight,
                            sigreg_projections=args.world_sigreg_projections,
                            learning_rate_scale=args.world_learning_rate_scale,
                            target=('the reader\'s wrist tokens of the next state (frame h + 1, 0.4-0.6 s later), the '
                                    'state rebuilt as the Actor\'s from the very pictures this state read (masked alike) '
                                    'plus frame h + 1 (patch-masked the same way from its own generator, seed + 12), '
                                    'read without gradient with the current reader; mirrored rows with both flipped'
                                    if args.world_target == 'tokens' else
                                    'the change of the wrist picture over the block (frame h to h + 1, 0.4-0.6 s): its '
                                    'frozen DINOv3-L patches minus the basis mean, in the top components, before the '
                                    'patch mask; mirrored rows with both pictures flipped'),
                            inputs='the reader\'s wrist tokens the Actor reads, the pose, the block\'s five recorded commands',
                            loss=('squared error of the next tokens over that of "the tokens do not change" + smooth L1 '
                                  'of the read-backs over their scale + SIGReg on the reader\'s wrist tokens (the Actor\'s '
                                  'read, one slice per token)' if args.world_target == 'tokens' else
                                  'squared change error over the squared change + smooth L1 of the read-backs over their '
                                  'scale + SIGReg on the joint query and the mean patch query before the last LayerNorm'),
                            leak_check='the frame h + 1 is read only for the target; the chain the reader sees ends at h')
        print(json.dumps(dict(world_head=world_record)), flush=True)

    protocol = dict(kind=KIND, version=VERSION, doc='docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md',
        visual_encoder=ENCODER,
        source_config=str(args.source_config), source_config_sha256=sha256(args.source_config),
        camera_layout=layout.identity, archive_rows=dict(train=layout.coverage(source.archive, splits['train']['archive']),
                                                         development=layout.coverage(source.archive, splits['development']['archive'])),
        archive_train_rows=int(len(archive_train)), archive_development_rows_evaluated=int(len(archive_dev)),
        archive_rows_without_wrist_camera='left out (wrist_only)' if model_config.wrist_only else 'kept, slot 0 empty',
        archive_collections={name: int(count) for name, count in zip(collections, collection_rows) if count},
        archive_draws=('uniform over admitted TRAIN rows' if draw_probability is None else dict(
            rule=f'collections with at least {args.large_collection_rows} admitted TRAIN rows share their draws equally; '
                 'the others keep their share; per pass as many draws as rows, with replacement, sorted into windows',
            share={name: float(draw_probability[train_collection == i].sum()) for i, name in enumerate(collections)
                   if collection_rows[i]})),
        fish_train_episodes=list(FISH_TRAIN), fish_exam_episodes=list(FISH_EXAM), fish_episode_2_read=False,
        fish_train_rows=int(len(fish_train)), fish_exam_rows=int(len(fish_exam)),
        lewm=dict(checkpoint=str(args.lewm), sha256=sha256(args.lewm), state_hash=lewm_hash, frozen=True),
        model=asdict(core.config), stages=['actor_archive', 'actor_fish'],
        actor_archive=dict(trained=['reader', 'actor'], batch=args.batch_size, passes=args.archive_passes,
                           cap=args.archive_updates, learning_rate=args.learning_rate,
                           **(dict(weight_average=dict(decay=args.ema, section='17.75', warmup='min(decay, (1+n)/(10+n))',
                                                       used_for='every Archive evaluation and checkpoint, and the '
                                                                'weights the fish stage starts from'))
                              if args.ema else {}),
                           **(dict(schedule=dict(rule=args.archive_schedule, warmup=args.archive_warmup,
                                                 section='17.74', shape='linear rise to --learning-rate, linear fall '
                                                                         'to 0 at the last Archive update'))
                              if args.archive_schedule != 'constant' else {}),
                           **(dict(selection=dict(rule=args.archive_select, section='17.89',
                                                  metric='archive future_mae: the 40-step error on the development rows',
                                                  patience=args.archive_patience,
                                                  handed_on='the evaluated weights (the average with --ema) with the '
                                                            'lowest error, the last weights included',
                                                  as_in='transformers load_best_model_at_end + EarlyStoppingCallback; '
                                                        'Keras EarlyStopping(restore_best_weights=True)'))
                              if args.archive_select != 'last' else {}),
                           supervised_commands=args.archive_horizon,
                           sampling=(f'each batch once per pass, locality windows of {args.window_rows}, '
                                     'random context length 0..3' if draw_probability is None else
                                     f'balanced draws (archive_draws), locality windows of {args.window_rows}, '
                                     'random context length 0..3'),
                           reads=('per-batch row cache' if args.random_reads else
                                  'one ascending read of each window into GPU memory; batches shuffled inside the window'),
                           window_readers=args.window_readers,
                           window_reading=('section 17.82: separate reading processes (forked), int8 values and scales '
                                           'in shared memory' if device.type == 'cuda' else 'threads'),
                           pool_windows=args.pool_windows,
                           cpus=('section 17.82: the training thread on two fast cores of its own, readers, decoders and '
                                 'the batch builder on the other cores' if device.type == 'cuda' else None),
                           prepared_ahead=('section 17.81: the next batch\'s labels and chain in a separate process '
                                           '(forked; never touches the GPU), pinned by the trainer'
                                           if device.type == 'cuda' else
                                           'section 17.78: the next batch\'s labels and chain on a background thread'),
                           deferred_checks=('section 17.80: every input check and count of an update is read once, '
                                            'after the next update is queued (the first failed check raises before '
                                            'anything is logged, evaluated or saved)' if device.type == 'cuda' else None),
                           checkpoint_reader=args.checkpoint_reader,
                           compiled_regions=('section 17.84: training updates run the FCPG read (a wrist-only reader '
                                             'without the other view), the Actor\'s layers, the memory shift, the '
                                             'picture checks and the mirror flip as torch.compile programs; '
                                             'evaluations run uncompiled' if args.compile else None)),
        mirror=(dict(share=.5, rows='training rows of both stages; evaluation rows as recorded (plus fish_mirrored)',
                     fish_roll_centre=args.fish_roll_centre, archive_roll_centre='twice the episode median wrist roll',
                     features='16 x 16 grid of frozen DINO tokens flipped', module='fcpg/v21_mirror.py',
                     sided_tasks=('Archive episodes whose task names a side are never mirrored'
                                  if args.mirror_skip_sided_tasks else 'mirrored like the rest (their words then point '
                                  'to the wrong side)'))
                if args.mirror else None),
        patch_mask=(dict(share=args.patch_mask, rows='training rows of both stages', value='zero', draws='new every time')
                    if args.patch_mask else None),
        **(dict(mirror_external=dict(section='17.52', external_picture='flipped with the wrist picture'))
           if args.mirror_external else {}),
        **(dict(view_dropout=dict(section='17.52', share_each=args.view_dropout,
                                  rows='training rows of both stages; evaluation rows as recorded',
                                  rule='one draw per row: external camera hidden below the share, wrist camera hidden '
                                       'from the share to twice it, only in rows where both cameras have a picture; '
                                       'for the whole chain', generator='torch seed + 8'))
           if args.view_dropout else {}),
        **(dict(cue_only=dict(share=args.cue_only_share, rows='training fish rows whose current wrist fish cue is seen',
                              rule='both cameras\' pictures blank (features zero, exposures still valid, so the Actor keeps the '
                                   'wrist cue) for the whole chain: the cue, the joints and the task remain',
                              generator='torch seed + 26'))
           if args.cue_only_share else {}),
        **(dict(archive_cue_only=dict(share=args.archive_cue_only_share,
                                      rows='training Archive rows whose current wrist grasp cue is seen',
                                      rule='as cue_only: both cameras\' pictures blank for the whole chain, exposures '
                                           'still valid', generator='torch seed + 27'))
           if args.archive_cue_only_share else {}),
        **(dict(wrist_hold_back=dict(section='17.63', share=args.wrist_hold_back,
                                     blocks=f'k uniform in 1..min({HOLD_BACK_BLOCKS}, context)',
                                     rows='training rows of both stages with at least one completed block whose wrist '
                                          'picture k blocks earlier is valid; evaluation rows as recorded',
                                     rule='the newest k wrist pictures do not arrive (invalid, zero features): the wrist '
                                          'memory keeps the picture of k blocks earlier with the joints at its exposure, '
                                          'no current wrist object cue; targets and the external camera unchanged; '
                                          'applied after --view-dropout, before --patch-mask',
                                     generator='torch seed + 9'))
           if args.wrist_hold_back else {}),
        **(dict(reread=dict(section='17.95', weight=args.reread_weight,
                            rows='Archive-stage rows that --wrist-hold-back held back (the current wrist picture existed)',
                            rule=reread_loss.__doc__.split('\n', 1)[0].strip() + ' ' + ' '.join(
                                line.strip() for line in reread_loss.__doc__.split('\n')[1:]).strip(),
                            target='the read of the same row composed before the hold-back (after the mirror and the '
                                   'view drop, before the patch mask), no gradient'))
           if args.reread_weight else {}),
        actor_fish=dict(trained=['reader', 'actor'] if args.fish_train_reader else ['actor'],
                        frozen=['lewm'] if args.fish_train_reader else ['reader', 'lewm'],
                        updates=args.fish_updates, batch=args.fish_batch_size,
                        halves='fish 0/1 and admitted Archive, sampled with replacement', learning_rate=args.fish_learning_rate,
                        archive_draws=(dict(rule='rolling pool of feature-store blocks (ArchivePool, section 17.59): '
                                                 'every block equally likely to enter, the pool\'s rows drawn uniformly, '
                                                 'the oldest blocks replaced after every update', blocks=args.fish_archive_pool,
                                            replaced_per_update=args.fish_archive_pool_refresh,
                                            prefetch='section 17.65: the entering blocks are decoded on a background '
                                                     'thread during the update before; values and draws unchanged')
                                       if args.fish_archive_pool else 'independent over the whole admitted Archive'),
                        keep_steps=sorted(set(args.keep_steps)),
                        **(dict(schedule=dict(rule=args.fish_schedule, warmup=args.fish_warmup, section='17.120',
                                              shape='linear rise to the fish learning rates, linear fall to 0 at the '
                                                    'last fish update'))
                           if args.fish_schedule != 'constant' else {}),
                        **(dict(weight_average=dict(decay=args.fish_ema, section='17.120 (17.75 for the fish stage)',
                                                    warmup='min(decay, (1+n)/(10+n))',
                                                    used_for='every fish evaluation, the kept steps and the final weights'))
                           if args.fish_ema else {}),
                        **(dict(fish_seed=dict(seed=args.fish_seed, generators='numpy fish seed + 4 (rows, mirror, cue '
                                               'dropout), patch mask + 5, torch + 6, numpy + 13 (box paste), numpy + 14 (scene paste)'))
                           if args.fish_seed is not None else {}),
                        supervised_commands=args.fish_horizon,
                        jaw_teacher=(dict(teacher='the model as the fish stage starts (frozen copy)', rows='demonstration rows',
                                          kept_from_demonstration=f'commands at least {JAW_OPENING:g} above the reading '
                                                                  '(the demonstrator opening the jaws)',
                                          archive_rows='unchanged', section='17.46')
                                     if args.fish_jaw_teacher else None),
                        horizon_rule=('fish and Archive rows of this stage: the loss covers the first 5 commands (0.5 s) '
                                      'only; the Actor reads all 40 command times (section 17.86), has 40 output steps '
                                      'and the exams score all 40'
                                      if args.fish_horizon == 5 else 'all 40 commands'),
                        idle_start=dict(trimmed=args.trim_idle_start, onset_rule=ONSET_RULE,
                                        onset_frame=onsets, fish_train_chains_drawn_per_context=[int(len(fish_draws[h])) for h in range(4)],
                                        rule=('chains whose current frame (start + 15 h) is before their demonstration\'s '
                                              'onset are never drawn' if args.trim_idle_start else 'all chains drawn'))),
        fish_exam_moving=dict(set='fish_moving', rule='demonstration 3 chains whose current frame is at or after its onset',
                              chains_per_context=[int(len(exam_moving[h])) for h in range(4)]),
        optimizer='AdamW weight decay 0.01, gradient norm clip 1.0', precision=args.precision, seed=args.seed,
        loss='fcpg/v21_horizon_training.actor_loss (unchanged)' + (
            ' + jaw_absolute_loss / 6 (section 17.46: the auxiliary absolute jaw, one joint\'s weight)'
            if args.gripper_absolute_aux else ''), vision_swap_rule=VISION_SWAP_RULE,
        forecast_in_training='not computed: a history-free model never reads q_error; equality of the proposal checked at step 1',
        reads_while_rebuilding='FCPG reads only in the last observation of a rebuilt state (final_read_only); '
                               'equality of the Actor inputs and the proposal checked at step 1',
        window_reads='the next window is read on a background thread while the current one trains; '
                     'the first such window is checked against the row cache',
        device=device_record, smoke=args.smoke, hardware_used=False)
    if args.place_table is not None:
        manifest = args.place_table.with_suffix('.manifest.json')
        protocol['place'] = dict(table=str(args.place_table), sha256=sha256(args.place_table),
                               manifest=json.loads(manifest.read_text()) if manifest.exists() else None,
                               projection=('added at the fish stage, zero output init' if place_added else
                                           'trained with the rest'),
                               fish_learning_rate=args.fish_learning_rate * args.place_learning_rate_scale,
                               learning_rate_scale=args.place_learning_rate_scale,
                               mirrored='dx and the sine of the doubled angle change sign, as the object token')
    if args.place_goal_tables is not None:
        goal_manifest = args.place_goal_tables / 'place_goal.manifest.json'
        protocol['place_goal'] = dict(
            tables=str(args.place_goal_tables),
            sha256={name: sha256(args.place_goal_tables / name)
                    for name in ('archive_place_goal.npy', 'fish_place_goal.npy', 'place_goal.manifest.json')},
            manifest=json.loads(goal_manifest.read_text()),
            projection=('added at the fish stage, zero output init' if goal_added else
                        'trained with the rest from the Archive stage on, zero output init from its own generator'),
            learning_rate_scale=args.place_goal_learning_rate_scale,
            archive_rows_with_goal=int((source.archive_place_goal_table[
                np.asarray(source.archive.row_endpoint_index)[archive_train]][..., 0] > 0).any(-1).sum()),
            fish_train_rows_with_goal=int((source.fish_place_goal_table[
                np.asarray(source.fish.starts)[fish_train]][:, 0] > 0).sum()),
            archive_rows_with_grasp_target=int((source.archive_place_goal_table[
                np.asarray(source.archive.row_endpoint_index)[archive_train]][..., GOAL_GRASP] > .5).any(-1).sum()),
            fish_train_rows_with_grasp_target=int((source.fish_place_goal_table[
                np.asarray(source.fish.starts)[fish_train]][:, GOAL_GRASP] > .5).sum()),
            mirrored='the turn about the pan axis changes sign (fcpg/v21_place_goal.GOAL_SIGN)',
            vision_swap='kept with its own row (not a visual input)',
            fish_goal_dropout=dict(share=args.fish_goal_dropout, rows='fish rows of the fish stage only; the Archive '
                                   'half and every evaluation keep their goals', draw='one per row per update',
                                   generator='torch seed + 7', section='17.10'))
    if args.fish_token_table is not None:
        protocol['fish_token'] = dict(
            table=str(args.fish_token_table), section='17.57',
            sha256={name: sha256(args.fish_token_table / name) for name in ('fish_token.npy', 'manifest.json')},
            convention=token_manifest['convention'], score_source=token_manifest['score_source'],
            manifest={k: v for k, v in token_manifest.items() if k != 'comparison_with_colour_table'},
            replaces='the colour body table (source config body_table) on fish rows; Archive rows have no token, as before',
            fish_train_rows_with_token=int((source.object_relative_table[
                np.asarray(source.fish.starts)[fish_train]][:, 0] > 0).sum()))
    if world_record is not None:
        protocol['world_head'] = world_record
    if args.object_token_learning_rate_scale != 1:
        protocol['object_token_learning_rate'] = dict(
            section='17.111', parameters='actor.object_relative.* (the wrist object token\'s projection)', stage='fish',
            learning_rate_scale=args.object_token_learning_rate_scale,
            fish_learning_rate=args.fish_learning_rate * args.object_token_learning_rate_scale,
            why='the Archive stage never sees a token (its last layer stays zero); at the fish rate a 40 px move of '
                'the token changed its output by about 0.01 after 1,000 updates (17.110)')
    if cem_table is not None:
        protocol['cem_teacher'] = dict(
            section='17.110', labels=str(args.cem_labels),
            sha256={name: sha256(args.cem_labels / name) for name in ('labels.npz', 'manifest.json')},
            rule=('fish stage, fish rows of a labelled (chain, context): with probability 1 - unshifted_share (a draw per '
                  'row, seed + 10) one of its usable shifted labels drawn uniformly, else its unshifted one; the first five '
                  'arm commands become the label (the alignment CEM\'s selection around the demonstrator\'s commands) and '
                  'the current wrist token moves by that label\'s shift, before the mirror; the gripper, the Archive half '
                  'and every evaluation keep the recorded commands'),
            unshifted_share=args.cem_unshifted_share,
            labelled_pairs=int(len(cem_table.arrays['rows'])),
            usable_plans=int(cem_table.arrays['usable'].sum()),
            manifest={k: cem_table.manifest[k] for k in ('model', 'model_fish_sha256', 'lewm_state_hash', 'response',
                                                         'response_sha256', 'teacher', 'shifts', 'summary', 'skipped')})
    if args.object_cues is not None:
        protocol['object_cues'] = dict(
            tables=str(args.object_cues), section='17.30',
            sha256={name: sha256(args.object_cues / name)
                    for name in ('archive_object_cue.npy', 'fish_object_cue.npy', 'object_cue.manifest.json')},
            manifest={k: v for k, v in cue_manifest.items() if k != 'labels'},
            replaces='the colour body table (source config body_table) on fish rows; Archive rows had no cue before',
            archive_train_rows_with_cue=int((source.archive_object_relative_table[
                np.asarray(source.archive.row_endpoint_index)[archive_train]][..., 0] > 0).any(-1).sum()),
            fish_train_rows_with_cue=int((source.object_relative_table[
                np.asarray(source.fish.starts)[fish_train]][:, 0] > 0).sum()),
            mirrored='dx and the sine of the doubled angle change sign (fcpg/v21_mirror.py CUE_SIGN)',
            vision_swap='swapped with the row\'s other visual inputs (VISION_SWAP_RULE)',
            checks=dict(shift=CUE_SHIFT, near=CUE_NEAR, at_command=GOAL_TURN_AT, moved=GOAL_TURN_MOVED))
        if getattr(source, 'archive_place_relative_table', None) is not None:
            protocol['object_cues']['place'] = dict(
                section='17.117', archive_sha256=sha256(args.object_cues / 'archive_place_cue.npy'),
                fish_table=str(args.place_table), fish_sha256=sha256(args.place_table),
                archive_train_rows_with_cue=int((source.archive_place_relative_table[
                    np.asarray(source.archive.row_endpoint_index)[archive_train]][..., 0] > 0).any(-1).sum()),
                fish_train_rows_with_cue=int((source.place_relative_table[
                    np.asarray(source.fish.starts)[fish_train]][:, 0] > 0).sum()))
        protocol['object_cues']['dropout'] = dict(
            share=args.cue_dropout, generator_seed=args.seed + 11,
            rule='both stages, every row of every update: the grasp cue and, by an independent draw, the place cue '
                 'zeroed (all context frames); evaluations keep them (section 17.117)')
    # section 17.120: the carry-lift penalty and the box- and scene-background pastes, fish stage only
    carry_gate, paste_bank, scene_bank = None, None, None
    if args.carry_lift_weight or args.box_paste_share or args.scene_paste_share:
        from fcpg.so101_kinematics import SO101MuJoCoKinematics
        from scripts.serve_so101_prod_v17_7 import DEFAULT_KINEMATICS_MODEL, TABLE_Z_MM, V177ClearanceGate
        carry_gate = V177ClearanceGate(kinematics=SO101MuJoCoKinematics(DEFAULT_KINEMATICS_MODEL), table_z_mm=TABLE_Z_MM,
                                       planned_clearance_mm=-10., readback_clearance_mm=-10., physical_floor_mm=-10.)
        protocol['carry_lift'] = dict(
            section='17.120', weight=args.carry_lift_weight, place_factor=args.carry_place_factor,
            margin_mm=CARRY_MARGIN_MM, scale_mm=CARRY_SCALE_MM, over_margin_mm=args.carry_over_margin_mm,
            fish_only=args.carry_fish_only,
            held=('rows with the grasp cue on (before --cue-dropout); fish rows also with the jaws reading below '
                  f'{HELD_GRIP:g}'),
            height=f'scripts/serve_so101_prod_v17_7.V177ClearanceGate.clearance_mm (table z {TABLE_Z_MM} mm), '
                   'linearised at the row\'s pose (Archive rows: our arm\'s kinematics at their joints)',
            rows=('fish training rows only (--carry-fish-only); every evaluation unchanged' if args.carry_fish_only else
                  'both stages, Archive and fish training rows; every evaluation unchanged'))
    def background_bank(path):
        blob = torch.load(path, map_location='cpu', weights_only=False)
        bank, background = blob.get('features'), blob.get('background')
        if (blob.get('schema') != BOX_BANK_SCHEMA or bank.ndim != 4 or bank.shape[1] != 3 or bank.shape[-1] != DIM
                or bank.shape[2] != len(background) or int(background.max()) >= PATCHES):
            raise ValueError(f'{path} is not a background bank of these pictures')
        return blob, dict(features=bank.to(device), background=background.to(device))
    if args.box_paste_share:
        blob, paste_bank = background_bank(args.box_paste_bank)
        protocol['box_paste'] = dict(
            section='17.120', bank=str(args.box_paste_bank), bank_sha256=sha256(args.box_paste_bank),
            entries=int(paste_bank['features'].shape[0]), patches=int(len(paste_bank['background'])),
            states=blob.get('states'), share=args.box_paste_share, lift_mm=PASTE_LIFT_MM, generator_seed=args.seed + 13,
            rule='fish stage, held fish rows whose demonstrator lifts the jaws more than lift_mm in the block: with '
                 'this share every wrist exposure (newest to newest) takes the background patches of a random bank '
                 'entry; the label stays the demonstrator\'s')
        del blob
    if args.scene_paste_share:
        blob, scene_bank = background_bank(args.scene_paste_bank)
        if paste_bank is not None and not torch.equal(scene_bank['background'].cpu(), paste_bank['background'].cpu()):
            raise ValueError('the scene bank holds other background patches than the box bank')
        protocol['scene_paste'] = dict(
            section='17.120', bank=str(args.scene_paste_bank), bank_sha256=sha256(args.scene_paste_bank),
            entries=int(scene_bank['features'].shape[0]), patches=int(len(scene_bank['background'])),
            collections=blob.get('bank_collections'), held_out_collections=blob.get('probe_collections'),
            share=args.scene_paste_share, lift_mm=PASTE_LIFT_MM, generator_seed=args.seed + 14,
            rule='fish stage, held fish rows whose demonstrator lifts the jaws more than lift_mm in the block that the '
                 'box paste left alone: with this share every wrist exposure (newest to newest) takes the background '
                 'patches of a random scene-bank entry (another room of the Archive); the label stays the '
                 'demonstrator\'s')
        del blob
    if args.archive_fish_share:
        beside = cotraining_rows(args.archive_fish_share, args.batch_size)
        protocol['actor_archive']['cotraining'] = dict(
            section='17.121', share_asked=args.archive_fish_share, fish_rows=beside, archive_rows=args.batch_size,
            share=beside / (args.batch_size + beside),
            rows='demonstrations 0 and 1: the fish stage\'s draws at the update\'s context, with replacement',
            weights='every row the same: the Archive rows carry archive_rows / (archive_rows + fish_rows) of the loss, '
                    'the fish rows the rest',
            supervised_commands=args.fish_horizon, reader='trained on both (as the Archive stage trains it)',
            augmentations=dict(mirror=bool(args.mirror), patch_mask=args.patch_mask, cue_dropout=args.cue_dropout,
                               box_paste=args.box_paste_share),
            generators='own: numpy seed + 21 (rows, mirror) and + 22 (box paste), patch mask seed + 23, cue dropout '
                       'seed + 24; the RNG of the model\'s dropout is forked around the fish rows',
            archive_rows_unchanged='the Archive rows, their mirror, patch masks and cue drops are drawn as without the '
                                   'fish rows; they get no carry rows; mirror and cue counts include the fish rows',
            as_in='Mobile ALOHA (Fu, Zhao, Finn 2024) co-training; sim-and-real co-training (Maddukuri et al. 2025)')
    if start_blob is not None:
        earlier = start_blob.get('overhaul') or {}
        for key in ('model', 'camera_layout', 'archive_train_rows', 'fish_train_rows', 'archive_draws', 'mirror', 'patch_mask'):
            theirs, mine = earlier.get(key), protocol.get(key)
            if key == 'model':
                theirs = saved_model_config(theirs)
            if key == 'model' and (place_added or goal_added):
                theirs = {k: v for k, v in (theirs or {}).items() if k not in added_switches}
                mine = {k: v for k, v in mine.items() if k not in added_switches}
            if theirs != mine:
                raise ValueError(f'the Archive-stage run differs in {key}')
        if (earlier.get('lewm') or {}).get('sha256') != protocol['lewm']['sha256']:
            raise ValueError('the Archive-stage run used another LeWM')
        protocol.update(stages=['actor_fish'], actor_archive=dict(
            not_run='weights loaded from an earlier run', checkpoint=str(args.archive_checkpoint),
            sha256=sha256(args.archive_checkpoint), step=start_blob.get('step'), run_protocol=earlier.get('actor_archive'),
            fish_stage_generators='new: numpy seed + 4 (draws, mirror), patch mask seed + 5, torch seed + 6',
            check='the loaded weights are evaluated (actor_archive-loaded.json) and compared with the earlier run\'s '
                  'actor_archive-after.json'))
    if continue_blob is not None:
        earlier = continue_blob.get('overhaul') or {}
        for key in ('model', 'camera_layout', 'archive_train_rows', 'fish_train_rows', 'archive_draws', 'mirror', 'patch_mask'):
            if (saved_model_config(earlier.get(key)) if key == 'model' else earlier.get(key)) != protocol.get(key):
                raise ValueError(f'the Archive-stage run differs in {key}')
        if (earlier.get('lewm') or {}).get('sha256') != protocol['lewm']['sha256']:
            raise ValueError('the Archive-stage run used another LeWM')
        protocol['actor_archive'].update(continued_from=dict(
            checkpoint=str(args.continue_archive), sha256=sha256(args.continue_archive), step=continue_blob.get('step'),
            run_protocol=earlier.get('actor_archive'), optimizer='new (AdamW, as a fresh run)',
            generators='new: numpy seed + 11 (windows, contexts, mirror), patch mask seed + 12, torch seed + 13',
            check='actor_archive-before.json evaluates the loaded weights; actor_archive-loaded-check.json compares it '
                  'with the earlier run\'s actor_archive-after.json'))
    write_json(args.output / 'protocol.json', protocol)

    seconds = dict(labels=0., chain=0., features=0., state=0., update=0., window=0., window_wait=0., host_wait=0.)

    def lap(name, since):
        if args.timing and device.type == 'cuda':
            torch.cuda.synchronize(device)
        now = time.monotonic()
        seconds[name] += now - since
        return now

    real_lewm, no_forecast = core.lewm, NoForecast().to(device)

    mask_generator = torch.Generator(device=device)
    mask_generator.manual_seed(args.seed + 3)
    mirror_counts = dict(asked=0, mirrored=0)
    mirrored_rows = torch.zeros((), dtype=torch.long, device=device)   # mirror_counts['mirrored'], on the GPU (17.78c)
    goal_generator = torch.Generator(device=device)    # --fish-goal-dropout only (section 17.10)
    goal_generator.manual_seed(args.seed + 7)
    goal_counts = dict(rows=0, dropped=0)
    cue_generator = torch.Generator(device=device)     # --cue-dropout only (section 17.117)
    cue_generator.manual_seed(args.seed + 11)
    cue_counts = dict(rows=0, grasp_dropped=0, place_dropped=0)
    view_generator = torch.Generator(device=device)    # --view-dropout only (section 17.52)
    view_generator.manual_seed(args.seed + 8)
    view_counts = dict(rows=0, external_hidden=0, wrist_hidden=0)
    cue_only_generator = torch.Generator(device=device)    # --cue-only-share only (11/03)
    cue_only_generator.manual_seed(args.seed + 26)
    cue_only_counts = dict(rows=0, hidden=0)
    archive_cue_only_generator = torch.Generator(device=device)    # --archive-cue-only-share only (11/03)
    archive_cue_only_generator.manual_seed(args.seed + 27)
    archive_cue_only_counts = dict(rows=0, hidden=0)
    hold_generator = torch.Generator(device=device)    # --wrist-hold-back only (section 17.63)
    hold_generator.manual_seed(args.seed + 9)
    hold_counts = dict(rows=0, **{f'held_{n}': 0 for n in range(1, HOLD_BACK_BLOCKS + 1)})
    cem_generator = np.random.default_rng(args.seed + 10)   # --cem-labels only (section 17.110)
    world_mask_generator = torch.Generator(device=device)     # --world-head tokens only (section 17.113): frame h + 1
    world_mask_generator.manual_seed(args.seed + 12)
    cem_counts = dict(rows=0, labelled=0, shifted=0)
    paste_rng = np.random.default_rng(args.seed + 13)       # --box-paste-share only (section 17.120)
    scene_rng = np.random.default_rng(args.seed + 14)       # --scene-paste-share only (section 17.120)
    # --carry-lift-weight / --box-paste-share / --scene-paste-share (section 17.120): prepare() leaves the held rows and
    # their height gradients here for update()
    carry_box, carry_counts = {}, dict(rows=0, held=0, held_seeing_place=0, lifting=0, pasted=0, scene_pasted=0)
    # --reread-weight only (section 17.95): prepare() leaves the target here for update(); the logged values arrive
    # through fcpg.v21_checks.later() once an update is read
    reread_box, reread_seen = {}, {}
    # --world-head only (section 17.113): prepare() leaves the world target here for update() and evaluate(); the
    # logged values arrive through fcpg.v21_checks.later() once an update is read
    world_box, world_seen = {}, {}

    def note_world(values):
        world_seen.update(change_explained=round(values[0], 4), readback_error=round(values[1], 4),
                          sigreg=round(values[2], 4), rows=int(values[3]))

    def world_terms(state, target, world_target):
        """The world head on this batch: -> (loss, explained [tensor], read-back error [tensor], sigreg [tensor]).
        The reader tokens are the Actor's own wrist read; commands and read-backs are the block's recorded ones."""
        goal, usable, _ = world_target
        with compute_context(device, args.precision):     # the reader runs as in core.act()
            tokens = temporal_actor_state(core, state).grounding[:, 0]
            predicted, joints, hidden = core.world(tokens, state.q, target.commands[:, :5])
        tokens_kind = core.world.target == 'tokens'
        change_part, explained = change_loss(predicted, goal, usable, current=tokens if tokens_kind else None)
        valid = target.valid[:, :5].to(joints.dtype)[..., None]
        scaled = torch.nn.functional.smooth_l1_loss(joints.float() / core.world.joint_scale,
                                                    target.readbacks[:, :5].float() / core.world.joint_scale,
                                                    reduction='none')
        readback = (scaled * valid).sum() / (valid.sum() * 6).clamp_min(1.)
        if tokens_kind:                    # the reader's wrist tokens: one slice per token, over the batch
            sigreg = world_sigreg(tokens.float().transpose(0, 1))
        else:
            sigreg = world_sigreg(torch.stack((hidden[:, 0], hidden[:, 1:].mean(1))).float())   # [2, B, width]
        return change_part + readback + args.world_sigreg_weight * sigreg, explained, readback.detach(), sigreg.detach()
    reread_totals = dict(updates=0, held_rows=0, loss_sum=0.)

    def note_reread(values):
        reread_seen.update(loss=round(values[0], 6), held_rows=int(values[1]))
        reread_totals['updates'] += 1
        reread_totals['held_rows'] += int(values[1])
        reread_totals['loss_sum'] += values[0]

    def pinned(value):
        """Every tensor of a TrainingChain, HorizonTargets or CommandTrace in page-locked memory on a GPU run, so that
        its copies to the GPU do not make this thread wait (section 17.78c)."""
        if device.type != 'cuda' or value is None:
            return value
        if isinstance(value, torch.Tensor):
            return value.pin_memory()
        return replace(value, **{f.name: (tuple(pinned(t) for t in getattr(value, f.name)) if f.name == 'traces'
                                          else pinned(getattr(value, f.name)))
                                 for f in fields(value) if isinstance(getattr(value, f.name), (torch.Tensor, tuple))
                                 or hasattr(getattr(value, f.name), '__dataclass_fields__')})

    def on_device(value):
        """pinned()'s tensors moved to the GPU without waiting; the same values as .to(device)."""
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        return replace(value, **{f.name: (tuple(on_device(t) for t in getattr(value, f.name)) if f.name == 'traces'
                                          else on_device(getattr(value, f.name)))
                                 for f in fields(value) if isinstance(getattr(value, f.name), (torch.Tensor, tuple))
                                 or hasattr(getattr(value, f.name), '__dataclass_fields__')})

    def prepare_host(kind, rows, h, mirror=None, pin=True, relabel=False):
        """prepare()'s CPU half: the rows with a future at context h, their labels and their recorded chain, the rows to
        mirror and their roll centres, with the seconds it took. Section 17.78: the Archive stage builds the next
        batch's on a background thread while the GPU trains on this one; 17.78c: in page-locked memory. Section 17.81:
        on a GPU in a separate process (pin=False there: the trainer pins what it receives, pin_built). Section 17.85:
        the rows to mirror are mirrored here, all but their pictures ('mirrored': the rows that were). Section 17.110:
        relabel (training fish rows with --cem-labels) puts the CEM teacher's labels and token shifts in first."""
        page = pinned if pin else (lambda value: value)
        clock = time.monotonic()
        available = horizon_metadata(source, rows, kind, h)[-1][:, 0]
        rows = rows[available]
        if not len(rows):
            return dict(available=available, rows=rows, targets=None, labels=time.monotonic() - clock, chain=0.,
                        drawn=None if mirror is None else np.array(mirror, bool))
        targets = horizon_targets(source, rows, kind, h)
        labelled = time.monotonic()
        host = source.chain(rows, kind, images=False)
        if relabel and kind == 'fish' and cem_table is not None:
            targets, host, taught, moved = cem_table.relabel(rows, h, targets, host, cem_generator,
                                                             args.cem_unshifted_share)
            cem_counts['rows'] += len(rows)
            cem_counts['labelled'] += taught
            cem_counts['shifted'] += moved
        built = dict(available=available, rows=rows, mirror=None, centre=None, mirrored=None,
                     drawn=None if mirror is None else np.array(mirror, bool))
        if mirror is not None:
            mirror = np.asarray(mirror, bool)[available]
            if kind == 'fish':
                centre = np.full(len(rows), args.fish_roll_centre, np.float32)
            else:
                episode = source.archive.endpoint_episode_index[np.asarray(source.archive.row_endpoint_index)[rows, 0]]
                centre = np.asarray(episode_roll_centre[episode])
                mirror = mirror & ~episode_sided[episode]
            # section 17.85: the joint values, object cues and goals of the rows to mirror are mirrored here, on the CPU
            # (in the batch-building process in the Archive stage; the same operations, so the same values as on the
            # GPU); prepare() only flips their pictures
            mirrored = torch.from_numpy(mirror)
            if mirror.any():
                host, targets, mirrored = mirror_rows(host, targets, mirrored, torch.from_numpy(centre))
            built.update(mirror=page(torch.from_numpy(mirror)), centre=page(torch.from_numpy(centre)),
                         mirrored=mirrored.numpy())
        built.update(targets=page(targets), host=page(host))
        return dict(built, labels=labelled - clock, chain=time.monotonic() - labelled)

    def pin_built(built):
        """A batch built in the batch-building process, its tensors copied to page-locked memory (the same values)."""
        return dict(built, **{k: pinned(built[k]) for k in ('targets', 'host', 'mirror', 'centre')
                              if built.get(k) is not None})

    def prepare(kind, rows, h, *, train_reader, window=None, forecast=False, mirror=None, augment=False, fast=True,
                hold=None, built=None, generators=None, carry_rows_too=True):
        """built: prepare_host(kind, rows, h, mirror), when it was made ahead. generators (section 17.121): the
        co-trained fish rows' own patch mask, cue dropout and box paste draws (cotraining_generators); carry_rows_too
        False: no carry rows (the Archive rows of a co-trained Archive stage)."""
        own = generators or dict(mask=mask_generator, cue=cue_generator, paste=paste_rng)
        if built is not None and not ((mirror is None and built['drawn'] is None)
                                      or (mirror is not None and built['drawn'] is not None
                                          and np.array_equal(np.asarray(mirror, bool), built['drawn']))):
            # section 17.81: 17.79 built the Archive batches ahead without their mirror draws, so no Archive row was
            # mirrored from 17.79 to 17.80; a batch built with another draw than this update's now stops the run
            raise ValueError('the batch was built ahead with another mirror draw')
        if built is not None and augment and kind == 'fish' and cem_table is not None:
            raise ValueError('the CEM teacher relabels fish rows while they are prepared, not ahead (section 17.110)')
        built = prepare_host(kind, rows, h, mirror, relabel=augment) if built is None else built
        seconds['labels'] += built['labels']
        seconds['chain'] += built['chain']
        targets, host, rows = built['targets'], built.get('host'), built['rows']
        if targets is None:
            return None
        carry = None
        if augment and carry_rows_too and carry_gate is not None and (kind == 'fish' or not args.carry_fish_only):
            # section 17.120, on the CPU and before the cue dropout: the held rows, the jaws' height gradient at their
            # pose, the demonstrator's lift over the block (mirrored rows with their joints as mirrored) and whether
            # they see where the thing goes
            cue = None if host.object_relative is None else host.object_relative[:, h].numpy()
            carry = carry_rows(carry_gate, host.q[:, h].numpy(), cue, targets.commands[:, :5].numpy(),
                               shut=kind == 'fish')
            place = (np.zeros(len(rows), bool) if host.place_relative is None
                     else host.place_relative[:, h, 0].numpy() > .5)
            carry = (*carry, place)
        clock = time.monotonic()
        mirror = None if built['mirror'] is None else built['mirror']
        asked = 0 if mirror is None else int(mirror.sum())       # counted on the CPU: no wait for the GPU
        targets = on_device(targets)
        world = getattr(core, 'world', None)
        frames = h + 2 if world is not None else h + 1     # section 17.113: frame h + 1 only for the world target
        features, checked = (feature_bank.load(rows, kind, context_frames=frames) if window is None
                             else window.features(rows, frames, as_stored=True))
        future = None
        if world is not None:
            future, future_valid = features[:, h + 1:h + 2], np.asarray(checked)[:, h + 1, 0]
            features, checked = features[:, :h + 1], np.asarray(checked)[:, :h + 1]
        if not np.array_equal(checked, host.view_valid[:, :h + 1].numpy()):
            raise ValueError('cached visual prefix identities differ')
        chain = replace(stage_chain_to_device(host, device), features=features.to(device))
        # section 17.76: a window's pictures stay float16 through the flips, view drops and patch masks below (exact:
        # they only move values or multiply them by 0 or 1). Section 17.78: they stay float16 in the FCPG memory too,
        # whose dtype follows the observation's; the reader turns the memory into float32 as its first step and the
        # history pools .float() values, so every value the model computes with is bit for bit the same, while a
        # batch's pictures and every memory step (three copies in VisualMemory.write) take half the GPU memory
        del features
        if asked:
            # manufactured rows: the same recording seen in a mirror (scene and fish on the other side); their joints,
            # cues and goals were mirrored in prepare_host() (section 17.85), their pictures are flipped here
            done = built['mirrored']
            if done.any():
                chain = replace(chain, features=flipped_features(chain.features, on_device(torch.from_numpy(done))))
                if future is not None:
                    future = flipped_features(future.to(device), on_device(torch.from_numpy(done)))
            if augment:
                mirror_counts['asked'] += asked
                mirrored_rows.add_(int(done.sum()))             # read when the update is logged (section 17.78c)
        if future is not None:
            # section 17.113: the world target, from the pictures as the rows show them (mirrored or not), before any
            # view drop, hold-back or patch mask; frame h + 1 never enters the chain the reader reads
            times = host.endpoint_times.numpy()
            gap = times[:, h + 1] - times[:, h]
            usable = future_valid & host.view_valid[:, h, 0].numpy() & (gap > .4) & (gap < .6)
            if world.target == 'patches':
                goal = world.project(future[:, 0, 0].to(device)) - world.project(chain.features[:, h, 0])
                world_box['target'] = (goal, on_device(torch.from_numpy(usable)), int(usable.sum()))
                future = None
        if carry is not None:
            held, gradient, lift, place = carry
            chosen = np.zeros(len(held), bool)
            if paste_bank is not None and kind == 'fish':
                # section 17.120: two draws per row whether or not it is pasted, so the stream does not depend on the data
                draws = own['paste'].random(len(held)) < args.box_paste_share
                picks = own['paste'].integers(0, len(paste_bank['features']), len(held))
                chosen = held & (lift > PASTE_LIFT_MM) & draws
                if chosen.any():
                    chain = replace(chain, features=paste_background(chain.features, h, np.flatnonzero(chosen),
                                                                     paste_bank['features'], picks[chosen],
                                                                     paste_bank['background']))
                carry_counts['lifting'] += int((held & (lift > PASTE_LIFT_MM)).sum())
                carry_counts['pasted'] += int(chosen.sum())
            if scene_bank is not None and kind == 'fish':
                # the scene paste the same way from its own generator, on the lifting rows the box paste left alone
                draws = scene_rng.random(len(held)) < args.scene_paste_share
                picks = scene_rng.integers(0, len(scene_bank['features']), len(held))
                scened = scene_paste_rows(held & (lift > PASTE_LIFT_MM), chosen, draws)
                if scened.any():
                    chain = replace(chain, features=paste_background(chain.features, h, np.flatnonzero(scened),
                                                                     scene_bank['features'], picks[scened],
                                                                     scene_bank['background']))
                if paste_bank is None:
                    carry_counts['lifting'] += int((held & (lift > PASTE_LIFT_MM)).sum())
                carry_counts['scene_pasted'] += int(scened.sum())
            carry_counts['rows'] += len(held)
            carry_counts['held'] += int(held.sum())
            carry_counts['held_seeing_place'] += int((held & place).sum())
            factor = np.where(place, args.carry_place_factor, 1.).astype(np.float32)
            carry_box['rows'] = (on_device(torch.from_numpy(held)), on_device(torch.from_numpy(gradient).float()),
                                 on_device(torch.from_numpy(factor)))
        if augment and kind == 'fish' and args.fish_goal_dropout:
            chain, dropped = drop_place_goals(chain, args.fish_goal_dropout, goal_generator)
            goal_counts['rows'] += len(rows)
            goal_counts['dropped'] += dropped
        if augment and args.cue_dropout:
            chain, grasp_lost, place_lost = drop_cues(chain, args.cue_dropout, own['cue'])
            cue_counts['rows'] += len(rows)
            cue_counts['grasp_dropped'] += grasp_lost
            cue_counts['place_dropped'] += place_lost
        if augment and args.view_dropout:
            chain, external, wrist = drop_views(chain, args.view_dropout, view_generator)
            view_counts['rows'] += len(rows)
            view_counts['external_hidden'] += external
            view_counts['wrist_hidden'] += wrist
        if augment and args.cue_only_share and kind == 'fish':
            chain, blind = cue_only_views(chain, h, args.cue_only_share, cue_only_generator)
            cue_only_counts['rows'] += len(rows)
            cue_only_counts['hidden'] += blind
        if augment and args.archive_cue_only_share and kind == 'archive':
            chain, blind = cue_only_views(chain, h, args.archive_cue_only_share, archive_cue_only_generator)
            archive_cue_only_counts['rows'] += len(rows)
            archive_cue_only_counts['hidden'] += blind
        if augment and args.wrist_hold_back:
            fresh = None
            if train_reader and args.reread_weight:
                # section 17.95: the read these rows give with their newest wrist pictures, as the Actor would get it
                # (after the mirror and the view drop, before the hold-back and the patch mask); made before the
                # hold-back makes a second copy of the pictures, so only one copy is on the GPU meanwhile
                with (torch.no_grad(), compute_context(device, args.precision),
                      lewm_replaced(core, real_lewm if forecast else no_forecast),
                      final_read_only(core, h) if fast else nullcontext()):
                    fresh = factual_state(core, chain, h, images=True, train_reader=False)
                    fresh = temporal_actor_state(core, fresh).grounding[:, 0]
                seen = chain.view_valid[:, h, 0]
            chain, held = hold_back_wrist(chain, h, args.wrist_hold_back, hold_generator)
            if fresh is not None:
                reread_box['target'] = (fresh, seen & ~chain.view_valid[:, h, 0])
                del fresh
            hold_counts['rows'] += len(rows)
            for n, count in enumerate(held, 1):
                hold_counts[f'held_{n}'] += count
        if hold is not None:                          # --exam-hold-back only (section 17.63)
            chain = withhold_wrist(chain, h, hold)
        if augment and args.patch_mask:
            keep = torch.rand(chain.features.shape[:-1], generator=own['mask'], device=device) >= args.patch_mask
            chain = replace(chain, features=chain.features * keep[..., None].to(chain.features.dtype))
        if future is not None:
            # section 17.113, 'tokens': the next state as the Actor's would be rebuilt one block later, from the very
            # pictures this state read (masked alike) plus frame h + 1 (masked the same way from its own generator, so
            # the Actor's draws are unchanged): only the new picture, the pose and the time differ. Its wrist read,
            # without gradient, is the target the world head predicts from this state's read
            future = future.to(device, chain.features.dtype)
            if augment and args.patch_mask:
                keep = torch.rand(future.shape[:-1], generator=world_mask_generator, device=device) >= args.patch_mask
                future = future * keep[..., None].to(future.dtype)
            full = replace(chain, features=torch.cat((chain.features, future), 1))
            with (torch.no_grad(), compute_context(device, args.precision), lewm_replaced(core, no_forecast),
                  final_read_only(core, h + 1) if fast else nullcontext()):
                ahead = factual_state(core, full, h + 1, images=True, train_reader=False)
                goal = temporal_actor_state(core, ahead).grounding[:, 0].float()
            world_box['target'] = (goal, on_device(torch.from_numpy(usable)), int(usable.sum()))
            del full, ahead, future
        clock = lap('features', clock)
        with (compute_context(device, args.precision), lewm_replaced(core, real_lewm if forecast else no_forecast),
              final_read_only(core, h) if fast else nullcontext()):
            core.lewm.eval()
            state = factual_state(core, chain, h, images=True, train_reader=train_reader)
        lap('state', clock)
        return state, chain.text_ids, targets

    held_development = []

    def development_window():
        """The development rows' pictures on the GPU for one evaluation (section 17.65). On the 32 GB A100 the row
        cache keeps only a few GiB (it leaves 24 GiB of the card free), so the fish stage's Archive pool pushed these
        rows out and most fish-stage evaluations read them again from the disk, once per context (about six minutes
        each). Now they are read once per run into pinned host memory and copied to the GPU for each evaluation;
        the copy is dropped after the evaluation. Checked once against the row cache."""
        first = not held_development
        if first:
            holder = WindowFeatures(source, torch.device('cpu'))
            holder.load(archive_dev)
            holder.close()
            if device.type == 'cuda':
                holder.tokens = holder.tokens.pin_memory()
            held_development.append(holder)
        window = held_development[0].copy_to(device)
        if first:
            mine, mine_valid = window.features(archive_dev[:32], 4)
            reference, reference_valid = feature_bank.load(archive_dev[:32], 'archive', context_frames=4)
            if not torch.equal(mine, reference.to(device)) or not np.array_equal(mine_valid, reference_valid):
                raise ValueError('the held development rows differ from the row cache')
        return window

    @torch.no_grad()
    def evaluate(sets=None):
        """Per set: the proposal's error, the error of "the arm stays where it is", per-joint first-five errors of
        both, and the vision-swap dependence."""
        reports = {}
        with_goal = bool(getattr(core.config, 'place_goal', False))
        with evaluating(core):
            default = [('archive', 'archive', archive_dev), ('fish', 'fish', fish_exam),
                       ('fish_moving', 'fish', exam_moving)]
            if args.mirror:
                default.append(('fish_mirrored', 'fish', fish_exam))
            for name, kind, rows in sets or default:
                entries, swaps, stays, joints, directions, turns, cues = [], [], [], [], [], [], []
                world_sums = torch.zeros(5, dtype=torch.float64)    # section 17.113: error, energy, |read-back|, count,
                                                                    # error with another row's commands
                held = development_window() if rows is archive_dev and not args.random_reads else None
                for h in range(4):
                    part = rows[h] if isinstance(rows, dict) else rows   # a dict gives the rows of each context
                    for begin in range(0, len(part), 32):
                        batch = part[begin:begin + 32]
                        prepared = prepare(kind, batch, h, train_reader=False, window=held,
                                           mirror=np.ones(len(batch), bool) if name == 'fish_mirrored' else None)
                        if prepared is None:
                            continue
                        state, ids, target = prepared
                        world_target = world_box.pop('target', None)
                        if world_target is not None:
                            goal, usable, _ = world_target
                            with compute_context(device, args.precision):
                                tokens = temporal_actor_state(core, state).grounding[:, 0]
                                predicted, joints_w, _ = core.world(tokens, state.q, target.commands[:, :5])
                                other = torch.roll(target.commands[:, :5], 1, 0)    # another row's commands
                                shuffled, _, _ = core.world(tokens, state.q, other)
                            change = goal.float() - (tokens.float() if core.world.target == 'tokens' else 0.)
                            weight_w = usable.to(torch.float32).reshape(-1, 1, 1)
                            valid_w = target.valid[:, :5].to(torch.float32)[..., None]
                            world_sums += torch.stack((
                                ((predicted.float() - goal.float()).square() * weight_w).sum(),
                                (change.square() * weight_w).sum(),
                                ((joints_w.float() - target.readbacks[:, :5].float()).abs() * valid_w).sum(),
                                valid_w.sum() * 6,
                                ((shuffled.float() - goal.float()).square() * weight_w).sum())).double().cpu()
                        with compute_context(device, args.precision):
                            pred = core.act(state, ids, target.command_offsets).commands_position.float()
                            entries.append(horizon_errors(pred, target.commands.float(), target.valid))
                            stay = state.q.float()[:, None].expand_as(pred)
                            stays.append(horizon_errors(stay, target.commands.float(), target.valid))
                            weight = target.valid[:, :5].to(pred)[..., None]
                            joints.append(torch.stack((((pred - target.commands.float()).abs()[:, :5] * weight).sum((0, 1)),
                                                       ((stay - target.commands.float()).abs()[:, :5] * weight).sum((0, 1)),
                                                       weight.sum().expand(6))).cpu())
                            # where the arm (five joints) ends up at the last labelled command: proposed against
                            # recorded direction, on rows whose recorded move is at least 5 units long
                            last = target.valid.long().sum(1).clamp_min(1) - 1
                            at = last[:, None, None].expand(-1, 1, 6)
                            proposed = (pred.gather(1, at)[:, 0] - state.q.float())[:, :5]
                            recorded = (target.commands.float().gather(1, at)[:, 0] - state.q.float())[:, :5]
                            moved = recorded.norm(dim=-1) >= 5.
                            cosine = torch.nn.functional.cosine_similarity(proposed, recorded, dim=-1)
                            directions.append(torch.stack((cosine[moved].sum(), (cosine[moved] > 0).float().sum(),
                                                           (proposed.norm(dim=-1) / recorded.norm(dim=-1).clamp_min(1e-6))[moved].sum(),
                                                           moved.float().sum())).cpu())
                            goal = getattr(state, 'place_goal', None)
                            if with_goal and goal is not None and bool((goal[:, 0] > 0).any()):
                                # section 17.7: the same rows with the goal's turn moved -/+ GOAL_TURN_SHIFT, and with no
                                # goal at all; a goal at larger azimuth needs a falling pan reading (fcpg/v21_place_goal.py)
                                has = goal[:, 0] > 0
                                pans = []
                                for sign in (-1., 1.):
                                    moved_goal = goal.clone()
                                    moved_goal[:, 1] = torch.where(has, goal[:, 1] + sign * GOAL_TURN_SHIFT, goal[:, 1])
                                    shifted = core.act(replace(state, place_goal=moved_goal), ids,
                                                       target.command_offsets).commands_position.float()
                                    pans.append(shifted[:, GOAL_TURN_AT, 0])
                                off = core.act(replace(state, place_goal=torch.zeros_like(goal)), ids,
                                               target.command_offsets).commands_position.float()
                                def turn_sums(rows):
                                    if not bool(rows.any()):
                                        return torch.zeros(8)
                                    difference = (pans[1] - pans[0])[rows]
                                    big = difference.abs() > GOAL_TURN_MOVED
                                    rows_valid = target.valid[rows]
                                    with_error = horizon_errors(pred[rows], target.commands.float()[rows], rows_valid)
                                    without_error = horizon_errors(off[rows], target.commands.float()[rows], rows_valid)
                                    stay_error = horizon_errors(stay[rows], target.commands.float()[rows], rows_valid)
                                    return torch.stack((rows.sum().float(), big.float().sum(),
                                                        (big & (difference < 0)).float().sum(), difference.sum(),
                                                        torch.tensor(with_error['first5_mae'] * with_error['examples'], device=difference.device),
                                                        torch.tensor(without_error['first5_mae'] * without_error['examples'], device=difference.device),
                                                        torch.tensor(float(with_error['examples']), device=difference.device),
                                                        torch.tensor(stay_error['first5_mae'] * stay_error['examples'], device=difference.device))).cpu()
                                # section 17.14: grasp targets (GRASP set) and place goals also separately
                                grasp = has & (goal[:, GOAL_GRASP] > .5)
                                turns.append(torch.stack((turn_sums(has), turn_sums(grasp), turn_sums(has & ~grasp))))
                            for cue_name, given in (('object_relative', args.object_cues is not None),
                                                    ('place_relative', args.place_table is not None)):
                                cue = getattr(state, cue_name, None)
                                if not (given and cue is not None and bool((cue[:, 0] > 0).any())):
                                    continue
                                # section 17.30 (17.117: the place cue too): the rows with a cue, also with the cue
                                # removed; on the rows whose cue is away from where it is held, the pan with the cue's
                                # dx moved -/+ CUE_SHIFT, and the recorded pan change against dx (its sign says which
                                # way the data turns)
                                has_cue = cue[:, 0] > 0
                                near = has_cue & (cue[:, 2:4].float().norm(dim=-1) >= CUE_NEAR)
                                off_cue = core.act(replace(state, **{cue_name: torch.zeros_like(cue)}), ids,
                                                   target.command_offsets).commands_position.float()
                                pans = []
                                for sign in (-1., 1.):
                                    moved_cue = cue.clone()
                                    moved_cue[:, 2] = torch.where(near, cue[:, 2] + sign * CUE_SHIFT, cue[:, 2])
                                    pans.append(core.act(replace(state, **{cue_name: moved_cue}), ids,
                                                         target.command_offsets).commands_position.float()[:, GOAL_TURN_AT, 0])
                                difference = pans[1] - pans[0]
                                recorded_pan = target.commands.float()[:, GOAL_TURN_AT, 0] - state.q.float()[:, 0]
                                related = near & target.valid[:, GOAL_TURN_AT]
                                dx = cue[:, 2].float()

                                def cue_errors(rows):
                                    if not bool(rows.any()):
                                        return [0., 0., 0., 0.]
                                    valid_rows = target.valid[rows]
                                    sums = []
                                    for proposal in (pred, off_cue, stay):
                                        e = horizon_errors(proposal[rows], target.commands.float()[rows], valid_rows)
                                        sums.append(float(e['first5_mae']) * float(e['examples']))
                                    return [*sums, float(horizon_errors(pred[rows], target.commands.float()[rows],
                                                                        valid_rows)['examples'])]
                                big = near & (difference.abs() > GOAL_TURN_MOVED)
                                cues.append((cue_name, torch.tensor([
                                    *cue_errors(has_cue), *cue_errors(near), float(near.sum()), float(big.sum()),
                                    float((big & (difference < 0)).sum()), float((big & (difference > 0)).sum()),
                                    float(difference[near].sum()), float((dx * recorded_pan)[related].sum()),
                                    float((dx ** 2)[related].sum()), float((recorded_pan ** 2)[related].sum()),
                                    float(related.sum())], dtype=torch.float64)))
                            if len(pred) > 1:
                                swapped = core.act(vision_swapped(state), ids, target.command_offsets).commands_position.float()
                                swaps.append(vision_swap_errors(pred, swapped, target.commands.float(), target.valid,
                                                                state.q.float()))
                total = sum(e['examples'] for e in entries)
                report = dict(examples=total, **{key: sum(e[key] * e['examples'] for e in entries) / total
                                                 for key in ('first5_mae', 'future_mae')})
                report.update({'stay_' + key: sum(e[key] * e['examples'] for e in stays) / total
                               for key in ('first5_mae', 'future_mae')})
                summed = torch.stack(joints).sum(0)
                report.update(first5_joint=(summed[0] / summed[2]).tolist(), stay_first5_joint=(summed[1] / summed[2]).tolist())
                heading = torch.stack(directions).sum(0)
                if heading[3] > 0:
                    report['arm_direction'] = dict(rows_that_moved=int(heading[3]), cosine=float(heading[0] / heading[3]),
                                                   same_side=float(heading[1] / heading[3]),
                                                   length_ratio=float(heading[2] / heading[3]))
                count = sum(e['examples'] for e in swaps)
                if count:
                    report['vision_swap'] = dict(examples=count,
                        **{key: sum(e[key] * e['examples'] for e in swaps) / count
                           for key in ('swap_first5_mae', 'swap_future_mae', 'change_first5', 'recorded_difference_first5',
                                       'first5_mae_arm', 'swap_first5_mae_arm')},
                        **{key: (sum(np.asarray(e[key]) * e['examples'] for e in swaps) / count).tolist()
                           for key in ('change_first5_joint', 'recorded_difference_first5_joint')})
                    report['vision_swap']['first5_increase'] = report['vision_swap']['swap_first5_mae'] / report['first5_mae'] - 1
                    report['vision_swap']['first5_increase_arm'] = (report['vision_swap']['swap_first5_mae_arm']
                                                                    / report['vision_swap']['first5_mae_arm'] - 1)
                if turns:
                    summed = torch.stack(turns).sum(0)
                    for key, t in zip(('goal_turn', 'goal_turn_grasp', 'goal_turn_place'), summed):
                        if key != 'goal_turn' and (t[0] == 0 or summed[1, 0] == 0):
                            continue      # the split only for models whose rows carry grasp targets
                        report[key] = dict(
                            rows=int(t[0]), shift_rad=GOAL_TURN_SHIFT, at_command=GOAL_TURN_AT, moved_pairs=int(t[1]),
                            moved_share=float(t[1] / t[0]), towards=float(t[2] / t[1]) if t[1] > 0 else None,
                            mean_pan_difference=float(t[3] / t[0]),
                            first5_mae_with_goal=float(t[4] / t[6]) if t[6] > 0 else None,
                            first5_mae_without_goal=float(t[5] / t[6]) if t[6] > 0 else None,
                            first5_mae_stay=float(t[7] / t[6]) if t[6] > 0 else None,
                            rule='pan at the scored command with the turn moved +shift minus with it moved -shift; '
                                 'towards = share of moved pairs whose pan falls (the goal at larger azimuth)'
                                 + ('; rows with a grasp target' if key == 'goal_turn_grasp' else
                                    '; rows with a place goal' if key == 'goal_turn_place' else ''))
                for cue_name, key in (('object_relative', 'object_cue'), ('place_relative', 'place_cue')):
                    mine = [t for n, t in cues if n == cue_name]
                    if not mine:
                        continue
                    c = torch.stack(mine).sum(0)

                    def cue_block(o):
                        n = float(c[o + 3])
                        return dict(examples=int(n), first5_mae_with_cue=float(c[o] / n) if n > 0 else None,
                                    first5_mae_without_cue=float(c[o + 1] / n) if n > 0 else None,
                                    first5_mae_stay=float(c[o + 2] / n) if n > 0 else None)
                    report[key] = dict(
                        rows_with_cue=cue_block(0), away_from_jaws=cue_block(4),
                        shift=dict(rows=int(c[8]), shift=CUE_SHIFT, near=CUE_NEAR, at_command=GOAL_TURN_AT,
                                   moved_pairs=int(c[9]), falling=int(c[10]), rising=int(c[11]),
                                   mean_pan_difference=float(c[12] / c[8]) if c[8] > 0 else None,
                                   recorded=dict(rows=int(c[16]), dx_times_pan=float(c[13]), dx_squared=float(c[14]),
                                                 pan_squared=float(c[15]),
                                                 correlation=(float(c[13] / torch.sqrt(c[14] * c[15]))
                                                              if c[14] > 0 and c[15] > 0 else None))),
                        rule='examples = rows with a cue at the current frame (away_from_jaws: the cue at least CUE_NEAR from '
                             'where it is held); without_cue = the same rows with the cue zeroed; shift: pan at the scored '
                             'command with dx moved +shift minus -shift, pairs above GOAL_TURN_MOVED counted by sign; '
                             'recorded: the recorded pan change at that command against dx on the same rows (uncentred '
                             'correlation), whose sign the judge takes as the direction towards the cue')
                if getattr(core, 'world', None) is not None and float(world_sums[1]) > 0:
                    report['world'] = dict(change_explained=float(1. - world_sums[0] / world_sums[1]),
                                           shuffled_commands_explained=float(1. - world_sums[4] / world_sums[1]),
                                           readback_mae=float(world_sums[2] / world_sums[3].clamp_min(1.)),
                                           rule='section 17.113: share of the change explained (next wrist tokens, '
                                                'or the wrist picture\'s change; 1 - squared error / squared change; '
                                                'nothing changes = 0), the same with every row given the next row\'s '
                                                'commands, and the mean absolute read-back error over the block')
                reports[name] = report
        return reports

    @torch.no_grad()
    def hold_back_exam():
        """--exam-hold-back (section 17.63): demonstration 3 from its first commanded motion on (fish_moving), rows whose
        wrist pictures of the last k + 1 frames all exist; per k = 1..3 (contexts k..3) the first-five arm error against
        the demonstrator and the distance to the fresh proposal (units) of
          fresh: as recorded; held: the newest k wrist pictures withheld (the memory keeps the one of k blocks earlier,
          with the joints at its exposure; no current wrist cue); here: held, the wrist pictures said to be taken at the
          current joints; zero: held, the wrist pictures' body fields zeroed (read address and the Actor's plain tokens,
          the 17.62 probe's C); blind: no wrist picture at all."""
        hooked = core.reader.frame_metadata

        def zero_wrist_body(memory, actual_q, now, history):
            values = hooked(memory, actual_q, now, history).clone()
            values[:, 0, :, :WRIST_BODY_FIELDS] = 0
            return values

        def propose(state, ids, target, body=True):
            core.reader.frame_metadata = hooked if body else zero_wrist_body
            try:
                with compute_context(device, args.precision):
                    return core.act(state, ids, target.command_offsets).commands_position.float()
            finally:
                core.reader.frame_metadata = hooked

        names = ('fresh', 'held', 'here', 'zero', 'blind')
        report = {}
        with evaluating(core):
            for k in range(1, HOLD_BACK_BLOCKS + 1):
                error = {n: [] for n in names}
                to_fresh = {n: [] for n in names[1:]}
                moves = {n: [] for n in ('here', 'zero', 'blind')}      # held against each: what that input changes
                for h in range(k, 4):
                    part = exam_moving[h]
                    for begin in range(0, len(part), 32):
                        batch = part[begin:begin + 32]
                        whole = source.chain(batch, 'fish', images=False).view_valid[:, h - k:h + 1, 0].all(1).numpy()
                        batch = batch[whole]
                        if not len(batch):
                            continue
                        prepared = [prepare('fish', batch, h, train_reader=False, hold=hold) for hold in (None, k, 'all')]
                        if prepared[0] is None:
                            continue
                        (fresh, ids, target), held, blind = prepared[0], prepared[1][0], prepared[2][0]
                        here = held.memory.q_at_image.clone()
                        here[:, 0] = held.q[:, None, :].to(here)
                        p = dict(fresh=propose(fresh, ids, target), held=propose(held, ids, target),
                                 here=propose(replace(held, memory=replace(held.memory, q_at_image=here)), ids, target),
                                 zero=propose(held, ids, target, body=False), blind=propose(blind, ids, target))
                        weight = target.valid[:, :5].to(p['fresh'])[..., None]
                        rows = weight.sum((1, 2)) > 0
                        count = (weight.sum((1, 2)) * ARM).clamp_min(1)
                        commands = target.commands.float()
                        for n in names:
                            error[n].append(((p[n] - commands).abs()[:, :5, :ARM] * weight).sum((1, 2))[rows] / count[rows])
                            if n != 'fresh':
                                to_fresh[n].append(((p[n] - p['fresh']).abs()[:, :5, :ARM] * weight).sum((1, 2))[rows]
                                                   / count[rows])
                            if n in moves:
                                moves[n].append(((p[n] - p['held']).abs()[:, :5, :ARM] * weight).sum((1, 2))[rows]
                                                / count[rows])
                if not error['fresh']:
                    continue
                error = {n: torch.cat(v).cpu() for n, v in error.items()}
                to_fresh = {n: torch.cat(v).cpu() for n, v in to_fresh.items()}
                report[f'held_{k}'] = dict(
                    rows=int(len(error['fresh'])), contexts=list(range(k, 4)),
                    first5_arm_mae={n: float(v.mean()) for n, v in error.items()},
                    distance_to_fresh={n: float(v.mean()) for n, v in to_fresh.items()},
                    held_closer_to_demonstrator_than={n: float((error['held'] < error[n]).float().mean())
                                                      for n in ('here', 'zero', 'blind')},
                    held_against={n: float(torch.cat(v).mean()) for n, v in moves.items()})
                print(json.dumps({f'held_{k}': report[f'held_{k}']['first5_arm_mae']}), flush=True)
        return report

    def save(stage, step, complete, name, extra=None):
        payload = dict(version=VERSION, contract=CONTRACT, position_unit=POSITION_UNIT, config=asdict(core.config),
                       vocabulary=list(vocabulary.words), text_max_length=vocabulary.max_length,
                       model={k: v.detach().cpu() for k, v in core.state_dict().items()},
                       stage=stage, step=step, complete=complete, smoke=args.smoke, overhaul=protocol,
                       temporal_training_contract=core.temporal_training_contract,
                       temporal_feature_coverage=core.temporal_feature_coverage, **(extra or {}))
        directory = args.output / stage
        directory.mkdir(exist_ok=True)
        path = directory / name
        if path.exists() and name != 'latest.pt':
            raise FileExistsError(f'{path} exists')
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    jaw_teacher = {}                # section 17.46: set as the fish stage starts (--fish-jaw-teacher)

    def update(groups, optimizer, parameters, window=None, commands=40, drawn=None, deferred=False):
        """drawn: per group, its mirror draw and prepare_host() output, made ahead (section 17.78). deferred (17.80):
        called inside fcpg.v21_checks.deferred(); nothing here waits for the GPU, the group losses and the gradient
        norm come back as GPU tensors and are read with the update's checks once the next update is queued. Section
        17.121: a group's optional fifth element holds its own options: window, commands, generators and
        carry_rows_too (as prepare's), weight (its share of the loss instead of 1 / len(groups)) and dropout (its own
        stream of the model's dropout, dropout_stream)."""
        losses = []
        for i, (kind, rows, h, train_reader, *options) in enumerate(groups):
            options = options[0] if options else {}
            if drawn is None:
                mirror, built = (rng.random(len(rows)) < .5 if args.mirror else None), None
            else:
                mirror, built = drawn[i]
            with (compiled_regions(args.compile),        # section 17.84: this update's forward and backward passes
                  dropout_stream(options['dropout'], device) if 'dropout' in options else nullcontext()):
                reread_box.clear()
                world_box.clear()
                carry_box.clear()
                prepared = prepare(kind, rows, h, train_reader=train_reader, window=options.get('window', window),
                                   mirror=mirror, augment=True, built=built, generators=options.get('generators'),
                                   carry_rows_too=options.get('carry_rows_too', True))
                if prepared is None:
                    continue
                state, ids, target = prepared
                reread = reread_box.pop('target', None)        # --reread-weight, Archive stage (section 17.95)
                # section 17.86: the Actor always reads all 40 command times, as in the exams and on the robot; only
                # the loss is cut to the first `commands` (--fish-horizon 5). The v2 Actor's command queries attend to
                # one another, so 5 queries alone are another computation than the first 5 of 40 (until 17.85 the fish
                # stage trained the v2 Actor on 5 queries only; the v1 Actor's causal queries never saw later ones)
                offsets = target.command_offsets
                target = first_commands(target, options.get('commands', commands))
                n = target.commands.shape[1]
                clock = time.monotonic()
                teacher = jaw_teacher.get('core') if kind == 'fish' else None
                with compute_context(device, args.precision):
                    if reread is None:
                        out = core.act(state, ids, offsets)
                    else:
                        # core.act() written out, so that the wrist read the Actor gets is also the re-read's
                        inputs = core.actor_inputs(state)
                        stale = inputs['grounding'][:, 0]
                        out = core.actor(inputs.pop('grounding'), inputs.pop('visual_error'), inputs.pop('q_position'),
                                         ids, command_offsets=offsets, **inputs)
                    action = out.commands_position[:, :n]
                    if teacher is not None:
                        with torch.no_grad(), compiled_regions(False):
                            taught = teacher.act(state, ids, offsets).commands_position[:, :n, ARM]
                if teacher is not None:
                    target, opening = teacher_jaw_targets(target, state.q[:, ARM], taught)
                    jaw_teacher['commands'] += int(target.valid.sum())
                    jaw_teacher['kept_opening'] += int((opening & target.valid).sum())
                loss = actor_loss(action, target, state.q, core.q_loss_scale)
                carry = carry_box.pop('rows', None)
                if carry is not None and args.carry_lift_weight:     # section 17.120: a carry below the demonstrator
                    held, gradient, factor = carry
                    loss = loss + args.carry_lift_weight * carry_lift_penalty(
                        action.float(), target.commands.float(), target.valid, gradient, held, factor,
                        args.carry_over_margin_mm)
                if out.jaw_absolute is not None:
                    loss = loss + jaw_absolute_loss(out.jaw_absolute[:, :n], target, core.q_loss_scale) / 6
                world_target = world_box.pop('target', None)
                if world_target is not None:            # section 17.113: the world head, on the same read
                    world_loss, explained, readback, world_sig = world_terms(state, target, world_target)
                    loss = loss + args.world_weight * world_loss
                    later(torch.stack((explained.float(), readback.float(), world_sig.float(),
                                       explained.new_tensor(float(world_target[2])))), note_world)
                loss = loss * options['weight'] if 'weight' in options else loss / len(groups)
                fail_if(~torch.isfinite(loss), 'nonfinite Actor loss')
                if reread is None:
                    loss.backward()
                else:
                    fresh, held_rows = reread
                    value = reread_loss(stale, fresh, held_rows)
                    fail_if(~torch.isfinite(value), 'nonfinite re-read loss')
                    later(torch.stack((value.detach(), held_rows.sum().to(value))), note_reread)
                    (loss + args.reread_weight * value / len(groups)).backward()
                    del stale, fresh, value
                losses.append(loss.detach())         # the action loss, as logged before
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=not deferred)
        if deferred:
            fail_if(~torch.isfinite(gradient), 'the total norm of the gradients is non-finite, so it cannot be clipped',
                    RuntimeError)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        constrain_training_age_columns(core, optimizer)
        lap('update', clock)
        if deferred:
            return losses, gradient
        return sum((float(loss) for loss in losses), 0.), float(gradient)

    rng = np.random.default_rng(args.seed + 2)
    try:
        if args.evaluate_checkpoint is not None:
            # read-only: how well saved weights fit rows they were trained on, against rows they never saw
            if list(blob.get('vocabulary') or []) != list(vocabulary.words):
                raise ValueError('checkpoint is not a model of this vocabulary')
            core.load_state_dict(blob['model'])
            if args.exam_hold_back:
                write_json(args.output / 'hold-back-exam.json', dict(
                    checkpoint=str(args.evaluate_checkpoint), checkpoint_sha256=sha256(args.evaluate_checkpoint),
                    stage=blob.get('stage'), step=blob.get('step'), section='17.63',
                    rule=hold_back_exam.__doc__.split('\n', 1)[1].strip(), exam=hold_back_exam()))
                status(state='evaluated', evaluation=str(args.output / 'hold-back-exam.json'))
                return
            seen = np.sort(np.random.default_rng(args.seed + 801).choice(
                archive_train, min(len(archive_train), args.development_rows), replace=False))
            # fish_moving: the exam the fish stage's kept steps report (section 17.120: averaged weights are read on it)
            sets = [('archive_development', 'archive', archive_dev), ('archive_train_sample', 'archive', seen),
                    ('fish_train', 'fish', fish_train), ('fish_exam', 'fish', fish_exam),
                    ('fish_moving', 'fish', exam_moving)]
            # the same question per collection: rows it trained on against held-out rows of the same collection
            _, dev_collection = layout.row_collections(source.archive, archive_dev_all)
            for split, rows, index in (('train', archive_train, train_collection), ('development', archive_dev_all, dev_collection)):
                for i, name in enumerate(collections):
                    members = rows[index == i]
                    if len(members) >= args.collection_rows:
                        pick = np.sort(np.random.default_rng(args.seed + 802 + i).choice(
                            members, min(256, len(members)), replace=False))
                        sets.append((f'{split}:{name}', 'archive', pick))
                        if args.by_camera_order:
                            episode = layout.endpoint_episode[np.asarray(source.archive.row_endpoint_index)[members, 0]]
                            swapped = layout.order[episode, 0] == 1
                            for label, part in (('in-order', members[~swapped]), ('swapped', members[swapped])):
                                if len(part) >= 64:
                                    sets.append((f'{split}:{name}:{label}', 'archive', np.sort(
                                        np.random.default_rng(args.seed + 803 + i).choice(part, min(256, len(part)), replace=False))))
            report = evaluate(sets)
            write_json(args.output / 'evaluation.json', dict(
                checkpoint=str(args.evaluate_checkpoint), checkpoint_sha256=sha256(args.evaluate_checkpoint),
                stage=blob.get('stage'), step=blob.get('step'), sets=report))
            status(state='evaluated', evaluation=str(args.output / 'evaluation.json'))
            return
        if start_blob is None:
            # ---- Archive: FCPG + Actor learn to act from what they see ----
            if continue_blob is not None:
                # section 17.72: an earlier run's Archive-stage weights, checked, then more Archive updates
                core.load_state_dict(continue_blob['model'])
                core.temporal_feature_coverage = deepcopy(continue_blob.get('temporal_feature_coverage'))
                loaded_hash = hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes()
                                                      for _, v in sorted(core.lewm.state_dict().items()))).hexdigest()
                if loaded_hash != lewm_hash:
                    raise ValueError('the Archive-stage checkpoint holds another LeWM')
                rng = np.random.default_rng(args.seed + 11)
                mask_generator.manual_seed(args.seed + 12)
                torch.manual_seed(args.seed + 13)
            status(state='evaluating', stage='actor_archive', phase='before')
            loaded = evaluate()
            write_json(args.output / 'actor_archive-before.json', loaded)
            earlier = None if continue_blob is None else args.continue_archive.parent.parent / 'actor_archive-after.json'
            if earlier is not None and earlier.exists():
                # the same weights on the same rows: the numbers should agree up to arithmetic on another device
                before = json.loads(earlier.read_text())
                differences = {f'{name}.{key}': abs(loaded[name][key] - before[name][key])
                               for name in loaded if name in before for key in ('first5_mae', 'future_mae')}
                write_json(args.output / 'actor_archive-loaded-check.json', dict(
                    earlier=str(earlier), largest_difference=max(differences.values()), differences=differences))
                print(json.dumps(dict(loaded_check=max(differences.values()))), flush=True)
            core.eval().requires_grad_(False)
            core.reader.requires_grad_(True).train()
            core.actor.requires_grad_(True).train()
            if getattr(core, 'world', None) is not None:      # section 17.113: trained with them, at the same rate
                core.world.requires_grad_(True).train()
            parameters = [p for p in core.parameters() if p.requires_grad]
            optimizer = fish_optimizer(core, parameters, args.learning_rate, goal_scale=args.place_goal_learning_rate_scale,
                                       fused=device.type == 'cuda',
                                       world_scale=args.world_learning_rate_scale if args.world_head else 1.)
            peak_rates = [group['lr'] for group in optimizer.param_groups]
            average = WeightAverage(parameters, args.ema) if args.ema else None
            averaged = average.swapped if average is not None else nullcontext
            best = BestWeights(parameters) if args.archive_select == 'development' else None     # section 17.89
            stopped_early = None
            size = args.window_rows
            windows = [archive_train[i:i + size] for i in range(0, len(archive_train), size)]

            def balanced_windows():
                # as many draws as rows; ascending, so every window is still one near-sequential read
                rows = np.sort(rng.choice(archive_train, len(archive_train), replace=True, p=draw_probability))
                return [rows[i:i + size] for i in range(0, len(rows), size)]

            step, stage_start, window_bytes, prefetch_checked = 0, time.monotonic(), 0, False
            total = planned_archive_updates(len(archive_train), size, args.batch_size, args.archive_passes,
                                            args.archive_updates, pool=args.pool_windows)
            # every draw of the Archive stage in the order the loop always made them (section 17.82: pool_batches; with
            # --pool-windows 1 the same draws as before), one batch ahead (section 17.78)
            batches = pool_batches(windows, args.archive_passes, args.pool_windows, args.batch_size, size,
                                   args.window_readers, rng, args.mirror,
                                   redraw=balanced_windows if draw_probability is not None else None)
            # section 17.81: on a GPU the next batch is built in a separate process, so that its Python work no longer
            # holds the trainer's interpreter lock (at 17.80 the training thread waited for it half of the time).
            # Section 17.82: the window readers are processes too; both kinds are forked here, before any thread of
            # this stage starts, and a receiving thread pins what the building process returns
            if device.type == 'cuda':
                _CPUS['main'], _CPUS['workers'] = cpu_roles()
            if window_features is not None:
                window_features.start_readers()
            if device.type == 'cuda':
                _BUILDING['prepare'] = lambda *a: prepare_host(*a, pin=False)
                builder = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context('fork'),
                                              initializer=_start_building_process)
                builder.submit(_start_building_process).result()      # forked now
                receiver = ThreadPoolExecutor(1)
                def received(future):
                    on_worker_cpus()
                    return pin_built(future.result())
                build = lambda kind, rows, h, mirror: receiver.submit(
                    received, builder.submit(_build_in_process, kind, rows, h, mirror))
            else:
                builder, receiver = ThreadPoolExecutor(1), None
                build = lambda kind, rows, h, mirror: builder.submit(prepare_host, kind, rows, h, mirror)
            upcoming = next(batches)
            building = build('archive', upcoming[4], upcoming[5], upcoming[6])     # rows, context, mirror draw
            if args.compile:
                start_compile_workers()     # section 17.84: before the pinning below, so they get every core
            if _CPUS.get('main'):
                os.sched_setaffinity(0, _CPUS['main'])      # this thread: fast cores of its own (section 17.82)
            # section 17.80: on a GPU an update's checks and numbers are read once the next update is queued, so the
            # GPU always has the next update to work on; `waiting` is the update not read yet
            defer, waiting = device.type == 'cuda', None
            # section 17.121 (--archive-fish-share): the co-trained fish rows beside every update's Archive rows
            beside = cotraining_rows(args.archive_fish_share, args.batch_size)
            if beside:
                cotrain = cotraining_generators(args.seed, device)
                share = beside / (args.batch_size + beside)
                archive_options = dict(weight=1 - share, carry_rows_too=False)
                fish_options = dict(weight=share, window=None, commands=args.fish_horizon, generators=cotrain,
                                    dropout=cotrain['dropout'])
                cotrain_counts = dict(updates=0, rows=0)

            def logged(entry):
                """Read one update's checks and numbers (the first failed check raises) and log the update."""
                (at, at_epoch, at_h, at_factor, at_elapsed, at_mirror, at_views, at_hold, pending, loss,
                 gradient, at_cue_only) = entry
                mirrored, parts = None, None
                if pending is not None:
                    losses, (gradient,), (mirrored,) = pending.finish((len(loss), 1, 1))
                    loss, parts = sum(losses, 0.), losses
                elif args.mirror:
                    mirrored = int(mirrored_rows)
                record = dict(stage='actor_archive', step=at, of=total, epoch=at_epoch, context=at_h, loss=loss,
                              gradient_norm=gradient, stage_elapsed_s=at_elapsed,
                              **(dict(learning_rate=args.learning_rate * at_factor)
                                 if args.archive_schedule != 'constant' else {}),
                              **(dict(mirrored=dict(at_mirror, mirrored=int(mirrored))) if args.mirror else {}),
                              **(dict(views_hidden=at_views) if args.view_dropout else {}),
                              **(dict(cue_only=at_cue_only) if args.archive_cue_only_share else {}),
                              **(dict(wrist_held_back=at_hold) if args.wrist_hold_back else {}),
                              **(dict(reread=dict(reread_seen, weight=args.reread_weight,
                                                  so_far=dict(reread_totals, loss_mean=round(
                                                      reread_totals['loss_sum'] / max(1, reread_totals['updates']), 6))))
                                 if args.reread_weight else {}),
                              **(dict(carry=dict(carry_counts)) if carry_gate is not None else {}),
                              **(dict(world=dict(world_seen)) if args.world_head else {}),
                              **(dict(cotrain=dict(cotrain_counts, **(dict(
                                  archive_loss=round(float(parts[0]) / (1 - share), 6),
                                  fish_loss=round(float(parts[1]) / share, 6))
                                  if parts is not None and len(parts) == 2 else {}))) if beside else {}))
                log.write(json.dumps(record) + '\n')
                if at == 1 or at % 25 == 0:
                    log.flush()
                    record = dict(record, seconds={k: round(v, 1) for k, v in seconds.items()},
                                  window_read_gib=round(window_bytes / 1024**3, 1))
                    status(state='running', **record)
                    print(json.dumps(record), flush=True)

            while upcoming is not None:
                epoch, entering, leaving, following, rows, h, mirror = upcoming
                clock = time.monotonic()
                built = building.result()       # from here until the next submit, only this thread reads the sources
                seconds['host_wait'] += time.monotonic() - clock
                pending = Pending() if defer else None
                if window_features is not None:
                    if entering:
                        clock = time.monotonic()
                        window_features.prefetch(entering)
                        with deferred(pending) if defer else nullcontext():
                            for key, window_rows in entering:
                                window_bytes += window_features.admit(key, window_rows)
                        lap('window', clock)
                    window_features.prefetch(following)         # section 17.82: the disk always reads the next windows
                    seconds['window_wait'] = window_features.waited
                    if entering and window_features.prefetched and not prefetch_checked:
                        # self-check, once per run: a window read ahead equals the row cache
                        mine, mine_valid = window_features.features(rows, 4)
                        reference, reference_valid = feature_bank.load(rows, 'archive', context_frames=4)
                        if not torch.equal(mine, reference.to(device)) or not np.array_equal(mine_valid, reference_valid):
                            raise ValueError('prefetched window read differs from the row cache')
                        del mine, reference
                        prefetch_checked = True
                step += 1
                if step == 1:
                    # self-check, once per run: the proposal does not depend on the LeWM forecast, and reading
                    # only in the last observation gives the same Actor inputs as the full rebuild
                    with torch.no_grad(), evaluating(core):
                        proposals, inputs = [], []
                        for forecast, fast in ((False, True), (True, False)):
                            state, ids, target = prepare('archive', rows, 3, train_reader=False,
                                                         window=window_features, forecast=forecast, fast=fast)
                            inputs.append((state.grounding.float(), state.visual_error.float()))
                            with compute_context(device, args.precision):
                                proposals.append(core.act(state, ids, target.command_offsets).commands_position.float())
                        if not torch.allclose(proposals[0], proposals[1], rtol=0, atol=1e-4):
                            raise ValueError('the proposal depends on the LeWM forecast or on the skipped reads')
                        if not all(torch.allclose(a, b, rtol=0, atol=1e-4) for a, b in zip(*inputs)):
                            raise ValueError('reading only in the last observation changed the Actor inputs')
                        del proposals, inputs, state
                if step == 1 and window_features is not None:
                    # self-check, once per run: the window read equals the per-batch row cache
                    mine, mine_valid = window_features.features(rows, 4)
                    reference, reference_valid = feature_bank.load(rows, 'archive', context_frames=4)
                    if not torch.equal(mine, reference.to(device)) or not np.array_equal(mine_valid, reference_valid):
                        raise ValueError('sequential window read differs from the row cache')
                    del mine, reference
                upcoming = next(batches, None) if step < total else None
                building = None if upcoming is None else build('archive', upcoming[4], upcoming[5], upcoming[6])
                factor = archive_learning_rate_factor(step, total, args.archive_warmup, args.archive_schedule)
                for group, peak in zip(optimizer.param_groups, peak_rates):
                    group['lr'] = peak * factor
                groups, drawn = [('archive', rows, h, True)], [(mirror, built)]
                if beside:
                    # section 17.121: this update's co-trained fish rows and their mirror draw, from their own generator
                    groups = [('archive', rows, h, True, archive_options),
                              ('fish', cotrain['rows'].choice(fish_draws[h], beside, replace=True), h, True,
                               fish_options)]
                    drawn.append((cotrain['rows'].random(beside) < .5 if args.mirror else None, None))
                    cotrain_counts['updates'] += 1
                    cotrain_counts['rows'] += beside
                with deferred(pending) if defer else nullcontext():
                    loss, gradient = update(groups, optimizer, parameters, window_features,
                                            commands=args.archive_horizon, drawn=drawn, deferred=defer)
                if average is not None:
                    average.update()
                if defer:
                    pending.start(*loss, gradient, mirrored_rows)       # queued after this update, read after the next
                for key in (leaving if window_features is not None else ()):
                    window_features.release(key)        # section 17.82: used once; its last rows are in this update
                entry = (step, epoch, h, factor, round(time.monotonic() - stage_start, 1), dict(mirror_counts),
                         dict(view_counts), dict(hold_counts), pending, loss, gradient, dict(archive_cue_only_counts))
                if not defer:
                    logged(entry)           # nothing was deferred: logged at once, as before
                else:
                    if waiting is not None:
                        logged(waiting)
                    waiting = entry
                if step % args.evaluate_every == 0 and step < total:
                    if waiting is not None:
                        logged(waiting)
                        waiting = None
                    if building is not None:
                        building.result()       # the evaluation reads the same sources: one reader at a time
                    with averaged():
                        evaluated = evaluate()
                        write_json(args.output / f'actor_archive-step-{step:05d}.json', evaluated)
                        save('actor_archive', step, False, 'latest.pt')
                        if best is not None:
                            best.offer(step, evaluated['archive']['future_mae'])
                    if args.archive_patience and best.since >= args.archive_patience:
                        stopped_early = step        # section 17.89: no lower development error for --archive-patience
                        break
                if step >= total:
                    break
            if waiting is not None:
                logged(waiting)
            builder.shutdown(wait=True)
            if receiver is not None:
                receiver.shutdown(wait=True)
            if window_features is not None:
                window_features.close()
                window_features.tokens = None
            if average is not None:
                average.keep()      # the average is the Archive stage's result: evaluated, saved, and fish-trained
            selected = None
            if best is not None:
                # section 17.89: the last weights compete too (after an early end they were the last evaluated ones);
                # the lowest development error is evaluated, saved and fish-trained
                if stopped_early is None:
                    evaluated = evaluate()
                    write_json(args.output / 'actor_archive-last.json', evaluated)
                    best.offer(step, evaluated['archive']['future_mae'])
                if best.weights is None:
                    raise ValueError('no finite development error to choose the Archive weights by')
                if best.step != step:
                    best.restore()
                selected = dict(rule='development', section='17.89', metric='archive future_mae',
                                selected_step=best.step, selected_error=best.error, last_step=step,
                                last_error=evaluated['archive']['future_mae'], planned_updates=total,
                                patience=args.archive_patience, stopped_early=stopped_early)
                write_json(args.output / 'actor_archive-selected.json', selected)
                print(json.dumps(dict(archive_selected=selected)), flush=True)
            write_json(args.output / 'actor_archive-after.json', evaluate())
            save('actor_archive', step, True, 'final.pt', extra=dict(archive_selected=selected) if selected else None)
        else:
            # ---- Archive stage of an earlier run: its saved weights, checked, then the fish stage with new generators ----
            missing, unexpected = core.load_state_dict(start_blob['model'], strict=False)
            added = sorted(k for k in core.state_dict() if (place_added and k.startswith('actor.place_relative.'))
                           or (goal_added and k.startswith('actor.place_goal.')))
            if unexpected or sorted(missing) != added:
                raise ValueError(f'the Archive-stage weights do not fit this model: missing {sorted(missing)}, '
                                 f'unexpected {sorted(unexpected)}')
            core.temporal_feature_coverage = deepcopy(start_blob.get('temporal_feature_coverage'))
            loaded_hash = hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes()
                                                  for _, v in sorted(core.lewm.state_dict().items()))).hexdigest()
            if loaded_hash != lewm_hash:
                raise ValueError('the Archive-stage checkpoint holds another LeWM')
            status(state='evaluating', stage='actor_archive', phase='loaded')
            loaded = evaluate()
            write_json(args.output / 'actor_archive-loaded.json', loaded)
            earlier = args.archive_checkpoint.parent.parent / 'actor_archive-after.json'
            if earlier.exists():
                # the same weights on the same rows: the numbers should agree up to arithmetic on another device
                before = json.loads(earlier.read_text())
                differences = {f'{name}.{key}': abs(loaded[name][key] - before[name][key])
                               for name in loaded if name in before for key in ('first5_mae', 'future_mae')}
                write_json(args.output / 'actor_archive-loaded-check.json', dict(
                    earlier=str(earlier), largest_difference=max(differences.values()), differences=differences))
                print(json.dumps(dict(loaded_check=max(differences.values()))), flush=True)
            rng = np.random.default_rng(args.seed + 4)
            mask_generator.manual_seed(args.seed + 5)
            torch.manual_seed(args.seed + 6)
        if args.fish_seed is not None:
            # section 17.120: another fish stage on the same Archive weights (the seed is part of the model config,
            # so --seed cannot change under --archive-checkpoint); --fish-seed equal to --seed draws as above
            rng, paste_rng, scene_rng = fish_stage_generators(args.fish_seed, mask_generator)

        # ---- Fish: demonstrations 0 and 1; FCPG frozen (unless --fish-train-reader); demonstration 3 is the exam ----
        core.eval().requires_grad_(False)
        core.actor.requires_grad_(True).train()
        if args.fish_train_reader:
            core.reader.requires_grad_(True).train()
        if getattr(core, 'world', None) is not None:          # section 17.113: the world head keeps learning
            core.world.requires_grad_(True).train()
        parameters = [p for p in core.parameters() if p.requires_grad]
        optimizer = fish_optimizer(core, parameters, args.fish_learning_rate, args.place_learning_rate_scale,
                                   fused=device.type == 'cuda', goal_scale=args.place_goal_learning_rate_scale,
                                   token_scale=args.object_token_learning_rate_scale,
                                   world_scale=args.world_learning_rate_scale if args.world_head else 1.)
        half = args.fish_batch_size // 2
        keep = set(args.keep_steps)
        fish_peaks = [group['lr'] for group in optimizer.param_groups]          # --fish-schedule (section 17.120)
        fish_average = WeightAverage(parameters, args.fish_ema) if args.fish_ema else None
        fish_averaged = fish_average.swapped if fish_average is not None else nullcontext
        if args.fish_jaw_teacher:
            # section 17.46: the model as the fish stage starts, frozen, gives the demonstration rows' gripper targets
            # (except where the demonstrator opens the jaws). Built like the core; the LeWM is not read by act().
            teacher = V21ResearchCore(model_config).float()
            teacher.set_loss_scale(scale)
            teacher = teacher.to(device)
            begin_training_temporal_contract(teacher, PROGRESS_TRAINING_POLICY)
            weights = {k: v for k, v in core.state_dict().items() if not k.startswith('lewm.')}
            missing, unexpected = teacher.load_state_dict(weights, strict=False)
            if unexpected or any(not k.startswith('lewm.') for k in missing):
                raise ValueError(f'the jaw teacher does not take the model weights: {missing}, {unexpected}')
            teacher.temporal_feature_coverage = deepcopy(getattr(core, 'temporal_feature_coverage', None))
            teacher.eval().requires_grad_(False)
            jaw_teacher.update(core=teacher, commands=0, kept_opening=0)
        pool = (ArchivePool(archive_train, store_blocks(source, archive_train), args.fish_archive_pool,
                            args.fish_archive_pool_refresh, rng) if args.fish_archive_pool else None)
        # section 17.65: the blocks entering the pool are decoded on a background thread during the update before
        # their rows can be drawn; one at a time, and a failure stops the run
        warm, warming = (ThreadPoolExecutor(1) if pool is not None else None), None
        stage_start = time.monotonic()
        for step in range(1, args.fish_updates + 1):
            h = step % 4
            fish_rows = rng.choice(fish_draws[h], half, replace=True)
            archive_rows = (pool.draw(half) if pool is not None
                            else rng.choice(archive_train, half, replace=True, p=draw_probability))
            if warm is not None:
                if warming is not None:
                    warming.result()
                warming = warm.submit(warm_archive_blocks, source, pool.entered_rows(),
                                      feature_bank.cache if feature_bank is not None else None)
            groups = [('fish', fish_rows, h, args.fish_train_reader),
                      ('archive', archive_rows, h, args.fish_train_reader)]
            factor = archive_learning_rate_factor(step, args.fish_updates, args.fish_warmup, args.fish_schedule)
            if args.fish_schedule != 'constant':
                for group, peak in zip(optimizer.param_groups, fish_peaks):
                    group['lr'] = peak * factor
            loss, gradient = update(groups, optimizer, parameters, commands=args.fish_horizon)
            if fish_average is not None:
                fish_average.update()
            record = dict(stage='actor_fish', step=step, of=args.fish_updates, context=h, loss=loss,
                          gradient_norm=gradient, stage_elapsed_s=round(time.monotonic() - stage_start, 1),
                          **(dict(learning_rate_factor=factor) if args.fish_schedule != 'constant' else {}),
                          **(dict(mirrored=dict(mirror_counts, mirrored=int(mirrored_rows))) if args.mirror else {}),
                          **(dict(goal_dropped=dict(goal_counts)) if args.fish_goal_dropout else {}),
                          **(dict(cue_dropped=dict(cue_counts)) if args.cue_dropout else {}),
                          **(dict(views_hidden=dict(view_counts)) if args.view_dropout else {}),
                          **(dict(cue_only=dict(cue_only_counts)) if args.cue_only_share else {}),
                          **(dict(wrist_held_back=dict(hold_counts)) if args.wrist_hold_back else {}),
                          **(dict(jaw_teacher={k: v for k, v in jaw_teacher.items() if k != 'core'})
                             if jaw_teacher else {}),
                          **(dict(cem_teacher=dict(cem_counts)) if cem_table is not None else {}),
                          **(dict(carry=dict(carry_counts)) if carry_gate is not None else {}),
                          **(dict(world=dict(world_seen)) if args.world_head else {}))
            log.write(json.dumps(record) + '\n')
            if step == 1 or step % 25 == 0:
                log.flush()
                status(state='running', **record)
                print(json.dumps(record), flush=True)
            if step in keep and step < args.fish_updates:
                with fish_averaged():        # --fish-ema: the kept steps are the average (section 17.120)
                    save('actor_fish', step, False, f'step-{step:05d}.pt')
                    write_json(args.output / f'actor_fish-step-{step:05d}.json', evaluate())
        if warm is not None:
            if warming is not None:
                warming.result()
            warm.shutdown()
        if fish_average is not None:
            fish_average.keep()             # the final weights are the average
        write_json(args.output / 'actor_fish-after.json', evaluate())
        save('fish', args.fish_updates, True, 'final.pt')
        lewm_after = hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes()
                                             for _, v in sorted(core.lewm.state_dict().items()))).hexdigest()
        if lewm_after != lewm_hash:
            raise ValueError('the frozen LeWM changed')
        write_json(args.output / 'config.json', dict(kind=KIND, contract=CONTRACT, smoke=args.smoke, complete=True,
            model=asdict(core.config), vocabulary=list(vocabulary.words), fish_updates=args.fish_updates,
            protocol=protocol, deployment_approved=False, physical_success_verified=False))
        status(state='completed', checkpoint=str(args.output / 'fish' / 'final.pt'),
               max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
               feature_bank=feature_bank.stats())
    except BaseException as error:
        status(state='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        log.close()


if __name__ == '__main__':
    main()
