"""C4 scene scorer (docs/V21_LEWM_C4_INTEGRATION_PLAN_20261009.md sections 3-5, step 1).

C4 is an 8-time scene scorer: for each candidate command sequence it predicts the scene at
t+0.5 ... t+4.0 s (one vector per camera) and scores the t+4.0 s prediction against a goal
embedding with the le-wm latent goal cost. It predicts no joints and no uncertainty; joint
responses and physical admission stay with the production LeWM and the existing checks.

Scoring only: this module imports no robot, service, Actor, FCPG or CEM code, keeps no state
between calls, writes no memory and returns no executable command.

One window, time-major, cameras in VIEWS order:
  history_patches [3, 2, 256, 384]  DINOv2-S/14 banks at t-1.0, t-0.5, t
  history_valid   [3, 2]            bool: that camera has an exposure at that time
  commands        [N, 50, 6]        per candidate: the actually dispatched past 10 + a candidate
                                    future 40 (lerobot_calibrated_position, 10 Hz, 5 per 0.5 s block)
  command_valid   [50]              bool, the same for every candidate
  goal_patches    [2, 256, 384], goal_valid [2]   the t+4.0 s target (a hindsight frame or a goal image)

`offline_candidates` builds the eight candidates of the offline test for one window exactly as
scripts/eval_v21_lewm_standalone.py `alternatives` does; the partner window is named by the caller.
`batch_partners` is the offline test's partner rule inside one evaluation batch.
"""
import numpy as np
import torch

from .v21_lewm_standalone import HISTORY, latent_goal_cost
from .v21_lewm_standalone_data import action_blocks

VIEWS = ('wrist', 'column')
CANDIDATES = ('factual', 'hold', 'shuffle', 'reversed', 'other_row', 'noise_a', 'noise_b', 'noise_c')
PAST, FUTURE, JOINTS = 10, 40, 6
PATCHES, DINO_DIM = 256, 384


def _eval_mode(model):
    if model.training:
        raise ValueError('C4 must be in eval mode: BatchNorm statistics are never updated here')


def offline_candidates(commands, command_valid, q_now, row, partner_commands):
    """[50, 6] dispatched commands of one window -> [8, 50, 6] candidates (past 10 untouched).

    hold = q at t; shuffle/reversed reorder the valid future slots (shuffle seeded by the window
    id); other_row = the partner's future change moved onto this window's first future command;
    noise a/b/c = smooth random walks (std 10/sqrt(40) units per step) seeded by the window id.
    """
    if commands.shape != (PAST + FUTURE, JOINTS) or partner_commands.shape != commands.shape:
        raise ValueError('commands and partner commands must be [50, 6]')
    if command_valid.shape != (PAST + FUTURE,) or q_now.shape != (JOINTS,):
        raise ValueError('command validity must be [50] and q_now [6]')
    out = commands[None].repeat(len(CANDIDATES), 1, 1)
    future = slice(PAST, PAST + FUTURE)
    out[1, future] = q_now[None]
    slots = torch.nonzero(command_valid[future]).flatten() + PAST
    order = torch.as_tensor(np.random.default_rng(21004 + int(row)).permutation(len(slots)))
    out[2, slots] = commands[slots[order]]
    out[3, slots] = commands[slots.flip(0)]
    out[4, future] = commands[PAST:PAST + 1] + (partner_commands[future] - partner_commands[PAST:PAST + 1])
    rng = np.random.default_rng(31004 + int(row))
    for k in range(3):
        steps = torch.as_tensor(rng.normal(0., 10. / np.sqrt(40.), size=(FUTURE, JOINTS)), dtype=commands.dtype)
        out[5 + k, future] = commands[future] + steps.cumsum(0)
    return out


def batch_partners(complete):
    """Offline partner rule inside one evaluation batch: the next window (cyclically) whose 50
    commands are all valid; the next window if none is."""
    n = len(complete)
    return [next((j % n for j in range(i + 1, i + n) if complete[j % n]), (i + 1) % n) for i in range(n)]


def rank_of_first(costs):
    """1 + number of candidates cheaper than the first (ties count half), as in the offline test."""
    c = np.asarray(costs, np.float64)
    return float(1 + (c[1:] < c[0]).sum() + 0.5 * (c[1:] == c[0]).sum())


def _check(history_valid, commands, command_valid, goal_valid):
    if history_valid.shape != (HISTORY, len(VIEWS)) or history_valid.dtype != torch.bool:
        raise ValueError('history validity must be bool [3, 2] (t-1.0, t-0.5, t; wrist, column)')
    if commands.ndim != 3 or commands.shape[1:] != (PAST + FUTURE, JOINTS) or not len(commands):
        raise ValueError('commands must be [N, 50, 6]: dispatched past 10 + candidate future 40')
    if command_valid.shape != (PAST + FUTURE,) or command_valid.dtype != torch.bool:
        raise ValueError('command validity must be bool [50]')
    if goal_valid.shape != (len(VIEWS),) or goal_valid.dtype != torch.bool:
        raise ValueError('goal validity must be bool [2]')
    if not torch.equal(commands[:, :PAST], commands[:1, :PAST].expand(len(commands), -1, -1)):
        raise ValueError('every candidate shares the dispatched past 10 commands')


@torch.no_grad()
def encode_window(model, history_patches, history_valid, goal_patches, goal_valid):
    """-> (history embeddings [3, 2, D], goal embedding [2, D]); the four frames in one batch."""
    _eval_mode(model)
    if history_patches.shape != (HISTORY, len(VIEWS), PATCHES, DINO_DIM):
        raise ValueError('history patches must be [3, 2, 256, 384]')
    if goal_patches.shape != (len(VIEWS), PATCHES, DINO_DIM):
        raise ValueError('goal patches must be [2, 256, 384]')
    patches = torch.cat((history_patches, goal_patches[None]))[None]
    valid = torch.cat((history_valid, goal_valid[None]))[None]
    emb = model.encode(patches, valid)[0]
    return emb[:HISTORY], emb[HISTORY]


@torch.no_grad()
def score_embeddings(model, history_emb, history_valid, commands, command_valid, goal_emb, goal_valid):
    """N candidates of one encoded window -> dict(costs [N] float or None, goal_cameras, predicted [N, 8, 2, D]).

    A cost exists when every history time has an exposure, all ten command blocks are complete and
    at least one camera has a goal; it averages over those cameras (offline test rule).
    """
    _eval_mode(model)
    _check(history_valid, commands, command_valid, goal_valid)
    n = len(commands)
    blocks, blocks_ok = action_blocks(commands, command_valid[None].expand(n, -1))
    act = model.action_encoder(blocks.to(history_emb.device))
    predicted = model.rollout(history_emb[None].expand(n, -1, -1, -1),
                              history_valid[None].expand(n, -1, -1).to(history_emb.device), act)
    history_ok = bool(history_valid.any(-1).all())
    goal_ok = goal_valid & history_ok & bool(blocks_ok[0].all())
    if not goal_ok.any():
        return dict(costs=None, goal_cameras=goal_ok.tolist(), predicted=predicted)
    costs = latent_goal_cost(predicted[:, -1], goal_emb, goal_ok)
    return dict(costs=costs, goal_cameras=goal_ok.tolist(), predicted=predicted)


def score_window(model, history_patches, history_valid, commands, command_valid, goal_patches, goal_valid,
                 *, autocast_dtype=None):
    """Encode the four frames and score N candidates; autocast_dtype None (fp32), bfloat16 or float16."""
    device = next(model.parameters()).device
    move = lambda x: x.to(device)
    with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        history_emb, goal_emb = encode_window(model, move(history_patches), move(history_valid),
                                              move(goal_patches), move(goal_valid))
        return score_embeddings(model, history_emb, move(history_valid), move(commands), move(command_valid),
                                goal_emb, move(goal_valid))
