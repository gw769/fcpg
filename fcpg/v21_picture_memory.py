"""17.166 (11/01; James 「3張我覺得不太足夠，萬一是30張呢」「直接30張」): the LeWM's memory of the last N pictures each
camera took (fcpg/v21_full_patch_lewm.py memory_pictures=N), built for a training or evaluation row from the same
places the chain's three pictures come from: the frozen DINO cache rows (V21Sources._get_archive_dino /
_feature_values, the camera layout's remap included) and the recorded clock's exposure packets (validity, capture
time, joints at the exposure; the layout's LayoutClock). The chain's memory (fcpg/v21_memory.py VisualMemory.write)
keeps per camera the last valid, new exposures; this does the same over the recording's earlier pictures:

* the candidates, newest first, are the row's current picture (Archive: its endpoint row_endpoint_index[row, h];
  demonstrations: its chain endpoint frame) and the ones `stride` half-seconds before it, ... (Archive endpoint ids
  of one episode are consecutive, 0.5 s apart; demonstration pictures are every frame of 30 Hz recordings, so a
  half-second is 15 frames), never before the start of the recording (nor into another one);
* per camera, a candidate counts when its view is valid and its capture time is earlier than the one counted before
  it (the same exposure twice, or one out of order, is skipped, as VisualMemory.write skips or refuses it);
* `drop` [B, 2] pictures are left out from the newest end first (the camera did not see for that long: training's
  view loss; 0 for every evaluation), then the next N are kept, right-aligned oldest -> newest.

With stride 1 and nothing dropped, a camera's newest three are the chain's own three whenever the chain had three
(tests/test_v21_picture_memory.py checks this against factual_state on real rows and every value of the slots).
Demonstration frames are checked with the source's validate_frames (the excluded demonstration is never read).
"""
from dataclasses import dataclass

import numpy as np
import torch

from .v21_encoder import DIM, PATCHES

DEMO_HALF_SECOND = 15          # frames of the 30 Hz demonstrations in 0.5 s (their chain endpoints are 15 apart)


@dataclass(frozen=True)
class PictureMemory:
    """Per camera the last N pictures taken, right-aligned, oldest -> newest; slots before a camera's first picture
    are empty (valid False, zero features). ids: the picture's identity (Archive endpoint id or demonstration frame),
    -1 where empty; strides: the half-seconds between the candidates of each row."""
    features: torch.Tensor      # [B, 2, N, PATCHES, DIM]
    q_at_image: torch.Tensor    # [B, 2, N, 6]
    image_times: torch.Tensor   # [B, 2, N] float64
    valid: torch.Tensor         # [B, 2, N] bool
    ids: np.ndarray             # [B, 2, N] int64
    strides: np.ndarray         # [B] int64


def select_pictures(valid, times, pictures, drop):
    """valid / times [B, M, 2] of the candidates, newest first -> [B, 2, N] candidate positions (oldest -> newest,
    right-aligned, -1 empty): per camera the valid candidates whose capture time is earlier than the last one counted,
    the newest drop[b, c] of them left out, then the next `pictures`."""
    b, m, _ = valid.shape
    out = np.full((b, 2, pictures), -1, np.int64)
    for row in range(b):
        for camera in range(2):
            kept, last = [], np.inf
            for j in range(m):
                if valid[row, j, camera] and times[row, j, camera] < last:
                    kept.append(j)
                    last = times[row, j, camera]
            kept = kept[int(drop[row, camera]):int(drop[row, camera]) + pictures]
            if kept:
                out[row, camera, pictures - len(kept):] = kept[::-1]
    return out


def _assemble(source, ids, inside, kind, pictures, drop, strides, features_for, device):
    """ids / inside [B, M] candidate identities newest first -> PictureMemory."""
    b, m = ids.shape
    safe = np.where(inside, ids, ids[:, :1])
    valid, times, q = (np.asarray(x) for x in source.recorded_clock.observations(safe.reshape(-1), kind))
    valid = valid.reshape(b, m, 2) & inside[..., None]
    times, q = times.reshape(b, m, 2).astype(np.float64), q.reshape(b, m, 2, 6)
    where = select_pictures(valid, times, pictures, drop)                      # [B, 2, N] candidate positions
    rows = np.broadcast_to(np.arange(b)[:, None, None], where.shape)
    cameras = np.broadcast_to(np.arange(2)[None, :, None], where.shape)
    taken = where >= 0
    chosen = np.where(taken, ids[rows, np.maximum(where, 0)], -1)
    unique, inverse = np.unique(chosen[taken], return_inverse=True)
    position = np.where(taken, where, 0)
    if len(unique):
        tokens, checked = features_for(unique)                                  # [U, 2, PATCHES, DIM], [U, 2]
        tokens = torch.as_tensor(tokens) if not isinstance(tokens, torch.Tensor) else tokens
        if not np.array_equal(np.asarray(checked)[inverse, cameras[taken]], valid[rows, position, cameras][taken]):
            raise ValueError('DINO cache and recorded clock disagree on which views were taken')
        dtype = tokens.dtype if tokens.is_floating_point() else torch.float32
        features = torch.zeros(b, 2, pictures, PATCHES, DIM, device=device, dtype=dtype)
        index = torch.as_tensor(inverse, device=tokens.device)
        view = torch.as_tensor(cameras[taken], device=tokens.device)
        features[torch.as_tensor(taken, device=device)] = tokens[index, view].to(device=device, dtype=dtype)
    else:
        features = torch.zeros(b, 2, pictures, PATCHES, DIM, device=device)
    q_at = np.where(taken[..., None], q[rows, position, cameras], 0.).astype(np.float32)
    at = np.where(taken, times[rows, position, cameras], 0.)
    if not np.isfinite(q_at[taken]).all() or not np.isfinite(at[taken]).all():
        raise ValueError('nonfinite capture q or time in the picture memory')
    return PictureMemory(features, torch.as_tensor(q_at, device=device), torch.as_tensor(at, device=device),
                         torch.as_tensor(taken, device=device), chosen, np.asarray(strides, np.int64))


def candidates(current, episodes_of, strides, count, step=1):
    """current [B] identities, strides [B] (half-seconds) -> ids [B, count] newest first and whether each is in the
    current one's recording; `step` identities per half-second (Archive endpoints 1, demonstration frames 15)."""
    current = np.asarray(current, np.int64)
    ids = current[:, None] - (np.asarray(strides, np.int64) * step)[:, None] * np.arange(count)[None]
    inside = ids >= 0
    episode = episodes_of(np.maximum(ids, 0))
    inside &= episode == episodes_of(current)[:, None]
    return ids, inside


def archive_picture_memory(source, rows, h, pictures, *, strides=None, drop=None, features_for=None, device='cpu',
                           extra=8):
    """The memory of Archive chain rows at context h (after h completed blocks): candidates from the row's current
    endpoint back; features_for(ids) -> (tokens, valid) defaults to the source's own cache reading (the training run
    passes its window's pictures, the same values)."""
    rows = np.asarray(rows, np.int64)
    b = len(rows)
    strides = np.ones(b, np.int64) if strides is None else np.asarray(strides, np.int64)
    drop = np.zeros((b, 2), np.int64) if drop is None else np.asarray(drop, np.int64)
    current = np.asarray(source.archive.row_endpoint_index)[rows, h]
    episode = np.asarray(source.archive.endpoint_episode_index)
    count = pictures + int(drop.max(initial=0)) + extra
    ids, inside = candidates(current, lambda x: episode[x], strides, count)
    inside &= np.arange(count)[None] < (pictures + drop.max(1) + extra)[:, None]     # each row's own reach
    reader = features_for or source._get_archive_dino
    return _assemble(source, ids, inside, 'archive', pictures, drop, strides, reader, device)


def fish_picture_memory(source, rows, h, pictures, *, strides=None, drop=None, device='cpu', extra=8,
                        features_for=None):
    """The memory of demonstration chain rows at context h: candidates from the chain's endpoint frame h back, 15
    frames per half-second, inside its own demonstration; every frame read is checked by validate_frames.
    features_for(ids) -> (tokens, valid) defaults to the source's own cache reading (the training run passes its
    feature bank's, fcpg/v21_scratch_features.py pictures: the same values)."""
    rows = np.asarray(rows, np.int64)
    b = len(rows)
    strides = np.ones(b, np.int64) if strides is None else np.asarray(strides, np.int64)
    drop = np.zeros((b, 2), np.int64) if drop is None else np.asarray(drop, np.int64)
    fish = source.fish
    current = np.asarray(fish.chains(rows)['endpoint_index'])[:, h]
    episode = np.asarray(fish.episodes)
    count = pictures + int(drop.max(initial=0)) + extra
    ids, inside = candidates(current, lambda x: episode[np.minimum(x, len(episode) - 1)], strides, count,
                             step=DEMO_HALF_SECOND)
    inside &= np.arange(count)[None] < (pictures + drop.max(1) + extra)[:, None]
    present = np.asarray(source.fish_dino.endpoint_index)
    at = np.searchsorted(present, np.maximum(ids, 0)).clip(max=len(present) - 1)
    inside &= present[at] == ids                                    # a frame without cached features is no picture
    fish.validate_frames(np.where(inside, ids, current[:, None]).reshape(-1))

    def reader(unique):
        fish.validate_frames(unique)
        return (source._feature_values(source.fish_dino, unique) if features_for is None
                else features_for(unique))
    return _assemble(source, ids, inside, 'fish', pictures, drop, strides, reader, device)


def from_visual_memory(memory, pictures):
    """A VisualMemory (the chain's or a robot snapshot's three pictures per camera, right-aligned oldest -> newest)
    as a PictureMemory of N slots: its pictures in the newest three, the rest empty."""
    b, n = memory.features.shape[0], memory.features.shape[2]
    if n > pictures:
        raise ValueError('more pictures than the memory holds')
    if (memory.valid[..., :-1] & ~memory.valid[..., 1:]).any():
        raise ValueError('a three-picture memory must be right-aligned')
    pad = pictures - n
    grow = lambda x, value: torch.cat((torch.full((b, 2, pad, *x.shape[3:]), value, dtype=x.dtype, device=x.device), x), 2)
    return PictureMemory(grow(memory.features, 0), grow(memory.q_at_image, 0), grow(memory.image_times, 0),
                         grow(memory.valid, False), np.full((b, 2, pictures), -1, np.int64), np.ones(b, np.int64))


ROBOT_MEMORY_SCHEMA = 'v21-robot-picture-memory-v2'     # v2 (11/01): one stream per actual state, not per session


def robot_states(session, counter, now):
    """[blocks] the actual state of every block (numbered across the pack), the blocks in session and record order.
    The service starts every state (each typed task, even the same text again: fcpg/v21_core.py initial_state) with an
    empty memory and its completed-readback counter (memory.current_readback_index) at 0, and the counter grows with
    every completed block, so a counter that does not grow starts a new state (robot_state_bounds_20261101.py on the
    10/31 packs: no picture of the block before, 25 s or more after it; inside a state at most 7 s between blocks)."""
    session, counter, now = (np.asarray(x) for x in (session, counter, now))
    state = np.zeros(len(session), np.int64)
    for r in range(1, len(session)):
        same = session[r] == session[r - 1] and counter[r] > counter[r - 1]
        if same and not now[r] > now[r - 1]:
            raise ValueError('the blocks of a state are not in time order')
        state[r] = state[r - 1] + (not same)
    return state


def robot_memory_index(pack, slots):
    """17.166: the robot's picture streams and, per block, its last `slots` pictures per camera. The service keeps
    three pictures per camera (one per plan, ~1.2 s apart) and every block's snapshot holds them; the LeWM's memory
    starts empty with every actual state, as the service's own (James 11/01 「可以空著阿大哥，動態的」), so a state's
    stream is the union of its blocks' pictures (robot_states; one per camera and capture time, the same exposure must
    have the same tokens and joints wherever it appears); a block's memory is its state's stream pictures of that camera
    taken no later than the newest one its own snapshot held, the last `slots` of them, right-aligned (its own three
    are their newest three: checked). Returns the stream (tokens [U, PATCHES, DIM] float16, q [U, 6], times [U]
    float64, camera [U], session [U], state [U]), every block's state and index [blocks, 2, slots] into the stream
    (-1 empty)."""
    data = pack['data']
    memory, session = data['memory'], data['session'].numpy()
    if 'current_readback_index' not in memory:
        raise ValueError('the pack keeps no completed-readback counter: the actual states cannot be told apart')
    state = robot_states(session, memory['current_readback_index'].numpy(), data['now'].numpy())
    features, q_at, times, valid = (memory[k] for k in ('features', 'q_at_image', 'image_times', 'valid'))
    stream, where = [], {}
    for r in range(len(session)):
        for c in range(2):
            for j in range(features.shape[2]):
                if not bool(valid[r, c, j]):
                    continue
                key = (int(session[r]), int(state[r]), c, float(times[r, c, j]))
                if key in where:
                    u = where[key]
                    if not (torch.equal(stream[u][0], features[r, c, j]) and torch.equal(stream[u][1], q_at[r, c, j])):
                        raise ValueError('one exposure with two sets of tokens or joints in a state')
                    continue
                where[key] = len(stream)
                stream.append((features[r, c, j], q_at[r, c, j], key))
    order = sorted(range(len(stream)), key=lambda u: stream[u][2])          # by session, state, camera, time
    rank = {u: i for i, u in enumerate(order)}
    tokens = torch.stack([stream[u][0] for u in order]).to(torch.float16)
    q = torch.stack([stream[u][1] for u in order]).float()
    keys = [stream[u][2] for u in order]
    s_of, st_of, c_of, t_of = (np.asarray([k[i] for k in keys]) for i in range(4))
    index = np.full((len(session), 2, slots), -1, np.int64)
    for r in range(len(session)):
        for c in range(2):
            own = valid[r, c].numpy()
            if not own.any():
                continue
            newest = float(times[r, c][torch.as_tensor(own)].max())
            mine = np.flatnonzero((st_of == state[r]) & (c_of == c) & (t_of <= newest))
            kept = mine[-slots:]
            index[r, c, slots - len(kept):] = kept
            held = sorted(rank[where[(int(session[r]), int(state[r]), c, float(times[r, c, j]))]]
                          for j in np.flatnonzero(own))
            if index[r, c, slots - len(held):].tolist() != held:
                raise ValueError("a block's own pictures are not the newest of its memory")
    return dict(schema=ROBOT_MEMORY_SCHEMA, slots=slots, split=pack['split'], blocks=len(session),
                tokens=tokens, q=q, times=torch.as_tensor(t_of, dtype=torch.float64),
                camera=torch.as_tensor(c_of), session=torch.as_tensor(s_of), state=torch.as_tensor(st_of),
                block_state=torch.as_tensor(state), index=torch.as_tensor(index))


class RobotPictureMemory:
    """The robot blocks' picture memories (robot_memory_index on the device): memory(ids, pictures, lost) gives the
    PictureMemory of N pictures per camera; lost [B, 2] seconds: a camera's pictures younger than that at the block's
    time are left out (the view lost that long), the memory then holds the N before."""

    def __init__(self, index, device):
        if index.get('schema') != ROBOT_MEMORY_SCHEMA:
            raise ValueError(f'not a {ROBOT_MEMORY_SCHEMA} index')
        self.slots = int(index['slots'])
        to = lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
        self.tokens, self.q, self.times = to(index['tokens']), to(index['q']), to(index['times'])
        self.index = to(index['index'])
        self.device = torch.device(device)

    def memory(self, ids, now, pictures, lost=None):
        ids = torch.as_tensor(ids, device=self.device)
        index = self.index[ids]                                            # [B, 2, slots]
        b = len(ids)
        taken = index >= 0
        times = torch.where(taken, self.times[index.clamp(min=0)], torch.zeros((), dtype=torch.float64,
                                                                            device=self.device))
        if lost is not None:
            age = now.to(self.device, torch.float64)[:, None, None] - times
            taken &= age >= torch.as_tensor(lost, dtype=torch.float64, device=self.device)[..., None]
        # right-align what is left, newest last, then keep the last N
        order = torch.argsort(taken.to(torch.int8), dim=-1, stable=True)
        index, taken = index.gather(-1, order)[..., -pictures:], taken.gather(-1, order)[..., -pictures:]
        safe = index.clamp(min=0)
        features = torch.where(taken[..., None, None], self.tokens[safe], torch.zeros((), dtype=self.tokens.dtype,
                                                                                      device=self.device))
        q = torch.where(taken[..., None], self.q[safe], torch.zeros((), device=self.device))
        at = torch.where(taken, self.times[safe], torch.zeros((), dtype=torch.float64, device=self.device))
        return PictureMemory(features, q, at, taken, torch.where(taken, index, -1).cpu().numpy(), np.ones(b, np.int64))


def lookback_endpoints(pictures, strides, seconds, evaluation_stride=1, extra=8):
    """How many endpoints before a chain's first one archive_picture_memory can reach: at stride s it reads
    pictures + (the most view_loss_drops drops at s) + extra candidates, s endpoints apart (a row's current endpoint
    is never before its chain's first)."""
    most = lambda s: max(1, int(seconds / (.5 * s)))
    return max((pictures + most(s) + extra - 1) * s for s in (*strides, evaluation_stride))


def view_loss_drops(rng, b, pictures, strides, probability, seconds):
    """[B, 2] training drops: with `probability` a row loses one camera (either) for 0.5 s up to `seconds`, in its own
    stride's half-seconds (at least one picture)."""
    drop = np.zeros((b, 2), np.int64)
    lose = rng.random(b) < probability
    camera = rng.integers(0, 2, b)
    most = np.maximum(1, (seconds / (.5 * np.asarray(strides))).astype(np.int64))
    count = 1 + (rng.random(b) * most).astype(np.int64)
    drop[np.flatnonzero(lose), camera[lose]] = np.minimum(count[lose], most[lose])
    return drop


# 17.167 (11/01; James 「改好阿，可以空著阿大哥，動態的」): the robot service's own picture memory. A runtime's state
# (fcpg/v21_core.py ActualState.pictures, kept when load_runtime sets core.live_pictures) starts every task with N
# empty slots per camera and takes in each new exposure as the three-picture memory does (VisualMemory.write's
# `inserted`), the oldest leaving once N are held: the rule robot_memory_index rebuilds from the snapshots.

def empty_pictures(batch, pictures, *, device='cpu', dtype=torch.float32):
    """A state's picture memory before its first exposure: N empty slots per camera (no identities, no strides)."""
    with torch.inference_mode(False), torch.no_grad():
        return PictureMemory(torch.zeros(batch, 2, pictures, PATCHES, DIM, device=device, dtype=dtype),
                             torch.zeros(batch, 2, pictures, 6, device=device),
                             torch.zeros(batch, 2, pictures, device=device, dtype=torch.float64),
                             torch.zeros(batch, 2, pictures, device=device, dtype=torch.bool), None, None)


def write_pictures(pictures, observation, inserted):
    """The picture memory after one observation: every camera with a new exposure (inserted [B, 2], as VisualMemory.write
    took it in) gets it in its newest slot, its oldest leaving; the other cameras stay as they were."""
    def push(old, new):
        with torch.inference_mode(False), torch.no_grad():
            select = inserted.to(old.device).reshape(*inserted.shape, *([1] * (old.ndim - 2)))
            return torch.where(select, torch.cat((old[:, :, 1:], new.to(old).detach()[:, :, None]), 2), old)
    return PictureMemory(push(pictures.features, observation.features), push(pictures.q_at_image, observation.q_at_image),
                         push(pictures.image_times, observation.image_times),
                         push(pictures.valid, torch.ones_like(inserted)), None, None)


def expand_pictures(pictures, n):
    """One state's picture memory for the planner's n candidates: views, nothing copied, so the LeWM makes the memory
    once for all of them (fcpg/v21_full_patch_lewm.py)."""
    if len(pictures.valid) != 1:
        raise ValueError("the planner expands one state's picture memory")
    grow = lambda x: x.expand(n, *x.shape[1:])
    return PictureMemory(grow(pictures.features), grow(pictures.q_at_image), grow(pictures.image_times),
                         grow(pictures.valid), None, None)


def snapshot_pictures(pictures):
    """A state's picture memory as the service's snapshots keep it: capture times, joints and validity on the CPU,
    without the tokens (60 pictures would add 24 MB to every block's snapshot; robot_memory_index puts them back
    together from the snapshots' own three per camera). A LeWM refuses it: its patches are empty."""
    cpu = lambda x: x.detach().cpu().clone()
    return PictureMemory(pictures.features.new_zeros((*pictures.features.shape[:3], 0, 0), device='cpu'),
                         cpu(pictures.q_at_image), cpu(pictures.image_times), cpu(pictures.valid), None, None)
