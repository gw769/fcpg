"""Factual 40-step targets from the existing V21 recordings.

Only labels are extended. The visual input and feedback remain a causal prefix
of the original chain. Episode tails are masked, never repeated as labels.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .v21_checks import checking, fail_if
from .v21_memory import VisualObservation
from .v21_models import CommandTrace

PREDICTION_STEPS = 40
EXECUTION_STEPS = 5


@dataclass(frozen=True)
class HorizonTargets:
    trace: CommandTrace
    command_offsets: torch.Tensor
    readback_offsets: torch.Tensor
    commands: torch.Tensor
    readbacks: torch.Tensor
    valid: torch.Tensor
    episode_ids: np.ndarray

    def to(self, device):
        tr = self.trace
        return HorizonTargets(CommandTrace(tr.positions.to(device), tr.offsets.to(device),
            tr.valid.to(device), tr.source, tr.complete), self.command_offsets.to(device),
            self.readback_offsets.to(device), self.commands.to(device),
            self.readbacks.to(device), self.valid.to(device), self.episode_ids)


def frame_grid(fps):
    """Nearest recorded samples to a 10 Hz grid, with their actual source times."""
    fps = np.asarray(fps, dtype=np.float64)
    if fps.ndim != 1 or not np.isfinite(fps).all() or (fps < 10).any() or (fps > 60).any():
        raise ValueError('40-step source requires declared 10 to 60 Hz recording clocks')
    frames = np.rint(fps[:, None] * np.arange(41)[None] / 10).astype(np.int64)
    if not (np.diff(frames, axis=1) > 0).all():
        raise ValueError('future readbacks must use distinct recorded frames')
    return frames, frames / fps[:, None]


def horizon_metadata(source, rows, kind, h):
    rows = np.asarray(rows, dtype=np.int64)
    if (kind not in ('archive', 'fish') or h not in range(4) or rows.ndim != 1
            or not len(rows) or (rows < 0).any() or (rows >= source.counts[kind]).any()):
        raise ValueError('invalid original chain identity or context index')
    if kind == 'fish':
        begin = np.asarray(source.fish.starts[rows]) + 15 * h
        source.fish.validate_frames(begin)
        episodes = np.asarray(source.fish.episodes[begin])
        end_by_episode = {int(e): int(np.flatnonzero(source.fish.episodes == e)[-1]) + 1
                          for e in (0, 1, 3)}
        end = np.array([end_by_episode[int(e)] for e in episodes])
        fps = np.full(len(rows), 30.)
    else:
        endpoint = source.archive.row_endpoint_index[rows, h]
        episodes = np.asarray(source.archive.endpoint_episode_index[endpoint])
        begin = source.offsets[episodes] + source.archive.endpoint_local_frame_index[endpoint]
        end = source.offsets[episodes + 1]
        fps = source.archive.fps[episodes]
    clock = getattr(source, 'recorded_clock', None)
    if clock is None:
        grid, times = frame_grid(fps)
        valid = begin[:, None] + grid[:, 1:] < end[:, None]
    else:
        from .v21_recorded_clock import future_grid
        grid, times, valid = future_grid(clock.times(kind), begin, end)
    return begin, end, episodes, fps, grid, times, valid


def horizon_targets(source, rows, kind, h):
    begin, end, episodes, fps, grid, times, valid = horizon_metadata(source, rows, kind, h)
    if not valid[:, 0].all():
        raise ValueError('source has no actual future readback on the 10 Hz grid')
    b = len(begin)
    commands = np.zeros((b, 40, 6), np.float32)
    readbacks = np.zeros_like(commands)
    # All commands before the last scored readback, not just the 10 Hz BC targets.
    lengths = grid[np.arange(b), valid.sum(1)]
    values = np.zeros((b, int(lengths.max()), 6), np.float32)
    slots = np.arange(values.shape[1])[None]
    available = slots < lengths[:, None]
    clocks = np.where(available, slots / fps[:, None], 0.)
    recorded = getattr(source, 'recorded_clock', None)
    if recorded is not None:
        timebank = recorded.times(kind)
        clocks[:] = 0.
        clocks[available] = timebank[(begin[:, None]+slots)[available]] - np.broadcast_to(timebank[begin, None], available.shape)[available]
    qbank, abank = ((source.fish.q, source.fish.actions) if kind == 'fish'
                    else (source.raw_q, source.raw_actions))
    command_ids = (begin[:, None] + grid[:, :-1])[valid]
    readback_ids = (begin[:, None] + grid[:, 1:])[valid]
    trace_ids = (begin[:, None] + slots)[available]
    if not ((begin + lengths) < end).all():
        raise ValueError('future crosses source episode')
    if kind == 'fish':
        source.fish.validate_frames(np.concatenate((command_ids, readback_ids, trace_ids)))
        if not (source.fish.episodes[readback_ids] == np.broadcast_to(episodes[:, None], valid.shape)[valid]).all():
            raise ValueError('fish future crosses episode')
    # Gather only valid identities; padding never indexes a later episode.
    commands[valid] = abank[command_ids]
    readbacks[valid] = qbank[readback_ids]
    values[available] = abank[trace_ids]
    if not all(np.isfinite(x).all() for x in (commands, readbacks, values)):
        raise ValueError('nonfinite factual horizon labels')
    tensor = lambda x: torch.from_numpy(np.array(x, copy=True))
    trace = CommandTrace(tensor(values), tensor(clocks), tensor(available), 'recorded', True)
    trace.validate(b, max_commands=256)
    return HorizonTargets(trace, tensor(times[:, :-1]), tensor(times[:, 1:]),
                          tensor(commands), tensor(readbacks), tensor(valid), episodes)


def factual_state(core, chain, h, *, images, train_reader=False):
    with torch.set_grad_enabled(torch.is_grad_enabled() and train_reader):
        return _factual_state(core, chain, h, images=images)


def _factual_state(core, chain, h, *, images):
    """Only the h completed real blocks are made available to the policy."""
    def observation(index):
        if not images:
            return None
        if chain.features is None:
            raise ValueError('Actor context requires recorded DINO features')
        return VisualObservation(chain.features[:, index],
            chain.q[:, index, None].expand(-1, 2, -1) if chain.capture_q is None else chain.capture_q[:, index],
            chain.endpoint_times[:, index, None].expand(-1, 2) if chain.capture_times is None else chain.capture_times[:, index], chain.view_valid[:, index],
            object_relative=None if chain.object_relative is None else chain.object_relative[:, index],
            place_relative=None if getattr(chain, 'place_relative', None) is None else chain.place_relative[:, index],
            place_goal=None if getattr(chain, 'place_goal', None) is None else chain.place_goal[:, index],
            camera_types=chain.camera_types,
            view_order=('wrist', 'column') if chain.camera_types is None else ('slot0', 'slot1'))
    # only a task-reading FCPG (V21Config.reader_task) carries the task in its states; earlier models are unchanged
    task = dict(task_ids=chain.text_ids) if getattr(getattr(core, 'config', None), 'reader_task', False) else {}
    state = core.observe(None, observation(0), chain.q[:, 0], chain.endpoint_times[:, 0], **task)
    for index in range(h):
        with torch.no_grad():
            forecast = core.predict(state, chain.traces[index], chain.readback_offsets[:, index, 1:])
        state = core.complete(state, forecast, chain.traces[index], chain.paths[:, index],
                              state.time[:, None] + chain.readback_offsets[:, index], observation(index + 1))
    return state


def masked_mean(values, valid):
    if values.ndim != 3 or valid.shape != values.shape[:2] or valid.dtype != torch.bool:
        raise ValueError('loss requires an explicit per-readback label mask')
    if checking():
        fail_if(~valid.any(1).all(), 'every example needs actual labels')
    weights = valid.to(values)[..., None]
    return (values * weights).sum((1, 2)).div(valid.sum(1) * values.shape[-1]).mean()


def actor_loss(action, targets, q, scale):
    position = F.smooth_l1_loss(action / scale, targets.commands / scale, reduction='none')
    velocity = F.smooth_l1_loss(
        torch.diff(torch.cat((q[:, None], action), 1), dim=1) / scale,
        torch.diff(torch.cat((q[:, None], targets.commands), 1), dim=1) / scale, reduction='none')
    # Equal attention to the executable prefix and the entire supervised future.
    return (masked_mean(position + .2 * velocity, targets.valid)
            + masked_mean(position[:, :5] + .2 * velocity[:, :5], targets.valid[:, :5])) / 2


def lewm_loss(response, targets, scale):
    prediction = response.predicted_q_path[:, 1:]
    error = (prediction - targets.readbacks) / scale
    regression = F.smooth_l1_loss(prediction / scale, targets.readbacks / scale, reduction='none')
    sigma = (response.q_uncertainty / scale).clamp_min(1e-3)
    uncertainty = .5 * (error.detach() / sigma).square() + sigma.log()
    values = regression + .05 * uncertainty
    return (masked_mean(values, targets.valid) + masked_mean(values[:, :5], targets.valid[:, :5])) / 2


@torch.no_grad()
def horizon_errors(prediction, target, valid):
    error = (prediction - target).abs()
    return {'first5_mae': float(masked_mean(error[:, :5], valid[:, :5])),
            'future_mae': float(masked_mean(error, valid)),
            'full40_examples': int(valid.all(1).sum()), 'examples': len(valid)}
