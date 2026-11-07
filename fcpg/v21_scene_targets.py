"""Actual future camera exposures, looked up by same-episode frame identity.

Scene supervision defaults to query 5/10/.../40; dense40 is explicit opt-in.
Only a cached image from that EXACT recorded frame is a label. No nearest
image, duplicated episode tail, synthetic image, or future input memory.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .v21_horizon_training import horizon_metadata

SCENE_QUERY_INDICES = np.arange(4, 40, 5, dtype=np.int64)
SCENE_SUPERVISION_MODES = ('sparse8', 'dense40')


def scene_query_indices(supervision='sparse8'):
    if supervision not in SCENE_SUPERVISION_MODES:
        raise ValueError('unknown scene supervision policy')
    return (SCENE_QUERY_INDICES.copy() if supervision == 'sparse8'
            else np.arange(40, dtype=np.int64))


@dataclass(frozen=True)
class SceneTargets:
    features: torch.Tensor  # [B,T,2,256,384], original frozen observed features
    valid: torch.Tensor  # [B,T,2]
    query_indices: torch.Tensor
    sigreg_query_indices: torch.Tensor | None = None
    # [B,T,2,6] recorded joint angles at each target exposure's own capture
    # time; a loss label only (zero where invalid), never a model input.
    capture_q: torch.Tensor | None = None

    def regularization_slots(self):
        """Dense MSE must not silently expand the original SIGReg time slots."""
        queries = self.query_indices
        if self.sigreg_query_indices is None:
            return torch.arange(len(queries), device=queries.device)
        wanted = self.sigreg_query_indices.to(queries)
        slots = torch.searchsorted(queries, wanted)
        if ((slots >= len(queries)).any() or
                not torch.equal(queries[slots], wanted)):
            raise ValueError('SIGReg query missing from visual targets')
        return slots


class FutureSceneTargets:
    def __init__(self, source, feature_bank=None, *, supervision='sparse8'):
        self.source = source
        self.supervision = supervision
        self.query_indices = scene_query_indices(supervision)
        self.feature_source = source if feature_bank is None else feature_bank._features
        episodes = np.asarray(source.archive.endpoint_episode_index)
        frames = source.offsets[episodes] + source.archive.endpoint_local_frame_index
        order = np.argsort(frames)
        self.frames, self.endpoint_ids = frames[order], order
        if (np.diff(self.frames) <= 0).any():
            raise ValueError('future scene endpoints must have unique episode/frame identities')
        self.archive_cached = np.sort(np.concatenate([cache.endpoint_index for cache in source.archive_dino]))

    @staticmethod
    def _contains(sorted_ids, wanted):
        if not len(sorted_ids):
            return np.zeros(wanted.shape, dtype=bool)
        positions = np.searchsorted(sorted_ids, wanted)
        return (positions < len(sorted_ids)) & (sorted_ids[np.minimum(positions, len(sorted_ids)-1)] == wanted)

    def identities(self, rows, kind, h):
        begin, end, episodes, fps, grid, times, readback_valid = horizon_metadata(self.source, rows, kind, h)
        frames = begin[:, None] + grid[:, self.query_indices + 1]
        valid = readback_valid[:, self.query_indices].copy()
        ids = np.full(frames.shape, -1, dtype=np.int64)
        if kind == 'fish':
            valid &= self._contains(self.source.fish_dino.endpoint_index, frames)
            admitted = frames[valid]
            self.source.fish.validate_frames(admitted)
            if not np.array_equal(self.source.fish.episodes[admitted],
                                  np.broadcast_to(episodes[:, None], frames.shape)[valid]):
                raise ValueError('future fish scene crossed episode')
            ids[valid] = admitted
        else:
            locations = np.searchsorted(self.frames, frames)
            candidates = self.endpoint_ids[np.minimum(locations, len(self.frames)-1)]
            valid &= self._contains(self.frames, frames)
            valid &= self._contains(self.archive_cached, candidates)
            ids[valid] = candidates[valid]
            selected_episodes = self.source.archive.endpoint_episode_index[ids[valid]]
            if not np.array_equal(selected_episodes, np.broadcast_to(episodes[:, None], frames.shape)[valid]):
                raise ValueError('future archive scene crossed episode/split')
        return ids

    def load(self, rows, kind, h, device):
        ids = self.identities(rows, kind, h)
        available = ids >= 0
        unique, inverse = np.unique(ids[available], return_inverse=True)
        features = torch.zeros(*ids.shape, 2, 256, 384, device=device, dtype=torch.float32)
        valid = torch.zeros(*ids.shape, 2, device=device, dtype=torch.bool)
        if len(unique):
            loader = self.feature_source
            if kind == 'fish':
                self.source.fish.validate_frames(unique)
                tokens, checked = loader._feature_values(self.source.fish_dino, unique)
            else:
                tokens, checked = loader._get_archive_dino(unique)
            tokens = torch.as_tensor(tokens, device=device, dtype=torch.float32)
            index = torch.as_tensor(inverse, device=device)
            mask = torch.as_tensor(available, device=device)
            features[mask] = tokens[index]
            valid[mask] = torch.as_tensor(np.asarray(checked), device=device, dtype=torch.bool)[index]
        clock = getattr(self.source, 'recorded_clock', None)
        capture_q = (torch.zeros(*ids.shape, 2, 6, device=device, dtype=torch.float32)
                     if clock is not None else None)
        if clock is not None and len(unique):
            begin = horizon_metadata(self.source, rows, kind, h)[0]
            now = np.asarray(clock.times(kind)[begin])
            _, capture, recorded_q = clock.observations(unique, kind)
            # A retained old exposure is not a future supervision target.
            causal_future = capture[inverse] > np.broadcast_to(now[:, None, None], (*ids.shape, 2))[available]
            valid[mask] &= torch.as_tensor(causal_future, device=device)
            recorded_q = torch.as_tensor(np.asarray(recorded_q, np.float32), device=device)
            if recorded_q.shape != (len(unique), 2, 6):
                raise ValueError('recorded capture q must be [exposures,2,6]')
            capture_q[mask] = torch.where(valid[mask][..., None], recorded_q[index], 0.)
        return SceneTargets(features, valid, torch.as_tensor(self.query_indices, device=device),
            torch.as_tensor(SCENE_QUERY_INDICES, device=device) if self.supervision == 'dense40' else None,
            capture_q)
