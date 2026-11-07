"""Scratch-run cache for immutable recorded DINO rows; model inputs stay FP32.

Motion, text and their validation remain on the original CPU source. This
adapter changes feature storage and transfers only; it never caches a learned
projection or samples additional examples.
"""
from __future__ import annotations

from copy import copy

import numpy as np
import torch

from .v21_encoder import DIM, PATCHES
from .prod_v20_1_acceleration import TensorRowBlockCache
from .prod_v20_timing import FISH_COMMAND_STEPS, FISH_COMMAND_STRIDE_FRAMES


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class ScratchDinoFeatureBank:
    """Lazy row LRU with the same endpoint mapping as ``V21Sources.chain``."""

    def __init__(self, host_source, device, max_bytes,
                 workspace_reserve_bytes=None):
        _require(torch.device(host_source.device).type == 'cpu',
            'scratch feature bank requires the original CPU source')
        _require(host_source.device_cache is None,
            'scratch feature bank must not replace an existing source cache')
        _require(not getattr(host_source, 'motion_only', False),
            'scratch feature bank requires complete recorded visual sources')
        self.host_source = host_source
        self.device = torch.device(device)
        _require(self.device in (torch.device('cpu'), torch.device('cuda:0')),
            'scratch feature bank requires cpu or logical cuda:0')
        self.cache = TensorRowBlockCache(self.device, max_bytes,
            workspace_reserve_bytes=workspace_reserve_bytes)
        # The CPU loader retains its device, dtype policy and cache ownership.
        # Existing feature helpers bind to this separate shallow proxy instead.
        self._features = copy(host_source)
        self._features.device = self.device
        self._features.device_cache = self.cache

    def load(self, rows, kind='archive', *, context_frames=5, as_stored=False):
        """as_stored (17.173): the float16 values as held, which their float32 copy holds exactly; the LeWM trainer's
        memory keeps them so and its LeWM converts them where it reads them (the same values)."""
        source = self.host_source
        _require(type(context_frames) is int and 1 <= context_frames <= 5,
            'feature prefix must contain one to five actual exposures')
        _require(kind in ('archive', 'dense', 'fish'), 'unknown scratch feature source')
        rows = np.asarray(rows)
        _require(rows.ndim == 1 and rows.dtype.kind in 'iu' and len(rows) > 0
            and ((rows >= 0) & (rows < source.counts[kind])).all(),
            'scratch feature rows must be valid original integer identities')
        if kind == 'fish':
            endpoint = (np.asarray(source.fish.starts)[rows, None]
                + np.arange(5, dtype=np.int64)[None]
                * (FISH_COMMAND_STEPS * FISH_COMMAND_STRIDE_FRAMES))
            # Exclude held episodes before any feature payload access, including
            # a request that could otherwise be served entirely by cache hits.
            source.fish.validate_frames(endpoint.reshape(-1))
            expected_valid = np.ones((len(rows), 5, 2), dtype=bool)
        else:
            mapped = np.asarray(source.dense_rows)[rows] if kind == 'dense' else rows
            endpoint = np.asarray(source.archive.row_endpoint_index[mapped])
            expected_valid = np.asarray(source.archive.endpoint_view_valid[endpoint])
        if getattr(source, 'recorded_clock', None) is not None:
            expected_valid = source.recorded_clock.observations(endpoint, kind)[0]
        _require(endpoint.shape == (len(rows), 5) and endpoint.dtype.kind in 'iu'
            and (endpoint >= 0).all(), 'scratch feature chain endpoint identities differ')
        _require(expected_valid.shape == (len(rows), 5, 2) and expected_valid.dtype == np.bool_,
            'scratch feature chain view validity differs')
        endpoint = endpoint[:, :context_frames]
        expected_valid = expected_valid[:, :context_frames]
        unique, inverse = np.unique(endpoint.reshape(-1), return_inverse=True)
        if kind == 'fish':
            tokens, checked = self._features._feature_values(source.fish_dino, unique)
        else:
            tokens, checked = self._features._get_archive_dino(unique)
        _require(isinstance(tokens, torch.Tensor) and tokens.device == self.device
            and tokens.dtype == torch.float16 and tuple(tokens.shape) == (len(unique), 2, PATCHES, DIM),
            'scratch feature bank requires original FP16 DINO rows on the selected device')
        checked = np.asarray(checked)
        _require(checked.shape == (len(unique), 2) and checked.dtype == np.bool_,
            'scratch cached feature view validity differs')
        valid = checked[inverse].reshape(len(rows), context_frames, 2)
        _require(np.array_equal(valid, expected_valid),
            'DINO and current chain view identities differ')
        index = torch.as_tensor(inverse, dtype=torch.int64, device=self.device)
        features = tokens.index_select(0, index).reshape(len(rows), context_frames, 2, PATCHES, DIM)
        return (features if as_stored else features.float()), valid

    def pictures(self, kind, ids):
        """17.166 picture memories (fcpg/v21_picture_memory.py features_for): these Archive endpoints' or
        demonstration frames' rows through this bank, as load() reads a chain's (an int8 store's rows are multiplied
        on the bank's GPU, fcpg/v21_dequantize.py), as stored (float16) with their views' validity; the host
        source's own reading gives the same values on one CPU core, 4x slower (11/02, 17.170)."""
        _require(kind in ('archive', 'fish'), 'unknown picture memory source')
        ids = np.asarray(ids, np.int64)
        if kind == 'fish':
            self.host_source.fish.validate_frames(ids)
            return self._features._feature_values(self.host_source.fish_dino, ids)
        return self._features._get_archive_dino(ids)

    def clear(self, release_allocator=False):
        return self.cache.clear(release_allocator=release_allocator)

    def stats(self):
        return self.cache.stats()
