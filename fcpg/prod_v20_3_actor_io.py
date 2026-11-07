"""Bounded, immutable endpoint windows for Actor training.

Storage reads are sorted/coalesced, model gathers restore requested chain order.
DINO/pooling stay fp16 until the original model-facing fp32 conversion. The
worker pool only touches CPU arrays; CUDA and the LRU have one owning thread.
No new cache files, image reconstruction, point substitution or quantization.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

import numpy as np
import torch

from .prod_v20_1_cache import SHAPES

GIB = 2**30
ENDPOINT_BYTES = 2*256*384*2 + 2 + sum(
    int(np.prod(shape))*(2 if name == "pooling_map" else 8)
    for name, shape in SHAPES.items()) + 8


@dataclass
class HostEndpointPool:
    endpoint_ids: np.ndarray
    arrays: dict
    load_seconds: float = 0.
    read_bytes: int = 0

    @property
    def nbytes(self):
        return int(self.endpoint_ids.nbytes + sum(value.nbytes for value in self.arrays.values()))


def _ids(value, count=None):
    raw = np.asarray(value)
    if raw.ndim != 1 or not len(raw) or raw.dtype.kind not in "iu":
        raise ValueError("endpoint/row indices must be nonempty integer vectors")
    ids = raw.astype(np.int64, copy=False)
    if (ids < 0).any() or (count is not None and (ids >= count).any()):
        raise ValueError("endpoint/row index out of range")
    return ids


def _copy_sorted(source, rows, target, positions):
    """A bounded CPU job. Adjacent disk rows become one slice read.

    Destination positions need not be contiguous (train/dev interleaving).
    Validate in <=128-row buffers, not a dataset-sized boolean temporary.
    """
    cuts = np.r_[0, np.flatnonzero(np.diff(rows) != 1)+1, len(rows)]
    for left, right in zip(cuts[:-1], cuts[1:]):
        for begin in range(int(left), int(right), 128):
            end = min(begin+128, int(right))
            value = np.asarray(source[int(rows[begin]):int(rows[end-1])+1])
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError("nonfinite cached Actor endpoint field")
            target[positions[begin:end]] = value


class ActorEndpointReader:
    """Read actual H0-H3 fields once per mixed-locality window, not per batch."""
    def __init__(self, sources, bank, *, workers=4, max_host_bytes=16*GIB):
        if workers < 1 or max_host_bytes < 1:
            raise ValueError("positive reader workers/host budget required")
        self.sources, self.bank = sources, bank
        self.workers, self.max_host_bytes = int(workers), int(max_host_bytes)

    def endpoint_ids(self, rows, kind="archive"):
        if kind not in ("archive", "dense"):
            raise ValueError("window reader accepts archive/dense only, never held fish")
        count = self.sources.counts[kind]
        rows = _ids(rows, count)
        selected = rows if kind == "archive" else np.asarray(self.sources.dense_rows)[rows]
        return np.unique(self.sources.archive.row_endpoint_index[selected, :4].reshape(-1))

    def load(self, rows, kind="archive"):
        started = time.monotonic()
        ids = self.endpoint_ids(rows, kind)
        if len(ids)*ENDPOINT_BYTES > self.max_host_bytes:
            raise MemoryError("Actor host window exceeds explicit budget; split on batch boundaries")
        if not np.asarray(self.bank.written[ids]).all():
            raise RuntimeError("raw RGB observation missing; no feature-only substitution")
        # Preserve the bank's recorded precision exactly. Production banks
        # require the V20.3 fp64 binding at open; standalone old-data timing
        # may still read a validated legacy fp32 bank, never relabel it.
        arrays = {name: np.empty((len(ids), *shape), dtype=self.bank.arrays[name].dtype)
                  for name, shape in SHAPES.items()}
        arrays["dino"] = np.empty((len(ids), 2, 256, 384), np.float16)
        arrays["view_valid"] = np.empty((len(ids), 2), bool)
        jobs = []
        for name in SHAPES:
            if self.bank.arrays[name].dtype != arrays[name].dtype:
                raise ValueError("observation cache precision differs from the frozen contract")
            jobs.append((self.bank.arrays[name], ids, arrays[name], np.arange(len(ids))))
        found = np.zeros(len(ids), bool)
        for cache in self.sources.archive_dino:
            positions = np.searchsorted(cache.endpoint_index, ids)
            matches = (positions < len(cache.endpoint_index))
            matches &= cache.endpoint_index[np.minimum(positions, len(cache.endpoint_index)-1)] == ids
            matches &= ~found
            dst = np.flatnonzero(matches)
            if not len(dst):
                continue
            if cache.tokens.dtype != np.float16:
                raise ValueError("DINO cache must remain fp16, no new quantization")
            src = positions[dst]
            arrays["view_valid"][dst] = cache.view_valid[src]
            jobs.append((cache.tokens, src, arrays["dino"], dst))
            found[dst] = True
        if not found.all():
            raise ValueError("missing frozen DINO endpoint")
        expected = self.sources.archive.endpoint_view_valid[ids]
        if not np.array_equal(arrays["view_valid"], expected):
            raise ValueError("DINO/source endpoint view validity disagrees")
        # Each job owns disjoint target rows/fields; no CUDA, RNG or mutable
        # model/source cache is accessed by the workers.
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="actor-disk") as pool:
            futures = []
            for source, src, target, dst in jobs:
                for begin in range(0, len(src), 512):
                    futures.append(pool.submit(_copy_sorted, source, src[begin:begin+512],
                                               target, dst[begin:begin+512]))
            for future in futures:
                future.result()
        for begin in range(0, len(ids), 128):
            valid = arrays["view_valid"][begin:begin+128]
            if np.any(arrays["dino"][begin:begin+128][~valid] != 0):
                raise ValueError("invalid DINO view has nonzero features")
        result = HostEndpointPool(ids, arrays, time.monotonic()-started)
        result.read_bytes = result.nbytes
        return result


class ActorWindowLRU:
    """One contiguous GPU tensor per field/window, not thousands of tiny blocks.

    Active CPU data are retained for deterministic OOM replay. Eviction or a
    failed optional upload never consumes the iterator, alters data, or drops
    a batch. At most the current and one prefetched host pool exist externally.
    """
    def __init__(self, device, max_bytes, *, workspace_bytes=24*GIB):
        self.device = torch.device(device)
        if max_bytes < 0 or workspace_bytes < 0:
            raise ValueError("cache/workspace budgets must be nonnegative")
        self.max_bytes, self.workspace_bytes = int(max_bytes), int(workspace_bytes)
        self.blocks = OrderedDict()
        self.resident_bytes = 0
        self.current_key = self.host = None
        self.upload_bytes = self.evictions = self.oom_bypasses = self.requests = 0
        self.window_loads = self.endpoint_reads = 0
        self.read_wall_s = 0.
        self.row_hits = self.row_misses = 0

    def _budget(self):
        if self.device.type != "cuda":
            return self.max_bytes
        free, _ = torch.cuda.mem_get_info(self.device)
        unused = torch.cuda.memory_reserved(self.device)-torch.cuda.memory_allocated(self.device)
        return max(0, min(self.max_bytes, int(free)+unused+self.resident_bytes-self.workspace_bytes))

    def _evict(self):
        _, (_, _, size) = self.blocks.popitem(last=False)
        self.resident_bytes -= size
        self.evictions += 1

    def activate(self, key, host):
        ids = _ids(host.endpoint_ids)
        if (np.diff(ids) <= 0).any() or any(len(a) != len(ids) for a in host.arrays.values()):
            raise ValueError("resident window needs sorted unique endpoints and aligned arrays")
        self.current_key, self.host = key, host
        if key in self.blocks:
            if not np.array_equal(self.blocks[key][0], ids):
                raise ValueError("LRU window key rebound to different endpoints")
            self.blocks.move_to_end(key)
            return
        self.window_loads += 1
        self.endpoint_reads += len(ids)
        self.read_wall_s += host.load_seconds
        size = sum(a.nbytes for a in host.arrays.values())
        budget = self._budget()
        while self.blocks and self.resident_bytes+size > budget:
            self._evict()
        if size > budget:
            self.oom_bypasses += 1
            return  # exact bounded per-request staging, not data truncation
        tensors = {}
        try:
            for name, array in host.arrays.items():
                if self.device.type == "cpu":
                    tensors[name] = torch.from_numpy(array).clone()
                    continue
                tensors[name] = torch.empty(array.shape, dtype=torch.from_numpy(array[:1]).dtype, device=self.device)
                # Do not pin an entire 13 GiB host pool. Reusable allocator
                # buffers and periodic stream waits bound outstanding pins.
                for begin in range(0, len(ids), 128):
                    pinned = torch.from_numpy(array[begin:begin+128]).pin_memory()
                    tensors[name][begin:begin+128].copy_(pinned, non_blocking=True)
                    if begin % 1024 == 0:
                        torch.cuda.current_stream(self.device).synchronize()
                torch.cuda.current_stream(self.device).synchronize()
            self.blocks[key] = (ids.copy(), tensors, size)
            self.resident_bytes += size
            self.upload_bytes += size
        except torch.cuda.OutOfMemoryError:
            tensors.clear()
            self.clear()
            torch.cuda.empty_cache()
            self.oom_bypasses += 1

    def get_arrays(self, endpoint_ids):
        if self.host is None:
            raise RuntimeError("no active Actor window")
        requested = _ids(endpoint_ids)
        positions = np.searchsorted(self.host.endpoint_ids, requested)
        if ((positions >= len(self.host.endpoint_ids)).any() or
                not np.array_equal(self.host.endpoint_ids[np.minimum(positions, len(self.host.endpoint_ids)-1)], requested)):
            raise ValueError("requested endpoint outside active actual-data window")
        self.requests += 1
        if self.current_key in self.blocks:
            self.blocks.move_to_end(self.current_key)
            self.row_hits += len(requested)
            index = torch.as_tensor(positions, device=self.device)
            return {name: value.index_select(0, index) for name, value in self.blocks[self.current_key][1].items()}
        self.row_misses += len(requested)
        return {name: torch.as_tensor(np.array(value[positions], copy=True), device=self.device)
                for name, value in self.host.arrays.items()}

    def clear(self):
        self.blocks.clear()
        self.resident_bytes = 0
        # Retain the active immutable CPU pool: OOM replays may request it.

    def close(self):
        self.clear()
        self.host = self.current_key = None

    def stats(self):
        count = self.row_hits+self.row_misses
        return dict(kind="mixed-locality-endpoint-window-lru-v1", resident_bytes=self.resident_bytes,
                    max_bytes=self.max_bytes, budget_bytes=self.max_bytes, resident_blocks=len(self.blocks),
                    workspace_reserve_bytes=self.workspace_bytes, upload_bytes=self.upload_bytes,
                    evictions=self.evictions, oom_bypass_windows=self.oom_bypasses, requests=self.requests,
                    window_loads=self.window_loads, unique_endpoints_read=self.endpoint_reads,
                    read_wall_s=self.read_wall_s, row_hits=self.row_hits, row_misses=self.row_misses,
                    hit_rate=self.row_hits/count if count else 0.,
                    active_host_bytes=self.host.nbytes if self.host is not None else 0)


def observation_from_arrays(values, observer):
    """Same frozen current repooling as RawObservationCache.get."""
    from .prod_v20_1_belief import JointPairBelief
    from .prod_v20_1_observer import ObserverOutput
    scene = values["dino"].float()
    valid = values["view_valid"].bool()
    pooling = values["pooling_map"].float()
    belief = JointPairBelief(**{name: values[name] for name in SHAPES if name != "pooling_map"})
    belief.validate()
    if not torch.isfinite(scene).all() or not torch.isfinite(pooling).all():
        raise ValueError("nonfinite staged current visual field")
    observation = ObserverOutput(belief, scene.new_zeros(len(scene),2,4,384),
                                scene.detach(), scene.new_empty(len(scene),2,4,0), pooling)
    return observer.repool(observation, belief, valid)
