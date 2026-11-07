"""Bounded read-only tensor-row cache for V20.1 frozen endpoint data.

Rows are packed in contiguous batch blocks, never individual CUDA allocations.
One cache shares a byte budget across DINO/observation-field namespaces.  The
loader sees only unique misses; callers always receive their original order,
duplicates, shape, dtype and values. No sampling or model math changes here.

This is a single-owner training-process utility, not a thread-safe datastore.
Source data must be immutable for a namespace's lifetime. Use a distinct
namespace or clear it when a frozen encoder/checkpoint changes. Trainable
tensor graphs are explicitly rejected, rather than silently detached/cached.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Hashable

import torch

GIB = 1024**3
MIN_WORKSPACE_BYTES = 24 * GIB


def minimum_workspace_bytes(total_bytes=None):
    """The workspace a cache must leave free: 24 GiB, or half of a smaller card (section 17.65 of
    docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md: on the 32 GiB A100 a 24 GiB reserve left the
    requested feature bank almost empty, while the trainers' own peak was about 10 GiB)."""
    return MIN_WORKSPACE_BYTES if total_bytes is None else min(MIN_WORKSPACE_BYTES, int(total_bytes) // 2)


def adaptive_cache_budget(max_bytes, resident_bytes, free_bytes, *,
                          workspace_reserve_bytes=MIN_WORKSPACE_BYTES,
                          reclaimable_bytes=0, total_bytes=None):
    """Cache capacity consistent with a live shared-device free-memory sample.

    ``free_bytes`` excludes our current residents. PyTorch's unallocated
    reserved pool is optionally reusable; active tensors are never counted as
    reclaimable. At least 24 GiB (half of a smaller device, ``total_bytes``) is
    left outside the cache for workspace.
    The sample is not a reservation against another process allocating later.
    """
    values = (max_bytes, resident_bytes, free_bytes, workspace_reserve_bytes, reclaimable_bytes)
    if any(int(value) < 0 for value in values):
        raise ValueError("memory byte counts cannot be negative")
    if workspace_reserve_bytes < minimum_workspace_bytes(total_bytes):
        raise ValueError("cache must retain at least 24 GiB (half of a smaller device) activation workspace")
    available = int(free_bytes) + int(reclaimable_bytes) - int(workspace_reserve_bytes)
    return min(int(max_bytes), max(0, int(resident_bytes) + available))


@dataclass
class _TensorBlock:
    namespace: Hashable
    row_ids: tuple[int, ...]
    tensor: torch.Tensor
    byte_count: int


class TensorRowBlockCache:
    """LRU tensor batch-block cache with adaptive shared-GPU capacity.

    ``get_rows(namespace, row_ids, loader)`` calls ``loader(list_of_unique_ids)``
    exactly once for misses. The loader returns a Tensor or NumPy-compatible
    array in that order. A CUDA input must already belong to the target GPU;
    the utility never silently copies another process/device's cache.

    ``memory_info`` is an optional zero-argument ``(free_bytes,total_bytes)``
    provider, also enabling deterministic CPU tests of shared-device pressure.
    Without it, CUDA uses torch.cuda.mem_get_info and CPU uses max_bytes alone.
    """
    def __init__(self, device, max_bytes, *, block_rows=128,
                 workspace_reserve_bytes=None, pin_memory=True,
                 memory_info: Callable | None = None):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("row cache supports explicit CPU or CUDA devices")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA row cache requested but CUDA is unavailable")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
        if int(max_bytes) < 0 or int(block_rows) < 1:
            raise ValueError("cache max_bytes must be non-negative and block_rows positive")
        # the device's size sets the smallest workspace (24 GiB, or half of a smaller card); unset = that minimum
        self.total_bytes = (int(memory_info()[1]) if memory_info is not None
                            else torch.cuda.get_device_properties(self.device).total_memory if self.device.type == "cuda"
                            else None)
        minimum = minimum_workspace_bytes(self.total_bytes)
        if workspace_reserve_bytes is None:
            workspace_reserve_bytes = minimum
        if int(workspace_reserve_bytes) < minimum:
            raise ValueError("cache must retain at least 24 GiB (half of a smaller device) activation workspace")
        self.max_bytes = int(max_bytes)
        self.block_rows = int(block_rows)
        self.workspace_reserve_bytes = int(workspace_reserve_bytes)
        self.pin_memory = bool(pin_memory)
        self._memory_info = memory_info
        self._blocks = OrderedDict()
        self._index = {}
        self._schemas = {}
        self._next_block = 0
        self._resident_bytes = 0
        self._budget_bytes = self.max_bytes
        self._counts = dict(row_hits=0, row_misses=0, hit_bytes=0, miss_bytes=0,
                            loader_calls=0, upload_bytes=0, blocks_inserted=0,
                            rows_inserted=0, evictions=0, evicted_bytes=0,
                            capacity_bypass_rows=0, oom_bypass_rows=0,
                            peak_resident_bytes=0)

    @staticmethod
    def _ids(row_ids):
        if isinstance(row_ids, torch.Tensor):
            if row_ids.device.type != "cpu" or row_ids.ndim != 1 or row_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError("row_ids tensor must be one-dimensional CPU integers")
            row_ids = row_ids.tolist()
        # ndarray.tolist avoids converting each numpy scalar with tensor ops.
        elif hasattr(row_ids, "tolist"):
            if getattr(row_ids, "ndim", 1) != 1:
                raise ValueError("row_ids must be one-dimensional")
            row_ids = row_ids.tolist()
        ids = []
        for value in row_ids:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("row IDs must be non-negative integers")
            ids.append(value)
        return ids

    def _value(self, namespace, value, rows):
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.requires_grad or tensor.grad_fn is not None:
            raise ValueError("cannot cache tensors with training gradients")
        if tensor.layout != torch.strided or tensor.ndim < 1 or len(tensor) != rows:
            raise ValueError("loader must return a dense tensor with one row per requested ID")
        if tensor.device.type not in ("cpu", self.device.type) or (tensor.device.type == "cuda" and tensor.device != self.device):
            raise ValueError("loader tensor must be on CPU or the selected cache GPU")
        schema = (tuple(tensor.shape[1:]), tensor.dtype)
        previous = self._schemas.get(namespace)
        if previous is not None and previous != schema:
            raise ValueError("namespace row shape/dtype changed; use a new namespace or clear")
        self._schemas[namespace] = schema
        return tensor

    def _current_budget(self):
        if self._memory_info is None and self.device.type == "cpu":
            self._budget_bytes = self.max_bytes
            return self._budget_bytes
        free, total = (self._memory_info() if self._memory_info is not None else torch.cuda.mem_get_info(self.device))
        if int(free) < 0 or int(total) < 0 or int(free) > int(total):
            raise ValueError("memory_info returned invalid free/total byte counts")
        reclaimable = 0
        if self.device.type == "cuda":
            reclaimable = max(0, torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device))
        self._budget_bytes = adaptive_cache_budget(
            self.max_bytes, self._resident_bytes, free,
            workspace_reserve_bytes=self.workspace_reserve_bytes,
            reclaimable_bytes=reclaimable, total_bytes=total)
        return self._budget_bytes

    def _remove_block(self, block_id, *, eviction=True):
        block = self._blocks.pop(block_id)
        for offset, row_id in enumerate(block.row_ids):
            key = (block.namespace, row_id)
            if self._index.get(key) == (block_id, offset):
                del self._index[key]
        self._resident_bytes -= block.byte_count
        if eviction:
            self._counts["evictions"] += 1
            self._counts["evicted_bytes"] += block.byte_count

    def trim(self, target_bytes=None):
        """Evict least-recently used blocks after a live capacity refresh."""
        budget = self._current_budget()
        if target_bytes is not None:
            if int(target_bytes) < 0:
                raise ValueError("trim target cannot be negative")
            budget = min(budget, int(target_bytes))
        before = self._resident_bytes
        while self._blocks and self._resident_bytes > budget:
            self._remove_block(next(iter(self._blocks)))
        return before - self._resident_bytes

    def clear(self, namespace=None, *, release_allocator=False, reset_stats=False):
        """Drop only this cache's tensor references, optionally one namespace.

        CUDA allocator release is opt-in; empty_cache does not free live model
        or other process tensors, and is not run on normal LRU evictions.
        """
        before = self._resident_bytes
        selected = [key for key, block in self._blocks.items()
                    if namespace is None or block.namespace == namespace]
        for key in selected:
            self._remove_block(key, eviction=False)
        if namespace is None:
            self._schemas.clear()
        else:
            self._schemas.pop(namespace, None)
        if reset_stats:
            for key in self._counts:
                self._counts[key] = 0
            self._counts["peak_resident_bytes"] = self._resident_bytes
        if release_allocator and self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        return before - self._resident_bytes

    def _upload(self, tensor):
        if tensor.device == self.device:
            return tensor
        if self.pin_memory and self.device.type == "cuda" and not tensor.is_pinned():
            tensor = tensor.pin_memory()
        output = tensor.to(device=self.device, non_blocking=self.device.type == "cuda")
        self._counts["upload_bytes"] += tensor.numel() * tensor.element_size()
        return output

    def put_rows(self, namespace, row_ids, value):
        """Copy immutable rows into batch blocks; return number retained.

        Existing identical keys are left unchanged: this is an immutable cache,
        not a write-through store. No full old block clone occurs for repeats.
        The caller must clear the namespace when the underlying values change.
        """
        ids = self._ids(row_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("put_rows expects unique row IDs")
        tensor = self._value(namespace, value, len(ids))
        if not ids:
            return 0
        self.trim()
        positions = [position for position, row in enumerate(ids) if (namespace, row) not in self._index]
        start = 0
        row_bytes = tensor[0].numel() * tensor.element_size()
        if row_bytes == 0:
            self._counts["capacity_bypass_rows"] += len(positions)
            return 0
        while start < len(positions):
            budget = self._current_budget()
            if row_bytes > budget:
                self._counts["capacity_bypass_rows"] += len(positions) - start
                break
            count = min(self.block_rows, budget // row_bytes)
            selection = positions[start:start + count]
            byte_count = len(selection) * row_bytes
            while self._blocks and self._resident_bytes + byte_count > budget:
                self._remove_block(next(iter(self._blocks)))
            try:
                index = torch.tensor(selection, dtype=torch.int64, device=tensor.device)
                # One independent contiguous allocation per multi-row block.
                # index_select owns the storage even for already-device values.
                packed = tensor.index_select(0, index)
                packed = self._upload(packed)
            except torch.OutOfMemoryError:
                # Cache allocation is optional; values/sampling remain intact.
                schemas = dict(self._schemas)
                self.clear(release_allocator=True)
                self._schemas = schemas
                self._counts["oom_bypass_rows"] += len(positions) - start
                return 0
            rows = tuple(ids[position] for position in selection)
            block_id = self._next_block
            self._next_block += 1
            self._blocks[block_id] = _TensorBlock(namespace, rows, packed, byte_count)
            for offset, row_id in enumerate(rows):
                self._index[(namespace, row_id)] = (block_id, offset)
            self._resident_bytes += byte_count
            self._counts["rows_inserted"] += len(rows)
            self._counts["blocks_inserted"] += 1
            self._counts["peak_resident_bytes"] = max(self._counts["peak_resident_bytes"], self._resident_bytes)
            start += len(selection)
        return sum((namespace, ids[position]) in self._index for position in positions)

    def get_rows(self, namespace, row_ids, loader):
        """Return a new tensor in exact requested order, including duplicates."""
        ids = self._ids(row_ids)
        self.trim()
        hit_groups = {}
        misses, missing_index = [], {}
        # Keep references until output gather completes if cache pressure
        # evicts a referenced block later in the same call.
        for destination, row_id in enumerate(ids):
            cached = self._index.get((namespace, row_id))
            if cached is None:
                self._counts["row_misses"] += 1
                if row_id not in missing_index:
                    missing_index[row_id] = len(misses)
                    misses.append(row_id)
            else:
                self._counts["row_hits"] += 1
                block_id, offset = cached
                block = self._blocks[block_id]
                self._blocks.move_to_end(block_id)
                group = hit_groups.setdefault(block_id, (block.tensor, [], []))
                group[1].append(destination)
                group[2].append(offset)
        loaded = None
        if misses:
            self._counts["loader_calls"] += 1
            loaded = self._value(namespace, loader(misses), len(misses))
        schema = self._schemas.get(namespace)
        if schema is None:
            raise ValueError("empty lookup requires an existing namespace shape/dtype")
        shape, dtype = schema
        output = torch.empty((len(ids), *shape), dtype=dtype, device=self.device)
        row_bytes = output[0].numel() * output.element_size() if ids else 0
        with torch.no_grad():
            for block, destinations, offsets in hit_groups.values():
                source_index = torch.tensor(offsets, dtype=torch.int64, device=self.device)
                destination_index = torch.tensor(destinations, dtype=torch.int64, device=self.device)
                output.index_copy_(0, destination_index, block.index_select(0, source_index))
                self._counts["hit_bytes"] += len(destinations) * row_bytes
            if loaded is not None:
                uploaded = self._upload(loaded)
                destinations = [position for position, row in enumerate(ids) if row in missing_index]
                offsets = [missing_index[ids[position]] for position in destinations]
                output.index_copy_(0, torch.tensor(destinations, dtype=torch.int64, device=self.device),
                                    uploaded.index_select(0, torch.tensor(offsets, dtype=torch.int64, device=self.device)))
                self._counts["miss_bytes"] += len(destinations) * row_bytes
                self.put_rows(namespace, misses, uploaded)
        return output

    def stats(self):
        total = self._counts["row_hits"] + self._counts["row_misses"]
        return {**self._counts, "resident_bytes": self._resident_bytes,
                "budget_bytes": self._budget_bytes, "max_bytes": self.max_bytes,
                "workspace_reserve_bytes": self.workspace_reserve_bytes,
                "resident_blocks": len(self._blocks), "resident_rows": len(self._index),
                "namespaces": len(self._schemas), "hit_rate": self._counts["row_hits"] / total if total else 0.,
                "device": str(self.device), "block_rows": self.block_rows}
