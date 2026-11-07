"""Bounded disk/CPU/GPU cache policy for temporal FCPG-H."""

from __future__ import annotations

import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Hashable, Mapping, TypeVar

import numpy as np
import torch

from fcpg.offline_token_cache import DeviceBatchLRU
from fcpg.temporal_grounding_protocol import (
    CPU_DECODE_LRU_BYTES,
    DATA_DISK_RESERVE_BYTES,
    DEFAULT_GPU_PATCH_LRU_BYTES,
    GPU_RESERVE_BYTES,
    GROUNDED_TOKEN_CACHE_BYTES,
)


T = TypeVar("T")
HOST_MEMORY_RESERVE_BYTES = 16 * 1024**3
GPU_ACTIVE_WORKSPACE_BYTES = 4 * 1024**3


def _linux_mem_available_bytes() -> int | None:
    """Read Linux MemAvailable without adding a runtime dependency."""

    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def safe_temporal_host_lru_bytes(
    requested_bytes: int = CPU_DECODE_LRU_BYTES,
    *,
    available_bytes: int | None = None,
    reserve_bytes: int = HOST_MEMORY_RESERVE_BYTES,
) -> int:
    """Bound the nominal 96 GiB host LRU to the machine actually running it."""

    if requested_bytes <= 0:
        raise ValueError("host LRU request must be positive")
    if reserve_bytes < 0:
        raise ValueError("host memory reserve cannot be negative")
    available = _linux_mem_available_bytes() if available_bytes is None else int(available_bytes)
    if available is None:
        return int(requested_bytes)
    return min(int(requested_bytes), max(1, available - int(reserve_bytes)))


def safe_temporal_gpu_lru_bytes(
    device: torch.device,
    requested_bytes: int = DEFAULT_GPU_PATCH_LRU_BYTES,
) -> int:
    """Compute the LRU budget after model/optimizer allocation.

    The latest protocol keeps only an 8 GiB free-device reserve.  Callers must
    invoke this after all persistent model and optimizer tensors are resident.
    """

    target = torch.device(device)
    if requested_bytes < 0:
        raise ValueError("GPU LRU request cannot be negative")
    if requested_bytes == 0 or target.type != "cuda":
        return 0
    free_bytes, _ = torch.cuda.mem_get_info(target)
    return min(int(requested_bytes), max(0, int(free_bytes) - GPU_RESERVE_BYTES))


def assert_disk_budget(
    root: Path,
    *,
    planned_bytes: int,
    cache_limit_bytes: int = GROUNDED_TOKEN_CACHE_BYTES,
    reserve_bytes: int = DATA_DISK_RESERVE_BYTES,
) -> None:
    if planned_bytes <= 0:
        raise ValueError("planned cache bytes must be positive")
    if planned_bytes > cache_limit_bytes:
        raise ValueError("grounded-token cache exceeds the 120 GiB protocol limit")
    free = shutil.disk_usage(root).free
    if free - planned_bytes < reserve_bytes:
        raise OSError("cache plan would violate the 128 GiB /data reserve")


def capped_window_count(
    *,
    available_windows: int,
    bytes_per_window: int,
    cache_limit_bytes: int = GROUNDED_TOKEN_CACHE_BYTES,
) -> int:
    if available_windows < 0 or bytes_per_window <= 0:
        raise ValueError("window count must be non-negative and row bytes positive")
    return min(available_windows, cache_limit_bytes // bytes_per_window)


@dataclass(frozen=True)
class HostCacheStats:
    requested_bytes: int
    max_bytes: int
    memory_reserve_bytes: int
    resident_bytes: int
    peak_bytes: int
    entries: int
    hits: int
    misses: int
    evictions: int
    pressure_evictions: int


class HostByteLRU(Generic[T]):
    """A byte-accounted 96 GiB host LRU for decoded/resized frame blocks."""

    def __init__(self, *, max_bytes: int | None = None) -> None:
        requested = CPU_DECODE_LRU_BYTES if max_bytes is None else int(max_bytes)
        if requested <= 0:
            raise ValueError("host LRU max_bytes must be positive")
        self.requested_bytes = requested
        if max_bytes is None:
            self.max_bytes = safe_temporal_host_lru_bytes(requested)
            self.memory_reserve_bytes = HOST_MEMORY_RESERVE_BYTES
        else:
            # Explicit small test/application budgets remain exact.  The
            # formal pipeline uses the dynamically bounded default.
            self.max_bytes = requested
            self.memory_reserve_bytes = 0
        self._entries: OrderedDict[Hashable, tuple[T, int]] = OrderedDict()
        self.current_bytes = 0
        self.peak_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.pressure_evictions = 0
        self._puts_since_pressure_check = 0

    def get(self, key: Hashable) -> T | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry[0]

    def put(self, key: Hashable, value: T, *, bytes_used: int) -> bool:
        if bytes_used <= 0:
            raise ValueError("cache entry bytes must be positive")
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.current_bytes -= previous[1]
        if bytes_used > self.max_bytes:
            return False
        while self._entries and self.current_bytes + bytes_used > self.max_bytes:
            _, (_, removed_bytes) = self._entries.popitem(last=False)
            self.current_bytes -= removed_bytes
            self.evictions += 1
        self._entries[key] = (value, int(bytes_used))
        self.current_bytes += int(bytes_used)
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        self._puts_since_pressure_check += 1
        if self.memory_reserve_bytes and self._puts_since_pressure_check >= 128:
            self.enforce_memory_reserve()
            self._puts_since_pressure_check = 0
        return True

    def enforce_memory_reserve(self, *, available_bytes: int | None = None) -> int:
        """Yield host rows when concurrent GPU jobs pressure shared RAM."""

        if not self.memory_reserve_bytes:
            return 0
        available = (
            _linux_mem_available_bytes() if available_bytes is None else int(available_bytes)
        )
        if available is None or available >= self.memory_reserve_bytes:
            return 0
        target = self.memory_reserve_bytes - available
        released = 0
        removed = 0
        while self._entries and released < target:
            _, (_, row_bytes) = self._entries.popitem(last=False)
            self.current_bytes -= row_bytes
            released += row_bytes
            removed += 1
        self.evictions += removed
        self.pressure_evictions += removed
        return released

    def stats(self) -> HostCacheStats:
        return HostCacheStats(
            requested_bytes=self.requested_bytes,
            max_bytes=self.max_bytes,
            memory_reserve_bytes=self.memory_reserve_bytes,
            resident_bytes=self.current_bytes,
            peak_bytes=self.peak_bytes,
            entries=len(self._entries),
            hits=self.hits,
            misses=self.misses,
            evictions=self.evictions,
            pressure_evictions=self.pressure_evictions,
        )


class DeviceTensorRowLRU:
    """On-demand GPU LRU for clean frozen-ViT patch rows.

    Rows are produced by the frozen encoder on a miss and never written to
    disk.  Corrupted images intentionally bypass this cache because their
    affine/blur realization is step-specific.
    """

    def __init__(self, *, device: torch.device, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("device tensor LRU max_bytes must be positive")
        self.device = torch.device(device)
        self.max_bytes = int(max_bytes)
        self.rows: OrderedDict[Hashable, torch.Tensor] = OrderedDict()
        self.current_bytes = 0
        self.peak_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.workspace_evictions = 0
        self.minimum_observed_free_bytes: int | None = None

    @staticmethod
    def _bytes(value: torch.Tensor) -> int:
        return value.numel() * value.element_size()

    def _insert(self, key: Hashable, value: torch.Tensor) -> None:
        # ``value`` is commonly a slice of one large encoder batch.  Keeping
        # that view would pin the whole batch storage after all but one row had
        # been evicted, so logical LRU bytes and physical CUDA bytes diverged.
        # Give every cached row its own storage; the CUDA allocator can then
        # reuse evicted blocks at the declared byte bound.
        row = (
            value.detach()
            .to(device=self.device, dtype=torch.float16)
            .contiguous()
            .clone(memory_format=torch.contiguous_format)
        )
        row_bytes = self._bytes(row)
        if row_bytes > self.max_bytes:
            return
        previous = self.rows.pop(key, None)
        if previous is not None:
            self.current_bytes -= self._bytes(previous)
        while self.rows and self.current_bytes + row_bytes > self.max_bytes:
            _, removed = self.rows.popitem(last=False)
            self.current_bytes -= self._bytes(removed)
            self.evictions += 1
        self.rows[key] = row
        self.current_bytes += row_bytes
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)

    def cached_keys(self):
        """Expose a read-only membership view without changing LRU recency/stats."""

        return self.rows.keys()

    def encode(
        self,
        keys: list[Hashable],
        images: torch.Tensor,
        encoder,
        *,
        chunk_rows: int = 256,
    ) -> torch.Tensor:
        if images.ndim != 4 or len(keys) != len(images):
            raise ValueError("keys/images must align as [N,3,H,W]")
        resolved: dict[Hashable, torch.Tensor] = {}
        missing_keys: list[Hashable] = []
        missing_images: list[torch.Tensor] = []
        for position, key in enumerate(keys):
            row = self.rows.get(key)
            if row is None:
                if key not in resolved and key not in missing_keys:
                    self.misses += 1
                    missing_keys.append(key)
                    missing_images.append(images[position])
            else:
                self.hits += 1
                self.rows.move_to_end(key)
                resolved[key] = row
        if missing_images:
            stacked = torch.stack(missing_images).to(self.device, non_blocking=True)
            encoded_chunks = [
                encoder(stacked[start : start + chunk_rows])
                for start in range(0, len(stacked), chunk_rows)
            ]
            encoded = torch.cat(encoded_chunks).to(dtype=torch.float16)
            for missing_index, key in enumerate(missing_keys):
                row = encoded[missing_index]
                resolved[key] = row
                self._insert(key, row)
        if any(key not in resolved for key in keys):
            raise AssertionError("device tensor LRU left an unresolved row")
        return torch.stack([resolved[key] for key in keys])

    def evict_bytes(self, minimum_bytes: int, *, workspace: bool = False) -> int:
        """Evict oldest rows until at least ``minimum_bytes`` were released."""

        if minimum_bytes < 0:
            raise ValueError("eviction byte target cannot be negative")
        removed_bytes = 0
        removed_rows = 0
        while self.rows and removed_bytes < minimum_bytes:
            _, removed = self.rows.popitem(last=False)
            row_bytes = self._bytes(removed)
            self.current_bytes -= row_bytes
            removed_bytes += row_bytes
            removed_rows += 1
        self.evictions += removed_rows
        if workspace:
            self.workspace_evictions += removed_rows
        return removed_bytes

    def reserve_active_workspace(self, minimum_free_bytes: int) -> int:
        """Temporarily make room for corruption/attention while idle cache stays large."""

        if minimum_free_bytes < 0:
            raise ValueError("minimum free GPU bytes cannot be negative")
        if self.device.type != "cuda":
            return 0
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        self.minimum_observed_free_bytes = (
            int(free_bytes)
            if self.minimum_observed_free_bytes is None
            else min(self.minimum_observed_free_bytes, int(free_bytes))
        )
        needed = max(0, int(minimum_free_bytes) - int(free_bytes))
        if needed == 0:
            return 0
        removed = self.evict_bytes(needed, workspace=True)
        if removed:
            torch.cuda.empty_cache()
        free_after, _ = torch.cuda.mem_get_info(self.device)
        self.minimum_observed_free_bytes = min(
            int(self.minimum_observed_free_bytes), int(free_after)
        )
        if free_after < minimum_free_bytes and self.rows:
            raise RuntimeError("GPU LRU could not release the requested active workspace")
        return removed

    def stats(self) -> dict[str, int | float | str | bool]:
        total = self.hits + self.misses
        return {
            "enabled": True,
            "device": str(self.device),
            "max_bytes": self.max_bytes,
            "resident_bytes": self.current_bytes,
            "peak_resident_bytes": self.peak_bytes,
            "resident_rows": len(self.rows),
            "row_hits": self.hits,
            "row_misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "evictions": self.evictions,
            "workspace_evictions": self.workspace_evictions,
            "minimum_observed_free_bytes": self.minimum_observed_free_bytes,
        }


def make_temporal_device_lru(
    arrays: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    requested_bytes: int = DEFAULT_GPU_PATCH_LRU_BYTES,
) -> DeviceBatchLRU | None:
    effective = safe_temporal_gpu_lru_bytes(device, requested_bytes)
    if effective == 0:
        return None
    return DeviceBatchLRU(arrays, device=device, max_bytes=effective)
