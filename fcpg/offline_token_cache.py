"""Bounded token-cache and balanced-sampling contracts for forward pipeline v1."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from fcpg.offline_pipeline_manifest import PROTOCOL_ID, sha256_file, sha256_json


CACHE_FORMAT = "so101-fcpg-lewm-token-cache-v1"
CACHE_GRID_SIZE = 7
CACHE_GRID_RADIUS_PX = 24.0
FCPG_GRID_SIZE = 5
MAX_CACHE_BYTES = 120 * 1024**3
MIN_FREE_AFTER_CACHE_BYTES = 64 * 1024**3
DEFAULT_GPU_CACHE_BYTES = 48 * 1024**3
GPU_CACHE_TOTAL_FRACTION = 0.65
GPU_CACHE_RESERVE_BYTES = 16 * 1024**3


def safe_gpu_cache_bytes(device: torch.device, requested_bytes: int) -> int:
    """Bound a requested cache while retaining activation/workspace headroom."""

    if requested_bytes < 0:
        raise ValueError("GPU cache bytes cannot be negative")
    if requested_bytes == 0 or device.type != "cuda":
        return 0
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return min(
        int(requested_bytes),
        int(total_bytes * GPU_CACHE_TOTAL_FRACTION),
        max(0, int(free_bytes) - GPU_CACHE_RESERVE_BYTES),
    )


@dataclass(frozen=True)
class ArrayContract:
    shape: tuple[int, ...]
    dtype: str
    required: bool = True


BASE_ARRAYS = {
    "global_tokens": ("float16", (2, -1)),
    "tip_local_tokens": ("float16", (2, 2, CACHE_GRID_SIZE**2, -1)),
    "q_nominal_px": ("float16", (2, 3, 2)),
    "q_pseudo_px": ("float16", (2, 3, 2)),
    "tip_confidence": ("float16", (2, 2)),
    "camera_descriptor": ("float32", (2, 15)),
    "anchor_xyz": ("float32", (3, 3)),
    "state": ("float32", (6,)),
    "source_index": ("int32", ()),
    "task_index": ("int32", ()),
    "episode_index": ("int64", ()),
    "timestamp_s": ("float64", ()),
    "usage_index": ("uint8", ()),
}

PIPELINE_ARRAYS = {
    "executed_action_chunks": ("float32", (-1, 5, 6)),
    "frozen_base_prediction": ("float16", (-1, 3, -1)),
    "common_frozen_target": ("float16", (-1, 3, -1)),
    "current_frozen_target": ("float16", (3, -1)),
}


def estimate_array_bytes(specifications: Mapping[str, tuple[tuple[int, ...], np.dtype]]) -> int:
    total = 0
    for shape, dtype in specifications.values():
        if any(int(size) <= 0 for size in shape):
            raise ValueError("cache byte estimation requires concrete positive shapes")
        total += int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    return total


def assert_cache_budget(total_bytes: int) -> None:
    if total_bytes <= 0:
        raise ValueError("cache size must be positive")
    if total_bytes > MAX_CACHE_BYTES:
        raise ValueError(
            f"token cache would use {total_bytes / 1024**3:.2f} GiB, "
            f"exceeding the 120 GiB protocol limit"
        )


def write_cache_manifest(
    root: Path,
    *,
    canonical_manifest_sha256: str,
    v12_checkpoint_sha256: str,
    arrays: Mapping[str, np.ndarray],
    source_names: list[str],
    task_names: list[str],
    include_pipeline: bool,
) -> dict:
    root = root.resolve()
    required = set(BASE_ARRAYS) | (set(PIPELINE_ARRAYS) if include_pipeline else set())
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"cache is missing required arrays: {sorted(missing)}")
    sample_counts = {int(value.shape[0]) for value in arrays.values()}
    if len(sample_counts) != 1:
        raise ValueError("all cache arrays must have the same leading sample count")
    array_rows = {}
    total_bytes = 0
    for name in sorted(required):
        value = arrays[name]
        path = root / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        total_bytes += path.stat().st_size
        array_rows[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "file": path.name,
            "file_sha256": sha256_file(path),
        }
    assert_cache_budget(total_bytes)
    manifest = {
        "format": CACHE_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "canonical_manifest_sha256": canonical_manifest_sha256,
        "v12_checkpoint_sha256": v12_checkpoint_sha256,
        "samples": sample_counts.pop(),
        "cache_grid_size": CACHE_GRID_SIZE,
        "cache_grid_radius_px": CACHE_GRID_RADIUS_PX,
        "fcpg_grid_size": FCPG_GRID_SIZE,
        "arrays": array_rows,
        "source_names": list(source_names),
        "task_names": list(task_names),
        "include_pipeline": bool(include_pipeline),
        "total_bytes": total_bytes,
    }
    manifest["content_sha256"] = sha256_json(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _validate_tail(shape: tuple[int, ...], expected: tuple[int, ...], name: str) -> None:
    if len(shape) != len(expected) + 1:
        raise ValueError(f"{name} rank does not match cache contract")
    for actual, required in zip(shape[1:], expected, strict=True):
        if required != -1 and int(actual) != required:
            raise ValueError(f"{name} shape {shape} violates tail {expected}")


class ForwardTokenCache:
    def __init__(
        self,
        root: Path,
        *,
        require_pipeline: bool,
        verify_file_hashes: bool = False,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("format") != CACHE_FORMAT:
            raise ValueError("unexpected token-cache format")
        if self.manifest.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("token cache belongs to another protocol")
        if int(self.manifest.get("cache_grid_size", -1)) != CACHE_GRID_SIZE:
            raise ValueError("token cache grid size changed")
        if float(self.manifest.get("cache_grid_radius_px", -1)) != CACHE_GRID_RADIUS_PX:
            raise ValueError("token cache grid radius changed")
        if require_pipeline and not self.manifest.get("include_pipeline"):
            raise ValueError("pipeline training requires future/base target arrays")
        expected = dict(BASE_ARRAYS)
        if require_pipeline:
            expected.update(PIPELINE_ARRAYS)
        self.arrays: dict[str, np.ndarray] = {}
        self._device_lru: DeviceBatchLRU | None = None
        samples = int(self.manifest["samples"])
        for name, (dtype, tail) in expected.items():
            record = self.manifest.get("arrays", {}).get(name)
            if not isinstance(record, dict):
                raise ValueError(f"cache manifest has no {name} record")
            path = self.root / record["file"]
            array = np.load(path, mmap_mode="r")
            if str(array.dtype) != dtype:
                raise ValueError(f"{name} dtype {array.dtype} != {dtype}")
            _validate_tail(tuple(array.shape), tail, name)
            if len(array) != samples or list(array.shape) != record["shape"]:
                raise ValueError(f"{name} manifest shape mismatch")
            if verify_file_hashes and sha256_file(path) != record["file_sha256"]:
                raise ValueError(f"{name} file SHA-256 mismatch")
            self.arrays[name] = array
        self.samples = samples

    def enable_gpu_lru(self, *, device: torch.device, requested_bytes: int) -> int:
        """Enable a bounded, lazy GPU row cache and return its effective budget."""

        safe_bytes = safe_gpu_cache_bytes(device, requested_bytes)
        if safe_bytes == 0:
            self._device_lru = None
            return 0
        bytes_per_row = sum(int(array[0].nbytes) for array in self.arrays.values())
        if safe_bytes < bytes_per_row:
            self._device_lru = None
            return 0
        self._device_lru = DeviceBatchLRU(
            self.arrays,
            device=device,
            max_bytes=safe_bytes,
        )
        return safe_bytes

    def device_lru_stats(self) -> dict[str, int | float | str | bool]:
        if self._device_lru is None:
            return {"enabled": False}
        return self._device_lru.stats()

    def batch(self, indices: np.ndarray, *, device: torch.device) -> dict[str, torch.Tensor]:
        index = np.asarray(indices, dtype=np.int64)
        if index.ndim != 1 or len(index) == 0:
            raise ValueError("batch indices must be one non-empty vector")
        if np.any(index < 0) or np.any(index >= self.samples):
            raise IndexError("batch index outside token cache")
        if self._device_lru is not None:
            if device != self._device_lru.device:
                raise ValueError("batch device differs from configured device LRU")
            return self._device_lru.batch(index)
        result = {}
        for name, array in self.arrays.items():
            # Materialize one contiguous CPU batch before one device transfer.
            value = np.asarray(array[index]).copy()
            result[name] = torch.from_numpy(value).to(device)
        return result


@dataclass
class _DevicePage:
    row_indices: tuple[int, ...]
    tensors: dict[str, torch.Tensor]
    bytes: int


class DeviceBatchLRU:
    """Lazy page LRU for rows read from an on-disk NumPy token cache.

    Each miss set is transferred once per array and retained as one page.  The
    cache never preallocates its full budget and evicts whole least-recently-used
    pages before the resident page total can exceed ``max_bytes``.  The training
    batch itself is not counted as cache residency.
    """

    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        *,
        device: torch.device,
        max_bytes: int,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("device LRU max_bytes must be positive")
        if not arrays:
            raise ValueError("device LRU requires at least one array")
        samples = {len(array) for array in arrays.values()}
        if len(samples) != 1:
            raise ValueError("device LRU arrays must share a leading row count")
        self.arrays = dict(arrays)
        self.device = torch.device(device)
        self.max_bytes = int(max_bytes)
        self.pages: OrderedDict[int, _DevicePage] = OrderedDict()
        self.row_locations: dict[int, tuple[int, int]] = {}
        self.next_page_id = 0
        self.current_bytes = 0
        self.peak_bytes = 0
        self.requests = 0
        self.row_hits = 0
        self.row_misses = 0
        self.evictions = 0
        self.workspace_evictions = 0
        self.bypassed_pages = 0
        self.admit_misses = True
        self.minimum_observed_free_bytes: int | None = None

    @staticmethod
    def _unique_in_order(index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions: dict[int, int] = {}
        unique: list[int] = []
        inverse = np.empty(len(index), dtype=np.int64)
        for position, raw in enumerate(index):
            row = int(raw)
            location = positions.get(row)
            if location is None:
                location = len(unique)
                positions[row] = location
                unique.append(row)
            inverse[position] = location
        return np.asarray(unique, dtype=np.int64), inverse

    @staticmethod
    def _tensor_bytes(tensors: Mapping[str, torch.Tensor]) -> int:
        return sum(value.numel() * value.element_size() for value in tensors.values())

    def _load_rows(self, rows: np.ndarray) -> dict[str, torch.Tensor]:
        tensors = {}
        for name, array in self.arrays.items():
            contiguous = np.asarray(array[rows]).copy()
            tensors[name] = torch.from_numpy(contiguous).to(self.device)
        return tensors

    def _evict_for(self, incoming_bytes: int) -> None:
        while self.pages and self.current_bytes + incoming_bytes > self.max_bytes:
            page_id, page = self.pages.popitem(last=False)
            self.current_bytes -= page.bytes
            for row in page.row_indices:
                if self.row_locations.get(row, (None, None))[0] == page_id:
                    del self.row_locations[row]
            self.evictions += 1

    def evict_bytes(self, minimum_bytes: int, *, workspace: bool = False) -> int:
        """Release oldest pages until at least ``minimum_bytes`` are removed."""

        if minimum_bytes < 0:
            raise ValueError("device LRU eviction target cannot be negative")
        removed = 0
        before = self.evictions
        while self.pages and removed < minimum_bytes:
            page_id, page = self.pages.popitem(last=False)
            self.current_bytes -= page.bytes
            removed += page.bytes
            for row in page.row_indices:
                if self.row_locations.get(row, (None, None))[0] == page_id:
                    del self.row_locations[row]
            self.evictions += 1
        if workspace:
            self.workspace_evictions += self.evictions - before
        return removed

    def resize(self, max_bytes: int, *, workspace: bool = False) -> int:
        """Change the residency budget and evict old pages when it shrinks.

        A zero-byte budget is valid after construction: misses are still
        materialized for the active batch, but no page is retained.  This lets
        a multi-source trainer lend the full aggregate cache to its current
        sequential phase without creating a second cache or changing samples.
        """

        if int(max_bytes) < 0:
            raise ValueError("device LRU max bytes cannot be negative")
        self.max_bytes = int(max_bytes)
        removed = 0
        before = self.evictions
        while self.pages and self.current_bytes > self.max_bytes:
            page_id, page = self.pages.popitem(last=False)
            self.current_bytes -= page.bytes
            removed += page.bytes
            for row in page.row_indices:
                if self.row_locations.get(row, (None, None))[0] == page_id:
                    del self.row_locations[row]
            self.evictions += 1
        if workspace:
            self.workspace_evictions += self.evictions - before
        return removed

    def reserve_active_workspace(self, minimum_free_bytes: int) -> int:
        """Evict idle pages so model activations retain a hard free-VRAM floor."""

        if minimum_free_bytes < 0:
            raise ValueError("minimum free GPU bytes cannot be negative")
        if self.device.type != "cuda":
            return 0
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        observed = int(free_bytes)
        self.minimum_observed_free_bytes = (
            observed
            if self.minimum_observed_free_bytes is None
            else min(self.minimum_observed_free_bytes, observed)
        )
        needed = max(0, int(minimum_free_bytes) - observed)
        if needed == 0:
            return 0
        removed = self.evict_bytes(needed, workspace=True)
        if removed:
            torch.cuda.empty_cache()
        free_after, _ = torch.cuda.mem_get_info(self.device)
        self.minimum_observed_free_bytes = min(
            int(self.minimum_observed_free_bytes), int(free_after)
        )
        if free_after < minimum_free_bytes:
            raise RuntimeError(
                "device LRU could not release the requested active workspace"
            )
        return removed

    def batch(self, indices: np.ndarray) -> dict[str, torch.Tensor]:
        index = np.asarray(indices, dtype=np.int64)
        if index.ndim != 1 or len(index) == 0:
            raise ValueError("device LRU indices must be one non-empty vector")
        self.requests += len(index)
        unique, inverse = self._unique_in_order(index)
        locations: dict[int, tuple[_DevicePage, int]] = {}
        misses: list[int] = []
        batch_hits = 0
        for row in unique:
            row_value = int(row)
            location = self.row_locations.get(row_value)
            if location is None:
                misses.append(row_value)
                continue
            page_id, offset = location
            page = self.pages.get(page_id)
            if page is None:
                misses.append(row_value)
                continue
            self.pages.move_to_end(page_id)
            locations[row_value] = (page, offset)
            self.row_hits += 1
            batch_hits += 1
        self.row_misses += len(misses)

        miss_tensors: dict[str, torch.Tensor] | None = None
        if misses:
            miss_rows = np.asarray(misses, dtype=np.int64)
            miss_tensors = self._load_rows(miss_rows)
            page_bytes = self._tensor_bytes(miss_tensors)
            page = _DevicePage(tuple(misses), miss_tensors, page_bytes)
            for offset, row in enumerate(misses):
                locations[row] = (page, offset)
            if self.admit_misses and page_bytes <= self.max_bytes:
                self._evict_for(page_bytes)
                page_id = self.next_page_id
                self.next_page_id += 1
                self.pages[page_id] = page
                self.current_bytes += page_bytes
                self.peak_bytes = max(self.peak_bytes, self.current_bytes)
                for offset, row in enumerate(misses):
                    self.row_locations[row] = (page_id, offset)
            else:
                self.bypassed_pages += 1

        no_duplicates = len(unique) == len(index)
        if not batch_hits and miss_tensors is not None and no_duplicates:
            return miss_tensors

        unique_position = {int(row): position for position, row in enumerate(unique)}
        result: dict[str, torch.Tensor] = {}
        for name, array in self.arrays.items():
            output = torch.empty(
                (len(unique), *array.shape[1:]),
                dtype=torch.from_numpy(np.empty((), dtype=array.dtype)).dtype,
                device=self.device,
            )
            by_page: dict[int, tuple[_DevicePage, list[int], list[int]]] = {}
            for row, (page, offset) in locations.items():
                key = id(page)
                if key not in by_page:
                    by_page[key] = (page, [], [])
                by_page[key][1].append(unique_position[row])
                by_page[key][2].append(offset)
            for page, destinations, offsets in by_page.values():
                destination = torch.as_tensor(destinations, dtype=torch.long, device=self.device)
                source = torch.as_tensor(offsets, dtype=torch.long, device=self.device)
                output.index_copy_(0, destination, page.tensors[name].index_select(0, source))
            if no_duplicates:
                result[name] = output
            else:
                gather = torch.from_numpy(inverse).to(self.device)
                result[name] = output.index_select(0, gather)
        return result

    def freeze_admission(self) -> None:
        """Keep current pages resident while serving later misses transiently."""

        self.admit_misses = False

    def insert_preloaded_page(
        self,
        rows: np.ndarray,
        values: Mapping[str, np.ndarray],
    ) -> int:
        """Insert one CPU-prefetched page without rereading backing arrays."""

        selected = np.asarray(rows, dtype=np.int64)
        if selected.ndim != 1 or len(selected) == 0:
            raise ValueError("preloaded device page rows must be non-empty [N]")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("preloaded device page rows must be unique")
        if set(values) != set(self.arrays):
            raise ValueError("preloaded device page arrays differ from LRU")
        if any(int(row) in self.row_locations for row in selected):
            raise ValueError("preloaded device page repeats a resident row")
        tensors: dict[str, torch.Tensor] = {}
        for name, array in self.arrays.items():
            value = np.ascontiguousarray(values[name])
            expected = (len(selected), *array.shape[1:])
            if value.shape != expected or value.dtype != array.dtype:
                raise ValueError(f"preloaded {name} violates shape or dtype")
            tensors[name] = torch.from_numpy(value).to(self.device)
        page_bytes = self._tensor_bytes(tensors)
        if page_bytes > self.max_bytes:
            raise ValueError("preloaded device page exceeds LRU budget")
        self._evict_for(page_bytes)
        page_id = self.next_page_id
        self.next_page_id += 1
        page = _DevicePage(tuple(map(int, selected)), tensors, page_bytes)
        self.pages[page_id] = page
        self.current_bytes += page_bytes
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        for offset, row in enumerate(selected):
            self.row_locations[int(row)] = (page_id, offset)
        return page_bytes

    def stats(self) -> dict[str, int | float | str | bool]:
        total = self.row_hits + self.row_misses
        return {
            "enabled": True,
            "device": str(self.device),
            "max_bytes": self.max_bytes,
            "resident_bytes": self.current_bytes,
            "peak_resident_bytes": self.peak_bytes,
            "resident_pages": len(self.pages),
            "resident_rows": len(self.row_locations),
            "requests": self.requests,
            "unique_row_hits": self.row_hits,
            "unique_row_misses": self.row_misses,
            "hit_rate": self.row_hits / total if total else 0.0,
            "evictions": self.evictions,
            "workspace_evictions": self.workspace_evictions,
            "bypassed_pages": self.bypassed_pages,
            "admit_misses": self.admit_misses,
            "minimum_observed_free_bytes": self.minimum_observed_free_bytes,
        }


def fcpg_local_tokens_from_cache(grid_tokens: torch.Tensor) -> torch.Tensor:
    """Extract the centered 5x5 FCPG neighborhood from a cached 7x7 grid."""

    if grid_tokens.ndim < 3 or grid_tokens.shape[-2] != CACHE_GRID_SIZE**2:
        raise ValueError("grid_tokens must end in [49,D]")
    leading = grid_tokens.shape[:-2]
    latent = grid_tokens.shape[-1]
    grid = grid_tokens.reshape(*leading, CACHE_GRID_SIZE, CACHE_GRID_SIZE, latent)
    inner = grid[..., 1:-1, 1:-1, :]
    return inner.reshape(*leading, FCPG_GRID_SIZE**2, latent)


def fcpg_relative_offsets(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coordinates = torch.linspace(-1.0, 1.0, FCPG_GRID_SIZE, device=device, dtype=dtype)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def sample_corrected_anchor_tokens(
    tip_grid_tokens: torch.Tensor,
    tip_delta_px: torch.Tensor,
) -> torch.Tensor:
    """Sample two corrected tip tokens and derive the center token by mean."""

    if tip_grid_tokens.ndim != 5:
        raise ValueError("tip_grid_tokens must have shape [B,2,2,49,D]")
    batch, views, tips, cells, latent = tip_grid_tokens.shape
    if (views, tips, cells) != (2, 2, CACHE_GRID_SIZE**2):
        raise ValueError("tip grid violates [B,2,2,49,D] contract")
    if tip_delta_px.shape != (batch, views, tips, 2):
        raise ValueError("tip_delta_px must have shape [B,2,2,2]")
    feature = tip_grid_tokens.reshape(
        batch * views * tips, CACHE_GRID_SIZE, CACHE_GRID_SIZE, latent
    ).permute(0, 3, 1, 2)
    grid = (tip_delta_px / CACHE_GRID_RADIUS_PX).reshape(
        batch * views * tips, 1, 1, 2
    )
    sampled = F.grid_sample(
        feature,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).reshape(batch, views, tips, latent)
    center = sampled.mean(dim=2, keepdim=True)
    return torch.cat((center, sampled), dim=2)


class BalancedCappedSampler:
    """Hierarchical source/task-balanced sampler with a per-source row cap.

    Sources are sampled uniformly, then tasks are cycled uniformly within each
    source.  A batch of at least two is allocated in source pairs so every
    selected source contributes at least two examples.  This is required for
    the protocol's per-source SIGReg rather than a pooled surrogate.
    """

    def __init__(
        self,
        source_index: np.ndarray,
        task_index: np.ndarray,
        eligible: np.ndarray,
        *,
        per_source_cap: int,
        seed: int,
    ) -> None:
        source = np.asarray(source_index, dtype=np.int64)
        task = np.asarray(task_index, dtype=np.int64)
        mask = np.asarray(eligible, dtype=bool)
        if source.shape != task.shape or source.shape != mask.shape or source.ndim != 1:
            raise ValueError("source/task/eligible arrays must share shape [N]")
        if per_source_cap <= 0:
            raise ValueError("per_source_cap must be positive")
        rng = np.random.default_rng(seed)
        admitted: list[int] = []
        for source_value in sorted(np.unique(source[mask])):
            rows = np.flatnonzero(mask & (source == source_value))
            if len(rows) > per_source_cap:
                rows = np.sort(rng.choice(rows, size=per_source_cap, replace=False))
            admitted.extend(int(row) for row in rows)
        groups: dict[tuple[int, int], np.ndarray] = {}
        admitted_array = np.asarray(sorted(admitted), dtype=np.int64)
        for source_value, task_value in sorted(
            set(zip(source[admitted_array], task[admitted_array], strict=True))
        ):
            rows = admitted_array[
                (source[admitted_array] == source_value)
                & (task[admitted_array] == task_value)
            ]
            if len(rows):
                groups[(int(source_value), int(task_value))] = rows
        if not groups:
            raise ValueError("balanced sampler has no eligible source/task groups")
        self.groups = groups
        self.group_keys = tuple(sorted(groups))
        self.source_keys = tuple(sorted({key[0] for key in self.group_keys}))
        self.groups_by_source = {
            source_value: tuple(
                key for key in self.group_keys if key[0] == source_value
            )
            for source_value in self.source_keys
        }
        self.rng = rng

    def batch(self, batch_size: int) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size == 1:
            source_order = [self.source_keys[int(self.rng.integers(len(self.source_keys)))]]
        else:
            pairs = batch_size // 2
            source_order: list[int] = []
            while len(source_order) < pairs:
                order = self.rng.permutation(len(self.source_keys))
                source_order.extend(self.source_keys[int(index)] for index in order)
            source_order = source_order[:pairs]
            source_order = [source for source in source_order for _ in range(2)]
            if len(source_order) < batch_size:
                source_order.append(source_order[0])

        selected = np.empty(batch_size, dtype=np.int64)
        task_orders: dict[int, list[tuple[int, int]]] = {}
        for index, source_value in enumerate(source_order):
            choices = task_orders.get(source_value)
            if not choices:
                permutation = self.rng.permutation(len(self.groups_by_source[source_value]))
                choices = [
                    self.groups_by_source[source_value][int(task_index)]
                    for task_index in permutation
                ]
                task_orders[source_value] = choices
            group = choices.pop()
            rows = self.groups[group]
            selected[index] = rows[int(self.rng.integers(0, len(rows)))]
        return selected
