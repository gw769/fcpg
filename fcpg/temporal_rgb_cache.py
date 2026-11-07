"""Schedule-bound, mmap-compatible RGB cache for temporal FCPG-H.

Compressed archive video is decoded once during cache materialization.  The
training reader then fetches raw 224x224 uint8 rows from NVMe.  The cache is
bound to the exact sampler and view-selection schedule; it is not a generic
copy of the 17.6M-frame archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from fcpg.lerobot_archive import decode_episode_views
from fcpg.offline_pipeline_manifest import sha256_file, sha256_json
from fcpg.temporal_grounding_protocol import (
    DATA_DISK_RESERVE_BYTES,
    GROUNDED_TOKEN_CACHE_BYTES,
    TEMPORAL_PROTOCOL_ID,
    TEMPORAL_RGB_CACHE_BYTES,
    select_roleless_views,
)


RGB_CACHE_FORMAT = "so101-temporal-rgb-cache-v1"
RGB_CACHE_PLAN_FORMAT = "so101-temporal-rgb-cache-plan-v1"
RGB_CACHE_COVERAGE_FULL = "full_rgb_schedule"
RGB_CACHE_COVERAGE_CAUSAL_EVENT = "causal_event_training_schedule"
RGB_FRAME_SHAPE = (224, 224, 3)
RGB_FRAME_BYTES = int(np.prod(RGB_FRAME_SHAPE))
VIEW_SELECTION_FORMULA = "selection_seed=seed*1000003+zero_based_step"


@dataclass(frozen=True)
class RGBSchedule:
    mode: str
    seed: int
    steps: int
    batch_size: int
    low_motion_fraction: float = 0.25
    per_source_cap: int = 100_000
    block_size: int = 8
    cluster_pair_rows: int = 12
    require_cluster_pairs: bool = True
    max_views_per_window: int = 1
    view_selection_formula: str = VIEW_SELECTION_FORMULA


@dataclass(frozen=True)
class PlannedRGBPack:
    pack_id: int
    episode_index: int
    canonical_episode_id: str
    view: str
    offsets: np.ndarray
    access_count: int


@dataclass(frozen=True)
class RGBDiskCacheStats:
    frame_requests: int
    frame_hits: int
    frame_misses: int
    bytes_read: int
    pack_opens: int
    pack_evictions: int
    open_packs: int


def schedule_for_training(
    *, mode: str, seed: int, steps: int, batch_size: int
) -> RGBSchedule:
    return RGBSchedule(
        mode=str(mode),
        seed=int(seed),
        steps=int(steps),
        batch_size=int(batch_size),
        require_cluster_pairs=mode != "contract_smoke",
    )


def _schedule_payload(schedule: RGBSchedule) -> dict[str, object]:
    return asdict(schedule)


def _pack_layout(index: Any) -> tuple[list[dict[str, object]], dict[tuple[int, str], tuple[int, int]]]:
    specs: list[dict[str, object]] = []
    lookup: dict[tuple[int, str], tuple[int, int]] = {}
    cursor = 0
    for episode_index, record in enumerate(index.episodes):
        length = int(record["frame_stop"]) - int(record["frame_start"])
        if length <= 0:
            raise ValueError("temporal episode has non-positive frame span")
        row = record["manifest_row"]
        for view in map(str, row["view_keys"]):
            pack_id = len(specs)
            specs.append(
                {
                    "pack_id": pack_id,
                    "episode_index": episode_index,
                    "canonical_episode_id": str(record["canonical_episode_id"]),
                    "view": view,
                    "length": length,
                    "count_start": cursor,
                }
            )
            lookup[(episode_index, view)] = (pack_id, cursor)
            cursor += length
    return specs, lookup


def plan_temporal_rgb_cache(
    index: Any,
    *,
    schedule: RGBSchedule,
    minimum_use_count: int,
) -> tuple[dict[str, object], list[PlannedRGBPack]]:
    """Replay a deterministic training schedule and select cache rows."""

    if minimum_use_count <= 0:
        raise ValueError("minimum_use_count must be positive")
    if schedule.steps <= 0 or schedule.batch_size <= 0:
        raise ValueError("RGB cache schedule must be non-empty")
    # Local import avoids making the runtime TemporalIndex/cache relationship
    # circular during module initialization.
    from fcpg.temporal_index import EpisodeBlockSampler

    specs, lookup = _pack_layout(index)
    total_view_frames = sum(int(spec["length"]) for spec in specs)
    counts = np.zeros(total_view_frames, dtype=np.uint16)
    sampler = EpisodeBlockSampler(
        index,
        seed=schedule.seed,
        per_source_cap=schedule.per_source_cap,
        block_size=schedule.block_size,
        cluster_pair_rows=schedule.cluster_pair_rows,
        require_cluster_pairs=schedule.require_cluster_pairs,
    )
    batch_unioned_accesses = 0
    for step in range(schedule.steps):
        windows = sampler.batch(
            schedule.batch_size,
            low_motion_fraction=schedule.low_motion_fraction,
        )
        requested: dict[tuple[int, str], set[int]] = defaultdict(set)
        for raw_window in windows:
            window = int(raw_window)
            episode_index = int(index.arrays["window_episode"][window])
            record = index.episodes[episode_index]
            views = select_roleless_views(
                record,
                maximum_views=schedule.max_views_per_window,
                selection_seed=schedule.seed * 1_000_003 + step,
            )
            offsets = np.asarray(
                index.arrays["window_image_indices"][window], dtype=np.int64
            )
            for view in views:
                requested[(episode_index, view)].update(map(int, offsets))
        for key, offsets_set in requested.items():
            _, count_start = lookup[key]
            offsets = np.fromiter(sorted(offsets_set), dtype=np.int64)
            if offsets.size and (int(offsets[0]) < 0 or int(offsets[-1]) >= int(specs[lookup[key][0]]["length"])):
                raise ValueError("training schedule requested an out-of-range frame")
            np.add.at(counts, count_start + offsets, 1)
            batch_unioned_accesses += int(offsets.size)

    unique_schedule_frames = int(np.count_nonzero(counts))
    selected_frame_count = int(np.count_nonzero(counts >= minimum_use_count))
    packs: list[PlannedRGBPack] = []
    use_histogram: dict[str, int] = {}
    active_counts = counts[counts > 0]
    if active_counts.size:
        values, frequencies = np.unique(active_counts, return_counts=True)
        use_histogram = {
            str(int(value)): int(frequency)
            for value, frequency in zip(values, frequencies, strict=True)
        }
    for spec in specs:
        start = int(spec["count_start"])
        stop = start + int(spec["length"])
        local_counts = counts[start:stop]
        offsets = np.flatnonzero(local_counts >= minimum_use_count).astype(np.int32)
        if not offsets.size:
            continue
        packs.append(
            PlannedRGBPack(
                pack_id=int(spec["pack_id"]),
                episode_index=int(spec["episode_index"]),
                canonical_episode_id=str(spec["canonical_episode_id"]),
                view=str(spec["view"]),
                offsets=offsets,
                access_count=int(local_counts[offsets].sum()),
            )
        )
    planned_array_bytes = selected_frame_count * RGB_FRAME_BYTES
    planned_offset_bytes = selected_frame_count * np.dtype(np.int32).itemsize
    plan = {
        "format": RGB_CACHE_PLAN_FORMAT,
        "protocol_id": TEMPORAL_PROTOCOL_ID,
        "schedule": _schedule_payload(schedule),
        "minimum_use_count": int(minimum_use_count),
        "unique_schedule_frames": unique_schedule_frames,
        "selected_frame_count": selected_frame_count,
        "selected_pack_count": len(packs),
        "batch_unioned_frame_accesses": int(batch_unioned_accesses),
        "use_histogram": use_histogram,
        "planned_rgb_array_bytes": planned_array_bytes,
        "planned_offset_array_bytes": planned_offset_bytes,
        "planned_bytes": planned_array_bytes + planned_offset_bytes,
        "complete_schedule_coverage": minimum_use_count == 1,
        "sampler_stats": sampler.stats(),
    }
    plan["content_sha256"] = sha256_json(plan)
    return plan, packs


def assert_rgb_cache_disk_budget(
    root: Path,
    *,
    planned_bytes: int,
    cache_limit_bytes: int = TEMPORAL_RGB_CACHE_BYTES,
    data_reserve_bytes: int = DATA_DISK_RESERVE_BYTES,
    future_token_allowance_bytes: int = GROUNDED_TOKEN_CACHE_BYTES,
) -> None:
    if planned_bytes <= 0:
        raise ValueError("planned RGB cache bytes must be positive")
    if planned_bytes > cache_limit_bytes:
        raise ValueError("temporal RGB cache exceeds the 320 GiB protocol limit")
    free = shutil.disk_usage(root).free
    required_after_build = data_reserve_bytes + future_token_allowance_bytes
    if free - planned_bytes < required_after_build:
        raise OSError(
            "RGB cache plan would violate the 128 GiB data reserve plus "
            "120 GiB future grounded-token allowance"
        )


def _write_npy_atomic(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous)).hexdigest()


def _materialize_pack_job(payload: tuple[dict, str, np.ndarray, Path, Path, Path]) -> dict[str, object]:
    row, view, offsets, offsets_path, rgb_path, receipt_path = payload
    frames = decode_episode_views(
        row,
        (view,),
        map(int, offsets),
        decoder_threads=1,
    )[view]
    expected = (len(offsets), *RGB_FRAME_SHAPE)
    if frames.shape != expected or frames.dtype != np.uint8:
        raise ValueError(f"decoded RGB pack has {frames.shape}/{frames.dtype}, expected {expected}/uint8")
    _write_npy_atomic(offsets_path, np.asarray(offsets, dtype=np.int32))
    _write_npy_atomic(rgb_path, np.ascontiguousarray(frames))
    receipt = {
        "offsets_file": offsets_path.name,
        "rgb_file": rgb_path.name,
        "frame_count": len(offsets),
        "offsets_sha256": _array_sha256(np.asarray(offsets, dtype=np.int32)),
        "rgb_content_sha256": _array_sha256(frames),
        "offsets_file_bytes": offsets_path.stat().st_size,
        "rgb_file_bytes": rgb_path.stat().st_size,
    }
    temporary = receipt_path.with_name(receipt_path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, receipt_path)
    return receipt


def materialize_temporal_rgb_cache(
    index: Any,
    *,
    output: Path,
    plan: Mapping[str, object],
    packs: Iterable[PlannedRGBPack],
    index_manifest_sha256: str,
    cluster_manifest_sha256: str,
    workers: int = 4,
    resume: bool = False,
) -> dict[str, object]:
    """Decode planned packs once and atomically publish a completed manifest."""

    output = Path(output).expanduser().resolve()
    pack_list = list(packs)
    if workers <= 0:
        raise ValueError("RGB cache builder workers must be positive")
    if output.exists() and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    packs_root = output / "packs"
    plan_root = output / "plan_offsets"
    packs_root.mkdir(exist_ok=True)
    plan_root.mkdir(exist_ok=True)
    plan_path = output / "plan.json"
    prior_plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    persisted_plan = dict(plan)
    persisted_plan.update(
        {
            "index_manifest_sha256": index_manifest_sha256,
            "motion_cluster_manifest_sha256": cluster_manifest_sha256,
        }
    )
    persisted_plan["artifact_sha256"] = sha256_json(persisted_plan)
    if prior_plan is not None and prior_plan != persisted_plan:
        raise ValueError("resume plan does not match the existing RGB cache plan")
    if prior_plan is None:
        plan_path.write_text(json.dumps(persisted_plan, indent=2, sort_keys=True) + "\n")

    payloads: list[tuple[dict, str, np.ndarray, Path, Path, Path]] = []
    pack_metadata: dict[int, dict[str, object]] = {}
    by_id = {pack.pack_id: pack for pack in pack_list}
    for pack in pack_list:
        stem = f"{pack.pack_id:06d}"
        plan_offsets = plan_root / f"{stem}.npy"
        if not plan_offsets.exists():
            _write_npy_atomic(plan_offsets, pack.offsets)
        offsets_path = packs_root / f"{stem}.offsets.npy"
        rgb_path = packs_root / f"{stem}.rgb.npy"
        receipt_path = packs_root / f"{stem}.json"
        if resume and receipt_path.exists() and offsets_path.exists() and rgb_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (
                int(receipt.get("frame_count", -1)) == len(pack.offsets)
                and int(receipt.get("offsets_file_bytes", -1)) == offsets_path.stat().st_size
                and int(receipt.get("rgb_file_bytes", -1)) == rgb_path.stat().st_size
            ):
                pack_metadata[pack.pack_id] = receipt
                continue
        row = index.episodes[pack.episode_index]["manifest_row"]
        payloads.append((row, pack.view, pack.offsets, offsets_path, rgb_path, receipt_path))

    if payloads and workers == 1:
        receipts = map(_materialize_pack_job, payloads)
        for payload, receipt in zip(payloads, receipts, strict=True):
            stem = payload[3].name.split(".", 1)[0]
            pack_metadata[int(stem)] = receipt
    elif payloads:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for payload, receipt in zip(payloads, pool.map(_materialize_pack_job, payloads), strict=True):
                stem = payload[3].name.split(".", 1)[0]
                pack_metadata[int(stem)] = receipt

    manifest_packs: list[dict[str, object]] = []
    for pack_id in sorted(by_id):
        pack = by_id[pack_id]
        receipt = pack_metadata.get(pack_id)
        if receipt is None:
            receipt_path = packs_root / f"{pack_id:06d}.json"
            if not receipt_path.exists():
                raise RuntimeError(f"RGB cache pack {pack_id} did not complete")
            receipt = json.loads(receipt_path.read_text())
        manifest_packs.append(
            {
                "pack_id": pack_id,
                "episode_index": pack.episode_index,
                "canonical_episode_id": pack.canonical_episode_id,
                "view": pack.view,
                "access_count": pack.access_count,
                **receipt,
            }
        )
    actual_bytes = sum(
        int(pack["offsets_file_bytes"]) + int(pack["rgb_file_bytes"])
        for pack in manifest_packs
    )
    manifest = {
        "format": RGB_CACHE_FORMAT,
        "protocol_id": TEMPORAL_PROTOCOL_ID,
        "complete": True,
        "index_manifest_sha256": index_manifest_sha256,
        "motion_cluster_manifest_sha256": cluster_manifest_sha256,
        "schedule": dict(plan["schedule"]),
        "plan_content_sha256": str(plan["content_sha256"]),
        "minimum_use_count": int(plan["minimum_use_count"]),
        "unique_schedule_frames": int(plan["unique_schedule_frames"]),
        "selected_frame_count": int(plan["selected_frame_count"]),
        "batch_unioned_frame_accesses": int(plan["batch_unioned_frame_accesses"]),
        "complete_schedule_coverage": bool(plan["complete_schedule_coverage"]),
        "coverage_scope": str(
            plan.get("coverage_scope", RGB_CACHE_COVERAGE_FULL)
        ),
        "event_ledger_manifest_sha256": plan.get(
            "event_ledger_manifest_sha256"
        ),
        "frame_shape": list(RGB_FRAME_SHAPE),
        "dtype": "uint8",
        "actual_pack_bytes": actual_bytes,
        "packs": manifest_packs,
    }
    manifest["content_sha256"] = sha256_json(manifest)
    manifest_path = output / "manifest.json"
    temporary = output / f"manifest.json.tmp-{os.getpid()}"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    return manifest


class TemporalRGBDiskCache:
    """Read-only mmap pack reader with a bounded open-file LRU."""

    def __init__(self, root: Path, *, max_open_packs: int = 128) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("format") != RGB_CACHE_FORMAT or not self.manifest.get("complete"):
            raise ValueError("temporal RGB cache is incomplete or has the wrong format")
        if self.manifest.get("protocol_id") != TEMPORAL_PROTOCOL_ID:
            raise ValueError("temporal RGB cache protocol mismatch")
        if tuple(self.manifest.get("frame_shape", ())) != RGB_FRAME_SHAPE:
            raise ValueError("temporal RGB cache frame shape mismatch")
        if self.manifest.get("dtype") != "uint8":
            raise ValueError("temporal RGB cache dtype mismatch")
        if max_open_packs <= 0:
            raise ValueError("max_open_packs must be positive")
        self.max_open_packs = int(max_open_packs)
        self._pack_by_key = {
            (str(pack["canonical_episode_id"]), str(pack["view"])): dict(pack)
            for pack in self.manifest["packs"]
        }
        if len(self._pack_by_key) != len(self.manifest["packs"]):
            raise ValueError("temporal RGB cache has duplicate episode/view packs")
        self._open: OrderedDict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.frame_requests = 0
        self.frame_hits = 0
        self.frame_misses = 0
        self.bytes_read = 0
        self.pack_opens = 0
        self.pack_evictions = 0

    @property
    def complete_schedule_coverage(self) -> bool:
        return bool(self.manifest["complete_schedule_coverage"])

    def assert_binding(
        self,
        *,
        index_manifest: Path,
        cluster_manifest: Path,
        schedule: RGBSchedule,
        require_complete: bool,
        allow_complete_prefix: bool = False,
        event_ledger_manifest: Path | None = None,
    ) -> None:
        if sha256_file(index_manifest) != self.manifest["index_manifest_sha256"]:
            raise ValueError("RGB cache temporal-index SHA mismatch")
        if sha256_file(cluster_manifest) != self.manifest["motion_cluster_manifest_sha256"]:
            raise ValueError("RGB cache motion-cluster SHA mismatch")
        requested = _schedule_payload(schedule)
        cached = self.manifest["schedule"]
        if requested != cached:
            prefix_fields = set(requested) - {"mode", "steps"}
            prefix_match = (
                allow_complete_prefix
                and self.complete_schedule_coverage
                and int(requested["steps"]) <= int(cached["steps"])
                and all(requested[name] == cached.get(name) for name in prefix_fields)
            )
            if not prefix_match:
                raise ValueError("RGB cache training schedule mismatch")
        if require_complete and not self.complete_schedule_coverage:
            raise ValueError("training requires a complete RGB cache")
        coverage_scope = str(
            self.manifest.get("coverage_scope", RGB_CACHE_COVERAGE_FULL)
        )
        if coverage_scope == RGB_CACHE_COVERAGE_CAUSAL_EVENT:
            if event_ledger_manifest is None:
                raise ValueError("causal-event RGB cache requires its event ledger")
            if self.manifest.get("event_ledger_manifest_sha256") != sha256_file(
                event_ledger_manifest
            ):
                raise ValueError("causal-event RGB cache ledger binding mismatch")
        elif coverage_scope != RGB_CACHE_COVERAGE_FULL:
            raise ValueError("temporal RGB cache has an unknown coverage scope")

    @staticmethod
    def _close_array(array: np.ndarray) -> None:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def _open_pack(self, key: tuple[str, str]) -> tuple[np.ndarray, np.ndarray] | None:
        opened = self._open.get(key)
        if opened is not None:
            self._open.move_to_end(key)
            return opened
        metadata = self._pack_by_key.get(key)
        if metadata is None:
            return None
        offsets_path = self.root / "packs" / str(metadata["offsets_file"])
        rgb_path = self.root / "packs" / str(metadata["rgb_file"])
        offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        frames = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
        expected = (int(metadata["frame_count"]), *RGB_FRAME_SHAPE)
        if offsets.shape != (expected[0],) or frames.shape != expected:
            self._close_array(offsets)
            self._close_array(frames)
            raise ValueError("temporal RGB cache pack shape mismatch")
        self._open[key] = (offsets, frames)
        self.pack_opens += 1
        while len(self._open) > self.max_open_packs:
            _, (old_offsets, old_frames) = self._open.popitem(last=False)
            self._close_array(old_offsets)
            self._close_array(old_frames)
            self.pack_evictions += 1
        return offsets, frames

    def get_many(
        self,
        canonical_episode_id: str,
        view: str,
        offsets: Iterable[int],
    ) -> dict[int, np.ndarray]:
        requested = np.asarray(sorted(set(map(int, offsets))), dtype=np.int64)
        self.frame_requests += int(requested.size)
        if not requested.size:
            return {}
        pack = self._open_pack((str(canonical_episode_id), str(view)))
        if pack is None:
            self.frame_misses += int(requested.size)
            return {}
        cached_offsets, frames = pack
        positions = np.searchsorted(cached_offsets, requested)
        valid = positions < len(cached_offsets)
        matched = np.zeros(len(requested), dtype=bool)
        matched[valid] = cached_offsets[positions[valid]] == requested[valid]
        hit_offsets = requested[matched]
        hit_frames = np.asarray(frames[positions[matched]])
        hits = int(matched.sum())
        self.frame_hits += hits
        self.frame_misses += int(requested.size) - hits
        self.bytes_read += hits * RGB_FRAME_BYTES
        return {
            int(offset): frame
            for offset, frame in zip(hit_offsets, hit_frames, strict=True)
        }

    def stats(self) -> RGBDiskCacheStats:
        return RGBDiskCacheStats(
            frame_requests=self.frame_requests,
            frame_hits=self.frame_hits,
            frame_misses=self.frame_misses,
            bytes_read=self.bytes_read,
            pack_opens=self.pack_opens,
            pack_evictions=self.pack_evictions,
            open_packs=len(self._open),
        )

    def close(self) -> None:
        while self._open:
            _, (offsets, frames) = self._open.popitem(last=False)
            self._close_array(offsets)
            self._close_array(frames)
