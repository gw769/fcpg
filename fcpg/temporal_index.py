"""Episode-block reader and sampler for the immutable temporal index."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Container, Iterable

import numpy as np
import torch

from fcpg.lerobot_archive import decode_episode_views
from fcpg.temporal_cache import HostByteLRU
from fcpg.temporal_grounding_protocol import TEMPORAL_PROTOCOL_ID, select_roleless_views


_DECODE_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _initialize_decode_worker() -> None:
    """Keep each spawned codec process inside its CPU thread budget."""

    for name in _DECODE_THREAD_ENV:
        os.environ[name] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:
        pass


def _decode_worker_ping(value: int) -> tuple[int, int]:
    """Force a spawned worker to exist before the parent initializes CUDA."""

    import os
    import time

    time.sleep(0.1)
    return int(value), os.getpid()


def _decode_episode_view_job(
    payload: tuple[dict, str, tuple[int, ...]],
) -> np.ndarray:
    """Decode one independent episode/view block without nested codec pools."""

    row, view, missing = payload
    return decode_episode_views(
        row,
        (view,),
        missing,
        decoder_threads=1,
    )[view]


@dataclass(frozen=True)
class TemporalTrainingBatch:
    images: torch.Tensor
    states: torch.Tensor
    completed_actions: torch.Tensor
    motion_signatures: torch.Tensor
    view_valid: torch.Tensor
    source_index: torch.Tensor
    task_index: torch.Tensor
    motion_cluster: torch.Tensor
    high_motion: torch.Tensor
    frame_keys: tuple[tuple[tuple[str, int, str] | None, ...], ...]
    part_motion_signatures: torch.Tensor | None = None


class TemporalIndex:
    """Read-only mmap index; archived RGB is decoded only for requested blocks."""

    REQUIRED_ARRAYS = (
        "states",
        "actions",
        "motion_signatures",
        "episode_offsets",
        "window_episode",
        "window_image_indices",
        "window_future_indices",
        "window_future_history_indices",
        "window_future_action_indices",
        "window_source_index",
        "window_task_index",
        "window_split_index",
    )

    def __init__(
        self,
        root: Path,
        *,
        cluster_artifact: Path | None = None,
        host_lru: HostByteLRU[np.ndarray] | None = None,
        decode_workers: int = 1,
        rgb_cache: Path | object | None = None,
        part_motion_cache: Path | object | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("protocol_id") != TEMPORAL_PROTOCOL_ID:
            raise ValueError("temporal index protocol mismatch")
        self.arrays = {
            name: np.load(self.root / f"{name}.npy", mmap_mode="r")
            for name in self.REQUIRED_ARRAYS
        }
        self.episodes = list(self.manifest["episodes"])
        self.host_lru = host_lru or HostByteLRU()
        if rgb_cache is None:
            self.rgb_cache = None
        elif isinstance(rgb_cache, (str, Path)):
            from fcpg.temporal_rgb_cache import TemporalRGBDiskCache

            self.rgb_cache = TemporalRGBDiskCache(Path(rgb_cache))
        else:
            self.rgb_cache = rgb_cache
        if part_motion_cache is None:
            self.part_motion_cache = None
        elif isinstance(part_motion_cache, (str, Path)):
            from fcpg.temporal_part_motion_cache import TemporalPartMotionDiskCache

            self.part_motion_cache = TemporalPartMotionDiskCache(Path(part_motion_cache))
        else:
            self.part_motion_cache = part_motion_cache
        if decode_workers < 0 or (decode_workers == 0 and self.rgb_cache is None):
            raise ValueError("zero temporal decode workers requires an RGB disk cache")
        self.decode_workers = int(decode_workers)
        self._decode_pool: ProcessPoolExecutor | None = None
        self.decode_worker_pids: tuple[int, ...] = ()
        if cluster_artifact is None:
            self.motion_cluster = np.full(len(self), -1, dtype=np.int16)
            self.high_motion = np.ones(len(self), dtype=bool)
        else:
            cluster_root = Path(cluster_artifact).expanduser().resolve()
            payload = json.loads((cluster_root / "manifest.json").read_text())
            if payload.get("protocol_id") != TEMPORAL_PROTOCOL_ID:
                raise ValueError("motion-cluster protocol mismatch")
            self.motion_cluster = np.load(cluster_root / "window_motion_cluster.npy", mmap_mode="r")
            self.high_motion = np.load(cluster_root / "window_high_motion.npy", mmap_mode="r")
            if len(self.motion_cluster) != len(self) or len(self.high_motion) != len(self):
                raise ValueError("motion cluster arrays do not align with temporal windows")

    def __len__(self) -> int:
        return len(self.arrays["window_episode"])

    def start_decode_pool(self) -> tuple[int, ...]:
        """Spawn persistent codec workers before any CUDA context is created."""

        if self.decode_workers <= 1:
            return ()
        if self._decode_pool is not None:
            return self.decode_worker_pids
        previous = {name: os.environ.get(name) for name in _DECODE_THREAD_ENV}
        for name in _DECODE_THREAD_ENV:
            os.environ[name] = "1"
        try:
            self._decode_pool = ProcessPoolExecutor(
                max_workers=self.decode_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_initialize_decode_worker,
            )
            futures = [
                self._decode_pool.submit(_decode_worker_ping, index)
                for index in range(self.decode_workers)
            ]
            for future in futures:
                future.result()
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        # Waiting for every submitted ping guarantees the executor manager has
        # completed startup. A fast worker may consume more than one ping, so
        # unique *result* PIDs are not a valid worker-count check.
        processes = getattr(self._decode_pool, "_processes", {})
        self.decode_worker_pids = tuple(
            sorted(
                int(process.pid)
                for process in processes.values()
                if process.pid is not None and process.is_alive()
            )
        )
        if len(self.decode_worker_pids) != self.decode_workers:
            self.close_decode_pool()
            raise RuntimeError(
                "temporal decode process pool did not start every requested worker"
            )
        return self.decode_worker_pids

    def close_decode_pool(self) -> None:
        if self._decode_pool is not None:
            self._decode_pool.shutdown(wait=True, cancel_futures=True)
            self._decode_pool = None

    def close(self) -> None:
        self.close_decode_pool()
        if self.rgb_cache is not None:
            close = getattr(self.rgb_cache, "close", None)
            if close is not None:
                close()
        if self.part_motion_cache is not None:
            close = getattr(self.part_motion_cache, "close", None)
            if close is not None:
                close()

    def _episode_window(self, window_index: int) -> tuple[int, dict, np.ndarray]:
        if not 0 <= window_index < len(self):
            raise IndexError("temporal window index outside artifact")
        episode_index = int(self.arrays["window_episode"][window_index])
        record = self.episodes[episode_index]
        return episode_index, record, np.asarray(
            self.arrays["window_image_indices"][window_index], dtype=np.int64
        )

    def load_batch(
        self,
        indices: Iterable[int],
        *,
        max_views_per_window: int | None = None,
        view_selection_seed: int = 0,
        view_selection_seeds: Iterable[int] | None = None,
        image_endpoint_indices: Iterable[int] | None = None,
        cached_image_keys: Container[tuple[str, int, str]] | None = None,
    ) -> TemporalTrainingBatch:
        selected = [int(value) for value in indices]
        if not selected:
            raise ValueError("temporal batch cannot be empty")
        if view_selection_seeds is None:
            selection_seeds = [int(view_selection_seed)] * len(selected)
        else:
            selection_seeds = [int(value) for value in view_selection_seeds]
            if len(selection_seeds) != len(selected):
                raise ValueError("per-row view-selection seeds must match the batch")
        if image_endpoint_indices is None:
            image_endpoints = [15] * len(selected)
        else:
            image_endpoints = [int(value) for value in image_endpoint_indices]
            if len(image_endpoints) != len(selected):
                raise ValueError("per-row image endpoints must match the batch")
            if any(endpoint < 0 or endpoint > 15 for endpoint in image_endpoints):
                raise ValueError("image endpoint must lie in [0,15]")
        records = [self._episode_window(index) for index in selected]
        selected_views: list[tuple[str, ...]] = []
        for (_, record, _), selection_seed in zip(
            records, selection_seeds, strict=True
        ):
            selected_views.append(
                select_roleless_views(
                    record,
                    maximum_views=max_views_per_window,
                    selection_seed=selection_seed,
                )
            )
        maximum_views = max(map(len, selected_views))
        batch = len(selected)
        images = np.zeros((batch, 16, maximum_views, 3, 224, 224), dtype=np.uint8)
        states = np.empty((batch, 16, 6), dtype=np.float32)
        completed = np.empty((batch, 15, 6), dtype=np.float32)
        signatures = np.empty((batch, 16, 64), dtype=np.float16)
        view_valid = np.zeros((batch, maximum_views), dtype=bool)
        frame_keys: list[tuple[tuple[str, int, str] | None, ...]] = []
        global_frame_indices = np.empty((batch, 16), dtype=np.int64)
        requested: dict[tuple[int, str], set[int]] = {}
        gpu_cached = cached_image_keys if cached_image_keys is not None else ()
        for (episode_index, record, local_indices), views, endpoint in zip(
            records, selected_views, image_endpoints, strict=True
        ):
            episode_id = str(record["canonical_episode_id"])
            for view in views:
                missing_offsets = (
                    int(offset)
                    for offset in local_indices[: endpoint + 1]
                    if (episode_id, int(offset), str(view)) not in gpu_cached
                )
                requested.setdefault((episode_index, str(view)), set()).update(
                    missing_offsets
                )
        decoded_frames: dict[tuple[str, int, str], np.ndarray] = {}
        decode_jobs: list[tuple[int, str, list[int]]] = []
        for (episode_index, view), offsets in sorted(requested.items()):
            record = self.episodes[episode_index]
            episode_id = str(record["canonical_episode_id"])
            missing: list[int] = []
            for offset in sorted(offsets):
                key = (episode_id, offset, view)
                cached = self.host_lru.get(key)
                if cached is None:
                    missing.append(offset)
                else:
                    decoded_frames[key] = cached
            if missing and self.rgb_cache is not None:
                disk_frames = self.rgb_cache.get_many(episode_id, view, missing)
                remaining: list[int] = []
                for offset in missing:
                    frame = disk_frames.get(offset)
                    if frame is None:
                        remaining.append(offset)
                        continue
                    key = (episode_id, offset, view)
                    decoded_frames[key] = frame
                    self.host_lru.put(key, frame, bytes_used=frame.nbytes)
                missing = remaining
            if missing:
                decode_jobs.append((episode_index, view, missing))

        if decode_jobs:
            if self.decode_workers == 0:
                missing_frames = sum(len(job[2]) for job in decode_jobs)
                raise KeyError(
                    f"complete RGB cache missed {missing_frames} requested frames"
                )
            payloads = [
                (
                    self.episodes[episode_index]["manifest_row"],
                    view,
                    tuple(missing),
                )
                for episode_index, view, missing in decode_jobs
            ]
            if self._decode_pool is None:
                blocks = [_decode_episode_view_job(payload) for payload in payloads]
            else:
                blocks = list(self._decode_pool.map(_decode_episode_view_job, payloads))
            # LRU mutation remains in deterministic parent-process order.
            for (episode_index, view, missing), block in zip(
                decode_jobs, blocks, strict=True
            ):
                episode_id = str(self.episodes[episode_index]["canonical_episode_id"])
                for offset, frame in zip(missing, block, strict=True):
                    key = (episode_id, offset, view)
                    decoded_frames[key] = frame
                    self.host_lru.put(key, frame, bytes_used=frame.nbytes)
        for batch_index, ((_, record, local_indices), views, endpoint) in enumerate(
            zip(records, selected_views, image_endpoints, strict=True)
        ):
            row = record["manifest_row"]
            episode_id = str(record["canonical_episode_id"])
            global_start = int(record["frame_start"])
            global_indices = global_start + local_indices
            global_frame_indices[batch_index] = global_indices
            states[batch_index] = self.arrays["states"][global_indices]
            completed[batch_index] = self.arrays["actions"][global_indices[:-1]]
            signatures[batch_index] = self.arrays["motion_signatures"][global_indices]
            view_valid[batch_index, : len(views)] = True
            keys_for_window: list[tuple[str, int, str] | None] = []
            for time_index, offset in enumerate(local_indices):
                visible_offset = int(
                    offset if time_index <= endpoint else local_indices[endpoint]
                )
                for view_index in range(maximum_views):
                    if view_index >= len(views):
                        keys_for_window.append(None)
                        continue
                    view = views[view_index]
                    key = (episode_id, visible_offset, view)
                    frame = decoded_frames.get(key)
                    if frame is not None:
                        images[batch_index, time_index, view_index] = frame.transpose(
                            2, 0, 1
                        )
                    elif key not in gpu_cached:
                        raise KeyError("temporal image row is absent from every cache")
                    keys_for_window.append(key)
            frame_keys.append(tuple(keys_for_window))
        selected_array = np.asarray(selected, dtype=np.int64)
        part_motion = (
            None
            if self.part_motion_cache is None
            else torch.from_numpy(
                self.part_motion_cache.get_many(global_frame_indices)
            )
        )
        return TemporalTrainingBatch(
            images=torch.from_numpy(images),
            states=torch.from_numpy(states),
            completed_actions=torch.from_numpy(completed),
            motion_signatures=torch.from_numpy(signatures),
            view_valid=torch.from_numpy(view_valid),
            source_index=torch.from_numpy(np.asarray(self.arrays["window_source_index"][selected_array]).copy()),
            task_index=torch.from_numpy(np.asarray(self.arrays["window_task_index"][selected_array]).copy()),
            motion_cluster=torch.from_numpy(np.asarray(self.motion_cluster[selected_array]).copy()),
            high_motion=torch.from_numpy(np.asarray(self.high_motion[selected_array]).copy()),
            frame_keys=tuple(frame_keys),
            part_motion_signatures=part_motion,
        )


class EpisodeBlockSampler:
    """Source/task-balanced sampling whose units are contiguous episode blocks."""

    def __init__(
        self,
        index: TemporalIndex,
        *,
        seed: int,
        per_source_cap: int = 100_000,
        block_size: int = 8,
        cluster_pair_rows: int = 12,
        require_cluster_pairs: bool = False,
    ) -> None:
        if per_source_cap <= 0 or block_size <= 0 or cluster_pair_rows < 0:
            raise ValueError("sampler caps and block size must be positive")
        if cluster_pair_rows % 2:
            raise ValueError("cluster_pair_rows must be even")
        self.index = index
        self.rng = np.random.default_rng(seed)
        self.block_size = int(block_size)
        self.cluster_pair_rows = int(cluster_pair_rows)
        self.cluster_pair_rows_sampled = 0
        by_group_episode: dict[tuple[int, int], dict[int, list[int]]] = {}
        by_motion_group_episode: dict[bool, dict[tuple[int, int], dict[int, list[int]]]] = {
            False: {},
            True: {},
        }
        cluster_groups: dict[int, dict[tuple[int, int], list[int]]] = {}
        source_count: dict[int, int] = {}
        order = self.rng.permutation(len(index))
        for raw in order:
            window = int(raw)
            if int(index.arrays["window_split_index"][window]) != 0:
                continue
            source = int(index.arrays["window_source_index"][window])
            if source_count.get(source, 0) >= per_source_cap:
                continue
            source_count[source] = source_count.get(source, 0) + 1
            task = int(index.arrays["window_task_index"][window])
            episode = int(index.arrays["window_episode"][window])
            by_group_episode.setdefault((source, task), {}).setdefault(episode, []).append(window)
            high = bool(index.high_motion[window])
            by_motion_group_episode[high].setdefault((source, task), {}).setdefault(
                episode, []
            ).append(window)
            cluster = int(index.motion_cluster[window])
            if high and cluster >= 0:
                groups = cluster_groups.setdefault(cluster, {})
                group = (source, task)
                if group in groups:
                    if len(groups[group]) < 2:
                        groups[group].append(window)
                elif len(groups) < 32:
                    groups[group] = [window]
        self.groups = {
            group: {
                episode: np.asarray(sorted(windows), dtype=np.int64)
                for episode, windows in episodes.items()
            }
            for group, episodes in by_group_episode.items()
        }
        if not self.groups:
            raise ValueError("episode-block sampler has no groups")
        self.group_keys = tuple(sorted(self.groups))
        self.motion_groups = {
            high: {
                group: {
                    episode: np.asarray(sorted(windows), dtype=np.int64)
                    for episode, windows in episodes.items()
                }
                for group, episodes in grouped.items()
            }
            for high, grouped in by_motion_group_episode.items()
        }
        self._group_order: dict[bool, tuple[tuple[int, int], ...]] = {}
        self._group_cursor = {False: 0, True: 0}
        self._episode_order: dict[
            tuple[bool, tuple[int, int]], tuple[int, ...]
        ] = {}
        self._episode_cursor: dict[tuple[bool, tuple[int, int]], int] = {}
        self._window_cursor: dict[tuple[bool, tuple[int, int], int], int] = {}
        for high, grouped in self.motion_groups.items():
            keys = tuple(sorted(grouped))
            if keys:
                permutation = self.rng.permutation(len(keys))
                self._group_order[high] = tuple(keys[int(index)] for index in permutation)
            for group, episodes in grouped.items():
                episode_ids = tuple(sorted(episodes))
                permutation = self.rng.permutation(len(episode_ids))
                key = (high, group)
                self._episode_order[key] = tuple(
                    episode_ids[int(index)] for index in permutation
                )
                self._episode_cursor[key] = 0
        self._cluster_groups = {
            cluster: groups for cluster, groups in cluster_groups.items() if len(groups) >= 2
        }
        cluster_ids = tuple(sorted(self._cluster_groups))
        if cluster_ids:
            permutation = self.rng.permutation(len(cluster_ids))
            self._cluster_order = tuple(cluster_ids[int(index)] for index in permutation)
        else:
            self._cluster_order = ()
        if require_cluster_pairs and self.cluster_pair_rows and not self._cluster_order:
            raise ValueError("formal sensorimotor NCE requires cross-group motion clusters")
        self._cluster_cursor = 0
        self._cluster_group_cursor: dict[int, int] = {}
        self._cluster_window_cursor: dict[tuple[int, tuple[int, int]], int] = {}

    def _block_containing_cluster_row(
        self,
        *,
        cluster: int,
        group: tuple[int, int],
    ) -> list[int]:
        candidates = self._cluster_groups[cluster][group]
        key = (cluster, group)
        candidate_cursor = self._cluster_window_cursor.get(key, 0)
        candidate = int(candidates[candidate_cursor % len(candidates)])
        self._cluster_window_cursor[key] = candidate_cursor + 1
        episode = int(self.index.arrays["window_episode"][candidate])
        windows = self.motion_groups[True][group][episode]
        position = int(np.searchsorted(windows, candidate))
        if position >= len(windows) or int(windows[position]) != candidate:
            raise AssertionError("motion-cluster candidate left its source/task block")
        if len(windows) >= self.block_size:
            start = min(position, len(windows) - self.block_size)
            return list(map(int, windows[start : start + self.block_size]))
        # Exhaust every distinct row in a short episode before repeating.
        return list(map(int, np.resize(windows, self.block_size)))

    def _sample_cluster_pair_blocks(self, rows: int) -> list[int]:
        if rows == 0 or not self._cluster_order:
            return []
        result: list[int] = []
        for _ in range(rows // 2):
            cluster = self._cluster_order[self._cluster_cursor % len(self._cluster_order)]
            self._cluster_cursor += 1
            groups = tuple(sorted(self._cluster_groups[cluster]))
            cursor = self._cluster_group_cursor.get(cluster, 0)
            selected_groups = (groups[cursor % len(groups)], groups[(cursor + 1) % len(groups)])
            self._cluster_group_cursor[cluster] = cursor + 2
            for group in selected_groups:
                result.extend(
                    self._block_containing_cluster_row(cluster=cluster, group=group)
                )
        self.cluster_pair_rows_sampled += rows
        return result

    def _next_group_block(
        self,
        *,
        high: bool,
        group: tuple[int, int],
        take: int,
    ) -> list[int]:
        """Advance a persistent episode/window cursor for maximum frame reuse."""

        result: list[int] = []
        key = (high, group)
        episode_order = self._episode_order[key]
        while len(result) < take:
            episode_cursor = self._episode_cursor[key]
            episode = episode_order[episode_cursor]
            windows = self.motion_groups[high][group][episode]
            window_key = (high, group, episode)
            start = self._window_cursor.get(window_key, 0)
            remaining = take - len(result)
            stop = min(len(windows), start + remaining)
            result.extend(map(int, windows[start:stop]))
            if stop == len(windows):
                self._window_cursor[window_key] = 0
                self._episode_cursor[key] = (episode_cursor + 1) % len(episode_order)
            else:
                self._window_cursor[window_key] = stop
        return result

    def _sample(self, high: bool, batch_size: int) -> list[int]:
        if batch_size == 0:
            return []
        groups = self.motion_groups[high]
        if not groups:
            raise ValueError("requested motion stratum has no windows")
        result: list[int] = []
        group_order = self._group_order[high]
        while len(result) < batch_size:
            cursor = self._group_cursor[high]
            group = group_order[cursor % len(group_order)]
            self._group_cursor[high] = cursor + 1
            take = min(self.block_size, batch_size - len(result))
            result.extend(self._next_group_block(high=high, group=group, take=take))
        return result

    def batch(self, batch_size: int, *, low_motion_fraction: float = 0.25) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if not 0.0 <= low_motion_fraction <= 1.0:
            raise ValueError("low_motion_fraction must lie in [0,1]")
        low_count = int(round(batch_size * low_motion_fraction))
        high_count = batch_size - low_count
        available_blocks = high_count // self.block_size
        pair_count = min(self.cluster_pair_rows, available_blocks)
        pair_count -= pair_count % 2
        if not self._cluster_order:
            pair_count = 0
        paired_window_count = pair_count * self.block_size
        result = self._sample(True, high_count - paired_window_count)
        result.extend(self._sample_cluster_pair_blocks(pair_count))
        result.extend(self._sample(False, low_count))
        return np.asarray(result, dtype=np.int64)

    def stats(self) -> dict[str, int]:
        return {
            "block_size": self.block_size,
            "configured_cluster_pair_anchor_rows_per_batch": self.cluster_pair_rows,
            "valid_cross_group_motion_clusters": len(self._cluster_order),
            "cluster_pair_rows_sampled": self.cluster_pair_rows_sampled,
        }
