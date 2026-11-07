"""Raw-image, episode-safe training data for the unified V10 model."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from fcpg.so101_real_image_drift import (
    SO101ImageAffineDrift,
    points_in_bounds,
    transform_points,
    warp_rgb,
)
from fcpg.so101_synthetic_dataset import (
    DATASET_FORMAT,
    LEGACY_DATASET_FORMAT,
    SO101SyntheticFCPGDataset,
)
from v10.config import FUTURE_OFFSETS, HISTORY_OFFSETS, V10Config
from v10.task_class import parse_task_class


JOINT_SCALE = 100.0
PROJECTION_FORMAT = "so101-v10-projection-v1"
WRIST_CAMERA_ROLE = "wrist"
EXTERNAL_CAMERA_ROLE = "external"
_CAMERA_ROLE_TO_VIEW = {
    WRIST_CAMERA_ROLE: 0,
    EXTERNAL_CAMERA_ROLE: 1,
}
_TRAJECTORY_CACHE_FILES = (
    "manifest.json",
    "frames.npy",
    "states.npy",
    "actions.npy",
    "episodes.npy",
    "tasks.npy",
    "frame_indices.npy",
)
CAMERA_CONTEXT = np.asarray(
    ((1.0, 0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0, 0.0)),
    dtype=np.float32,
)


def _frame_cache_bytes(explicit: int | None) -> int:
    if explicit is not None:
        return max(0, int(explicit))
    return max(0, int(os.environ.get("V10_FRAME_CACHE_MB", "1024"))) * 1024 * 1024


class _FrameSliceLRU:
    """Per-process decoded-frame LRU so mmap faults are not paid twice.

    DataLoader workers fork after the dataset is built, so each worker gets
    its own empty store. Keep the per-worker budget small; a shared 20 GB
    copy in every worker would OOM.
    """

    def __init__(self, budget_bytes: int) -> None:
        self.budget_bytes = max(0, int(budget_bytes))
        self._store: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._used = 0
        self.hits = 0
        self.misses = 0

    def gather(
        self, cache_id: int, frames: np.ndarray, rows: np.ndarray
    ) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        if self.budget_bytes <= 0 or len(rows) == 0:
            self.misses += int(len(rows))
            return np.asarray(frames[rows], dtype=np.uint8)
        out = np.empty((len(rows),) + frames.shape[1:], dtype=np.uint8)
        for index, row in enumerate(rows):
            key = (cache_id, int(row))
            cached = self._store.get(key)
            if cached is not None:
                self._store.move_to_end(key)
                out[index] = cached
                self.hits += 1
                continue
            frame = np.array(frames[int(row)], copy=True, order="C")
            self.misses += 1
            nbytes = int(frame.nbytes)
            while self._store and self._used + nbytes > self.budget_bytes:
                _, old = self._store.popitem(last=False)
                self._used -= int(old.nbytes)
            if nbytes <= self.budget_bytes:
                self._store[key] = frame
                self._used += nbytes
            out[index] = frame
        return out


class TrajectoryCache:
    """Memory-map one decoded LeRobot trajectory cache."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != "so101-lewm-cache-v1":
            raise ValueError(f"unsupported trajectory cache: {self.root}")
        self.frames = np.load(self.root / "frames.npy", mmap_mode="r")
        self.states = np.load(self.root / "states.npy", mmap_mode="r")
        self.actions = np.load(self.root / "actions.npy", mmap_mode="r")
        self.episodes = np.load(self.root / "episodes.npy", mmap_mode="r")
        self.tasks = np.load(self.root / "tasks.npy", mmap_mode="r")
        self.frame_indices = np.load(self.root / "frame_indices.npy", mmap_mode="r")
        rows = int(self.manifest["frames"])
        arrays = (
            self.frames,
            self.states,
            self.actions,
            self.episodes,
            self.tasks,
            self.frame_indices,
        )
        if any(len(array) != rows for array in arrays):
            raise ValueError(f"trajectory arrays disagree: {self.root}")
        if self.frames.shape[1:] != (224, 224, 3) or self.frames.dtype != np.uint8:
            raise ValueError("V10 requires uint8 224x224 RGB caches")
        if self.states.shape[1:] != (6,) or self.actions.shape[1:] != (6,):
            raise ValueError("SO-101 states and actions must have six joints")
        self.dataset_id = str(
            self.manifest.get("dataset_id", self.manifest.get("name", self.root.name))
        )
        self.name = str(self.manifest.get("name", self.root.name))


class GeometryCache:
    """Read only deployable FK anchor geometry from an older context cache."""

    def __init__(self, root: str | Path, rows: int) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest = json.loads((self.root / "manifest.json").read_text())
        if int(manifest.get("rows", -1)) != rows:
            raise ValueError("geometry and trajectory row counts differ")
        self.anchors_xyz = np.load(self.root / "anchors_xyz.npy", mmap_mode="r")
        if self.anchors_xyz.shape != (rows, 3, 3):
            raise ValueError("anchors_xyz must have shape [rows,3,3]")


class RealGroundingLookup:
    """Recover clean wrist/column anchor pixels for the local real trajectory."""

    def __init__(self, root: str | Path, rows: int) -> None:
        root_path = Path(root).expanduser().resolve()
        manifest = json.loads((root_path / "manifest.json").read_text())
        dataset_format = manifest.get("format")
        legacy_true_only = (
            dataset_format == LEGACY_DATASET_FORMAT
            and not (root_path / "q_nominal.npy").is_file()
        )
        if legacy_true_only:
            arrays = self._load_legacy_true_only(root_path, manifest)
        else:
            dataset = SO101SyntheticFCPGDataset(root_path)
            arrays = dataset.arrays
        if (
            int(manifest["width"]) != 224
            or int(manifest["height"]) != 224
        ):
            raise ValueError("real grounding must use 224x224 image coordinates")
        required = ("source_row", "camera_index", "drift", "q_true", "valid")
        if any(name not in arrays for name in required):
            raise ValueError("real grounding data lacks clean partner identity")

        has_nominal = "q_nominal" in arrays
        if dataset_format == DATASET_FORMAT:
            if "drift_variant" not in arrays or not has_nominal:
                raise ValueError("v2 real grounding lacks nominal/true separation")
            clean = np.flatnonzero(np.asarray(arrays["drift_variant"]) == 0)
            if not self._is_clean_identity(arrays, clean):
                raise ValueError(
                    "v2 drift_variant=0 rows must be clean identity grounding"
                )
        elif dataset_format == LEGACY_DATASET_FORMAT:
            all_rows = np.arange(len(arrays["q_true"]), dtype=np.int64)
            if not self._is_clean_identity(arrays, all_rows):
                raise ValueError(
                    "legacy real grounding is supported only for clean identity data"
                )
            clean = all_rows
        else:  # SO101SyntheticFCPGDataset rejects this before exposing arrays.
            raise ValueError(f"unsupported real grounding format: {dataset_format}")

        self.q_nominal = np.zeros((rows, 2, 3, 2), dtype=np.float32)
        self.valid_nominal = np.zeros((rows, 2, 3), dtype=bool)
        self.q_true = np.zeros((rows, 2, 3, 2), dtype=np.float32)
        self.valid_true = np.zeros((rows, 2, 3), dtype=bool)
        seen = np.zeros((rows, 2), dtype=bool)
        for index in clean:
            row = int(arrays["source_row"][index])
            view = int(arrays["camera_index"][index])
            if not 0 <= row < rows or view not in (0, 1):
                raise ValueError("clean grounding identity is outside the trajectory")
            if seen[row, view]:
                raise ValueError("duplicate clean grounding row")
            self.q_nominal[row, view] = np.asarray(
                arrays["q_nominal" if has_nominal else "q_true"][index],
                dtype=np.float32,
            )
            self.q_true[row, view] = np.asarray(
                arrays["q_true"][index], dtype=np.float32
            )
            valid = np.asarray(arrays["valid"][index], dtype=bool)
            self.valid_nominal[row, view] = valid
            self.valid_true[row, view] = valid & points_in_bounds(
                self.q_true[row, view], 224, 224
            )
            seen[row, view] = True
        if not bool(seen.all()):
            raise ValueError("clean grounding does not cover every local frame and view")

        # Preserve callers that still use the original clean-grounding API.
        self.q = self.q_nominal
        self.valid = self.valid_nominal
        self.target_fallback_to_nominal = False

    @staticmethod
    def _load_legacy_true_only(
        root: Path, manifest: dict[str, object]
    ) -> dict[str, np.ndarray]:
        names = ("source_row", "camera_index", "drift", "q_true", "valid")
        arrays = {
            name: np.load(root / f"{name}.npy", mmap_mode="r") for name in names
        }
        samples = int(manifest["samples"])
        if any(len(value) != samples for value in arrays.values()):
            raise ValueError("legacy real grounding arrays disagree with manifest")
        if arrays["q_true"].shape != (samples, 3, 2):
            raise ValueError("legacy q_true must have shape [samples,3,2]")
        if arrays["valid"].shape != (samples, 3) or arrays["valid"].dtype != np.bool_:
            raise ValueError("legacy valid must have bool shape [samples,3]")
        if arrays["drift"].shape != (samples, 3):
            raise ValueError("legacy drift must have shape [samples,3]")
        return arrays

    @staticmethod
    def _is_clean_identity(
        arrays: dict[str, np.ndarray], indices: np.ndarray
    ) -> bool:
        if not len(indices):
            return False
        drift = np.asarray(arrays["drift"][indices])
        true = np.asarray(arrays["q_true"][indices])
        nominal = (
            None
            if "q_nominal" not in arrays
            else np.asarray(arrays["q_nominal"][indices])
        )
        return bool(
            np.all(np.isfinite(true))
            and np.allclose(drift, 0.0, rtol=0.0, atol=1.0e-7)
            and (
                nominal is None
                or (
                    np.all(np.isfinite(nominal))
                    and np.allclose(nominal, true, rtol=0.0, atol=1.0e-5)
                )
            )
        )


class V10ProjectionLookup:
    """Load nominal history projections and optional true future targets."""

    def __init__(self, root: str | Path, rows: int) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest.get("format") != PROJECTION_FORMAT:
            raise ValueError(f"unsupported V10 projection cache: {self.root}")
        if int(manifest.get("rows", -1)) != rows:
            raise ValueError("projection and trajectory row counts differ")
        self.q_nominal = np.load(self.root / "q_nominal_px.npy", mmap_mode="r")
        valid_nominal_path = self.root / "valid_nominal.npy"
        if not valid_nominal_path.is_file():
            valid_nominal_path = self.root / "valid.npy"
        self.valid_nominal = np.load(valid_nominal_path, mmap_mode="r")
        self._validate_points(self.q_nominal, rows=rows, name="q_nominal_px")
        self._validate_validity(
            self.valid_nominal, rows=rows, name=valid_nominal_path.stem
        )

        q_true_path = self.root / "q_true_px.npy"
        valid_true_path = self.root / "valid_true.npy"
        has_q_true = q_true_path.is_file()
        has_valid_true = valid_true_path.is_file()
        if has_q_true != has_valid_true:
            raise ValueError(
                "projection target requires both q_true_px.npy and valid_true.npy"
            )
        self.target_fallback_to_nominal = not has_q_true
        if self.target_fallback_to_nominal:
            # Keep legacy nominal geometry available for history, but never
            # turn it into pixel supervision for a future target.
            self.q_true = np.full(self.q_nominal.shape, 112.0, dtype=np.float32)
            self.valid_true = np.zeros(self.valid_nominal.shape, dtype=bool)
        else:
            self.q_true = np.load(q_true_path, mmap_mode="r")
            raw_valid_true = np.load(valid_true_path, mmap_mode="r")
            self._validate_points(self.q_true, rows=rows, name="q_true_px")
            self._validate_validity(raw_valid_true, rows=rows, name="valid_true")
            self.valid_true = np.asarray(raw_valid_true) & points_in_bounds(
                self.q_true, 224, 224
            )

        # Preserve callers that still use the v1 nominal-only lookup API.
        self.q = self.q_nominal
        self.valid = self.valid_nominal

    @staticmethod
    def _validate_points(value: np.ndarray, *, rows: int, name: str) -> None:
        if value.shape != (rows, 2, 3, 2):
            raise ValueError(f"{name} must have shape [rows,2,3,2]")
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"{name} must use a floating dtype")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite")

    @staticmethod
    def _validate_validity(value: np.ndarray, *, rows: int, name: str) -> None:
        if value.shape != (rows, 2, 3):
            raise ValueError(f"{name} must have shape [rows,2,3]")
        if value.dtype != np.bool_:
            raise ValueError(f"{name} must use bool dtype")


@dataclass(frozen=True)
class V10SourceSpec:
    trajectory: Path
    geometry: Path
    secondary_trajectory: Path | None = None
    real_grounding: Path | None = None
    projection: Path | None = None
    primary_view: int = 1
    validation_episodes: tuple[int, ...] = ()
    excluded_episodes: tuple[int, ...] = ()
    primary_camera_role: str | None = None
    secondary_camera_role: str | None = None
    task_class: int | None = None


@dataclass
class _LoadedSource:
    primary: TrajectoryCache
    secondary: TrajectoryCache | None
    geometry: GeometryCache
    grounding: RealGroundingLookup | V10ProjectionLookup | None
    primary_view: int
    secondary_view: int | None
    validation_episodes: frozenset[int]
    task_class: int | None = None


def _validation_episodes(
    trajectory: TrajectoryCache,
    *,
    split_seed: int,
    val_fraction: float,
) -> frozenset[int]:
    result: set[int] = set()
    episodes = np.asarray(trajectory.episodes)
    tasks = np.asarray(trajectory.tasks)
    for raw_task in np.unique(tasks):
        task = int(raw_task)
        candidates = [int(value) for value in np.unique(episodes[tasks == task])]
        if len(candidates) < 2:
            raise ValueError(
                f"task {trajectory.dataset_id}:{task} has fewer than two episodes"
            )
        ordered = sorted(
            candidates,
            key=lambda episode: hashlib.sha256(
                f"{split_seed}:{trajectory.dataset_id}:{task}:{episode}".encode()
            ).digest(),
        )
        count = max(1, int(round(len(ordered) * val_fraction)))
        result.update(ordered[: min(count, len(ordered) - 1)])
    return frozenset(result)


def _assert_paired(primary: TrajectoryCache, secondary: TrajectoryCache) -> None:
    if len(primary.frames) != len(secondary.frames):
        raise ValueError(
            "paired camera caches have different lengths: "
            f"{primary.root} != {secondary.root}"
        )
    for name in ("states", "actions", "episodes", "tasks", "frame_indices"):
        if not np.array_equal(getattr(primary, name), getattr(secondary, name)):
            raise ValueError(
                f"paired camera caches disagree on {name}: "
                f"{primary.root} != {secondary.root}"
            )


def _camera_views(
    spec: V10SourceSpec,
    *,
    paired: bool,
) -> tuple[int, int | None]:
    if spec.primary_view not in (0, 1):
        raise ValueError("primary_view must be wrist=0 or external=1")

    primary_role = spec.primary_camera_role
    secondary_role = spec.secondary_camera_role
    for role in (primary_role, secondary_role):
        if role is not None and role not in _CAMERA_ROLE_TO_VIEW:
            raise ValueError("physical camera role must be wrist or external")

    if not paired:
        if secondary_role is not None:
            raise ValueError(
                "secondary_camera_role requires a paired secondary trajectory"
            )
        view = (
            int(spec.primary_view)
            if primary_role is None
            else _CAMERA_ROLE_TO_VIEW[primary_role]
        )
        return view, None

    if (primary_role is None) != (secondary_role is None):
        raise ValueError("paired camera roles must describe both trajectories")
    if primary_role is None:
        # Preserve the original paired-source contract for existing callers.
        return 0, 1
    if (primary_role, secondary_role) != (
        WRIST_CAMERA_ROLE,
        EXTERNAL_CAMERA_ROLE,
    ):
        raise ValueError(
            "paired cameras require trajectory=wrist and secondary=external"
        )
    return 0, 1


class V10TrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    """Joint raw-image action/dynamics examples with multiscale history."""

    def __init__(
        self,
        specs: Sequence[V10SourceSpec],
        *,
        split: str,
        config: V10Config,
        split_seed: int = 7310,
        val_fraction: float = 0.1,
        trim_static_tail: int = 15,
        frame_cache_bytes: int | None = None,
    ) -> None:
        config.validate()
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        if not 0.0 < val_fraction < 0.5:
            raise ValueError("val_fraction must lie between zero and one half")
        if trim_static_tail < 0:
            raise ValueError("trim_static_tail must be non-negative")
        self.config = config
        self.split = split
        self.sources: list[_LoadedSource] = []
        self.samples: list[tuple[int, int]] = []
        self.group_keys: list[tuple[str, int]] = []
        self._episode_terminal_row: dict[tuple[int, int], int] = {}
        self._frame_lru = _FrameSliceLRU(_frame_cache_bytes(frame_cache_bytes))

        for source_index, spec in enumerate(specs):
            primary = TrajectoryCache(spec.trajectory)
            secondary = (
                None
                if spec.secondary_trajectory is None
                else TrajectoryCache(spec.secondary_trajectory)
            )
            if secondary is not None:
                _assert_paired(primary, secondary)
            primary_view, secondary_view = _camera_views(
                spec, paired=secondary is not None
            )
            geometry = GeometryCache(spec.geometry, len(primary.frames))
            if spec.real_grounding is not None and spec.projection is not None:
                raise ValueError(
                    "source must choose real_grounding or projection, not both"
                )
            if spec.real_grounding is not None:
                grounding: RealGroundingLookup | V10ProjectionLookup | None = (
                    RealGroundingLookup(spec.real_grounding, len(primary.frames))
                )
            elif spec.projection is not None:
                grounding = V10ProjectionLookup(spec.projection, len(primary.frames))
            else:
                grounding = None
            if grounding is not None and secondary is None:
                raise ValueError("two-view grounding requires a paired secondary camera")
            validation = (
                frozenset(int(value) for value in spec.validation_episodes)
                if spec.validation_episodes
                else _validation_episodes(
                    primary, split_seed=split_seed, val_fraction=val_fraction
                )
            )
            tagged = (
                None
                if spec.task_class is None
                else parse_task_class(spec.task_class)
            )
            # Episodes withheld from *both* splits, for the task-level
            # split in cache/heldout_task_split_v1.json. Empty for every
            # default source, so the within-task behaviour above is
            # unchanged and the frozen cross-eval stays reproducible.
            excluded = frozenset(
                int(value) for value in spec.excluded_episodes
            )
            if excluded & validation:
                validation = frozenset(validation) - excluded
            loaded = _LoadedSource(
                primary=primary,
                secondary=secondary,
                geometry=geometry,
                grounding=grounding,
                primary_view=primary_view,
                secondary_view=secondary_view,
                validation_episodes=validation,
                task_class=tagged,
            )
            self.sources.append(loaded)

            episodes = np.asarray(primary.episodes)
            tasks = np.asarray(primary.tasks)
            frames = np.asarray(primary.frame_indices)
            states = np.asarray(primary.states)
            for raw_episode in np.unique(episodes):
                episode = int(raw_episode)
                rows = np.flatnonzero(episodes == episode)
                if len(rows) == 0:
                    raise ValueError(f"empty episode {primary.name}:{episode}")
                self._episode_terminal_row[(source_index, episode)] = int(rows[-1])
                if episode in excluded:
                    continue
                if len(rows) <= abs(config.history_offsets[0]) + config.action_horizon:
                    continue
                if not np.all(np.diff(rows) == 1) or not np.all(
                    np.diff(frames[rows]) == 1
                ):
                    raise ValueError(f"non-contiguous episode {primary.name}:{episode}")
                task_ids = np.unique(tasks[rows])
                if len(task_ids) != 1:
                    raise ValueError("one physical episode spans multiple tasks")
                episode_split = "val" if episode in validation else "train"
                if episode_split != split:
                    continue
                active_stop = len(rows)
                delta = np.max(np.abs(np.diff(states[rows], axis=0)), axis=1)
                moving = np.flatnonzero(delta > 0.02)
                if len(moving):
                    active_stop = min(
                        len(rows), int(moving[-1]) + 2 + int(trim_static_tail)
                    )
                first = abs(config.history_offsets[0])
                last_exclusive = active_stop - config.action_horizon
                task = int(task_ids[0])
                for local in range(first, last_exclusive):
                    current = int(rows[local])
                    self.samples.append((source_index, current))
                    self.group_keys.append((primary.dataset_id, task))
        if not self.samples:
            raise ValueError(f"no {split} V10 trajectory samples")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _images(
        source: _LoadedSource,
        rows: np.ndarray,
        lru: _FrameSliceLRU | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        count = len(rows)
        rgb = np.zeros((count, 2, 224, 224, 3), dtype=np.uint8)
        valid = np.zeros((count, 2), dtype=bool)
        cache = lru if lru is not None else _FrameSliceLRU(0)
        if source.secondary is None:
            view = source.primary_view
            rgb[:, view] = cache.gather(id(source.primary), source.primary.frames, rows)
            valid[:, view] = True
        else:
            if source.secondary_view is None:
                raise RuntimeError("paired source has no secondary camera view")
            rgb[:, source.primary_view] = cache.gather(
                id(source.primary), source.primary.frames, rows
            )
            rgb[:, source.secondary_view] = cache.gather(
                id(source.secondary), source.secondary.frames, rows
            )
            valid[:, (source.primary_view, source.secondary_view)] = True
        return rgb, valid

    @staticmethod
    def _action_landmarks(
        actions: np.ndarray,
        current: int,
        offsets: tuple[int, ...],
    ) -> np.ndarray:
        """Encode every irregular history interval with five real commands."""

        result = []
        for begin_offset, end_offset in zip(offsets[:-1], offsets[1:], strict=True):
            begin = current + begin_offset
            end = current + end_offset
            if end <= begin:
                raise ValueError("history offsets must be strictly increasing")
            indices = np.linspace(begin, end - 1, 5).round().astype(np.int64)
            result.append(
                np.asarray(actions[indices], dtype=np.float32).reshape(-1) / JOINT_SCALE
            )
        return np.stack(result)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index, current = self.samples[index]
        source = self.sources[source_index]
        history_rows = current + np.asarray(self.config.history_offsets, dtype=np.int64)
        future_rows = current + np.asarray(self.config.future_offsets, dtype=np.int64)
        action_rows = current + np.arange(self.config.action_horizon, dtype=np.int64)
        history_rgb, history_view_valid = self._images(
            source, history_rows, self._frame_lru
        )
        future_rgb, future_view_valid = self._images(
            source, future_rows, self._frame_lru
        )

        history_q = np.full(
            (self.config.obs_horizon, 2, 3, 2), 112.0, dtype=np.float32
        )
        future_q = np.full(
            (self.config.future_horizon, 2, 3, 2), 112.0, dtype=np.float32
        )
        history_anchor_valid = np.zeros(
            (self.config.obs_horizon, 2, 3), dtype=bool
        )
        future_anchor_valid = np.zeros(
            (self.config.future_horizon, 2, 3), dtype=bool
        )
        projection_target_fallback_to_nominal = False
        if source.grounding is not None:
            history_q = source.grounding.q_nominal[history_rows].copy()
            future_q = source.grounding.q_true[future_rows].copy()
            history_anchor_valid = source.grounding.valid_nominal[
                history_rows
            ].copy()
            future_anchor_valid = source.grounding.valid_true[future_rows].copy()
            projection_target_fallback_to_nominal = bool(
                source.grounding.target_fallback_to_nominal
            )

        result: dict[str, np.ndarray] = {
            "rgb_history": history_rgb.transpose(0, 1, 4, 2, 3),
            "rgb_future": future_rgb.transpose(0, 1, 4, 2, 3),
            "q_nominal_px": history_q,
            "anchor_xyz_history": np.asarray(
                source.geometry.anchors_xyz[history_rows], dtype=np.float32
            ),
            "state_history": np.asarray(
                source.primary.states[history_rows], dtype=np.float32
            )
            / JOINT_SCALE,
            "camera_context": np.broadcast_to(
                CAMERA_CONTEXT[None], (self.config.obs_horizon, 2, 5)
            ).copy(),
            "anchor_valid": history_anchor_valid,
            "view_valid": history_view_valid,
            "past_action_chunks": self._action_landmarks(
                source.primary.actions, current, self.config.history_offsets
            ),
            "target_action": np.asarray(
                source.primary.actions[action_rows], dtype=np.float32
            )
            / JOINT_SCALE,
            "target_action_mask": np.ones(self.config.action_horizon, dtype=bool),
            "target_future_view_valid": future_view_valid,
            "target_future_state": np.asarray(
                source.primary.states[future_rows], dtype=np.float32
            )
            / JOINT_SCALE,
            "target_future_anchor_xyz": np.asarray(
                source.geometry.anchors_xyz[future_rows], dtype=np.float32
            ),
            "target_future_anchor_px": future_q,
            "target_future_anchor_valid": future_anchor_valid,
            "projection_target_fallback_to_nominal": np.asarray(
                projection_target_fallback_to_nominal, dtype=bool
            ),
            "source_index": np.asarray(source_index, dtype=np.int64),
            "task": np.asarray(source.primary.tasks[current], dtype=np.int64),
            "episode": np.asarray(source.primary.episodes[current], dtype=np.int64),
            "is_dual_camera": np.asarray(source.secondary is not None, dtype=bool),
            "episode_terminal_state": self._terminal_state(source_index, current),
        }
        if source.task_class is not None:
            result["task_class"] = np.asarray(source.task_class, dtype=np.int64)
        return {
            name: torch.from_numpy(np.array(value, copy=True, order="C"))
            for name, value in result.items()
        }

    def _terminal_state(self, source_index: int, current: int) -> np.ndarray:
        source = self.sources[source_index]
        episode = int(source.primary.episodes[current])
        key = (source_index, episode)
        if key not in self._episode_terminal_row:
            raise RuntimeError(
                f"missing episode terminal row source={source_index} episode={episode}"
            )
        return (
            np.asarray(
                source.primary.states[self._episode_terminal_row[key]],
                dtype=np.float32,
            )
            / JOINT_SCALE
        )

    def action_contract(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit normalized limits on train episodes only."""

        states: list[np.ndarray] = []
        deltas: list[np.ndarray] = []
        allowed: dict[int, set[int]] = {}
        for source_index, current in self.samples:
            source = self.sources[source_index].primary
            allowed.setdefault(source_index, set()).add(int(source.episodes[current]))
        for source_index, episodes in sorted(allowed.items()):
            trajectory = self.sources[source_index].primary
            episode_array = np.asarray(trajectory.episodes)
            selected = np.isin(episode_array, list(episodes))
            actions = np.asarray(trajectory.actions, dtype=np.float32) / JOINT_SCALE
            states.append(actions[selected])
            same = (episode_array[1:] == episode_array[:-1]) & selected[1:]
            deltas.append(np.abs(np.diff(actions, axis=0))[same])
        all_states = np.concatenate(states)
        all_deltas = np.concatenate(deltas)
        lower = np.minimum(np.quantile(all_states, 0.001, axis=0), -0.01).astype(
            np.float32
        )
        upper = np.maximum(np.quantile(all_states, 0.999, axis=0), 0.01).astype(
            np.float32
        )
        cap = np.clip(
            np.quantile(all_deltas, 0.995, axis=0), 0.005, 0.20
        ).astype(np.float32)
        return cap, lower, upper


class V10FCPGDataset(Dataset[dict[str, torch.Tensor]]):
    """Expose real-image synthetic-drift supervision without cached features."""

    def __init__(self, root: str | Path, *, split: str) -> None:
        self.dataset = SO101SyntheticFCPGDataset(root)
        self.rows = self.dataset.indices(split)
        if not len(self.rows):
            raise ValueError(f"no {split} FCPG rows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        arrays = self.dataset.arrays
        row = int(self.rows[index])
        result = {
            "rgb": np.asarray(arrays["rgb"][row], dtype=np.uint8).transpose(2, 0, 1),
            "state": np.asarray(arrays["states"][row], dtype=np.float32) / JOINT_SCALE,
            "anchors_xyz": np.asarray(arrays["anchors_xyz"][row], dtype=np.float32),
            "q_nominal_px": np.asarray(arrays["q_nominal"][row], dtype=np.float32),
            "q_true_px": np.asarray(arrays["q_true"][row], dtype=np.float32),
            "valid": np.asarray(arrays["valid"][row], dtype=bool),
            "camera_context": np.asarray(
                arrays["camera_context"][row], dtype=np.float32
            ),
            "camera_index": np.asarray(arrays["camera_index"][row], dtype=np.int64),
        }
        return {
            name: torch.from_numpy(np.array(value, copy=True, order="C"))
            for name, value in result.items()
        }


class V10FixedAffineTrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    """Apply one temporally consistent camera perturbation to a local trajectory."""

    def __init__(
        self,
        base: V10TrajectoryDataset,
        *,
        drift: SO101ImageAffineDrift,
    ) -> None:
        if len(base.sources) != 1 or base.sources[0].secondary is None:
            raise ValueError("drift evaluation requires one paired dual-camera source")
        self.base = base
        self.drift = drift
        self.matrix = drift.matrix(base.config.image_size, base.config.image_size)

    @property
    def samples(self) -> list[tuple[int, int]]:
        return self.base.samples

    @property
    def group_keys(self) -> list[tuple[str, int]]:
        return self.base.group_keys

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = self.base[index]

        def warp_frames(value: torch.Tensor) -> np.ndarray:
            frames = value.numpy().transpose(0, 1, 3, 4, 2)
            warped = np.empty_like(frames)
            for frame in range(frames.shape[0]):
                for view in range(frames.shape[1]):
                    warped[frame, view] = warp_rgb(frames[frame, view], self.matrix)
            return warped.transpose(0, 1, 4, 2, 3)

        history_nominal = result["q_nominal_px"].numpy()
        future_nominal = result["target_future_anchor_px"].numpy()
        history_true = transform_points(history_nominal, self.matrix)
        future_true = transform_points(future_nominal, self.matrix)
        history_valid = (
            result["anchor_valid"].numpy()
            & points_in_bounds(history_true, self.base.config.image_size, self.base.config.image_size)
        )
        future_valid = (
            result["target_future_anchor_valid"].numpy()
            & points_in_bounds(future_true, self.base.config.image_size, self.base.config.image_size)
        )

        replacements = {
            "rgb_history": warp_frames(result["rgb_history"]),
            "rgb_future": warp_frames(result["rgb_future"]),
            "q_nominal_px": history_nominal,
            "history_q_true_px": history_true,
            "anchor_valid": history_valid,
            "target_future_anchor_px": future_true,
            "target_future_anchor_valid": future_valid,
            "projection_target_fallback_to_nominal": np.asarray(False, dtype=bool),
        }
        result.update(
            {
                name: torch.from_numpy(np.array(value, copy=True, order="C"))
                for name, value in replacements.items()
            }
        )
        return result


LOCKED_LOCAL_FISH_TO_BOX_NAME = "my_first_task_clean_v1"
LOCKED_LOCAL_FISH_TO_BOX_TRAIN_EPISODES = (0, 1, 3)
LOCKED_LOCAL_FISH_TO_BOX_VAL_EPISODES = (2,)
LOCKED_LOCAL_FISH_TO_BOX_MANIFEST = (
    Path(__file__).resolve().parent
    / "local_sources"
    / "my_first_task_clean_v1_locked_val2.json"
)


def locked_local_fish_to_box_source(data_root: str | Path) -> V10SourceSpec:
    """Local-only fish-to-box source. Public SO-101 mix is forbidden here."""

    root = Path(data_root).expanduser().resolve()
    return V10SourceSpec(
        trajectory=root / "cache" / "my_first_task_clean_v1_wrist_224",
        secondary_trajectory=root / "cache" / "my_first_task_clean_v1_column_224",
        geometry=root / "predictive_context" / "my_first_task_clean_v1_wrist_expanded5_10k",
        real_grounding=root
        / "fcpg"
        / "my_first_task_clean_v1_real_warp_val2_seed6124",
        primary_view=0,
        validation_episodes=LOCKED_LOCAL_FISH_TO_BOX_VAL_EPISODES,
        primary_camera_role=WRIST_CAMERA_ROLE,
        secondary_camera_role=EXTERNAL_CAMERA_ROLE,
    )


def v12_expansion_sources(data_root: str | Path) -> list[V10SourceSpec]:
    """Three extra public datasets, added for the V12.1 dynamics pool only.

    These carry no fish-to-box supervision and no task class, so they may feed
    the goal-free LeWM and must never reach the actor loss. Mounts were measured
    rather than read off the field names: `pen_mug` and `tape_in_square` ship two
    fixed cameras each and no eye-in-hand view at all, while `tea2`'s wrist
    stream moves like one. Each dataset contributes a single viewpoint so that no
    episode can straddle the train and validation split through a second camera.
    """

    root = Path(data_root).expanduser().resolve()
    cache = root / "cache"
    geometry = root / "predictive_context"
    suffix = "_expanded5_10k"

    def external_only(name: str) -> V10SourceSpec:
        return V10SourceSpec(
            trajectory=cache / name,
            geometry=geometry / f"{name}{suffix}",
            primary_view=1,
            primary_camera_role=EXTERNAL_CAMERA_ROLE,
        )

    return [
        external_only("so101_pen_mug_camera2_224"),
        external_only("so101_tape_birdeye_224"),
        V10SourceSpec(
            trajectory=cache / "so101_tea2_wrist_224",
            secondary_trajectory=cache / "so101_tea2_phone_224",
            geometry=geometry / f"so101_tea2_phone_224{suffix}",
            primary_view=0,
            primary_camera_role=WRIST_CAMERA_ROLE,
            secondary_camera_role=EXTERNAL_CAMERA_ROLE,
        ),
    ]


V12_4_SELECTION_NAME = "v12_4_source_selection.json"


def _trajectory_cache_ready(path: Path) -> bool:
    return path.is_dir() and all(
        (path / name).is_file() for name in _TRAJECTORY_CACHE_FILES
    )


def v12_4_expansion_sources(data_root: str | Path) -> list[V10SourceSpec]:
    """Normalized-unit public streams added for the V12.4 dynamics pool.

    Selection is measured (`scripts/select_v12_4_sources.py`), not read off
    camera field names. Missing trajectory or geometry files are skipped so a
    partial decode cannot crash the loader. These sources carry no task class
    and must never reach the actor loss.
    """

    root = Path(data_root).expanduser().resolve()
    cache = root / "cache"
    geometry = root / "predictive_context"
    suffix = "_expanded5_10k"
    selection_path = cache / V12_4_SELECTION_NAME
    if not selection_path.is_file():
        return []

    payload = json.loads(selection_path.read_text())
    if payload.get("format") != "so101-v12-4-source-selection-v1":
        raise ValueError(f"unsupported V12.4 selection: {selection_path}")

    sources: list[V10SourceSpec] = []
    for item in payload.get("sources") or []:
        primary = cache / str(item["primary"])
        geom = geometry / f"{item['geometry']}{suffix}"
        if not _trajectory_cache_ready(primary) or not (geom / "manifest.json").is_file():
            continue
        secondary_name = item.get("secondary")
        primary_role = item["primary_role"]
        secondary_role = item.get("secondary_role")
        if secondary_name:
            secondary = cache / str(secondary_name)
            if not _trajectory_cache_ready(secondary):
                continue
            if (primary_role, secondary_role) != (
                WRIST_CAMERA_ROLE,
                EXTERNAL_CAMERA_ROLE,
            ):
                raise ValueError(
                    f"V12.4 paired source {item['dataset_id']} is not wrist+external"
                )
            sources.append(
                V10SourceSpec(
                    trajectory=primary,
                    secondary_trajectory=secondary,
                    geometry=geom,
                    primary_view=0,
                    primary_camera_role=WRIST_CAMERA_ROLE,
                    secondary_camera_role=EXTERNAL_CAMERA_ROLE,
                )
            )
            continue
        if primary_role not in _CAMERA_ROLE_TO_VIEW:
            raise ValueError(f"unknown V12.4 camera role {primary_role!r}")
        sources.append(
            V10SourceSpec(
                trajectory=primary,
                geometry=geom,
                primary_view=_CAMERA_ROLE_TO_VIEW[primary_role],
                primary_camera_role=primary_role,
            )
        )
    return sources


def default_v10_sources(data_root: str | Path) -> list[V10SourceSpec]:
    """Return the fixed public-plus-local V10 source list."""

    root = Path(data_root).expanduser().resolve()
    cache = root / "cache"
    geometry = root / "predictive_context"
    suffix = "_expanded5_10k"
    # Entries explicitly name (fixed external, physical eye-in-hand) caches.
    public = (
        ("so101_fruit_front_224", "so101_fruit_wrist_224"),
        ("so101_table_cleanup_front_224", "so101_table_cleanup_wrist_224"),
        ("so101_drawing_front_224", None),
        (
            "so101_multitask_7tasks_700epi_top",
            "so101_multitask_7tasks_700epi_left_wrist_224",
        ),
        # This dataset's field names describe the original capture setup, not
        # the physical role visible in the decoded frames: overhead is the
        # eye-in-hand close view, while "wrist" is the fixed table view.
        ("so101_multi_task_v1_external", "so101_multi_task_v1_overhead_224"),
    )

    def trajectory_cache_ready(path: Path) -> bool:
        return _trajectory_cache_ready(path)

    def public_source(external_name: str, wrist_name: str | None) -> V10SourceSpec:
        external = cache / external_name
        source_geometry = geometry / f"{external_name}{suffix}"
        wrist = None if wrist_name is None else cache / wrist_name
        if (
            wrist is not None
            and trajectory_cache_ready(wrist)
            and trajectory_cache_ready(external)
            and source_geometry.is_dir()
        ):
            return V10SourceSpec(
                trajectory=wrist,
                secondary_trajectory=external,
                geometry=source_geometry,
                primary_view=0,
                primary_camera_role=WRIST_CAMERA_ROLE,
                secondary_camera_role=EXTERNAL_CAMERA_ROLE,
            )
        return V10SourceSpec(
            trajectory=external,
            geometry=source_geometry,
            primary_view=1,
            primary_camera_role=EXTERNAL_CAMERA_ROLE,
        )

    result = [
        public_source(external_name, wrist_name)
        for external_name, wrist_name in public
    ]
    # Declutter is intentionally excluded from the default training contract.
    # Its wrist stream contains long frozen runs while state and the external
    # view continue changing, so treating the pair as synchronized supervision
    # would teach an unlabelled stale-camera shortcut.
    result.append(locked_local_fish_to_box_source(root))
    return result


def require_batch_task_class(
    batch: Mapping[str, torch.Tensor],
    task_class: int,
) -> torch.Tensor:
    """Read actor tags from the batch. Do not invent them from --task-class."""

    if "task_class" not in batch:
        raise ValueError(
            "batch is missing task_class; actor tag has no data supervision"
        )
    tags = batch["task_class"]
    wanted = parse_task_class(task_class)
    if tags.ndim != 1:
        raise ValueError("task_class must have shape [B]")
    if tags.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("task_class must be an integer tensor")
    if torch.any(tags != wanted):
        raise ValueError("batch task_class does not match the requested class")
    return tags.long()


__all__ = [
    "CAMERA_CONTEXT",
    "EXTERNAL_CAMERA_ROLE",
    "JOINT_SCALE",
    "PROJECTION_FORMAT",
    "TrajectoryCache",
    "V10FCPGDataset",
    "V10FixedAffineTrajectoryDataset",
    "V10SourceSpec",
    "V10ProjectionLookup",
    "V10TrajectoryDataset",
    "WRIST_CAMERA_ROLE",
    "LOCKED_LOCAL_FISH_TO_BOX_MANIFEST",
    "LOCKED_LOCAL_FISH_TO_BOX_NAME",
    "LOCKED_LOCAL_FISH_TO_BOX_TRAIN_EPISODES",
    "LOCKED_LOCAL_FISH_TO_BOX_VAL_EPISODES",
    "default_v10_sources",
    "locked_local_fish_to_box_source",
    "require_batch_task_class",
]
