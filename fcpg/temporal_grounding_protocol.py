"""Pure contracts for the calibration-free temporal FCPG-H protocol.

This module intentionally does not import the v1 camera-role or calibration
contracts.  It defines only identities, causal time windows, split continuity,
and bounded-cache policy for ``so101-fcpg-h-lewm-temporal-v2``.
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

from fcpg.offline_pipeline_manifest import (
    SPLIT_SEED,
    _UnionFind,
    sha256_json,
    stable_fraction,
)


TEMPORAL_MANIFEST_FORMAT = "so101-fcpg-h-temporal-manifest-v2"
TEMPORAL_PROTOCOL_ID = "so101-fcpg-h-lewm-temporal-v2"
CANONICAL_HZ = 10.0
IMAGE_OFFSETS_S = tuple(round(-1.5 + 0.1 * index, 9) for index in range(16))
# An action row at image index i is the command whose control interval ends at
# image index i + 1.  Therefore the final included command has ended by t=0.
COMPLETED_ACTION_OFFSETS_S = IMAGE_OFFSETS_S[:-1]
GROUNDING_ENDPOINT_INDICES = (10, 15)
GROUNDING_HISTORY_SLICES = ((0, 11), (5, 16))
FUTURE_OFFSETS_S = (0.5, 1.0, 1.5, 2.0)
FUTURE_ACTION_OFFSETS_S = tuple(round(0.1 * index, 9) for index in range(20))
V12_4_EXPOSURE_VALUES = frozenset({"exact", "duplicate", "unseen", "unknown"})

CPU_DECODE_LRU_BYTES = 96 * 1024**3
DEFAULT_GPU_PATCH_LRU_BYTES = 80 * 1024**3
GPU_RESERVE_BYTES = 8 * 1024**3
GROUNDED_TOKEN_CACHE_BYTES = 120 * 1024**3
DATA_DISK_RESERVE_BYTES = 128 * 1024**3
TEMPORAL_RGB_CACHE_BYTES = 320 * 1024**3


def select_roleless_views(
    record: Mapping[str, object],
    *,
    maximum_views: int | None,
    selection_seed: int,
) -> tuple[str, ...]:
    """Apply the one shared deterministic view-selection contract.

    Cache planning and online batch loading call this same function so an
    exact-schedule cache cannot silently disagree with training.
    """

    manifest_row = record.get("manifest_row")
    if not isinstance(manifest_row, Mapping):
        raise ValueError("temporal episode record has no manifest_row")
    views = tuple(map(str, manifest_row.get("view_keys", ())))
    if not views:
        raise ValueError("temporal episode has no declared view")
    if maximum_views is not None:
        if maximum_views <= 0:
            raise ValueError("maximum_views must be positive")
        if len(views) > maximum_views:
            episode_id = str(record.get("canonical_episode_id", ""))
            if not episode_id:
                raise ValueError("temporal episode has no canonical identity")
            views = tuple(
                sorted(
                    views,
                    key=lambda view: hashlib.sha256(
                        f"{selection_seed}\0{episode_id}\0{view}".encode()
                    ).digest(),
                )[:maximum_views]
            )
    return views


@dataclass(frozen=True)
class TemporalGroundingWindow:
    """One causal 16-frame union window and its four real future targets."""

    current_time_s: float
    image_indices: tuple[int, ...]
    completed_action_indices: tuple[int, ...]
    future_indices: tuple[int, ...]
    future_history_indices: tuple[tuple[int, ...], ...]
    future_action_indices: tuple[int, ...]

    def endpoint_image_indices(self, endpoint: int) -> tuple[int, ...]:
        start, stop = GROUNDING_HISTORY_SLICES[endpoint]
        return self.image_indices[start:stop]

    def endpoint_action_indices(self, endpoint: int) -> tuple[int, ...]:
        start, stop = GROUNDING_HISTORY_SLICES[endpoint]
        return self.completed_action_indices[start : stop - 1]


def _strictly_increasing(values: Sequence[float]) -> bool:
    return all(right > left for left, right in zip(values, values[1:]))


def temporal_grounding_windows(
    *,
    timestamps_s: Sequence[float] | None = None,
    length: int | None = None,
    fps: float | None = None,
    canonical_hz: float = CANONICAL_HZ,
    max_time_error_s: float | None = None,
) -> list[TemporalGroundingWindow]:
    """Resample by physical timestamps without admitting a future command.

    The 16 images cover ``[-1.5, 0]`` seconds.  The 15 action rows correspond
    to the completed intervals between adjacent images, never the command that
    starts at the current endpoint.  Only the four declared future image
    targets are returned; the function cannot manufacture H5/H10 targets.
    """

    if timestamps_s is None:
        if length is None or length <= 0 or fps is None or fps <= 0:
            raise ValueError("length and positive fps are required without timestamps")
        timestamps = [index / float(fps) for index in range(length)]
    else:
        timestamps = [float(value) for value in timestamps_s]
        if not timestamps or not all(math.isfinite(value) for value in timestamps):
            raise ValueError("timestamps must be non-empty and finite")
        if length is not None and length != len(timestamps):
            raise ValueError("length does not match timestamps")
    if not _strictly_increasing(timestamps):
        raise ValueError("timestamps must be strictly increasing")
    if canonical_hz <= 0:
        raise ValueError("canonical_hz must be positive")

    step = 1.0 / canonical_hz
    tolerance = step / 2.0 + 1e-9 if max_time_error_s is None else max_time_error_s
    if tolerance < 0:
        raise ValueError("max_time_error_s cannot be negative")

    def nearest_index(target: float) -> int | None:
        low, high = 0, len(timestamps)
        while low < high:
            middle = (low + high) // 2
            if timestamps[middle] < target:
                low = middle + 1
            else:
                high = middle
        candidates = [index for index in (low - 1, low) if 0 <= index < len(timestamps)]
        index = min(candidates, key=lambda candidate: (abs(timestamps[candidate] - target), candidate))
        return index if abs(timestamps[index] - target) <= tolerance else None

    start = timestamps[0] - min(IMAGE_OFFSETS_S)
    stop = timestamps[-1] - max(FUTURE_OFFSETS_S)
    first_tick = math.ceil((start - timestamps[0]) / step - 1e-9)
    last_tick = math.floor((stop - timestamps[0]) / step + 1e-9)
    windows: list[TemporalGroundingWindow] = []
    for tick in range(first_tick, last_tick + 1):
        current = timestamps[0] + tick * step
        images = tuple(nearest_index(current + offset) for offset in IMAGE_OFFSETS_S)
        future_histories = tuple(
            tuple(nearest_index(current + horizon + offset) for offset in IMAGE_OFFSETS_S)
            for horizon in FUTURE_OFFSETS_S
        )
        futures = tuple(history[-1] for history in future_histories)
        future_actions = tuple(
            nearest_index(current + offset) for offset in FUTURE_ACTION_OFFSETS_S
        )
        if any(index is None for index in (*images, *futures, *future_actions, *(index for history in future_histories for index in history))):
            continue
        image_indices = tuple(int(index) for index in images)
        future_indices = tuple(int(index) for index in futures)
        future_history_indices = tuple(
            tuple(int(index) for index in history) for history in future_histories
        )
        future_action_indices = tuple(int(index) for index in future_actions)
        if len(set(image_indices)) != len(image_indices):
            # A source below 10 Hz cannot satisfy the temporal contract.
            continue
        action_indices = image_indices[:-1]
        if action_indices[-1] >= image_indices[-1]:
            raise AssertionError("future action leaked into temporal window")
        windows.append(
            TemporalGroundingWindow(
                current_time_s=round(current, 9),
                image_indices=image_indices,
                completed_action_indices=action_indices,
                future_indices=future_indices,
                future_history_indices=future_history_indices,
                future_action_indices=future_action_indices,
            )
        )
    return windows


def validate_roleless_views(view_keys: Sequence[str]) -> tuple[str, ...]:
    """Return deterministic stream keys without assigning camera semantics."""

    views = tuple(sorted(set(str(key) for key in view_keys if str(key))))
    if not views:
        raise ValueError("temporal FCPG-H requires at least one image stream")
    return views


def classify_v12_4_exposure(
    row: Mapping[str, object],
    *,
    exact_episode_ids: set[str],
    duplicate_groups: set[str],
) -> str:
    """Classify exposure conservatively; missing identity is never 'unseen'."""

    episode_id = str(row.get("canonical_episode_id", ""))
    duplicate = str(row.get("duplicate_group", ""))
    state_hash = str(row.get("state_action_sha256", ""))
    if not episode_id or not duplicate or not state_hash:
        return "unknown"
    if episode_id in exact_episode_ids:
        return "exact"
    if duplicate in duplicate_groups:
        return "duplicate"
    return "unseen"


def assign_preserving_splits(
    rows: Sequence[Mapping[str, object]],
    *,
    previous_rows: Sequence[Mapping[str, object]],
    seed: int = SPLIT_SEED,
) -> dict[str, dict[str, str]]:
    """Extend source/duplicate components without changing a prior split.

    A new episode connected to a previous source or duplicate inherits that
    component's split.  If new evidence connects two previously different
    sealed identities, the build fails rather than silently moving either.
    """

    episode_ids = [str(row["canonical_episode_id"]) for row in rows]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("canonical_episode_id values must be unique")
    union = _UnionFind(episode_ids)
    for field in ("source_group", "duplicate_group"):
        groups: dict[str, list[str]] = {}
        for row, episode_id in zip(rows, episode_ids, strict=True):
            value = str(row.get(field, ""))
            if not value:
                raise ValueError(f"each episode requires {field}")
            groups.setdefault(value, []).append(episode_id)
        for members in groups.values():
            for episode_id in members[1:]:
                union.union(members[0], episode_id)

    previous_by_id = {
        str(row["canonical_episode_id"]): str(row.get("split", ""))
        for row in previous_rows
        if row.get("canonical_episode_id") and row.get("split")
    }
    previous_by_source: dict[str, set[str]] = {}
    previous_by_duplicate: dict[str, set[str]] = {}
    for row in previous_rows:
        split = str(row.get("split", ""))
        if not split:
            continue
        previous_by_source.setdefault(str(row.get("source_group", "")), set()).add(split)
        previous_by_duplicate.setdefault(str(row.get("duplicate_group", "")), set()).add(split)

    components: dict[str, list[Mapping[str, object]]] = {}
    for row, episode_id in zip(rows, episode_ids, strict=True):
        components.setdefault(union.find(episode_id), []).append(row)

    result: dict[str, dict[str, str]] = {}
    for members in components.values():
        inherited: set[str] = set()
        for row in members:
            episode_id = str(row["canonical_episode_id"])
            if episode_id in previous_by_id:
                inherited.add(previous_by_id[episode_id])
            inherited.update(previous_by_source.get(str(row["source_group"]), set()))
            inherited.update(previous_by_duplicate.get(str(row["duplicate_group"]), set()))
        inherited.discard("")
        if len(inherited) > 1:
            raise ValueError(f"new duplicate/source evidence joins prior splits: {sorted(inherited)}")
        component_members = sorted(str(row["canonical_episode_id"]) for row in members)
        component_id = sha256_json(component_members)
        if inherited:
            split = next(iter(inherited))
        else:
            fraction = stable_fraction(seed, "temporal-v2-split-component", component_id)
            split = "train" if fraction < 0.70 else "development" if fraction < 0.85 else "sealed_test"
        for episode_id in component_members:
            result[episode_id] = {"split": split, "split_component": component_id}
    return result


def build_temporal_manifest(
    rows: Sequence[Mapping[str, object]],
    *,
    search_roots: Sequence[str],
    scan_mode: str,
    previous_manifest_sha256: str | None,
    seed: int = SPLIT_SEED,
) -> dict[str, object]:
    if scan_mode not in {"metadata_audit", "content_audit", "formal"}:
        raise ValueError("invalid temporal manifest scan mode")
    normalized = [dict(row) for row in rows]
    manifest: dict[str, object] = {
        "format": TEMPORAL_MANIFEST_FORMAT,
        "protocol_id": TEMPORAL_PROTOCOL_ID,
        "scan_mode": scan_mode,
        "formal_ready": scan_mode == "formal",
        "split_seed": int(seed),
        "split_fractions": {"train": 0.70, "development": 0.15, "sealed_test": 0.15},
        "canonical_hz": CANONICAL_HZ,
        "image_offsets_s": list(IMAGE_OFFSETS_S),
        "completed_action_offsets_s": list(COMPLETED_ACTION_OFFSETS_S),
        "grounding_endpoint_indices": list(GROUNDING_ENDPOINT_INDICES),
        "grounding_history_slices": [list(value) for value in GROUNDING_HISTORY_SLICES],
        "future_offsets_s": list(FUTURE_OFFSETS_S),
        "future_action_offsets_s": list(FUTURE_ACTION_OFFSETS_S),
        "view_contract": "roleless_shared_parameters_no_camera_metadata",
        "previous_manifest_sha256": previous_manifest_sha256,
        "cache_limits_bytes": {
            "cpu_decode_lru": CPU_DECODE_LRU_BYTES,
            "gpu_patch_lru_default": DEFAULT_GPU_PATCH_LRU_BYTES,
            "gpu_reserve": GPU_RESERVE_BYTES,
            "grounded_token_cache": GROUNDED_TOKEN_CACHE_BYTES,
            "data_disk_reserve": DATA_DISK_RESERVE_BYTES,
        },
        "search_roots": list(search_roots),
        "rows": normalized,
    }
    manifest["content_sha256"] = sha256_json(manifest)
    return manifest


def temporal_window_to_dict(window: TemporalGroundingWindow) -> dict[str, object]:
    return asdict(window)


def assert_training_rows(rows: Iterable[Mapping[str, object]]) -> None:
    for row in rows:
        if str(row.get("split", "")) != "train":
            raise ValueError("FCPG-H training may consume train split rows only")
        exposure = str(row.get("v12_4_exposure", "unknown"))
        if exposure not in V12_4_EXPOSURE_VALUES:
            raise ValueError(f"invalid V12.4 exposure label: {exposure!r}")
