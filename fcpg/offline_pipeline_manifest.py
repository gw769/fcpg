"""Canonical data contracts for the SO-101 FCPG -> LeWM forward pipeline.

The module is deliberately independent of PyArrow/OpenCV.  Dataset discovery
and sparse media decoding live in ``scripts/build_so101_canonical_manifest.py``;
the identity, role, split and time-window rules below are pure and unit-testable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


MANIFEST_FORMAT = "so101-fcpg-lewm-canonical-manifest-v1"
PROTOCOL_ID = "so101-fcpg-lewm-forward-v1"
SPLIT_SEED = 12_600
CANONICAL_HZ = 10.0
HISTORY_OFFSETS_S = (-0.5, 0.0)
FUTURE_OFFSETS_S = (0.5, 1.0, 1.5, 2.0)
ACTION_CHUNK_OFFSETS_S = tuple(
    tuple(round(0.5 * horizon + 0.1 * step, 9) for step in range(1, 6))
    for horizon in range(4)
)
ACTION_OFFSETS_S = tuple(
    offset for chunk in ACTION_CHUNK_OFFSETS_S for offset in chunk
)


@dataclass(frozen=True)
class ViewRoleAssignment:
    external: str
    wrist: str
    stress_views: tuple[str, ...]
    available_views: tuple[str, ...]
    rule: str


@dataclass(frozen=True)
class CanonicalWindow:
    current_time_s: float
    history_indices: tuple[int, ...]
    future_indices: tuple[int, ...]
    action_indices: tuple[int, ...]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def stable_fraction(seed: int, namespace: str, *parts: object) -> float:
    payload = "\0".join((str(seed), namespace, *(str(part) for part in parts)))
    value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
    return value / float(1 << 64)


def _normalized_view_name(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def _role_score(key: str, role: str) -> int | None:
    name = _normalized_view_name(key)
    tokens = set(name.split("_"))
    if role == "wrist":
        scores = (
            ("left_wrist", 100),
            ("wrist_left", 100),
            ("wrist", 90),
            ("gripper", 80),
            ("hand", 70),
            ("eef", 70),
        )
    elif role == "external":
        scores = (
            ("base_0", 100),
            ("desk_view", 95),
            ("column", 90),
            ("overhead", 85),
            ("top", 80),
            ("front", 75),
            ("external", 70),
            ("third_person", 65),
            ("base", 60),
            ("desk", 55),
        )
    else:
        raise ValueError(f"unknown role: {role!r}")
    for phrase, score in scores:
        if phrase in name or phrase in tokens:
            return score
    return None


def infer_view_roles(
    view_keys: Sequence[str],
    *,
    dataset_hint: str = "",
    overrides: Mapping[str, str] | None = None,
) -> ViewRoleAssignment:
    """Resolve one external and one wrist view or fail on ambiguity.

    Overrides use the keys ``external`` and ``wrist``.  They are intended for
    a checked-in provenance map, never for an unrecorded command-line guess.
    """

    views = tuple(sorted(set(view_keys)))
    if len(views) < 2:
        raise ValueError("a canonical source requires at least two distinct image views")
    override = dict(overrides or {})
    unknown_override = set(override) - {"external", "wrist"}
    if unknown_override:
        raise ValueError(f"unknown role overrides: {sorted(unknown_override)}")

    hint = dataset_hint.lower()
    if "pi05" in hint or "pi0.5" in hint or "pi0_5" in hint:
        pi_external = next((v for v in views if "base_0_rgb" in v.lower()), None)
        pi_wrist = next((v for v in views if "left_wrist_0_rgb" in v.lower()), None)
        if pi_external is not None and pi_wrist is not None:
            override.setdefault("external", pi_external)
            override.setdefault("wrist", pi_wrist)

    chosen: dict[str, str] = {}
    rules: list[str] = []
    for role in ("external", "wrist"):
        if role in override:
            value = override[role]
            if value not in views:
                raise ValueError(f"{role} override {value!r} is not an available view")
            chosen[role] = value
            rules.append(f"override:{role}")
            continue
        scored = sorted(
            ((score, view) for view in views if (score := _role_score(view, role)) is not None),
            key=lambda item: (-item[0], item[1]),
        )
        if not scored:
            raise ValueError(f"cannot infer a {role} view from {list(views)!r}")
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            raise ValueError(
                f"ambiguous {role} views at equal priority: {scored[0][1]!r}, "
                f"{scored[1][1]!r}; add a checked-in override"
            )
        chosen[role] = scored[0][1]
        rules.append(f"heuristic:{role}:{scored[0][0]}")

    if chosen["external"] == chosen["wrist"]:
        raise ValueError("external and wrist roles cannot use the same stream")
    stress = tuple(view for view in views if view not in set(chosen.values()))
    return ViewRoleAssignment(
        external=chosen["external"],
        wrist=chosen["wrist"],
        stress_views=stress,
        available_views=views,
        rule=";".join(rules),
    )


def episode_semantic_fingerprint(
    *,
    task: str,
    length: int,
    state_action_sha256: str,
    sparse_visual_hashes: Mapping[str, Sequence[str]],
) -> str:
    if length <= 0:
        raise ValueError("episode length must be positive")
    if not state_action_sha256:
        raise ValueError("state/action hash is required for formal duplicate grouping")
    if set(sparse_visual_hashes) != {"external", "wrist"}:
        raise ValueError("formal duplicate grouping requires external and wrist visual hashes")
    if any(not hashes for hashes in sparse_visual_hashes.values()):
        raise ValueError("each primary view must contain at least one sparse visual hash")
    return sha256_json(
        {
            "task": " ".join(task.split()).casefold(),
            "length": int(length),
            "state_action_sha256": state_action_sha256,
            "sparse_visual_hashes": {
                role: list(sparse_visual_hashes[role])
                for role in ("external", "wrist")
            },
        }
    )


def metadata_duplicate_candidate_key(row: Mapping[str, object]) -> str:
    """Cheap first-stage key used before sparse media decoding."""

    state_action = str(row.get("state_action_sha256", ""))
    if not state_action:
        raise ValueError("metadata duplicate candidate requires state_action_sha256")
    tasks = row.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("metadata duplicate candidate requires non-empty tasks")
    return sha256_json(
        {
            "tasks": sorted(" ".join(str(task).split()).casefold() for task in tasks),
            "length": int(row["length"]),
            "state_action_sha256": state_action,
        }
    )


class _UnionFind:
    def __init__(self, keys: Iterable[str]) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        smaller, larger = sorted((a, b))
        self.parent[larger] = smaller


def assign_source_disjoint_splits(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int = SPLIT_SEED,
) -> dict[str, dict[str, str]]:
    """Assign source/duplicate connected components to deterministic splits."""

    episode_ids = [str(row["canonical_episode_id"]) for row in rows]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("canonical_episode_id values must be unique")
    union = _UnionFind(episode_ids)
    by_source: dict[str, list[str]] = {}
    by_duplicate: dict[str, list[str]] = {}
    for row, episode_id in zip(rows, episode_ids, strict=True):
        source = str(row.get("source_group", ""))
        duplicate = str(row.get("duplicate_group", ""))
        if not source or not duplicate:
            raise ValueError("each episode requires source_group and duplicate_group")
        by_source.setdefault(source, []).append(episode_id)
        by_duplicate.setdefault(duplicate, []).append(episode_id)
    for groups in (by_source, by_duplicate):
        for members in groups.values():
            for episode_id in members[1:]:
                union.union(members[0], episode_id)

    components: dict[str, list[str]] = {}
    for episode_id in episode_ids:
        components.setdefault(union.find(episode_id), []).append(episode_id)

    result: dict[str, dict[str, str]] = {}
    for members in components.values():
        component_id = sha256_json(sorted(members))
        fraction = stable_fraction(seed, "split-component", component_id)
        split = "train" if fraction < 0.70 else "development" if fraction < 0.85 else "sealed_test"
        for episode_id in members:
            result[episode_id] = {
                "split": split,
                "split_component": component_id,
            }
    return result


def assign_calibration_and_scoring_usage(
    rows: Sequence[Mapping[str, object]],
    split_assignments: Mapping[str, Mapping[str, str]],
    *,
    calibration_fraction: float = 0.10,
) -> dict[str, str]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie strictly between zero and one")
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        episode_id = str(row["canonical_episode_id"])
        split = split_assignments[episode_id]["split"]
        grouped.setdefault((str(row["source_group"]), split), []).append(row)

    usage: dict[str, str] = {}
    for (_, split), group_rows in grouped.items():
        ordered = sorted(
            group_rows,
            key=lambda row: (int(row.get("episode_index", -1)), str(row["canonical_episode_id"])),
        )
        calibration_count = max(1, int(math.ceil(len(ordered) * calibration_fraction)))
        for index, row in enumerate(ordered):
            episode_id = str(row["canonical_episode_id"])
            if split == "train":
                usage[episode_id] = "train"
            elif index < calibration_count:
                usage[episode_id] = "calibration_fit_only"
            else:
                usage[episode_id] = "development_score" if split == "development" else "sealed_score"
    return usage


def _strictly_increasing(values: Sequence[float]) -> bool:
    return all(right > left for left, right in zip(values, values[1:]))


def canonical_windows(
    *,
    timestamps_s: Sequence[float] | None = None,
    length: int | None = None,
    fps: float | None = None,
    canonical_hz: float = CANONICAL_HZ,
    max_time_error_s: float | None = None,
) -> list[CanonicalWindow]:
    """Resample an episode by timestamps and return fixed physical-time windows."""

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

    canonical_step = 1.0 / canonical_hz
    tolerance = canonical_step / 2.0 + 1e-9 if max_time_error_s is None else max_time_error_s
    if tolerance < 0:
        raise ValueError("max_time_error_s cannot be negative")

    def nearest_index(target: float) -> int | None:
        # Binary search without a NumPy dependency.
        low, high = 0, len(timestamps)
        while low < high:
            middle = (low + high) // 2
            if timestamps[middle] < target:
                low = middle + 1
            else:
                high = middle
        candidates = [idx for idx in (low - 1, low) if 0 <= idx < len(timestamps)]
        index = min(candidates, key=lambda idx: (abs(timestamps[idx] - target), idx))
        return index if abs(timestamps[index] - target) <= tolerance else None

    start = timestamps[0] - min(HISTORY_OFFSETS_S)
    stop = timestamps[-1] - max(FUTURE_OFFSETS_S)
    first_tick = math.ceil((start - timestamps[0]) / canonical_step - 1e-9)
    last_tick = math.floor((stop - timestamps[0]) / canonical_step + 1e-9)
    windows: list[CanonicalWindow] = []
    for tick in range(first_tick, last_tick + 1):
        current = timestamps[0] + tick * canonical_step
        history = tuple(nearest_index(current + offset) for offset in HISTORY_OFFSETS_S)
        future = tuple(nearest_index(current + offset) for offset in FUTURE_OFFSETS_S)
        action = tuple(nearest_index(current + offset) for offset in ACTION_OFFSETS_S)
        if any(index is None for index in (*history, *future, *action)):
            continue
        all_indices = tuple(int(index) for index in (*history, *future, *action))
        if len(set(action)) != len(action):
            # Sources below 10 Hz cannot satisfy the executed-action chunk.
            continue
        windows.append(
            CanonicalWindow(
                current_time_s=round(current, 9),
                history_indices=all_indices[: len(history)],
                future_indices=all_indices[len(history) : len(history) + len(future)],
                action_indices=all_indices[-len(action) :],
            )
        )
    return windows


def assert_training_usage(rows: Iterable[Mapping[str, object]]) -> None:
    prohibited = {"development_score", "sealed_score"}
    for row in rows:
        usage = str(row.get("usage", ""))
        if usage in prohibited:
            raise ValueError(f"training input contains prohibited {usage} episode")
        if usage not in {"train", "calibration_fit_only"}:
            raise ValueError(f"training input has unknown usage {usage!r}")


def build_manifest(
    rows: Sequence[Mapping[str, object]],
    *,
    search_roots: Sequence[str],
    scan_mode: str,
    role_rules_sha256: str,
    seed: int = SPLIT_SEED,
) -> dict[str, object]:
    if scan_mode not in {"metadata_audit", "content_audit", "formal"}:
        raise ValueError("scan_mode must be metadata_audit, content_audit or formal")
    normalized_rows = [dict(row) for row in rows]
    if scan_mode == "formal" and any(not row.get("duplicate_group") for row in normalized_rows):
        raise ValueError("formal manifest rows require duplicate_group")
    manifest: dict[str, object] = {
        "format": MANIFEST_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "scan_mode": scan_mode,
        "formal_ready": scan_mode == "formal",
        "split_seed": int(seed),
        "split_fractions": {"train": 0.70, "development": 0.15, "sealed_test": 0.15},
        "canonical_hz": CANONICAL_HZ,
        "history_offsets_s": list(HISTORY_OFFSETS_S),
        "future_offsets_s": list(FUTURE_OFFSETS_S),
        "action_offsets_s": list(ACTION_OFFSETS_S),
        "action_chunk_offsets_s": [list(chunk) for chunk in ACTION_CHUNK_OFFSETS_S],
        "search_roots": list(search_roots),
        "role_rules_sha256": role_rules_sha256,
        "rows": normalized_rows,
    }
    manifest["content_sha256"] = sha256_json(manifest)
    return manifest


def window_to_dict(window: CanonicalWindow) -> dict[str, object]:
    return asdict(window)
