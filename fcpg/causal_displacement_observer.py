"""Causal acquire-then-predict displacement events for FCPG-H/LeWM.

The first command-CAD/FK 100 mm event is observation-only acquisition.  A
second accumulator is reset at that measured endpoint; only its following
100 mm command interval is predicted.  The prediction endpoint contributes a
visual measurement token but no measured state or FK to the transition prior.
"""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fcpg.offline_pipeline_manifest import sha256_file, sha256_json
from fcpg.temporal_index import EpisodeBlockSampler
from fcpg.temporal_rgb_cache import RGBSchedule


CAUSAL_DISPLACEMENT_OBSERVER_PROTOCOL_ID = (
    "so101-query-token-observer-acquire100-predict100-v4"
)
CAUSAL_DISPLACEMENT_OBSERVER_LEDGER_FORMAT = (
    "so101-causal-displacement-observer-ledger-v1"
)
CAUSAL_DISPLACEMENT_OBSERVER_CHECKPOINT_FORMAT = (
    "so101-causal-displacement-observer-checkpoint-v4"
)
DISPLACEMENT_SEGMENT_MM = 100.0
ACTION_PROGRESS_SAMPLES = 5


@dataclass(frozen=True)
class CausalDisplacementObserverBatch:
    occurrence_indices: np.ndarray
    window_indices: np.ndarray
    view_selection_seeds: np.ndarray
    acquisition_endpoint_indices: np.ndarray
    prediction_endpoint_indices: np.ndarray
    action_progress_indices: np.ndarray
    command_motion_endpoint: np.ndarray


def causal_displacement_observer_contract(
    measurement_tokens: torch.Tensor,
    states: torch.Tensor,
    completed_actions: torch.Tensor,
    motion_signatures: torch.Tensor,
    *,
    acquisition_endpoint_indices: torch.Tensor,
    prediction_endpoint_indices: torch.Tensor,
    action_progress_indices: torch.Tensor,
    command_motion_endpoint: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build one causal transition from acquisition to prediction event."""

    batch = len(measurement_tokens)
    if measurement_tokens.ndim != 5 or measurement_tokens.shape[1] != 16:
        raise ValueError("measurement_tokens must be [B,16,V,Q,D]")
    if states.shape != (batch, 16, 6):
        raise ValueError("states must be [B,16,6]")
    if completed_actions.shape != (batch, 15, 6):
        raise ValueError("completed_actions must be [B,15,6]")
    if motion_signatures.shape != (batch, 16, 64):
        raise ValueError("motion_signatures must be [B,16,64]")
    acquisition = acquisition_endpoint_indices.to(
        device=measurement_tokens.device, dtype=torch.long
    )
    prediction = prediction_endpoint_indices.to(
        device=measurement_tokens.device, dtype=torch.long
    )
    progress = action_progress_indices.to(
        device=measurement_tokens.device, dtype=torch.long
    )
    command_motion = command_motion_endpoint.to(
        device=measurement_tokens.device, dtype=motion_signatures.dtype
    )
    if acquisition.shape != (batch,) or prediction.shape != (batch,):
        raise ValueError("event endpoints must be [B]")
    if bool(
        ((acquisition < 1) | (acquisition > 14)).any()
        or ((prediction <= acquisition) | (prediction > 15)).any()
    ):
        raise ValueError("observer events require 1 <= acquisition < prediction <= 15")
    if progress.shape != (batch, ACTION_PROGRESS_SAMPLES):
        raise ValueError("event action-progress indices must be [B,5]")
    if bool(
        ((progress < acquisition[:, None]) | (progress >= prediction[:, None])).any()
        or (progress[:, 1:] < progress[:, :-1]).any()
    ):
        raise ValueError("prediction action progress must lie after acquisition and before target")
    if command_motion.shape != (batch, 64) or not bool(
        torch.isfinite(command_motion).all()
    ):
        raise ValueError("command-motion endpoint must be finite [B,64]")

    rows = torch.arange(batch, device=measurement_tokens.device)
    token_endpoints = torch.stack(
        (
            measurement_tokens[rows, acquisition],
            measurement_tokens[rows, prediction],
        ),
        dim=1,
    )
    acquisition_state = states[rows, acquisition]
    state_endpoints = acquisition_state[:, None].expand(-1, 2, -1).clone()
    acquisition_motion = motion_signatures[rows, acquisition]
    motion_endpoints = torch.stack((acquisition_motion, command_motion), dim=1)
    action_rows = rows[:, None].expand_as(progress)
    progress_actions = completed_actions[action_rows, progress]
    action_chunk = progress_actions.reshape(batch, 1, ACTION_PROGRESS_SAMPLES * 6)
    endpoints = torch.stack((acquisition, prediction), dim=-1)
    return token_endpoints, state_endpoints, action_chunk, motion_endpoints, endpoints


def _global_image_indices(index: Any, windows: np.ndarray) -> np.ndarray:
    episodes = np.asarray(index.arrays["window_episode"][windows], dtype=np.int64)
    starts = np.fromiter(
        (int(index.episodes[int(episode)]["frame_start"]) for episode in episodes),
        dtype=np.int64,
        count=len(episodes),
    )
    local = np.asarray(index.arrays["window_image_indices"][windows], dtype=np.int64)
    return starts[:, None] + local


def _scheduled_occurrences(
    index: Any, schedule: RGBSchedule
) -> tuple[np.ndarray, np.ndarray]:
    sampler = EpisodeBlockSampler(
        index,
        seed=schedule.seed,
        per_source_cap=schedule.per_source_cap,
        block_size=schedule.block_size,
        cluster_pair_rows=schedule.cluster_pair_rows,
        require_cluster_pairs=schedule.require_cluster_pairs,
    )
    windows: list[np.ndarray] = []
    seeds: list[np.ndarray] = []
    for step in range(schedule.steps):
        rows = sampler.batch(
            schedule.batch_size,
            low_motion_fraction=schedule.low_motion_fraction,
        )
        windows.append(rows.astype(np.int64, copy=False))
        seeds.append(
            np.full(
                len(rows), schedule.seed * 1_000_003 + step, dtype=np.int64
            )
        )
    return np.concatenate(windows), np.concatenate(seeds)


def causal_displacement_events_for_windows(
    index: Any,
    windows: np.ndarray,
    *,
    kinematics: Any,
    segment_mm: float = DISPLACEMENT_SEGMENT_MM,
) -> dict[str, np.ndarray]:
    """Resolve two independent command-CAD displacement events.

    The returned ``input_occurrence_index`` binds every accepted row to its
    position in ``windows``.  This keeps duplicate occurrences distinct and
    lets callers attach pre-existing view-selection seeds without changing the
    physical event calculation.
    """

    if segment_mm != DISPLACEMENT_SEGMENT_MM:
        raise ValueError("the observer fixes both displacement segments at 100 mm")
    selected = np.asarray(windows, dtype=np.int64)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("causal displacement windows must be a non-empty vector")
    if bool(((selected < 0) | (selected >= len(index))).any()):
        raise IndexError("causal displacement window lies outside the temporal index")
    global_images = _global_image_indices(index, selected)
    command_rows = global_images[:, :-1]
    unique_commands, inverse = np.unique(command_rows, return_inverse=True)
    command_states = np.asarray(
        index.arrays["actions"][unique_commands], dtype=np.float32
    )
    command_tips, _ = kinematics.forward_tips_and_jacobians(command_states)
    command_tips = np.asarray(command_tips, dtype=np.float32)[inverse].reshape(
        len(selected), 15, 2, 3
    )
    start_signature = np.asarray(
        index.arrays["motion_signatures"][global_images[:, 0]], dtype=np.float32
    )
    start_tips = start_signature[:, 24:30].reshape(len(selected), 2, 3)

    acquisition_path = np.concatenate((start_tips[:, None], command_tips), axis=1)
    acquisition_step_mm = (
        np.linalg.norm(np.diff(acquisition_path, axis=1), axis=-1).max(axis=-1)
        * 1000.0
    )
    acquisition_cumulative = np.cumsum(acquisition_step_mm, axis=1)
    acquisition_reached = acquisition_cumulative >= float(segment_mm)
    acquisition_valid = acquisition_reached.any(axis=1)
    acquisition_action = np.argmax(acquisition_reached, axis=1).astype(np.int16)
    acquisition_endpoint = acquisition_action + 1

    occurrence = np.arange(len(selected))
    acquisition_global = global_images[occurrence, acquisition_endpoint]
    acquisition_signature = np.asarray(
        index.arrays["motion_signatures"][acquisition_global], dtype=np.float32
    )
    acquisition_state = np.asarray(
        index.arrays["states"][acquisition_global], dtype=np.float32
    )
    acquisition_tips = acquisition_signature[:, 24:30].reshape(
        len(selected), 2, 3
    )

    prediction_step_mm = np.zeros((len(selected), 15), dtype=np.float32)
    for action_index in range(15):
        active = acquisition_endpoint <= action_index
        previous_tips = np.where(
            (acquisition_endpoint == action_index)[:, None, None],
            acquisition_tips,
            command_tips[:, max(0, action_index - 1)],
        )
        step = (
            np.linalg.norm(command_tips[:, action_index] - previous_tips, axis=-1)
            .max(axis=-1)
            .astype(np.float32)
            * 1000.0
        )
        prediction_step_mm[:, action_index] = np.where(active, step, 0.0)
    prediction_cumulative = np.cumsum(prediction_step_mm, axis=1)
    prediction_reached = prediction_cumulative >= float(segment_mm)
    prediction_valid = prediction_reached.any(axis=1)
    prediction_action = np.argmax(prediction_reached, axis=1).astype(np.int16)
    prediction_endpoint = prediction_action + 1
    valid = (
        acquisition_valid
        & prediction_valid
        & (acquisition_endpoint < prediction_endpoint)
    )

    progress_mm = np.linspace(
        segment_mm / ACTION_PROGRESS_SAMPLES,
        segment_mm,
        ACTION_PROGRESS_SAMPLES,
        dtype=np.float32,
    )
    progress_indices = np.argmax(
        prediction_cumulative[:, :, None] >= progress_mm[None, None, :], axis=1
    ).astype(np.int16)
    endpoint_command = np.asarray(
        index.arrays["actions"][command_rows[occurrence, prediction_action]],
        dtype=np.float32,
    )
    endpoint_tips = command_tips[occurrence, prediction_action]
    joint_delta = endpoint_command - acquisition_state
    tip_delta = endpoint_tips - acquisition_tips
    command_motion = acquisition_signature.copy()
    command_motion[:, 0:6] = acquisition_state
    command_motion[:, 6:12] = joint_delta
    command_motion[:, 12:18] = endpoint_command
    command_motion[:, 18:24] = joint_delta
    command_motion[:, 24:30] = endpoint_tips.reshape(len(selected), 6)
    command_motion[:, 30:36] = tip_delta.reshape(len(selected), 6)
    command_motion[:, 36:42] = 0.0
    command_motion[:, 42:48] = tip_delta.reshape(len(selected), 6)
    command_motion[:, 60:62] = 0.0
    command_motion[:, 62:64] = np.linalg.norm(tip_delta, axis=-1)

    return {
        "input_occurrence_index": occurrence[valid].astype(np.int64, copy=False),
        "window_index": selected[valid].astype(np.int64, copy=False),
        "acquisition_endpoint_index": acquisition_endpoint[valid].astype(
            np.int16, copy=False
        ),
        "prediction_endpoint_index": prediction_endpoint[valid].astype(
            np.int16, copy=False
        ),
        "action_progress_index": progress_indices[valid].astype(np.int16, copy=False),
        "command_motion_endpoint": command_motion[valid].astype(np.float16),
    }


def build_causal_displacement_observer_ledger(
    index: Any,
    *,
    schedule: RGBSchedule,
    kinematics: Any,
    segment_mm: float = DISPLACEMENT_SEGMENT_MM,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Build two independently accumulated 100 mm events per occurrence."""

    if segment_mm != DISPLACEMENT_SEGMENT_MM:
        raise ValueError("the observer fixes both displacement segments at 100 mm")
    windows, view_seeds = _scheduled_occurrences(index, schedule)
    event = causal_displacement_events_for_windows(
        index, windows, kinematics=kinematics, segment_mm=segment_mm
    )
    if len(event["window_index"]) == 0:
        raise ValueError("RGB schedule contains no causal acquire/predict event pair")
    accepted_occurrence = event.pop("input_occurrence_index")
    arrays = {
        "window_index": event["window_index"],
        "view_selection_seed": view_seeds[accepted_occurrence].astype(
            np.int64, copy=False
        ),
        "acquisition_endpoint_index": event["acquisition_endpoint_index"],
        "prediction_endpoint_index": event["prediction_endpoint_index"],
        "action_progress_index": event["action_progress_index"],
        "command_motion_endpoint": event["command_motion_endpoint"],
    }
    accepted_windows = arrays["window_index"]
    sources = np.asarray(
        index.arrays["window_source_index"][accepted_windows], dtype=np.int64
    )
    tasks = np.asarray(
        index.arrays["window_task_index"][accepted_windows], dtype=np.int64
    )
    acquisition = arrays["acquisition_endpoint_index"]
    prediction = arrays["prediction_endpoint_index"]
    manifest: dict[str, object] = {
        "format": CAUSAL_DISPLACEMENT_OBSERVER_LEDGER_FORMAT,
        "protocol_id": CAUSAL_DISPLACEMENT_OBSERVER_PROTOCOL_ID,
        "segment_mm": float(segment_mm),
        "acquisition_metric": (
            "first_passage_from_frame0_sum_t_max_tip_command_CAD_FK_arc"
        ),
        "prediction_metric": (
            "reset_at_measured_acquisition_state_then_first_passage_"
            "sum_t_max_tip_command_CAD_FK_arc"
        ),
        "query_selector_last_visible_frame": "acquisition_endpoint",
        "prediction_target_state_used_by_prior": False,
        "schedule": asdict(schedule),
        "considered_occurrences": int(len(windows)),
        "accepted_occurrences": int(len(accepted_windows)),
        "accepted_unique_windows": int(len(np.unique(accepted_windows))),
        "source_count": int(len(np.unique(sources))),
        "task_count": int(len(np.unique(tasks))),
        "source_task_count": int(
            len(np.unique(np.stack((sources, tasks), axis=-1), axis=0))
        ),
        "acquisition_endpoint_quantiles": np.quantile(
            acquisition, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        ).tolist(),
        "prediction_endpoint_quantiles": np.quantile(
            prediction, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        ).tolist(),
        "prediction_interval_frame_quantiles": np.quantile(
            prediction - acquisition, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        ).tolist(),
        "kinematics_scene": str(kinematics.model_path),
        "kinematics_scene_sha256": sha256_file(Path(kinematics.model_path)),
    }
    return manifest, arrays


def write_causal_displacement_observer_ledger(
    output: Path,
    *,
    manifest: dict[str, object],
    arrays: dict[str, np.ndarray],
    index_manifest: Path,
    cluster_manifest: Path,
    git: dict[str, object],
) -> dict[str, object]:
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name, value in arrays.items():
            np.save(temporary / f"{name}.npy", value, allow_pickle=False)
        payload = dict(manifest)
        payload.update(
            {
                "git": git,
                "temporal_index_manifest_sha256": sha256_file(index_manifest),
                "motion_cluster_manifest_sha256": sha256_file(cluster_manifest),
                "arrays": {
                    name: {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "sha256": sha256_file(temporary / f"{name}.npy"),
                    }
                    for name, value in arrays.items()
                },
            }
        )
        payload["content_sha256"] = sha256_json(payload)
        (temporary / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        temporary.rename(output)
        return payload
    except Exception:
        for path in temporary.glob("*"):
            path.unlink(missing_ok=True)
        temporary.rmdir()
        raise


def exclude_causal_observer_media_sources(
    index: Any,
    *,
    manifest: dict[str, object],
    arrays: dict[str, np.ndarray],
    source_indices: tuple[int, ...],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Fail closed on source-level media faults before RGB materialization."""

    excluded = tuple(sorted(set(map(int, source_indices))))
    if not excluded:
        return dict(manifest), arrays
    source_names = tuple(map(str, index.manifest.get("source_names", ())))
    if not source_names or any(value < 0 or value >= len(source_names) for value in excluded):
        raise ValueError("media-excluded source index is outside the temporal index")
    windows = np.asarray(arrays["window_index"], dtype=np.int64)
    sources = np.asarray(
        index.arrays["window_source_index"][windows], dtype=np.int64
    )
    keep = ~np.isin(sources, np.asarray(excluded, dtype=np.int64))
    removed = int((~keep).sum())
    if removed == 0:
        raise ValueError("media exclusion did not remove any observer occurrence")
    if not bool(keep.any()):
        raise ValueError("media exclusion removed every observer occurrence")
    filtered = {name: np.asarray(value[keep]).copy() for name, value in arrays.items()}
    kept_windows = np.asarray(filtered["window_index"], dtype=np.int64)
    kept_sources = np.asarray(
        index.arrays["window_source_index"][kept_windows], dtype=np.int64
    )
    kept_tasks = np.asarray(
        index.arrays["window_task_index"][kept_windows], dtype=np.int64
    )
    acquisition = np.asarray(filtered["acquisition_endpoint_index"])
    prediction = np.asarray(filtered["prediction_endpoint_index"])
    payload = dict(manifest)
    payload.update(
        {
            "accepted_occurrences_before_media_exclusion": int(len(windows)),
            "accepted_occurrences": int(len(kept_windows)),
            "accepted_unique_windows": int(len(np.unique(kept_windows))),
            "source_count": int(len(np.unique(kept_sources))),
            "task_count": int(len(np.unique(kept_tasks))),
            "source_task_count": int(
                len(
                    np.unique(
                        np.stack((kept_sources, kept_tasks), axis=-1), axis=0
                    )
                )
            ),
            "media_excluded_occurrences": removed,
            "media_excluded_source_indices": list(excluded),
            "media_excluded_source_names": [source_names[value] for value in excluded],
            "media_exclusion_policy": "exclude_entire_source_after_exact_video_bounds_failure",
            "acquisition_endpoint_quantiles": np.quantile(
                acquisition, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            ).tolist(),
            "prediction_endpoint_quantiles": np.quantile(
                prediction, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            ).tolist(),
            "prediction_interval_frame_quantiles": np.quantile(
                prediction - acquisition,
                (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
            ).tolist(),
        }
    )
    return payload, filtered


class CausalDisplacementObserverLedger:
    """Read-only two-event ledger bound to the cached RGB schedule."""

    ARRAY_NAMES = (
        "window_index",
        "view_selection_seed",
        "acquisition_endpoint_index",
        "prediction_endpoint_index",
        "action_progress_index",
        "command_motion_endpoint",
    )

    def __init__(
        self,
        root: Path,
        *,
        index_manifest: Path,
        cluster_manifest: Path,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != CAUSAL_DISPLACEMENT_OBSERVER_LEDGER_FORMAT:
            raise ValueError("causal displacement observer ledger format mismatch")
        if self.manifest.get("protocol_id") != CAUSAL_DISPLACEMENT_OBSERVER_PROTOCOL_ID:
            raise ValueError("causal displacement observer protocol mismatch")
        if float(self.manifest.get("segment_mm", -1)) != DISPLACEMENT_SEGMENT_MM:
            raise ValueError("causal displacement observer segment mismatch")
        if self.manifest.get("temporal_index_manifest_sha256") != sha256_file(
            index_manifest
        ):
            raise ValueError("observer ledger/index binding mismatch")
        if self.manifest.get("motion_cluster_manifest_sha256") != sha256_file(
            cluster_manifest
        ):
            raise ValueError("observer ledger/cluster binding mismatch")
        self.arrays = {
            name: np.load(self.root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in self.ARRAY_NAMES
        }
        count = int(self.manifest["accepted_occurrences"])
        expected_shapes = {
            "window_index": (count,),
            "view_selection_seed": (count,),
            "acquisition_endpoint_index": (count,),
            "prediction_endpoint_index": (count,),
            "action_progress_index": (count, ACTION_PROGRESS_SAMPLES),
            "command_motion_endpoint": (count, 64),
        }
        if any(
            self.arrays[name].shape != shape
            for name, shape in expected_shapes.items()
        ):
            raise ValueError("observer ledger arrays are misaligned")
        for name, value in self.arrays.items():
            expected = self.manifest["arrays"][name]
            if list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]:
                raise ValueError(f"observer ledger {name} metadata mismatch")
            if sha256_file(self.root / f"{name}.npy") != expected["sha256"]:
                raise ValueError(f"observer ledger {name} hash mismatch")
        acquisition = np.asarray(self.arrays["acquisition_endpoint_index"])
        prediction = np.asarray(self.arrays["prediction_endpoint_index"])
        progress = np.asarray(self.arrays["action_progress_index"])
        if bool(
            ((acquisition < 1) | (acquisition > 14)).any()
            or ((prediction <= acquisition) | (prediction > 15)).any()
            or ((progress < acquisition[:, None]) | (progress >= prediction[:, None])).any()
            or (progress[:, 1:] < progress[:, :-1]).any()
        ):
            raise ValueError("observer ledger violates causal event order")


class CausalDisplacementObserverSampler:
    """Source/task-balanced deterministic sampler over two-event rows."""

    def __init__(
        self, index: Any, ledger: CausalDisplacementObserverLedger, *, seed: int
    ) -> None:
        self.index = index
        self.ledger = ledger
        self.rng = np.random.default_rng(seed)
        windows = np.asarray(ledger.arrays["window_index"], dtype=np.int64)
        sources = np.asarray(index.arrays["window_source_index"][windows], dtype=np.int64)
        tasks = np.asarray(index.arrays["window_task_index"][windows], dtype=np.int64)
        grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
        for row, group in enumerate(zip(sources.tolist(), tasks.tolist(), strict=True)):
            grouped[group].append(row)
        self.groups = {
            group: np.asarray(rows, dtype=np.int64)[self.rng.permutation(len(rows))]
            for group, rows in grouped.items()
        }
        keys = tuple(sorted(self.groups))
        self.group_order = tuple(keys[int(i)] for i in self.rng.permutation(len(keys)))
        self.group_cursor = 0
        self.row_cursor = {group: 0 for group in self.groups}
        self.rows_sampled = 0

    def batch(self, batch_size: int) -> CausalDisplacementObserverBatch:
        if batch_size <= 0:
            raise ValueError("observer event batch size must be positive")
        selected: list[int] = []
        for _ in range(batch_size):
            group = self.group_order[self.group_cursor % len(self.group_order)]
            self.group_cursor += 1
            rows = self.groups[group]
            cursor = self.row_cursor[group]
            selected.append(int(rows[cursor % len(rows)]))
            self.row_cursor[group] = cursor + 1
        occurrence = np.asarray(selected, dtype=np.int64)
        self.rows_sampled += batch_size
        return CausalDisplacementObserverBatch(
            occurrence_indices=occurrence,
            window_indices=np.asarray(
                self.ledger.arrays["window_index"][occurrence], dtype=np.int64
            ),
            view_selection_seeds=np.asarray(
                self.ledger.arrays["view_selection_seed"][occurrence], dtype=np.int64
            ),
            acquisition_endpoint_indices=np.asarray(
                self.ledger.arrays["acquisition_endpoint_index"][occurrence],
                dtype=np.int64,
            ),
            prediction_endpoint_indices=np.asarray(
                self.ledger.arrays["prediction_endpoint_index"][occurrence],
                dtype=np.int64,
            ),
            action_progress_indices=np.asarray(
                self.ledger.arrays["action_progress_index"][occurrence], dtype=np.int64
            ),
            command_motion_endpoint=np.asarray(
                self.ledger.arrays["command_motion_endpoint"][occurrence],
                dtype=np.float32,
            ),
        )

    def stats(self) -> dict[str, int]:
        return {
            "eligible_occurrences": int(len(self.ledger.arrays["window_index"])),
            "source_task_groups": len(self.groups),
            "rows_sampled": self.rows_sampled,
        }


__all__ = [
    "CAUSAL_DISPLACEMENT_OBSERVER_CHECKPOINT_FORMAT",
    "CAUSAL_DISPLACEMENT_OBSERVER_PROTOCOL_ID",
    "CausalDisplacementObserverLedger",
    "CausalDisplacementObserverSampler",
    "build_causal_displacement_observer_ledger",
    "causal_displacement_events_for_windows",
    "causal_displacement_observer_contract",
    "exclude_causal_observer_media_sources",
    "write_causal_displacement_observer_ledger",
]
