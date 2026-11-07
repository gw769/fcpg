"""Frozen real-RGB candidate interface for the v4.6 PartGraph.

This module contains only prediction-blind construction and physical-control
contracts.  It never accepts a target pixel and cannot train a model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from fcpg.offline_pipeline_manifest import sha256_json
from fcpg.part_graph_track_association import (
    GLOBAL_ACQUISITION_SOURCE,
    LOCAL_PERSISTENCE_SOURCE,
)
from fcpg.query_initialized_displacement_panel import (
    QUERY_DISPLACEMENT_PANEL_FORMAT,
    QUERY_DISPLACEMENT_PROTOCOL_ID,
)
from fcpg.query_initialized_measurement_annotations import (
    QUERY_DISPLACEMENT_TRUTH_FORMAT,
)
from fcpg.temporal_corruptions import causal_trailing_blur, sample_smooth_jitter


REAL_RGB_V4_6_PROTOCOL_ID = (
    "fcpg-h-canonical-part-graph-v4.6-real-rgb-interface-v1"
)
REAL_RGB_V4_6_QUERY_FORMAT = "so101-fcpg-h-v4.6-real-rgb-queries-v1"
REAL_RGB_V4_6_PREDICTION_FORMAT = "so101-fcpg-h-v4.6-real-rgb-predictions-v1"
REAL_RGB_V4_6_REPORT_FORMAT = "so101-fcpg-h-v4.6-real-rgb-report-v1"

REAL_CANDIDATE_COUNT = 32
GLOBAL_CANDIDATE_COUNT = 16
LOCAL_CANDIDATES_PER_TIP = 8
REAL_HISTORY_SAMPLES = 11
ONLINE_TRACKER_WINDOW = 16
TIP_COUNT = 2
DISPLACEMENT_LEVELS_MM = np.linspace(0.0, 100.0, REAL_HISTORY_SAMPLES)
REAL_RGB_CONDITIONS = ("clean", "causal_blur", "smooth_jitter", "blur_jitter")
REAL_RGB_ASSOCIATION_METHODS = (
    "FCPGH",
    "H0",
    "reverse",
    "cyclic",
    "query_swap",
)
REAL_RGB_CORRUPTION_SEED_BASE = 12_692_000

FROZEN_PANEL_FILE_SHA256 = (
    "45fd3ff9358e4d1eb34922a792f04d0b832b94ca0bb4be8c901e73ab8099400a"
)
FROZEN_PANEL_CONTENT_SHA256 = (
    "e0254a32a38f5d38e670332929cacf2be644b5a992836bfe06d0b47c54fdf8b8"
)
FROZEN_IMAGE_MANIFEST_FILE_SHA256 = (
    "080432f92295593df4b1b4647fd825a21954e7a47047e091cf6893038a1c7923"
)
FROZEN_IMAGE_MANIFEST_CONTENT_SHA256 = (
    "8979f1cb306c7c21f7c479a4786307a004fc262537fa08c33e63cadfcc3fc568"
)
FROZEN_FCPGH_CHECKPOINT_SHA256 = (
    "0d99e0479fc72087797b8f14f7d1568305db65489ac46b3a08d8113102a300d6"
)
FROZEN_H0_CHECKPOINT_SHA256 = (
    "ba05fc0394b0b2ec2294b11e3be39269d4cbe9e2b791d27af565a7006217936b"
)
FROZEN_V4_6_GATE_SHA256 = (
    "63f3eca8713d068b8bfbee81223655da75a89d661632261c2924d7f27ef04fe3"
)
FROZEN_V12_4_SHA256 = (
    "aca718b0a7335fd3ed3afce4bf70548ede1e194b6f9f0463ba4de0a36379c339"
)
FROZEN_COTRACKER_COMMIT = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
FROZEN_COTRACKER_CHECKPOINT_SHA256 = (
    "205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218"
)


@dataclass(frozen=True)
class RealCandidateQueries:
    points_px: np.ndarray
    source: np.ndarray
    owner: np.ndarray
    primary_index: np.ndarray


@dataclass(frozen=True)
class DisplacementProgressSample:
    states: np.ndarray
    completed_actions: np.ndarray
    observation_indices: np.ndarray
    command_indices: np.ndarray
    cumulative_command_tip_arc_mm: np.ndarray


@dataclass(frozen=True)
class ConditionedCausalPrefix:
    rgb: np.ndarray
    query_tip_px: np.ndarray
    source_to_target_affine: np.ndarray
    blur_frames: int | None


def extract_query_only_payload(
    panel: dict[str, Any],
    truth: dict[str, Any],
    *,
    panel_file_sha256: str,
    truth_file_sha256: str,
) -> dict[str, Any]:
    """Remove every target coordinate before model inference."""

    if (
        panel.get("format") != QUERY_DISPLACEMENT_PANEL_FORMAT
        or panel.get("protocol_id") != QUERY_DISPLACEMENT_PROTOCOL_ID
        or panel.get("split") != "development"
        or not bool(panel.get("prediction_blind"))
    ):
        raise ValueError("unexpected/non-blind query displacement panel")
    if (
        truth.get("format") != QUERY_DISPLACEMENT_TRUTH_FORMAT
        or truth.get("protocol_id") != QUERY_DISPLACEMENT_PROTOCOL_ID
        or truth.get("panel_sha256") != panel_file_sha256
    ):
        raise ValueError("merged truth does not bind to the frozen panel")
    panel_keys = {
        (str(record["entry_id"]), view_index)
        for record in panel.get("records", [])
        for view_index, _ in enumerate(record.get("views", []))
    }
    rows = []
    seen: set[tuple[str, int]] = set()
    for record in truth.get("records", []):
        key = (str(record["entry_id"]), int(record["view_index"]))
        if key in seen:
            raise ValueError("merged truth repeats an entry/view")
        seen.add(key)
        query = _finite_array(record.get("query_tip_px"), (2, 2), "query_tip_px")
        if np.any(query < 0.0) or np.any(query > 224.0):
            raise ValueError("query tip lies outside the annotation image")
        rows.append(
            {
                "entry_id": key[0],
                "view_index": key[1],
                "query_tip_px": query.tolist(),
            }
        )
    if seen != panel_keys:
        raise ValueError("merged truth is incomplete or contains foreign keys")
    rows.sort(key=lambda row: (row["entry_id"], row["view_index"]))
    payload: dict[str, Any] = {
        "format": REAL_RGB_V4_6_QUERY_FORMAT,
        "protocol_id": REAL_RGB_V4_6_PROTOCOL_ID,
        "source_protocol_id": QUERY_DISPLACEMENT_PROTOCOL_ID,
        "panel_file_sha256": panel_file_sha256,
        "truth_file_sha256_commitment": truth_file_sha256,
        "target_coordinates_present": False,
        "records": rows,
    }
    payload["content_sha256"] = sha256_json(payload)
    return payload


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def build_real_candidate_queries(query_tip_px: object) -> RealCandidateQueries:
    """Create the fixed 16-global + 2x8-local query bank.

    The local ring is regular rather than an axis-aligned grid so the seven
    non-primary candidates do not privilege one image direction.
    """

    tips = _finite_array(query_tip_px, (TIP_COUNT, 2), "query_tip_px")
    if np.any(tips < 0.0) or np.any(tips > 223.0):
        raise ValueError("query fingertips must lie inside the 224px image")
    grid_axis = np.asarray((28.0, 84.0, 140.0, 196.0), dtype=np.float32)
    global_points = np.stack(
        np.meshgrid(grid_axis, grid_axis, indexing="xy"), axis=-1
    ).reshape(-1, 2)
    angles = 2.0 * math.pi * np.arange(7, dtype=np.float32) / 7.0
    ring = 4.0 * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    local = np.concatenate(
        [
            np.concatenate((tips[tip : tip + 1], tips[tip : tip + 1] + ring), axis=0)
            for tip in range(TIP_COUNT)
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    # CoTracker accepts points slightly outside the frame poorly and the
    # canonical solver correctly rejects duplicate clipped tubes.  Fail closed
    # instead of clipping a pre-registered query bank.
    if np.any(local < 0.0) or np.any(local > 223.0):
        raise ValueError("radius-4 local query ring leaves the image")
    points = np.concatenate((global_points, local), axis=0)
    if points.shape != (REAL_CANDIDATE_COUNT, 2):
        raise AssertionError("real candidate count drifted")
    source = np.concatenate(
        (
            np.full(GLOBAL_CANDIDATE_COUNT, GLOBAL_ACQUISITION_SOURCE),
            np.full(2 * LOCAL_CANDIDATES_PER_TIP, LOCAL_PERSISTENCE_SOURCE),
        )
    ).astype(np.int64)
    owner = np.concatenate(
        (
            np.full(GLOBAL_CANDIDATE_COUNT, -1),
            np.repeat(np.arange(TIP_COUNT), LOCAL_CANDIDATES_PER_TIP),
        )
    ).astype(np.int64)
    primary = np.asarray(
        (
            GLOBAL_CANDIDATE_COUNT,
            GLOBAL_CANDIDATE_COUNT + LOCAL_CANDIDATES_PER_TIP,
        ),
        dtype=np.int64,
    )
    if not np.array_equal(points[primary], tips):
        raise AssertionError("primary candidates are not the exact query tips")
    return RealCandidateQueries(points, source, owner, primary)


def real_rgb_corruption_seed(
    entry_ordinal: int,
    view_index: int,
    condition_index: int,
) -> int:
    """Return the frozen, collision-free seed for one panel row."""

    if entry_ordinal < 0 or view_index not in (0, 1):
        raise ValueError("invalid panel entry/view ordinal")
    if not 0 <= condition_index < len(REAL_RGB_CONDITIONS):
        raise ValueError("invalid corruption index")
    return (
        REAL_RGB_CORRUPTION_SEED_BASE
        + 100 * int(entry_ordinal)
        + 10 * int(view_index)
        + int(condition_index)
    )


def _transform_points_by_affine(
    points_px: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float32)
    matrix = np.asarray(affine, dtype=np.float32)
    if points.ndim != 2 or points.shape[-1] != 2 or matrix.shape != (2, 3):
        raise ValueError("points/affine must be [N,2] and [2,3]")
    homogeneous = np.concatenate(
        (points, np.ones((len(points), 1), dtype=np.float32)), axis=-1
    )
    return np.einsum("ij,nj->ni", matrix, homogeneous).astype(
        np.float32, copy=False
    )


def apply_prediction_blind_condition(
    rgb: object,
    query_tip_px: object,
    *,
    condition: str,
    seed: int,
) -> ConditionedCausalPrefix:
    """Corrupt a causal RGB prefix and transform only its known query clicks.

    There is deliberately no target-coordinate parameter.  The returned
    endpoint affine is sufficient for the later, independent scorer to
    transform target truth after predictions have been frozen.
    """

    frames = np.asarray(rgb)
    tips = _finite_array(query_tip_px, (TIP_COUNT, 2), "query_tip_px")
    if (
        frames.ndim != 4
        or frames.shape[1:] != (224, 224, 3)
        or frames.dtype != np.uint8
        or not 2 <= len(frames) <= 11
    ):
        raise ValueError("real RGB prefix must be uint8 [T,224,224,3], 2<=T<=11")
    if condition not in REAL_RGB_CONDITIONS:
        raise ValueError("unknown real-RGB corruption condition")
    if seed < 0:
        raise ValueError("corruption seed must be nonnegative")
    images = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)
    images = images[None, :, None].float().div_(255.0)
    blur_frames: int | None = None
    if condition in {"causal_blur", "blur_jitter"}:
        blur_frames = 3 if seed % 2 == 0 else 5
        images = causal_trailing_blur(images, frames=blur_frames)
    affine = torch.eye(2, 3, dtype=torch.float32)[None].repeat(len(frames), 1, 1)
    if condition in {"smooth_jitter", "blur_jitter"}:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        images, sampled_affine = sample_smooth_jitter(
            images, generator=generator
        )
        affine = sampled_affine[0, :, 0].float().cpu()
    transformed_query = _transform_points_by_affine(tips, affine[0].numpy())
    output = (
        images[0, :, 0]
        .mul(255.0)
        .round()
        .clamp_(0.0, 255.0)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .numpy()
    )
    return ConditionedCausalPrefix(
        rgb=output,
        query_tip_px=transformed_query,
        source_to_target_affine=affine.numpy().astype(np.float32, copy=False),
        blur_frames=blur_frames,
    )


def resample_real_candidate_tracks(
    tracks_px: object,
    visible: object,
    observation_indices: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Put causal online tracks on the frozen 0..100mm progress clock."""

    tracks = np.asarray(tracks_px, dtype=np.float32)
    mask = np.asarray(visible, dtype=bool)
    indices = np.asarray(observation_indices, dtype=np.int64)
    if tracks.ndim != 3 or tracks.shape[1:] != (REAL_CANDIDATE_COUNT, 2):
        raise ValueError("online tracks must be [T,32,2]")
    if mask.shape != tracks.shape[:2]:
        raise ValueError("online visibility must be [T,32]")
    if indices.shape != (REAL_HISTORY_SAMPLES,):
        raise ValueError("observation indices must contain eleven levels")
    if (
        indices[0] != 0
        or indices[-1] != len(tracks) - 1
        or np.any(indices[1:] < indices[:-1])
        or np.any(indices < 0)
        or np.any(indices >= len(tracks))
    ):
        raise ValueError("progress indices violate the causal query/target prefix")
    sampled_tracks = tracks[indices].transpose(1, 0, 2).copy()
    sampled_visible = mask[indices].transpose(1, 0).copy()
    if not np.all(np.isfinite(sampled_tracks)):
        raise ValueError("online tracker returned non-finite coordinates")
    # The user-provided query measurement is known exactly at displacement 0.
    sampled_visible[:, 0] = True
    return sampled_tracks, sampled_visible


def causal_left_pad_online_window(
    rgb: object,
    *,
    window: int = ONLINE_TRACKER_WINDOW,
) -> tuple[np.ndarray, int]:
    """Left-pad with the query frame; never read or create a post-target frame."""

    frames = np.asarray(rgb)
    if (
        frames.ndim != 4
        or frames.shape[1:] != (224, 224, 3)
        or frames.dtype != np.uint8
    ):
        raise ValueError("rgb must be uint8 [T,224,224,3]")
    if not 2 <= len(frames) <= window:
        raise ValueError("real candidate prefix must contain 2..window frames")
    padding = window - len(frames)
    padded = np.concatenate(
        (np.repeat(frames[:1], padding, axis=0), frames), axis=0
    )
    if not np.array_equal(padded[padding:], frames):
        raise AssertionError("online window changed causal source frames")
    if padding and not np.array_equal(
        padded[:padding], np.repeat(frames[:1], padding, axis=0)
    ):
        raise AssertionError("online padding is not the query frame")
    return padded, padding


def displacement_progress_sample(
    states: object,
    completed_actions: object,
    *,
    kinematics: object,
    levels_mm: np.ndarray = DISPLACEMENT_LEVELS_MM,
) -> DisplacementProgressSample:
    """Sample a variable-duration command-CAD event at fixed arc progress."""

    state = np.asarray(states, dtype=np.float32)
    action = np.asarray(completed_actions, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 6 or not 2 <= len(state) <= 11:
        raise ValueError("states must be [T,6] with 2<=T<=11")
    if action.shape != (len(state) - 1, 6):
        raise ValueError("completed_actions must be [T-1,6]")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        raise ValueError("state/action event contains non-finite values")
    levels = np.asarray(levels_mm, dtype=np.float32)
    if (
        levels.shape != (REAL_HISTORY_SAMPLES,)
        or levels[0] != 0.0
        or levels[-1] != 100.0
        or np.any(np.diff(levels) <= 0.0)
    ):
        raise ValueError("displacement levels must be strict 0..100mm samples")
    command_tips, _ = kinematics.forward_tips_and_jacobians(action)
    initial_tips, _ = kinematics.forward_tips_and_jacobians(state[:1])
    command_tips = np.asarray(command_tips, dtype=np.float32)
    initial_tips = np.asarray(initial_tips, dtype=np.float32)[0]
    if command_tips.shape != (len(action), TIP_COUNT, 3):
        raise ValueError("kinematics returned unexpected command-tip shape")
    command_path = np.concatenate((initial_tips[None], command_tips), axis=0)
    step_mm = (
        np.linalg.norm(np.diff(command_path, axis=0), axis=-1).max(axis=-1)
        * 1000.0
    )
    cumulative = np.cumsum(step_mm, dtype=np.float64)
    if not np.all(np.isfinite(cumulative)) or cumulative[-1] < 100.0 - 1e-3:
        raise ValueError("panel transition does not reach the frozen 100mm event")
    command_index = np.searchsorted(cumulative, levels[1:], side="left")
    command_index = np.minimum(command_index, len(action) - 1).astype(np.int64)
    observation_index = np.concatenate(
        (np.zeros(1, dtype=np.int64), command_index + 1)
    )
    if observation_index[-1] != len(state) - 1:
        raise ValueError("panel target is not the first frozen 100mm crossing")
    sampled_state = state[observation_index].copy()
    sampled_action = action[command_index].copy()
    return DisplacementProgressSample(
        states=sampled_state,
        completed_actions=sampled_action,
        observation_indices=observation_index,
        command_indices=command_index,
        cumulative_command_tip_arc_mm=cumulative.astype(np.float32),
    )


def endpoint_matched_increment_control(
    states: object,
    completed_actions: object,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder physical increments while retaining both endpoints exactly."""

    state = _finite_array(states, (REAL_HISTORY_SAMPLES, 6), "states")
    action = _finite_array(
        completed_actions,
        (REAL_HISTORY_SAMPLES - 1, 6),
        "completed_actions",
    )
    # The final increment is held fixed.  Reordering only the first nine
    # increments preserves both the penultimate and current state/command, so
    # the complete 44-D current PartGraph row is not a control cue.
    interior = np.arange(REAL_HISTORY_SAMPLES - 2)
    if mode == "reverse":
        changed_interior = interior[::-1]
    elif mode == "cyclic":
        changed_interior = np.roll(interior, 1)
    else:
        raise ValueError("increment control mode must be reverse or cyclic")
    order = np.concatenate(
        (changed_interior, np.asarray((REAL_HISTORY_SAMPLES - 2,)))
    )
    state_increment = np.diff(state, axis=0)
    command_path = np.concatenate((state[:1], action), axis=0)
    command_increment = np.diff(command_path, axis=0)
    changed_state = np.concatenate(
        (state[:1], state[:1] + np.cumsum(state_increment[order], axis=0)), axis=0
    ).astype(np.float32)
    changed_action = (
        state[:1] + np.cumsum(command_increment[order], axis=0)
    ).astype(np.float32)
    if not np.allclose(changed_state[-1], state[-1], atol=2e-5, rtol=0.0):
        raise AssertionError("state increment control changed its endpoint")
    if not np.allclose(changed_action[-1], action[-1], atol=2e-5, rtol=0.0):
        raise AssertionError("command increment control changed its endpoint")
    # Overwrite accumulated fp32 roundoff so current-row controls are exact.
    changed_state[0] = state[0]
    changed_state[-1] = state[-1]
    changed_action[-1] = action[-1]
    changed_state[-2] = state[-2]
    changed_action[-2] = action[-2]
    return changed_state, changed_action


def forced_primary_dropout_pattern(
    entry_ordinal: int,
    view_index: int,
    condition_index: int,
) -> np.ndarray:
    """Return balanced none/fixed/moving/both forced fallback flags."""

    if entry_ordinal < 0 or view_index not in (0, 1):
        raise ValueError("invalid panel entry/view ordinal")
    if not 0 <= condition_index < len(REAL_RGB_CONDITIONS):
        raise ValueError("invalid corruption index")
    patterns = np.asarray(
        ((False, False), (True, False), (False, True), (True, True)),
        dtype=bool,
    )
    return patterns[(2 * entry_ordinal + view_index + condition_index) % 4].copy()


def apply_primary_dropout(
    candidate_valid: torch.Tensor,
    primary_candidate_index: torch.Tensor,
    forced_fallback: torch.Tensor,
) -> torch.Tensor:
    """Invalidate a forced primary for exactly the final two progress samples."""

    valid = torch.as_tensor(candidate_valid)
    primary = torch.as_tensor(primary_candidate_index)
    forced = torch.as_tensor(forced_fallback)
    if valid.ndim != 3 or valid.dtype != torch.bool:
        raise ValueError("candidate_valid must be bool [B,32,11]")
    batch, candidates, time = valid.shape
    if (candidates, time) != (REAL_CANDIDATE_COUNT, REAL_HISTORY_SAMPLES):
        raise ValueError("candidate validity differs from the real interface")
    if primary.shape != (batch, TIP_COUNT) or primary.dtype != torch.long:
        raise ValueError("primary_candidate_index must be int64 [B,2]")
    if forced.shape != (batch, TIP_COUNT) or forced.dtype != torch.bool:
        raise ValueError("forced_fallback must be bool [B,2]")
    result = valid.clone()
    for row in range(batch):
        for tip in range(TIP_COUNT):
            if bool(forced[row, tip]):
                result[row, int(primary[row, tip]), -2:] = False
    return result


__all__ = [
    "DISPLACEMENT_LEVELS_MM",
    "ConditionedCausalPrefix",
    "DisplacementProgressSample",
    "FROZEN_COTRACKER_CHECKPOINT_SHA256",
    "FROZEN_COTRACKER_COMMIT",
    "FROZEN_FCPGH_CHECKPOINT_SHA256",
    "FROZEN_H0_CHECKPOINT_SHA256",
    "FROZEN_IMAGE_MANIFEST_CONTENT_SHA256",
    "FROZEN_IMAGE_MANIFEST_FILE_SHA256",
    "FROZEN_PANEL_CONTENT_SHA256",
    "FROZEN_PANEL_FILE_SHA256",
    "FROZEN_V12_4_SHA256",
    "FROZEN_V4_6_GATE_SHA256",
    "GLOBAL_CANDIDATE_COUNT",
    "LOCAL_CANDIDATES_PER_TIP",
    "ONLINE_TRACKER_WINDOW",
    "REAL_CANDIDATE_COUNT",
    "REAL_HISTORY_SAMPLES",
    "REAL_RGB_ASSOCIATION_METHODS",
    "REAL_RGB_CONDITIONS",
    "REAL_RGB_V4_6_PREDICTION_FORMAT",
    "REAL_RGB_V4_6_PROTOCOL_ID",
    "REAL_RGB_V4_6_QUERY_FORMAT",
    "REAL_RGB_V4_6_REPORT_FORMAT",
    "RealCandidateQueries",
    "apply_primary_dropout",
    "apply_prediction_blind_condition",
    "build_real_candidate_queries",
    "causal_left_pad_online_window",
    "displacement_progress_sample",
    "endpoint_matched_increment_control",
    "forced_primary_dropout_pattern",
    "real_rgb_corruption_seed",
    "resample_real_candidate_tracks",
]
