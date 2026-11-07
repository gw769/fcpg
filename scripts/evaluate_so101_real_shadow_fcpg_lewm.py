#!/usr/bin/env python3
"""Run FCPG and LeWM ranking on a paired real SO-101 shadow capture."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw

from fcpg.so101_active_gripper import apply_live_similarity_transform
from fcpg.so101_camera_calibration import (
    ProjectionFit,
    predict_projection,
    valid_projection_mask,
)
from fcpg.coarse_fine_planner import (
    apply_coarse_fine_observe_points,
    assert_coarse_fine_planner_policy,
    planner_grounding_ranking_contract,
)
from fcpg.grounding_confidence_fusion import (
    COARSE_FINE_VARIANT,
    GROUNDING_GATED_FCPG,
    GROUNDING_NOMINAL,
    RANKING_ON,
    GroundingFusionState,
)
from fcpg.so101_fcpg_lewm_bridge import (
    SO101FCPGConditionedLeWM,
    build_so101_fcpg_corrector,
    infer_so101_fcpg_corrector_variant,
)
from fcpg.so101_kinematics import SO101_ANCHOR_NAMES, SO101MuJoCoKinematics
from fcpg.so101_retrieval_mpc import (
    SO101_POSITION_LOWER,
    SO101_POSITION_UPPER,
    candidate_safety_checks,
    candidate_start_rows,
    chunk_and_normalize_actions,
    raw_action_prefixes,
    rebase_action_prefixes,
    retrieve_diverse_candidates,
    terminal_goal_cost,
    terminal_goal_rows,
    terminal_state_cost,
    validate_paired_camera_data,
)
from fcpg.so101_shadow_session import SO101ShadowSession, load_shadow_session
from fcpg.so101_state_residual_lewm import StateResidualLeWM
from scripts.evaluate_so101_retrieval_mpc import (
    _display_neighbor,
    _rollout_candidates,
    _training_episodes,
)
from scripts.train_so101_fcpg_context_bridge import patch_feature_map
from scripts.train_so101_lewm import episode_splits, parse_exclusions
from scripts.train_so101_synthetic_fcpg import normalize_rgb
from scripts.visualize_so101_lewm import (
    CacheView,
    build_checkpoint_model,
    checkpoint_action_normalizer,
    checkpoint_contract,
    encode_rows,
    load_caches,
    normalized_state_rows,
)


VIEWS = ("wrist", "column")
COLORS = ((32, 181, 205), (255, 196, 77), (237, 92, 92))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--bridge-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--secondary-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--history-stride", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidates-per-episode", type=int, default=8)
    parser.add_argument("--candidate-min-gap", type=int, default=20)
    parser.add_argument("--goal-tail-frames", type=int, default=15)
    parser.add_argument("--goal-final-steps", type=int, default=3)
    parser.add_argument("--state-retrieval-weight", type=float, default=1.0)
    parser.add_argument("--state-goal-weight", type=float, default=1.0)
    parser.add_argument("--secondary-visual-goal-weight", type=float, default=1.0)
    parser.add_argument("--max-first-command-jump", type=float, default=8.0)
    parser.add_argument("--max-prefix-step-jump", type=float, default=8.0)
    parser.add_argument("--prediction-bound-margin", type=float, default=0.0)
    parser.add_argument("--coarse-register-column", action="store_true")
    parser.add_argument("--retrieval-cost-weight", type=float, default=1.0)
    parser.add_argument(
        "--goal-mode",
        choices=("demo-terminal", "current-fish", "current-box"),
        default="demo-terminal",
    )
    parser.add_argument("--red-object-phase-guard", action="store_true")
    parser.add_argument("--object-phase-weight", type=float, default=4.0)
    parser.add_argument("--max-object-phase-distance", type=float, default=0.08)
    parser.add_argument(
        "--grounding-mode",
        choices=("nominal", "gated_fcpg"),
        default="gated_fcpg",
    )
    parser.add_argument(
        "--ranking-mode",
        choices=("off", "on"),
        default="on",
    )
    parser.add_argument(
        "--allow-uncalibrated-coarse-fine-shadow",
        action="store_true",
        help=(
            "Permit an uncalibrated / no-physical-label coarse_fine checkpoint "
            "on this offline shadow path."
        ),
    )
    args = parser.parse_args()
    if args.device.startswith("cuda:1"):
        parser.error("CUDA1 is forbidden for this project")
    integer_values = (
        args.history_stride,
        args.horizon,
        args.batch_size,
        args.candidates_per_episode,
        args.goal_tail_frames,
        args.goal_final_steps,
    )
    if min(integer_values) <= 0 or args.candidate_min_gap < 0:
        parser.error("integer settings are invalid")
    if min(args.retrieval_cost_weight, args.object_phase_weight) < 0:
        parser.error("retrieval and object-phase weights must be non-negative")
    if args.max_prefix_step_jump <= 0 or args.max_object_phase_distance <= 0:
        parser.error("safety and object-phase thresholds must be positive")
    if (
        args.prediction_bound_margin < 0
        or not np.isfinite(args.prediction_bound_margin)
    ):
        parser.error("prediction-bound-margin must be finite and non-negative")
    if args.goal_mode != "demo-terminal" and args.coarse_register_column:
        parser.error(
            "current-scene goals must use the raw current column frame; "
            "coarse registration assumes a training-scene reference"
        )
    return args


def load_paired_session(root: Path) -> dict[str, SO101ShadowSession]:
    root = root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "so101-paired-shadow-session-v1":
        raise ValueError("real shadow evaluator requires a paired session")
    sessions = {view: load_shadow_session(root / view) for view in VIEWS}
    first = sessions[VIEWS[0]]
    for view in VIEWS[1:]:
        other = sessions[view]
        if not np.array_equal(first.states, other.states):
            raise ValueError("paired shadow states are not identical")
        if not np.array_equal(first.timestamps_s, other.timestamps_s):
            raise ValueError("paired shadow timestamps are not identical")
    return sessions


def load_projection_fits(
    path: Path, *, expected_geometry: str | None = None
) -> dict[str, ProjectionFit]:
    """Load the per-view projection fits, optionally under a geometry contract.

    A projection fit is a map from anchor coordinates to *pixels*, so it is
    only meaningful in the pixel geometry it was fitted in.  Historically this
    loader read ``feature_names``/``coefficients``/``ridge`` and nothing else,
    which let a calibration fitted under one geometry load silently into a
    runtime using another.  ``expected_geometry`` is the opt-in contract that
    closes that hole, and it is opt-in precisely because legacy and offline
    callers (including the frozen B0 Tier-2 replay test) load unversioned rig
    calibrations by design.

    Args:
        path: calibration JSON path.
        expected_geometry: ``None`` (default) keeps the historical permissive
            behavior -- the ``geometry_protocol`` field is not consulted at
            all.  ``letterbox_v1``/``squash_v0`` require the payload to
            *declare* that exact geometry and fail closed otherwise.

    Raises:
        ValueError: if ``expected_geometry`` is not ``None`` and the payload
            either declares no geometry or declares a different one; also if
            ``expected_geometry`` itself is not a known protocol.  Geometry is
            never inferred: an unversioned fit is not assumed to be either
            protocol, because a 224x224 canvas is consistent with both.
    """

    from fcpg.letterbox_geometry import LETTERBOX_V1, SQUASH_V0

    if expected_geometry is not None and expected_geometry not in (
        SQUASH_V0,
        LETTERBOX_V1,
    ):
        raise ValueError(
            f"expected_geometry must be None, {SQUASH_V0!r} or {LETTERBOX_V1!r}, "
            f"got {expected_geometry!r}; a geometry is never inferred or guessed"
        )
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if expected_geometry is not None:
        remedy = (
            f"it must either go through the analytic {SQUASH_V0} -> "
            f"{LETTERBOX_V1} migration or be re-fitted in {LETTERBOX_V1} pixels"
            if expected_geometry == LETTERBOX_V1
            else f"it must declare {expected_geometry!r} explicitly"
        )
        declared = payload.get("geometry_protocol")
        if declared is None:
            raise ValueError(
                f"projection calibration {source} is unversioned: it carries no "
                "'geometry_protocol' field, so the pixel geometry its "
                f"coefficients were fitted in is undeclared, while "
                f"{expected_geometry!r} was required. An unversioned fit is "
                f"never reinterpreted as {expected_geometry!r} -- {remedy}."
            )
        if declared != expected_geometry:
            raise ValueError(
                f"projection calibration {source} declares "
                f"geometry_protocol={declared!r}, but {expected_geometry!r} was "
                f"required; pixel fits are not portable across geometries, so "
                f"{remedy}."
            )
    result: dict[str, ProjectionFit] = {}
    for view in VIEWS:
        row = payload["fits"][view]
        result[view] = ProjectionFit(
            view=view,
            feature_names=tuple(row["feature_names"]),
            coefficients=np.asarray(row["coefficients"], dtype=np.float64),
            ridge=float(row["ridge"]),
        )
    return result


def project_history(
    states: np.ndarray,
    fits: dict[str, ProjectionFit],
    kinematics: SO101MuJoCoKinematics,
    *,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    base_xyz, gripper_xyz = kinematics.forward_anchor_frames(states)
    nominal = {
        view: predict_projection(
            fits[view], base_xyz, gripper_xyz, states[:, -1]
        ).astype(np.float32)
        for view in VIEWS
    }
    valid = {
        view: valid_projection_mask(points, image_size=image_size)
        for view, points in nominal.items()
    }
    return base_xyz.astype(np.float32), nominal, valid


def estimate_coarse_affine(
    reference: np.ndarray,
    current: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    detector = cv2.ORB_create(nfeatures=2000, fastThreshold=5)
    reference_gray = cv2.cvtColor(np.asarray(reference), cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(np.asarray(current), cv2.COLOR_RGB2GRAY)
    reference_keypoints, reference_descriptors = detector.detectAndCompute(
        reference_gray, None
    )
    current_keypoints, current_descriptors = detector.detectAndCompute(
        current_gray, None
    )
    if reference_descriptors is None or current_descriptors is None:
        raise ValueError("column registration lacks ORB descriptors")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        reference_descriptors, current_descriptors, k=2
    )
    matches = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    if len(matches) < 6:
        raise ValueError(f"column registration has only {len(matches)} ratio-test matches")
    source = np.float32(
        [reference_keypoints[match.queryIdx].pt for match in matches]
    )
    target = np.float32([current_keypoints[match.trainIdx].pt for match in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.999,
    )
    if matrix is None or inliers is None:
        raise ValueError("column registration could not estimate an affine transform")
    inlier_count = int(inliers.sum())
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    rotation_degrees = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    translation = float(np.linalg.norm(matrix[:, 2]))
    if inlier_count < 6:
        raise ValueError(f"column registration has only {inlier_count} RANSAC inliers")
    if not 0.7 <= scale <= 1.3 or abs(rotation_degrees) > 15.0 or translation > 80.0:
        raise ValueError("column registration transform is outside the safety envelope")
    return matrix.astype(np.float32), {
        "ratio_test_matches": len(matches),
        "inliers": inlier_count,
        "scale": scale,
        "rotation_degrees": rotation_degrees,
        "translation_px": translation,
    }


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    affine = np.asarray(matrix, dtype=np.float32)
    if values.shape[-1] != 2 or affine.shape != (2, 3):
        raise ValueError("affine point transform has incompatible shapes")
    return values @ affine[:, :2].T + affine[:, 2]


def align_current_to_reference(frames: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(frames, dtype=np.uint8)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("camera frames must have shape [T,H,W,3]")
    inverse = cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float32))
    height, width = values.shape[1:3]
    return np.stack(
        [
            cv2.warpAffine(
                frame,
                inverse,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            for frame in values
        ]
    ).astype(np.uint8, copy=False)


def red_object_descriptor(
    frame: np.ndarray, *, gripper_center_px: np.ndarray | None = None
) -> dict[str, Any]:
    rgb = np.asarray(frame, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("object-phase frame must be RGB")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 65, 45), (14, 255, 255)) | cv2.inRange(
        hsv, (165, 65, 45), (179, 255, 255)
    )
    components, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if components <= 1:
        return {"visible": False, "centroid_px": None, "area_px": 0}
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[component, cv2.CC_STAT_AREA])
    if area < 15:
        return {"visible": False, "centroid_px": None, "area_px": area}
    centroid = np.asarray(centroids[component], dtype=np.float32)
    result: dict[str, Any] = {
        "visible": True,
        "centroid_px": centroid,
        "area_px": area,
    }
    if gripper_center_px is not None:
        center = np.asarray(gripper_center_px, dtype=np.float32)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("gripper_center_px must be a finite 2-vector")
        result["gripper_center_px"] = center
        result["relative_centroid_px"] = centroid - center
    return result


def red_object_phase_distances(
    query: dict[str, Any],
    frames: np.ndarray,
    *,
    gripper_centers_px: np.ndarray | None = None,
) -> np.ndarray:
    if not query["visible"]:
        raise ValueError("red-object phase guard cannot see the current object")
    values = np.asarray(frames, dtype=np.uint8)
    if gripper_centers_px is not None:
        centers = np.asarray(gripper_centers_px, dtype=np.float32)
        if centers.shape != (len(values), 2) or not np.all(np.isfinite(centers)):
            raise ValueError("gripper_centers_px must have shape [N, 2]")
    else:
        centers = None
    height, width = values.shape[1:3]
    if gripper_centers_px is None:
        query_center = np.asarray(query["centroid_px"], dtype=np.float32)
    else:
        query_relative = query.get("relative_centroid_px")
        if query_relative is None:
            raise ValueError("query must contain relative_centroid_px")
        query_center = np.asarray(query_relative, dtype=np.float32)
    query_center = query_center / np.asarray(
        [max(width - 1, 1), max(height - 1, 1)], dtype=np.float32
    )
    query_area = np.log1p(float(query["area_px"])) / 10.0
    distance = np.empty(len(values), dtype=np.float32)
    for index, frame in enumerate(values):
        anchor = None
        if centers is not None:
            anchor = centers[index]
        descriptor = red_object_descriptor(frame, gripper_center_px=anchor)
        if not descriptor["visible"]:
            distance[index] = 4.0
            continue
        center_key = (
            "relative_centroid_px"
            if gripper_centers_px is not None
            else "centroid_px"
        )
        center = np.asarray(descriptor[center_key], dtype=np.float32) / np.asarray(
            [max(width - 1, 1), max(height - 1, 1)], dtype=np.float32
        )
        area = np.log1p(float(descriptor["area_px"])) / 10.0
        distance[index] = float(
            np.sum(np.square(center - query_center))
            + 0.25 * np.square(area - query_area)
        )
    return distance


def brown_box_descriptor(frame: np.ndarray) -> dict[str, Any]:
    """Locate the current cardboard box without assuming fixed table pixels."""

    rgb = np.asarray(frame, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("box frame must be RGB")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (5, 30, 20), (30, 255, 220))
    red = cv2.inRange(hsv, (0, 65, 45), (4, 255, 255)) | cv2.inRange(
        hsv, (165, 65, 45), (179, 255, 255)
    )
    mask[red > 0] = 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    components, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    height, width = rgb.shape[:2]
    candidates: list[int] = []
    for component in range(1, components):
        x, y, box_width, box_height, area = map(int, stats[component])
        touches_border = (
            x <= 1
            or y <= 1
            or x + box_width >= width - 1
            or y + box_height >= height - 1
        )
        if area >= 200 and box_width >= 15 and box_height >= 15 and not touches_border:
            candidates.append(component)
    if not candidates:
        return {"visible": False, "centroid_px": None, "area_px": 0, "bbox_xyxy": None}
    component = max(candidates, key=lambda item: int(stats[item, cv2.CC_STAT_AREA]))
    x, y, box_width, box_height, area = map(int, stats[component])
    return {
        "visible": True,
        "centroid_px": np.asarray(centroids[component], dtype=np.float32),
        "area_px": area,
        "bbox_xyxy": np.asarray(
            [x, y, x + box_width, y + box_height], dtype=np.int32
        ),
    }


def project_future_anchor_pixels(
    predicted_states: np.ndarray,
    fit: ProjectionFit,
    kinematics: SO101MuJoCoKinematics,
    *,
    current_correction_px: np.ndarray,
    live_transform_px: np.ndarray | None = None,
) -> np.ndarray:
    """Project LeWM future states into the current camera's local coordinates."""

    states = np.asarray(predicted_states, dtype=np.float32)
    if states.ndim != 3 or states.shape[-1] != 6:
        raise ValueError("predicted_states must have shape [candidates, horizon, 6]")
    correction = np.asarray(current_correction_px, dtype=np.float32)
    if correction.shape != (3, 2) or not np.all(np.isfinite(correction)):
        raise ValueError("current_correction_px must have shape [3, 2]")
    flat = states.reshape(-1, 6)
    base_xyz, gripper_xyz = kinematics.forward_anchor_frames(flat)
    nominal = predict_projection(fit, base_xyz, gripper_xyz, flat[:, -1])
    if live_transform_px is not None:
        nominal = apply_live_similarity_transform(nominal, live_transform_px)
    corrected = nominal + correction[None]
    return corrected.reshape(*states.shape[:2], 3, 2).astype(np.float32)


def current_pixel_goal_cost(
    predicted_anchor_px: np.ndarray,
    target_px: np.ndarray,
    *,
    final_steps: int,
    image_size: tuple[int, int],
) -> np.ndarray:
    points = np.asarray(predicted_anchor_px, dtype=np.float32)
    target = np.asarray(target_px, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("predicted_anchor_px must have shape [candidates, horizon, 2]")
    if target.shape != (2,) or not np.all(np.isfinite(target)):
        raise ValueError("target_px must be a finite 2-vector")
    if final_steps <= 0 or final_steps > points.shape[1]:
        raise ValueError("final_steps is outside the prediction horizon")
    width, height = map(float, image_size)
    scale = np.asarray([max(width - 1.0, 1.0), max(height - 1.0, 1.0)])
    error = (points[:, -final_steps:] - target[None, None]) / scale[None, None]
    return np.mean(np.sum(np.square(error), axis=-1), axis=-1).astype(np.float32)


def draw_goal_projection(
    frame: np.ndarray,
    *,
    target_px: np.ndarray,
    current_gripper_px: np.ndarray,
    predicted_gripper_px: np.ndarray,
) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    target = tuple(map(float, target_px))
    current = tuple(map(float, current_gripper_px))
    predicted = tuple(map(float, predicted_gripper_px))
    draw.line((*current, *target), fill=(240, 191, 46), width=3)
    draw.ellipse(
        (target[0] - 6, target[1] - 6, target[0] + 6, target[1] + 6),
        outline=(240, 191, 46),
        width=3,
    )
    draw.ellipse(
        (
            predicted[0] - 5,
            predicted[1] - 5,
            predicted[0] + 5,
            predicted[1] + 5,
        ),
        outline=(43, 150, 92),
        width=3,
    )
    draw.line((*current, *predicted), fill=(43, 150, 92), width=3)
    return image


@torch.inference_mode()
def encode_fcpg_view(
    system: SO101FCPGConditionedLeWM,
    frames: np.ndarray,
    normalized_states: np.ndarray,
    anchors_xyz: np.ndarray,
    q_nominal: np.ndarray,
    valid: np.ndarray,
    camera_context: np.ndarray,
    device: torch.device,
    *,
    correction_seed_px: np.ndarray | None = None,
    derive_center_from_tips: bool = False,
    precomputed_vision: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    return_native: bool = False,
    return_bridge: bool = False,
) -> tuple[Any, ...]:
    system.last_fcpg_native = None
    if precomputed_vision is None:
        pixels = normalize_rgb(np.asarray(frames)).to(device).unsqueeze(0)
        context, patches, feature_map = system._encode_shared_vision(pixels)
    else:
        context, patches, feature_map = precomputed_vision
    frames_count = len(frames)
    if context.shape[:2] != (1, frames_count):
        raise ValueError("precomputed context does not match the frame history")
    if patches.shape[:2] != (1, frames_count):
        raise ValueError("precomputed patches do not match the frame history")
    if feature_map.shape[0] != frames_count:
        raise ValueError("precomputed feature map does not match the frame history")
    observations = np.concatenate(
        (normalized_states, camera_context), axis=-1
    ).astype(np.float32)
    seed = q_nominal if correction_seed_px is None else np.asarray(
        correction_seed_px, dtype=np.float32
    )
    if seed.shape != q_nominal.shape or not np.all(np.isfinite(seed)):
        raise ValueError("correction_seed_px must be finite and match q_nominal")
    try:
        fcpg = system.corrector(
            feature_map,
            torch.from_numpy(seed).to(device),
            torch.from_numpy(anchors_xyz).to(device),
            torch.from_numpy(observations).to(device),
            image_size=(int(frames.shape[2]), int(frames.shape[1])),
        )
        q_corrected = fcpg.q_corrected_px.reshape(frames_count, 3, 2)
        bridge_valid = torch.from_numpy(valid).to(device).unsqueeze(0)
        if derive_center_from_tips:
            # SO-101 anchor order is [center, fixed tip, moving tip]. Keep the
            # learned fingertip measurements and derive the redundant center
            # before the corrected anchors are encoded for LeWM.
            q_corrected = q_corrected.clone()
            q_corrected[:, 0] = q_corrected[:, 1:].mean(dim=1)
            bridge_valid = bridge_valid.clone()
            bridge_valid[:, :, 0] = bridge_valid[:, :, 1] & bridge_valid[:, :, 2]
        bridge = system.bridge(
            context,
            patches,
            torch.from_numpy(q_nominal).to(device).unsqueeze(0),
            q_corrected.unsqueeze(0),
            torch.from_numpy(anchors_xyz).to(device).unsqueeze(0),
            bridge_valid,
            image_size=(int(frames.shape[2]), int(frames.shape[1])),
        )
        system.last_fcpg_native = fcpg
        encoded = (
            context[0].cpu().numpy().astype(np.float32),
            bridge.context_embedding[0].cpu().numpy().astype(np.float32),
            q_corrected.cpu().numpy().astype(np.float32),
        )
        if return_native and return_bridge:
            return (*encoded, fcpg, bridge)
        if return_native:
            return (*encoded, fcpg)
        if return_bridge:
            return (*encoded, bridge)
        return encoded
    except Exception:
        system.last_fcpg_native = None
        raise


def draw_grounding(
    frame: np.ndarray,
    nominal: np.ndarray,
    corrected: np.ndarray,
    valid: np.ndarray,
) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    for index, (before, after, supported) in enumerate(
        zip(nominal, corrected, valid, strict=True)
    ):
        if not supported:
            continue
        color = COLORS[index]
        bx, by = map(float, before)
        ax, ay = map(float, after)
        draw.line((bx, by, ax, ay), fill=color, width=2)
        draw.line((bx - 4, by - 4, bx + 4, by + 4), fill=(255, 255, 255), width=2)
        draw.line((bx + 4, by - 4, bx - 4, by + 4), fill=(255, 255, 255), width=2)
        draw.ellipse((ax - 5, ay - 5, ax + 5, ay + 5), outline=color, width=3)
    return image


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def html_page(report: dict[str, Any]) -> str:
    selected = report["selection"]
    execution = "QUALIFIED" if selected["execution_qualified"] else "BLOCKED"
    card_rows = [
        ("Wrist: white X nominal / colored circle FCPG", "wrist_grounding.jpg"),
        ("Column raw camera frame", "column_raw.jpg"),
        ("Column: white X nominal / colored circle FCPG", "column_grounding.jpg"),
    ]
    if report.get("current_goal") is not None:
        card_rows.append(
            (
                "Current target: yellow; selected LeWM H8 gripper: green",
                "current_goal_projection.jpg",
            )
        )
    card_rows.extend(
        [
            ("Selected demonstration start", "selected_start.jpg"),
            ("Selected demonstration H8 end", "selected_end.jpg"),
            ("LeWM predicted terminal nearest frame", "predicted_neighbor.jpg"),
            ("Demonstrated successful goal (diagnostic only)", "goal.jpg"),
        ]
    )
    cards = "".join(
        f'<article><h2>{html.escape(title)}</h2><img src="{html.escape(path)}"></article>'
        for title, path in card_rows
    )
    candidates = "".join(
        "<article><h2>Candidate {index}: ep {episode} / frame {frame} - {status}</h2>"
        '<div class="pair"><img src="candidates/{index:02d}_wrist.jpg">'
        '<img src="candidates/{index:02d}_column.jpg"></div>{goal_image}'
        "<p>cost={cost:.4f}; goal={goal_cost}; geometry={geometry_cost}; "
        "phase={phase:.4f}; first jump={jump:.3f}; "
        "prefix step={prefix_jump:.3f}; rejected={reasons}</p></article>".format(
            index=row["index"],
            episode=row["episode"],
            frame=row["frame"],
            status="SAFE" if row["safe"] else "BLOCKED",
            cost=row["cost"],
            goal_cost=(
                f"{row['current_goal_cost']:.4f}"
                if row["current_goal_cost"] is not None
                else "n/a"
            ),
            geometry_cost=(
                f"{row['geometry_goal_cost']:.4f}"
                if row["geometry_goal_cost"] is not None
                else "n/a"
            ),
            goal_image=(
                f'<img src="candidates/{row["index"]:02d}_goal.jpg">'
                if row["current_goal_cost"] is not None
                else ""
            ),
            phase=row["object_phase_distance"],
            jump=row["first_command_jump"],
            prefix_jump=row["prefix_max_step_jump"],
            reasons=html.escape(", ".join(row["rejection_reasons"]) or "none"),
        )
        for row in report["candidates"]
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SO-101 FCPG + LeWM Shadow</title>
<style>body{{margin:0;background:#eef1f2;color:#1d2529;font:15px system-ui}}header{{padding:22px 5%;background:#202b30;color:white}}main{{max-width:1300px;margin:auto;padding:20px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}article{{background:white;border:1px solid #c8d0d4;padding:12px}}img{{width:100%;image-rendering:auto}}h1{{margin:0 0 8px}}h2{{font-size:15px}}.full{{grid-column:1/-1}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}code{{font-family:ui-monospace,monospace}}@media(max-width:760px){{main{{grid-template-columns:1fr}}}}</style></head><body>
<header><h1>SO-101 Real Shadow: FCPG + 5M Dynamics</h1><div>只读推理；robot_commands_sent=false；当前画面没有 q_true，彩色圆只供人工核对，不能当精度证据。</div></header>
<main><article class="full"><b>执行判定：</b>{execution}（shadow candidate={selected['shadow_candidate_qualified']}）；<b>目标模式：</b>{report['goal_contract']['mode']}；<b>候选：</b>{selected['candidate_count']}；<b>安全候选：</b>{selected['safe_candidate_count']}；<b>诊断选中：</b>episode {selected['episode']} / frame {selected['frame']}；<b>首命令最大关节跳变：</b>{selected['first_command_max_jump']:.3f}<br><b>总 cost：</b>{selected['cost']:.6f}；<b>FCPG 平均修正：</b>wrist {report['grounding']['wrist']['mean_correction_px']:.2f}px / column {report['grounding']['column']['mean_correction_px']:.2f}px</article>{cards}<article class="full"><h2>All candidates</h2><p>鱼和盒子的绝对坐标不被假设固定。黄色是当前帧目标，绿色是 LeWM 对候选动作 H8 后夹爪位置的预测。所有结果只读，不发送命令。</p></article>{candidates}</main></body></html>"""


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    sessions = load_paired_session(args.session)
    frame_count = sessions["wrist"].frame_count
    final_index = frame_count - 1
    history_indices = final_index - np.arange(2, -1, -1) * args.history_stride
    if history_indices[0] < 0:
        raise ValueError("shadow session is too short for three history frames")
    states = np.asarray(sessions["wrist"].states[history_indices], dtype=np.float32)

    device = torch.device(args.device)
    bridge_payload = torch.load(
        args.bridge_checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_path = Path(bridge_payload["args"]["checkpoint"]).resolve()
    checkpoint, embedded_args, cache_paths, frameskip, history, checkpoint_sha = (
        checkpoint_contract(checkpoint_path, None)
    )
    if frameskip != args.history_stride or history != 3:
        raise ValueError("shadow sampling must match checkpoint frameskip/history")
    model = build_checkpoint_model(checkpoint, frameskip, history, device)
    if not isinstance(model, StateResidualLeWM):
        raise TypeError("shadow ranking requires a state-residual LeWM")
    corrector_variant = infer_so101_fcpg_corrector_variant(bridge_payload)
    assert_coarse_fine_planner_policy(
        variant=corrector_variant,
        metadata=bridge_payload,
        recursive_grounding=False,
        global_tip_checkpoint=None,
        request_live_execute=False,
        allow_uncalibrated_coarse_fine_shadow=bool(
            args.allow_uncalibrated_coarse_fine_shadow
        ),
    )
    corrector = build_so101_fcpg_corrector(
        model,
        camera_context_dim=5,
        variant=corrector_variant,
    )
    system = SO101FCPGConditionedLeWM(
        model,
        corrector=corrector,
        camera_context_dim=5,
        freeze_model=True,
    ).to(device)
    system.corrector.load_state_dict(bridge_payload["corrector"], strict=True)
    system.bridge.load_state_dict(bridge_payload["bridge"], strict=True)
    system.eval()

    caches = load_caches(cache_paths)
    if len(caches) != 1:
        raise ValueError("shadow checkpoint must bind exactly one primary cache")
    primary: CacheView = caches[0]
    secondary: CacheView = load_caches([args.secondary_cache.resolve()])[0]
    validate_paired_camera_data(primary.data, secondary.data)

    contract = checkpoint["run_config"]["contract"]
    state_center = np.asarray(contract["normalization_center"], dtype=np.float32)
    state_scale = np.asarray(contract["normalization_scale"], dtype=np.float32)
    normalized_states = (states - state_center) / state_scale
    kinematics = SO101MuJoCoKinematics(
        Path(__file__).resolve().parents[1]
        / "third_party/so_arm100/Simulation/SO101/so101_new_calib.xml"
    )
    fits = load_projection_fits(args.calibration)
    anchors_xyz, nominal, valid = project_history(
        states, fits, kinematics, image_size=(224, 224)
    )
    view_frames = {
        view: np.asarray(sessions[view].frames[history_indices]).copy()
        for view in VIEWS
    }
    Image.fromarray(view_frames["column"][-1]).save(
        output / "column_raw.jpg", quality=92
    )
    registration: dict[str, Any] | None = None
    if args.coarse_register_column:
        distance = np.mean(
            np.square(
                (
                    np.asarray(primary.data.states, dtype=np.float32)
                    - states[-1][None]
                )
                / state_scale[None]
            ),
            axis=1,
        )
        reference_row = int(np.argmin(distance))
        matrix, registration_metrics = estimate_coarse_affine(
            np.asarray(secondary.data.frames[reference_row]),
            view_frames["column"][-1],
        )
        view_frames["column"] = align_current_to_reference(
            view_frames["column"], matrix
        )
        registration = {
            "reference_row": reference_row,
            "reference_episode": int(secondary.data.episodes[reference_row]),
            "reference_frame": int(secondary.data.frame_indices[reference_row]),
            "normalized_state_distance": float(np.sqrt(distance[reference_row])),
            "matrix": matrix,
            "model_input": "current column frames warped into training-camera coordinates",
            **registration_metrics,
        }
    contexts: dict[str, np.ndarray] = {}
    base_contexts: dict[str, np.ndarray] = {}
    corrected: dict[str, np.ndarray] = {}
    fusion_state = GroundingFusionState()
    grounding_contract = planner_grounding_ranking_contract(
        grounding_mode=getattr(args, "grounding_mode", GROUNDING_GATED_FCPG),
        ranking_mode=getattr(args, "ranking_mode", RANKING_ON),
    )
    for view_index, view in enumerate(VIEWS):
        frames = view_frames[view]
        camera_context = np.zeros((history, 5), dtype=np.float32)
        camera_context[:, view_index] = 1.0
        if corrector_variant == COARSE_FINE_VARIANT:
            base, bridged, points, native = encode_fcpg_view(
                system,
                frames,
                normalized_states,
                anchors_xyz,
                nominal[view],
                valid[view],
                camera_context,
                device,
                return_native=True,
            )
            system.last_fcpg_native = None
            points, _fusion = apply_coarse_fine_observe_points(
                native,
                seed_px=nominal[view],
                nominal_px=nominal[view],
                view=view,
                view_valid=valid[view],
                contract=grounding_contract,
                fusion_state=fusion_state,
            )
        else:
            base, bridged, points = encode_fcpg_view(
                system,
                frames,
                normalized_states,
                anchors_xyz,
                nominal[view],
                valid[view],
                camera_context,
                device,
            )
            if grounding_contract.grounding_mode == GROUNDING_NOMINAL:
                points = np.asarray(nominal[view], dtype=np.float32).copy()
        base_contexts[view] = base
        contexts[view] = bridged
        corrected[view] = points
        draw_grounding(
            frames[-1], nominal[view][-1], points[-1], valid[view][-1]
        ).save(output / f"{view}_grounding.jpg", quality=92)

    current_goal: dict[str, Any] | None = None
    if args.goal_mode == "current-fish":
        descriptor = red_object_descriptor(view_frames["column"][-1])
        if not descriptor["visible"]:
            raise ValueError("current column frame does not contain a detectable red fish")
        current_goal = {
            "mode": args.goal_mode,
            "view": "column",
            "target_kind": "red_fish_centroid",
            "descriptor": descriptor,
            "target_px": np.asarray(descriptor["centroid_px"], dtype=np.float32),
        }
    elif args.goal_mode == "current-box":
        descriptor = brown_box_descriptor(view_frames["column"][-1])
        if not descriptor["visible"]:
            raise ValueError("current column frame does not contain a detectable box")
        current_goal = {
            "mode": args.goal_mode,
            "view": "column",
            "target_kind": "cardboard_box_centroid",
            "descriptor": descriptor,
            "target_px": np.asarray(descriptor["centroid_px"], dtype=np.float32),
        }

    assignments = episode_splits(
        [primary.data],
        parse_exclusions(embedded_args.get("exclude_episode", [])),
        split_seed=int(embedded_args["split_seed"]),
        val_fraction=float(embedded_args["val_fraction"]),
    )
    train_episodes = _training_episodes(primary, assignments)
    horizon_raw = frameskip * args.horizon
    bank_rows = candidate_start_rows(
        primary.data.episodes,
        primary.data.frame_indices,
        train_episodes,
        horizon_raw=horizon_raw,
        stride=frameskip,
    )
    goal_rows = terminal_goal_rows(
        primary.data.episodes,
        train_episodes,
        tail_frames=args.goal_tail_frames,
    )
    train_rows = np.concatenate(
        [np.flatnonzero(np.asarray(primary.data.episodes) == episode) for episode in sorted(train_episodes)]
    )
    unique_rows = np.unique(np.concatenate((bank_rows, goal_rows, train_rows)))
    primary_encoded = encode_rows(model, primary, unique_rows, args.batch_size)
    secondary_encoded = encode_rows(model, secondary, unique_rows, args.batch_size)
    primary_by_row = dict(zip(map(int, unique_rows), primary_encoded, strict=True))
    secondary_by_row = dict(zip(map(int, unique_rows), secondary_encoded, strict=True))
    primary_bank = np.stack([primary_by_row[int(row)] for row in bank_rows])
    secondary_bank = np.stack([secondary_by_row[int(row)] for row in bank_rows])
    primary_goals = np.stack([primary_by_row[int(row)] for row in goal_rows])
    secondary_goals = np.stack([secondary_by_row[int(row)] for row in goal_rows])
    primary_train = np.stack([primary_by_row[int(row)] for row in train_rows])
    secondary_train = np.stack([secondary_by_row[int(row)] for row in train_rows])

    current_object: dict[str, Any] | None = None
    bank_object_distance: np.ndarray | None = None
    bank_gripper_centers: np.ndarray | None = None
    if args.red_object_phase_guard:
        # The wrist camera is rigidly mounted to the gripper.  Compare the
        # object relative to the current gripper anchor, not table/world pixels.
        current_object = red_object_descriptor(
            view_frames["wrist"][-1],
            gripper_center_px=corrected["wrist"][-1, 0],
        )
        _, bank_nominal, _ = project_history(
            np.asarray(primary.data.states[bank_rows], dtype=np.float32),
            fits,
            kinematics,
            image_size=(224, 224),
        )
        bank_gripper_centers = np.asarray(bank_nominal["wrist"][:, 0], dtype=np.float32)
        bank_object_distance = red_object_phase_distances(
            current_object,
            np.asarray(primary.data.frames[bank_rows]),
            gripper_centers_px=bank_gripper_centers,
        )

    current_state = states[-1]
    retrieval = retrieve_diverse_candidates(
        np.concatenate((contexts["wrist"][-1], contexts["column"][-1])),
        current_state,
        np.concatenate((primary_bank, secondary_bank), axis=1),
        np.asarray(primary.data.states[bank_rows], dtype=np.float32),
        bank_rows,
        np.asarray(primary.data.episodes[bank_rows], dtype=np.int64),
        state_scale=state_scale,
        per_episode=args.candidates_per_episode,
        minimum_row_gap=args.candidate_min_gap,
        state_weight=args.state_retrieval_weight,
        extra_distance=bank_object_distance,
        extra_weight=args.object_phase_weight,
    )
    object_distance_by_row = (
        dict(zip(map(int, bank_rows), bank_object_distance, strict=True))
        if bank_object_distance is not None
        else {}
    )
    selected_object_distance = np.asarray(
        [object_distance_by_row.get(int(row), 0.0) for row in retrieval.rows],
        dtype=np.float32,
    )
    raw_prefixes = raw_action_prefixes(
        primary.data.actions, retrieval.rows, horizon_raw=horizon_raw
    )
    raw_prefixes = rebase_action_prefixes(
        raw_prefixes,
        np.asarray(primary.data.states[retrieval.rows], dtype=np.float32),
        current_state,
    )
    action_center, action_scale, action_normalization = checkpoint_action_normalizer(
        checkpoint, embedded_args
    )
    chunks = chunk_and_normalize_actions(
        raw_prefixes,
        frameskip=frameskip,
        center=action_center,
        scale=action_scale,
    )
    hold_chunk = np.tile((current_state - action_center) / action_scale, frameskip)
    past_actions = np.stack((hold_chunk, hold_chunk)).astype(np.float32)
    primary_prediction, state_prediction = _rollout_candidates(
        model, contexts["wrist"], normalized_states, past_actions, chunks
    )
    secondary_prediction, _ = _rollout_candidates(
        model, contexts["column"], normalized_states, past_actions, chunks
    )
    predicted_states_raw = (
        state_prediction * state_scale[None, None] + state_center[None, None]
    )
    predicted_goal_pixels: np.ndarray | None = None
    commanded_goal_pixels: np.ndarray | None = None
    geometry_goal_cost: np.ndarray | None = None
    if current_goal is None:
        visual_cost = terminal_goal_cost(
            primary_prediction, primary_goals, final_steps=args.goal_final_steps
        ) + args.secondary_visual_goal_weight * terminal_goal_cost(
            secondary_prediction, secondary_goals, final_steps=args.goal_final_steps
        )
        goal_states = normalized_state_rows(
            primary, goal_rows, state_center, state_scale
        )
        state_cost = terminal_state_cost(
            state_prediction, goal_states, final_steps=args.goal_final_steps
        )
    else:
        view = str(current_goal["view"])
        correction = corrected[view][-1] - nominal[view][-1]
        predicted_all_pixels = project_future_anchor_pixels(
            predicted_states_raw,
            fits[view],
            kinematics,
            current_correction_px=correction,
        )
        predicted_goal_pixels = predicted_all_pixels[:, :, 0]
        visual_cost = current_pixel_goal_cost(
            predicted_goal_pixels,
            np.asarray(current_goal["target_px"], dtype=np.float32),
            final_steps=args.goal_final_steps,
            image_size=(224, 224),
        )
        commanded_states = raw_prefixes[:, frameskip - 1 :: frameskip]
        if commanded_states.shape[1] != args.horizon:
            raise ValueError("raw candidate prefix does not align with LeWM horizon")
        commanded_all_pixels = project_future_anchor_pixels(
            commanded_states,
            fits[view],
            kinematics,
            current_correction_px=correction,
        )
        commanded_goal_pixels = commanded_all_pixels[:, :, 0]
        geometry_goal_cost = current_pixel_goal_cost(
            commanded_goal_pixels,
            np.asarray(current_goal["target_px"], dtype=np.float32),
            final_steps=args.goal_final_steps,
            image_size=(224, 224),
        )
        state_cost = np.zeros_like(visual_cost)
    retrieval_cost = np.asarray(retrieval.combined_distance, dtype=np.float32)
    costs = (
        visual_cost
        + args.state_goal_weight * state_cost
        + args.retrieval_cost_weight * retrieval_cost
    )
    safety = candidate_safety_checks(
        raw_prefixes[:, 0],
        current_state,
        max_joint_jump=args.max_first_command_jump,
        max_prefix_step_jump=args.max_prefix_step_jump,
        raw_prefixes=raw_prefixes,
        predicted_states=predicted_states_raw,
        position_lower=SO101_POSITION_LOWER,
        position_upper=SO101_POSITION_UPPER,
        prediction_bound_margin=args.prediction_bound_margin,
    )
    phase_safe = (
        selected_object_distance <= args.max_object_phase_distance
        if args.red_object_phase_guard
        else np.ones(len(retrieval.rows), dtype=bool)
    )
    safe = safety.safe & phase_safe
    if np.any(safe):
        selected = int(np.argmin(np.where(safe, costs, np.inf)))
    else:
        selected = int(np.argmin(safety.first_command_jump))
    selected_row = int(retrieval.rows[selected])
    predicted_state = predicted_states_raw[selected, -1]
    primary_neighbor = _display_neighbor(
        primary_prediction[selected, -1],
        predicted_state,
        primary_train,
        np.asarray(primary.data.states[train_rows], dtype=np.float32),
        state_scale,
    )
    display_row = int(train_rows[primary_neighbor])
    goal_row = int(goal_rows[0])
    Image.fromarray(np.asarray(primary.data.frames[selected_row])).save(
        output / "selected_start.jpg", quality=90
    )
    Image.fromarray(np.asarray(primary.data.frames[selected_row + horizon_raw])).save(
        output / "selected_end.jpg", quality=90
    )
    Image.fromarray(np.asarray(primary.data.frames[display_row])).save(
        output / "predicted_neighbor.jpg", quality=90
    )
    Image.fromarray(np.asarray(primary.data.frames[goal_row])).save(
        output / "goal.jpg", quality=90
    )
    if current_goal is not None:
        view = str(current_goal["view"])
        draw_goal_projection(
            view_frames[view][-1],
            target_px=np.asarray(current_goal["target_px"], dtype=np.float32),
            current_gripper_px=corrected[view][-1, 0],
            predicted_gripper_px=predicted_goal_pixels[selected, -1],
        ).save(output / "current_goal_projection.jpg", quality=92)

    candidate_dir = output / "candidates"
    candidate_dir.mkdir()
    candidate_reports: list[dict[str, Any]] = []
    for index, row_value in enumerate(retrieval.rows):
        row = int(row_value)
        Image.fromarray(np.asarray(primary.data.frames[row])).save(
            candidate_dir / f"{index:02d}_wrist.jpg", quality=88
        )
        Image.fromarray(np.asarray(secondary.data.frames[row])).save(
            candidate_dir / f"{index:02d}_column.jpg", quality=88
        )
        if current_goal is not None:
            view = str(current_goal["view"])
            draw_goal_projection(
                view_frames[view][-1],
                target_px=np.asarray(current_goal["target_px"], dtype=np.float32),
                current_gripper_px=corrected[view][-1, 0],
                predicted_gripper_px=predicted_goal_pixels[index, -1],
            ).save(candidate_dir / f"{index:02d}_goal.jpg", quality=88)
        reasons = []
        for passed, name in (
            (safety.finite[index], "non_finite"),
            (safety.first_jump_ok[index], "first_jump"),
            (safety.prefix_step_ok[index], "prefix_step"),
            (safety.command_bounds_ok[index], "command_bounds"),
            (safety.prediction_bounds_ok[index], "prediction_bounds"),
            (phase_safe[index], "object_phase"),
        ):
            if not passed:
                reasons.append(name)
        candidate_reports.append(
            {
                "index": index,
                "row": row,
                "episode": int(primary.data.episodes[row]),
                "frame": int(primary.data.frame_indices[row]),
                "cost": float(costs[index]),
                "visual_cost": float(visual_cost[index]),
                "current_goal_cost": (
                    float(visual_cost[index]) if current_goal is not None else None
                ),
                "geometry_goal_cost": (
                    float(geometry_goal_cost[index])
                    if geometry_goal_cost is not None
                    else None
                ),
                "state_cost": float(state_cost[index]),
                "retrieval_cost": float(retrieval_cost[index]),
                "object_phase_distance": float(selected_object_distance[index]),
                "safe": bool(safe[index]),
                "hard_command_safe": bool(safety.hard_command_safe[index]),
                "prediction_plausible": bool(safety.prediction_plausible[index]),
                "rejection_reasons": reasons,
                "first_command_jump": float(safety.first_command_jump[index]),
                "prefix_max_step_jump": float(safety.prefix_max_step_jump[index]),
                "first_command": raw_prefixes[index, 0],
                "command_min": np.min(raw_prefixes[index], axis=0),
                "command_max": np.max(raw_prefixes[index], axis=0),
                "prediction_min": np.min(predicted_states_raw[index], axis=0),
                "prediction_max": np.max(predicted_states_raw[index], axis=0),
            }
        )

    report = {
        "format": "so101-real-shadow-fcpg-lewm-v2",
        "scope": "read-only real dual-camera shadow inference",
        "robot_commands_sent": False,
        "session": str(args.session.resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "bridge_checkpoint": str(args.bridge_checkpoint.resolve()),
        "history_indices": history_indices,
        "action_normalization": action_normalization,
        "anchor_names": SO101_ANCHOR_NAMES,
        "grounding": {
            view: {
                "nominal_px": nominal[view][-1],
                "corrected_px": corrected[view][-1],
                "mean_correction_px": float(
                    np.mean(np.linalg.norm(corrected[view][-1] - nominal[view][-1], axis=-1))
                ),
                "valid": valid[view][-1],
            }
            for view in VIEWS
        },
        "current_goal": current_goal,
        "goal_contract": {
            "mode": args.goal_mode,
            "uses_current_scene_target": current_goal is not None,
            "uses_demonstration_terminal_latent": current_goal is None,
            "absolute_fish_or_box_coordinates_assumed_fixed": False,
            "state_or_fk_coordinates_used_only_for_robot_anchor_prediction": True,
        },
        "column_registration": registration,
        "object_phase_guard": {
            "enabled": bool(args.red_object_phase_guard),
            "kind": "largest red HSV component, compared as fish-minus-gripper in wrist view",
            "invariance": "only the wrist-camera-to-gripper mounting is fixed; fish/table/column/world coordinates are not assumed fixed",
            "current": current_object,
            "current_gripper_center_px": corrected["wrist"][-1, 0],
            "weight": float(args.object_phase_weight),
            "maximum_distance": float(args.max_object_phase_distance),
            "paper_claim_allowed": False,
        },
        "safety_envelope": {
            "position_lower": SO101_POSITION_LOWER,
            "position_upper": SO101_POSITION_UPPER,
            "maximum_first_command_jump": float(args.max_first_command_jump),
            "maximum_prefix_step_jump": float(args.max_prefix_step_jump),
            "prediction_bound_margin": float(args.prediction_bound_margin),
            "checks_entire_raw_prefix": True,
            "checks_entire_predicted_rollout": True,
            "command_bounds_are_never_relaxed": True,
            "prediction_margin_is_not_a_command_safety_margin": True,
            "clips_unsafe_commands": False,
        },
        "candidates": candidate_reports,
        "selection": {
            "candidate_count": len(retrieval.rows),
            "safe_candidate_count": int(np.sum(safe)),
            "selected_index": selected,
            "episode": int(primary.data.episodes[selected_row]),
            "frame": int(primary.data.frame_indices[selected_row]),
            "row": selected_row,
            "cost": float(costs[selected]),
            "visual_cost": float(visual_cost[selected]),
            "current_goal_cost": (
                float(visual_cost[selected]) if current_goal is not None else None
            ),
            "geometry_goal_cost": (
                float(geometry_goal_cost[selected])
                if geometry_goal_cost is not None
                else None
            ),
            "state_cost": float(state_cost[selected]),
            "retrieval_cost": float(retrieval_cost[selected]),
            "retrieval_cost_weight": float(args.retrieval_cost_weight),
            "first_command": raw_prefixes[selected, 0],
            "first_command_max_jump": float(safety.first_command_jump[selected]),
            "safe_candidate_available": bool(np.any(safe)),
            "shadow_candidate_qualified": bool(
                np.any(safe) and current_goal is not None
            ),
            # This program is intentionally read-only. A fresh recapture and a
            # separate guarded executor are required before any command.
            "execution_qualified": False,
            "deployment_block_reason": (
                "read_only_shadow_evaluator"
                if current_goal is not None
                else "demo_terminal_goal_is_not_current_scene_relative"
            ),
            "diagnostic_fallback_used": bool(not np.any(safe)),
            "object_phase_distance": float(selected_object_distance[selected]),
            "predicted_terminal_state": predicted_state,
            "display_row": display_row,
        },
    }
    (output / "report.json").write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(html_page(report), encoding="utf-8")
    print(json.dumps(jsonable(report["selection"]), indent=2), flush=True)
    print(f"html: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
