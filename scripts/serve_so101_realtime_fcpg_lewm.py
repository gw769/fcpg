#!/usr/bin/env python3
"""Serve a confirmed one-chunk SO-101 FCPG + LeWM MPC loop."""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import faulthandler
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
import mimetypes
import os
from pathlib import Path
import signal
import threading
import time
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image, ImageDraw

from fcpg.so101_active_gripper import (
    localize_gripper_from_pulse,
    motion_gripper_roi_from_localization,
    motion_gripper_rois_from_pulse,
    next_gripper_position,
)
from fcpg.self_contact_nominal_predictor import attach_nominal_prediction
from fcpg.so101_self_contact import (
    SelfContactConfig,
    SelfContactFatalError,
    SelfContactHardwareHooks,
    infer_close_direction_from_pulse,
    persistable_self_contact_arrays,
    run_self_contact_seek,
)
from fcpg.so101_kinematics import SO101_JOINT_NAMES, SO101_MUJOCO_JOINT_NAMES
from fcpg.so101_planning_snapshots import (
    VIEWS as SNAPSHOT_VIEWS,
    build_snapshot_meta,
    load_latest_snapshot_jpeg,
    load_latest_snapshot_meta,
    load_round_snapshot_jpeg,
    load_round_snapshot_meta,
    plan_anchor_paths,
    planning_frame,
    prod_v13_observed_tip_trajectory,
    render_planning_snapshot,
    snapshot_jpeg_bytes,
    write_planning_snapshots,
)
from fcpg.coarse_fine_planner import peek_bridge_corrector_variant
from fcpg.grounding_confidence_fusion import COARSE_FINE_VARIANT
from fcpg.so101_realtime_planner import (
    PreparedPlannerConfig,
    PreparedSO101RealtimePlanner,
    RealtimePlan,
    plan_json,
)
from fcpg.so101_realtime_runtime import (
    ENDPOINT_MISS_WITH_ACTUAL_STATE,
    ENDPOINT_SETTLED,
    ChunkExecution,
    EndpointEnvelope,
    EndpointSettlement,
    LiveObservation,
    PlanRejectedError,
    RealtimeMPCController,
    ReplaySO101IO,
    SafeHoldVerificationError,
    gripper_capture_radius_m,
)
from fcpg.letterbox_geometry import (
    LETTERBOX_V1,
    PLANNING_IMAGE_SIZE,
    SQUASH_V0,
    letterbox_rgb_image,
)
from fcpg.so101_pilot_preflight import FROZEN_B2_CHECKPOINT
from fcpg.so101_retrieval_mpc import SO101_POSITION_LOWER, SO101_POSITION_UPPER
from fcpg.so101_safe_energize import preseed_goal_position
from fcpg.so101_shadow_session import observation_state
from scripts.evaluate_so101_real_shadow_fcpg_lewm import (
    VIEWS,
    draw_grounding,
    load_paired_session,
)
from scripts.create_so101_peg_context_hardware_manifest import (
    iter_trials as iter_hardware_experiment_trials,
    validate_manifest as validate_hardware_experiment_manifest,
)
from scripts.preflight_so101_phase_target import (
    build_hardware_preflight_report,
    require_hardware_preflight,
)
from scripts.render_so101_mujoco_prediction import (
    DEFAULT_SCENE,
    SO101ReplayRenderer,
    fit_panel_image,
)
from scripts.visualize_so101_lewm import load_font


REPO = Path(__file__).resolve().parents[1]
DEFAULT_BRIDGE = REPO / (
    "outputs/lerobot/so101/"
    "my_first_task_clean_v1_fcpg_context_bridge_val2_seed6126_2k/latest.pt"
)
#: The frozen, shadow-only B2 head -- the checkpoint the pilot compares.
#:
#: This default used to name the V9 multiscale seed5131 checkpoint, whose
#: protocol IS in ``MEMORY_PROTOCOLS`` and which therefore builds a memory model
#: with offsets [5,15,30,60] and runs the causal-memory rollout.  That made a
#: forgotten ``--model-checkpoint`` flag silently swap the model under test for a
#: different one with a different conditioning path -- the failure mode this
#: whole patch exists to remove, since neither checkpoint announces itself in
#: the logs the operator watches.
#:
#: Imported rather than written as a path literal: ``so101_pilot_preflight`` is
#: the single owner of where the frozen head lives, so the preflight that
#: authorizes a session and the launcher that starts one cannot drift apart and
#: name different checkpoints.  Safe against circularity -- that module's only
#: reference back here is a deferred, function-local import.
DEFAULT_MODEL = FROZEN_B2_CHECKPOINT
#: Historical, UNVERSIONED rig calibration.  It declares no
#: ``geometry_protocol`` at all, so the strict loader refuses it whenever a
#: caller declares `letterbox_v1`.  Kept as the default for the replay /
#: offline route only, which declares `squash_v0` and therefore loads it
#: permissively, exactly as it always has.  Never modified.
DEFAULT_CALIBRATION = REPO / (
    "outputs/lerobot/so101/"
    "my_first_task_clean_v1_dual_camera_calibration/calibration.json"
)
#: Geometry-versioned calibration for the B1 LIVE path: same directory, same
#: frozen coefficients, plus the explicit ``geometry_protocol=letterbox_v1``
#: declaration (a declaration-only upgrade -- no coefficient was transformed
#: or refitted).  The live path declares `letterbox_v1`
#: (`_declared_pixel_geometry`), and the planner then loads the projection fit
#: under a strict geometry contract, so pointing the live default at the
#: unversioned artifact above makes startup fail closed.  That refusal is
#: correct; the launcher default is what has to name the versioned artifact.
#: Single owner for the live default: no path literal is repeated at the CLI.
DEFAULT_LIVE_CALIBRATION = REPO / (
    "outputs/lerobot/so101/"
    "my_first_task_clean_v1_dual_camera_calibration/calibration_letterbox_v1.json"
)
DEFAULT_SECONDARY = REPO / (
    "data/lerobot/so101/cache/my_first_task_clean_v1_column_224"
)
DEFAULT_GLOBAL_TIP = REPO / (
    "outputs/lerobot/so101/"
    "my_first_task_clean_v1_global_tip_seed6131_2k/best.pt"
)
DEFAULT_REPLAY = REPO / (
    "outputs/lerobot/so101/real_dual_shadow_20260724_pre_traj02"
)
DEFAULT_OUTPUT = REPO / "outputs/lerobot/so101/realtime_fcpg_lewm_mpc"
DEFAULT_ROBOT_PORT = (
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E115543-if00"
)
DEFAULT_WRIST_DEVICE = (
    "/dev/v4l/by-id/usb-XHH-260202-H_Integrated_Camera-video-index0"
)
FALLBACK_COLUMN_DEVICE = (
    "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"
)
DEFAULT_COLUMN_DEVICE = FALLBACK_COLUMN_DEVICE
#: Raw capture resolution both live cameras are opened at, and therefore the
#: raw geometry the live `letterbox_v1` planning frames are derived from.  One
#: constant feeds the OpenCVCameraConfig request, the declared
#: `source_image_size` on the planner config, and the per-frame ingestion
#: guard, so the requested mode and the declared geometry cannot drift apart.
#: Geometry is never inferred from the 224x224 planning canvas: both protocols
#: produce that same shape, so it carries no information about which one ran.
LIVE_SOURCE_IMAGE_SIZE = (640, 480)
EXPERIMENT_SESSION_FORMAT = "so101-peg-context-hardware-session-v1"
COMMAND_COLOR = "#087e8b"
PREDICTION_COLOR = "#d1495b"
STATE_READ_RETRIES = 4
STATE_WRITE_RETRIES = 4
# Encoder noise on a held STS3215 is a couple of ticks; a larger jump is real motion.
DIAGNOSTIC_MATERIAL_MOTION_TICKS = 2
#: Allowance when verifying a Goal_Position read-back against what was written.
#: A calibrated goal is quantized to an integer STS3215 count on the way down
#: (360/4096 ~ 0.088 deg per count) and re-normalized on the way back, so one
#: count of slack in each direction is round-trip noise rather than a lost write.
#: Same physical basis as the 0.1 quantization allowance the pre-motion command
#: check already grants.
GOAL_READBACK_TOLERANCE = 0.2
#: A safe hold gets one retry: the arm may legitimately have crept between the
#: write and the read-back, and the retry re-reads the pose before rewriting it.
#: Beyond that a mismatch is a bus or servo fault, not a race.
SAFE_HOLD_VERIFY_ATTEMPTS = 2
# Previously inlined in HardwareSO101IO.calibrate_gripper. Lifted so an
# in-place open can reuse the same pulse, not a second set of numbers.
GRIPPER_PULSE_SAMPLE_HZ = 20.0
GRIPPER_PULSE_MAXIMUM_STEP = 1.5
GRIPPER_PULSE_SETTLE_TOLERANCE = 2.0
GRIPPER_PULSE_STABLE_REQUIRED = 6
# The real gripper readback advances much more slowly than the 1.5-unit command
# cap. A full 30-unit close took about 125 samples on hardware, so 100 samples
# could stop mid-pulse even though the servo was still making normal progress.
GRIPPER_PULSE_MAXIMUM_PHASE_FRAMES = 180
GRIPPER_PULSE_MAXIMUM_ARM_DRIFT = 1.5
MONITOR_STREAM_FPS = 10.0
CAMERA_READ_TIMEOUT_MS = 1000.0
CAMERA_RECOVERY_TIMEOUT_MS = 2000.0
CAMERA_RECOVERY_ATTEMPTS = 3
CAMERA_RECOVERY_DELAY_S = 0.25


def _select_auto_column_device(
    wrist_device: str | Path,
    *,
    by_id_root: Path = Path("/dev/v4l/by-id"),
) -> str:
    """Select one stable capture alias physically distinct from the wrist."""

    wrist = Path(wrist_device).expanduser()
    wrist_target = wrist.resolve(strict=False)
    candidates: list[Path] = []
    if by_id_root.is_dir():
        for alias in sorted(by_id_root.glob("*-video-index0")):
            if not alias.is_symlink() or not alias.exists():
                continue
            if alias.resolve() == wrist_target:
                continue
            candidates.append(alias)
    if not candidates:
        return FALLBACK_COLUMN_DEVICE
    if len(candidates) > 1:
        joined = ", ".join(str(path) for path in candidates)
        raise ValueError(
            "multiple stable column-camera candidates found; pass "
            f"--column-device explicitly: {joined}"
        )
    return str(candidates[0])


def _save_gripper_calibration_capture(
    root: Path,
    frames: dict[str, np.ndarray],
    states: np.ndarray,
    commands: np.ndarray,
    phase_ranges: list[dict[str, int | str]],
    *,
    sample_hz: float,
    closed_position: float,
    open_position: float,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Persist one complete pulse before attempting visual localization.

    ``extra_metadata`` is written into ``metadata.json`` only.  The historical
    ``pulse.npz`` key set is unchanged so older captures keep loading.
    """

    capture_dir = root / f"pulse_{time.time_ns()}"
    capture_dir.mkdir(parents=True, exist_ok=False)
    arrays = {view: np.asarray(frames[view], dtype=np.uint8) for view in VIEWS}
    state_array = np.asarray(states, dtype=np.float32)
    command_array = np.asarray(commands, dtype=np.float32)
    npz_path = capture_dir / "pulse.npz"
    np.savez_compressed(
        npz_path,
        wrist_frames=arrays["wrist"],
        column_frames=arrays["column"],
        states=state_array,
        commands=command_array,
        sample_hz=np.asarray(sample_hz, dtype=np.float32),
        closed_position=np.asarray(closed_position, dtype=np.float32),
        open_position=np.asarray(open_position, dtype=np.float32),
    )
    metadata = {
        "frame_count": int(len(state_array)),
        "sample_hz": float(sample_hz),
        "closed_position": float(closed_position),
        "open_position": float(open_position),
        "phase_ranges": phase_ranges,
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    metadata_path = capture_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    duration_ms = max(20, int(round(1000.0 / sample_hz)))
    gif_paths: dict[str, Path] = {}
    for view in VIEWS:
        images = [Image.fromarray(frame) for frame in arrays[view]]
        gif_path = capture_dir / f"{view}_pulse.gif"
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        gif_paths[view] = gif_path

    dual = [
        Image.fromarray(np.concatenate((wrist, column), axis=1))
        for wrist, column in zip(
            arrays["wrist"], arrays["column"], strict=True
        )
    ]
    dual_path = capture_dir / "dual_pulse.gif"
    dual[0].save(
        dual_path,
        save_all=True,
        append_images=dual[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    open_index = int(np.argmax(state_array[:, 5]))
    closed_index = int(np.argmin(state_array[:, 5]))
    for view in VIEWS:
        Image.fromarray(arrays[view][open_index]).save(capture_dir / f"{view}_open.png")
        Image.fromarray(arrays[view][closed_index]).save(
            capture_dir / f"{view}_closed.png"
        )
    index_path = capture_dir / "index.html"
    index_path.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>SO-101 gripper pulse</title>\n"
        "<style>body{font-family:sans-serif;margin:20px;background:#f5f6f7;color:#17202a}"
        "img{image-rendering:auto;max-width:100%;border:1px solid #ccd1d1}"
        ".grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}</style></head><body>"
        "<h1>SO-101 gripper calibration capture</h1><img src=\"dual_pulse.gif\">"
        "<div class=\"grid\"><section><h2>Wrist</h2><img src=\"wrist_pulse.gif\"></section>"
        "<section><h2>Column</h2><img src=\"column_pulse.gif\"></section></div>"
        "<p><a href=\"pulse.npz\">pulse.npz</a> | "
        "<a href=\"metadata.json\">metadata.json</a></p></body></html>",
        encoding="utf-8",
    )
    return {
        "directory": str(capture_dir),
        "npz": str(npz_path),
        "metadata": str(metadata_path),
        "index_html": str(index_path),
        "dual_gif": str(dual_path),
        "wrist_gif": str(gif_paths["wrist"]),
        "column_gif": str(gif_paths["column"]),
    }


def _persist_self_contact_capture(
    capture: Mapping[str, str] | None,
    record: Mapping[str, Any],
) -> dict[str, str] | None:
    """Write the opt-in seek beside a historical pulse without rewriting it."""

    if capture is None:
        return None
    directory = Path(capture["directory"])
    sidecar = directory / "self_contact.json"
    arrays = persistable_self_contact_arrays(record)
    npz_path = directory / "self_contact.npz"
    np.savez_compressed(npz_path, **arrays)
    persistable = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "visual_column",
            "visual_wrist",
            "plateau",
            "arm_states",
            "column_planning_224",
            "wrist_planning_224",
            "column_native",
            "wrist_native",
        }
    }
    persistable["visual_column"] = record.get("visual_column")
    persistable["visual_wrist"] = record.get("visual_wrist")
    persistable["plateau"] = record.get("plateau")
    persistable["arm_states"] = record.get("arm_states")
    persistable = attach_nominal_prediction(persistable)
    sidecar.write_text(
        json.dumps(_jsonable(persistable), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    metadata_path = Path(capture["metadata"])
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["self_contact"] = {
            "schema": record.get("schema"),
            "evidence_schema_version": record.get("evidence_schema_version"),
            "images_persisted_complete": record.get("images_persisted_complete"),
            "contact": record.get("contact"),
            "abort_reason": record.get("abort_reason"),
            "json": str(sidecar),
            "npz": str(npz_path),
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
        )
    paths = {
        "self_contact_json": str(sidecar),
        "self_contact_npz": str(npz_path),
    }
    return paths


def _preview_safe_state(state: np.ndarray) -> tuple[np.ndarray, bool]:
    """Clamp a copy for MuJoCo display without changing the executable command."""

    values = np.asarray(state, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("preview state must be one finite six-joint vector")
    safe = values.copy()
    safe[:5] = np.clip(safe[:5], -100.0, 100.0)
    safe[5] = np.clip(safe[5], 0.0, 100.0)
    return safe, not np.array_equal(safe, values)


def _draw_live_xy_trace(
    draw: ImageDraw.ImageDraw,
    commanded: np.ndarray,
    predicted: np.ndarray,
    current: int,
) -> None:
    """Draw command-reference and LeWM-predicted EEF paths in base-frame XY."""

    left, top, width, height = 60, 655, 520, 120
    both = np.concatenate([commanded[:, :2], predicted[:, :2]], axis=0) * 1000.0
    low = np.min(both, axis=0)
    high = np.max(both, axis=0)
    span = np.maximum(high - low, 1.0)
    margin = np.maximum(0.08 * span, 2.0)
    low -= margin
    high += margin

    def points(values: np.ndarray) -> list[tuple[float, float]]:
        xy = values[:, :2] * 1000.0
        return [
            (
                left + width * float((point[0] - low[0]) / (high[0] - low[0])),
                top + height * (1.0 - float((point[1] - low[1]) / (high[1] - low[1]))),
            )
            for point in xy[: current + 1]
        ]

    draw.rectangle(
        (left, top, left + width, top + height),
        fill="#f7f8f9",
        outline="#bcc5ca",
    )
    command_points = points(commanded)
    prediction_points = points(predicted)
    if len(command_points) > 1:
        draw.line(command_points, fill=COMMAND_COLOR, width=4)
        draw.line(prediction_points, fill=PREDICTION_COLOR, width=4)
    if command_points:
        cx, cy = command_points[-1]
        px, py = prediction_points[-1]
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=COMMAND_COLOR)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=PREDICTION_COLOR)
    draw.text(
        (left, top - 24),
        "Current H1-H8 base-frame XY path",
        fill="#20242a",
        font=load_font(18, True),
    )


def _compose_live_mujoco_frame(
    real_wrist: np.ndarray,
    predicted_image: np.ndarray,
    *,
    phase: str,
    candidate_index: int,
    horizon: int,
    raw_ahead: int,
    command_xyz: np.ndarray,
    predicted_xyz: np.ndarray,
    command_path: np.ndarray,
    prediction_path: np.ndarray,
    predicted_clipped: bool,
) -> Image.Image:
    """Match the established frozen-real / moving-prediction replay layout."""

    canvas = Image.new("RGB", (1280, 800), "#eef1f2")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26, True)
    label_font = load_font(20, True)
    text_font = load_font(16)
    draw.text(
        (40, 22),
        "SO-ARM101 MuJoCo: live observation vs LeWM forecast",
        fill="#182024",
        font=title_font,
    )
    draw.text(
        (40, 58),
        f"phase {phase} | selected candidate #{candidate_index} | human-confirmed MPC preview",
        fill="#4a5358",
        font=text_font,
    )
    draw.text(
        (40, 84),
        f"H{horizon}: {raw_ahead} raw control steps ahead from the current RGB/state and candidate actions",
        fill="#4a5358",
        font=text_font,
    )
    left = Image.fromarray(fit_panel_image(real_wrist), mode="RGB")
    right = Image.fromarray(fit_panel_image(predicted_image), mode="RGB")
    canvas.paste(left, (40, 130))
    canvas.paste(right, (670, 130))
    draw.rectangle((40, 130, 610, 610), outline=COMMAND_COLOR, width=5)
    draw.rectangle((670, 130, 1240, 610), outline=PREDICTION_COLOR, width=5)
    draw.text(
        (56, 145),
        "REAL WRIST CAMERA OBSERVATION (FROZEN)",
        fill=COMMAND_COLOR,
        font=label_font,
    )
    draw.text(
        (686, 145),
        "LEWM -> MUJOCO PREDICTION",
        fill=PREDICTION_COLOR,
        font=label_font,
    )

    divergence_mm = 1000.0 * float(np.linalg.norm(predicted_xyz - command_xyz))
    draw.text(
        (630, 666),
        f"Pred-command EEF gap {divergence_mm:.1f} mm\n"
        + (
            "Prediction clipped for display only"
            if predicted_clipped
            else "No prediction display clipping"
        ),
        fill="#20242a",
        font=text_font,
    )
    _draw_live_xy_trace(draw, command_path, prediction_path, horizon)
    draw.text(
        (910, 678),
        "Teal: candidate command prefix via FK\n"
        "Red: LeWM-predicted state via FK\n"
        "Real RGB is frozen; MuJoCo is preview only",
        fill="#4a5358",
        font=text_font,
    )
    return canvas


def _parse_dicg_shadow_extrinsic_index(value: str) -> int | str:
    text = str(value).strip()
    if not text:
        raise argparse.ArgumentTypeError("DICG shadow extrinsic index is empty")
    try:
        return int(text, 10)
    except ValueError:
        return text


def _validate_dicg_shadow_cli(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    companions = (
        args.dicg_shadow_vit_checkpoint,
        args.dicg_shadow_dataset_contract,
        args.dicg_shadow_bundle,
        args.dicg_shadow_mujoco_model,
        args.dicg_shadow_extrinsic_index,
        args.dicg_shadow_log,
        args.dicg_shadow_max_latency_ms,
    )
    checkpoint = args.dicg_shadow_checkpoint
    if checkpoint is None:
        if any(value is not None for value in companions):
            parser.error(
                "DICG shadow geometry/log flags require --dicg-shadow-checkpoint; "
                "the scorer checkpoint never replaces --model-checkpoint or "
                "--v11-control-checkpoint"
            )
        return
    if any(value is None for value in companions):
        parser.error(
            "--dicg-shadow-checkpoint requires --dicg-shadow-vit-checkpoint, "
            "--dicg-shadow-dataset-contract, --dicg-shadow-bundle, "
            "--dicg-shadow-mujoco-model, --dicg-shadow-extrinsic-index, "
            "--dicg-shadow-log, and --dicg-shadow-max-latency-ms"
        )
    if args.replay_session is not None:
        parser.error("DICG shadow is live-observation-only and refuses --replay-session")
    latency = args.dicg_shadow_max_latency_ms
    if not np.isfinite(latency) or float(latency) <= 0.0:
        parser.error("--dicg-shadow-max-latency-ms must be finite and positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-checkpoint", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--model-checkpoint", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--global-tip-checkpoint",
        type=Path,
        default=DEFAULT_GLOBAL_TIP,
        help="Markerless full-image fingertip locator used before local FCPG.",
    )
    parser.add_argument(
        "--disable-global-tip-locator",
        action="store_true",
        help="Use only calibrated/recursive local FCPG seeds.",
    )
    parser.add_argument(
        "--global-tip-views",
        nargs="+",
        choices=VIEWS,
        default=("column",),
        help=(
            "Views that use markerless full-image fingertip seeds. The wrist "
            "defaults to its rigid camera-to-gripper geometry."
        ),
    )
    parser.add_argument(
        "--memory-policy",
        choices=("auto", "off", "always"),
        default="auto",
        help=(
            "V9 history policy: auto preserves task-local parity while current "
            "FCPG is reliable and uses memory only as a fallback."
        ),
    )
    # Defaults to None so an omitted flag is distinguishable from an explicit
    # one, and is then resolved per ingestion path below: the live path needs
    # the geometry-versioned artifact, replay keeps the historical one.  An
    # explicit --calibration always wins and is never rewritten.
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=(
            "Projection calibration. Defaults to the geometry-versioned "
            f"{DEFAULT_LIVE_CALIBRATION.name} on the live path (which declares "
            "letterbox_v1) and to the historical unversioned "
            f"{DEFAULT_CALIBRATION.name} under --replay-session."
        ),
    )
    parser.add_argument(
        "--live-anchor-calibration",
        type=Path,
        help="Previously captured open/closed moving-tip track for column geometry.",
    )
    parser.add_argument(
        "--disable-live-anchor-calibration",
        action="store_true",
        help="Ignore both CLI and session live-anchor calibration for this run.",
    )
    parser.add_argument("--secondary-cache", type=Path, default=DEFAULT_SECONDARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--target-profile",
        type=Path,
        help="Persistent appearance templates; defaults to OUTPUT/target_profile.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument(
        "--control-mode",
        choices=("phase_target", "anchor_trajectory"),
        default="phase_target",
        help=(
            "phase_target generates target-conditioned IK candidates and ranks "
            "their consequences with LeWM; anchor_trajectory is replay-only "
            "demonstration continuation."
        ),
    )
    parser.add_argument(
        "--experiment-method",
        choices=("full", "no_context"),
        default="full",
        help=(
            "Candidate-selection assignment. no_context hides Peg geometry "
            "only from selection; the common execution supervisor still "
            "checks the registered Peg and may reject with zero commands."
        ),
    )
    parser.add_argument(
        "--experiment-manifest",
        type=Path,
        help="Frozen H0-H2 schedule used to bind this hardware attempt.",
    )
    parser.add_argument(
        "--experiment-attempt-id",
        help="Exact attempt_id selected from --experiment-manifest.",
    )
    parser.add_argument(
        "--operator-id",
        help="Non-secret operator label recorded with a formal hardware attempt.",
    )
    parser.add_argument(
        "--max-plan-age-s",
        type=float,
        default=120.0,
        help=(
            "Maximum plan age in seconds; 0 disables time-based expiry and is "
            "only accepted without hardware motion permissions."
        ),
    )
    parser.add_argument("--max-state-drift", type=float, default=2.0)
    parser.add_argument("--max-target-drift-px", type=float, default=15.0)
    parser.add_argument("--max-command-jump", type=float, default=8.0)
    parser.add_argument(
        "--wrist-registration-distance-m",
        type=float,
        default=0.13,
        help=(
            "Pause an approach at or inside this planar distance until the "
            "pickup has been registered in the frozen wrist frame."
        ),
    )
    parser.add_argument(
        "--grounding-feedback-config",
        type=Path,
        default=None,
        help=(
            "Enable FCPG grounding feedback from this perception YAML. When "
            "set, the near-field acquisition window is resolved autonomously: "
            "grounding verifies the planner's proposal, requests one near-view "
            "observation if uncertainty exceeds the grasp tolerance, registers "
            "a reference, and resumes. Any verification failure halts and "
            "reports. Omit to keep the operator-registration path unchanged."
        ),
    )
    parser.add_argument(
        "--grounding-object-width-m",
        type=float,
        default=None,
        help=(
            "Override the object width used to derive grasp tolerance. This "
            "value decides whether an observation is requested at all, and it "
            "is ASSUMED in config until measured with calipers."
        ),
    )
    parser.add_argument(
        "--grounding-observe-only",
        action="store_true",
        help=(
            "Run every grounding check including near-field detection, report "
            "the result, but never commit a reference. For measuring the "
            "feasibility boundary across poses without each measurement "
            "changing what the next one observes. Execution still halts on a "
            "withheld registration, so the arm cannot proceed on a reference "
            "that was not created."
        ),
    )
    parser.add_argument(
        "--grounding-dry-run",
        action="store_true",
        help=(
            "Bootstrap, print one grounding trace against the live plan, and "
            "exit without starting the HTTP server. Reads cameras and runs the "
            "planner; commands no motion and registers nothing. Use this to "
            "verify the verdict path before arming execution."
        ),
    )
    parser.add_argument(
        "--ik-arrival-raw-steps",
        type=int,
        default=20,
        help="Raw commands used by realtime IK to reach its terminal target.",
    )
    parser.add_argument(
        "--max-planar-step-m",
        type=float,
        default=0.025,
        help="Maximum XY travel of the tool per planning round, meters.",
    )
    parser.add_argument(
        "--max-z-step-m",
        type=float,
        default=0.020,
        help="Maximum vertical travel of the tool per planning round, meters.",
    )
    parser.add_argument(
        "--live-action-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply FCPG image-servo gain/step, column servo gain, and "
            "GRASP jaw rates. 1.0 keeps the 0.65-0.95 deg/step caps; "
            "live launch uses 3.0."
        ),
    )
    parser.add_argument(
        "--column-extrinsic-max-offset-px",
        type=float,
        default=60.0,
        help=(
            "Maximum accepted column metal-tip translation from FK, in the "
            "224-pixel planning frame. Values above the registered 60-pixel "
            "default require --v168-supervised-debug and are capped at 80."
        ),
    )
    parser.add_argument(
        "--column-extrinsic-max-residual-px",
        type=float,
        default=8.0,
        help=(
            "Maximum RMS disagreement between the observed jaw sweep and FK, "
            "in planning pixels. Values above the registered 8-pixel default "
            "require --v168-supervised-debug and are capped at 16."
        ),
    )
    parser.add_argument(
        "--peg-clearance-m",
        type=float,
        default=0.018,
        help=(
            "Operator-declared composite base-XY envelope from the peg center "
            "to each nominal command-FK anchor polyline; not a link/mesh "
            "collision guarantee."
        ),
    )
    parser.add_argument("--peg-radius-m", type=float)
    parser.add_argument("--represented-gripper-extent-m", type=float)
    parser.add_argument("--calibration-error-bound-m", type=float)
    parser.add_argument("--tracking-backprojection-error-bound-m", type=float)
    parser.add_argument("--additional-clearance-margin-m", type=float)
    parser.add_argument(
        "--trial-contact-height-m",
        type=float,
        default=None,
        help=(
            "Explicit trial contact height (m) above base datum. "
            "When provided, overrides bridge-derived contact_height_m. "
            "Use for experimental geometry not represented in training data."
        ),
    )
    parser.add_argument(
        "--minimum-expected-motion",
        type=float,
        default=3.0,
        help=(
            "No-follow watchdog floor. Must not be below "
            "--settle-position-tolerance: the settle loop accepts any pose "
            "within that band, so a commanded move smaller than the band can "
            "legitimately produce zero observed motion."
        ),
    )
    parser.add_argument("--minimum-observed-motion", type=float, default=0.1)
    parser.add_argument(
        "--mpc-only-collection",
        action="store_true",
        help=(
            "Phase 2.5: LeWM shadow logging only. Predictions computed but have "
            "zero influence on cost, ranking, safety filtering, or rejection. "
            "Deterministic MPC-native safety still enforced."
        ),
    )
    parser.add_argument("--command-guard-margin", type=float, default=0.5)
    parser.add_argument("--settle-timeout-s", type=float, default=3.0)
    parser.add_argument("--settle-position-tolerance", type=float, default=3.0)
    parser.add_argument("--settle-step-tolerance", type=float, default=0.25)
    parser.add_argument("--settle-samples", type=int, default=3)
    parser.add_argument("--replay-session", type=Path)
    parser.add_argument("--replay-index", type=int, default=54)
    parser.add_argument(
        "--home-state-report",
        "--object-prior-report",
        dest="home_state_report",
        type=Path,
        help="Load only the six-joint task-start state from a prior report.",
    )
    parser.add_argument("--robot-port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--robot-id", default="myfollower01")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--wrist-device", default=DEFAULT_WRIST_DEVICE)
    parser.add_argument("--column-device", default=DEFAULT_COLUMN_DEVICE)
    parser.add_argument("--execute-enabled", action="store_true")
    parser.add_argument(
        "--bounded-autonomy-enabled",
        action="store_true",
        help=(
            "Allow one explicit non-formal authorization to run multiple "
            "freshly supervised five-command chunks."
        ),
    )
    parser.add_argument("--max-autonomous-chunks", type=int, default=16)
    parser.add_argument(
        "--max-autonomous-duration-s", type=float, default=180.0
    )
    parser.add_argument("--gripper-calibration-enabled", action="store_true")
    parser.add_argument(
        "--self-contact-calibration",
        action="store_true",
        help=(
            "After the nominal close-open-restore pulse, run an opt-in empty-jaw "
            "self-contact seek. Default off. Does not change closed_state or "
            "the live planner path. Requires --i-accept-empty-jaw-self-contact "
            "and --gripper-calibration-enabled."
        ),
    )
    parser.add_argument(
        "--i-accept-empty-jaw-self-contact",
        action="store_true",
        default=False,
        help=(
            "Self-contact-specific acknowledgement required in addition to "
            "--self-contact-calibration. Default off. Does not change the "
            "historical pulse or closed_state."
        ),
    )
    parser.add_argument("--acknowledge-workspace-clear", action="store_true")
    parser.add_argument("--no-mujoco-preview", action="store_true")
    parser.add_argument(
        "--v11-shadow-checkpoint",
        type=Path,
        default=None,
        help="V11 checkpoint for non-actuating shadow inference (trial4+)",
    )
    parser.add_argument(
        "--v11-control-checkpoint",
        type=Path,
        default=None,
        help=(
            "V11 checkpoint for actuating control. Replaces B2/LeWM ranking "
            "as the executed action source. Safety supervisor still applies."
        ),
    )
    parser.add_argument(
        "--v11-vit-checkpoint",
        type=Path,
        default=REPO / "outputs/lerobot/so101/so101_v9_local_multiscale_seed5131/latest.pt",
        help="Frozen V9 ViT checkpoint for V11 initialization",
    )
    parser.add_argument(
        "--prod-v13-observer-shadow",
        action="store_true",
        help=(
            "Attach the non-actuating Prod-V13 measurement/prior/posterior "
            "observer. Off by default and never used by planning or control."
        ),
    )
    parser.add_argument(
        "--prod-v13-checkpoint",
        type=Path,
        help="Fixed Prod-V13 recursive checkpoint used only by the shadow.",
    )
    parser.add_argument(
        "--prod-v13-v12-checkpoint",
        type=Path,
        help="Exact frozen V12.4 step-20000 dependency for Prod-V13 shadow.",
    )
    parser.add_argument(
        "--prod-v13-fcpg-checkpoint",
        type=Path,
        help="Exact frozen protected-memory FCPG-H v3.1 dependency.",
    )
    parser.add_argument(
        "--prod-v13-actor-checkpoint",
        type=Path,
        help=(
            "Exact frozen V12.1 actor_best@4750 reference; validated but never "
            "loaded into the Prod-V13 optimizer or shadow action path."
        ),
    )
    parser.add_argument(
        "--prod-v13-multirate-shadow",
        action="store_true",
        help=(
            "Attach the fixed CoTracker/V12.4/PartTokenLeWM observer. Off by "
            "default, non-actuating, and distinct from recursive-v3.2."
        ),
    )
    parser.add_argument(
        "--prod-v13-multirate-checkpoint",
        type=Path,
        help="Fixed step-20000 PartTokenLeWM endpoint used only by the shadow.",
    )
    parser.add_argument(
        "--prod-v13-multirate-v12-checkpoint",
        type=Path,
        help="Exact frozen V12.4 step-20000 source for live patch tokens.",
    )
    parser.add_argument(
        "--prod-v13-cotracker-repo",
        type=Path,
        help="Pinned local CoTracker repository; live startup remains offline.",
    )
    parser.add_argument(
        "--prod-v13-cotracker-checkpoint",
        type=Path,
        help="Pinned local CoTracker3-online weights.",
    )
    parser.add_argument(
        "--prod-v14-shadow",
        action="store_true",
        help=(
            "Attach the selected V14 two-stage step-2000 dual-view ranker. "
            "Off by default and counterfactual-only."
        ),
    )
    parser.add_argument(
        "--prod-v14-checkpoint",
        type=Path,
        help="Selected V14 step-2000 or explicit archived step-500 checkpoint.",
    )
    parser.add_argument(
        "--prod-v16-live-secant-shadow",
        action="store_true",
        help=(
            "Attach the causal V16.1 dual-view live secant candidate. It uses "
            "only completed actions and remains post-plan/non-actuating."
        ),
    )
    parser.add_argument(
        "--task-class",
        type=int,
        default=None,
        help=(
            "Human-set actor skill ID. Required with --v11-control-checkpoint. "
            "Class 1 is grasp-the-red-fish / fish_to_box_task_v1. The model "
            "does not infer the skill."
        ),
    )
    parser.add_argument(
        "--recursive-observer-shadow",
        action="store_true",
        help=(
            "Run the 3-D column information filter each plan. Off by default. "
            "Does not move the arm by itself; pair with --observer-close-sigma-m "
            "to let the posterior gate closes."
        ),
    )
    parser.add_argument(
        "--observer-close-sigma-m",
        type=float,
        default=None,
        help=(
            "Fail-closed jaw-close gate: refuse close unless the observer's "
            "along-close posterior std is at or below this many metres. "
            "Typical live value is 0.012."
        ),
    )
    parser.add_argument(
        "--observer-active-excitation",
        action="store_true",
        help=(
            "Log a D-optimal APPROACH micro-step. The live command path does "
            "not apply that step yet; this flag only records the decision."
        ),
    )
    parser.add_argument(
        "--observer-bundle",
        type=Path,
        default=None,
        help=(
            "column_intrinsics_from_task_v1.json used to seed camera pose. "
            "K stays the 45-degree FOV prior; only T_cb / joint offsets seed."
        ),
    )
    parser.add_argument(
        "--observer-drop-3d",
        type=Path,
        default=None,
        dest="observer_drop_3d_path",
        help=(
            "JSON path for a base-frame drop point. Loaded at serve start and "
            "rewritten when the observer first sees a drop. Reprojection "
            "replaces brown-blob re-hunting."
        ),
    )
    parser.add_argument(
        "--observer-servo-load",
        action="store_true",
        help=(
            "Persist Present_Load and Present_Current once per planning round. "
            "Off by default. Observability only: two extra servo group-reads, "
            "never an input to planning, gating, or the servo command."
        ),
    )
    parser.add_argument(
        "--grounding-mode",
        choices=("nominal", "gated_fcpg"),
        default="gated_fcpg",
        help=(
            "Planner-facing anchors: nominal leaves FK uncorrected; "
            "gated_fcpg consumes FCPG / CoarseFine fusion. Ranking cannot "
            "change this choice."
        ),
    )
    parser.add_argument(
        "--anchor-authority",
        choices=("nominal", "original_fcpg", "tuned_fcpg"),
        default=None,
        help=(
            "Enable the immutable V16.5 runtime bundle and freeze the sole "
            "planner anchor authority for the entire server session."
        ),
    )
    parser.add_argument(
        "--v165-original-fcpg-checkpoint",
        type=Path,
        default=None,
        help="Registered ep2-blind original independent3 FCPG checkpoint.",
    )
    parser.add_argument(
        "--v165-source-checkpoint",
        type=Path,
        default=None,
        help="Registered V12.4 source carrying the frozen FCPG patch projector.",
    )
    parser.add_argument(
        "--v165-tuned-fcpg-checkpoint",
        type=Path,
        default=None,
        help="Frozen V16.5 physical-click FCPG checkpoint.",
    )
    parser.add_argument(
        "--v165-lewm-checkpoint",
        type=Path,
        default=None,
        help="Frozen V16.5 State/FK LeWM checkpoint.",
    )
    parser.add_argument(
        "--v165-actor-checkpoint",
        type=Path,
        default=None,
        help="Frozen V16.5 320k structured-goal Actor checkpoint.",
    )
    parser.add_argument(
        "--v165-no-contact-approach",
        action="store_true",
        help=(
            "One-chunk approach-only hardware gate: prohibit jaw close, "
            "GRASP phase, calibration pulses, and TCP descent into the "
            "registered contact plane."
        ),
    )
    parser.add_argument(
        "--v165-contact-authorization-receipt",
        type=Path,
        default=None,
        help=(
            "Separately reviewed receipt binding replay, shadow, zero-motion, "
            "and no-contact GO evidence to this exact V16.5 bundle. Without "
            "it, a V16.5 execute session is forced into no-contact mode."
        ),
    )
    parser.add_argument(
        "--v168-mode",
        choices=("shadow", "candidate"),
        help=(
            "Run the receipt-bound V16.8 full stack. shadow forwards original "
            "FCPG, State/FK LeWM and Actor but sends zero commands; candidate "
            "adds exactly one Actor proposal to the common LeWM-ranked bank."
        ),
    )
    parser.add_argument(
        "--v168-original-fcpg-checkpoint",
        type=Path,
        help="Registered original independent3 FCPG checkpoint for V16.8.",
    )
    parser.add_argument(
        "--v168-source-checkpoint",
        type=Path,
        help="Registered V12 source carrying V16.8 FCPG patch projection.",
    )
    parser.add_argument(
        "--v168-lewm-checkpoint",
        type=Path,
        help="Exact formally-GO frozen State/FK LeWM checkpoint for V16.8.",
    )
    parser.add_argument(
        "--v168-lewm-receipt",
        type=Path,
        help="Exact GO receipt binding --v168-lewm-checkpoint.",
    )
    parser.add_argument(
        "--v168-training-receipt",
        type=Path,
        help="Exact V16.8 pass8 Actor training receipt.",
    )
    parser.add_argument(
        "--v168-actor-checkpoint",
        type=Path,
        help="Exact V16.8 full/main/pass8 Actor checkpoint.",
    )
    parser.add_argument(
        "--v168-live-execute",
        action="store_true",
        help=(
            "Second explicit permission for candidate-mode hardware execution; "
            "defaults to one-chunk no-contact unless supervised debug is also "
            "explicitly enabled; never a performance gate."
        ),
    )
    parser.add_argument(
        "--v168-supervised-debug",
        action="store_true",
        help=(
            "Operator-supervised V16.8 debug session: allow the existing "
            "bounded multi-chunk, calibration, excitation, and home entrypoints. "
            "Does not bypass candidate legality, hard joint bounds, step caps, "
            "projector checks, fresh-plan checks, or the runtime watchdog."
        ),
    )
    parser.add_argument(
        "--v170-mode",
        choices=("shadow", "candidate"),
        help=(
            "Run the plain V17.0 FCPG + LeWM + Actor stack. shadow records "
            "settled-endpoint inference with commands_sent=0; candidate adds "
            "exactly one Actor proposal to the common bounded planner."
        ),
    )
    parser.add_argument(
        "--v170-fcpg-checkpoint",
        type=Path,
        help="Plain final V17.0 FCPG checkpoint.",
    )
    parser.add_argument(
        "--v170-lewm-checkpoint",
        type=Path,
        help="Plain final V17.0 State/FK LeWM checkpoint.",
    )
    parser.add_argument(
        "--v170-actor-checkpoint",
        type=Path,
        help="Plain final V17.0 five-command Actor checkpoint.",
    )
    parser.add_argument(
        "--v170-live-execute",
        action="store_true",
        help=(
            "Second explicit permission for V17.0 candidate hardware "
            "execution. The common bounds, projector, freshness checks, and "
            "runtime watchdog remain active."
        ),
    )
    parser.add_argument(
        "--ranking-mode",
        choices=("off", "on"),
        default="on",
        help=(
            "LeWM candidate ranking. off is FCPG-only selection; on is "
            "FCPG+LeWM. Does not change corrected anchors."
        ),
    )
    parser.add_argument(
        "--allow-uncalibrated-coarse-fine-shadow",
        action="store_true",
        help=(
            "Permit an uncalibrated / no-physical-label coarse_fine "
            "checkpoint in offline or shadow only. Live execute stays refused."
        ),
    )
    parser.add_argument(
        "--disable-wrist-correction",
        action="store_true",
        help="Disable the eye-in-hand wrist overlay. CoarseFine forces this off.",
    )
    parser.add_argument(
        "--dicg-shadow-checkpoint",
        type=Path,
        default=None,
        help=(
            "Non-actuating DICG hypothesis-scorer checkpoint. Does not replace "
            "--model-checkpoint or --v11-control-checkpoint."
        ),
    )
    parser.add_argument(
        "--dicg-shadow-vit-checkpoint",
        type=Path,
        default=None,
        help="Frozen ViT checkpoint required with --dicg-shadow-checkpoint.",
    )
    parser.add_argument(
        "--dicg-shadow-dataset-contract",
        type=Path,
        default=None,
        help="Training-identical DICG motion NPZ required with the shadow checkpoint.",
    )
    parser.add_argument(
        "--dicg-shadow-bundle",
        type=Path,
        default=None,
        help="Observer bundle JSON required with --dicg-shadow-checkpoint.",
    )
    parser.add_argument(
        "--dicg-shadow-mujoco-model",
        type=Path,
        default=None,
        help="Explicit MuJoCo model required with --dicg-shadow-checkpoint.",
    )
    parser.add_argument(
        "--dicg-shadow-extrinsic-index",
        type=_parse_dicg_shadow_extrinsic_index,
        default=None,
        help="Explicit bundle extrinsic index or audited group key. No silent default.",
    )
    parser.add_argument(
        "--dicg-shadow-log",
        type=Path,
        default=None,
        help="JSONL path for non-actuating DICG shadow records.",
    )
    parser.add_argument(
        "--dicg-shadow-max-latency-ms",
        type=float,
        default=None,
        help="Shadow worker latency budget in milliseconds. Required with the checkpoint.",
    )
    args = parser.parse_args()
    # B1 calibration default, resolved per ingestion path and never inferred
    # from the artifact's contents.  The live path declares `letterbox_v1`
    # (`_declared_pixel_geometry`) and the planner loads the projection fit
    # under a strict geometry contract, so the live default must be the
    # geometry-versioned artifact; the historical unversioned file declares no
    # protocol and is correctly refused there.  Replay declares `squash_v0`
    # and keeps the historical default, so no legacy/offline behaviour moves.
    # An explicitly passed --calibration is left exactly as given, for either
    # path, so an operator can still point at any artifact on purpose.
    if args.calibration is None:
        args.calibration = (
            DEFAULT_CALIBRATION
            if args.replay_session is not None
            else DEFAULT_LIVE_CALIBRATION
        )
    if args.column_device == "auto":
        try:
            args.column_device = _select_auto_column_device(args.wrist_device)
        except ValueError as exc:
            parser.error(str(exc))
    if args.device.startswith("cuda:1"):
        parser.error("CUDA1 is forbidden for this project")
    if args.http_port <= 0 or args.replay_index < 0:
        parser.error("HTTP port and replay index are invalid")
    if (
        isinstance(args.max_autonomous_chunks, bool)
        or not 1 <= args.max_autonomous_chunks <= 16
    ):
        parser.error("--max-autonomous-chunks must be in [1,16]")
    if (
        not np.isfinite(args.max_autonomous_duration_s)
        or args.max_autonomous_duration_s <= 0
        or args.max_autonomous_duration_s > 180.0
    ):
        parser.error(
            "--max-autonomous-duration-s must be in (0,180]"
        )
    if (
        args.bounded_autonomy_enabled
        and args.replay_session is None
        and not args.execute_enabled
    ):
        parser.error(
            "hardware bounded autonomy requires --execute-enabled"
        )
    for value in (
        args.control_hz,
        args.max_state_drift,
        args.max_target_drift_px,
        args.max_command_jump,
        args.wrist_registration_distance_m,
        args.command_guard_margin,
        args.settle_timeout_s,
        args.settle_position_tolerance,
        args.settle_step_tolerance,
        args.max_planar_step_m,
        args.max_z_step_m,
        args.live_action_scale,
        args.column_extrinsic_max_offset_px,
        args.column_extrinsic_max_residual_px,
        args.peg_clearance_m,
    ):
        if not np.isfinite(value) or value <= 0:
            parser.error("runtime thresholds must be finite and positive")
    if args.live_action_scale > 8.0:
        parser.error("--live-action-scale must be in (0, 8]")
    widened_column_calibration = (
        args.column_extrinsic_max_offset_px > 60.0
        or args.column_extrinsic_max_residual_px > 8.0
    )
    if widened_column_calibration and not args.v168_supervised_debug:
        parser.error(
            "widened column calibration bounds require "
            "--v168-supervised-debug"
        )
    if (
        args.column_extrinsic_max_offset_px > 80.0
        or args.column_extrinsic_max_residual_px > 16.0
    ):
        parser.error(
            "supervised-debug column calibration bounds are capped at "
            "80 px offset and 16 px RMS residual"
        )
    if args.observer_close_sigma_m is not None and (
        not np.isfinite(args.observer_close_sigma_m)
        or args.observer_close_sigma_m <= 0
        or args.observer_close_sigma_m > 0.2
    ):
        parser.error("--observer-close-sigma-m must be in (0, 0.2]")
    if not np.isfinite(args.max_plan_age_s) or args.max_plan_age_s < 0:
        parser.error("--max-plan-age-s must be finite and non-negative")
    if args.settle_samples <= 0:
        parser.error("--settle-samples must be positive")
    if args.ik_arrival_raw_steps <= 0 or args.ik_arrival_raw_steps % 5:
        parser.error("--ik-arrival-raw-steps must be a positive multiple of 5")
    if args.command_guard_margin >= args.max_command_jump:
        parser.error("--command-guard-margin must be below --max-command-jump")
    if args.self_contact_calibration and not args.i_accept_empty_jaw_self_contact:
        parser.error(
            "self-contact calibration is not armed without "
            "--i-accept-empty-jaw-self-contact in addition to "
            "--self-contact-calibration; default remains off and "
            "does not change closed_state"
        )
    if args.self_contact_calibration and not args.gripper_calibration_enabled:
        parser.error(
            "self-contact calibration requires --gripper-calibration-enabled; "
            "the historical pulse still supplies closed_state"
        )
    if (
        args.execute_enabled
        or args.gripper_calibration_enabled
        or args.self_contact_calibration
    ) and not args.acknowledge_workspace_clear:
        parser.error(
            "motion permissions require --acknowledge-workspace-clear"
        )
    if args.replay_session is not None and (
        args.execute_enabled
        or args.gripper_calibration_enabled
        or args.self_contact_calibration
    ):
        parser.error("replay mode does not accept hardware motion permissions")
    if args.max_plan_age_s == 0 and (
        args.execute_enabled
        or args.gripper_calibration_enabled
        or args.self_contact_calibration
    ):
        parser.error(
            "--max-plan-age-s 0 (no expiry) is not allowed with hardware "
            "motion permissions"
        )
    if args.control_mode == "anchor_trajectory" and (
        args.execute_enabled
        or args.gripper_calibration_enabled
        or args.self_contact_calibration
    ):
        parser.error(
            "anchor_trajectory is replay-only; hardware motion requires "
            "--control-mode phase_target"
        )
    _validate_dicg_shadow_cli(parser, args)
    prod_v13_paths = (
        args.prod_v13_checkpoint,
        args.prod_v13_v12_checkpoint,
        args.prod_v13_fcpg_checkpoint,
        args.prod_v13_actor_checkpoint,
    )
    if args.prod_v13_observer_shadow:
        if any(path is None for path in prod_v13_paths):
            parser.error(
                "--prod-v13-observer-shadow requires its checkpoint, V12.4, "
                "FCPG-H v3.1 and V12.1 actor reference"
            )
    elif any(path is not None for path in prod_v13_paths):
        parser.error(
            "Prod-V13 shadow checkpoint paths require "
            "--prod-v13-observer-shadow"
        )
    multirate_paths = (
        args.prod_v13_multirate_checkpoint,
        args.prod_v13_multirate_v12_checkpoint,
        args.prod_v13_cotracker_repo,
        args.prod_v13_cotracker_checkpoint,
    )
    if args.prod_v13_multirate_shadow and args.prod_v13_observer_shadow:
        parser.error("recursive-v3.2 and multi-rate Prod-V13 shadows are exclusive")
    if args.prod_v13_multirate_shadow:
        if any(path is None for path in multirate_paths):
            parser.error(
                "--prod-v13-multirate-shadow requires its checkpoint, V12.4, "
                "CoTracker repository and CoTracker checkpoint"
            )
    elif any(path is not None for path in multirate_paths):
        parser.error(
            "multi-rate Prod-V13 paths require --prod-v13-multirate-shadow"
        )
    if args.prod_v14_shadow:
        if not args.prod_v13_multirate_shadow:
            parser.error("--prod-v14-shadow requires --prod-v13-multirate-shadow")
        if args.prod_v14_checkpoint is None:
            parser.error("--prod-v14-shadow requires --prod-v14-checkpoint")
    elif args.prod_v14_checkpoint is not None:
        parser.error("--prod-v14-checkpoint requires --prod-v14-shadow")
    if args.prod_v16_live_secant_shadow and not args.prod_v13_multirate_shadow:
        parser.error(
            "--prod-v16-live-secant-shadow requires "
            "--prod-v13-multirate-shadow"
        )
    v165_paths = (
        args.v165_original_fcpg_checkpoint,
        args.v165_source_checkpoint,
        args.v165_tuned_fcpg_checkpoint,
        args.v165_lewm_checkpoint,
        args.v165_actor_checkpoint,
    )
    v168_paths = (
        args.v168_original_fcpg_checkpoint,
        args.v168_source_checkpoint,
        args.v168_lewm_checkpoint,
        args.v168_lewm_receipt,
        args.v168_training_receipt,
        args.v168_actor_checkpoint,
    )
    v170_paths = (
        args.v170_fcpg_checkpoint,
        args.v170_lewm_checkpoint,
        args.v170_actor_checkpoint,
    )
    v170_requested = (
        args.v170_mode is not None
        or any(path is not None for path in v170_paths)
        or args.v170_live_execute
    )
    if v170_requested:
        v165_requested = (
            args.anchor_authority is not None
            or any(path is not None for path in v165_paths)
            or args.v165_no_contact_approach
            or args.v165_contact_authorization_receipt is not None
        )
        v168_requested = (
            args.v168_mode is not None
            or any(path is not None for path in v168_paths)
            or args.v168_live_execute
            or args.v168_supervised_debug
        )
        if v165_requested or v168_requested:
            parser.error("V17.0 cannot coexist with a V16.5 or V16.8 bundle")
    if args.v168_mode is not None:
        if any(path is not None for path in v165_paths):
            parser.error("V16.8 must not load tuned FCPG or a legacy V16.5 Actor")
        if any(path is None for path in v168_paths):
            parser.error(
                "--v168-mode requires original FCPG, V12 source, GO LeWM "
                "checkpoint/receipt, and pass8 Actor receipt/checkpoint"
            )
        if args.anchor_authority != "original_fcpg":
            parser.error("V16.8 requires --anchor-authority original_fcpg")
        if args.max_command_jump - args.command_guard_margin > 4.0 + 1.0e-6:
            parser.error(
                "V16.8 requires guarded arm jump <= 4 degrees; for the "
                "default 0.5 margin use --max-command-jump 4.5 or lower"
            )
        if args.control_mode != "phase_target" or args.grounding_mode != "gated_fcpg":
            parser.error("V16.8 requires phase_target control with gated_fcpg")
        if args.ranking_mode != "on" or args.mpc_only_collection:
            parser.error("V16.8 requires active frozen State/FK LeWM ranking")
        if args.v11_control_checkpoint is not None:
            parser.error("V16.8 Actor and legacy V11 control cannot share one session")
        if args.observer_active_excitation or args.observer_close_sigma_m is not None:
            parser.error("V16.8 keeps legacy observer excitation/sigma controls off")
        if (
            args.gripper_calibration_enabled or args.self_contact_calibration
        ) and not args.v168_supervised_debug:
            parser.error(
                "V16.8 calibration requires --v168-supervised-debug"
            )
        if args.trial_contact_height_m is not None:
            parser.error("V16.8 requires the registered bridge-derived contact plane")
        if args.v165_contact_authorization_receipt is not None:
            parser.error("V16.8 has no contact-hardware authorization receipt")
        if args.v168_mode == "shadow":
            if (
                args.execute_enabled
                or args.v168_live_execute
                or args.v168_supervised_debug
            ):
                parser.error("V16.8 shadow requires commands_sent=0")
        elif args.v168_live_execute and not args.execute_enabled:
            parser.error("--v168-live-execute requires --execute-enabled")
        elif args.execute_enabled and not args.v168_live_execute:
            parser.error("V16.8 candidate hardware execute needs --v168-live-execute")
        if args.v168_supervised_debug and (
            args.v168_mode != "candidate"
            or not args.execute_enabled
            or not args.v168_live_execute
        ):
            parser.error(
                "--v168-supervised-debug requires candidate live execute"
            )
        if args.execute_enabled:
            # The default continues to reuse the mature one-chunk no-contact
            # wrapper.  An explicit supervised-debug launch instead enables
            # the ordinary bounded runtime entrypoints while retaining their
            # deterministic safety validation.
            args.v165_no_contact_approach = not args.v168_supervised_debug
        if args.v165_no_contact_approach and args.bounded_autonomy_enabled:
            parser.error("V16.8 no-contact execution forbids multi-chunk autonomy")
    elif (
        any(path is not None for path in v168_paths)
        or args.v168_live_execute
        or args.v168_supervised_debug
    ):
        parser.error("V16.8 artifacts/permission require --v168-mode")
    elif args.anchor_authority is None:
        if any(path is not None for path in v165_paths):
            parser.error(
                "V16.5 checkpoint paths require --anchor-authority"
            )
        if args.v165_no_contact_approach:
            parser.error(
                "--v165-no-contact-approach requires a complete V16.5 bundle"
            )
        if args.v165_contact_authorization_receipt is not None:
            parser.error(
                "--v165-contact-authorization-receipt requires a complete V16.5 bundle"
            )
    else:
        if any(path is None for path in v165_paths):
            parser.error(
                "--anchor-authority requires original/tuned FCPG, V12 source, LeWM, and Actor checkpoints"
            )
        if args.v11_control_checkpoint is not None:
            parser.error(
                "V16.5 Actor and legacy V11 control cannot share one session"
            )
        if args.grounding_mode != "gated_fcpg":
            parser.error(
                "V16.5 uses --anchor-authority and requires --grounding-mode gated_fcpg"
            )
        if args.ranking_mode != "on" or args.mpc_only_collection:
            parser.error("V16.5 requires active LeWM ranking")
        if args.observer_active_excitation or args.observer_close_sigma_m is not None:
            parser.error(
                "V16.5 keeps legacy observer excitation/sigma close control decision-off"
            )
        if args.gripper_calibration_enabled or args.self_contact_calibration:
            parser.error(
                "V16.5 uses a frozen calibration; calibration and self-contact pulses are forbidden"
            )
        if args.v165_no_contact_approach and args.v165_contact_authorization_receipt is not None:
            parser.error(
                "V16.5 no-contact and contact-authorized modes are mutually exclusive"
            )
        if args.v165_contact_authorization_receipt is not None:
            parser.error(
                "V16.5 contact hardware remains sealed until post-training integration review"
            )
        if args.trial_contact_height_m is not None:
            parser.error(
                "V16.5 requires the registered bridge-derived contact plane and "
                "forbids --trial-contact-height-m"
            )
        if args.execute_enabled:
            # Fail-safe default: forgetting an optional flag may never turn a
            # deployment-ineligible training bundle into a contact controller.
            args.v165_no_contact_approach = True
        if args.v165_no_contact_approach and args.bounded_autonomy_enabled:
            parser.error(
                "V16.5 no-contact approach forbids bounded multi-chunk autonomy"
            )
    if args.v170_mode is None:
        if any(path is not None for path in v170_paths) or args.v170_live_execute:
            parser.error("V17.0 artifacts/permission require --v170-mode")
    else:
        if any(path is None for path in v170_paths):
            parser.error(
                "--v170-mode requires FCPG, LeWM, and Actor checkpoints"
            )
        if args.v170_mode == "shadow":
            if (
                args.execute_enabled
                or args.v170_live_execute
                or args.gripper_calibration_enabled
                or args.self_contact_calibration
            ):
                parser.error("V17.0 shadow requires commands_sent=0")
        elif args.v170_live_execute and not args.execute_enabled:
            parser.error("--v170-live-execute requires --execute-enabled")
        elif args.execute_enabled and not args.v170_live_execute:
            parser.error(
                "V17.0 candidate hardware execute needs --v170-live-execute"
            )
    if args.v11_control_checkpoint is not None and args.task_class is None:
        parser.error("V11 control requires a human-set --task-class")
    if args.task_class is not None:
        from v10.task_class import parse_task_class

        try:
            args.task_class = parse_task_class(args.task_class)
        except ValueError as exc:
            parser.error(str(exc))
    experiment_fields = (
        args.experiment_manifest,
        args.experiment_attempt_id,
        args.operator_id,
    )
    if any(value is not None for value in experiment_fields) and not all(
        value is not None for value in experiment_fields
    ):
        parser.error(
            "--experiment-manifest, --experiment-attempt-id, and --operator-id "
            "must be provided together"
        )
    clearance_components = (
        args.peg_radius_m,
        args.represented_gripper_extent_m,
        args.calibration_error_bound_m,
        args.tracking_backprojection_error_bound_m,
        args.additional_clearance_margin_m,
    )
    if any(value is not None for value in clearance_components):
        if not all(value is not None for value in clearance_components):
            parser.error(
                "all five physical clearance component arguments must be "
                "provided together"
            )
        if any(not np.isfinite(value) or value < 0 for value in clearance_components):
            parser.error("physical clearance components must be finite and non-negative")
    required_paths = [
        args.bridge_checkpoint,
        args.model_checkpoint,
        args.calibration,
        args.secondary_cache,
        args.experiment_manifest,
    ]
    if not args.disable_global_tip_locator:
        required_paths.append(args.global_tip_checkpoint)
    if args.v168_mode is not None:
        required_paths.extend(v168_paths)
    elif args.anchor_authority is not None:
        required_paths.extend(v165_paths)
        if args.v165_contact_authorization_receipt is not None:
            required_paths.append(args.v165_contact_authorization_receipt)
    if args.v170_mode is not None:
        required_paths.extend(v170_paths)
    if args.dicg_shadow_checkpoint is not None:
        dicg_input_paths = (
            args.dicg_shadow_checkpoint,
            args.dicg_shadow_vit_checkpoint,
            args.dicg_shadow_dataset_contract,
            args.dicg_shadow_bundle,
            args.dicg_shadow_mujoco_model,
        )
        for path in dicg_input_paths:
            if path is None or not Path(path).expanduser().exists():
                parser.error(f"required DICG shadow input does not exist: {path}")
    if args.prod_v13_observer_shadow:
        for path in prod_v13_paths:
            if path is None or not path.expanduser().is_file():
                parser.error(f"required Prod-V13 shadow input does not exist: {path}")
    if args.prod_v13_multirate_shadow:
        for path in multirate_paths:
            if path is None or not path.expanduser().exists():
                parser.error(
                    f"required multi-rate Prod-V13 input does not exist: {path}"
                )
    if args.prod_v14_shadow and (
        args.prod_v14_checkpoint is None
        or not args.prod_v14_checkpoint.expanduser().is_file()
    ):
        parser.error(
            f"required Prod-V14 shadow input does not exist: "
            f"{args.prod_v14_checkpoint}"
        )
    for path in required_paths:
        if path is None:
            continue
        if not path.expanduser().exists():
            parser.error(f"required planner input does not exist: {path}")
    try:
        args.experiment_binding = _resolve_hardware_experiment_binding(args)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(f"invalid hardware experiment binding: {exc}")
    return args


def _resolve_hardware_experiment_binding(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Resolve one manifest attempt without opening cameras or robot hardware."""

    manifest_path = getattr(args, "experiment_manifest", None)
    if manifest_path is None:
        return None
    if getattr(args, "replay_session", None) is not None:
        raise ValueError("formal hardware attempts cannot run in replay mode")
    operator_id = getattr(args, "operator_id", None)
    if (
        not isinstance(operator_id, str)
        or not operator_id.strip()
        or operator_id != operator_id.strip()
        or len(operator_id) > 128
    ):
        raise ValueError("operator_id must be a trimmed non-empty string up to 128 chars")
    attempt_id = getattr(args, "experiment_attempt_id", None)
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("experiment_attempt_id must be a non-empty string")

    resolved_manifest = Path(manifest_path).expanduser().resolve()
    raw_manifest = json.loads(resolved_manifest.read_text(encoding="ascii"))
    manifest = validate_hardware_experiment_manifest(raw_manifest)
    matches = [
        row
        for row in iter_hardware_experiment_trials(manifest)
        if row["attempt_id"] == attempt_id
    ]
    if len(matches) != 1:
        raise ValueError(f"attempt_id {attempt_id!r} is not unique in the manifest")
    assignment = matches[0]
    expected_method = assignment.get("method", "full")
    if getattr(args, "experiment_method", None) != expected_method:
        raise ValueError(
            f"attempt {attempt_id!r} requires --experiment-method {expected_method}"
        )
    if getattr(args, "control_mode", None) != "phase_target":
        raise ValueError("formal hardware attempts require phase_target control")
    if bool(getattr(args, "bounded_autonomy_enabled", False)):
        raise ValueError(
            "the frozen formal protocol requires one confirmation per chunk; "
            "bounded autonomy is non-formal"
        )

    stage = assignment["stage"]
    execute_enabled = bool(getattr(args, "execute_enabled", False))
    calibration_enabled = bool(
        getattr(args, "gripper_calibration_enabled", False)
    )
    if stage in {"h1", "h2"} and not execute_enabled:
        raise ValueError(f"{stage.upper()} attempts require --execute-enabled")
    if stage == "h0":
        adversarial = assignment.get("round_kind") == "adversarial"
        if adversarial != execute_enabled:
            required = "enabled" if adversarial else "disabled"
            raise ValueError(
                "H0 adversarial attempts require execution enabled for a "
                "durable rejection; H0 ordinary attempts require it disabled "
                f"(this assignment requires {required})"
            )
    if calibration_enabled:
        raise ValueError("formal attempts cannot include a calibration pulse")

    component_names = (
        "peg_radius_m",
        "represented_gripper_extent_m",
        "calibration_error_bound_m",
        "tracking_backprojection_error_bound_m",
        "additional_clearance_margin_m",
    )
    components: dict[str, float] = {}
    for name in component_names:
        value = getattr(args, name, None)
        if value is None or not np.isfinite(value) or value < 0:
            raise ValueError(
                "formal attempts require all five measured, finite, "
                "non-negative clearance components"
            )
        components[name] = float(value)
    component_sum = float(sum(components.values()))
    frozen_clearance = float(getattr(args, "peg_clearance_m"))
    if component_sum > frozen_clearance + 1e-12:
        raise ValueError(
            "measured clearance component sum exceeds --peg-clearance-m: "
            f"{component_sum:.6f} > {frozen_clearance:.6f}"
        )

    output = Path(args.output).expanduser().resolve()
    if output.name != attempt_id or output.parent.name != "trials":
        raise ValueError(
            "formal output must be EVIDENCE_ROOT/trials/<experiment-attempt-id>"
        )
    return {
        "manifest_path": str(resolved_manifest),
        "manifest_format": manifest["format"],
        "protocol": manifest["protocol"],
        "manifest_seed": manifest["seed"],
        "budgets": manifest["budgets"],
        "operator_id": operator_id,
        "clearance_measurement": {
            **components,
            "component_sum_m": component_sum,
            "frozen_clearance_m": frozen_clearance,
        },
        **assignment,
    }


def _write_hardware_experiment_session(args: argparse.Namespace) -> Path | None:
    """Create the exclusive attempt record before hardware preflight starts."""

    binding = getattr(args, "experiment_binding", None)
    if binding is None:
        return None
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.name for path in output.iterdir())
    if existing:
        raise FileExistsError(
            f"formal attempt output must be empty; found {existing} in {output}"
        )
    destination = output / "session.json"
    source_paths = {
        "bridge_checkpoint": args.bridge_checkpoint,
        "model_checkpoint": args.model_checkpoint,
        "v165_original_fcpg_checkpoint": getattr(
            args, "v165_original_fcpg_checkpoint", None
        ),
        "v165_source_checkpoint": getattr(args, "v165_source_checkpoint", None),
        "v165_tuned_fcpg_checkpoint": getattr(
            args, "v165_tuned_fcpg_checkpoint", None
        ),
        "v165_lewm_checkpoint": getattr(args, "v165_lewm_checkpoint", None),
        "v165_actor_checkpoint": getattr(args, "v165_actor_checkpoint", None),
        "v165_contact_authorization_receipt": getattr(
            args, "v165_contact_authorization_receipt", None
        ),
        "v168_original_fcpg_checkpoint": getattr(
            args, "v168_original_fcpg_checkpoint", None
        ),
        "v168_source_checkpoint": getattr(args, "v168_source_checkpoint", None),
        "v168_lewm_checkpoint": getattr(args, "v168_lewm_checkpoint", None),
        "v168_lewm_receipt": getattr(args, "v168_lewm_receipt", None),
        "v168_training_receipt": getattr(args, "v168_training_receipt", None),
        "v168_actor_checkpoint": getattr(args, "v168_actor_checkpoint", None),
        "global_tip_checkpoint": (
            None
            if args.disable_global_tip_locator
            else args.global_tip_checkpoint
        ),
        "projection_calibration": args.calibration,
        "secondary_cache": args.secondary_cache,
        "live_anchor_calibration": args.live_anchor_calibration,
        "home_state_report": args.home_state_report,
        "experiment_manifest": args.experiment_manifest,
        "planner_source": REPO / "fcpg/so101_realtime_planner.py",
        "runtime_source": REPO / "fcpg/so101_realtime_runtime.py",
        "server_source": Path(__file__),
    }
    source_records: dict[str, dict[str, Any] | None] = {}
    for name, raw_path in source_paths.items():
        if raw_path is None:
            source_records[name] = None
            continue
        path = Path(raw_path).expanduser().resolve()
        stat = path.stat()
        file_sha256 = None
        if path.is_file() and name.startswith(("v165_", "v168_")):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            file_sha256 = digest.hexdigest()
        source_records[name] = {
            "path": str(path),
            "kind": "file" if path.is_file() else "directory",
            "byte_count": stat.st_size if path.is_file() else None,
            "sha256": file_sha256,
        }
    payload = {
        "format": EXPERIMENT_SESSION_FORMAT,
        "session_id": binding["attempt_id"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "binding": _jsonable(binding),
        "source_records": source_records,
        "hardware_binding": {
            "robot_id": args.robot_id,
            "robot_port": args.robot_port,
            "wrist_device": args.wrist_device,
            "column_device": args.column_device,
        },
        "runtime": {
            "control_mode": args.control_mode,
            "experiment_method": args.experiment_method,
            "anchor_authority": getattr(args, "anchor_authority", None),
            "anchor_authority_session_immutable": getattr(
                args, "anchor_authority", None
            )
            is not None,
            "v165_no_contact_approach": bool(
                getattr(args, "v165_no_contact_approach", False)
            ),
            "v165_contact_authorization_receipt": (
                None
                if getattr(args, "v165_contact_authorization_receipt", None) is None
                else str(
                    Path(args.v165_contact_authorization_receipt)
                    .expanduser()
                    .resolve()
                )
            ),
            "v168_mode": getattr(args, "v168_mode", None),
            "v168_live_execute": bool(getattr(args, "v168_live_execute", False)),
            "v168_supervised_debug": bool(
                getattr(args, "v168_supervised_debug", False)
            ),
            "v170_mode": getattr(args, "v170_mode", None),
            "v170_live_execute": bool(
                getattr(args, "v170_live_execute", False)
            ),
            "execute_enabled": bool(args.execute_enabled),
            "max_chunks_per_confirmation": (
                int(getattr(args, "max_autonomous_chunks", 1))
                if bool(getattr(args, "bounded_autonomy_enabled", False))
                else 1
            ),
            "max_plan_age_s": args.max_plan_age_s,
            "peg_clearance_m": args.peg_clearance_m,
        },
    }
    with destination.open("x", encoding="ascii") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return destination


def _load_live_anchor_calibration(
    planner: PreparedSO101RealtimePlanner, path: Path
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("view") != "column":
        raise ValueError("unsupported live-anchor calibration contract")
    result = planner.set_live_moving_tip_track(
        "column",
        np.asarray(payload["moving_tip_track_candidates_px"], dtype=np.float32),
        np.asarray(payload["open_state"], dtype=np.float32),
        np.asarray(payload["closed_state"], dtype=np.float32),
    )
    _record_live_anchor_calibration_contract(planner, resolved)
    return result


def _record_live_anchor_calibration_contract(
    planner: Any, path: Path
) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    metadata = {
        "live_anchor_calibration_source": str(resolved),
        "live_anchor_calibration_sha256": hashlib.sha256(
            resolved.read_bytes()
        ).hexdigest(),
    }
    planner.model_contract.update(metadata)
    return metadata


def _save_live_anchor_calibration(
    output: Path,
    calibration: dict[str, Any],
    *,
    planner: Any | None = None,
) -> Path:
    views = calibration.get("views")
    geometry = calibration.get("geometry")
    if not isinstance(views, dict) or not isinstance(geometry, dict):
        raise ValueError("calibration result is missing views or geometry")
    column_view = views.get("column")
    column_geometry = geometry.get("column")
    if not isinstance(column_view, dict) or not isinstance(column_geometry, dict):
        raise ValueError("calibration result is missing column geometry")
    payload = {
        "version": 1,
        "view": "column",
        "moving_tip_track_candidates_px": column_view[
            "moving_tip_track_candidates_px"
        ],
        "open_state": column_view["open_state"],
        "closed_state": column_view["closed_state"],
        "selected_candidate_index": column_geometry["candidate_index"],
        "source_capture": calibration.get("capture"),
    }
    path = output.expanduser().resolve() / "live_anchor_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    if planner is not None:
        _record_live_anchor_calibration_contract(planner, path)
    return path


def _commit_live_anchor_calibration(
    output: Path,
    planner: PreparedSO101RealtimePlanner,
    calibration: dict[str, Any],
) -> dict[str, str]:
    """Durably save one calibration before runtime replans and re-arms."""

    path = _save_live_anchor_calibration(
        output,
        calibration,
        planner=planner,
    )
    return {
        "live_anchor_calibration": str(path),
        "live_anchor_calibration_source": planner.model_contract[
            "live_anchor_calibration_source"
        ],
        "live_anchor_calibration_sha256": planner.model_contract[
            "live_anchor_calibration_sha256"
        ],
    }


def _load_home_state_report(path: Path) -> np.ndarray:
    """Read a task-start state without importing stale image coordinates."""

    report = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if "current_state" not in report:
        raise ValueError("home-state report does not contain current_state")
    state = np.asarray(report["current_state"], dtype=np.float32)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("home-state report has an invalid current_state")
    return state.copy()


def _resolve_camera_device(
    role: str,
    device: str | Path,
    *,
    require_stable_alias: bool = True,
    sysfs_root: Path = Path("/sys/class/video4linux"),
) -> dict[str, str]:
    """Resolve one camera alias and identify its underlying physical device."""

    requested = Path(device).expanduser()
    requested_text = str(requested)
    stable_alias = requested.parent == Path("/dev/v4l/by-id")
    if require_stable_alias and not stable_alias:
        raise ValueError(
            f"{role} camera must use /dev/v4l/by-id, "
            f"not the volatile device name {requested_text!r}"
        )
    if not requested.exists():
        raise FileNotFoundError(
            f"{role} camera alias is missing or disconnected: {requested_text}"
        )

    resolved = requested.resolve(strict=True)
    sysfs_device = sysfs_root / resolved.name / "device"
    physical_device = resolved
    if sysfs_device.exists():
        physical_device = sysfs_device.resolve(strict=True)
        for candidate in (physical_device, *physical_device.parents):
            if (candidate / "idVendor").is_file() and (
                candidate / "idProduct"
            ).is_file():
                physical_device = candidate
                break

    return {
        "role": role,
        "requested_alias": requested_text,
        "resolved_device": str(resolved),
        "physical_device": str(physical_device),
        "binding": "by-id" if "/by-id/" in requested_text else (
            "by-path" if "/by-path/" in requested_text else "resolved-path"
        ),
    }


CAMERA_RECONNECT_IDENTITY_FIELDS = (
    "role",
    "requested_alias",
    "physical_device",
    "binding",
)


def _resolve_reconnect_camera_binding(
    view: str,
    configured_alias: str | Path,
    startup_manifest: dict[str, Any],
    *,
    resolver: Any | None = None,
) -> dict[str, str]:
    """Re-resolve one configured alias and reject any identity drift."""

    if view not in VIEWS:
        raise ValueError(f"unsupported camera role {view!r}")
    if startup_manifest.get("schema") != "so101-camera-role-manifest-v1":
        raise ValueError("camera startup manifest has an unsupported schema")
    cameras = startup_manifest.get("cameras")
    if not isinstance(cameras, dict) or not isinstance(cameras.get(view), dict):
        raise ValueError(f"camera startup manifest lacks role {view!r}")
    expected = cameras[view]
    if not isinstance(configured_alias, (str, Path)):
        raise ValueError(f"camera {view!r} has no configured requested_alias")
    requested_alias = str(Path(configured_alias).expanduser())
    resolve = _resolve_camera_device if resolver is None else resolver
    current = resolve(
        view,
        requested_alias,
        require_stable_alias=True,
    )
    if not isinstance(current, dict):
        raise TypeError("camera resolver returned an invalid binding")
    for field in CAMERA_RECONNECT_IDENTITY_FIELDS:
        if field not in expected or field not in current:
            raise ValueError(f"camera binding lacks identity field {field!r}")
        if expected[field] != current[field]:
            raise ValueError(
                f"camera {view!r} identity changed at {field}: "
                f"startup={expected[field]!r}, current={current[field]!r}"
            )
    return current


def _validate_camera_devices(
    wrist_device: str | Path,
    column_device: str | Path,
    *,
    require_stable_alias: bool = True,
    sysfs_root: Path = Path("/sys/class/video4linux"),
) -> dict[str, Any]:
    """Fail closed when camera roles are missing, volatile, or duplicated."""

    cameras = {
        "wrist": _resolve_camera_device(
            "wrist",
            wrist_device,
            require_stable_alias=require_stable_alias,
            sysfs_root=sysfs_root,
        ),
        "column": _resolve_camera_device(
            "column",
            column_device,
            require_stable_alias=require_stable_alias,
            sysfs_root=sysfs_root,
        ),
    }
    wrist_identity = cameras["wrist"]["physical_device"]
    column_identity = cameras["column"]["physical_device"]
    if wrist_identity == column_identity:
        raise ValueError(
            "wrist and column camera aliases resolve to the same physical "
            f"device: {wrist_identity}"
        )
    return {
        "schema": "so101-camera-role-manifest-v1",
        "cameras": cameras,
    }


def connect_follower_with_retry(
    follower: Any,
    *,
    attempts: int = 3,
    delay_s: float = 2.0,
    sleep: Any = time.sleep,
) -> None:
    """Open the follower after a native crash without assuming the bus is clean.

    Overnight deaths left ``wrist_flex`` / id 6 missing on the first
    ``connect()``.  A short retry with a full disconnect between attempts is
    enough for a clean reload; this does not home or pulse the arm.
    """

    if attempts < 1:
        raise ValueError("connect attempts must be at least 1")
    connect_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            follower.connect(calibrate=False)
            return
        except (ConnectionError, TimeoutError, OSError, RuntimeError) as exc:
            connect_error = exc
            if getattr(follower, "is_connected", False):
                try:
                    follower.disconnect()
                except Exception:
                    pass
            bus = getattr(follower, "bus", None)
            if bus is not None and getattr(bus, "is_connected", False):
                try:
                    bus.disconnect(disable_torque=False)
                except Exception:
                    pass
            if attempt < attempts - 1:
                sleep(delay_s)
    if connect_error is not None:
        raise connect_error


def _invoke_compatible(fn: Any, candidates: Mapping[str, Any]) -> Any:
    """Call ``fn`` with the intersection of ``candidates`` and its signature."""

    import inspect

    signature = inspect.signature(fn)
    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    accepts_var_kw = False
    for name, param in signature.parameters.items():
        if name in {"self", "cls"}:
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_var_kw = True
            continue
        if name in candidates:
            kwargs[name] = candidates[name]
        elif param.default is inspect.Parameter.empty:
            missing.append(name)
    if missing:
        raise TypeError(
            f"{getattr(fn, '__qualname__', fn)} missing required argument(s): "
            + ", ".join(missing)
        )
    if accepts_var_kw:
        for name, value in candidates.items():
            kwargs.setdefault(name, value)
    return fn(**kwargs)


def _import_dicg_shadow_sidecar() -> Any:
    from fcpg.dicg_shadow_sidecar import DICGShadowSidecar

    return DICGShadowSidecar


def _dead_dicg_shadow(reason: str) -> Any:
    """Best-effort stand-in when spawn fails.  Submit is a no-op."""

    payload = {
        "closed": True,
        "worker_alive": False,
        "ready": False,
        "loaded": False,
        "engine_wired": False,
        "load_error": str(reason),
        "shadow_only": True,
        "control_influence": "none",
        "planner_dataflow_influence": "none",
        "process_isolated": True,
        "resource_contention_possible": True,
        "timing_influence_possible": True,
        "heldout_gate_passed": False,
        "live_ready": False,
    }

    class _DeadDicgShadow:
        def submit(self, observation: Any) -> bool:
            del observation
            return False

        def status(self) -> dict[str, Any]:
            return dict(payload)

        def close(self) -> None:
            return None

    return _DeadDicgShadow()


def _submit_dicg_shadow_observation(runner: Any, observation: LiveObservation) -> None:
    """Best-effort live bypass. Return value and errors never reach the caller."""

    if runner is None:
        return
    submit = getattr(runner, "submit", None)
    if not callable(submit):
        return
    try:
        native = observation.native_frames
        observation = LiveObservation(
            timestamp_s=float(observation.timestamp_s),
            state=np.asarray(observation.state, dtype=np.float32).copy(),
            frames={
                view: np.asarray(frame).copy()
                for view, frame in observation.frames.items()
            },
            native_frames=(
                None
                if native is None
                else {
                    view: np.asarray(frame).copy()
                    for view, frame in native.items()
                }
            ),
        )
        submit(observation)
    except Exception:
        return


def _close_dicg_shadow_runner(runner: Any) -> None:
    if runner is None:
        return
    close = getattr(runner, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        return


def _dicg_shadow_http_status(controller: Any) -> dict[str, Any] | None:
    """Read-only HTTP sidecar. Control paths must not call this helper."""

    runner = getattr(getattr(controller, "io", None), "dicg_shadow", None)
    if runner is None:
        return None
    status_fn = getattr(runner, "status", None)
    if not callable(status_fn):
        return None
    try:
        payload = status_fn()
    except Exception:
        return {
            "shadow_only": True,
            "control_influence": "none",
            "heldout_gate_passed": False,
            "error": "status_unavailable",
        }
    if not isinstance(payload, Mapping):
        return {
            "shadow_only": True,
            "control_influence": "none",
            "heldout_gate_passed": False,
        }
    status = dict(payload)
    status.setdefault("shadow_only", True)
    status.setdefault("control_influence", "none")
    status.setdefault("heldout_gate_passed", False)
    return status


def _attach_dicg_shadow_runner(args: argparse.Namespace, io: Any) -> Any:
    checkpoint = getattr(args, "dicg_shadow_checkpoint", None)
    if checkpoint is None:
        return None
    if not isinstance(io, HardwareSO101IO):
        raise RuntimeError(
            "DICG shadow is live-observation-only and cannot attach to replay IO"
        )
    extrinsic = args.dicg_shadow_extrinsic_index
    if isinstance(extrinsic, str):
        extrinsic_index: int | str | None = None
        extrinsic_group_key: int | str | None = extrinsic
    else:
        extrinsic_index = extrinsic
        extrinsic_group_key = None
    try:
        sidecar_cls = _import_dicg_shadow_sidecar()
        runner = _invoke_compatible(
            sidecar_cls,
            {
                "checkpoint_path": args.dicg_shadow_checkpoint,
                "vit_checkpoint_path": args.dicg_shadow_vit_checkpoint,
                "log_path": args.dicg_shadow_log,
                "dataset_contract": args.dicg_shadow_dataset_contract,
                "observer_bundle": args.dicg_shadow_bundle,
                "mujoco_model": args.dicg_shadow_mujoco_model,
                "extrinsic_index": extrinsic_index,
                "extrinsic_group_key": extrinsic_group_key,
                "device": args.device,
                "latency_budget_s": float(args.dicg_shadow_max_latency_ms) / 1000.0,
            },
        )
    except Exception as exc:
        print(
            "[DICG-SHADOW] attach failed "
            f"({type(exc).__name__}: {exc}); control service continues "
            "NON-ACTUATING",
            flush=True,
        )
        runner = _dead_dicg_shadow(f"{type(exc).__name__}: {exc}")
    io.dicg_shadow = runner
    print(
        "[DICG-SHADOW] "
        f"checkpoint={Path(checkpoint).name} "
        f"vit={Path(args.dicg_shadow_vit_checkpoint).name} "
        f"log={args.dicg_shadow_log} "
        f"max_latency_ms={args.dicg_shadow_max_latency_ms} "
        "PROCESS-ISOLATED NON-ACTUATING",
        flush=True,
    )
    return runner


def configured_camera_views(args):
    """Explicit partial acquisition for the V21 cached-Column experiment only.

    Defaults remain dual-view. No automatic substitution on camera failure.
    The caller must supply a truthful view-valid mask for inactive views.
    """
    views = tuple(getattr(args, "active_camera_views", VIEWS))
    if views not in (tuple(VIEWS), ("wrist",)):
        raise ValueError("active camera views must be wrist,column or explicit wrist-only")
    return views


class HardwareSO101IO:
    """Persistent follower connection with bounded per-camera recovery."""

    is_hardware = True

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        gripper_calibration_closed: float,
        gripper_calibration_open: float,
        endpoint_envelope: EndpointEnvelope | None = None,
    ) -> None:
        #: Bounds a stalled endpoint must satisfy to be replanned from rather
        #: than aborted on.  Left ``None`` the settle contract fails closed and
        #: every stall aborts, which is the pre-existing behaviour.
        self.endpoint_envelope = endpoint_envelope
        self.active_camera_views = configured_camera_views(args)
        self.camera_manifest = _validate_camera_devices(
            args.wrist_device,
            args.column_device,
        )
        if self.active_camera_views != tuple(VIEWS):
            self.camera_manifest["active_camera_views"] = list(self.active_camera_views)
        manifest_path = (
            args.output.expanduser().resolve() / "camera_device_manifest.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(self.camera_manifest, indent=2),
            encoding="utf-8",
        )

        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        source_width, source_height = LIVE_SOURCE_IMAGE_SIZE
        cameras = {
            "wrist": OpenCVCameraConfig(
                index_or_path=args.wrist_device,
                fps=60,
                width=source_width,
                height=source_height,
                fourcc="MJPG",
            ),
            "column": OpenCVCameraConfig(
                index_or_path=args.column_device,
                fps=30,
                width=source_width,
                height=source_height,
                fourcc="YUYV",
            ),
        }
        cameras = {name: camera for name, camera in cameras.items() if name in self.active_camera_views}
        config = SO101FollowerConfig(
            port=args.robot_port,
            id=args.robot_id,
            calibration_dir=args.calibration_dir,
            use_degrees=False,
            disable_torque_on_disconnect=False,
            max_relative_target=args.max_command_jump,
            cameras=cameras,
        )
        self.follower = SO101Follower(config)
        self.max_command_jump = float(args.max_command_jump)
        self.settle_timeout_s = float(args.settle_timeout_s)
        self.settle_position_tolerance = float(args.settle_position_tolerance)
        self.settle_step_tolerance = float(args.settle_step_tolerance)
        self.settle_samples = int(args.settle_samples)
        calibration_positions = np.asarray(
            [gripper_calibration_closed, gripper_calibration_open],
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(calibration_positions))
            or np.any(calibration_positions < 0.0)
            or np.any(calibration_positions > 100.0)
            or calibration_positions[1] - calibration_positions[0] < 20.0
        ):
            raise ValueError(
                "gripper calibration positions must be in [0,100] with "
                "at least 20 units of opening span"
            )
        self.gripper_calibration_closed = float(calibration_positions[0])
        self.gripper_calibration_open = float(calibration_positions[1])
        self.self_contact_calibration_enabled = bool(
            getattr(args, "self_contact_calibration", False)
            and getattr(args, "i_accept_empty_jaw_self_contact", False)
        )
        self.calibration_capture_root = (
            args.output.expanduser().resolve() / "gripper_calibration_captures"
        )
        self.command_journal_root = (
            args.output.expanduser().resolve() / "command_journal"
        )
        self.command_journal_root.mkdir(parents=True, exist_ok=True)
        self.last_command_journal: str | None = None
        self.experiment_binding = getattr(args, "experiment_binding", None)
        # The raw resolution every live planning frame is letterboxed FROM.
        # Held on the IO because `_read_camera` is the one place a raw frame
        # exists, and the pad band is a function of this size: a camera that
        # silently came up in another mode would still yield a 224x224 canvas,
        # just with the content rectangle in the wrong place.
        self.source_image_size = (int(source_width), int(source_height))
        self.camera_lock = threading.Lock()
        self.preview_cache_lock = threading.Lock()
        self.preview_cache: dict[str, np.ndarray] = {}
        self.camera_reconnect_counts = {view: 0 for view in VIEWS}
        self.camera_last_errors: dict[str, str | None] = {
            view: None for view in VIEWS
        }
        if not self.follower.calibration_fpath.is_file():
            raise FileNotFoundError(
                f"missing follower calibration {self.follower.calibration_fpath}"
            )
        # COLD-START SAFETY, measured on this rig: raw Goal_Position reads 0 on
        # all six motors after a power cycle while Present_Position is 863..3152
        # ticks.  `connect()` ends in `configure()`, whose `torque_disabled()`
        # context manager re-enables torque in its `finally:` and never writes a
        # goal -- so connecting cold would drive every joint toward tick 0, into
        # the table and the hard stops.  Normalized reads hide this: raw 0 clamps
        # to each joint's calibrated floor, surfacing as [-100..-100, 0], which is
        # exactly SO101_POSITION_LOWER and indistinguishable from a real command.
        #
        # So the goal register is seeded to the measured position FIRST, on the
        # bus alone, and verified by exact raw readback.  `connect()` is still
        # called afterwards rather than replaced, because `configure()` also
        # writes Operating_Mode and the P/I/D coefficients that nothing else in
        # this repo writes; replacing it would silently drop that setup.  The bus
        # is released in between only because `connect()` refuses an already-open
        # bus -- the seed persists, since Goal_Position lives in the servo.
        initializer = getattr(args, "connection_initializer", None)
        if initializer is None:
            self.safe_energize_preseed = preseed_goal_position(self.follower.bus)
            self.follower.bus.disconnect(disable_torque=False)
            connect_follower_with_retry(self.follower)
        else:
            if not callable(initializer):raise TypeError("connection_initializer must be callable")
            self.safe_energize_preseed = initializer(self)
        if not self.follower.is_calibrated:
            self.close()
            raise RuntimeError("follower calibration does not match the connected bus")
        self.pre_motion_identity_required = False
        self.pre_motion_reference: dict[str, Any] | None = None
        self.last_pre_motion_identity: dict[str, Any] | None = None

    def run_pre_motion_identity_handshake(
        self,
        *,
        initial_observation: LiveObservation | None = None,
        n_frames: int = 3,
    ) -> dict[str, Any]:
        """Grab N frames and match setup reference before any Goal write."""

        from fcpg.pre_motion_identity import (
            PRE_MOTION_N_FRAMES,
            evaluate_pre_motion_identity,
            grab_pre_motion_frames,
        )

        needed = int(n_frames or PRE_MOTION_N_FRAMES)
        columns: list[np.ndarray] = []
        wrists: list[np.ndarray] = []
        if initial_observation is not None:
            columns.append(np.asarray(initial_observation.frames["column"]))
            wrists.append(np.asarray(initial_observation.frames["wrist"]))
        remaining = max(0, needed - len(columns))
        if remaining:
            extra = grab_pre_motion_frames(self.observe, n_frames=remaining)
            columns.extend(extra["column"])
            wrists.extend(extra["wrist"])
        reference = getattr(self, "pre_motion_reference", None)
        setup = getattr(self, "pre_motion_setup_manifest", None)
        verdict = evaluate_pre_motion_identity(
            columns,
            wrists,
            reference_column=None if not isinstance(reference, Mapping) else reference.get("column"),
            reference_wrist=None if not isinstance(reference, Mapping) else reference.get("wrist"),
            setup_manifest=setup if isinstance(setup, Mapping) else None,
            n_frames=needed,
        )
        fingerprints = verdict.get("role_fingerprints")
        if bool(verdict.get("pre_motion_identity_valid")) and isinstance(fingerprints, Mapping):
            cameras = self.camera_manifest.setdefault("cameras", {})
            for role, fingerprint in fingerprints.items():
                if isinstance(cameras.get(role), dict) and isinstance(fingerprint, Mapping):
                    cameras[role]["role_fingerprint"] = dict(fingerprint)
        return verdict

    def observe(self) -> LiveObservation:
        state = self._read_state_only()
        with self.camera_lock:
            active = getattr(self, "active_camera_views", VIEWS)
            frames = {view: self._read_camera(view) for view in active}
            raw = getattr(self, "_last_raw_frames", None)
            native = None
            if isinstance(raw, dict) and set(raw) >= set(active):
                native = {view: np.asarray(raw[view]).copy() for view in active}
        observation = LiveObservation(
            timestamp_s=time.perf_counter(),
            state=state,
            frames=frames,
            native_frames=native,
        )
        _submit_dicg_shadow_observation(
            getattr(self, "dicg_shadow", None),
            observation,
        )
        return observation

    def _read_raw_camera_frame(self, view: str) -> np.ndarray:
        """Read one RAW camera frame, recovering once, with no preprocessing.

        Split out of :meth:`_read_camera` so the startup resolution probe and
        the model's own ingestion path read through exactly one code path: a
        probe that reconnected differently from the planning read would be
        checking a geometry the planner never sees.
        """

        camera = self.follower.cameras[view]
        try:
            raw_frame = camera.async_read(timeout_ms=CAMERA_READ_TIMEOUT_MS)
        except (TimeoutError, ConnectionError, RuntimeError) as error:
            raw_frame = self._recover_camera(view, error)
        return np.asarray(raw_frame)

    def observed_source_image_sizes(self) -> dict[str, tuple[int, int]]:
        """Measure each view's RAW ``(width, height)`` before any resize.

        Read from the raw frame, never from the 224x224 planning canvas: both
        geometry protocols produce that same canvas, so its shape cannot
        identify the raw resolution it came from.  The caller checks these
        against the planner's declared ``source_image_size``.
        """

        sizes: dict[str, tuple[int, int]] = {}
        with self.camera_lock:
            for view in VIEWS:
                raw = self._read_raw_camera_frame(view)
                if raw.ndim != 3 or raw.shape[2] != 3:
                    raise ValueError(
                        f"camera {view!r} returned a raw frame of shape "
                        f"{raw.shape}; expected [H, W, 3]"
                    )
                sizes[view] = (int(raw.shape[1]), int(raw.shape[0]))
        return sizes

    def _read_camera(self, view: str) -> np.ndarray:
        """Read one camera while the caller owns ``camera_lock``.

        Live ingestion produces `letterbox_v1` planning frames: an
        aspect-preserving resize onto a black 224x224 canvas, which is the
        geometry every trained component's cache was actually built with.  The
        historical anisotropic `resize_rgb` squash is NOT used here anymore;
        it remains the offline capture geometry for its own callers.
        """

        raw = self._read_raw_camera_frame(view)
        cache = getattr(self, "_last_raw_frames", None)
        if cache is None:
            cache = {}
            self._last_raw_frames = cache
        cache[view] = np.asarray(raw).copy()
        observed = (int(raw.shape[1]), int(raw.shape[0]))
        if observed != self.source_image_size:
            # Fail closed rather than letterboxing an unexpected aspect ratio:
            # the result would still be a 224x224 frame, so nothing downstream
            # could detect that the content rectangle had moved.
            raise RuntimeError(
                f"camera {view!r} delivered a {observed[0]}x{observed[1]} raw "
                f"frame, but live geometry is declared for "
                f"{self.source_image_size[0]}x{self.source_image_size[1]}; "
                "the letterbox padding band is a function of the raw aspect "
                "ratio, so planning pixels would be biased silently"
            )
        frame = letterbox_rgb_image(raw, 224)
        with self.preview_cache_lock:
            self.preview_cache[view] = frame.copy()
        return frame

    @staticmethod
    def _clear_camera_async_state(camera: Any) -> None:
        """Discard frames/events belonging to a released OpenCV handle."""

        event = getattr(camera, "new_frame_event", None)
        if event is not None:
            event.clear()
        if not hasattr(camera, "latest_frame"):
            return
        frame_lock = getattr(camera, "frame_lock", None)
        if frame_lock is None:
            camera.latest_frame = None
            return
        with frame_lock:
            camera.latest_frame = None

    @classmethod
    def _disconnect_camera(cls, camera: Any) -> None:
        """Stop a stale read thread and release its device handle."""

        try:
            camera.disconnect()
        except (ConnectionError, RuntimeError):
            pass
        finally:
            cls._clear_camera_async_state(camera)

    def _recover_camera(self, view: str, cause: Exception) -> np.ndarray:
        """Reconnect one failed camera through its stable configured path."""

        camera = self.follower.cameras[view]
        last_error: Exception = cause
        for attempt in range(1, CAMERA_RECOVERY_ATTEMPTS + 1):
            self._disconnect_camera(camera)
            if CAMERA_RECOVERY_DELAY_S > 0:
                time.sleep(CAMERA_RECOVERY_DELAY_S)
            try:
                _resolve_reconnect_camera_binding(
                    view,
                    getattr(camera, "index_or_path", None),
                    self.camera_manifest,
                )
            except (OSError, TypeError, ValueError, RuntimeError) as error:
                self.camera_last_errors[view] = str(error)
                raise RuntimeError(
                    f"camera {view!r} reconnect identity validation failed: "
                    f"{error}"
                ) from error
            try:
                camera.connect(warmup=False)
                # connect() does not start the asynchronous reader. Clearing here
                # guarantees async_read waits for a frame from the new handle.
                self._clear_camera_async_state(camera)
                frame = camera.async_read(
                    timeout_ms=CAMERA_RECOVERY_TIMEOUT_MS
                )
            except (TimeoutError, ConnectionError, RuntimeError) as error:
                last_error = error
                self._disconnect_camera(camera)
                continue
            self.camera_reconnect_counts[view] += 1
            self.camera_last_errors[view] = None
            return np.asarray(frame)

        self.camera_last_errors[view] = str(last_error)
        raise TimeoutError(
            f"camera {view!r} did not recover after "
            f"{CAMERA_RECOVERY_ATTEMPTS} reconnect attempts; "
            f"last error: {last_error}"
        ) from last_error

    def _cached_preview(self, view: str) -> np.ndarray | None:
        with self.preview_cache_lock:
            cached = self.preview_cache.get(view)
            return None if cached is None else cached.copy()

    def preview_frame(self, view: str) -> np.ndarray:
        """Read a monitor-only frame without advancing task/model state."""

        if view not in VIEWS:
            raise ValueError(f"unsupported preview view: {view}")
        if not self.camera_lock.acquire(blocking=False):
            cached = self._cached_preview(view)
            if cached is not None:
                return cached
            return np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            try:
                return self._read_camera(view)
            except TimeoutError:
                cached = self._cached_preview(view)
                if cached is not None:
                    return cached
                raise
        finally:
            self.camera_lock.release()

    def _read_state_only(self) -> np.ndarray:
        """Read joint positions without polling either camera."""

        positions = self.follower.bus.sync_read(
            "Present_Position", num_retry=STATE_READ_RETRIES
        )
        observation = {
            f"{motor}.pos": value for motor, value in positions.items()
        }
        return observation_state(observation)

    def read_servo_load_registers(self) -> dict[str, Any]:
        """Observability-only load/current sample.  Never used for control."""

        from fcpg.servo_load_telemetry import read_servo_load_registers

        return read_servo_load_registers(self.follower.bus)

    def _write_gripper_position(self, position: float) -> float:
        """Write one already-bounded gripper target without a redundant state read."""

        target = float(position)
        if not np.isfinite(target) or not 0.0 <= target <= 100.0:
            raise ValueError("gripper calibration target must lie in [0,100]")
        self.follower.bus.sync_write(
            "Goal_Position",
            {"gripper": target},
            num_retry=STATE_WRITE_RETRIES,
        )
        return target

    def _read_optional_servo_telemetry(self) -> dict[str, Any]:
        """Best-effort Goal / Velocity / Moving / Load / Current sample.

        Never raises.  Availability and read errors are always recorded.
        Optional registers never define contact.

        ``Present_Load`` / ``Present_Current`` are the raw unsigned Feetech
        words from ``sync_read`` (LeRobot does not decode them).  Safety
        callers must run ``fcpg.servo_register_decode`` before comparing
        magnitudes.  ``Present_Velocity`` is already signed (bit 15).
        """

        keys = (
            "goal",
            "present_velocity",
            "moving",
            "present_load",
            "present_current",
        )
        sample: dict[str, Any] = {
            "goal": None,
            "present_velocity": None,
            "moving": None,
            "present_load": None,
            "present_current": None,
            "availability": {key: False for key in keys},
            "read_errors": {},
        }
        bus = getattr(getattr(self, "follower", None), "bus", None)
        if bus is None or not hasattr(bus, "sync_read"):
            for key in keys:
                sample["read_errors"][key] = "bus_unavailable"
            return sample
        register_map = {
            "goal": "Goal_Position",
            "present_velocity": "Present_Velocity",
            "moving": "Moving",
            "present_load": "Present_Load",
            "present_current": "Present_Current",
        }
        for key, register in register_map.items():
            try:
                values = bus.sync_read(register, num_retry=0)
            except Exception as exc:  # noqa: BLE001 — optional witness
                sample["read_errors"][key] = f"{type(exc).__name__}: {exc}"
                continue
            sample[key] = values
            sample["availability"][key] = True
        return sample

    def _write_position_command(
        self,
        command: np.ndarray,
        previous: np.ndarray,
    ) -> np.ndarray:
        """Write one prevalidated arm command without LeRobot's zero-retry read."""

        target = np.asarray(command, dtype=np.float32)
        reference = np.asarray(previous, dtype=np.float32)
        if target.shape != (6,) or reference.shape != (6,):
            raise ValueError("SO-101 command and reference must be six-vectors")
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(reference)):
            raise ValueError("SO-101 command and reference must be finite")
        if np.any(target < SO101_POSITION_LOWER) or np.any(
            target > SO101_POSITION_UPPER
        ):
            raise RuntimeError("SO-101 command lies outside hard position bounds")
        jump = float(np.max(np.abs(target - reference)))
        if jump > self.max_command_jump + 1e-5:
            raise RuntimeError(
                "SO-101 command exceeds the configured relative limit: "
                f"jump={jump:.3f}, limit={self.max_command_jump:.3f}"
            )
        goals = {
            name.removesuffix(".pos"): float(value)
            for name, value in zip(SO101_JOINT_NAMES, target, strict=True)
        }
        self.follower.bus.sync_write(
            "Goal_Position",
            goals,
            num_retry=STATE_WRITE_RETRIES,
        )
        return target.copy()

    def capture_initial(self, count: int, period_s: float) -> list[LiveObservation]:
        if count <= 0 or period_s <= 0:
            raise ValueError("initial capture settings are invalid")
        started = time.perf_counter()
        rows: list[LiveObservation] = []
        for index in range(count):
            deadline = started + index * period_s
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            rows.append(self.observe())
        return rows

    def execute_chunk(
        self,
        chunk: np.ndarray,
        *,
        period_s: float,
        stop_event: threading.Event,
    ) -> ChunkExecution:
        values = np.asarray(chunk, dtype=np.float32)
        if values.shape != (5, 6) or not np.all(np.isfinite(values)):
            raise ValueError("hardware command chunk must be finite [5,6]")
        journal_path = self._start_command_journal(values)
        self.last_command_journal = (
            None if journal_path is None else str(journal_path)
        )
        sent: list[np.ndarray] = []
        observations: list[LiveObservation] = []
        started = time.perf_counter()

        try:
            previous_command = self._read_state_only()
            self._append_command_journal(
                journal_path,
                {
                    "event": "chunk_started",
                    "monotonic_s": started,
                    "before_state": previous_command,
                },
            )
            for index, command in enumerate(values):
                if stop_event.is_set():
                    raise RuntimeError("execution stopped by operator")
                deadline = started + index * period_s
                delay = deadline - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                self._append_command_journal(
                    journal_path,
                    {
                        "event": "command_intent",
                        "command_index": index,
                        "monotonic_s": time.perf_counter(),
                        "requested": command,
                        "previous_command": previous_command,
                    },
                )
                actual = self._write_position_command(command, previous_command)
                if not np.allclose(actual, command, atol=1e-4, rtol=0.0):
                    raise RuntimeError("LeRobot clipped or changed an interpolated command")
                if np.any(actual < SO101_POSITION_LOWER) or np.any(
                    actual > SO101_POSITION_UPPER
                ):
                    raise RuntimeError("LeRobot returned a command outside hard bounds")
                sent.append(actual)
                previous_command = actual

                # Each command needs its own measured state, timestamp, and raw
                # frames. Repeating one settled endpoint five times makes the
                # execution receipt look denser than the hardware evidence is.
                observation = self.observe()
                observations.append(observation)
                self._append_command_journal(
                    journal_path,
                    {
                        "event": "command_result",
                        "command_index": index,
                        "monotonic_s": time.perf_counter(),
                        "sent": actual,
                        "observation_timestamp_s": observation.timestamp_s,
                        "observed_state": observation.state,
                    },
                )

            settlement = self._settle_commanded_endpoint(
                sent[-1],
                period_s=period_s,
                stop_event=stop_event,
                journal_path=journal_path,
            )

            # Use a fresh settled endpoint for the final command observation so
            # the next plan starts from the pose that actually stopped moving.
            settled_observation = self.observe()
            observations[-1] = settled_observation
            self._append_command_journal(
                journal_path,
                {
                    "event": "chunk_completed",
                    "monotonic_s": time.perf_counter(),
                    "sent_commands": len(sent),
                    "settled_timestamp_s": settled_observation.timestamp_s,
                    "settled_state": settled_observation.state,
                    "endpoint_outcome": settlement.outcome,
                    "endpoint_joint_residual": settlement.joint_residual,
                    "endpoint_tcp_miss_m": settlement.tcp_miss_m,
                },
            )
            return ChunkExecution(
                requested=values,
                sent=np.stack(sent),
                observations=tuple(observations),
                duration_s=time.perf_counter() - started,
                command_journal=(
                    None if journal_path is None else str(journal_path)
                ),
                settlement=settlement,
            )
        except BaseException as exc:
            self._append_command_journal(
                journal_path,
                {
                    "event": "chunk_fault",
                    "monotonic_s": time.perf_counter(),
                    "sent_commands_confirmed": len(sent),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

    def _settle_commanded_endpoint(
        self,
        target: np.ndarray,
        *,
        period_s: float,
        stop_event: threading.Event,
        journal_path: Path | None,
        timeout_prefix: str = "robot did not settle after the five-command chunk",
    ) -> EndpointSettlement:
        """Wait until the arm is quiet on ``target[:5]``, or declare a stall.

        Gripper residual is ignored: a jaw can stall against an object (or a
        mechanical stop) without meaning the arm missed.  Thresholds are the
        instance settle contract already declared on this IO object.
        """

        commanded = np.asarray(target, dtype=np.float32)
        if commanded.shape != (6,) or not np.all(np.isfinite(commanded)):
            raise ValueError("settlement target must be a finite six-vector")
        previous_state = self._read_state_only()
        stable_samples = 0
        quiet_samples = 0
        settle_samples_seen = 0
        settle_deadline = time.perf_counter() + self.settle_timeout_s
        final_position_error = float("inf")
        final_step_delta = float("inf")
        settlement: EndpointSettlement | None = None
        current_state = previous_state
        # One slot more than the stability quorum, so "is the residual still
        # improving?" is measured across exactly the window a successful
        # settle would have needed to accumulate.
        residual_window: deque[float] = deque(maxlen=self.settle_samples + 1)

        while stable_samples < self.settle_samples:
            if stop_event.is_set():
                raise RuntimeError(
                    "execution stopped while waiting for the robot to settle"
                )
            if time.perf_counter() >= settle_deadline:
                self._hold_at_measured_pose(
                    journal_path,
                    reason="settle_timeout",
                )
                raise RuntimeError(
                    f"{timeout_prefix}: "
                    f"target_error={final_position_error:.3f}, "
                    f"step_delta={final_step_delta:.3f}"
                )

            time.sleep(period_s)
            current_state = self._read_state_only()
            settle_samples_seen += 1
            # A grasped object can stall the gripper; arm joints must still
            # reach their target while the full state stops changing.
            final_position_error = float(
                np.max(np.abs(current_state[:5] - commanded[:5]))
            )
            final_step_delta = float(
                np.max(np.abs(current_state - previous_state))
            )
            residual_window.append(final_position_error)
            # `settle_step_tolerance` is the operator's declared per-sample
            # change that is not real motion, so it -- and not a new
            # constant -- decides whether the arm is still moving.
            motion_stopped = final_step_delta <= self.settle_step_tolerance
            quiet_samples = quiet_samples + 1 if motion_stopped else 0
            if (
                final_position_error <= self.settle_position_tolerance
                and motion_stopped
            ):
                stable_samples += 1
            else:
                stable_samples = 0
            previous_state = current_state
            # A stall implies the residual is outside tolerance, so a
            # partially accumulated settle can never be pre-empted by one.
            if stable_samples == 0 and self._endpoint_stalled(
                residual_window=residual_window,
                quiet_samples=quiet_samples,
                joint_residual=final_position_error,
            ):
                settlement = self._resolve_stalled_endpoint(
                    journal_path,
                    commanded=commanded,
                    actual_state=current_state,
                    joint_residual=final_position_error,
                    step_delta=final_step_delta,
                    residual_improvement=float(
                        residual_window[0] - residual_window[-1]
                    ),
                    quiet_samples=quiet_samples,
                    samples=settle_samples_seen,
                )
                break

        if settlement is None:
            settlement = EndpointSettlement(
                outcome=ENDPOINT_SETTLED,
                commanded_state=commanded,
                actual_state=current_state,
                joint_residual=final_position_error,
                step_delta=final_step_delta,
                residual_improvement=(
                    0.0
                    if len(residual_window) < 2
                    else max(0.0, float(residual_window[0] - residual_window[-1]))
                ),
                quiet_samples=quiet_samples,
                samples=max(1, settle_samples_seen),
            )
        return settlement

    def _start_command_journal(self, requested: np.ndarray) -> Path | None:
        root = getattr(self, "command_journal_root", None)
        if root is None:
            return None
        directory = Path(root) / f"chunk_{time.time_ns()}"
        directory.mkdir(parents=False, exist_ok=False)
        path = directory / "events.jsonl"
        self._append_command_journal(
            path,
            {
                "event": "chunk_intent",
                "recorded_wall_time_ns": time.time_ns(),
                "experiment_binding": getattr(self, "experiment_binding", None),
                "requested": requested,
                "requested_commands": len(requested),
            },
        )
        return path

    @staticmethod
    def _append_command_journal(
        path: Path | None,
        record: dict[str, Any],
    ) -> None:
        if path is None:
            return
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _endpoint_stalled(
        self,
        *,
        residual_window: Sequence[float],
        quiet_samples: int,
        joint_residual: float,
    ) -> bool:
        """Decide whether motion provably stopped short of the commanded endpoint.

        Both halves of the test reuse thresholds the settle contract already
        declares, so no new constant gets to decide when the arm is finished:

        * Motion stopped -- ``settle_step_tolerance`` already defines the
          per-sample change that is not real motion, and it must hold for
          ``settle_samples`` consecutive samples: the same quorum a successful
          settle needs.  A stall is therefore never declared on weaker evidence
          than a success, and one quiet sample during a slow start cannot pass.
        * Residual no longer improving -- across that same window the residual
          must not have fallen by more than one ``settle_step_tolerance``.  A
          joint that closed more than one indistinguishable-from-noise step is
          still converging and keeps the remaining timeout.

        The commanded target is still *missed* here; this only distinguishes a
        physically finished pose from "we gave up waiting".
        """

        if joint_residual <= self.settle_position_tolerance:
            return False
        if quiet_samples < self.settle_samples:
            return False
        if len(residual_window) < self.settle_samples + 1:
            return False
        improvement = float(residual_window[0] - residual_window[-1])
        return improvement <= self.settle_step_tolerance

    def _resolve_stalled_endpoint(
        self,
        journal_path: Path | None,
        *,
        commanded: np.ndarray,
        actual_state: np.ndarray,
        joint_residual: float,
        step_delta: float,
        residual_improvement: float,
        quiet_samples: int,
        samples: int,
    ) -> EndpointSettlement:
        """Accept a stalled endpoint as measured, or hold and abort.

        Accepting never means the commanded endpoint was reached: the returned
        settlement carries the measured pose so the next round replans from
        where the arm actually is.
        """

        envelope = getattr(self, "endpoint_envelope", None)
        if envelope is None:
            # Fail closed.  Without an envelope there is nothing to judge the
            # miss against, so it cannot be declared survivable.
            violations: tuple[str, ...] = ("no_endpoint_envelope",)
            tcp_miss_m: float | None = None
        else:
            violations, tcp_miss_m = envelope.evaluate(
                commanded_state=commanded,
                actual_state=actual_state,
            )
        evidence: dict[str, Any] = {
            "commanded_state": commanded,
            "actual_state": actual_state,
            "joint_residual": joint_residual,
            "step_delta": step_delta,
            "residual_improvement": residual_improvement,
            "quiet_samples": quiet_samples,
            "settle_samples_seen": samples,
            "tcp_miss_m": tcp_miss_m,
            "envelope_violations": list(violations),
        }
        if violations:
            self._append_command_journal(
                journal_path,
                {
                    "event": "endpoint_miss_outside_envelope",
                    "monotonic_s": time.perf_counter(),
                    **evidence,
                },
            )
            self._hold_at_measured_pose(
                journal_path,
                reason="endpoint_miss_outside_envelope",
            )
            raise RuntimeError(
                "settled endpoint lies outside the safety envelope: "
                f"joint_residual={joint_residual:.3f}, "
                f"tcp_miss_m={'n/a' if tcp_miss_m is None else f'{tcp_miss_m:.4f}'}, "
                f"violations={','.join(violations)}"
            )
        self._append_command_journal(
            journal_path,
            {
                "event": "endpoint_miss_with_actual_state",
                "monotonic_s": time.perf_counter(),
                **evidence,
            },
        )
        return EndpointSettlement(
            outcome=ENDPOINT_MISS_WITH_ACTUAL_STATE,
            commanded_state=commanded,
            actual_state=actual_state,
            joint_residual=joint_residual,
            step_delta=step_delta,
            residual_improvement=max(0.0, residual_improvement),
            quiet_samples=quiet_samples,
            samples=samples,
            tcp_miss_m=tcp_miss_m,
        )

    def _hold_at_measured_pose(
        self,
        journal_path: Path | None,
        *,
        reason: str,
        actuator_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Clamp the servos onto the pose they are actually in, and prove it landed.

        Uses the same reliability contract as the production command write --
        ``num_retry=STATE_WRITE_RETRIES`` -- because a hold that is silently
        dropped leaves the arm still chasing the goal this hold exists to
        cancel, which is the one state an abort must never leave behind.

        Torque is deliberately left enabled: cutting it on a loaded arm drops
        the payload and the arm with it, so an unverifiable hold is escalated by
        raising rather than by de-energizing.
        """

        # A normalized float32 round trip can truncate one encoder cell. HOLD
        # must copy the decoded Present cells without using that inverse codec.
        bus = self.follower.bus
        motor_names = tuple(SO101_MUJOCO_JOINT_NAMES)

        def raw_position(register: str) -> dict[str, int]:
            values = bus.sync_read(
                register, normalize=False, num_retry=STATE_READ_RETRIES,
            )
            if not isinstance(values, Mapping) or set(values) != set(motor_names):
                raise ValueError(f"{register} raw read must contain exactly six motors")
            # STS3215 position registers use bit-15 sign-magnitude encoding.
            # This is a packet validity limit, not a calibrated workspace limit.
            if any(
                isinstance(values[name], (bool, np.bool_))
                or not isinstance(values[name], (int, float, np.integer, np.floating))
                or not -32767 <= values[name] <= 32767
                or not np.isfinite(values[name])
                or float(values[name]) != int(values[name])
                for name in motor_names
            ):
                raise ValueError(f"{register} raw read must contain six representable finite integer cells")
            return {name: int(values[name]) for name in motor_names}

        def normalized_position(raw: dict[str, int]) -> dict[str, float]:
            ids = {name: bus.motors[name].id for name in motor_names}
            values = bus._normalize({ids[name]: raw[name] for name in motor_names})
            normalized = {name: float(values[ids[name]]) for name in motor_names}
            if not np.isfinite(list(normalized.values())).all():
                raise ValueError("raw HOLD calibration produced nonfinite positions")
            return normalized

        attempts: list[dict[str, Any]] = []
        mismatched: dict[str, list[int]] = {}
        for attempt in range(SAFE_HOLD_VERIFY_ATTEMPTS):
            # Re-read on every attempt: holding a stale pose would itself be a
            # motion command if the arm crept between the write and the check.
            sample_started_s = time.monotonic()
            goal_raw = raw_position("Present_Position")
            sample_returned_s = time.monotonic()
            pose_error: Exception | None = None
            goal = None
            measured = None
            try:
                goal = normalized_position(goal_raw)
                measured = np.asarray([goal[name] for name in motor_names], dtype=np.float64)
            except Exception as exc:
                # Valid measured raw cells still cancel the old target even
                # if calibration metadata cannot describe the pose safely.
                pose_error = exc
            # Optional V21 telemetry stays in memory until physical handling
            # finishes. It never delays a freeze with filesystem work.
            event = dict(event="goal_write", role="emergency_freeze", reason=reason,
                         attempt=attempt, register="Goal_Position", normalize=False,
                         requested_position=goal, requested_goal_raw=dict(goal_raw),
                         present_read_started_s=sample_started_s,
                         present_read_returned_s=sample_returned_s,
                         position_source="same raw Present_Position sample",
                         goal_register_verified=False, pose_validated=False,
                         write_started_s=time.monotonic())
            if actuator_events is not None:
                actuator_events.append(event)
            try:
                bus.sync_write(
                    "Goal_Position", goal_raw, normalize=False, num_retry=STATE_WRITE_RETRIES,
                )
                event["write_returned_s"] = time.monotonic()
                event["verification_started_s"] = time.monotonic()
                readback_raw = raw_position("Goal_Position")
                event["verification_returned_s"] = time.monotonic()
                event["goal_readback_raw"] = dict(readback_raw)
            except BaseException as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
                event["failed_s"] = time.monotonic()
                raise
            mismatched = {
                name: [goal_raw[name], readback_raw[name]]
                for name in motor_names if readback_raw[name] != goal_raw[name]
            }
            event["mismatched_joints"] = mismatched
            event["goal_register_verified"] = not mismatched
            readback = None
            try:
                readback = normalized_position(readback_raw)
            except Exception as exc:
                pose_error = pose_error or exc
            event["goal_readback"] = readback
            attempt_record = dict(attempt=attempt, held_state=measured,
                requested_goal_raw=dict(goal_raw), goal_readback_raw=dict(readback_raw),
                goal_readback=readback, mismatched_joints=mismatched,
                goal_register_verified=not mismatched, pose_validated=False)
            attempts.append(attempt_record)
            if not mismatched:
                # A workspace violation must not leave the old motion goal
                # active. Check this SAME sample after freezing, then preserve
                # the fault; clamped normalization is never proof of safety.
                try:
                    if pose_error is not None:
                        raise pose_error
                    for name in motor_names:
                        cal = bus.calibration[name]
                        if not cal.range_min <= goal_raw[name] <= cal.range_max or cal.range_min >= cal.range_max:
                            raise ValueError(f"raw HOLD {name} is outside its valid calibration range")
                    ids = {name: bus.motors[name].id for name in motor_names}
                    reconstructed = bus._unnormalize({ids[name]: goal[name] for name in motor_names})
                    if any(abs(reconstructed[ids[name]] - goal_raw[name]) > 1 for name in motor_names):
                        raise ValueError("raw HOLD calibration normalization hides the measured encoder cells")
                    if np.any(measured < SO101_POSITION_LOWER) or np.any(measured > SO101_POSITION_UPPER):
                        raise ValueError("raw HOLD calibration position is outside joint bounds")
                    gate = getattr(self, "clearance_gate", None)
                    if gate is not None:
                        gate.require_readback(measured, label="raw HOLD measured pose")
                    event["pose_validated"] = attempt_record["pose_validated"] = True
                except Exception as exc:
                    event.update(error=f"{type(exc).__name__}: {exc}", failed_s=time.monotonic())
                    self._append_command_journal(journal_path, {
                        "event": "safe_hold_pose_rejected", "reason": reason,
                        "monotonic_s": time.perf_counter(), "attempts": attempts,
                        "goal_register_verified": True, "error": event["error"],
                    })
                    raise
                record = {
                    "event": "safe_hold_verified",
                    "monotonic_s": time.perf_counter(),
                    "reason": reason,
                    "attempts": attempts,
                }
                self._append_command_journal(journal_path, record)
                return record
        self._append_command_journal(
            journal_path,
            {
                "event": "safe_hold_unverified",
                "monotonic_s": time.perf_counter(),
                "reason": reason,
                "attempts": attempts,
            },
        )
        raise SafeHoldVerificationError(
            f"safe hold ({reason}) was not confirmed by Goal_Position read-back "
            f"after {SAFE_HOLD_VERIFY_ATTEMPTS} attempts; mismatched joints "
            f"(requested raw, read-back raw): {mismatched}"
        )

    def _trace_final_waypoint_write(self, calibrated_goal: np.ndarray) -> dict[str, Any]:
        """Trace final waypoint Goal_Position write with immediate verification."""
        bus = self.follower.bus

        # Compute expected raw goal via _unnormalize
        goal_dict = {
            name: float(value)
            for name, value in zip(SO101_MUJOCO_JOINT_NAMES, calibrated_goal, strict=True)
        }
        ids_values = {bus.motors[name].id: value for name, value in goal_dict.items()}
        raw_ids = bus._unnormalize(ids_values)
        id_to_name = {bus.motors[name].id: name for name in SO101_MUJOCO_JOINT_NAMES}
        expected_raw = {id_to_name[id_]: int(raw_ids[id_]) for id_ in raw_ids}

        t_write = time.perf_counter()

        # Read back raw Goal_Position
        actual_raw_goal = bus.sync_read("Goal_Position", normalize=False, num_retry=STATE_READ_RETRIES)
        actual_raw_goal = {name: int(actual_raw_goal[name]) for name in SO101_MUJOCO_JOINT_NAMES}

        # Read raw Present_Position
        present_raw = bus.sync_read("Present_Position", normalize=False, num_retry=STATE_READ_RETRIES)
        present_raw = {name: int(present_raw[name]) for name in SO101_MUJOCO_JOINT_NAMES}

        # Check matches
        matches = {name: (expected_raw[name] == actual_raw_goal[name]) for name in SO101_MUJOCO_JOINT_NAMES}

        # Test calibration roundtrip
        roundtrip_calibrated_ids = bus._normalize(raw_ids)
        roundtrip = {}
        for name in SO101_MUJOCO_JOINT_NAMES:
            original_deg = goal_dict[name]
            raw_ticks = expected_raw[name]
            roundtrip_deg = float(roundtrip_calibrated_ids[bus.motors[name].id])
            roundtrip[name] = {
                "original_deg": original_deg,
                "expected_raw_ticks": raw_ticks,
                "roundtrip_deg": roundtrip_deg,
                "error_deg": roundtrip_deg - original_deg,
            }

        return {
            "t_write": t_write,
            "desired_calibrated_goal": goal_dict,
            "expected_raw_goal": expected_raw,
            "actual_raw_goal": actual_raw_goal,
            "goal_write_match": matches,
            "all_match": all(matches.values()),
            "present_raw": present_raw,
            "calibration_roundtrip": roundtrip,
        }

    def _calibrate_raw_locally(self, raw_by_name: dict[str, int]) -> dict[str, float]:
        """Apply the bus normalisation math without spending a serial round-trip."""
        bus = self.follower.bus
        raw_ids = {bus.motors[name].id: raw_by_name[name] for name in raw_by_name}
        calibrated_ids = bus._normalize(raw_ids)
        return {
            name: float(calibrated_ids[bus.motors[name].id]) for name in raw_by_name
        }

    def _trace_settle_sample(self, target_calibrated: np.ndarray) -> dict[str, Any]:
        """Record one settle polling sample.

        Budgeted to four serial transactions so the trace does not stretch the
        settle loop it is measuring; calibrated values are derived in software
        from the raw ticks instead of costing two more reads.
        """
        bus = self.follower.bus
        t = time.perf_counter()

        goal_raw = bus.sync_read("Goal_Position", normalize=False, num_retry=STATE_READ_RETRIES)
        goal_raw = {name: int(goal_raw[name]) for name in SO101_MUJOCO_JOINT_NAMES}

        present_raw = bus.sync_read("Present_Position", normalize=False, num_retry=STATE_READ_RETRIES)
        present_raw = {name: int(present_raw[name]) for name in SO101_MUJOCO_JOINT_NAMES}

        goal_calibrated = self._calibrate_raw_locally(goal_raw)
        present_calibrated = self._calibrate_raw_locally(present_raw)

        servo_state: dict[str, Any] = {}
        for register in ("Present_Velocity", "Moving"):
            try:
                values = bus.sync_read(register, num_retry=STATE_READ_RETRIES)
                servo_state[register] = {
                    name: int(values[name]) for name in SO101_MUJOCO_JOINT_NAMES
                }
            except Exception as exc:
                servo_state[register] = f"UNSUPPORTED ({type(exc).__name__})"

        target_dict = {
            name: float(target_calibrated[i])
            for i, name in enumerate(SO101_MUJOCO_JOINT_NAMES)
        }
        residual_deg = {
            name: present_calibrated[name] - target_dict[name]
            for name in SO101_MUJOCO_JOINT_NAMES
        }
        residual_ticks = {
            name: present_raw[name] - goal_raw[name]
            for name in SO101_MUJOCO_JOINT_NAMES
        }

        return {
            "t": t,
            "goal_raw": goal_raw,
            "present_raw": present_raw,
            "goal_calibrated": goal_calibrated,
            "present_calibrated": present_calibrated,
            "target_calibrated": target_dict,
            "residual_deg": residual_deg,
            "residual_ticks": residual_ticks,
            "servo_state": servo_state,
            "sample_duration_ms": (time.perf_counter() - t) * 1000.0,
        }

    def _trace_safe_hold_execution(self) -> dict[str, Any]:
        """Trace SAFE HOLD execution with before/after comparison."""
        bus = self.follower.bus
        t_before = time.perf_counter()

        # BEFORE HOLD
        goal_raw_before = bus.sync_read("Goal_Position", normalize=False, num_retry=STATE_READ_RETRIES)
        goal_raw_before = {name: int(goal_raw_before[name]) for name in SO101_MUJOCO_JOINT_NAMES}

        present_raw_before = bus.sync_read("Present_Position", normalize=False, num_retry=STATE_READ_RETRIES)
        present_raw_before = {name: int(present_raw_before[name]) for name in SO101_MUJOCO_JOINT_NAMES}

        # Fresh read for hold
        t_fresh = time.perf_counter()
        present_raw_fresh = bus.sync_read("Present_Position", normalize=False, num_retry=STATE_READ_RETRIES)
        present_raw_fresh = {name: int(present_raw_fresh[name]) for name in SO101_MUJOCO_JOINT_NAMES}

        present_calibrated_fresh = bus.sync_read("Present_Position", normalize=True, num_retry=STATE_READ_RETRIES)
        present_calibrated_fresh = {name: float(present_calibrated_fresh[name]) for name in SO101_MUJOCO_JOINT_NAMES}

        # Compute expected hold goal via safe hold roundtrip
        raw_ids = {bus.motors[name].id: present_raw_fresh[name] for name in SO101_MUJOCO_JOINT_NAMES}
        calibrated_ids = bus._normalize(raw_ids)
        hold_raw_ids = bus._unnormalize(calibrated_ids)
        id_to_name = {bus.motors[name].id: name for name in SO101_MUJOCO_JOINT_NAMES}
        expected_hold_raw = {id_to_name[id_]: int(hold_raw_ids[id_]) for id_ in hold_raw_ids}

        safe_hold_roundtrip = {}
        for name in SO101_MUJOCO_JOINT_NAMES:
            safe_hold_roundtrip[name] = {
                "original_present_raw": present_raw_fresh[name],
                "calibrated": float(calibrated_ids[bus.motors[name].id]),
                "hold_goal_raw": expected_hold_raw[name],
                "error_ticks": expected_hold_raw[name] - present_raw_fresh[name],
            }

        # NOTE: The actual SAFE HOLD write happens in the calling code
        # We just compute what it SHOULD write

        return {
            "t_before": t_before,
            "goal_raw_before": goal_raw_before,
            "present_raw_before": present_raw_before,
            "t_fresh_read": t_fresh,
            "present_raw_fresh": present_raw_fresh,
            "present_calibrated_fresh": present_calibrated_fresh,
            "safe_hold_roundtrip": safe_hold_roundtrip,
        }

    def _save_diagnostic_trace(self, trace: dict[str, Any], journal_path: Path | None) -> None:
        """Save diagnostic trace to a JSON file."""
        if journal_path is None:
            output_path = self.command_journal_root / f"diagnostic_trace_{time.time_ns()}.json"
        else:
            output_path = journal_path.parent / "diagnostic_servo_trace.json"

        output_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
        print(f"[DIAGNOSTIC] Trace saved to {output_path}")

    def _ramp_gripper_to(
        self,
        target: float,
        *,
        stop_event: threading.Event,
        pinned_arm: np.ndarray,
        phase_name: str,
    ) -> tuple[list[float], list[np.ndarray], float]:
        """Step only the gripper toward ``target`` with the calibration pulse limits.

        Arm Goal is not written here.  The caller must already have pinned it
        to Present via ``_hold_at_measured_pose``.  Drift against ``pinned_arm``
        still aborts, using the same limit ``calibrate_gripper`` used.
        """

        period_s = 1.0 / GRIPPER_PULSE_SAMPLE_HZ
        commands: list[float] = []
        states: list[np.ndarray] = []
        stable = 0
        maximum_observed_arm_drift = 0.0
        for _ in range(GRIPPER_PULSE_MAXIMUM_PHASE_FRAMES):
            started = time.perf_counter()
            if stop_event.is_set():
                raise PlanRejectedError(
                    "in-place gripper open stopped by operator"
                )
            state = self._read_state_only()
            drift = float(np.max(np.abs(state[:5] - pinned_arm)))
            maximum_observed_arm_drift = max(maximum_observed_arm_drift, drift)
            if drift > GRIPPER_PULSE_MAXIMUM_ARM_DRIFT:
                raise PlanRejectedError(
                    "arm moved during in-place gripper open: "
                    f"{drift:.3f} > {GRIPPER_PULSE_MAXIMUM_ARM_DRIFT:.3f}"
                )
            states.append(state.copy())
            command = next_gripper_position(
                float(state[5]), target, GRIPPER_PULSE_MAXIMUM_STEP
            )
            sent = self._write_gripper_position(command)
            commands.append(sent)
            # Keep-open must land on the calibrated stop, not the 2.0 pulse
            # band: that band includes values below C1e's 30.0 floor.
            # settle_step_tolerance is the IO's already-declared "not real
            # motion" quantum (production 0.25).
            if abs(float(state[5]) - target) <= self.settle_step_tolerance:
                stable += 1
            else:
                stable = 0
            if stable >= GRIPPER_PULSE_STABLE_REQUIRED:
                break
            delay = period_s - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)
        else:
            phase_gripper = np.asarray(
                [row[5] for row in states], dtype=np.float32
            )
            raise PlanRejectedError(
                "gripper did not settle during in-place open "
                f"{phase_name}: target={target:.3f}, "
                f"start={phase_gripper[0]:.3f}, "
                f"final={phase_gripper[-1]:.3f}, "
                f"observed_range=[{np.min(phase_gripper):.3f},"
                f"{np.max(phase_gripper):.3f}], "
                f"last_command={commands[-1]:.3f}"
            )
        return commands, states, maximum_observed_arm_drift

    def open_gripper_in_place(
        self,
        *,
        stop_event: threading.Event,
    ) -> dict[str, Any]:
        """Open the jaws to ``gripper_calibration_open`` without moving the arm.

        Arm Goal is rewritten to the measured Present, then only the gripper
        register is ramped.  Stall-aware settle is the same loop the five-
        command chunk uses.  There is no restore leg.
        """

        before = self._read_state_only()
        pinned_arm = before[:5].copy()
        commanded_aperture = float(self.gripper_calibration_open)
        commanded = np.concatenate(
            [pinned_arm, np.asarray([commanded_aperture], dtype=np.float32)]
        ).astype(np.float32)
        journal_path = self._start_command_journal(commanded[None])
        hold = self._hold_at_measured_pose(
            journal_path,
            reason="open_gripper_in_place_pin",
        )
        pinned_arm = np.asarray(
            hold["attempts"][-1]["held_state"], dtype=np.float32
        )[:5].copy()
        commanded[0:5] = pinned_arm
        try:
            commands, states, maximum_observed_arm_drift = self._ramp_gripper_to(
                commanded_aperture,
                stop_event=stop_event,
                pinned_arm=pinned_arm,
                phase_name="open",
            )
            try:
                settlement = self._settle_commanded_endpoint(
                    commanded,
                    period_s=1.0 / GRIPPER_PULSE_SAMPLE_HZ,
                    stop_event=stop_event,
                    journal_path=journal_path,
                    timeout_prefix=(
                        "robot did not settle after in-place gripper open"
                    ),
                )
            except PlanRejectedError:
                raise
            except RuntimeError as exc:
                raise PlanRejectedError(str(exc)) from exc
        except BaseException:
            self._hold_at_measured_pose(
                journal_path,
                reason="open_gripper_in_place_abort",
            )
            raise
        self._hold_at_measured_pose(
            journal_path,
            reason="open_gripper_in_place_landed",
        )
        after = self._read_state_only()
        opened_aperture = float(after[5])
        # One-sided floor: more-open than commanded is fine.  The 2.0
        # position band would accept 29.7 and still fail C1e (>= 30).
        # settle_step_tolerance is the existing "not real motion" quantum.
        minimum_aperture = commanded_aperture - self.settle_step_tolerance
        if opened_aperture < minimum_aperture:
            raise PlanRejectedError(
                "in-place gripper open missed the calibrated stop: "
                f"actual={opened_aperture:.3f}, "
                f"target={commanded_aperture:.3f}, "
                f"minimum={minimum_aperture:.3f}"
            )
        return {
            "kind": "open_gripper_in_place",
            "before_state": before,
            "after_state": after,
            "opened_aperture": opened_aperture,
            "pinned_arm": pinned_arm,
            "commanded_aperture": commanded_aperture,
            "commanded_state": commanded,
            "gripper_commands": np.asarray(commands, dtype=np.float32),
            "maximum_observed_arm_drift": maximum_observed_arm_drift,
            "ramp_states": np.stack(states).astype(np.float32) if states else None,
            "settlement_outcome": settlement.outcome,
            "settlement_joint_residual": settlement.joint_residual,
            "settlement_actual_state": settlement.actual_state,
            "command_journal": None if journal_path is None else str(journal_path),
        }

    def fcpg_identity_probe(
        self,
        *,
        stop_event: threading.Event,
    ) -> dict[str, Any]:
        """Move only the gripper exactly ten units, observe, then restore.

        The arm is pinned to fresh Present_Position before either leg.  The
        preferred direction is toward the calibrated open stop; if less than
        ten units remain there, the closed direction is used.  No arm-joint
        Goal is sent after the initial safe hold.
        """

        before_observation = self.observe()
        before = np.asarray(before_observation.state, dtype=np.float32)
        initial_aperture = float(before[5])
        open_room = float(self.gripper_calibration_open - initial_aperture)
        closed_room = float(initial_aperture - self.gripper_calibration_closed)
        if open_room >= 10.0:
            target_aperture = initial_aperture + 10.0
            direction = "toward_calibrated_open"
        elif closed_room >= 10.0:
            target_aperture = initial_aperture - 10.0
            direction = "toward_calibrated_closed"
        else:
            raise PlanRejectedError(
                "identity probe has no calibrated ten-unit gripper corridor"
            )
        if abs(target_aperture - initial_aperture) != 10.0:
            raise AssertionError("identity probe displacement drifted from ten units")
        pinned_arm = before[:5].copy()
        target_state = np.concatenate(
            [pinned_arm, np.asarray([target_aperture], dtype=np.float32)]
        ).astype(np.float32)
        restored_state = np.concatenate(
            [pinned_arm, np.asarray([initial_aperture], dtype=np.float32)]
        ).astype(np.float32)
        journal_path = self._start_command_journal(
            np.stack((target_state, restored_state))
        )
        hold = self._hold_at_measured_pose(
            journal_path,
            reason="fcpg_identity_probe_pin",
        )
        pinned_arm = np.asarray(
            hold["attempts"][-1]["held_state"], dtype=np.float32
        )[:5].copy()
        target_state[:5] = pinned_arm
        restored_state[:5] = pinned_arm
        maximum_drift = 0.0
        moved_observation = None
        try:
            outward_commands, outward_states, outward_drift = self._ramp_gripper_to(
                target_aperture,
                stop_event=stop_event,
                pinned_arm=pinned_arm,
                phase_name="fcpg_identity_probe_outward",
            )
            maximum_drift = max(maximum_drift, outward_drift)
            moved_observation = self.observe()
            return_commands, return_states, return_drift = self._ramp_gripper_to(
                initial_aperture,
                stop_event=stop_event,
                pinned_arm=pinned_arm,
                phase_name="fcpg_identity_probe_restore",
            )
            maximum_drift = max(maximum_drift, return_drift)
            try:
                settlement = self._settle_commanded_endpoint(
                    restored_state,
                    period_s=1.0 / GRIPPER_PULSE_SAMPLE_HZ,
                    stop_event=stop_event,
                    journal_path=journal_path,
                    timeout_prefix="robot did not settle after FCPG identity probe",
                )
            except PlanRejectedError:
                raise
            except RuntimeError as exc:
                raise PlanRejectedError(str(exc)) from exc
        except BaseException:
            self._hold_at_measured_pose(
                journal_path,
                reason="fcpg_identity_probe_abort",
            )
            raise
        self._hold_at_measured_pose(
            journal_path,
            reason="fcpg_identity_probe_restored",
        )
        restored_observation = self.observe()
        after = np.asarray(restored_observation.state, dtype=np.float32)
        arm_drift = float(np.max(np.abs(after[:5] - pinned_arm)))
        maximum_drift = max(maximum_drift, arm_drift)
        if maximum_drift > GRIPPER_PULSE_MAXIMUM_ARM_DRIFT:
            raise PlanRejectedError(
                "arm moved during FCPG identity probe: "
                f"{maximum_drift:.3f} > {GRIPPER_PULSE_MAXIMUM_ARM_DRIFT:.3f}"
            )
        restore_error = abs(float(after[5]) - initial_aperture)
        if restore_error > self.settle_step_tolerance:
            raise PlanRejectedError(
                "FCPG identity probe failed to restore initial aperture: "
                f"error={restore_error:.3f}"
            )
        if moved_observation is None:
            raise AssertionError("identity probe did not capture its moved endpoint")
        return {
            "kind": "fcpg_identity_probe",
            "direction": direction,
            "before_state": before,
            "moved_state": np.asarray(moved_observation.state, dtype=np.float32),
            "restored_state": after,
            "pinned_arm": pinned_arm,
            "initial_aperture": initial_aperture,
            "target_aperture": target_aperture,
            "displacement_units": abs(target_aperture - initial_aperture),
            "outward_commands": np.asarray(outward_commands, dtype=np.float32),
            "return_commands": np.asarray(return_commands, dtype=np.float32),
            "outward_states": np.stack(outward_states).astype(np.float32),
            "return_states": np.stack(return_states).astype(np.float32),
            "maximum_observed_arm_drift": maximum_drift,
            "settlement_outcome": settlement.outcome,
            "settlement_joint_residual": settlement.joint_residual,
            "command_journal": None if journal_path is None else str(journal_path),
            "before_frames": before_observation.frames,
            "moved_frames": moved_observation.frames,
            "restored_frames": restored_observation.frames,
            "restored_observation": restored_observation,
        }

    def calibrate_gripper(
        self,
        *,
        stop_event: threading.Event,
        self_contact: bool | None = None,
        dual_view_motion_roi: bool = False,
        active_initializer_only: bool = False,
    ) -> dict[str, Any]:
        """Run a pulse and restore its starting aperture after any failure."""

        initial = self.observe()
        try:
            return self._calibrate_gripper_impl(
                stop_event=stop_event,
                self_contact=self_contact,
                dual_view_motion_roi=dual_view_motion_roi,
                active_initializer_only=active_initializer_only,
            )
        except BaseException as pulse_error:
            # An explicit stop means stop now. For every other failure, leaving
            # the jaws at the failed close/open endpoint is itself an unsafe
            # side effect; return only the gripper to its pre-pulse aperture.
            if stop_event.is_set():
                raise
            try:
                self._restore_failed_gripper_pulse(
                    initial_state=np.asarray(initial.state, dtype=np.float32),
                    stop_event=stop_event,
                )
            except BaseException as restore_error:
                raise RuntimeError(
                    f"{pulse_error}; failed to restore pre-pulse aperture: "
                    f"{restore_error}"
                ) from pulse_error
            raise

    def _restore_failed_gripper_pulse(
        self,
        *,
        initial_state: np.ndarray,
        stop_event: threading.Event,
    ) -> None:
        target = float(initial_state[5])
        pinned_arm = np.asarray(initial_state[:5], dtype=np.float32)
        stable = 0
        period_s = 1.0 / GRIPPER_PULSE_SAMPLE_HZ
        for _ in range(GRIPPER_PULSE_MAXIMUM_PHASE_FRAMES):
            started = time.perf_counter()
            if stop_event.is_set():
                raise RuntimeError("pulse restore stopped by operator")
            observation = self.observe()
            state = np.asarray(observation.state, dtype=np.float32)
            drift = float(np.max(np.abs(state[:5] - pinned_arm)))
            if drift > GRIPPER_PULSE_MAXIMUM_ARM_DRIFT:
                raise RuntimeError(
                    "arm moved while restoring failed gripper pulse: "
                    f"{drift:.3f} > {GRIPPER_PULSE_MAXIMUM_ARM_DRIFT:.3f}"
                )
            command = next_gripper_position(
                float(state[5]), target, GRIPPER_PULSE_MAXIMUM_STEP
            )
            self._write_gripper_position(command)
            if abs(float(state[5]) - target) <= GRIPPER_PULSE_SETTLE_TOLERANCE:
                stable += 1
            else:
                stable = 0
            if stable >= GRIPPER_PULSE_STABLE_REQUIRED:
                return
            delay = period_s - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)
        raise RuntimeError(
            "gripper did not restore its pre-pulse aperture: "
            f"target={target:.3f}"
        )

    def _calibrate_gripper_impl(
        self,
        *,
        stop_event: threading.Event,
        self_contact: bool | None = None,
        dual_view_motion_roi: bool = False,
        active_initializer_only: bool = False,
    ) -> dict[str, Any]:
        """Move only the gripper through one slow close-open-close pulse.

        The default path is the historical pulse.  Pass ``self_contact=True``
        or start the server with ``--self-contact-calibration`` to append an
        opt-in empty-jaw seek; ``closed_state`` stays the pulse argmin.
        ``dual_view_motion_roi`` localizes wrist independently for visual
        preflight; it does not change the default pulse step or closed_state.
        ``active_initializer_only`` returns the exact saved RGB/state pulse
        before any legacy image localizer runs.  Prod-V13 FCPG-H uses this path
        so its automatic initialization has no optical-flow dependency.
        """

        sample_hz = GRIPPER_PULSE_SAMPLE_HZ
        period_s = 1.0 / sample_hz
        maximum_step = GRIPPER_PULSE_MAXIMUM_STEP
        settle_tolerance = GRIPPER_PULSE_SETTLE_TOLERANCE
        stable_required = GRIPPER_PULSE_STABLE_REQUIRED
        maximum_phase_frames = GRIPPER_PULSE_MAXIMUM_PHASE_FRAMES
        maximum_arm_drift = GRIPPER_PULSE_MAXIMUM_ARM_DRIFT
        initial = self.observe()
        initial_arm = initial.state[:5].copy()
        initial_aperture = float(initial.state[5])
        if bool(getattr(self, "pre_motion_identity_required", False)):
            handshake = self.run_pre_motion_identity_handshake(
                initial_observation=initial
            )
            self.last_pre_motion_identity = handshake
            if not bool(handshake.get("pre_motion_identity_valid")):
                raise RuntimeError(
                    "pre_motion_identity_refused:"
                    + str(handshake.get("reason") or "identity_break")
                )
        open_position = self.gripper_calibration_open
        closed_position = self.gripper_calibration_closed
        frames: dict[str, list[np.ndarray]] = {view: [] for view in VIEWS}
        states: list[np.ndarray] = []
        commands: list[float] = []
        phase_ranges: list[dict[str, int | str]] = []
        maximum_observed_arm_drift = 0.0

        # The pulse must end where it began: the task flow may need the
        # gripper open (approach requires it), and a calibration that always
        # parks the jaws closed silently re-blocks the next plan.
        for phase_name, target in (
            ("close", closed_position),
            ("open", open_position),
            ("restore", initial_aperture),
        ):
            phase_start = len(states)
            stable = 0
            for _ in range(maximum_phase_frames):
                started = time.perf_counter()
                if stop_event.is_set():
                    raise RuntimeError("grounding calibration stopped by operator")
                observation = self.observe()
                drift = float(
                    np.max(np.abs(observation.state[:5] - initial_arm))
                )
                maximum_observed_arm_drift = max(
                    maximum_observed_arm_drift, drift
                )
                if drift > maximum_arm_drift:
                    raise RuntimeError(
                        "arm moved during gripper calibration: "
                        f"{drift:.3f} > {maximum_arm_drift:.3f}"
                    )
                for view in VIEWS:
                    frames[view].append(observation.frames[view].copy())
                states.append(observation.state.copy())

                command = next_gripper_position(
                    float(observation.state[5]), target, maximum_step
                )
                sent = self._write_gripper_position(command)
                commands.append(sent)
                if abs(float(observation.state[5]) - target) <= settle_tolerance:
                    stable += 1
                else:
                    stable = 0
                if stable >= stable_required:
                    break
                delay = period_s - (time.perf_counter() - started)
                if delay > 0:
                    time.sleep(delay)
            else:
                phase_states = np.asarray(states[phase_start:], dtype=np.float32)
                phase_gripper = phase_states[:, 5]
                raise RuntimeError(
                    "gripper did not settle during calibration phase "
                    f"{phase_name}: target={target:.3f}, "
                    f"start={phase_gripper[0]:.3f}, "
                    f"final={phase_gripper[-1]:.3f}, "
                    f"observed_range=[{np.min(phase_gripper):.3f},"
                    f"{np.max(phase_gripper):.3f}], "
                    f"last_command={commands[-1]:.3f}"
                )
            phase_ranges.append(
                {
                    "name": phase_name,
                    "start": phase_start,
                    "stop": len(states),
                }
            )

        state_array = np.stack(states).astype(np.float32)
        frame_arrays = {
            view: np.stack(frames[view]).astype(np.uint8, copy=False)
            for view in VIEWS
        }
        capture_root = getattr(self, "calibration_capture_root", None)
        capture = None
        if capture_root is not None:
            capture = _save_gripper_calibration_capture(
                Path(capture_root),
                frame_arrays,
                state_array,
                np.asarray(commands, dtype=np.float32),
                phase_ranges,
                sample_hz=sample_hz,
                closed_position=closed_position,
                open_position=open_position,
            )
        if active_initializer_only:
            if capture is None:
                raise RuntimeError(
                    "automatic FCPG-H initialization requires pulse capture storage"
                )
            return {
                "kind": "fcpgh_active_initialization_pulse",
                "state": state_array[-1].copy(),
                "sample_hz": sample_hz,
                "frame_count": len(state_array),
                "maximum_command_step": maximum_step,
                "closed_position": closed_position,
                "open_position": open_position,
                "initial_position": initial_aperture,
                "final_position": float(state_array[-1, 5]),
                "restored_initial": bool(
                    abs(float(state_array[-1, 5]) - initial_aperture)
                    <= settle_tolerance
                ),
                "observed_gripper_range": [
                    float(np.min(state_array[:, 5])),
                    float(np.max(state_array[:, 5])),
                ],
                "maximum_observed_arm_drift": maximum_observed_arm_drift,
                "other_joint_commands_sent": False,
                "command_count": len(commands),
                "phase_ranges": phase_ranges,
                "capture": capture,
                "optical_flow_used": False,
                "localization": "deferred_to_fcpgh_command_locked_initializer",
            }
        # The column camera is external and needs a live image-plane offset.
        # The wrist camera is rigidly attached to the gripper; its nominal
        # moving-tip projection is already expressed in the local camera frame.
        try:
            localizations = {
                "column": localize_gripper_from_pulse(
                    frame_arrays["column"],
                    state_array,
                    maximum_arm_motion=maximum_arm_drift,
                )
            }
        except BaseException as exc:
            location = None if capture is None else capture["index_html"]
            raise RuntimeError(
                f"gripper localization failed; cached capture={location}: {exc}"
            ) from exc
        enabled = (
            bool(self_contact)
            if self_contact is not None
            else bool(getattr(self, "self_contact_calibration_enabled", False))
        )
        localize_wrist = bool(dual_view_motion_roi) or bool(enabled)
        if localize_wrist:
            try:
                localizations["wrist"] = localize_gripper_from_pulse(
                    frame_arrays["wrist"],
                    state_array,
                    maximum_arm_motion=maximum_arm_drift,
                )
            except Exception as exc:  # noqa: BLE001 — wrist is witness-only
                localizations["wrist_error"] = str(exc)
        view_results = {}
        for view, localization in localizations.items():
            if view == "wrist_error":
                continue
            view_results[view] = {
                **localization.to_dict(),
                "open_state": state_array[localization.open_frame_index].copy(),
                "closed_state": state_array[localization.closed_frame_index].copy(),
            }
        if "wrist_error" in localizations:
            view_results.setdefault(
                "wrist",
                {
                    "bbox_xyxy": None,
                    "moving_area_px": None,
                    "failure_reason": localizations["wrist_error"],
                },
            )
        column = localizations["column"]
        result = {
            **column.to_dict(),
            "center_px": np.asarray(column.center_px, dtype=np.float32),
            "state": state_array[-1].copy(),
            "views": view_results,
            "sample_hz": sample_hz,
            "frame_count": len(state_array),
            "maximum_command_step": maximum_step,
            "closed_position": closed_position,
            "open_position": open_position,
            "initial_position": initial_aperture,
            "final_position": float(state_array[-1, 5]),
            "restored_initial": bool(
                abs(float(state_array[-1, 5]) - initial_aperture)
                <= settle_tolerance
            ),
            "observed_gripper_range": [
                float(np.min(state_array[:, 5])),
                float(np.max(state_array[:, 5])),
            ],
            "maximum_observed_arm_drift": maximum_observed_arm_drift,
            "other_joint_commands_sent": False,
            "command_count": len(commands),
            "phase_ranges": phase_ranges,
            "capture": capture,
        }
        if localize_wrist:
            result["motion_roi_by_view"] = motion_gripper_rois_from_pulse(
                result,
                (224, 224),
                require_complete_in_frame=True,
            )
        if not enabled:
            return result
        result["self_contact"] = self._run_self_contact_calibration(
            stop_event=stop_event,
            pulse_result=result,
            state_array=state_array,
            initial_arm=initial_arm,
            restore_target=initial_aperture,
            capture=capture,
        )
        return result

    def capture_gripper_for_fcpgh(
        self, *, stop_event: threading.Event
    ) -> dict[str, Any]:
        """Capture one local dual-view excitation without legacy localization."""

        return self.calibrate_gripper(
            stop_event=stop_event,
            active_initializer_only=True,
        )

    def _run_self_contact_calibration(
        self,
        *,
        stop_event: threading.Event,
        pulse_result: Mapping[str, Any],
        state_array: np.ndarray,
        initial_arm: np.ndarray,
        restore_target: float,
        capture: Mapping[str, str] | None,
    ) -> dict[str, Any]:
        """Continue closing from the observed p1 neighborhood.  Opt-in only.

        ``start_present`` is session p1 (last close-phase Present), not the
        current aperture.  ``run_self_contact_seek`` must ramp from the
        live Present; it must not one-shot p1.
        """

        close_direction = infer_close_direction_from_pulse(
            state_array[:, 5],
            pulse_result["phase_ranges"],
        )
        close_phase = next(
            row
            for row in pulse_result["phase_ranges"]
            if row["name"] == "close"
        )
        start_present = float(state_array[int(close_phase["stop"]) - 1, 5])
        config = getattr(self, "self_contact_config", None)
        if config is None:
            config = SelfContactConfig()
        column_hw = (224, 224)
        rois = pulse_result.get("motion_roi_by_view")
        if not isinstance(rois, Mapping):
            rois = motion_gripper_rois_from_pulse(
                pulse_result,
                column_hw,
                require_complete_in_frame=True,
            )
        roi = rois.get("column") or motion_gripper_roi_from_localization(
            pulse_result, column_hw, view="column"
        )
        if not bool(roi.get("valid")):
            raise RuntimeError(
                "self-contact calibration aborted "
                f"({roi.get('reason') or 'gripper_roi_required'}): "
                "true-contact seek requires a motion-derived gripper ROI "
                "from the preceding open-close pulse; no full-image fallback"
            )
        hooks = SelfContactHardwareHooks(
            observe=self.observe,
            write_gripper=self._write_gripper_position,
            read_telemetry=self._read_optional_servo_telemetry,
            sleep=time.sleep,
        )
        evidence_dir = None if capture is None else capture.get("directory")
        try:
            reference = getattr(self, "pre_motion_reference", None)
            handshake = getattr(self, "last_pre_motion_identity", None)
            record = run_self_contact_seek(
                hooks,
                config,
                pinned_arm=np.asarray(initial_arm, dtype=np.float64),
                start_present=start_present,
                close_direction=close_direction,
                restore_target=float(restore_target),
                stop_event=stop_event,
                gripper_roi_xyxy=roi.get("xyxy"),
                gripper_roi_by_view=rois,
                gripper_roi_provenance=str(roi.get("provenance") or ""),
                allow_full_image=False,
                identity_reference_column=(
                    None
                    if not isinstance(reference, Mapping)
                    else reference.get("column")
                ),
                identity_reference_wrist=(
                    None
                    if not isinstance(reference, Mapping)
                    else reference.get("wrist")
                ),
                pre_motion_identity_valid=(
                    None
                    if not isinstance(handshake, Mapping)
                    else handshake.get("pre_motion_identity_valid")
                ),
                pre_motion_identity=(
                    None if not isinstance(handshake, Mapping) else handshake
                ),
                evidence_dir=evidence_dir,
                require_evidence=evidence_dir is not None,
                require_complete_in_frame=True,
            )
            if record.get("before_first_command") and record.get("abort_reason"):
                sidecar = _persist_self_contact_capture(capture, record)
                if sidecar is not None:
                    record = {**record, "capture": sidecar}
                raise RuntimeError(
                    "self-contact calibration aborted "
                    f"({record.get('abort_reason')})"
                )
        except SelfContactFatalError as exc:
            sidecar = _persist_self_contact_capture(capture, exc.payload)
            if sidecar is not None:
                exc.payload = {**exc.payload, "capture": sidecar}
            raise RuntimeError(
                f"self-contact calibration aborted ({exc.reason})"
            ) from exc
        sidecar = _persist_self_contact_capture(capture, record)
        if sidecar is not None:
            record = {**record, "capture": sidecar}
        return record

    def close(self) -> None:
        runner = getattr(self, "dicg_shadow", None)
        if runner is not None:
            _close_dicg_shadow_runner(runner)
            self.dicg_shadow = None
        follower = getattr(self, "follower", None)
        if follower is None:
            return
        if follower.is_connected:
            follower.disconnect()
        elif follower.bus.is_connected:
            follower.bus.disconnect(disable_torque=False)


class RoundArtifactWriter:
    def __init__(
        self,
        root: Path,
        *,
        render_mujoco: bool,
        experiment_binding: dict[str, Any] | None = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.render_mujoco = bool(render_mujoco)
        self.experiment_binding = (
            None if experiment_binding is None else _jsonable(experiment_binding)
        )

    def __call__(self, plan: RealtimePlan) -> dict[str, str]:
        round_root = self.root / "rounds" / plan.guard.round_id
        round_root.mkdir(parents=True, exist_ok=False)
        assets: dict[str, str] = {}
        for view in VIEWS:
            raw_name = f"{view}_raw.jpg"
            self._save_image_atomic(
                round_root / raw_name,
                plan.frames[view][-1],
            )
            grounding_name = f"{view}_grounding.jpg"
            model_contract = getattr(plan, "model_contract", {})
            prod_v13_pixels = (
                isinstance(model_contract, Mapping)
                and model_contract.get("v12_pixel_anchor_role")
                == "not_executed_current_geometry_owned_by_prod_v13"
            )
            grounding_overlay = (
                self._current_fingertip_overlay(plan, view)
                if prod_v13_pixels
                else draw_grounding(
                    plan.frames[view][-1],
                    plan.nominal[view][-1],
                    plan.corrected[view][-1],
                    plan.valid[view][-1],
                )
            )
            self._save_image_atomic(
                round_root / grounding_name,
                np.asarray(grounding_overlay, dtype=np.uint8),
            )
            assets[f"{view}_raw"] = self._url(plan, raw_name)
            assets[f"{view}_grounding"] = self._url(plan, grounding_name)

        for view in VIEWS:
            target_name = f"{view}_target_paths.jpg"
            self._save_image_atomic(
                round_root / target_name,
                np.asarray(self._target_overlay(plan, view), dtype=np.uint8),
            )
            assets[f"{view}_target_paths"] = self._url(plan, target_name)
        assets["target_paths"] = assets["column_target_paths"]
        snapshot_files = write_planning_snapshots(
            round_root,
            plan,
            latest_dir=self.root / "snapshots",
        )
        for view in SNAPSHOT_VIEWS:
            assets[f"{view}_planning_snapshot"] = self._url(
                plan, f"{view}_planning_snapshot.jpg"
            )
        assets["snapshots_json"] = self._url(plan, "snapshots.json")
        if snapshot_files.get("latest_wrist"):
            assets["latest_wrist_planning_snapshot"] = (
                "/assets/snapshots/latest_wrist.jpg"
            )
            assets["latest_column_planning_snapshot"] = (
                "/assets/snapshots/latest_column.jpg"
            )
            assets["latest_snapshots_json"] = "/assets/snapshots/latest.json"
        if self.render_mujoco and plan.selected_index is not None:
            # In-process GIF uses a second libmujoco model and has SIGSEGV'd
            # the serve process. Preview is optional; execute must not wait.
            error_name = "mujoco_preview_error.txt"
            (round_root / error_name).write_text(
                "skipped: in-process MuJoCo preview disabled; "
                "execute does not wait for GIF render\n",
                encoding="utf-8",
            )
            assets["mujoco_preview_error"] = self._url(plan, error_name)
            print(
                f"MuJoCo preview skipped for {plan.guard.round_id} "
                "(in-process GIF disabled)",
                flush=True,
            )

        plan_record = plan_json(plan)
        if self.experiment_binding is not None:
            plan_record["hardware_experiment_binding"] = self.experiment_binding
        telemetry = getattr(plan, "servo_load_telemetry", None)
        if telemetry is not None:
            plan_record["servo_load_telemetry"] = telemetry
        plan_path = round_root / "plan.json"
        plan_temporary = plan_path.with_suffix(".json.tmp")
        with plan_temporary.open("x", encoding="utf-8") as stream:
            json.dump(
                plan_record,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._commit_atomic(plan_temporary, plan_path)
        assets["plan_json"] = self._url(plan, "plan.json")
        if telemetry is not None:
            try:
                load_name = "servo_load.json"
                load_path = round_root / load_name
                load_temporary = load_path.with_suffix(".json.tmp")
                with load_temporary.open("x", encoding="utf-8") as stream:
                    json.dump(
                        telemetry,
                        stream,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=True,
                    )
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                self._commit_atomic(load_temporary, load_path)
                assets["servo_load_json"] = self._url(plan, load_name)
            except Exception as exc:
                print(
                    f"servo_load telemetry write failed for "
                    f"{plan.guard.round_id}: {exc}",
                    flush=True,
                )
        return assets

    @staticmethod
    def _current_fingertip_overlay(
        plan: RealtimePlan, view: str
    ) -> Image.Image:
        """Draw only measured Prod-V13 C/F/M points; no legacy V12 pixels."""

        image = Image.fromarray(
            np.asarray(plan.frames[view][-1], dtype=np.uint8), mode="RGB"
        )
        draw = ImageDraw.Draw(image)
        points = np.asarray(plan.corrected[view][-1], dtype=np.float32)
        valid = np.asarray(plan.valid[view][-1], dtype=bool)
        colors = ((255, 205, 64), (38, 208, 124), (232, 91, 155))
        labels = ("C", "F", "M")
        if bool(valid[1] and valid[2]):
            draw.line(
                (*map(float, points[1]), *map(float, points[2])),
                fill=(245, 245, 245),
                width=2,
            )
        for point, supported, color, label in zip(
            points, valid, colors, labels, strict=True
        ):
            if not supported:
                continue
            x, y = map(float, point)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=3)
            draw.text((x + 6, y - 8), label, fill=color)
        return image

    def write_execution(
        self,
        *,
        plan: RealtimePlan,
        receipt: dict[str, Any],
        fresh_observation: LiveObservation,
        observations: tuple[LiveObservation, ...],
    ) -> dict[str, str]:
        """Persist one complete five-command execution before the next round."""

        if receipt.get("round_id") != plan.guard.round_id:
            raise ValueError("execution receipt round does not match the plan")
        if len(observations) != 5:
            raise ValueError("execution artifacts require five observations")
        execution_root = self.root / "rounds" / plan.guard.round_id / "execution"
        execution_root.mkdir(parents=True, exist_ok=False)

        frame_files: dict[str, Any] = {"fresh": {}, "post_command": []}
        for view in VIEWS:
            name = f"fresh_{view}.jpg"
            self._save_image_atomic(
                execution_root / name,
                fresh_observation.frames[view],
            )
            frame_files["fresh"][view] = name
        for index, observation in enumerate(observations, start=1):
            row: dict[str, str] = {}
            for view in VIEWS:
                name = f"post_command_{index:02d}_{view}.jpg"
                self._save_image_atomic(
                    execution_root / name,
                    observation.frames[view],
                )
                row[view] = name
            frame_files["post_command"].append(row)

        trajectory_name = "trajectory.npz"
        trajectory_path = execution_root / trajectory_name
        trajectory_temporary = trajectory_path.with_suffix(".npz.tmp")
        with trajectory_temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                fresh_timestamp_s=np.asarray(
                    fresh_observation.timestamp_s, dtype=np.float64
                ),
                fresh_state=np.asarray(
                    fresh_observation.state, dtype=np.float32
                ),
                observation_timestamps_s=np.asarray(
                    [row.timestamp_s for row in observations], dtype=np.float64
                ),
                observed_states=np.stack(
                    [row.state for row in observations]
                ).astype(np.float32),
                requested=np.asarray(receipt["requested"], dtype=np.float32),
                sent=np.asarray(receipt["sent"], dtype=np.float32),
            )
            stream.flush()
            os.fsync(stream.fileno())
        self._commit_atomic(trajectory_temporary, trajectory_path)

        assets = {
            "execution_receipt": self._url(plan, "execution/receipt.json"),
            "execution_trajectory": self._url(
                plan, f"execution/{trajectory_name}"
            ),
        }
        record = {
            **_jsonable(receipt),
            "frame_files": frame_files,
            "trajectory_file": trajectory_name,
            "artifacts": assets,
        }
        if self.experiment_binding is not None:
            record["hardware_experiment_binding"] = self.experiment_binding
        receipt_path = execution_root / "receipt.json"
        receipt_temporary = receipt_path.with_suffix(".json.tmp")
        with receipt_temporary.open("w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._commit_atomic(receipt_temporary, receipt_path)
        return assets

    def write_execution_attempt(
        self,
        *,
        plan: RealtimePlan,
        record: dict[str, Any],
    ) -> dict[str, str]:
        """Append one pre-motion rejection or runtime-fault event."""

        if record.get("active_round_id") != plan.guard.round_id:
            raise ValueError("execution attempt does not match the active plan")
        attempts_root = (
            self.root / "rounds" / plan.guard.round_id / "execution_attempts"
        )
        attempts_root.mkdir(parents=True, exist_ok=True)
        stem = f"attempt_{time.time_ns()}_{threading.get_ident()}"
        destination = attempts_root / f"{stem}.json"
        if destination.exists():
            raise FileExistsError(f"execution attempt already exists: {destination}")
        temporary = attempts_root / f"{stem}.json.tmp"
        payload = {
            **_jsonable(record),
            "plan_phase": plan.phase.value,
            "selected_index": plan.selected_index,
            "selected_chunk_present": plan.selected_chunk is not None,
            "model_contract": _jsonable(plan.model_contract),
        }
        if self.experiment_binding is not None:
            payload["hardware_experiment_binding"] = self.experiment_binding
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._commit_atomic(temporary, destination)
        relative = f"execution_attempts/{destination.name}"
        return {"execution_attempt": self._url(plan, relative)}

    @staticmethod
    def _save_image_atomic(path: Path, frame: np.ndarray) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB").save(
                stream,
                format="JPEG",
                quality=93,
            )
            stream.flush()
            os.fsync(stream.fileno())
        RoundArtifactWriter._commit_atomic(temporary, path)

    @staticmethod
    def _commit_atomic(temporary: Path, destination: Path) -> None:
        temporary.replace(destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _url(self, plan: RealtimePlan, name: str) -> str:
        return f"/assets/rounds/{plan.guard.round_id}/{name}"

    @staticmethod
    def _target_overlay(plan: RealtimePlan, view: str = "column") -> Image.Image:
        base_image = Image.fromarray(plan.frames[view][-1], mode="RGB")
        image = base_image.copy()
        draw = ImageDraw.Draw(image)
        font = load_font(11, bold=True)
        current_anchors = np.asarray(plan.corrected[view][-1], dtype=np.float32)
        anchor_names = ("C", "L", "R")
        command_colors = ((37, 99, 235), (85, 135, 235), (30, 68, 170))
        prediction_colors = ((22, 163, 74), (46, 211, 255), (255, 92, 185))

        prediction_anchors, command_anchors = plan_anchor_paths(plan, view)
        prod_v13 = prod_v13_observed_tip_trajectory(plan, view)
        raw_trajectory = None
        if prod_v13 is not None:
            # The old V12 CAD/FK pixel path is not a seed13161 fingertip
            # prediction. Never display it under the PRED label once the
            # current observer is attached.
            prediction_anchors = None
            raw_trajectory = prod_v13.get("display_px")
            if raw_trajectory is not None:
                raw_trajectory = np.asarray(raw_trajectory, dtype=np.float32)
                current_anchors = raw_trajectory[-1]
                anchor_names = ("C", "A", "B")
        pending_automatic_h0 = prod_v13 is not None and raw_trajectory is None
        if pending_automatic_h0:
            command_anchors = None

        def draw_anchor_paths(
            values: np.ndarray | None,
            colors: tuple[tuple[int, int, int], ...],
            prefix: str,
            width: int,
        ) -> None:
            if values is None:
                return
            paths = np.asarray(values, dtype=np.float32)
            if paths.ndim != 3 or paths.shape[-1] != 2:
                raise ValueError("anchor paths must have shape [H,K,2]")
            if paths.shape[1] not in (1, 3):
                raise ValueError("anchor paths must contain one or three anchors")
            indices = range(paths.shape[1])
            for anchor_index in indices:
                start_index = 0 if paths.shape[1] == 1 else anchor_index
                line = [
                    tuple(map(float, current_anchors[start_index])),
                    *[
                        tuple(map(float, point))
                        for point in paths[:, anchor_index]
                    ],
                ]
                color = colors[start_index]
                draw.line(line, fill=color, width=width)
                for point in line[1:]:
                    draw.ellipse(
                        (
                            point[0] - 2,
                            point[1] - 2,
                            point[0] + 2,
                            point[1] + 2,
                        ),
                        fill=color,
                    )

        draw_anchor_paths(command_anchors, command_colors, "IK", 2)
        if raw_trajectory is not None:
            for anchor_index, color in enumerate(prediction_colors):
                points = raw_trajectory[:, anchor_index]
                finite = np.all(np.isfinite(points), axis=-1)
                previous = None
                for point, keep in zip(points, finite, strict=True):
                    if not keep:
                        previous = None
                        continue
                    current = tuple(map(float, point))
                    if previous is not None:
                        draw.line((*previous, *current), fill=color, width=3)
                    draw.ellipse(
                        (current[0] - 2, current[1] - 2, current[0] + 2, current[1] + 2),
                        fill=color,
                    )
                    previous = current
        else:
            draw_anchor_paths(prediction_anchors, prediction_colors, "PRED", 3)

        # Paths are intentionally drawn first. The target and current anchors
        # must remain visible even when a forecast starts at the current point.
        if not pending_automatic_h0:
            anchors = current_anchors
            anchor_colors = ((255, 255, 255), (46, 211, 255), (255, 92, 185))
            for index, (point, color) in enumerate(
                zip(anchors, anchor_colors, strict=True)
            ):
                radius = 5 if index == 0 else 3
                x, y = map(float, point)
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=color,
                    outline=(20, 20, 20),
                    width=2,
                )
        if prod_v13 is not None:
            draw.rectangle((3, 3, 220, 31), fill=(20, 20, 20))
            state = (
                "FCPG-H H0/TRACK + LEWM POST"
                if raw_trajectory is not None
                else "FCPG-H AUTO H0 PENDING"
            )
            draw.text((7, 5), state, fill=(190, 255, 210), font=font)
            draw.text(
                (7, 17),
                (
                    "CURRENT RGB COORDS; NO FUTURE PIXEL HEAD"
                    if raw_trajectory is not None
                    else "NO PIXEL CLICK; FIRST EXECUTE AUTO-PULSES"
                ),
                fill=(255, 210, 120),
                font=font,
            )

        # The registered drop zone is drawn in every phase (teal) so the
        # operator can verify the release point before the transport begins;
        # the active phase target keeps its yellow crosshair on top.
        descriptors = getattr(plan, "target_descriptors", None) or {}
        drop_descriptor = (
            descriptors.get("drop_zone_column") or {} if view == "column" else {}
        )
        if (
            drop_descriptor.get("visible", False)
            and plan.target.get("kind") != "drop_zone"
        ):
            x, y = map(float, drop_descriptor["centroid_px"])
            draw.ellipse(
                (x - 8, y - 8, x + 8, y + 8), outline=(64, 224, 208), width=3
            )
            draw.line((x - 11, y, x + 11, y), fill=(64, 224, 208), width=2)
            draw.line((x, y - 11, x, y + 11), fill=(64, 224, 208), width=2)
            draw.text(
                (x + 9, y - 15),
                _target_overlay_label("drop_zone", drop_descriptor.get("source")),
                fill=(64, 224, 208),
                font=font,
                stroke_width=2,
                stroke_fill=(20, 20, 20),
            )

        peg_descriptor = descriptors.get("peg_column") or {} if view == "column" else {}
        if peg_descriptor.get("visible", False):
            x, y = map(float, peg_descriptor["centroid_px"])
            peg_color = (239, 68, 68)
            clearance_m = peg_descriptor.get("clearance_m")
            clearance_label = (
                f" {1000.0 * float(clearance_m):.0f}mm"
                if clearance_m is not None and np.isfinite(clearance_m)
                else ""
            )
            draw.ellipse(
                (x - 9, y - 9, x + 9, y + 9), outline=peg_color, width=3
            )
            draw.line((x - 12, y, x + 12, y), fill=peg_color, width=2)
            draw.line((x, y - 12, x, y + 12), fill=peg_color, width=2)
            draw.text(
                (x + 10, y + 7),
                f"PEG{clearance_label}",
                fill=peg_color,
                font=font,
                stroke_width=2,
                stroke_fill=(20, 20, 20),
            )

        # plan.target.centroid_px is a column pixel. The wrist frame gets its own
        # detection, and when that one is missing the empty crosshair is the point:
        # it is the view the grasp verdict is read from.
        crosshair = (
            plan.target
            if view == "column"
            else (descriptors.get("pickup_wrist") or {})
        )
        if crosshair.get("visible", False) and crosshair.get("centroid_px") is not None:
            x, y = map(float, crosshair["centroid_px"])
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(255, 210, 36), width=3)
            draw.line((x - 11, y, x + 11, y), fill=(255, 210, 36), width=2)
            draw.line((x, y - 11, x, y + 11), fill=(255, 210, 36), width=2)
            draw.text(
                (x + 9, y - 15),
                _target_overlay_label(
                    plan.target.get("kind"), crosshair.get("source")
                ),
                fill=(255, 220, 40),
                font=font,
                stroke_width=2,
                stroke_fill=(20, 20, 20),
            )
        elif view != "column":
            draw.text(
                (6, 6),
                "PICKUP NOT DETECTED IN WRIST",
                fill=(255, 120, 120),
                font=font,
                stroke_width=2,
                stroke_fill=(20, 20, 20),
            )

        if prediction_anchors is None and command_anchors is None and raw_trajectory is None:
            return image

        path_arrays = [current_anchors]
        for values in (prediction_anchors, command_anchors):
            if values is not None:
                path_arrays.append(np.asarray(values, dtype=np.float32).reshape(-1, 2))
        if raw_trajectory is not None:
            finite_raw = raw_trajectory[np.all(np.isfinite(raw_trajectory), axis=-1)]
            if finite_raw.size:
                path_arrays.append(finite_raw.reshape(-1, 2))
        all_points = np.concatenate(path_arrays, axis=0)
        minimum = np.min(all_points, axis=0)
        maximum = np.max(all_points, axis=0)
        center = 0.5 * (minimum + maximum)
        crop_size = max(32.0, float(np.max(maximum - minimum)) + 18.0)
        crop_size = min(crop_size, 112.0)
        left = int(np.clip(round(center[0] - crop_size / 2), 0, 224 - crop_size))
        top = int(np.clip(round(center[1] - crop_size / 2), 0, 224 - crop_size))
        right = int(round(left + crop_size))
        bottom = int(round(top + crop_size))
        zoom = base_image.crop((left, top, right, bottom)).resize(
            (224, 224), resample=Image.Resampling.BILINEAR
        )
        zoom_draw = ImageDraw.Draw(zoom)

        def zoom_point(point: np.ndarray) -> tuple[float, float]:
            return (
                float((point[0] - left) * 224.0 / (right - left)),
                float((point[1] - top) * 224.0 / (bottom - top)),
            )

        def draw_zoom_paths(
            values: np.ndarray | None,
            colors: tuple[tuple[int, int, int], ...],
            width: int,
        ) -> None:
            if values is None:
                return
            paths = np.asarray(values, dtype=np.float32)
            for anchor_index in range(paths.shape[1]):
                line = [
                    zoom_point(current_anchors[anchor_index]),
                    *[zoom_point(point) for point in paths[:, anchor_index]],
                ]
                zoom_draw.line(line, fill=colors[anchor_index], width=width)
                for point in line[1:]:
                    zoom_draw.ellipse(
                        (point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3),
                        fill=colors[anchor_index],
                    )

        draw_zoom_paths(command_anchors, command_colors, 4)
        if raw_trajectory is not None:
            for anchor_index, color in enumerate(prediction_colors):
                points = raw_trajectory[:, anchor_index]
                previous = None
                for point in points:
                    if not np.all(np.isfinite(point)):
                        previous = None
                        continue
                    current = zoom_point(point)
                    if previous is not None:
                        zoom_draw.line((*previous, *current), fill=color, width=6)
                    zoom_draw.ellipse(
                        (current[0] - 3, current[1] - 3, current[0] + 3, current[1] + 3),
                        fill=color,
                    )
                    previous = current
        else:
            draw_zoom_paths(prediction_anchors, prediction_colors, 6)
        for index, point in enumerate(current_anchors):
            x, y = zoom_point(point)
            radius = 6 if index == 0 else 4
            zoom_draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=anchor_colors[index],
                outline=(20, 20, 20),
                width=2,
            )
        zoom_draw.rectangle((0, 0, 223, 223), outline=(255, 210, 36), width=3)
        zoom_draw.rectangle((5, 5, 218, 51), fill=(20, 20, 20))
        zoom_draw.text(
            (9, 7),
            f"ZOOM {224 / (right - left):.1f}x",
            fill=(255, 235, 120),
            font=font,
        )
        legend_x = 56
        first_label = "RAW" if prod_v13 is not None else "PRED"
        for row, (row_label, colors) in enumerate(
            ((first_label, prediction_colors), ("IK", command_colors))
        ):
            y = 23 + 15 * row
            zoom_draw.text((9, y - 6), row_label, fill=(245, 245, 245), font=font)
            for index, label in enumerate(anchor_names):
                x = legend_x + index * 48
                zoom_draw.line((x, y, x + 12, y), fill=colors[index], width=5)
                zoom_draw.text((x + 15, y - 6), label, fill=(245, 245, 245), font=font)
        composite = Image.new("RGB", (448, 224), color=(20, 20, 20))
        composite.paste(image, (0, 0))
        composite.paste(zoom, (224, 0))
        return composite

    def _mujoco_gif(self, plan: RealtimePlan, output: Path) -> None:
        if plan.commanded_states is None or plan.predicted_states is None:
            raise RuntimeError("MuJoCo preview requires selected command and prediction")
        renderer = SO101ReplayRenderer(
            DEFAULT_SCENE,
            azimuth=135.0,
            elevation=-25.0,
            distance=0.72,
        )
        commanded = np.concatenate(
            (plan.current_state[None], np.asarray(plan.commanded_states)), axis=0
        )
        predicted = np.concatenate(
            (plan.current_state[None], np.asarray(plan.predicted_states)), axis=0
        )
        command_xyz: list[np.ndarray] = []
        prediction_images: list[np.ndarray] = []
        prediction_xyz: list[np.ndarray] = []
        prediction_clipped: list[bool] = []
        try:
            for command_state, prediction_state in zip(
                commanded, predicted, strict=True
            ):
                command_display, command_was_clipped = _preview_safe_state(
                    command_state
                )
                prediction_display, prediction_was_clipped = _preview_safe_state(
                    prediction_state
                )
                command_image, command_position, command_renderer_clipped = (
                    renderer.render(command_display)
                )
                prediction_image, prediction_position, prediction_renderer_clipped = (
                    renderer.render(prediction_display)
                )
                del command_image
                command_xyz.append(command_position)
                prediction_images.append(prediction_image)
                prediction_xyz.append(prediction_position)
                prediction_clipped.append(
                    prediction_was_clipped or prediction_renderer_clipped
                )
                if command_was_clipped or command_renderer_clipped:
                    prediction_clipped[-1] = True
        finally:
            renderer.close()

        command_path = np.asarray(command_xyz, dtype=np.float64)
        prediction_path = np.asarray(prediction_xyz, dtype=np.float64)
        frozen_wrist = np.asarray(plan.frames["wrist"][-1], dtype=np.uint8)
        frames = [
            _compose_live_mujoco_frame(
                frozen_wrist,
                prediction_images[index],
                phase=plan.phase.value,
                candidate_index=int(plan.selected_index),
                horizon=index,
                raw_ahead=index * 5,
                command_xyz=command_path[index],
                predicted_xyz=prediction_path[index],
                command_path=command_path,
                prediction_path=prediction_path,
                predicted_clipped=prediction_clipped[index],
            )
            for index in range(len(predicted))
        ]
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=320,
            loop=0,
            optimize=False,
        )

    def close(self) -> None:
        return None


TIP_LOCATOR_PANEL_VIEWS = ("wrist", "column")
UNVERIFIED_ABSOLUTE_ACCURACY = "绝对精度未验证"
UNAVAILABLE_LABEL = "unavailable"
_PHYSICAL_GT_KEYS = (
    "independent_physical_gt",
    "has_independent_physical_gt",
    "is_physical_gt",
)


def _json_like(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_like(item) for key, item in value.items()}
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and not isinstance(value, (list, tuple)):
        try:
            return _json_like(tolist())
        except Exception:
            return None
    if isinstance(value, (list, tuple)):
        return [_json_like(item) for item in value]
    return value


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _xy_point(value: Any) -> list[float] | None:
    point = _json_like(value)
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    x = _finite_number(point[0])
    y = _finite_number(point[1])
    if x is None or y is None:
        return None
    return [x, y]


def last_tip_triple(value: Any) -> list[list[float]] | None:
    """Accept a [3,2] C/L/R triple or take the last frame of [H,3,2]."""

    points = _json_like(value)
    if points is None:
        return None
    if not isinstance(points, (list, tuple)) or not points:
        return None
    if len(points) == 3:
        triple = [_xy_point(point) for point in points]
        if all(point is not None for point in triple):
            return [point for point in triple if point is not None]
    return last_tip_triple(points[-1])


def last_confidence_min(value: Any) -> float | None:
    scores = _json_like(value)
    if scores is None:
        return None
    number = _finite_number(scores)
    if number is not None:
        return number
    if not isinstance(scores, (list, tuple)) or not scores:
        return None
    if all(_finite_number(item) is not None for item in scores):
        return min(item for item in (_finite_number(v) for v in scores) if item is not None)
    return last_confidence_min(scores[-1])


def _string_list(value: Any) -> list[str]:
    items = _json_like(value)
    if items is None:
        return []
    if isinstance(items, (str, bytes)):
        text = str(items).strip()
        return [text] if text else []
    if not isinstance(items, (list, tuple)):
        return []
    reasons: list[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            reasons.append(text)
    return reasons


def _explicit_physical_gt(value: Any) -> bool:
    payload = _json_like(value)
    if payload is True:
        return True
    if not isinstance(payload, Mapping):
        return False
    return any(payload.get(key) is True for key in _PHYSICAL_GT_KEYS) or (
        payload.get("present") is True
    )


def _has_independent_physical_gt(row: Mapping[str, Any], grounding: Mapping[str, Any]) -> bool:
    for source in (row, grounding):
        if any(_explicit_physical_gt(source.get(key)) for key in _PHYSICAL_GT_KEYS):
            return True
        physical = source.get("physical_gt")
        if _explicit_physical_gt(physical):
            return True
    return False


def _locator_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    locator = row.get("global_locator")
    nested = locator if isinstance(locator, Mapping) else {}
    reasons = _string_list(
        nested.get("rejection_reasons", row.get("global_locator_rejection_reasons"))
    )
    quality = nested.get("quality")
    quality_map = quality if isinstance(quality, Mapping) else {}
    if not reasons:
        reasons = _string_list(quality_map.get("rejection_reasons"))
    if "accepted" in nested:
        accepted = bool(nested.get("accepted"))
    else:
        accepted = bool(row.get("global_locator_accepted", False))
    points = last_tip_triple(
        nested.get("points_px", row.get("global_locator_points_px"))
    )
    if "enabled" in nested:
        enabled: bool | None = bool(nested.get("enabled"))
    elif "global_locator_enabled" in row:
        enabled = bool(row.get("global_locator_enabled"))
    elif reasons and all(
        reason
        in {"disabled", "disabled_for_view", "disabled_for_coarse_fine"}
        for reason in reasons
    ):
        enabled = False
    elif nested or points is not None or row.get("global_locator_confidence") is not None:
        enabled = True
    else:
        enabled = None
    confidence = last_confidence_min(
        nested.get("confidence", row.get("global_locator_confidence"))
    )
    if confidence is None:
        confidence = _finite_number(quality_map.get("minimum_tip_confidence"))
    if enabled is None and points is None and not reasons and "accepted" not in nested:
        status = UNAVAILABLE_LABEL
    elif enabled is False:
        status = "disabled"
    elif accepted:
        status = "accepted"
    elif enabled is True:
        status = "rejected"
    else:
        status = UNAVAILABLE_LABEL
    return {
        "enabled": enabled,
        "accepted": accepted,
        "adopted": bool(accepted),
        "role": "proposal",
        "status": status,
        "points": points,
        "min_confidence": confidence,
        "rejection_reasons": reasons,
        "label": "proposal",
    }


def _mean_correction_px(
    row: Mapping[str, Any],
    corrected: list[list[float]] | None,
    nominal: list[list[float]] | None,
) -> float | None:
    reported = _finite_number(
        row.get("mean_correction_px", row.get("mean_total_correction_px"))
    )
    if reported is not None:
        return reported
    if corrected is None or nominal is None:
        return None
    return sum(
        math.hypot(c[0] - n[0], c[1] - n[1]) for c, n in zip(corrected, nominal)
    ) / float(len(corrected))


def _column_extrinsic_panel(view: str, row: Mapping[str, Any]) -> dict[str, Any]:
    if view != "column":
        return {
            "applicable": False,
            "measurement_present": False,
            "hard_match": None,
            "residual_px": None,
            "residual_status": "n/a",
            "status": "n/a",
        }
    extrinsic = row.get("column_extrinsic")
    if not isinstance(extrinsic, Mapping):
        return {
            "applicable": True,
            "measurement_present": False,
            "hard_match": None,
            "residual_px": None,
            "residual_status": UNAVAILABLE_LABEL,
            "status": UNAVAILABLE_LABEL,
        }
    measurement = extrinsic.get("measurement")
    measurement_map = measurement if isinstance(measurement, Mapping) else None
    metal_points = (
        None
        if measurement_map is None
        else last_tip_triple(measurement_map.get("metal_points_px"))
    )
    measured_residual = (
        None
        if measurement_map is None
        else _finite_number(measurement_map.get("residual_rms_px"))
    )
    measurement_present = metal_points is not None or measured_residual is not None
    hard_match = extrinsic.get("hard_match_this_frame")
    hard_match_flag = None if hard_match is None else bool(hard_match)
    residual = measured_residual if measurement_present else None
    residual_status = (
        "measured" if residual is not None else UNAVAILABLE_LABEL
    )
    return {
        "applicable": True,
        "measurement_present": measurement_present,
        "hard_match": hard_match_flag,
        "residual_px": residual,
        "residual_status": residual_status,
        "status": "present" if measurement_present else UNAVAILABLE_LABEL,
    }


def _empty_tip_locator_view(view: str, *, grounding: Mapping[str, Any]) -> dict[str, Any]:
    has_gt = _has_independent_physical_gt({}, grounding)
    return {
        "view": view,
        "present": False,
        "nominal": {"role": "nominal", "points": None, "label": "nominal"},
        "correction_seed": {
            "role": "accepted_control_seed",
            "points": None,
            "label": "accepted control seed",
        },
        "corrected": {
            "role": "final_corrected",
            "points": None,
            "mean_correction_px": None,
            "label": "final corrected",
        },
        "locator": {
            "enabled": None,
            "accepted": False,
            "adopted": False,
            "role": "proposal",
            "status": UNAVAILABLE_LABEL,
            "points": None,
            "min_confidence": None,
            "rejection_reasons": [],
            "label": "proposal",
        },
        "column_extrinsic": _column_extrinsic_panel(view, {}),
        "prior_source": None,
        "has_independent_physical_gt": has_gt,
        "accuracy_disclaimer": None if has_gt else UNVERIFIED_ABSOLUTE_ACCURACY,
    }


def fcpg_tip_locator_panel_model(grounding: Any) -> dict[str, Any]:
    """Read-only view model for the 8765 FCPG tip-locator panel.

    Consumes ``plan.grounding`` from ``/api/status``. Missing fields and old
    archives degrade to unavailable. A rejected locator candidate is never
    marked adopted. Pulse / observer residuals are not metal measurements.
    Metal is not independent physical GT.
    """

    payload = grounding if isinstance(grounding, Mapping) else {}
    views: dict[str, Any] = {}
    any_gt = _has_independent_physical_gt({}, payload)
    for view in TIP_LOCATOR_PANEL_VIEWS:
        row = payload.get(view)
        if not isinstance(row, Mapping):
            views[view] = _empty_tip_locator_view(view, grounding=payload)
            any_gt = any_gt or bool(views[view]["has_independent_physical_gt"])
            continue
        nominal = last_tip_triple(row.get("nominal"))
        seed = last_tip_triple(row.get("correction_seed"))
        corrected = last_tip_triple(row.get("corrected"))
        has_gt = _has_independent_physical_gt(row, payload)
        any_gt = any_gt or has_gt
        views[view] = {
            "view": view,
            "present": True,
            "nominal": {"role": "nominal", "points": nominal, "label": "nominal"},
            "correction_seed": {
                "role": "accepted_control_seed",
                "points": seed,
                "label": "accepted control seed",
            },
            "corrected": {
                "role": "final_corrected",
                "points": corrected,
                "mean_correction_px": _mean_correction_px(row, corrected, nominal),
                "label": "final corrected",
            },
            "locator": _locator_payload(row),
            "column_extrinsic": _column_extrinsic_panel(view, row),
            "prior_source": (
                None
                if row.get("prior_source") is None
                else str(row.get("prior_source"))
            ),
            "has_independent_physical_gt": has_gt,
            "accuracy_disclaimer": None if has_gt else UNVERIFIED_ABSOLUTE_ACCURACY,
        }
    return {
        "views": views,
        "has_independent_physical_gt": any_gt,
        "accuracy_disclaimer": None if any_gt else UNVERIFIED_ABSOLUTE_ACCURACY,
    }


def page() -> str:
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 FCPG + LeWM MPC</title>
<style>
:root{color-scheme:light;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#eef0f2;color:#17191c}
*{box-sizing:border-box}body{margin:0}header{height:58px;background:#181b1f;color:#fff;display:flex;align-items:center;justify-content:space-between;padding:0 20px;position:sticky;top:0;z-index:3}
h1{font-size:18px;margin:0;font-weight:650}#badge{font:600 13px ui-monospace,monospace;padding:6px 9px;border:1px solid #59616b;border-radius:4px}
main{max-width:1320px;margin:auto;padding:14px 18px 94px}.statusbar{display:grid;grid-template-columns:repeat(7,minmax(100px,1fr));gap:1px;background:#c8cdd2;border:1px solid #c8cdd2}.metric{background:#fff;padding:9px 11px;min-height:57px}.metric span{display:block;font-size:11px;color:#616970;text-transform:uppercase}.metric strong{font-size:15px;overflow-wrap:anywhere}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}.panel{background:#fff;border:1px solid #c8cdd2;border-radius:6px;overflow:hidden}.panel h2{font-size:13px;margin:0;padding:9px 11px;border-bottom:1px solid #d7dadd}.panel img{width:100%;display:block;aspect-ratio:4/3;object-fit:contain;background:#151719}.panel.planning img{aspect-ratio:1/1;cursor:crosshair}.panel.wide{grid-column:1/-1}.panel.wide img{aspect-ratio:auto;max-height:490px}
.live-panel{border-color:#31835d}.live-panel h2{display:flex;align-items:center;justify-content:space-between}.live-tag{background:#17653a;color:#fff;border-radius:3px;padding:2px 6px;font:700 10px ui-monospace,monospace}.clock-note{color:#59616b;font-weight:500}
.target-tools{grid-column:1/-1;background:#fff;border:1px solid #c8cdd2;border-radius:6px;padding:10px 12px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}.target-tools.required{border:2px solid #b45309;background:#fff7ed}.target-tools[hidden]{display:none}.target-tools strong{font-size:13px}.target-modes{display:flex;border:1px solid #8b949c;border-radius:5px;overflow:hidden}.target-modes button{height:34px;border-radius:0;background:#fff;color:#2a3035;padding:0 13px}.target-modes button[aria-pressed="true"]{background:#235c82;color:#fff}.target-tools label{display:flex;align-items:center;gap:7px;font-size:12px}.target-tools input{width:130px}.target-state{margin-left:auto;font:12px ui-monospace,monospace;color:#4d555c}.target-tools.required .target-state{color:#8a3d06;font-weight:750}.trajectory-mode .panel.planning img{cursor:default}.trajectory-mode #column{aspect-ratio:2/1}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid #e1e4e7}tr.selected{background:#ecfdf3}tr.unsafe{color:#8a1c1c}.log{font:12px ui-monospace,monospace;white-space:pre-wrap;max-height:210px;overflow:auto;padding:10px;margin:0;background:#111418;color:#d8e4df;user-select:text;cursor:text}
.request-error{display:none;grid-column:1/-1;margin-top:12px;border:1px solid #b42318;background:#fff4f2;color:#731b14;padding:10px}.request-error.visible{display:block}.request-error header{height:auto;position:static;background:transparent;color:inherit;padding:0;display:flex;gap:10px}.request-error pre{white-space:pre-wrap;overflow:auto;user-select:text;cursor:text;margin:8px 0 0;max-height:180px}.copy-error,.copy-diagnostics{height:32px;color:#fff;padding:0 11px;font-size:12px}.copy-error{background:#731b14;margin-left:auto}.copy-diagnostics{background:#235c82;margin-left:auto}.runtime-title{display:flex;align-items:center;gap:8px}
.controls{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid #aeb4ba;padding:12px 18px;display:flex;justify-content:center;gap:10px;align-items:center;z-index:4}button{height:44px;border:0;border-radius:5px;padding:0 18px;font-size:14px;font-weight:650;cursor:pointer;white-space:nowrap}.execute{background:#17653a;color:#fff}.auto{background:#0f766e;color:#fff}.auto[aria-pressed="true"]{background:#b45309}.replan{background:#235c82;color:#fff}.calibrate{background:#5d4777;color:#fff}.home{background:#6b4f1d;color:#fff}.stop{background:#b42318;color:#fff}button:disabled{opacity:.4;cursor:not-allowed}#busy{min-width:150px;color:#4d555c;font-size:13px}
.tip-locator{border-color:#2a3035}.tip-locator>h2{display:flex;align-items:center;justify-content:space-between;gap:10px;background:#181b1f;color:#f2f4f6;border-bottom:1px solid #3a4046;letter-spacing:.03em}.tip-stamp{font:700 10px ui-monospace,monospace;color:#c9a227;border:1px solid #8a7320;padding:2px 6px;white-space:nowrap}.tip-disclaimer{margin:0;padding:7px 11px;background:#2a2114;color:#f3d39a;font:12px ui-monospace,monospace;border-bottom:1px solid #5a4a28}.tip-disclaimer.verified{background:#14261c;color:#b7e0c4;border-bottom-color:#2d5a3d}.tip-views{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#2a3035}.tip-view{min-width:0;background:#111418;color:#d8e4df;padding:10px 11px}.tip-view h3{margin:0 0 8px;font:700 11px ui-monospace,monospace;letter-spacing:.12em;text-transform:uppercase;color:#9aa3ab;display:flex;justify-content:space-between;align-items:center}.tip-row{display:grid;grid-template-columns:132px minmax(0,1fr);gap:8px;align-items:start;padding:5px 0;border-top:1px solid #262b30;font:12px ui-monospace,monospace}.tip-role{font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:#8b949c;padding-top:2px}.tip-role[data-role="nominal"]{color:#9aa3ab}.tip-role[data-role="proposal"]{color:#d4a017}.tip-role[data-role="accepted_control_seed"]{color:#3d9b8f}.tip-role[data-role="final_corrected"]{color:#5dbe7a}.tip-chips{display:flex;flex-wrap:wrap;gap:4px}.tip-chip{border:1px solid #3a4046;padding:2px 6px;color:#e8eee9}.tip-chip[data-role="proposal"]{border-style:dashed;border-color:#a67c12;color:#f0d48a}.tip-chip[data-role="proposal"][data-adopted="true"]{border-style:solid;border-color:#3d9b8f;color:#b7e0d4}.tip-chip[data-role="accepted_control_seed"]{border-color:#2f6f68;color:#b7e0d4}.tip-chip[data-role="final_corrected"]{border-color:#2d6a45;color:#c8edc8}.tip-meta{color:#8b949c}.tip-pill{display:inline-block;font:700 10px ui-monospace,monospace;padding:1px 5px;border:1px solid #3a4046;margin:0 4px 3px 0}.tip-pill.on{border-color:#3d9b8f;color:#7ed6c6}.tip-pill.off{border-color:#6b3030;color:#e0a0a0}.tip-pill.miss{border-color:#4d555c;color:#8b949c}.tip-unavail{color:#c9a227}
@media(max-width:800px){.statusbar{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}.panel.wide{grid-column:auto}.tip-views{grid-template-columns:1fr}.controls{justify-content:flex-start;overflow-x:auto}}
</style></head><body>
<header><h1>SO-101 FCPG + LeWM Receding-Horizon MPC</h1><div id="badge">LOADING</div></header>
<main><section class="statusbar">
<div class="metric"><span>Mode</span><strong id="mode">-</strong></div><div class="metric"><span>Phase</span><strong id="phase">-</strong></div><div class="metric"><span>Round</span><strong id="round">-</strong></div><div class="metric"><span>Candidate</span><strong id="candidate">-</strong></div><div class="metric"><span>Operation</span><strong id="operation">-</strong></div><div class="metric"><span>State</span><strong id="state">-</strong></div><div class="metric"><span>LeWM Clock</span><strong id="modelClock">-</strong></div>
</section><section class="grid">
<section class="request-error" id="requestError"><header><strong id="requestErrorTitle">请求失败</strong><button class="copy-error" id="copyError">复制错误</button></header><pre id="requestErrorText" tabindex="0"></pre></section>
<section class="target-tools" id="targetTools"><strong>规划图点击目标</strong><div class="target-modes"><button id="pickupMode" aria-pressed="true">抓取目标</button><button id="dropMode" aria-pressed="false">放置区域</button><button id="pegMode" aria-pressed="false">Peg 障碍</button></div><label>模板半径 <input id="targetRadius" type="range" min="8" max="40" step="1" value="18"><output id="radiusValue">18 px</output></label><span class="target-state" id="targetState">pickup - / drop - / peg -</span></section>
<article class="panel wide tip-locator" id="tipLocator" aria-labelledby="tipLocatorTitle">
<h2 id="tipLocatorTitle">当前 FCPG-H 双视角 measurement <span class="tip-stamp">A / B / MID · NO V12 ANCHORS</span></h2>
<p class="tip-disclaimer" id="tipLocatorDisclaimer" role="status">显示当前图像测量与实际记录轨迹；不是人工物理 GT</p>
<div class="tip-views" id="tipLocatorViews"></div>
</article>
<article class="panel live-panel"><h2>Wrist live monitor <span class="clock-note">not model history</span><span class="live-tag">LIVE</span></h2><img id="liveWrist" alt="Wrist live camera"></article>
<article class="panel live-panel"><h2>Column live monitor <span class="clock-note">not model history</span><span class="live-tag">LIVE</span></h2><img id="liveColumn" alt="Column live camera"></article>
<article class="panel planning"><h2>Wrist planning snapshot: FCPG anchors / pickup reference</h2><img id="wrist"></article>
<article class="panel planning"><h2>Column planning snapshot: current anchors / LeWM forecast / IK trajectory</h2><img id="column"></article>
<article class="panel wide"><h2>Frozen real wrist observation vs LeWM MuJoCo prediction <span id="previewWarning"></span></h2><img id="mujoco"></article>
<article class="panel wide"><h2>Ranked action prefixes</h2><table><thead><tr><th>#</th><th>Demo</th><th>Cost</th><th>Future</th><th>Retrieval</th><th>IK mm</th><th>Safe</th><th>Reason</th></tr></thead><tbody id="candidates"></tbody></table></article>
<article class="panel wide"><h2 class="runtime-title">Runtime <button class="copy-diagnostics" id="copyDiagnostics">复制完整诊断</button></h2><pre class="log" id="log" tabindex="0"></pre></article>
</section></main>
<div class="controls"><button class="execute" id="execute">执行 5 步</button><button class="execute" id="autoExecute">自动运行任务</button><button class="replan" id="replan">重新观测并规划</button><button class="calibrate" id="calibrate">标定夹爪位置</button><button class="calibrate" id="calibrateExcitation">自由空间标定</button><button class="home" id="home">回到初始位置</button><button class="stop" id="stop">停止</button><span id="busy"></span></div>
<script>
let latest=null,requestInFlight=false,refreshInFlight=false,stopInFlight=false,refreshTimer=null,targetRole='pickup';const $=id=>document.getElementById(id);
function fmt(v,n=3){return Number.isFinite(v)?v.toFixed(n):'-'}
function finiteNumber(v){if(v===true||v===false||v==null)return null;const n=Number(v);return Number.isFinite(n)?n:null}
function lastTipTriple(value){if(value==null)return null;if(!Array.isArray(value)||!value.length)return null;if(value.length===3){const pts=value.map(p=>Array.isArray(p)&&p.length===2?(()=>{const x=finiteNumber(p[0]),y=finiteNumber(p[1]);return x==null||y==null?null:[x,y]})():null);if(pts.every(Boolean))return pts}return lastTipTriple(value[value.length-1])}
function lastConfidenceMin(value){if(value==null)return null;const n=finiteNumber(value);if(n!=null)return n;if(!Array.isArray(value)||!value.length)return null;if(value.every(v=>finiteNumber(v)!=null))return Math.min(...value.map(finiteNumber));return lastConfidenceMin(value[value.length-1])}
function stringList(value){if(value==null)return [];if(typeof value==='string'){const t=value.trim();return t?[t]:[]}if(!Array.isArray(value))return [];return value.filter(v=>v!=null&&String(v).trim()).map(v=>String(v).trim())}
function explicitPhysicalGt(value){if(value===true)return true;if(!value||typeof value!=='object')return false;return value.independent_physical_gt===true||value.has_independent_physical_gt===true||value.is_physical_gt===true||value.present===true}
function hasIndependentPhysicalGt(row,grounding){row=row||{};grounding=grounding||{};return [row,grounding].some(src=>explicitPhysicalGt(src.independent_physical_gt)||explicitPhysicalGt(src.has_independent_physical_gt)||explicitPhysicalGt(src.is_physical_gt)||explicitPhysicalGt(src.physical_gt))}
function locatorPayload(row){row=row||{};const nested=(row.global_locator&&typeof row.global_locator==='object')?row.global_locator:{};const quality=(nested.quality&&typeof nested.quality==='object')?nested.quality:{};let reasons=stringList(nested.rejection_reasons!=null?nested.rejection_reasons:row.global_locator_rejection_reasons);if(!reasons.length)reasons=stringList(quality.rejection_reasons);const accepted=('accepted' in nested)?Boolean(nested.accepted):Boolean(row.global_locator_accepted);const points=lastTipTriple(nested.points_px!=null?nested.points_px:row.global_locator_points_px);let enabled=null;if('enabled' in nested)enabled=Boolean(nested.enabled);else if('global_locator_enabled' in row)enabled=Boolean(row.global_locator_enabled);else if(reasons.length&&reasons.every(r=>r==='disabled'||r==='disabled_for_view'||r==='disabled_for_coarse_fine'))enabled=false;else if(Object.keys(nested).length||points||row.global_locator_confidence!=null)enabled=true;let confidence=lastConfidenceMin(nested.confidence!=null?nested.confidence:row.global_locator_confidence);if(confidence==null)confidence=finiteNumber(quality.minimum_tip_confidence);let status='unavailable';if(enabled==null&&points==null&&!reasons.length&&!('accepted' in nested))status='unavailable';else if(enabled===false)status='disabled';else if(accepted)status='accepted';else if(enabled===true)status='rejected';return {enabled,accepted,adopted:Boolean(accepted),role:'proposal',status,points,min_confidence:confidence,rejection_reasons:reasons,label:'proposal'}}
function meanCorrectionPx(row,corrected,nominal){const reported=finiteNumber(row.mean_correction_px!=null?row.mean_correction_px:row.mean_total_correction_px);if(reported!=null)return reported;if(!corrected||!nominal)return null;return corrected.reduce((s,c,i)=>s+Math.hypot(c[0]-nominal[i][0],c[1]-nominal[i][1]),0)/corrected.length}
function columnExtrinsicPanel(view,row){if(view!=='column')return {applicable:false,measurement_present:false,hard_match:null,residual_px:null,residual_status:'n/a',status:'n/a'};const extrinsic=row&&row.column_extrinsic;if(!extrinsic||typeof extrinsic!=='object')return {applicable:true,measurement_present:false,hard_match:null,residual_px:null,residual_status:'unavailable',status:'unavailable'};const measurement=(extrinsic.measurement&&typeof extrinsic.measurement==='object')?extrinsic.measurement:null;const metalPoints=measurement?lastTipTriple(measurement.metal_points_px):null;const measuredResidual=measurement?finiteNumber(measurement.residual_rms_px):null;const measurementPresent=Boolean(metalPoints||measuredResidual!=null);const hardMatch=extrinsic.hard_match_this_frame==null?null:Boolean(extrinsic.hard_match_this_frame);const residual=measurementPresent?measuredResidual:null;return {applicable:true,measurement_present:measurementPresent,hard_match:hardMatch,residual_px:residual,residual_status:residual!=null?'measured':'unavailable',status:measurementPresent?'present':'unavailable'}}
function emptyTipView(view,grounding){const hasGt=hasIndependentPhysicalGt({},grounding);return {view,present:false,nominal:{role:'nominal',points:null,label:'nominal'},correction_seed:{role:'accepted_control_seed',points:null,label:'accepted control seed'},corrected:{role:'final_corrected',points:null,mean_correction_px:null,label:'final corrected'},locator:{enabled:null,accepted:false,adopted:false,role:'proposal',status:'unavailable',points:null,min_confidence:null,rejection_reasons:[],label:'proposal'},column_extrinsic:columnExtrinsicPanel(view,{}),prior_source:null,has_independent_physical_gt:hasGt,accuracy_disclaimer:hasGt?null:'绝对精度未验证'}}
function fcpgTipLocatorModel(grounding){const payload=(grounding&&typeof grounding==='object')?grounding:{};const views={};let anyGt=hasIndependentPhysicalGt({},payload);['wrist','column'].forEach(view=>{const row=payload[view];if(!row||typeof row!=='object'){views[view]=emptyTipView(view,payload);anyGt=anyGt||views[view].has_independent_physical_gt;return}const nominal=lastTipTriple(row.nominal);const seed=lastTipTriple(row.correction_seed);const corrected=lastTipTriple(row.corrected);const hasGt=hasIndependentPhysicalGt(row,payload);anyGt=anyGt||hasGt;views[view]={view,present:true,nominal:{role:'nominal',points:nominal,label:'nominal'},correction_seed:{role:'accepted_control_seed',points:seed,label:'accepted control seed'},corrected:{role:'final_corrected',points:corrected,mean_correction_px:meanCorrectionPx(row,corrected,nominal),label:'final corrected'},locator:locatorPayload(row),column_extrinsic:columnExtrinsicPanel(view,row),prior_source:row.prior_source==null?null:String(row.prior_source),has_independent_physical_gt:hasGt,accuracy_disclaimer:hasGt?null:'绝对精度未验证'}});return {views,has_independent_physical_gt:anyGt,accuracy_disclaimer:anyGt?null:'绝对精度未验证'}}
function tipEl(tag,attrs,children){const node=document.createElement(tag);Object.entries(attrs||{}).forEach(([k,v])=>{if(v==null||v===false)return;if(k==='className')node.className=v;else if(k==='text')node.textContent=v;else node.setAttribute(k,String(v))});(children||[]).forEach(child=>{if(!child)return;node.appendChild(typeof child==='string'?document.createTextNode(child):child)});return node}
function fmtTriple(points){if(!points)return null;return ['C','L','R'].map((name,i)=>tipEl('span',{className:'tip-chip','data-anchor':name,text:name+' '+fmt(points[i][0],1)+','+fmt(points[i][1],1)}))}
function tipPill(label,state){return tipEl('span',{className:'tip-pill '+(state==='yes'?'on':(state==='no'?'off':'miss')),text:label+' '+(state==='yes'?'yes':(state==='no'?'no':'unavailable'))})}
function renderTipStage(role,label,points,extra,adopted){const chips=fmtTriple(points);const body=tipEl('div',{className:'tip-chips'});if(chips)chips.forEach(chip=>{chip.setAttribute('data-role',role);if(role==='proposal')chip.setAttribute('data-adopted',adopted?'true':'false');body.appendChild(chip)});else body.appendChild(tipEl('span',{className:'tip-unavail',text:'unavailable'}));if(extra)body.appendChild(tipEl('span',{className:'tip-meta',text:extra}));return tipEl('div',{className:'tip-row'},[tipEl('div',{className:'tip-role','data-role':role,text:label}),body])}
function renderTipView(model){const loc=model.locator||{};const metal=model.column_extrinsic||{};const adopted=Boolean(loc.adopted);const locExtra=['status '+(loc.status||'unavailable'),loc.min_confidence==null?'min conf unavailable':'min conf '+fmt(loc.min_confidence,3),(loc.rejection_reasons||[]).length?('reject '+(loc.rejection_reasons||[]).join(', ')):(adopted?'':'no rejection reasons')].join(' · ');const metalText=metal.applicable===false?'n/a (column only)':[(metal.measurement_present?'metal measurement present':'metal measurement unavailable'),metal.hard_match==null?'hard_match unavailable':('hard_match '+(metal.hard_match?'yes':'no')),metal.residual_status==='measured'?('residual '+fmt(metal.residual_px,2)+' px'):'residual unavailable'].join(' · ');const prior=model.prior_source?String(model.prior_source):'unavailable';const gt=model.has_independent_physical_gt?'independent physical GT present':'no independent physical GT';const article=tipEl('section',{className:'tip-view','data-view':model.view,'aria-label':model.view+' FCPG tip locator'},[tipEl('h3',null,[tipEl('span',{text:model.view}),tipEl('span',{className:'tip-meta',text:model.present?'archive/live fields':'missing view'})]),renderTipStage('nominal','nominal',model.nominal&&model.nominal.points),renderTipStage('accepted_control_seed','accepted control seed',model.correction_seed&&model.correction_seed.points),renderTipStage('final_corrected','final corrected',model.corrected&&model.corrected.points,(!model.corrected||model.corrected.mean_correction_px==null)?'mean corr unavailable':'mean corr '+fmt(model.corrected.mean_correction_px,2)+' px'),renderTipStage('proposal','proposal',loc.points,locExtra,adopted),tipEl('div',{className:'tip-row'},[tipEl('div',{className:'tip-role',text:'locator'}),tipEl('div',null,[tipPill('enabled',loc.enabled==null?'miss':(loc.enabled?'yes':'no')),tipPill('accepted',loc.accepted?'yes':'no'),tipEl('span',{className:'tip-meta',text:adopted?'adopted as control seed':'not adopted'})])]),tipEl('div',{className:'tip-row'},[tipEl('div',{className:'tip-role',text:'column_extrinsic'}),tipEl('div',{className:metal.status==='unavailable'?'tip-unavail':'tip-meta',text:metalText})]),tipEl('div',{className:'tip-row'},[tipEl('div',{className:'tip-role',text:'prior / GT'}),tipEl('div',{className:'tip-meta',text:'source '+prior+' · '+gt+(model.accuracy_disclaimer?' · '+model.accuracy_disclaimer:'')})])]);return article}
function renderTipLocator(grounding){const model=fcpgTipLocatorModel(grounding);const root=$('tipLocatorViews');if(!root)return;root.replaceChildren();['wrist','column'].forEach(view=>root.appendChild(renderTipView(model.views[view]||emptyTipView(view,grounding))));const note=$('tipLocatorDisclaimer');if(note){note.textContent=model.accuracy_disclaimer||'independent physical GT present';note.classList.toggle('verified',Boolean(model.has_independent_physical_gt))}}
function point2(value){if(!Array.isArray(value)||value.length!==2)return null;const x=finiteNumber(value[0]),y=finiteNumber(value[1]);return x==null||y==null?null:[x,y]}
function pairAt(value,index){if(!Array.isArray(value)||!Array.isArray(value[index])||value[index].length!==2)return null;const pair=value[index].map(point2);return pair.every(Boolean)?pair:null}
function visiblePairAt(value,index){if(!Array.isArray(value)||!Array.isArray(value[index])||value[index].length!==2)return [false,false];return value[index].map(Boolean)}
function midpoint(pair){return pair&&pair.length===2?[(pair[0][0]+pair[1][0])/2,(pair[0][1]+pair[1][1])/2]:null}
function latestRecordedPair(rows,index){if(!Array.isArray(rows))return null;for(let i=rows.length-1;i>=0;i--){const pair=pairAt(rows[i]&&rows[i].measurement_tip_px,index);if(pair)return pair}return null}
function activeTipChips(pair){const mid=midpoint(pair);const values=pair?[pair[0],pair[1],mid]:null;const names=['A','B','MID'];const body=tipEl('div',{className:'tip-chips'});if(!values){body.appendChild(tipEl('span',{className:'tip-unavail',text:'unavailable'}));return body}values.forEach((point,i)=>body.appendChild(tipEl('span',{className:'tip-chip','data-role':'final_corrected','data-anchor':names[i],text:names[i]+' '+fmt(point[0],1)+','+fmt(point[1],1)})));return body}
function renderActiveFcpghView(payload,view,index){payload=payload||{};const measurement=payload.measurement||{};const rows=Array.isArray(payload.rows)?payload.rows:[];const current=pairAt(measurement.tracker_tip_px,index)||latestRecordedPair(rows,index);const visible=visiblePairAt(measurement.tracker_visible,index);const viewRows=rows.filter(row=>pairAt(row&&row.measurement_tip_px,index));const last=viewRows.length?viewRows[viewRows.length-1]:null;let travel=null;if(viewRows.length>1){const a=pairAt(viewRows[viewRows.length-2].measurement_tip_px,index),b=pairAt(viewRows[viewRows.length-1].measurement_tip_px,index);if(a&&b)travel=Math.max(Math.hypot(b[0][0]-a[0][0],b[0][1]-a[0][1]),Math.hypot(b[1][0]-a[1][0],b[1][1]-a[1][1]))}const source=measurement.source||'unavailable';const historyText='rows '+viewRows.length+' · latest '+(last&&last.kind?last.kind:'none')+' · last travel '+(travel==null?'-':fmt(travel,2)+' px');return tipEl('section',{className:'tip-view','data-view':view,'aria-label':view+' current FCPG-H measurement'},[tipEl('h3',null,[tipEl('span',{text:view}),tipEl('span',{className:'tip-meta',text:payload.initialized?'observer initialized':'observer unavailable'})]),tipEl('div',{className:'tip-row'},[tipEl('div',{className:'tip-role','data-role':'final_corrected',text:'current measurement'}),activeTipChips(current)]),tipEl('div',{className:'tip-row'},[tipEl('div',{className:'tip-role',text:'visibility'}),tipEl('div',null,[tipPill('A',visible[0]?'yes':'no'),tipPill('B',visible[1]?'yes':'no')])]),tipEl('div',{className:'tip-row'},[tipEl('div',{className:'tip-role',text:'recorded history'}),tipEl('div',{className:'tip-meta',text:historyText})]),tipEl('div',{className:'tip-row'},[tipEl('div',{className:'tip-role',text:'source'}),tipEl('div',{className:'tip-meta',text:source+' · current image only · no legacy V12 anchor'})])])}
function renderActiveFcpgh(payload){const root=$('tipLocatorViews');if(!root)return;root.replaceChildren();const active=payload&&typeof payload==='object'?payload:{};['wrist','column'].forEach((view,index)=>root.appendChild(renderActiveFcpghView(active,view,index)));const note=$('tipLocatorDisclaimer');if(note){note.textContent='A/B 是当前 FCPG-H/tracker 测量，MID 严格取中点；轨迹只连真实记录行，不画未来像素，不读取旧 V12 C/L/R。';note.classList.remove('verified')}}
function setAsset(id,url,round){const el=$(id);if(!url){el.style.display='none';el.removeAttribute('src');el.onload=null;el.onerror=null;delete el.dataset.assetKey;delete el.dataset.renderedRound;return}const key=url+'?r='+encodeURIComponent(round||'none');if(el.dataset.assetKey!==key){el.dataset.assetKey=key;delete el.dataset.renderedRound;el.onload=()=>{if(el.dataset.assetKey===key)el.dataset.renderedRound=String(round||'')};el.onerror=()=>{if(el.dataset.assetKey===key)delete el.dataset.renderedRound};el.src=key}el.style.display='block'}
function renderCandidates(rows,selected){const body=$('candidates');body.replaceChildren();rows.slice().sort((x,y)=>x.cost-y.cost).forEach(r=>{const tr=document.createElement('tr');const source=Number.isInteger(r.episode)?`ep${r.episode}/f${r.frame}`:(r.source||'-');const ikMm=Number.isFinite(r.anchor_ik_max_error_m)?1000*r.anchor_ik_max_error_m:NaN;tr.className=r.index===selected?'selected':(r.safe?'':'unsafe');[r.index,source,fmt(r.cost),fmt(r.demonstration_future_cost),fmt(r.retrieval_cost),fmt(ikMm),r.safe?'yes':'no',(r.reasons||[]).join(', ')].forEach(value=>{const td=document.createElement('td');td.textContent=String(value);tr.appendChild(td)});body.appendChild(tr)})}
function controls(s){const p=(s&&s.plan)||{};const backendBusy=Boolean(s&&s.active_operation);const busy=requestInFlight||backendBusy;const executable=Boolean(s&&s.execute_enabled&&s.execution_armed&&!s.stopped&&p.executable&&!s.wrist_registration_required);$('execute').disabled=busy||!executable;$('autoExecute').disabled=busy||!executable||!s.bounded_autonomy_enabled;$('replan').disabled=busy||!s||s.stopped;$('calibrate').disabled=busy||!s||!s.calibration_available;$('calibrateExcitation').disabled=busy||!s||!s.excitation_available;$('home').disabled=busy||!s||!s.execute_enabled||!s.home_available||s.stopped;$('pickupMode').disabled=busy;$('dropMode').disabled=busy;$('pegMode').disabled=busy;$('targetRadius').disabled=busy;$('stop').disabled=!s||s.stopped||stopInFlight}
function render(s){latest=s;const p=s.plan||{};const trajectoryMode=p.phase==='trajectory';const wristRequired=Boolean(s.wrist_registration_required);document.body.classList.toggle('trajectory-mode',trajectoryMode);$('targetTools').hidden=trajectoryMode;$('targetTools').classList.toggle('required',wristRequired);if(wristRequired&&targetRole!=='pickup')selectTargetRole('pickup');const ready=Boolean(s.execute_enabled&&s.execution_armed&&p.executable&&!s.stopped&&!wristRequired);const operation=s.active_operation||'-';const permission=s.execute_enabled?'armed':(s.calibration_enabled?'calibration-only':'read-only');$('mode').textContent=s.mode+' / '+permission;$('phase').textContent=p.phase||'-';$('round').textContent=p.round_id||'-';$('candidate').textContent=p.selected_index===null||p.selected_index===undefined?'BLOCKED':'#'+p.selected_index;$('operation').textContent=operation;$('state').textContent=s.input_exhausted?'REPLAY COMPLETE':(s.stopped?'STOPPED':(wristRequired?'WRIST CLICK REQUIRED':(ready?'READY':(s.execute_enabled&&!s.execution_armed?'REPLAN REQUIRED':'BLOCKED'))));$('modelClock').textContent=(s.observation_clock||'-')+' / '+(s.model_history_frames??'-')+' frames';$('badge').textContent=s.error?'FAULT':(s.active_operation?String(s.active_operation).toUpperCase():(s.input_exhausted?'COMPLETE':(wristRequired?'CLICK WRIST':(ready?'READY':'BLOCKED'))));$('badge').style.background=s.error?'#8f1d18':(wristRequired?'#b45309':(ready?'#17653a':'#5a626b'));
const a=s.assets||{};setAsset('wrist',a.wrist_target_paths||a.wrist_grounding,p.round_id);setAsset('column',a.column_target_paths||a.target_paths,p.round_id);setAsset('mujoco',a.mujoco_preview,p.round_id);$('previewWarning').textContent=a.mujoco_preview_error?'(preview unavailable)':'';renderCandidates(p.candidates||[],p.selected_index);renderActiveFcpgh(p.prod_v13_tip_trajectory);
const profile=p.target_profile||{};$('targetState').textContent=wristRequired?'请点击下方 Wrist planning snapshot 中的抓取目标':'pickup '+(profile.pickup_ready?'ready':'-')+' / wrist '+(profile.wrist_pickup_ready?'ready':'-')+' / drop '+(profile.drop_zone_ready?'ready':'-')+' / peg '+(profile.peg_ready?'ready':'-');const diagnostics={request_error:s.request_error||null,request_error_code:s.request_error_code||null,error:s.error,active_operation:s.active_operation,wrist_registration_required:wristRequired,wrist_registration_distance_m:s.wrist_registration_distance_m,observation_clock:s.observation_clock,monitor_frames_feed_model:s.monitor_frames_feed_model,model_history_frames:s.model_history_frames,blocked:p.blocked_reasons,phase:p.phase_status,target:p.target,target_profile:profile,fcpg_h_observer:p.prod_v13_tip_trajectory,prod_v14_plan_shadow:p.prod_v14_shadow,prod_v14_measurement_history:s.prod_v14_shadow_status,prod_v16_1_secant_plan_shadow:p.prod_v16_secant_shadow,prod_v16_1_secant_status:s.prod_v16_secant_shadow_status,calibration:s.calibration,preview_error_asset:a.mujoco_preview_error||null,last_execution_batch:s.last_execution_batch,latest_receipt:s.latest_receipt};$('log').textContent=JSON.stringify(diagnostics,null,2);controls(s);if(s.error&&!$('requestError').classList.contains('visible'))showRequestError(s.error)}
function errorTitle(message){const text=String(message||'');if(text.includes('status packet'))return '机械臂通信中断';if(text.includes('Timed out waiting for frame'))return '相机取帧超时';if(text.includes('live state drift')||text.includes('fresh plan')||text.includes('stale planning'))return '现场状态已变化，请重新规划';if(text.includes('did not settle'))return '机械臂未在时限内稳定';if(text.includes('clipped or changed'))return 'LeRobot 修改了命令，执行已停止';if(text.includes('another planning'))return '已有操作正在进行';if(text.includes('no complete next five-frame'))return '回放数据已经结束';return '请求失败'}
function showRequestError(message){$('requestErrorTitle').textContent=errorTitle(message);$('requestErrorText').textContent=String(message);$('requestError').classList.add('visible')}
function clearRequestError(){$('requestErrorText').textContent='';$('requestError').classList.remove('visible')}
async function readJson(response){const text=await response.text();try{return JSON.parse(text)}catch(e){throw new Error(`HTTP ${response.status}: ${text.slice(0,500)}`)}}
function scheduleRefresh(delay=1500){clearTimeout(refreshTimer);refreshTimer=setTimeout(refresh,delay)}
async function refresh(){if(requestInFlight||refreshInFlight){scheduleRefresh();return}refreshInFlight=true;try{const r=await fetch('/api/status',{cache:'no-store'});const d=await readJson(r);if(!r.ok)throw new Error(d.request_error||`HTTP ${r.status}`);render(d)}catch(e){showRequestError(String(e));$('badge').textContent='DISCONNECTED';$('badge').style.background='#8f1d18'}finally{refreshInFlight=false;scheduleRefresh()}}
async function post(path,payload={},label='处理中'){if(requestInFlight){showRequestError('another planning or execution request is active');return}requestInFlight=true;clearTimeout(refreshTimer);$('busy').textContent=label;clearRequestError();controls(latest);try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await readJson(r);render(d);if(!r.ok)showRequestError(d.request_error||d.error||`HTTP ${r.status}`)}catch(e){showRequestError(String(e))}finally{requestInFlight=false;$('busy').textContent='';controls(latest);scheduleRefresh(250)}}
async function stopNow(){if(stopInFlight)return;stopInFlight=true;$('busy').textContent='正在停止';controls(latest);try{const r=await fetch('/api/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});const d=await readJson(r);render(d);if(!r.ok)showRequestError(d.request_error||d.error||`HTTP ${r.status}`)}catch(e){showRequestError(String(e))}finally{stopInFlight=false;$('busy').textContent='';controls(latest);scheduleRefresh(250)}}
function selectTargetRole(role){targetRole=role;$('pickupMode').setAttribute('aria-pressed',String(role==='pickup'));$('dropMode').setAttribute('aria-pressed',String(role==='drop_zone'));$('pegMode').setAttribute('aria-pressed',String(role==='peg'));$('targetRadius').value=role==='drop_zone'?'28':'18';$('radiusValue').textContent=$('targetRadius').value+' px'}
function planningPoint(view,event){const el=event.currentTarget;const rect=el.getBoundingClientRect();const naturalWidth=el.naturalWidth||224;const naturalHeight=el.naturalHeight||224;const scale=Math.min(rect.width/naturalWidth,rect.height/naturalHeight);const imageLeft=rect.left+(rect.width-naturalWidth*scale)/2;const imageTop=rect.top+(rect.height-naturalHeight*scale)/2;const sourceX=(event.clientX-imageLeft)/scale;const sourceY=(event.clientY-imageTop)/scale;const panelWidth=Math.min(224,naturalWidth);if(sourceX<0||sourceY<0||sourceX>=panelWidth||sourceY>=naturalHeight){showRequestError(view==='column'&&naturalWidth>224?'请点击 column 规划图左侧的真实相机画面':'点击位置不在规划图内');return null}return [sourceX*224/panelWidth,sourceY*224/naturalHeight]}
function registerTarget(view,event){if(requestInFlight||latest?.plan?.phase==='trajectory')return;if((targetRole==='drop_zone'||targetRole==='peg')&&view!=='column'){showRequestError((targetRole==='peg'?'Peg 障碍':'放置区域')+'只能在 column 规划图中设置');return}const point=planningPoint(view,event);if(!point)return;const [x,y]=point;const radius=Number($('targetRadius').value);const renderedRound=event.currentTarget.dataset.renderedRound;if(!renderedRound){showRequestError(view+' 规划图尚未完成加载，请等待当前轮图像显示');return}if(targetRole==='peg'){if(confirm('使用当前 column 规划图中的点击区域设置 Peg 障碍？'))post('/api/context-reference',{round_id:renderedRound,kind:'peg',view:'column',centroid_px:[x,y],radius_px:radius},'正在登记 Peg 障碍并重新规划');return}const label=targetRole==='pickup'?'抓取目标':'放置区域';const required=Boolean(latest?.wrist_registration_required&&targetRole==='pickup'&&view==='wrist');if(required||confirm('使用当前 '+view+' 规划图中的点击区域设置'+label+'？'))post('/api/target-reference',{round_id:renderedRound,role:targetRole,view:view,centroid_px:[x,y],radius_px:radius},'正在保存目标外观并重新规划')}
function attachLiveStream(id,path){const el=$(id);let retry=null;const connect=()=>{clearTimeout(retry);el.src=path+'?t='+Date.now()};el.onerror=()=>{clearTimeout(retry);retry=setTimeout(connect,1000)};connect()}
$('pickupMode').onclick=()=>selectTargetRole('pickup');$('dropMode').onclick=()=>selectTargetRole('drop_zone');$('pegMode').onclick=()=>selectTargetRole('peg');$('targetRadius').oninput=()=>$('radiusValue').textContent=$('targetRadius').value+' px';$('wrist').onclick=e=>registerTarget('wrist',e);$('column').onclick=e=>registerTarget('column',e);$('execute').onclick=()=>{if(!latest||!latest.plan)return;post('/api/execute',{round_id:latest.plan.round_id,confirmed:true,max_chunks:1,task_class:latest.task_class},'正在执行 5 步并重新观测')};$('autoExecute').onclick=()=>{if(!latest||!latest.plan||!latest.bounded_autonomy_enabled)return;post('/api/execute',{round_id:latest.plan.round_id,confirmed:true,max_chunks:latest.max_autonomous_chunks,task_class:latest.task_class},'正在闭环运行任务')};$('replan').onclick=()=>post('/api/plan',{},'正在读取双相机并规划');$('calibrate').onclick=()=>{if(confirm('保持工作区清空。仅让夹爪慢速开合一次并标定 column 中的夹爪位置？'))post('/api/calibrate',{confirmed:true},'正在开合夹爪并标定')};$('calibrateExcitation').onclick=()=>{if(confirm('夹爪必须空载。机械臂先抬离桌面几厘米，在自由空间开合一次标定爪尖，再回到当前抓取位。继续？'))post('/api/calibrate-excitation',{confirmed:true},'正在自由空间标定')};$('home').onclick=()=>{if(confirm('缓慢回到记录的任务初始位置？'))post('/api/home',{confirmed:true},'正在返回初始位置')};$('stop').onclick=stopNow;attachLiveStream('liveWrist','/stream/wrist.mjpg');attachLiveStream('liveColumn','/stream/column.mjpg');refresh();
async function copyText(value,label){try{await navigator.clipboard.writeText(value)}catch(e){const area=document.createElement('textarea');area.value=value;area.style.position='fixed';area.style.opacity='0';document.body.appendChild(area);area.select();document.execCommand('copy');area.remove()}$('busy').textContent=label;setTimeout(()=>$('busy').textContent='',1200)}
$('copyError').onclick=()=>copyText($('requestErrorText').textContent,'错误已复制');
$('copyDiagnostics').onclick=()=>copyText($('log').textContent,'完整诊断已复制');
</script></body></html>"""


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _replace_nonfinite_json_numbers(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_nonfinite_json_numbers(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_nonfinite_json_numbers(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _dumps_http_json(payload: Any) -> str:
    return json.dumps(
        _replace_nonfinite_json_numbers(_jsonable(payload)),
        indent=2,
        allow_nan=False,
    )


def _safe_snapshot_round_id(round_id: str | None) -> str | None:
    if round_id is None:
        return None
    value = str(round_id).strip()
    if not value:
        raise ValueError("round_id must be a non-empty string")
    if "/" in value or "\\" in value or ".." in value:
        raise ValueError("round_id must be a single path segment")
    return value


def _query_round_id(query: dict[str, list[str]]) -> str | None:
    values = query.get("round_id") or []
    if not values:
        return None
    if len(values) != 1:
        raise ValueError("round_id must be a single value")
    return _safe_snapshot_round_id(values[0])


def _live_snapshot_frame(controller: RealtimeMPCController, view: str, plan: Any) -> np.ndarray:
    getter = getattr(controller, "preview_frame", None)
    if callable(getter):
        try:
            return planning_frame(getter(view))
        except Exception:
            pass
    return planning_frame(plan.frames[view])


def serve_planning_snapshot_jpeg(
    controller: RealtimeMPCController,
    view: str,
    *,
    asset_root: Path,
    round_id: str | None = None,
) -> bytes:
    if view not in SNAPSHOT_VIEWS:
        raise ValueError(f"unsupported snapshot view: {view}")
    root = asset_root.expanduser().resolve()
    requested = _safe_snapshot_round_id(round_id)
    if requested is not None:
        return load_round_snapshot_jpeg(root / "rounds" / requested, view)
    plan = getattr(controller, "plan", None)
    if plan is not None:
        frame = _live_snapshot_frame(controller, view, plan)
        return snapshot_jpeg_bytes(render_planning_snapshot(frame, plan, view))
    return load_latest_snapshot_jpeg(root / "snapshots", view)


def serve_planning_snapshot_meta(
    controller: RealtimeMPCController,
    *,
    asset_root: Path,
    round_id: str | None = None,
) -> dict[str, Any]:
    root = asset_root.expanduser().resolve()
    requested = _safe_snapshot_round_id(round_id)
    if requested is not None:
        return load_round_snapshot_meta(root / "rounds" / requested)
    plan = getattr(controller, "plan", None)
    if plan is not None:
        latest_dir = root / "snapshots"
        files = {
            "latest_wrist": str(latest_dir / "latest_wrist.jpg"),
            "latest_column": str(latest_dir / "latest_column.jpg"),
            "latest_meta": str(latest_dir / "latest.json"),
        }
        guard = getattr(plan, "guard", None)
        current_id = getattr(guard, "round_id", None)
        if current_id:
            round_root = root / "rounds" / str(current_id)
            files["wrist"] = str(round_root / "wrist_planning_snapshot.jpg")
            files["column"] = str(round_root / "column_planning_snapshot.jpg")
            files["meta"] = str(round_root / "snapshots.json")
        return build_snapshot_meta(plan, files=files)
    return load_latest_snapshot_meta(root / "snapshots")


def _jpeg_bytes(frame: np.ndarray) -> bytes:
    values = np.asarray(frame)
    if values.shape != (224, 224, 3) or values.dtype != np.uint8:
        raise ValueError("monitor frame must be uint8 [224,224,3]")
    output = BytesIO()
    Image.fromarray(values, mode="RGB").save(
        output,
        format="JPEG",
        quality=82,
        optimize=False,
    )
    return output.getvalue()


def _target_overlay_label(kind: str | None, source: str | None = None) -> str:
    base = {
        "pickup": "PICKUP TARGET",
        "drop_zone": "DROP ZONE",
        "peg": "PEG",
    }.get(kind, "TARGET")
    if source in {
        "frozen_vit",
        "open_vocab",
        "predicted",
        "reprojection",
        "unavailable",
    }:
        short = {
            "pickup": "PICKUP",
            "drop_zone": "DROP",
            "peg": "PEG",
        }.get(kind, "TARGET")
        return f"{short} [{source}]"
    return base


def _request_error_code(exc: Exception) -> str:
    message = str(exc).lower()
    if isinstance(exc, PlanRejectedError):
        return "plan_rejected"
    if isinstance(exc, (TypeError, ValueError, json.JSONDecodeError)):
        return "invalid_request"
    if "status packet" in message:
        return "robot_communication"
    if "timed out waiting for frame" in message:
        return "camera_timeout"
    if "did not settle" in message:
        return "robot_not_settled"
    if "clipped or changed" in message:
        return "command_changed"
    if "another planning or execution" in message:
        return "operation_busy"
    return "runtime_error"


def _error_response_payload(
    snapshot: dict[str, Any], exc: Exception
) -> dict[str, Any]:
    """Compose a refusal body that states why the request was refused.

    ``error`` is the latched controller fault, and a request can be refused
    without latching one -- every rejection before motion is. Leaving it null on
    a 4xx reported only that the controller was healthy, so a refused execute
    read as a mystery while the reason sat under a key nothing pointed at. A
    latched fault outranks the per-request reason, being the condition the
    operator has to clear first.
    """

    reason = f"{type(exc).__name__}: {exc}"
    return {
        **snapshot,
        "error": snapshot.get("error") or reason,
        "request_error_code": _request_error_code(exc),
        "request_error": reason,
    }


def _parse_context_reference_request(
    payload: dict[str, Any],
) -> tuple[str, np.ndarray, float]:
    if payload.get("kind") != "peg":
        raise ValueError("context kind must be exactly 'peg'")
    if payload.get("view") != "column":
        raise ValueError("peg context view must be exactly 'column'")
    round_id = payload.get("round_id")
    if not isinstance(round_id, str) or not round_id:
        raise ValueError("context round_id must be a non-empty string")
    centroid = np.asarray(payload.get("centroid_px"), dtype=np.float32)
    if centroid.shape != (2,) or not np.all(np.isfinite(centroid)):
        raise ValueError("context centroid_px must be a finite two-vector")
    if np.any(centroid < 0) or np.any(centroid >= 224):
        raise ValueError("context centroid_px lies outside the planning frame")
    radius_value = payload.get("radius_px")
    if isinstance(radius_value, bool):
        raise ValueError("context radius_px must be numeric")
    radius_px = float(radius_value)
    if not np.isfinite(radius_px) or not 6 <= radius_px <= 112:
        raise ValueError("context radius_px must be between 6 and 112")
    return round_id, centroid, radius_px


def _parse_target_reference_request(
    payload: dict[str, Any],
) -> tuple[str, str, str, np.ndarray, float]:
    # Drop is fail-closed until this endpoint (or an equivalent per-task
    # target-reference) registers it. The planner does not invent a drop
    # template from appearance.
    role = payload.get("role")
    if role not in {"pickup", "drop_zone"}:
        raise ValueError("target role must be exactly 'pickup' or 'drop_zone'")
    view = payload.get("view")
    if view not in VIEWS:
        raise ValueError("target view must be exactly 'wrist' or 'column'")
    if role == "drop_zone" and view != "column":
        raise ValueError("drop_zone target view must be exactly 'column'")
    round_id = payload.get("round_id")
    if not isinstance(round_id, str) or not round_id:
        raise ValueError("target round_id must be a non-empty string")
    centroid = np.asarray(payload.get("centroid_px"), dtype=np.float32)
    if centroid.shape != (2,) or not np.all(np.isfinite(centroid)):
        raise ValueError("target centroid_px must be a finite two-vector")
    if np.any(centroid < 0) or np.any(centroid >= 224):
        raise ValueError("target centroid_px lies outside the planning frame")
    radius_value = payload.get("radius_px")
    if isinstance(radius_value, bool):
        raise ValueError("target radius_px must be numeric")
    radius_px = float(radius_value)
    if not np.isfinite(radius_px) or not 6 <= radius_px <= 112:
        raise ValueError("target radius_px must be between 6 and 112")
    return round_id, role, view, centroid, radius_px


def _set_target_reference_from_request(
    controller: RealtimeMPCController,
    payload: dict[str, Any],
) -> dict[str, Any]:
    round_id, role, view, centroid_px, radius_px = (
        _parse_target_reference_request(payload)
    )
    return controller.set_target_reference(
        role,
        view,
        centroid_px,
        radius_px=radius_px,
        round_id=round_id,
    )


def handler_factory(
    controller: RealtimeMPCController, asset_root: Path
) -> type[BaseHTTPRequestHandler]:
    root = asset_root.expanduser().resolve()
    # Isolate MuJoCo FK / CUDA rank / arm I/O from ThreadingHTTPServer status
    # and asset threads. Overnight native deaths happened mid-replan while
    # status/asset GETs overlapped the 34-candidate rank path.
    native_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="so101-native"
    )

    def _run_native(fn, *args, **kwargs):
        return native_executor.submit(fn, *args, **kwargs).result()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = _dumps_http_json(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def stream_monitor(self, view: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            period_s = 1.0 / MONITOR_STREAM_FPS
            try:
                while True:
                    started = time.perf_counter()
                    payload = _jpeg_bytes(controller.preview_frame(view))
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
                    )
                    self.wfile.write(payload)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    delay = period_s - (time.perf_counter() - started)
                    if delay > 0:
                        time.sleep(delay)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:
                print(
                    f"{view} monitor stream stopped: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = page().encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/status":
                payload = controller.status()
                shadow_status = _dicg_shadow_http_status(controller)
                if shadow_status is not None:
                    payload["dicg_shadow"] = shadow_status
                self.send_json(payload)
                return
            if parsed.path == "/api/servo-debug":
                self.send_json(_run_native(controller.read_servo_debug))
                return
            if parsed.path == "/api/fcpg-observer-status":
                query = parse_qs(parsed.query, keep_blank_values=False)
                refresh = query.get("refresh", ("0",))[-1] in ("1", "true")
                self.send_json(
                    _run_native(
                        controller.prod_v13_observer_status,
                        refresh=refresh,
                    )
                )
                return
            if parsed.path in (
                "/api/snapshot/wrist",
                "/api/snapshot/column",
                "/api/snapshot/meta",
            ):
                try:
                    query = parse_qs(parsed.query, keep_blank_values=False)
                    requested = _query_round_id(query)
                    if parsed.path == "/api/snapshot/meta":
                        self.send_json(
                            serve_planning_snapshot_meta(
                                controller,
                                asset_root=root,
                                round_id=requested,
                            )
                        )
                        return
                    view = parsed.path.rsplit("/", 1)[-1]
                    payload = serve_planning_snapshot_jpeg(
                        controller,
                        view,
                        asset_root=root,
                        round_id=requested,
                    )
                except FileNotFoundError:
                    self.send_json(
                        {
                            "error": "planning snapshot not found",
                            "path": parsed.path,
                        },
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                except ValueError as exc:
                    self.send_json(
                        {
                            "error": str(exc),
                            "request_error_code": "invalid_request",
                        },
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if parsed.path in ("/stream/wrist.mjpg", "/stream/column.mjpg"):
                view = Path(parsed.path).stem
                self.stream_monitor(view)
                return
            if parsed.path.startswith("/assets/"):
                relative = Path(unquote(parsed.path.removeprefix("/assets/")))
                target = (root / relative).resolve()
                if root not in target.parents or not target.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                body = target.read_bytes()
                content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 4096:
                raise ValueError("request body is too large")
            if length == 0:
                return {}
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def do_POST(self) -> None:
            try:
                payload = self._body()
                if self.path == "/api/plan":
                    result = _run_native(controller.replan)
                elif self.path == "/api/execute":
                    requested_chunks = payload.get("max_chunks", 1)
                    if (
                        not isinstance(requested_chunks, int)
                        or isinstance(requested_chunks, bool)
                        or not 1
                        <= requested_chunks
                        <= controller.max_autonomous_chunks
                    ):
                        raise ValueError(
                            "max_chunks must be within the configured run budget"
                        )
                    if (
                        requested_chunks > 1
                        and not controller.autonomous_run_enabled
                    ):
                        raise ValueError(
                            "multi-chunk execution requires bounded autonomy"
                        )
                    result = _run_native(
                        controller.execute,
                        round_id=str(payload.get("round_id", "")),
                        confirmed=payload.get("confirmed") is True,
                        max_chunks=requested_chunks,
                        task_class=payload.get("task_class"),
                    )
                elif self.path == "/api/home":
                    result = _run_native(
                        controller.return_home,
                        confirmed=payload.get("confirmed") is True,
                    )
                elif self.path == "/api/calibrate":
                    result = _run_native(
                        controller.calibrate_grounding,
                        confirmed=payload.get("confirmed") is True,
                    )
                elif self.path == "/api/reanchor":
                    result = _run_native(
                        controller.reanchor_grounding,
                        confirmed=payload.get("confirmed") is True,
                    )
                elif self.path == "/api/calibrate-excitation":
                    result = _run_native(
                        controller.calibrate_with_excitation,
                        confirmed=payload.get("confirmed") is True,
                    )
                elif self.path == "/api/open-gripper":
                    result = _run_native(
                        controller.open_gripper_in_place,
                        confirmed=payload.get("confirmed") is True,
                    )
                elif self.path == "/api/fcpg-query-init":
                    result = _run_native(
                        controller.initialize_prod_v13_query,
                        payload,
                    )
                elif self.path == "/api/fcpg-auto-init":
                    result = _run_native(
                        controller.initialize_prod_v13_automatically,
                        confirmed=payload.get("confirmed") is True,
                    )
                elif self.path == "/api/fcpg-identity-probe":
                    result = _run_native(
                        controller.run_prod_v13_identity_probe,
                        confirmed=payload.get("confirmed") is True,
                        empty_gripper_confirmed=(
                            payload.get("empty_gripper_confirmed") is True
                        ),
                        workspace_clear_confirmed=(
                            payload.get("workspace_clear_confirmed") is True
                        ),
                    )
                elif self.path == "/api/stop":
                    result = controller.stop()
                elif self.path == "/api/target-reference":
                    result = _run_native(
                        _set_target_reference_from_request,
                        controller,
                        payload,
                    )
                elif self.path == "/api/context-reference":
                    round_id, centroid_px, radius_px = (
                        _parse_context_reference_request(payload)
                    )
                    result = _run_native(
                        controller.set_context_reference,
                        round_id,
                        centroid_px,
                        radius_px=radius_px,
                    )
                elif self.path == "/api/v11_preview":
                    result = _run_native(controller.preview_v11_chunk)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self.send_json(result)
            except Exception as exc:
                print(
                    f"request failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                refusal = _error_response_payload(controller.status(), exc)
                status = (
                    HTTPStatus.BAD_REQUEST
                    if refusal["request_error_code"] == "invalid_request"
                    else HTTPStatus.CONFLICT
                )
                self.send_json(refusal, status=status)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.client_address[0]} {fmt % args}", flush=True)

    return Handler


def _effective_live_anchor_calibration(args: argparse.Namespace) -> Path | None:
    if bool(getattr(args, "disable_live_anchor_calibration", False)):
        return None
    candidates: list[Path] = []
    if args.live_anchor_calibration is not None:
        cli_calibration = args.live_anchor_calibration.expanduser()
        if cli_calibration.exists():
            candidates.append(cli_calibration.resolve())
    session_calibration = (
        args.output.expanduser().resolve() / "live_anchor_calibration.json"
    )
    if session_calibration.exists():
        candidates.append(session_calibration)
    return (
        max(candidates, key=lambda path: path.stat().st_mtime_ns)
        if candidates
        else None
    )


def _run_startup_hardware_preflight(
    args: argparse.Namespace,
    *,
    live_anchor_calibration: Path | None,
) -> dict[str, Any] | None:
    """Persist and enforce a current zero-motion Gate before hardware is opened."""

    if args.replay_session is not None:
        return None
    report = build_hardware_preflight_report(
        bridge_checkpoint=args.bridge_checkpoint,
        model_checkpoint=args.model_checkpoint,
        global_tip_checkpoint=args.global_tip_checkpoint,
        disable_global_tip_locator=args.disable_global_tip_locator,
        calibration=args.calibration,
        secondary_cache=args.secondary_cache,
        live_anchor_calibration=live_anchor_calibration,
        home_state_report=args.home_state_report,
        output=args.output,
        robot_port=args.robot_port,
        robot_id=args.robot_id,
        calibration_dir=args.calibration_dir,
        wrist_device=args.wrist_device,
        column_device=args.column_device,
        execute_enabled=args.execute_enabled,
        gripper_calibration_enabled=args.gripper_calibration_enabled,
        acknowledge_workspace_clear=args.acknowledge_workspace_clear,
        control_mode=args.control_mode,
    )
    if getattr(args, "experiment_binding", None) is not None:
        report["hardware_experiment_binding"] = _jsonable(
            args.experiment_binding
        )
    preflight_root = args.output.expanduser().resolve() / "startup_preflight"
    preflight_root.mkdir(parents=True, exist_ok=True)
    report_path = preflight_root / f"preflight_{time.time_ns()}.json"
    with report_path.open("x", encoding="ascii") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(preflight_root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(f"startup hardware preflight: {report['decision']} ({report_path})")
    require_hardware_preflight(report)
    return {**report, "report_path": str(report_path)}


def _dump_planning_frames(plan: Any) -> None:
    """Write each planning view and report where the target appearance sits.

    The far view's own detector response answers a question the registered
    centroid cannot: whether the template sits at the frame edge because the
    OBJECT is at the edge, or because the registration was placed poorly. Only
    the first is fixable by moving the object.
    """
    import cv2
    import numpy as np

    out = Path("/tmp/fcpg_trial_state")
    out.mkdir(parents=True, exist_ok=True)
    frames = getattr(plan, "frames", None)
    if not isinstance(frames, dict):
        return

    print()
    print("planning frames")
    for role, stack in sorted(frames.items()):
        data = np.asarray(stack[-1], dtype=np.uint8)
        path = out / f"dryrun_{role}_view.png"
        cv2.imwrite(str(path), data)
        b, g, r = (data[:, :, i].astype(np.float64) for i in range(3))
        rg, rb = r - g, r - b
        print(f"  {role:8s} {data.shape}  -> {path}")
        print(f"           channel means B={b.mean():6.1f} G={g.mean():6.1f} "
              f"R={r.mean():6.1f}   R-G={rg.mean():+6.1f}")

        # Where is the most target-like region, on the frame's own terms?
        strict = (r > 140) & (rg > 50) & (rb > 25)
        if strict.sum() > 0:
            n, _, stats, centroids = cv2.connectedComponentsWithStats(
                strict.astype(np.uint8), 8)
            areas = stats[1:, cv2.CC_STAT_AREA]
            idx = int(np.argmax(areas))
            cx, cy = centroids[idx + 1]
            print(f"           target appearance: {int(strict.sum()):6d} px, "
                  f"{n - 1} components, largest {int(areas[idx])} px "
                  f"at ({cx:.0f},{cy:.0f})")
        else:
            score = rg + rb
            k = 15
            smooth = cv2.filter2D(
                score.astype(np.float32), -1,
                np.ones((k, k), np.float32) / (k * k))
            iy, ix = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
            print(f"           target appearance: 0 px at strict thresholds; "
                  f"most red-dominant 15x15 window at ({ix},{iy}) "
                  f"score {float(smooth[iy, ix]):+.1f}")


def _dump_detector_diagnosis(frame: Any, bridge: Any) -> None:
    """Explain a zero-candidate detection by reporting per-discriminant counts.

    A fail-closed stop that does not say why is a poor diagnostic: the operator
    cannot tell "target absent from the view" from "thresholds too strict for
    this lighting".  Writes the frame and mask next to the numbers so the two
    can be compared directly.
    """
    import cv2
    import numpy as np

    out = Path("/tmp/fcpg_trial_state")
    out.mkdir(parents=True, exist_ok=True)
    data = np.asarray(frame.data, dtype=np.uint8)
    cv2.imwrite(str(out / "dryrun_near_view.png"), data)

    channels = [data[:, :, i].astype(np.float64) for i in range(data.shape[2])]
    print()
    print("   detector diagnosis")
    print(f"     frame written    {out / 'dryrun_near_view.png'}")
    print(f"     channel means    "
          + "  ".join(f"c{i}={c.mean():6.1f}" for i, c in enumerate(channels)))
    print(f"     channel maxima   "
          + "  ".join(f"c{i}={c.max():6.0f}" for i, c in enumerate(channels)))

    detector = bridge._policy._detector
    weights = getattr(detector, "_weights", None)
    offsets = getattr(detector, "_offsets", None)
    if not weights or offsets is None:
        print("     (detector exposes no discriminants to report)")
        return

    values = data.astype(np.float64)
    mask = np.ones(data.shape[:2], dtype=bool)
    for i, (w, off) in enumerate(zip(weights, offsets)):
        value = values @ np.asarray(w, dtype=np.float64)
        passed = (value + float(off)) > 0.0
        mask &= passed
        print(f"     discriminant {i}: weights={list(w)} need > {-float(off):+8.1f},"
              f" observed max {float(value.max()):+8.1f},"
              f" {int(passed.sum()):6d} px pass")
    print(f"     all discriminants: {int(mask.sum()):6d} px pass")

    min_area = int(getattr(detector, "_min_area", 0))
    if mask.sum() > 0:
        cv2.imwrite(str(out / "dryrun_near_mask.png"),
                    (mask * 255).astype(np.uint8))
        n, _, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        order = np.argsort(areas)[::-1][:4]
        print(f"     mask written     {out / 'dryrun_near_mask.png'}")
        print(f"     components {n - 1}, largest first (area floor {min_area}):")
        for rank, idx in enumerate(order):
            cx, cy = centroids[idx + 1]
            area = int(areas[idx])
            print(f"       [{rank}] area {area:6d}  centroid ({cx:6.1f},{cy:6.1f})"
                  f"  {'clears' if area >= min_area else 'below'} floor")
        print("     -> pixels pass but no component cleared the area floor:")
        print("        the appearance is present; the floor is what rejected it")
    else:
        print("     -> no pixel passes every discriminant: on this frame the")
        print("        target appearance is absent, not merely small")


def _print_grounding_dry_run(controller: Any, bridge: Any) -> int:
    """Print one grounding trace against the live plan.  Commands no motion.

    Reads the frozen plan produced by bootstrap, runs the verifier, and reports
    the verdict.  When an observation is requested it additionally runs the
    detector and the pre-registration sanity checks on the near-field frame so
    the whole decision path is visible -- but it does NOT register, because a
    dry run that mutated the planner's reference store would not be a dry run.

    Returns a process exit code: 0 when the trace is coherent, 1 otherwise.
    """
    from fcpg.so101_grounding_bridge import (
        NEAR_VIEW,
        frame_from_planning_array,
        proposed_action_from_plan,
    )
    from interfaces.grounding import GroundingVerdict

    plan = controller.plan
    print()
    print("=" * 72)
    print("FCPG GROUNDING DRY RUN -- no motion is commanded, nothing registered")
    print("=" * 72)

    if plan is None:
        print("FAIL: bootstrap produced no plan; cannot review grounding")
        return 1

    # Write every planning view unconditionally. The frames are what any
    # subsequent diagnosis argues about, so capturing them is cheap insurance
    # against having to re-run hardware to look at a picture.
    _dump_planning_frames(plan)

    cfg = bridge._config
    print()
    print("physical inputs")
    print(f"  aperture                 {cfg.aperture_m * 1e3:7.2f} mm   MEASURED")
    print(f"  far-view sigma           {cfg.baseline_sigma_m * 1e3:7.2f} mm   MEASURED")
    print(f"  object width             {cfg.object_width_m * 1e3:7.2f} mm   ASSUMED")
    tolerance = cfg.grasp_tolerance_m()
    if tolerance is None:
        print("FAIL: no grasp slack for this object width; every action is unsafe")
        return 1
    print(f"  -> grasp tolerance       {tolerance * 1e3:7.2f} mm")
    print(f"  safety factor            {cfg.safety_factor:7.2f}")

    action = proposed_action_from_plan(plan, cfg)
    print()
    print("1. LeWM proposal")
    if action is None:
        print("   none: the plan carries no usable target pose.")
        print("   Grounding cannot review this; the runtime would preserve the")
        print("   previous halt rather than invent a verdict.")
        return 1
    xyz = action.target_world_xyz
    print(f"   phase           {action.phase}")
    print(f"   target          ({xyz[0]:+.5f}, {xyz[1]:+.5f}, {xyz[2]:+.5f}) m")
    print(f"   planar error    {action.metadata.get('planar_error_m')}")
    print(f"   round           {action.metadata.get('round_id')}")

    feedback, record = bridge.review(plan)
    if feedback is None or record is None:
        print("FAIL: verifier returned nothing for a reviewable action")
        return 1

    print()
    print("2. FCPG grounding verdict")
    print(f"   verdict         {feedback.verdict.value.upper()}")
    print(f"   sigma           {feedback.uncertainty.position_sigma_m * 1e3:.3f} mm")
    print(f"   safety*sigma    "
          f"{cfg.safety_factor * feedback.uncertainty.position_sigma_m * 1e3:.3f} mm")
    print(f"   tolerance       {tolerance * 1e3:.3f} mm")
    print(f"   calibrated      {feedback.uncertainty.calibration.is_calibrated}"
          "   (False: verdicts are a modelling choice, not a validated confidence)")
    if feedback.uncertainty.components:
        print("   evidence components")
        for name, value in sorted(feedback.uncertainty.components.items()):
            print(f"     {name:26s} {value:.4f}")
    if feedback.reason:
        print(f"   reason          {feedback.reason}")

    print()
    print("3. Observation request")
    if feedback.verdict is GroundingVerdict.PROCEED:
        print("   none. Uncertainty fits inside the grasp tolerance, so the")
        print("   near-field observation is not needed and execution would")
        print("   continue directly. This is the case the deployed distance")
        print("   gate could not express: it halted on geometry alone.")
    elif feedback.verdict is GroundingVerdict.REQUEST_OBSERVATION:
        req = feedback.observation_request
        print(f"   sensor          {req.sensor_role}")
        print(f"   reason          {req.reason}")
        frames = getattr(plan, "frames", None)
        if not isinstance(frames, dict) or NEAR_VIEW not in frames:
            print("   near frame      MISSING -> would halt fail-closed")
            return 1
        frame = frame_from_planning_array(frames[NEAR_VIEW][-1], NEAR_VIEW)
        candidates = bridge._policy._detector.detect(frame)
        print(f"   candidates      {len(candidates)}")
        for i, cand in enumerate(candidates[:4]):
            u, v = cand.centroid_px
            print(f"     [{i}] ({u:6.1f},{v:6.1f})  area {cand.area_px:6d}"
                  f"  solidity {cand.solidity:.3f}  score {cand.score:.4f}")
        if not candidates:
            print("   would halt fail-closed: no candidate in the near view")
            _dump_detector_diagnosis(frame, bridge)
            return 1
        passed, reason, checks = bridge._policy._sanity_check(
            candidates[0], frame, None
        )
        print(f"   sanity check    {'PASS' if passed else 'FAIL'}"
              f"{'' if passed else '  (' + str(reason) + ')'}")
        print(f"   checks          {checks}")
        print("   registration    NOT performed (dry run)")
        if not passed:
            return 1
    else:
        print(f"   none. Verdict is {feedback.verdict.value.upper()};")
        print("   the runtime would stop and report.")

    print()
    print("4. What execute() would do")
    if feedback.verdict is GroundingVerdict.PROCEED:
        print("   continue to _execute_one_locked without observing")
    elif feedback.verdict is GroundingVerdict.REQUEST_OBSERVATION:
        print("   register the near-field reference, replan, then continue")
    else:
        print(f"   stop with reason grounding_halt:{feedback.verdict.value}")

    print()
    print("no operator click required at any point in this trace")
    print("=" * 72)
    print()
    bridge.reset()
    return 0


def _declared_pixel_geometry(
    *, replay: bool
) -> tuple[str, tuple[int, int] | None]:
    """Return the ``(geometry_protocol, source_image_size)`` for one run.

    B1 pixel geometry, declared per ingestion path and never inferred.

    The LIVE hardware path letterboxes (``letterbox_v1``), which is the geometry
    every trained component's cache was actually built with, so the runtime
    finally scores pixels in the same frame the model was fitted in.

    Replay does NOT: it reads cached shadow-session frames that were written by
    ``so101_shadow_session.resize_rgb``, i.e. ``squash_v0``, and those
    recordings declare no protocol at all.  Declaring ``letterbox_v1`` for them
    would reinterpret pixels that were never letterboxed -- and because both
    protocols yield the same 224x224 canvas, nothing downstream could notice.
    ``ReplaySO101IO`` fails closed on exactly that combination, so the split
    here is load-bearing, not cosmetic.

    Lives outside ``main`` so the declaration is reachable by a test: the two
    branches differ only in metadata, and getting them backwards produces a run
    that looks completely normal while every pixel is off by the pad affine.
    """

    if replay:
        return SQUASH_V0, None
    return LETTERBOX_V1, LIVE_SOURCE_IMAGE_SIZE


def _controller_execute_enabled(
    args: argparse.Namespace,
    *,
    replay: bool,
) -> bool:
    """Resolve controller permission after the stack-mode startup checks."""

    if args.v168_mode == "shadow" or args.v170_mode == "shadow":
        return False
    return bool(replay or args.execute_enabled)


def main() -> None:
    faulthandler.enable(all_threads=True)
    args = parse_args()
    replay = args.replay_session is not None
    experiment_session = _write_hardware_experiment_session(args)
    if experiment_session is not None:
        print(f"hardware experiment session: {experiment_session}", flush=True)
    live_anchor_calibration = _effective_live_anchor_calibration(args)
    _run_startup_hardware_preflight(
        args,
        live_anchor_calibration=live_anchor_calibration,
    )
    guarded_command_jump = args.max_command_jump - args.command_guard_margin
    target_profile = (
        args.target_profile
        if args.target_profile is not None
        else args.output / "target_profile"
    )
    # B1 pixel geometry: declared per ingestion path, never inferred.  See
    # `_declared_pixel_geometry` for why the live/replay split is load-bearing.
    geometry_protocol, source_image_size = _declared_pixel_geometry(replay=replay)
    corrector_variant = peek_bridge_corrector_variant(args.bridge_checkpoint)
    disable_global_tip = bool(args.disable_global_tip_locator)
    if corrector_variant == COARSE_FINE_VARIANT and not disable_global_tip:
        disable_global_tip = True
        print(
            "[COARSE_FINE] global tip locator disabled; "
            "cannot chain with coarse_fine",
            flush=True,
        )
    planner = PreparedSO101RealtimePlanner(
        PreparedPlannerConfig(
            bridge_checkpoint=args.bridge_checkpoint,
            model_checkpoint=args.model_checkpoint,
            global_tip_checkpoint=(
                None if disable_global_tip else args.global_tip_checkpoint
            ),
            global_tip_views=tuple(args.global_tip_views),
            calibration=args.calibration,
            secondary_cache=args.secondary_cache,
            device=args.device,
            max_first_command_jump=guarded_command_jump,
            max_prefix_step_jump=guarded_command_jump,
            target_profile=target_profile,
            task_class=args.task_class,
            memory_policy=args.memory_policy,
            require_live_column_calibration=not bool(
                args.disable_live_anchor_calibration
            ),
            control_mode=args.control_mode,
            experiment_method=args.experiment_method,
            ik_arrival_raw_steps=args.ik_arrival_raw_steps,
            max_column_servo_step_m=args.max_planar_step_m,
            max_column_z_step_m=args.max_z_step_m,
            live_action_scale=args.live_action_scale,
            column_extrinsic_max_offset_px=(
                args.column_extrinsic_max_offset_px
            ),
            column_extrinsic_max_residual_px=(
                args.column_extrinsic_max_residual_px
            ),
            peg_clearance_m=args.peg_clearance_m,
            mpc_only_collection=args.mpc_only_collection,
            geometry_protocol=geometry_protocol,
            source_image_size=source_image_size,
            # Explicit rather than defaulted, so all three geometry fields are
            # declared at the same site.  The canonical constant already IS
            # 224, so this changes no behaviour -- it removes the asymmetry
            # where two thirds of the declaration was visible here and the
            # canvas size had to be inferred from a dataclass default.
            planning_image_size=PLANNING_IMAGE_SIZE,
            recursive_observer_shadow=bool(args.recursive_observer_shadow),
            observer_active_excitation=bool(args.observer_active_excitation),
            observer_close_sigma_m=args.observer_close_sigma_m,
            observer_bundle_path=args.observer_bundle,
            observer_drop_3d_path=args.observer_drop_3d_path,
            observer_servo_load=bool(args.observer_servo_load),
            grounding_mode=args.grounding_mode,
            ranking_mode=args.ranking_mode,
            anchor_authority=args.anchor_authority,
            v165_original_fcpg_checkpoint=args.v165_original_fcpg_checkpoint,
            v165_source_checkpoint=args.v165_source_checkpoint,
            v165_tuned_fcpg_checkpoint=args.v165_tuned_fcpg_checkpoint,
            v165_lewm_checkpoint=args.v165_lewm_checkpoint,
            v165_actor_checkpoint=args.v165_actor_checkpoint,
            v165_no_contact_approach=bool(args.v165_no_contact_approach),
            v165_contact_authorization_receipt=args.v165_contact_authorization_receipt,
            v168_mode=args.v168_mode,
            v168_original_fcpg_checkpoint=args.v168_original_fcpg_checkpoint,
            v168_source_checkpoint=args.v168_source_checkpoint,
            v168_lewm_checkpoint=args.v168_lewm_checkpoint,
            v168_lewm_receipt=args.v168_lewm_receipt,
            v168_training_receipt=args.v168_training_receipt,
            v168_actor_checkpoint=args.v168_actor_checkpoint,
            v168_live_execute=bool(args.v168_live_execute),
            v168_supervised_debug=bool(args.v168_supervised_debug),
            v170_mode=args.v170_mode,
            v170_fcpg_checkpoint=args.v170_fcpg_checkpoint,
            v170_lewm_checkpoint=args.v170_lewm_checkpoint,
            v170_actor_checkpoint=args.v170_actor_checkpoint,
            v170_live_execute=bool(args.v170_live_execute),
            wrist_correction_enabled=not bool(args.disable_wrist_correction),
            allow_uncalibrated_coarse_fine_shadow=bool(
                args.allow_uncalibrated_coarse_fine_shadow
            ),
            request_live_execute=bool(args.execute_enabled),
        )
    )

    # Apply explicit trial contact height override if provided
    if args.trial_contact_height_m is not None:
        bridge_derived = planner.contact_height_m
        planner.contact_height_m = float(args.trial_contact_height_m)
        print(
            f"[TRIAL OVERRIDE] contact_height_m: "
            f"{bridge_derived:.6f} m (bridge-derived) → "
            f"{planner.contact_height_m:.6f} m (explicit trial override)"
        )
    else:
        print(
            f"[CONTACT HEIGHT] contact_height_m = {planner.contact_height_m:.6f} m "
            "(bridge-derived, no trial override)"
        )
    print(
        "[RECURSIVE OBSERVER] "
        f"shadow={bool(args.recursive_observer_shadow)} "
        f"close_sigma_m={args.observer_close_sigma_m} "
        f"excitation={bool(args.observer_active_excitation)} "
        f"bundle={args.observer_bundle} "
        f"drop3d={args.observer_drop_3d_path} "
        f"servo_load={bool(args.observer_servo_load)}",
        flush=True,
    )

    if args.experiment_binding is not None:
        planner.model_contract["hardware_experiment_binding"] = _jsonable(
            args.experiment_binding
        )
    # In-session calibrations are saved to <output>/live_anchor_calibration
    # .json; loading only the CLI-provided file silently discarded every one
    # of them on restart. Prefer whichever calibration is newest.
    if live_anchor_calibration is not None:
        _load_live_anchor_calibration(planner, live_anchor_calibration)
        print(f"live anchor calibration loaded from: {live_anchor_calibration}")
    if (
        planner.v165_runtime_adapter is not None
        and planner.v165_runtime_adapter.contact_hardware_authorized
    ):
        # Fail before cameras or robot I/O if any reviewed planner/safety input
        # differs from the no-contact/shadow configuration that earned contact.
        planner.assert_v165_contact_runtime_binding()
    home_state: np.ndarray | None = None
    if args.home_state_report is not None:
        home_state = _load_home_state_report(args.home_state_report)
        # The report snapshots the arm POSE at task start, but its gripper
        # value reflects whatever the calibration pulse left behind (closed).
        # A grasp is only physically possible from an open gripper, and no
        # phase opens it during APPROACH, so returning home must restore the
        # task-ready aperture.
        home_state[5] = max(
            float(home_state[5]), planner.gripper_calibration_open
        )
    if replay:
        sessions = load_paired_session(args.replay_session)
        # Replay is a second, fully independent pixel-ingestion point: it reads
        # cached frames directly and never touches the hardware camera path.
        # Its geometry is therefore declared explicitly from the same planner
        # contract that will score those pixels, instead of being inferred from
        # the 224x224 cached canvas (which is consistent with both protocols).
        # A cached session recorded under one protocol and replayed under
        # another now fails closed here rather than biasing every derived pixel.
        io = ReplaySO101IO(
            sessions,
            start_index=args.replay_index,
            geometry_protocol=planner.config.geometry_protocol,
            source_image_size=planner.config.source_image_size,
            planning_image_size=planner.config.planning_image_size,
        )
    else:
        # Both bounds are measured or already declared, never newly tuned: the
        # joint bound is the same per-step motion limit the safety layer refuses
        # to exceed when *commanding*, and the task-space bound is the jaw span
        # read off the robot model.  The planner's kinematics is reused for IK
        # and settle FK.  MuJoCo GIF preview is skipped in-process so a
        # renderer crash cannot block or kill execute.
        io = HardwareSO101IO(
            args,
            gripper_calibration_closed=planner.gripper_calibration_closed,
            gripper_calibration_open=planner.gripper_calibration_open,
            endpoint_envelope=EndpointEnvelope(
                max_joint_miss=float(args.max_command_jump),
                max_tcp_miss_m=gripper_capture_radius_m(planner.kinematics),
                forward=planner.kinematics.forward,
            ),
        )
        # The planner config is frozen before any camera exists, so the
        # declared raw resolution is a promise until this point -- constructing
        # HardwareSO101IO is the first moment the cameras are open and their
        # ACTUAL mode is observable.  Check it here, from raw frames, before a
        # single plan is scored.  A camera that negotiated a different mode
        # would still produce a 224x224 planning canvas, just with the content
        # rectangle and the pad band in the wrong rows, so this is the only
        # place the mismatch is still visible.
        observed_source_sizes = io.observed_source_image_sizes()
        planner.config.require_source_image_size(observed_source_sizes)
        geometry = planner.config.planning_geometry
        left, top, right, bottom = geometry.content_rect
        print(
            f"live pixel geometry: {geometry.protocol} "
            f"source={'x'.join(str(v) for v in LIVE_SOURCE_IMAGE_SIZE)} "
            f"content_rect=({left},{top},{right},{bottom}) "
            f"pad_rows={geometry.pad_rows} pad_columns={geometry.pad_columns}",
            flush=True,
        )
    writer = RoundArtifactWriter(
        args.output,
        render_mujoco=not args.no_mujoco_preview,
        experiment_binding=args.experiment_binding,
    )

    def calibration_committer(
        calibration_result: dict[str, Any],
    ) -> dict[str, str]:
        return _commit_live_anchor_calibration(
            args.output,
            planner,
            calibration_result,
        )

    grounding_bridge = None
    if args.grounding_feedback_config is not None:
        from fcpg.so101_grounding_bridge import build_bridge_from_config

        overrides: dict[str, Any] = {}
        if args.grounding_object_width_m is not None:
            overrides["object_width_m"] = float(args.grounding_object_width_m)
        grounding_bridge = build_bridge_from_config(
            planner,
            args.grounding_feedback_config,
            log_path=Path(args.output) / "grounding_interactions.json",
            overrides=overrides,
            observe_only=args.grounding_observe_only,
        )
        tolerance = grounding_bridge._config.grasp_tolerance_m()
        print(
            "[grounding] FCPG feedback enabled from "
            f"{args.grounding_feedback_config}\n"
            f"[grounding]   object_width_m={grounding_bridge._config.object_width_m}"
            " (ASSUMED)\n"
            f"[grounding]   grasp tolerance={tolerance:.5f} m\n"
            f"[grounding]   safety_factor={grounding_bridge._config.safety_factor}\n"
            "[grounding]   uncertainty is UNCALIBRATED; verdicts are a "
            "modelling choice, not a validated confidence"
        )
        if args.grounding_observe_only:
            print(
                "[grounding]   OBSERVE-ONLY: detection runs, no reference is "
                "committed, execution halts on every observation request"
            )

    controller = RealtimeMPCController(
        planner,
        io,
        artifact_writer=writer,
        # A stack shadow session never takes the execute path or creates a
        # completed-command history entry, even when its frames come from
        # replay. Candidate mode retains the ordinary controller permission.
        execute_enabled=_controller_execute_enabled(args, replay=replay),
        calibration_enabled=args.execute_enabled
        or args.gripper_calibration_enabled,
        control_hz=args.control_hz,
        max_plan_age_s=args.max_plan_age_s,
        max_state_drift=args.max_state_drift,
        max_target_drift_px=args.max_target_drift_px,
        max_command_jump=guarded_command_jump,
        minimum_expected_motion=args.minimum_expected_motion,
        minimum_observed_motion=args.minimum_observed_motion,
        wrist_registration_distance_m=args.wrist_registration_distance_m,
        home_state=home_state,
        calibration_committer=calibration_committer,
        autonomous_run_enabled=args.bounded_autonomy_enabled,
        max_autonomous_chunks=args.max_autonomous_chunks,
        max_autonomous_duration_s=args.max_autonomous_duration_s,
        grounding_bridge=grounding_bridge,
        task_class=args.task_class,
    )
    if args.prod_v13_observer_shadow:
        from fcpg.offline_pipeline_receipt import git_state
        from fcpg.prod_v13_shadow_observer import ProdV13ShadowObserver

        git = git_state(REPO, require_clean=True)
        controller.prod_v13_shadow = ProdV13ShadowObserver.from_checkpoints(
            checkpoint=args.prod_v13_checkpoint,
            actor_checkpoint=args.prod_v13_actor_checkpoint,
            v12_checkpoint=args.prod_v13_v12_checkpoint,
            fcpg_checkpoint=args.prod_v13_fcpg_checkpoint,
            device=args.device,
            artifact_root=args.output.expanduser().resolve()
            / "prod_v13_observer_shadow",
            git_sha=str(git["head"]),
        )
        print(
            "[Prod-V13] recursive observer attached (NON-ACTUATING, "
            "query/identity reset at service start)",
            flush=True,
        )
    elif args.prod_v13_multirate_shadow:
        from fcpg.offline_pipeline_receipt import git_state
        from fcpg.prod_v13_multirate_shadow import (
            ProdV13MultirateShadowObserver,
        )

        git = git_state(REPO, require_clean=True)
        controller.prod_v13_shadow = (
            ProdV13MultirateShadowObserver.from_checkpoints(
                checkpoint=args.prod_v13_multirate_checkpoint,
                v12_checkpoint=args.prod_v13_multirate_v12_checkpoint,
                cotracker_repo=args.prod_v13_cotracker_repo,
                cotracker_checkpoint=args.prod_v13_cotracker_checkpoint,
                device=args.device,
                artifact_root=args.output.expanduser().resolve()
                / "prod_v13_multirate_shadow",
                git_sha=str(git["head"]),
            )
        )
        print(
            "[Prod-V13] multi-rate PartTokenLeWM observer attached "
            "(NON-ACTUATING, automatic FCPG-H H0 pending)",
            flush=True,
        )
    if args.prod_v14_shadow:
        from fcpg.prod_v14_live_shadow import ProdV14LiveShadow

        controller.prod_v14_shadow = ProdV14LiveShadow.from_checkpoint(
            args.prod_v14_checkpoint,
            device=args.device,
            kinematics=planner.kinematics,
        )
        print(
            "[Prod-V14] dual-view candidate ranker attached "
            f"(step-{controller.prod_v14_shadow.adapter.checkpoint_step}, "
            "POST-PLAN SHADOW, NON-ACTUATING)",
            flush=True,
        )
    if args.prod_v16_live_secant_shadow:
        from fcpg.prod_v16_live_secant import ProdV16LiveSecantShadow

        controller.prod_v16_secant_shadow = ProdV16LiveSecantShadow(
            kinematics=planner.kinematics,
            horizon_raw=planner.horizon_raw,
            command_chunk=planner.config.command_chunk,
        )
        print(
            "[Prod-V16.1] causal dual-view live secant attached "
            "(POST-PLAN SHADOW, NON-ACTUATING)",
            flush=True,
        )
    # V11 shadow runner (trial4+): non-actuating parallel inference.
    # When present it is also the independent LeWM ranking source for v11_6
    # (full checkpoint, not a grafted dynamics head).
    if args.v11_shadow_checkpoint is not None:
        from fcpg.so101_v11_shadow import V11ShadowRunner
        print(
            f"[V11] Loading shadow runner from {args.v11_shadow_checkpoint.name}",
            flush=True,
        )
        controller.v11_shadow = V11ShadowRunner(
            checkpoint_path=args.v11_shadow_checkpoint,
            vit_checkpoint_path=args.v11_vit_checkpoint,
            planner=planner,
            device=args.device,
        )
        print("[V11] Shadow runner attached (NON-ACTUATING)", flush=True)

    if args.v11_control_checkpoint is not None:
        from fcpg.so101_v11_controller import V11Controller
        from fcpg.so101_v11_shadow import TWO_VIEW_TASK_CLASS_FAMILIES
        print(
            f"[V11] Loading production controller from "
            f"{args.v11_control_checkpoint.name}",
            flush=True,
        )
        controller.v11_controller = V11Controller(
            checkpoint_path=args.v11_control_checkpoint,
            vit_checkpoint_path=args.v11_vit_checkpoint,
            planner=planner,
            device=args.device,
        )
        family = getattr(controller.v11_controller.shadow, "family", None)
        rank_uses = "b2_wrist"
        actor_checkpoint = args.v11_control_checkpoint.expanduser().resolve()
        if family in TWO_VIEW_TASK_CLASS_FAMILIES:
            # Dual-wire: independent shadow supplies LeWM when loaded; else the
            # control checkpoint ranks itself (today's single-checkpoint path).
            if controller.v11_shadow is not None:
                lewm_ranker = controller.v11_shadow
                lewm_checkpoint = args.v11_shadow_checkpoint.expanduser().resolve()
                lewm_source = "v11_shadow_checkpoint"
            else:
                lewm_ranker = controller.v11_controller.shadow
                lewm_checkpoint = actor_checkpoint
                lewm_source = "v11_control_checkpoint"
            planner.v11_lewm_ranker = lewm_ranker
            planner.v11_lewm_observation_fn = lambda: controller.observations
            try:
                planner.attach_v11_6_lewm(lewm_ranker.model)
            except (TypeError, ValueError, AttributeError) as exc:
                print(f"[V11] v11_6 CLS rank attach skipped: {exc}", flush=True)
            # Round archive previously omitted the actor path; plan.json only
            # carried dynamics/grounding contracts. Persist both full checkpoints.
            planner.model_contract.update(
                {
                    "v11_actor_checkpoint": str(actor_checkpoint),
                    "v11_lewm_checkpoint": str(lewm_checkpoint),
                    "v11_lewm_checkpoint_source": lewm_source,
                }
            )
            rank_uses = "v11_6_2view"
            print(
                "[V11] RANK_USES=v11_6_2view; B2 wrist CLS is backup only "
                f"lewm_source={lewm_source} "
                f"actor={actor_checkpoint.name} lewm={lewm_checkpoint.name}",
                flush=True,
            )
        else:
            planner.model_contract["v11_actor_checkpoint"] = str(actor_checkpoint)
        print(
            "[V11] Production controller attached "
            "(actor prefix ranked by LeWM+MPC; FCPG grounds actor goal only) "
            f"task_class={args.task_class} family={family} RANK_USES={rank_uses}",
            flush=True,
        )
    server: ThreadingHTTPServer | None = None
    try:
        _attach_dicg_shadow_runner(args, io)
        controller.bootstrap()
        if args.grounding_dry_run:
            if grounding_bridge is None:
                raise SystemExit(
                    "--grounding-dry-run requires --grounding-feedback-config"
                )
            code = _print_grounding_dry_run(controller, grounding_bridge)
            controller.close()
            raise SystemExit(code)
        server = ThreadingHTTPServer(
            (args.host, args.http_port), handler_factory(controller, args.output)
        )
        server.daemon_threads = True
        mode = (
            "replay"
            if replay
            else (
                "hardware-armed"
                if args.execute_enabled
                else (
                    "hardware-calibration-only"
                    if args.gripper_calibration_enabled
                    else "hardware-read-only"
                )
            )
        )
        print(
            f"SO-101 real-time FCPG + LeWM ({mode}): "
            f"http://{args.host}:{args.http_port}",
            flush=True,
        )
        # Launched as a background job from a non-interactive shell, this process
        # inherits SIGINT/SIGQUIT as SIG_IGN and Python keeps that disposition, so
        # `kill -INT` is a no-op and the operator's restart silently does nothing.
        # Re-arm both signals onto the `finally:` path below -- it is the only thing
        # that calls `cam.disconnect()`, and without it the replacement process
        # cannot open the V4L2 devices.  Raise rather than call `server.shutdown()`:
        # the handler runs on the same thread as `serve_forever()`, which `shutdown()`
        # waits on.  Torque survives either way (`disable_torque_on_disconnect=False`).
        def _stop(signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt(signal.Signals(signum).name)

        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)
        try:
            server.serve_forever()
        except KeyboardInterrupt as stop:
            print(
                f"SO-101 real-time server stopped by operator ({stop or 'SIGINT'})",
                flush=True,
            )
    finally:
        if server is not None:
            server.server_close()
        controller.close()
        writer.close()


if __name__ == "__main__":
    main()
