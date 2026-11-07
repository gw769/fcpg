#!/usr/bin/env python3
"""Direct V18.1 dual-view hardware service with no point-grounding planner."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Mapping
import uuid

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for source in (REPO, Path("/home/james/lerobot/src")):
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

from fcpg.prod_v18_1_live import STEP_CAP_DEG, V181LivePolicy
from fcpg.so101_realtime_mpc import validate_command_chunk
from fcpg.so101_realtime_runtime import EndpointEnvelope, LiveObservation
from fcpg.so101_retrieval_mpc import SO101_POSITION_LOWER, SO101_POSITION_UPPER
from scripts.serve_so101_realtime_fcpg_lewm import (
    HardwareSO101IO,
    LIVE_SOURCE_IMAGE_SIZE,
)


DEFAULT_ROOT = Path(
    "/home/james/so101-vda-disk/feedback-corrected-grounding/outputs/lerobot/so101/"
    "prod_v18_1_live_20260902"
)
DEFAULT_CHECKPOINT = DEFAULT_ROOT / "models/full-seed18000.pt"
DEFAULT_DINO_REPOSITORY = Path(
    "/home/james/.cache/torch/hub/facebookresearch_dinov2_main"
)
DEFAULT_DINO_WEIGHTS = Path(
    "/home/james/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
)
DEFAULT_HOME_STATE_REPORT = (
    REPO / "configs/prod_v18_1_task_home.json"
)
# These are the established commandable jaw stops from the production fish
# data (1st/99th percentiles).  They are deliberately much narrower than the
# motor calibration's generic normalized [0, 100] electrical range.  The
# 11.810/19.448 values seen in older plans are phase-classification thresholds,
# not positions to use as jaw stops.
DEFAULT_GRIPPER_CLOSED_POSITION = 1.3140604496002197
DEFAULT_GRIPPER_OPEN_POSITION = 31.603153228759766
# V18.1 was trained to begin this task with the episode-start jaw aperture.
# Closing the jaw at HOME instead aliases the trailing return phase and makes
# the first live input inconsistent with the training clock.
DEFAULT_HOME_GRIPPER_POSITION = 13.600525856018066
DEFAULT_JOINT_LIMIT_GUARD = 3.0
# The fish demonstrations are recorded at 30 Hz and select commands at
# t+[0,3,6,9,12].  The actuator clock is therefore 10 Hz: five commands span
# 0.5 s before the endpoint is reobserved.  A 30 Hz transport silently changes
# both the servo dynamics and every readback sample seen by LeWM.
DEFAULT_CONTROL_HZ = 10.0
# Frozen deployment choice, not a per-step gate.  The aggregate held
# intervention favors actual-history, but same-run replay of the physical task
# start shows that masking state innovation traps the controller in the
# demonstration's repeated two-second idle prefix.  The checkpoint's native
# full path supplies the otherwise-unobservable transition out of that prefix.
V181_DEPLOYMENT_ACTOR_CONDITIONING = "full"
MAX_CONSECUTIVE_ENDPOINT_MISSES = 2
SO101_MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
V181_DEPLOYMENT_POSITION_LOWER = SO101_POSITION_LOWER.copy()
V181_DEPLOYMENT_POSITION_UPPER = SO101_POSITION_UPPER.copy()
V181_DEPLOYMENT_POSITION_LOWER[:5] += DEFAULT_JOINT_LIMIT_GUARD
V181_DEPLOYMENT_POSITION_UPPER[:5] -= DEFAULT_JOINT_LIMIT_GUARD
V181_DEPLOYMENT_POSITION_LOWER[5] = DEFAULT_GRIPPER_CLOSED_POSITION
V181_DEPLOYMENT_POSITION_UPPER[5] = DEFAULT_GRIPPER_OPEN_POSITION
DEFAULT_ROBOT_PORT = (
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E115543-if00"
)
DEFAULT_WRIST_DEVICE = (
    "/dev/v4l/by-id/usb-XHH-260202-H_Integrated_Camera-video-index0"
)
DEFAULT_COLUMN_DEVICE = (
    "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"
)
PLAN_IMAGE_SIZE = (640, 280)
PLAN_JOINT_LABELS = ("J0", "J1", "J2", "J3", "J4", "GRIP")


def _validate_control_hz(value: float) -> float:
    control_hz = float(value)
    if not np.isfinite(control_hz) or control_hz != DEFAULT_CONTROL_HZ:
        raise ValueError(
            f"V18.1 command clock is fixed at {DEFAULT_CONTROL_HZ:g} Hz "
            "to match the fish training stride"
        )
    return control_hz


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(payload), ensure_ascii=False) + "\n")
        stream.flush()


def _dashed_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    *,
    thickness: int = 1,
    dash_px: int = 5,
) -> None:
    """Draw one compact dashed RGB line without adding a plotting dependency."""

    a = np.asarray(start, dtype=np.float32)
    b = np.asarray(end, dtype=np.float32)
    distance = float(np.linalg.norm(b - a))
    if distance <= 0.0:
        return
    pieces = max(1, int(np.ceil(distance / float(dash_px))))
    for index in range(0, pieces, 2):
        left = index / pieces
        right = min(index + 1, pieces) / pieces
        p0 = tuple(np.rint(a + (b - a) * left).astype(int))
        p1 = tuple(np.rint(a + (b - a) * right).astype(int))
        cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)


def _render_next_chunk_plan(
    frame: np.ndarray,
    *,
    view: str,
    plan: Mapping[str, Any] | None,
    predicted_q_path: np.ndarray | None,
) -> np.ndarray:
    """Compose the exact endpoint frame with the next joint-space chunk.

    V18.1 intentionally has no image-space keypoint or Cartesian trajectory.
    The honest live visualization is therefore the same-frame observation plus
    the Actor command trace and LeWM's predicted measured-q trace.
    """

    source_frame = np.asarray(frame)
    if source_frame.shape != (224, 224, 3) or source_frame.dtype != np.uint8:
        raise ValueError("plan frame must be uint8 [224,224,3]")
    if view not in ("wrist", "column"):
        raise ValueError("plan view must be wrist or column")

    width, height = PLAN_IMAGE_SIZE
    canvas = np.full((height, width, 3), (18, 22, 27), dtype=np.uint8)
    canvas[:, :height] = cv2.resize(
        source_frame, (height, height), interpolation=cv2.INTER_LINEAR
    )
    cv2.rectangle(canvas, (0, 0), (height - 1, height - 1), (72, 84, 94), 1)
    cv2.rectangle(canvas, (0, 0), (height, 31), (13, 18, 22), -1)
    cv2.putText(
        canvas,
        f"{view.upper()} / PLAN SOURCE",
        (12, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (236, 242, 244),
        1,
        cv2.LINE_AA,
    )

    chart_left = height
    cv2.putText(
        canvas,
        "NEXT 5 COMMANDS / JOINT SPACE",
        (chart_left + 18, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (236, 242, 244),
        1,
        cv2.LINE_AA,
    )
    if plan is None:
        cv2.putText(
            canvas,
            "WAITING FOR PLAN",
            (chart_left + 18, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 184, 72),
            1,
            cv2.LINE_AA,
        )
        return canvas

    source = np.asarray(plan.get("source_state_deg"), dtype=np.float32)
    commands = np.asarray(plan.get("commands_deg"), dtype=np.float32)
    if (
        source.shape != (6,)
        or commands.shape != (5, 6)
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(commands))
    ):
        raise ValueError("plan must contain finite source [6] and commands [5,6]")
    command_path = np.concatenate((source[None], commands), axis=0)
    prediction = None
    if predicted_q_path is not None:
        candidate = np.asarray(predicted_q_path, dtype=np.float32)
        if candidate.shape == (6, 6) and np.all(np.isfinite(candidate)):
            prediction = candidate

    safe = bool(plan.get("safe"))
    badge_color = (41, 180, 112) if safe else (224, 82, 76)
    badge_text = "LEGAL" if safe else "BLOCKED"
    cv2.rectangle(canvas, (width - 75, 8), (width - 12, 28), badge_color, -1)
    cv2.putText(
        canvas,
        badge_text,
        (width - 68, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (8, 15, 18),
        1,
        cv2.LINE_AA,
    )

    x_points = np.rint(np.linspace(chart_left + 68, width - 68, 6)).astype(int)
    command_color = (74, 221, 255)
    prediction_color = (255, 184, 72)
    neutral = (67, 78, 88)
    text_color = (202, 211, 216)
    lane_top = 46
    lane_stride = 33
    lane_amplitude = 11.0
    for joint, label in enumerate(PLAN_JOINT_LABELS):
        center_y = lane_top + joint * lane_stride
        cv2.line(
            canvas,
            (int(x_points[0]), center_y),
            (int(x_points[-1]), center_y),
            neutral,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (chart_left + 18, center_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            text_color,
            1,
            cv2.LINE_AA,
        )
        visible_delta = np.abs(command_path[:, joint] - source[joint])
        if prediction is not None:
            visible_delta = np.concatenate(
                (visible_delta, np.abs(prediction[:, joint] - source[joint]))
            )
        scale = max(float(np.max(visible_delta)), 0.1)
        command_y = np.rint(
            center_y
            - np.clip((command_path[:, joint] - source[joint]) / scale, -1.0, 1.0)
            * lane_amplitude
        ).astype(int)
        for step in range(5):
            cv2.line(
                canvas,
                (int(x_points[step]), int(command_y[step])),
                (int(x_points[step + 1]), int(command_y[step + 1])),
                command_color,
                2,
                cv2.LINE_AA,
            )
        for x, y in zip(x_points, command_y, strict=True):
            cv2.circle(canvas, (int(x), int(y)), 2, command_color, -1, cv2.LINE_AA)

        if prediction is not None:
            prediction_y = np.rint(
                center_y
                - np.clip((prediction[:, joint] - source[joint]) / scale, -1.0, 1.0)
                * lane_amplitude
            ).astype(int)
            for step in range(5):
                _dashed_line(
                    canvas,
                    (int(x_points[step]), int(prediction_y[step])),
                    (int(x_points[step + 1]), int(prediction_y[step + 1])),
                    prediction_color,
                )
        delta = float(command_path[-1, joint] - source[joint])
        cv2.putText(
            canvas,
            f"{delta:+.2f}",
            (width - 56, center_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            text_color,
            1,
            cv2.LINE_AA,
        )

    for step, x in enumerate(x_points):
        cv2.putText(
            canvas,
            str(step),
            (int(x) - 3, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (137, 149, 157),
            1,
            cv2.LINE_AA,
        )
    cv2.line(
        canvas,
        (chart_left + 18, height - 5),
        (chart_left + 35, height - 5),
        command_color,
        2,
    )
    cv2.putText(
        canvas,
        "ACTOR",
        (chart_left + 40, height - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.28,
        text_color,
        1,
        cv2.LINE_AA,
    )
    _dashed_line(
        canvas,
        (chart_left + 105, height - 5),
        (chart_left + 122, height - 5),
        prediction_color,
    )
    cv2.putText(
        canvas,
        "LEWM Q",
        (chart_left + 127, height - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.28,
        text_color,
        1,
        cv2.LINE_AA,
    )
    return canvas


def _load_home_state(path: Path, *, gripper_position: float) -> np.ndarray:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    state = np.asarray(payload.get("current_state"), dtype=np.float32)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("home-state report must contain one finite current_state")
    state[5] = float(gripper_position)
    if np.any(state < SO101_POSITION_LOWER) or np.any(
        state > SO101_POSITION_UPPER
    ):
        raise ValueError("fixed home state lies outside hard position bounds")
    return state


def _load_motor_calibration_contract(path: Path) -> dict[str, Any]:
    """Load the installed SO-101 raw motor limits used by LeRobot.

    The runtime operates in LeRobot's calibrated normalized coordinates, but
    the physical protection is the per-motor raw range stored in this file and
    written into the servos.  Keep the raw values visible in status so a
    generic [-100,100]/[0,100] box can never masquerade as the calibration.
    """

    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if set(payload) != set(SO101_MOTOR_ORDER):
        raise ValueError("motor calibration must contain exactly the six SO-101 motors")
    motors: dict[str, dict[str, int]] = {}
    for expected_id, name in enumerate(SO101_MOTOR_ORDER, start=1):
        row = payload[name]
        required = ("id", "drive_mode", "homing_offset", "range_min", "range_max")
        if not isinstance(row, Mapping) or any(key not in row for key in required):
            raise ValueError(f"motor calibration row {name!r} is incomplete")
        values = {key: int(row[key]) for key in required}
        if values["id"] != expected_id:
            raise ValueError(f"motor calibration id mismatch for {name!r}")
        if values["range_min"] >= values["range_max"]:
            raise ValueError(f"motor calibration range is invalid for {name!r}")
        motors[name] = values
    return {
        "file": str(source),
        "coordinate_unit": "lerobot_calibrated_normalized_position",
        "raw_motor_limits": motors,
        "normalized_hard_lower": SO101_POSITION_LOWER.copy(),
        "normalized_hard_upper": SO101_POSITION_UPPER.copy(),
        "deployment_lower": V181_DEPLOYMENT_POSITION_LOWER.copy(),
        "deployment_upper": V181_DEPLOYMENT_POSITION_UPPER.copy(),
        "joint_limit_guard": DEFAULT_JOINT_LIMIT_GUARD,
        "gripper_operational_closed": DEFAULT_GRIPPER_CLOSED_POSITION,
        "gripper_operational_open": DEFAULT_GRIPPER_OPEN_POSITION,
    }


def _audit_chunk(
    commands: np.ndarray,
    state: np.ndarray,
    *,
    allow_gripper_recovery: bool = False,
) -> dict[str, Any]:
    chunk = np.asarray(commands, dtype=np.float32)
    current = np.asarray(state, dtype=np.float32)
    reasons: list[str] = []
    if chunk.shape != (5, 6) or not np.all(np.isfinite(chunk)):
        reasons.append("finite_shape")
        return {"safe": False, "reasons": reasons}
    transitions = np.diff(
        np.concatenate((current[None], chunk), axis=0), axis=0
    )
    per_joint = np.max(np.abs(transitions), axis=0)
    if np.any(chunk < SO101_POSITION_LOWER) or np.any(
        chunk > SO101_POSITION_UPPER
    ):
        reasons.append("hard_position_bounds")
    if np.any(per_joint > STEP_CAP_DEG + 1.0e-4):
        reasons.append("v18_1_per_joint_step_cap")

    # The electrical calibration permits gripper positions through 100, but
    # this rig's established task working range is only 1.314..31.603.  A
    # learned proposal must remain inside it.  If the service inherits an
    # already-outside state, only a manual monotonic move back toward the range
    # is permitted; it may never move farther toward the mechanical endpoint.
    gripper = chunk[:, 5]
    current_gripper = float(current[5])
    low = DEFAULT_GRIPPER_CLOSED_POSITION
    high = DEFAULT_GRIPPER_OPEN_POSITION
    eps = 1.0e-4
    current_inside = low - eps <= current_gripper <= high + eps
    if current_inside:
        if np.any(gripper < low - eps) or np.any(gripper > high + eps):
            reasons.append("gripper_operational_bounds")
    elif not allow_gripper_recovery:
        reasons.append("current_gripper_outside_operational_bounds")
    else:
        previous_violation = max(low - current_gripper, current_gripper - high, 0.0)
        violations = np.maximum.reduce(
            (low - gripper, gripper - high, np.zeros_like(gripper))
        )
        if (
            np.any(violations > previous_violation + eps)
            or np.any(np.diff(violations) > eps)
            or float(violations[-1]) >= previous_violation - eps
        ):
            reasons.append("gripper_recovery_not_monotonic")
    generic = validate_command_chunk(
        chunk,
        current,
        max_first_jump=float(STEP_CAP_DEG.max() + 1.0e-4),
        max_step_jump=float(STEP_CAP_DEG.max() + 1.0e-4),
        limit_guard=DEFAULT_JOINT_LIMIT_GUARD,
        outward_tolerance=0.1,
    )
    # `validate_command_chunk` applies the electrical 3-unit guard to all six
    # motors.  For the gripper we have a narrower, independently established
    # operational interval above, so evaluate its outward direction against
    # that interval instead.  The five arm joints retain the electrical guard.
    generic_reasons = [
        reason
        for reason in generic.reasons
        if reason != "moving_outward_at_joint_limit"
    ]
    first_delta = chunk[0] - current
    arm_outward = bool(
        np.any(
            (current[:5] <= SO101_POSITION_LOWER[:5] + DEFAULT_JOINT_LIMIT_GUARD)
            & (first_delta[:5] < -0.1)
        )
        or np.any(
            (current[:5] >= SO101_POSITION_UPPER[:5] - DEFAULT_JOINT_LIMIT_GUARD)
            & (first_delta[:5] > 0.1)
        )
    )
    gripper_outward = bool(
        (
            current_gripper <= low + 0.1
            and float(first_delta[5]) < -0.1
        )
        or (
            current_gripper >= high - 0.1
            and float(first_delta[5]) > 0.1
        )
    )
    if arm_outward or gripper_outward:
        generic_reasons.append("moving_outward_at_joint_limit")
    reasons.extend(generic_reasons)
    return {
        "safe": not reasons,
        "reasons": sorted(set(reasons)),
        "maximum_step_per_joint_deg": per_joint,
    }


class V181HardwareService:
    """Single action source: fresh V18.1 proposals and actual innovations."""

    def __init__(
        self,
        policy: V181LivePolicy,
        io: HardwareSO101IO,
        *,
        output: Path,
        control_hz: float,
        max_auto_chunks: int,
        open_gripper_deg: float,
        home_state: np.ndarray,
        motor_calibration: Mapping[str, Any],
    ) -> None:
        self.policy = policy
        self.io = io
        self.output = output
        self.control_hz = _validate_control_hz(control_hz)
        self.period_s = 1.0 / self.control_hz
        self.max_auto_chunks = int(max_auto_chunks)
        self.open_gripper_deg = float(open_gripper_deg)
        self.operation_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.closed = False
        self.stopped = False
        self.stop_reason: str | None = None
        self.busy: str | None = None
        self.last_error: str | None = None
        self.plan: dict[str, Any] | None = None
        self.last_observation: LiveObservation | None = None
        self.home_state = np.asarray(home_state, dtype=np.float32).copy()
        self.motor_calibration = dict(motor_calibration)
        self.completed_chunks = 0
        self.consecutive_endpoint_misses = 0
        self.receipts: list[dict[str, Any]] = []
        self.events_path = self.output / "service-events.jsonl"

    def _observe_endpoint_outcome(self, *, reached: bool) -> dict[str, Any]:
        """Stop a batch before repeated tracking misses become oscillation."""

        with self.state_lock:
            if reached:
                self.consecutive_endpoint_misses = 0
            else:
                self.consecutive_endpoint_misses += 1
            tripped = (
                self.consecutive_endpoint_misses
                >= MAX_CONSECUTIVE_ENDPOINT_MISSES
            )
            if tripped:
                self.stopped = True
                self.stop_reason = "repeated_endpoint_miss"
        if tripped:
            self.stop_event.set()
        return {
            "consecutive_endpoint_misses": self.consecutive_endpoint_misses,
            "watchdog_stopped": tripped,
        }

    def bootstrap(self) -> dict[str, Any]:
        observed = self.io.observed_source_image_sizes()
        expected = tuple(int(value) for value in LIVE_SOURCE_IMAGE_SIZE)
        if any(tuple(size) != expected for size in observed.values()):
            raise RuntimeError(
                f"camera source size mismatch: {observed}, expected={expected}"
            )
        observation = self.io.observe()
        self._plan_from(observation, replace_pending=False)
        _append_jsonl(
            self.events_path,
            {
                "event": "service_started",
                "wall_time_s": time.time(),
                "service_start_state_deg": observation.state,
                "home_state_deg": self.home_state,
                "motor_calibration": self.motor_calibration,
                "camera_source_sizes": observed,
            },
        )
        return self.status()

    def _acquire(self, operation: str) -> None:
        if not self.operation_lock.acquire(blocking=False):
            raise RuntimeError("another hardware operation is active")
        with self.state_lock:
            self.busy = operation

    def _release(self) -> None:
        with self.state_lock:
            self.busy = None
        self.operation_lock.release()

    def _plan_from(
        self, observation: LiveObservation, *, replace_pending: bool
    ) -> dict[str, Any]:
        proposal = self.policy.propose(
            observation.frames,
            observation.state,
            replace_pending=replace_pending,
        )
        commands = np.asarray(proposal["commands_deg"], dtype=np.float32)
        audit = _audit_chunk(commands, observation.state)
        plan = {
            "round_id": f"v181-{time.time_ns()}-{uuid.uuid4().hex[:8]}",
            "created_monotonic_s": time.monotonic(),
            "source_timestamp_s": observation.timestamp_s,
            "source_state_deg": observation.state.copy(),
            "commands_deg": commands,
            "safe": bool(audit["safe"]),
            "reasons": audit["reasons"],
            "maximum_step_per_joint_deg": audit.get(
                "maximum_step_per_joint_deg"
            ),
            "inference_s": proposal["inference_s"],
            "memory_chunks": proposal["memory_chunks"],
        }
        with self.state_lock:
            self.last_observation = observation
            self.plan = plan
            self.last_error = None
        return plan

    def replan(self) -> dict[str, Any]:
        self._acquire("reobserve_and_plan")
        try:
            self.stop_event.clear()
            if self.stopped:
                self.policy.reset()
            observation = self.io.observe()
            self._plan_from(observation, replace_pending=True)
            with self.state_lock:
                self.stopped = False
                self.stop_reason = None
                self.consecutive_endpoint_misses = 0
            return self.status()
        except BaseException as exc:
            with self.state_lock:
                self.last_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._release()

    def _fresh_plan_for_transport(self) -> tuple[LiveObservation, dict[str, Any]]:
        live = self.io.observe()
        plan = self._plan_from(live, replace_pending=True)
        if not bool(plan["safe"]):
            raise RuntimeError(
                "V18.1 command rejected: " + ", ".join(plan["reasons"])
            )
        return live, plan

    def _execute_one(self) -> dict[str, Any]:
        live, plan = self._fresh_plan_for_transport()
        commands = np.asarray(plan["commands_deg"], dtype=np.float32)
        started = time.perf_counter()
        try:
            result = self.io.execute_chunk(
                commands,
                period_s=self.period_s,
                stop_event=self.stop_event,
            )
        except BaseException as exc:
            record = {
                "event": "execution_error",
                "wall_time_s": time.time(),
                "round_id": plan["round_id"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "command_journal": self.io.last_command_journal,
            }
            _append_jsonl(self.events_path, record)
            with self.state_lock:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.stopped = True
            self.stop_event.set()
            raise

        measured_path = np.concatenate(
            (
                live.state[None],
                np.stack([row.state for row in result.observations]),
            ),
            axis=0,
        ).astype(np.float32)
        innovation: dict[str, Any] | None = None
        innovation_error: str | None = None
        try:
            innovation = self.policy.complete(
                result.observations[-1].frames,
                measured_path,
                delta_t_s=float(result.duration_s),
            )
        except Exception as exc:
            # Motion is complete and must never be replayed. Start the next
            # observation as H0 if transition bookkeeping cannot be trusted.
            innovation_error = f"{type(exc).__name__}: {exc}"
            self.policy.reset()
        tracking = self._observe_endpoint_outcome(
            reached=result.reached_commanded_endpoint
        )
        receipt = {
            "event": "chunk_completed",
            "wall_time_s": time.time(),
            "round_id": plan["round_id"],
            "requested_deg": result.requested,
            "sent_deg": result.sent,
            "measured_q_path_deg": measured_path,
            "after_state_deg": result.observations[-1].state,
            "duration_s": result.duration_s,
            "settlement": (
                None
                if result.settlement is None
                else {
                    "outcome": result.settlement.outcome,
                    "joint_residual": result.settlement.joint_residual,
                    "actual_state": result.settlement.actual_state,
                }
            ),
            "innovation": innovation,
            **tracking,
            "command_journal": result.command_journal,
        }
        if innovation_error is not None:
            receipt["innovation_failure"] = innovation_error
        _append_jsonl(self.events_path, receipt)
        with self.state_lock:
            self.completed_chunks += 1
            self.receipts.append(receipt)
            self.last_observation = result.observations[-1]
            self.last_error = innovation_error
        return receipt

    def execute(self, *, round_id: str, max_chunks: int) -> dict[str, Any]:
        if not isinstance(max_chunks, int) or isinstance(max_chunks, bool):
            raise ValueError("max_chunks must be an integer")
        if not 1 <= max_chunks <= self.max_auto_chunks:
            raise ValueError(
                f"max_chunks must lie in [1,{self.max_auto_chunks}]"
            )
        with self.state_lock:
            active = self.plan
            if self.stopped:
                raise RuntimeError("motion is stopped; reobserve and plan to resume")
            if active is None or round_id != active["round_id"]:
                raise RuntimeError("stale planning round")
        self._acquire("execute")
        completed = 0
        try:
            self.stop_event.clear()
            for _ in range(max_chunks):
                if self.stop_event.is_set():
                    break
                self._execute_one()
                completed += 1
            if not self.stop_event.is_set():
                latest = self.last_observation
                if latest is None:
                    latest = self.io.observe()
                self._plan_from(latest, replace_pending=False)
            return self.status()
        finally:
            _append_jsonl(
                self.events_path,
                {
                    "event": "execute_batch_finished",
                    "wall_time_s": time.time(),
                    "requested_chunks": max_chunks,
                    "completed_chunks": completed,
                    "stopped": self.stop_event.is_set(),
                },
            )
            self._release()

    @staticmethod
    def _bounded_commands(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        previous = np.asarray(current, dtype=np.float32).copy()
        goal = np.asarray(target, dtype=np.float32)
        rows: list[np.ndarray] = []
        for _ in range(5):
            delta = np.clip(goal - previous, -STEP_CAP_DEG, STEP_CAP_DEG)
            previous = previous + delta
            rows.append(previous.copy())
        return np.stack(rows)

    def _run_manual_target(
        self,
        target: np.ndarray,
        *,
        label: str,
        allow_gripper_recovery: bool = False,
    ) -> None:
        goal = np.asarray(target, dtype=np.float32)
        tolerance = float(self.io.settle_position_tolerance)
        final_residual = float("inf")
        for _ in range(20):
            if self.stop_event.is_set():
                raise RuntimeError(f"{label} stopped by operator")
            live = self.io.observe()
            final_residual = float(np.max(np.abs(live.state - goal)))
            if final_residual <= tolerance:
                return
            # Never defeat the endpoint guard to reach a nominal HOME.  If one
            # component is already inside the 3-unit guard band, hold that
            # component while the other joints return home.
            bounded_goal = goal.copy()
            near_lower = live.state <= SO101_POSITION_LOWER + DEFAULT_JOINT_LIMIT_GUARD
            near_upper = live.state >= SO101_POSITION_UPPER - DEFAULT_JOINT_LIMIT_GUARD
            bounded_goal[near_lower & (bounded_goal < live.state)] = live.state[
                near_lower & (bounded_goal < live.state)
            ]
            bounded_goal[near_upper & (bounded_goal > live.state)] = live.state[
                near_upper & (bounded_goal > live.state)
            ]
            commands = self._bounded_commands(live.state, bounded_goal)
            audit = _audit_chunk(
                commands,
                live.state,
                allow_gripper_recovery=allow_gripper_recovery,
            )
            if not audit["safe"]:
                raise RuntimeError(
                    f"{label} command rejected: " + ", ".join(audit["reasons"])
                )
            self.io.execute_chunk(
                commands,
                period_s=self.period_s,
                stop_event=self.stop_event,
            )
        raise RuntimeError(
            f"{label} did not converge inside its fixed budget: "
            f"residual={final_residual:.3f} deg, tolerance={tolerance:.3f} deg"
        )

    def home(self) -> dict[str, Any]:
        if self.home_state is None:
            raise RuntimeError("startup home state is unavailable")
        self._acquire("home")
        try:
            self.stop_event.clear()
            self._run_manual_target(
                self.home_state,
                label="home",
                allow_gripper_recovery=True,
            )
            self.policy.reset()
            observation = self.io.observe()
            self._plan_from(observation, replace_pending=False)
            with self.state_lock:
                self.stopped = False
                self.stop_reason = None
                self.consecutive_endpoint_misses = 0
            return self.status()
        finally:
            self._release()

    def open_gripper(self) -> dict[str, Any]:
        self._acquire("open_gripper")
        try:
            self.stop_event.clear()
            live = self.io.observe()
            target = live.state.copy()
            target[5] = self.open_gripper_deg
            self._run_manual_target(
                target,
                label="open_gripper",
                allow_gripper_recovery=True,
            )
            self.policy.reset()
            observation = self.io.observe()
            self._plan_from(observation, replace_pending=False)
            with self.state_lock:
                self.stopped = False
                self.stop_reason = None
                self.consecutive_endpoint_misses = 0
            return self.status()
        finally:
            self._release()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        with self.state_lock:
            self.stopped = True
            self.stop_reason = "operator_stop"
        _append_jsonl(
            self.events_path,
            {"event": "operator_stop", "wall_time_s": time.time()},
        )
        return self.status()

    def next_chunk_plan_image(self, view: str) -> np.ndarray:
        """Return a same-round endpoint plus Actor/LeWM joint traces."""

        if view not in ("wrist", "column"):
            raise ValueError("plan view must be wrist or column")
        with self.state_lock:
            observation = self.last_observation
            plan = self.plan
            if observation is None:
                raise RuntimeError("no endpoint observation is available")
            frame = np.asarray(observation.frames[view]).copy()
            plan_copy = None if plan is None else dict(plan)
        proposal = self.policy.last_proposal or {}
        return _render_next_chunk_plan(
            frame,
            view=view,
            plan=plan_copy,
            predicted_q_path=proposal.get("predicted_q_path_deg"),
        )

    def status(self) -> dict[str, Any]:
        with self.state_lock:
            payload = {
                "version": "18.1",
                "mode": "hardware",
                "action_source": self.policy.action_source,
                "point_coordinates_present": False,
                "phase_state_present": False,
                "execute_enabled": True,
                "stopped": self.stopped,
                "stop_reason": self.stop_reason,
                "busy": self.busy,
                "plan": self.plan,
                "home_state_deg": self.home_state,
                "position_unit": "lerobot_calibrated_normalized_position",
                "command_hz": 1.0 / self.period_s,
                "nominal_chunk_duration_s": 5.0 * self.period_s,
                "motor_calibration": self.motor_calibration,
                "completed_chunks": self.completed_chunks,
                "consecutive_endpoint_misses": self.consecutive_endpoint_misses,
                "max_auto_chunks": self.max_auto_chunks,
                "latest_receipt": self.receipts[-1] if self.receipts else None,
                "model": self.policy.status(),
                "functions": {
                    "execute_5_steps": True,
                    "automatic_run": True,
                    "reobserve_and_plan": True,
                    "home": self.home_state is not None,
                    "open_gripper": True,
                    "stop": True,
                },
            }
            if self.last_error is not None:
                payload["runtime_failure"] = self.last_error
            return _jsonable(payload)

    def close(self) -> None:
        if self.closed:
            return
        self.stop_event.set()
        self.io.close()
        self.closed = True


HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-101 V18.1 Control</title>
<style>
:root{color-scheme:light;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#eef0f2;color:#17191c}
*{box-sizing:border-box}body{margin:0}header{height:58px;background:#181b1f;color:#fff;display:flex;align-items:center;justify-content:space-between;padding:0 20px;position:sticky;top:0;z-index:3}
h1{font-size:18px;margin:0;font-weight:650}#badge{font:700 13px ui-monospace,monospace;padding:6px 10px;border:1px solid #59616b;border-radius:4px;background:#5a626b;min-width:78px;text-align:center}
main{max-width:1320px;margin:auto;padding:14px 18px 102px}.statusbar{display:grid;grid-template-columns:repeat(7,minmax(100px,1fr));gap:1px;background:#c8cdd2;border:1px solid #c8cdd2}.metric{background:#fff;padding:9px 11px;min-height:57px}.metric span{display:block;font-size:11px;color:#616970;text-transform:uppercase}.metric strong{display:block;font-size:15px;overflow-wrap:anywhere;margin-top:3px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}.panel{background:#fff;border:1px solid #c8cdd2;border-radius:6px;overflow:hidden}.panel h2{font-size:13px;margin:0;padding:9px 11px;border-bottom:1px solid #d7dadd}.panel.wide{grid-column:1/-1}.live-panel{border-color:#31835d}.live-panel h2{display:flex;align-items:center;justify-content:space-between}.live-tag{background:#17653a;color:#fff;border-radius:3px;padding:2px 6px;font:700 10px ui-monospace,monospace}.camera{width:100%;display:block;aspect-ratio:4/3;object-fit:contain;background:#151719}.plan-camera{width:100%;display:block;aspect-ratio:16/7;object-fit:contain;background:#12161b}.camera-meta{display:flex;justify-content:space-between;padding:7px 11px;border-top:1px solid #d7dadd;color:#616970;font:11px ui-monospace,monospace}
.request-error{display:none;grid-column:1/-1;border:1px solid #b42318;background:#fff4f2;color:#731b14;padding:10px}.request-error.visible{display:block}.request-error-head{display:flex;align-items:center;gap:10px}.request-error pre{white-space:pre-wrap;overflow:auto;user-select:text;cursor:text;margin:8px 0 0;max-height:180px}.copy{height:30px;background:#731b14;color:#fff;margin-left:auto;padding:0 11px;font-size:12px}
.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:#d7dadd}.summary-item{background:#fff;padding:12px}.summary-item span{display:block;color:#697078;font-size:11px;text-transform:uppercase}.summary-item strong{display:block;margin-top:4px;font:650 16px ui-monospace,monospace}.summary-item.good strong{color:#17653a}.summary-item.warn strong{color:#9a4b08}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:right;padding:8px 10px;border-bottom:1px solid #e1e4e7;font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}thead{background:#f6f7f8;color:#59616b}.state-row{background:#eef6ff;font-weight:650}.step-ok{color:#17653a;font-weight:700}.step-bad{color:#b42318;font-weight:700}
.runtime-title{display:flex;align-items:center;gap:8px}.runtime-title .copy{background:#235c82}.log{font:12px ui-monospace,monospace;white-space:pre-wrap;max-height:260px;overflow:auto;padding:10px;margin:0;background:#111418;color:#d8e4df;user-select:text;cursor:text}.details summary{padding:10px 12px;cursor:pointer;font-size:13px;font-weight:650;border-bottom:1px solid #d7dadd}.plan-note{padding:9px 11px;color:#59616b;font:12px ui-monospace,monospace;border-top:1px solid #e1e4e7}
.controls{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid #aeb4ba;padding:12px 18px;display:flex;justify-content:center;gap:10px;align-items:center;z-index:4;box-shadow:0 -4px 14px rgba(24,27,31,.08)}button{height:44px;border:0;border-radius:5px;padding:0 18px;font-size:14px;font-weight:650;cursor:pointer;white-space:nowrap}.execute{background:#17653a;color:#fff}.auto{background:#0f766e;color:#fff}.replan{background:#235c82;color:#fff}.home{background:#6b4f1d;color:#fff}.gripper{background:#5d4777;color:#fff}.stop{background:#b42318;color:#fff;min-width:88px}button:disabled{opacity:.4;cursor:not-allowed}.busy{min-width:150px;color:#4d555c;font-size:13px}
@media(max-width:900px){.statusbar{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.panel.wide,.request-error{grid-column:auto}.summary{grid-template-columns:1fr 1fr}.controls{justify-content:flex-start;overflow-x:auto}}
</style></head><body>
<header><h1>SO-101 · V18.1 双视角无指尖闭环</h1><div id="badge">LOADING</div></header>
<main>
<section class="statusbar">
<div class="metric"><span>Mode</span><strong id="mode">-</strong></div>
<div class="metric"><span>Controller</span><strong id="controller">-</strong></div>
<div class="metric"><span>Round</span><strong id="round">-</strong></div>
<div class="metric"><span>Plan</span><strong id="planState">-</strong></div>
<div class="metric"><span>Operation</span><strong id="operation">-</strong></div>
<div class="metric"><span>Memory</span><strong id="memory">-</strong></div>
<div class="metric"><span>Completed</span><strong id="completed">-</strong></div>
</section>
<section class="grid">
<section class="request-error" id="requestError"><div class="request-error-head"><strong id="requestErrorTitle">请求失败</strong><button class="copy" id="copyError">复制错误</button></div><pre id="requestErrorText" tabindex="0"></pre></section>
<article class="panel live-panel"><h2>Wrist 实时输入 <span class="live-tag">LIVE</span></h2><img class="camera" id="liveWrist" alt="Wrist live camera"><div class="camera-meta"><span>ACTUAL OBSERVATION</span><span>不显示指尖点</span></div></article>
<article class="panel live-panel"><h2>Column 实时输入 <span class="live-tag">LIVE</span></h2><img class="camera" id="liveColumn" alt="Column live camera"><div class="camera-meta"><span>ACTUAL OBSERVATION</span><span>不显示指尖点</span></div></article>
<article class="panel"><h2>Wrist · 下一动作块</h2><img class="plan-camera" id="planWrist" alt="Wrist next five-command plan"><div class="camera-meta"><span>实线 ACTOR COMMAND</span><span>虚线 LEWM READBACK</span></div></article>
<article class="panel"><h2>Column · 下一动作块</h2><img class="plan-camera" id="planColumn" alt="Column next five-command plan"><div class="camera-meta"><span>实线 ACTOR COMMAND</span><span>虚线 LEWM READBACK</span></div></article>
<article class="panel wide"><h2>V18.1 当前闭环状态</h2><div class="summary">
<div class="summary-item"><span>推理延迟</span><strong id="latency">-</strong></div>
<div class="summary-item"><span>最大动作步长</span><strong id="maxStep">-</strong></div>
<div class="summary-item"><span>视觉 innovation RMS</span><strong id="visualInnovation">-</strong></div>
<div class="summary-item"><span>状态 innovation RMS</span><strong id="stateInnovation">-</strong></div>
</div></article>
<article class="panel wide"><h2>当前状态与五步命令（标定归一化位置）</h2><table><thead><tr><th>Step</th><th>Joint 0</th><th>Joint 1</th><th>Joint 2</th><th>Joint 3</th><th>Joint 4</th><th>Gripper</th><th>Max Δ</th></tr></thead><tbody id="commands"></tbody></table><div class="plan-note" id="planNote">等待规划</div></article>
<article class="panel wide details"><details><summary class="runtime-title">Runtime 技术详情 <button class="copy" id="copyDiagnostics">复制完整诊断</button></summary><pre class="log" id="log" tabindex="0"></pre></details></article>
</section></main>
<div class="controls">
<button class="execute" id="execute">执行 5 步</button>
<button class="auto" id="autoExecute">自动运行任务</button>
<button class="replan" id="replan">重新观测并规划</button>
<button class="gripper" id="openGripper">张开夹爪</button>
<button class="home" id="home">回到 HOME</button>
<button class="stop" id="stop">停止</button>
<span class="busy" id="busy"></span>
</div>
<script>
let latest=null,requestInFlight=false,refreshInFlight=false,stopInFlight=false,refreshTimer=null,renderedPlanRound=null;
const $=id=>document.getElementById(id);
function fmt(value,digits=2){const n=Number(value);return Number.isFinite(n)?n.toFixed(digits):'-'}
function shortRound(value){if(!value)return '-';const text=String(value);return text.length>20?text.slice(0,11)+'…'+text.slice(-6):text}
function maxAbs(values){if(!Array.isArray(values)||!values.length)return null;const finite=values.map(Number).filter(Number.isFinite);return finite.length?Math.max(...finite.map(Math.abs)):null}
function errorTitle(message){const text=String(message||'');if(text.includes('status packet'))return '机械臂通信中断';if(text.includes('Timed out waiting for frame'))return '相机取帧超时';if(text.includes('stale planning'))return '规划已过期，请重新观测';if(text.includes('did not settle'))return '机械臂未在时限内稳定';if(text.includes('another hardware'))return '已有真机操作正在进行';return '请求失败'}
function showRequestError(message){$('requestErrorTitle').textContent=errorTitle(message);$('requestErrorText').textContent=String(message);$('requestError').classList.add('visible')}
function clearRequestError(){$('requestErrorText').textContent='';$('requestError').classList.remove('visible')}
function renderCommands(plan){const body=$('commands');body.replaceChildren();if(!plan||!Array.isArray(plan.commands_deg)){body.innerHTML='<tr><td colspan="8">尚无有效规划</td></tr>';$('planNote').textContent='等待重新观测并规划';return}const source=Array.isArray(plan.source_state_deg)?plan.source_state_deg:null;const rows=[...(source?[{name:'当前',values:source,state:true}]:[]),...plan.commands_deg.map((values,index)=>({name:String(index+1),values,state:false}))];let previous=source;for(const row of rows){const tr=document.createElement('tr');if(row.state)tr.className='state-row';const delta=previous&&!row.state?Math.max(...row.values.map((v,index)=>Math.abs(Number(v)-Number(previous[index])))):null;const cells=[row.name,...row.values.map(value=>fmt(value,2)),delta==null?'-':fmt(delta,2)+' u'];cells.forEach((value,index)=>{const td=document.createElement('td');td.textContent=value;if(index===7&&!row.state)td.className=delta<=maxAbs(plan.maximum_step_per_joint_deg)+1e-4?'step-ok':'step-bad';tr.appendChild(td)});body.appendChild(tr);previous=row.values}$('planNote').textContent=plan.safe?'五步命令已通过电机标定边界、任务夹爪边界、步长与 projector 检查':'禁止执行：'+((plan.reasons||[]).join(', ')||'规划不合法')}
function controls(status){const busy=requestInFlight||Boolean(status&&status.busy);const plan=status&&status.plan;const ready=Boolean(status&&status.execute_enabled&&!status.stopped&&plan&&plan.safe&&!busy);$('execute').disabled=!ready;$('autoExecute').disabled=!ready;$('replan').disabled=busy;$('home').disabled=busy||!status?.functions?.home;$('openGripper').disabled=busy||!status?.functions?.open_gripper;$('stop').disabled=stopInFlight}
function refreshPlanImages(plan){const round=plan&&plan.round_id;if(!round||round===renderedPlanRound)return;renderedPlanRound=round;const stamp=encodeURIComponent(round);$('planWrist').src='/api/plan-image/wrist.jpg?round='+stamp;$('planColumn').src='/api/plan-image/column.jpg?round='+stamp}
function render(status){latest=status;const plan=status.plan||null;const model=status.model||{};const completion=model.last_completion||(status.latest_receipt&&status.latest_receipt.innovation)||null;const proposal=model.last_proposal||null;const error=status.runtime_failure||status.request_error||null;const ready=Boolean(status.execute_enabled&&!status.stopped&&plan&&plan.safe);$('mode').textContent=(status.mode||'-')+' / '+(status.execute_enabled?'armed':'read-only');$('controller').textContent='V18.1 / TIP-FREE';$('round').textContent=shortRound(plan&&plan.round_id);$('planState').textContent=plan?(plan.safe?'LEGAL':'BLOCKED'):'-';$('operation').textContent=status.busy||'-';$('memory').textContent=(model.memory_chunks??plan?.memory_chunks??0)+' / 3';$('completed').textContent=status.completed_chunks??0;$('latency').textContent=plan?fmt(plan.inference_s*1000,0)+' ms':'-';$('maxStep').textContent=plan?fmt(maxAbs(plan.maximum_step_per_joint_deg),2)+' u':'-';$('visualInnovation').textContent=completion?fmt(completion.visual_innovation_rms,4):'-';$('stateInnovation').textContent=completion?fmt(completion.state_innovation_rms_deg,3)+' u':'-';$('badge').textContent=error?'ERROR':(status.busy?String(status.busy).toUpperCase():(status.stopped?'STOPPED':(ready?'READY':'BLOCKED')));$('badge').style.background=error?'#8f1d18':(status.stopped?'#6b4f1d':(status.busy?'#235c82':(ready?'#17653a':'#5a626b')));renderCommands(plan);refreshPlanImages(plan);const diagnostics={action_source:status.action_source,point_coordinates_present:status.point_coordinates_present,phase_state_present:status.phase_state_present,position_unit:status.position_unit,motor_calibration:status.motor_calibration,plan,latest_receipt:status.latest_receipt,model:{version:model.version,tip_free:model.tip_free,view_order:model.view_order,training_phase:model.training_phase,seed:model.seed,deployment_action_lower:model.deployment_action_lower,deployment_action_upper:model.deployment_action_upper,pending_chunk:model.pending_chunk,memory_chunks:model.memory_chunks,last_proposal:proposal,last_completion:model.last_completion},functions:status.functions};if(error)diagnostics.runtime_failure=String(error);$('log').textContent=JSON.stringify(diagnostics,null,2);controls(status);if(error)showRequestError(error)}
async function readJson(response){const text=await response.text();try{return JSON.parse(text)}catch(e){throw new Error('HTTP '+response.status+': '+text.slice(0,500))}}
function scheduleRefresh(delay=1200){clearTimeout(refreshTimer);refreshTimer=setTimeout(refresh,delay)}
async function refresh(){if(requestInFlight||refreshInFlight){scheduleRefresh();return}refreshInFlight=true;try{const response=await fetch('/api/status',{cache:'no-store'});const data=await readJson(response);if(!response.ok)throw new Error(data.request_error||('HTTP '+response.status));render(data)}catch(error){showRequestError(String(error));$('badge').textContent='DISCONNECTED';$('badge').style.background='#8f1d18'}finally{refreshInFlight=false;scheduleRefresh()}}
async function post(path,payload={},label='处理中'){if(requestInFlight){showRequestError('已有真机操作正在进行');return}requestInFlight=true;clearTimeout(refreshTimer);$('busy').textContent=label;clearRequestError();controls(latest);try{const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await readJson(response);render(data);if(!response.ok)showRequestError(data.request_error||data.error||('HTTP '+response.status))}catch(error){showRequestError(String(error))}finally{requestInFlight=false;$('busy').textContent='';controls(latest);scheduleRefresh(250)}}
async function stopNow(){if(stopInFlight)return;stopInFlight=true;$('busy').textContent='正在停止';controls(latest);try{const response=await fetch('/api/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});const data=await readJson(response);render(data);if(!response.ok)showRequestError(data.request_error||data.error||('HTTP '+response.status))}catch(error){showRequestError(String(error))}finally{stopInFlight=false;$('busy').textContent='';controls(latest);scheduleRefresh(250)}}
function execute(chunks){if(!latest?.plan)return showRequestError('请先重新观测并规划');const count=Number(chunks);if(!Number.isInteger(count)||count<1||count>16)return showRequestError('自动运行轮数必须在 1 到 16 之间');post('/api/execute',{round_id:latest.plan.round_id,max_chunks:count},count===1?'正在执行 5 步并重新观测':'正在闭环运行任务')}
function attachLiveStream(id,path){const element=$(id);let retry=null;const connect=()=>{clearTimeout(retry);element.src=path+'?t='+Date.now()};element.onerror=()=>{clearTimeout(retry);retry=setTimeout(connect,1000)};connect()}
async function copyText(value,label){try{await navigator.clipboard.writeText(value)}catch(error){const area=document.createElement('textarea');area.value=value;area.style.position='fixed';area.style.opacity='0';document.body.appendChild(area);area.select();document.execCommand('copy');area.remove()}$('busy').textContent=label;setTimeout(()=>$('busy').textContent='',1200)}
$('execute').onclick=()=>execute(1);$('autoExecute').onclick=()=>execute(Number(latest?.max_auto_chunks||16));$('replan').onclick=()=>post('/api/plan',{},'正在读取双相机并规划');$('home').onclick=()=>{if(confirm('缓慢回到固定任务 HOME？'))post('/api/home',{},'正在返回 HOME')};$('openGripper').onclick=()=>{if(confirm('确认张开夹爪？'))post('/api/open-gripper',{},'正在张开夹爪')};$('stop').onclick=stopNow;$('copyError').onclick=()=>copyText($('requestErrorText').textContent,'错误已复制');$('copyDiagnostics').onclick=event=>{event.preventDefault();copyText($('log').textContent,'完整诊断已复制')};attachLiveStream('liveWrist','/stream/wrist.mjpg');attachLiveStream('liveColumn','/stream/column.mjpg');refresh();
</script></body></html>"""


def handler_factory(controller: V181HardwareService):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(_jsonable(payload), ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("request body must be an object")
            return value

        @staticmethod
        def _jpeg(view: str) -> bytes:
            frame = controller.io.preview_frame(view)
            ok, encoded = cv2.imencode(
                ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            return encoded.tobytes()

        @staticmethod
        def _plan_jpeg(view: str) -> bytes:
            frame = controller.next_chunk_plan_image(view)
            ok, encoded = cv2.imencode(
                ".jpg",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            if not ok:
                raise RuntimeError("plan JPEG encoding failed")
            return encoded.tobytes()

        def do_GET(self) -> None:  # noqa: N802
            request_path = self.path.partition("?")[0]
            if request_path == "/":
                body = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if request_path == "/api/status":
                self._json(controller.status())
                return
            for view in ("wrist", "column"):
                if request_path == f"/api/plan-image/{view}.jpg":
                    body = self._plan_jpeg(view)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if request_path == f"/api/snapshot/{view}":
                    body = self._jpeg(view)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if request_path == f"/stream/{view}.mjpg":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                    )
                    self.end_headers()
                    try:
                        while True:
                            body = self._jpeg(view)
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                            self.wfile.write(
                                f"Content-Length: {len(body)}\r\n\r\n".encode()
                            )
                            self.wfile.write(body + b"\r\n")
                            self.wfile.flush()
                            time.sleep(0.2)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            try:
                body = self._body()
                if self.path == "/api/plan":
                    result = controller.replan()
                elif self.path == "/api/execute":
                    result = controller.execute(
                        round_id=str(body.get("round_id", "")),
                        max_chunks=int(body.get("max_chunks", 1)),
                    )
                elif self.path == "/api/home":
                    result = controller.home()
                elif self.path == "/api/open-gripper":
                    result = controller.open_gripper()
                elif self.path == "/api/stop":
                    result = controller.stop()
                else:
                    self.send_error(404)
                    return
                self._json(result)
            except Exception as exc:
                payload = controller.status()
                payload["request_error"] = f"{type(exc).__name__}: {exc}"
                self._json(payload, status=409)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dino-repository", type=Path, default=DEFAULT_DINO_REPOSITORY)
    parser.add_argument("--dino-weights", type=Path, default=DEFAULT_DINO_WEIGHTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "service")
    parser.add_argument("--robot-port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--robot-id", default="myfollower01")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--wrist-device", default=DEFAULT_WRIST_DEVICE)
    parser.add_argument("--column-device", default=DEFAULT_COLUMN_DEVICE)
    parser.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument("--settle-timeout-s", type=float, default=20.0)
    parser.add_argument("--settle-position-tolerance", type=float, default=2.0)
    parser.add_argument("--settle-step-tolerance", type=float, default=0.25)
    parser.add_argument("--settle-samples", type=int, default=3)
    parser.add_argument(
        "--open-gripper-deg", type=float, default=DEFAULT_GRIPPER_OPEN_POSITION
    )
    parser.add_argument(
        "--home-state-report", type=Path, default=DEFAULT_HOME_STATE_REPORT
    )
    parser.add_argument(
        "--home-gripper-deg", type=float, default=DEFAULT_HOME_GRIPPER_POSITION
    )
    parser.add_argument("--max-auto-chunks", type=int, default=16)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.control_hz = _validate_control_hz(args.control_hz)
    if not 1 <= args.max_auto_chunks <= 16:
        raise ValueError("max-auto-chunks must lie in [1,16]")
    if not (
        DEFAULT_GRIPPER_CLOSED_POSITION
        <= args.open_gripper_deg
        <= DEFAULT_GRIPPER_OPEN_POSITION
    ):
        raise ValueError("open-gripper-deg must lie inside the calibrated task jaw range")
    if not (
        DEFAULT_GRIPPER_CLOSED_POSITION
        <= args.home_gripper_deg
        <= DEFAULT_GRIPPER_OPEN_POSITION
    ):
        raise ValueError("home-gripper-deg must lie inside the calibrated task jaw range")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    policy = V181LivePolicy(
        checkpoint=args.checkpoint,
        dino_repository=args.dino_repository,
        dino_weights=args.dino_weights,
        device=args.device,
        action_lower_position=V181_DEPLOYMENT_POSITION_LOWER,
        action_upper_position=V181_DEPLOYMENT_POSITION_UPPER,
        actor_conditioning=V181_DEPLOYMENT_ACTOR_CONDITIONING,
    )
    io_args = SimpleNamespace(
        wrist_device=args.wrist_device,
        column_device=args.column_device,
        output=output,
        robot_port=args.robot_port,
        robot_id=args.robot_id,
        calibration_dir=args.calibration_dir,
        max_command_jump=float(STEP_CAP_DEG.max() + 0.1),
        settle_timeout_s=args.settle_timeout_s,
        settle_position_tolerance=args.settle_position_tolerance,
        settle_step_tolerance=args.settle_step_tolerance,
        settle_samples=args.settle_samples,
        self_contact_calibration=False,
        i_accept_empty_jaw_self_contact=False,
        experiment_binding=None,
    )
    io = HardwareSO101IO(
        io_args,
        gripper_calibration_closed=DEFAULT_GRIPPER_CLOSED_POSITION,
        gripper_calibration_open=DEFAULT_GRIPPER_OPEN_POSITION,
        endpoint_envelope=EndpointEnvelope(
            max_joint_miss=float(STEP_CAP_DEG[:5].max()),
            max_tcp_miss_m=1.0,
            forward=None,
        ),
    )
    controller = V181HardwareService(
        policy,
        io,
        output=output,
        control_hz=args.control_hz,
        max_auto_chunks=args.max_auto_chunks,
        open_gripper_deg=args.open_gripper_deg,
        home_state=_load_home_state(
            args.home_state_report,
            gripper_position=args.home_gripper_deg,
        ),
        motor_calibration=_load_motor_calibration_contract(
            io.follower.calibration_fpath
        ),
    )
    server: ThreadingHTTPServer | None = None
    try:
        controller.bootstrap()
        server = ThreadingHTTPServer(
            (args.host, args.port), handler_factory(controller)
        )
        server.daemon_threads = True
        print(
            f"SO-101 V18.1 direct hardware service: http://{args.host}:{args.port}",
            flush=True,
        )

        def stop_service(signum: int, _frame: Any) -> None:
            controller.stop_event.set()
            raise KeyboardInterrupt(signal.Signals(signum).name)

        signal.signal(signal.SIGINT, stop_service)
        signal.signal(signal.SIGTERM, stop_service)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    finally:
        if server is not None:
            server.server_close()
        controller.close()


if __name__ == "__main__":
    main()
