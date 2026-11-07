#!/usr/bin/env python3
"""V17.7 safe runtime for the frozen V17.7 camera-aware predictive-belief stack.

Importing this module is side-effect free.  Hardware is opened only by
``main`` and continues to use the single established ``HardwareSO101IO``
transport from the production service.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
from http.server import ThreadingHTTPServer
from pathlib import Path
import signal
import sys
import termios
import threading
import time
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for source in (REPO, Path("/home/james/lerobot/src")):
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

from fcpg.prod_v17_7_live import (
    POSITION_UNIT,
    V177LivePolicy,
    _nonmonotonic_recovery_joints,
)
from fcpg.prod_v17_7_dino_targets import locate_dino_targets
from fcpg.prod_v17_7 import GRID_CELLS, GRID_SIZE
# The V17.7 runtime preserves the V17.6.3 calibrated endpoint identity
# tolerance.  It is deliberately local because the public V17.7 runtime API
# does not export hardware-specific encoder policy.
MAX_SOURCE_STATE_ENCODER_CELL_DELTA = 4
# The V17.7 branch intentionally keeps the live service self-contained.
# This retained advisory lease is the exact fail-closed ordering used by the
# V17.6.3 service: lock first, prove every resolved device is unused, then and
# only then construct HardwareSO101IO.
import os
import re
import subprocess
from typing import IO


class SO101HardwareLease:
    def __init__(self, handle: IO[str], resolved_devices: dict[str, str]) -> None:
        self._handle = handle
        self.resolved_devices = dict(resolved_devices)

    @property
    def acquired(self) -> bool:
        return self._handle is not None

    def close(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def acquire_so101_hardware_lease(
    *,
    lock_path: str | os.PathLike[str],
    serial_device: str | os.PathLike[str],
    wrist_device: str | os.PathLike[str],
    column_device: str | os.PathLike[str],
    fuser_binary: str = "fuser",
    fuser_timeout_s: float = 5.0,
) -> SO101HardwareLease:
    devices: dict[str, str] = {}
    for role, raw in {
        "serial": serial_device,
        "wrist": wrist_device,
        "column": column_device,
    }.items():
        expanded = os.path.expanduser(os.fspath(raw))
        resolved = os.path.realpath(expanded)
        if not expanded or not os.path.isabs(resolved) or not os.path.exists(resolved):
            raise RuntimeError(
                f"SO-101 {role} device does not exist after path resolution: "
                f"{expanded} -> {resolved}"
            )
        devices[role] = resolved
    if len(set(devices.values())) != len(devices):
        raise RuntimeError("SO-101 hardware device paths are not distinct")

    path = Path(lock_path).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            raise RuntimeError(
                f"another SO-101 hardware service already holds {path}"
            ) from exc
        for role in ("serial", "wrist", "column"):
            try:
                result = subprocess.run(
                    [fuser_binary, devices[role]],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=float(fuser_timeout_s),
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(
                    f"cannot verify ownership of SO-101 {role} device"
                ) from exc
            stderr = str(result.stderr or "").strip()
            if int(result.returncode) == 0:
                owners = sorted(set(re.findall(r"(?<!\\d)(\\d+)", result.stdout or "")))
                raise RuntimeError(
                    f"SO-101 {role} device is already in use: "
                    f"{devices[role]} (owner PID(s): {', '.join(owners) or 'unparsed'})"
                )
            if int(result.returncode) != 1 or stderr:
                raise RuntimeError(
                    f"could not establish whether SO-101 {role} device is free: "
                    f"{stderr or result.stdout or 'no diagnostic output'}"
                )
        return SO101HardwareLease(handle, devices)
    except BaseException:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
        raise
from fcpg.so101_kinematics import SO101MuJoCoKinematics
from fcpg.so101_realtime_runtime import (
    EndpointEnvelope,
    LiveObservation,
    gripper_capture_radius_m,
)
from fcpg.so101_retrieval_mpc import SO101_POSITION_LOWER, SO101_POSITION_UPPER
from fcpg.so101_visual_targets import VisualTargetTracker
import scripts.serve_so101_prod_v17_5 as service_base
from scripts.serve_so101_realtime_fcpg_lewm import (
    GOAL_READBACK_TOLERANCE,
    SAFE_HOLD_VERIFY_ATTEMPTS,
    STATE_READ_RETRIES,
)


DEFAULT_ROOT = Path(
    "/home/james/so101-vda-disk/feedback-corrected-grounding/outputs/lerobot/so101/"
    "prod_v17_7_camera_aware_belief"
)
DEFAULT_CHECKPOINT = DEFAULT_ROOT / "actor/final.pt"
DEFAULT_REAL_CALIBRATION = (
    REPO
    / "outputs/lerobot/so101/my_first_task_clean_v1_dual_camera_calibration"
    / "calibration_letterbox_v1.json"
)
DEFAULT_DINO_REPOSITORY = Path(
    "/home/james/.cache/torch/hub/facebookresearch_dinov2_main"
)
DEFAULT_DINO_WEIGHTS = Path(
    "/home/james/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
)
DEFAULT_KINEMATICS_MODEL = (
    REPO / "third_party/so_arm100/Simulation/SO101/so101_new_calib.xml"
)
DEFAULT_ROBOT_PORT = service_base.DEFAULT_ROBOT_PORT
DEFAULT_WRIST_DEVICE = service_base.DEFAULT_WRIST_DEVICE
DEFAULT_COLUMN_DEVICE = service_base.DEFAULT_COLUMN_DEVICE
DEFAULT_CONTROL_HZ = service_base.DEFAULT_CONTROL_HZ
DEFAULT_JOINT_LIMIT_GUARD = service_base.DEFAULT_JOINT_LIMIT_GUARD
DEFAULT_GRIPPER_CLOSED_POSITION = service_base.DEFAULT_GRIPPER_CLOSED_POSITION
DEFAULT_GRIPPER_OPEN_POSITION = service_base.DEFAULT_GRIPPER_OPEN_POSITION
DEFAULT_HOME_GRIPPER_POSITION = service_base.DEFAULT_HOME_GRIPPER_POSITION
MAX_PLAN_AGE_S = 5.0
MAX_CHUNKS_PER_REQUEST = 2
MAX_MODEL_CLOCK_LATENESS_S = 0.030
MAX_MODEL_ENDPOINT_HOLD_CORRECTIONS = 4
MAX_MODEL_ENDPOINT_ENCODER_TICKS = MAX_SOURCE_STATE_ENCODER_CELL_DELTA
HOME_RECOVERY_LIFT_STEP_POSITION = 2.0
TABLE_Z_MM = -17.118124
MINIMUM_ALLOWED_CLEARANCE_MM = -6.0
# A gate may be built with a lower physical floor only explicitly (V177ClearanceGate.physical_floor_mm), never below
# this: 10/16 the SO-101 jaws resting on the table read -6.4 to -6.7 mm against TABLE_Z_MM, below the -6.0 floor,
# so every firm table contact deadlocked the arm (docs/V21_SYSTEM_LEWM_CEM_AUDIT_20261013.md section 14.12).
LOWEST_PHYSICAL_FLOOR_MM = -10.0
PLANNED_CLEARANCE_MM = -6.0
READBACK_CLEARANCE_MM = -6.0
PATH_INTERPOLATION_STEP_POSITION = 0.25
DEFAULT_HARDWARE_LOCK = Path("/run/lock/so101-hardware.lock")
DEFAULT_DINO_TARGET_PROFILE = Path(
    "/home/james/so101-vda-disk/feedback-corrected-grounding/outputs/lerobot/"
    "so101/v12_1_actor_v12_4_lewm20k_live_20260817/target_profile/class_1"
)
OPERATOR_TARGET_ROLES = ("pickup", "drop_zone")
OPERATOR_TARGET_VIEWS = ("wrist", "column")
OPERATOR_TARGET_DEFAULT_RADIUS_PX = {
    "pickup": 18.0,
    "drop_zone": 28.0,
}

BELIEF_RENDER_COLORS = ((68, 211, 255), (255, 139, 72))
BELIEF_RENDER_LABELS = ("LF", "LM")

# User-authorized V17.7 physical-boundary mode: the live policy is limited by
# the calibrated mechanical range itself, without an additional software guard
# band or a narrower task-specific gripper interval.
V177_DEPLOYMENT_POSITION_LOWER = SO101_POSITION_LOWER.copy()
V177_DEPLOYMENT_POSITION_UPPER = SO101_POSITION_UPPER.copy()


def _parse_operator_target_request(
    payload: dict[str, Any],
) -> tuple[str, str, str, np.ndarray, float]:
    """Validate a current-plan, display-only operator annotation request."""

    round_id = payload.get("round_id")
    if not isinstance(round_id, str) or not round_id:
        raise ValueError("operator target round_id must be a non-empty string")
    role = payload.get("role")
    if role not in OPERATOR_TARGET_ROLES:
        raise ValueError(
            "operator target role must be exactly 'pickup' or 'drop_zone'"
        )
    view = payload.get("view")
    if view not in OPERATOR_TARGET_VIEWS:
        raise ValueError(
            "operator target view must be exactly 'wrist' or 'column'"
        )
    # Keep the established VisualTargetTracker contract: a drop region is a
    # fixed-overhead (column) reference.  We do not fabricate a corresponding
    # wrist pixel without a measured cross-view correspondence.
    if role == "drop_zone" and view != "column":
        raise ValueError("drop_zone operator target must use the column view")
    centroid_px = np.asarray(payload.get("centroid_px"), dtype=np.float32)
    if centroid_px.shape != (2,) or not np.all(np.isfinite(centroid_px)):
        raise ValueError("operator target centroid_px must be a finite two-vector")
    if np.any(centroid_px < 0.0) or np.any(centroid_px >= 224.0):
        raise ValueError("operator target centroid_px lies outside the current frame")
    radius_value = payload.get(
        "radius_px", OPERATOR_TARGET_DEFAULT_RADIUS_PX[role]
    )
    if isinstance(radius_value, bool):
        raise ValueError("operator target radius_px must be numeric")
    radius_px = float(radius_value)
    if not np.isfinite(radius_px) or not 6.0 <= radius_px <= 112.0:
        raise ValueError("operator target radius_px must be between 6 and 112")
    return round_id, role, view, centroid_px, radius_px


def _conditional_hpd_mask(log_belief: np.ndarray, mass: float) -> np.ndarray:
    """Return a conditional-visible HPD mask for display only."""

    values = np.asarray(log_belief, dtype=np.float64)
    if values.shape != (GRID_CELLS + 1,) or not np.all(np.isfinite(values)):
        raise ValueError("render belief must be finite [56*56+absent]")
    if not 0.0 < float(mass) < 1.0:
        raise ValueError("HPD mass must lie strictly between zero and one")
    spatial = values[:GRID_CELLS]
    probability = np.exp(spatial - float(np.max(spatial)))
    normalizer = float(probability.sum())
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("render belief has no finite visible mass")
    probability /= normalizer
    order = np.argsort(probability)[::-1]
    count = int(np.searchsorted(np.cumsum(probability[order]), mass, side="left")) + 1
    selected = np.zeros((GRID_CELLS,), dtype=np.uint8)
    selected[order[:count]] = 255
    return selected.reshape(GRID_SIZE, GRID_SIZE)


def _draw_belief_overlay(
    image: np.ndarray,
    log_posterior: np.ndarray | None,
    belief_means_px: np.ndarray | None,
    *,
    view_index: int,
    source_panel_size: int,
) -> None:
    """Draw learned 50/90 percent belief contours and diagnostic F/M means."""

    if log_posterior is not None:
        beliefs = np.asarray(log_posterior, dtype=np.float32)
        if beliefs.shape == (2, 2, GRID_CELLS + 1) and np.all(np.isfinite(beliefs)):
            for anchor, color in enumerate(BELIEF_RENDER_COLORS):
                for mass, thickness in ((0.90, 1), (0.50, 2)):
                    mask = _conditional_hpd_mask(
                        beliefs[view_index, anchor], mass
                    )
                    expanded = cv2.resize(
                        mask,
                        (source_panel_size, source_panel_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    contours, _ = cv2.findContours(
                        expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(
                        image, contours, -1, color, thickness, cv2.LINE_AA
                    )

    if belief_means_px is None:
        return
    means = np.asarray(belief_means_px, dtype=np.float32)
    if means.shape != (2, 2, 2) or not np.all(np.isfinite(means)):
        return
    scale = float(source_panel_size) / 224.0
    for anchor, (color, label) in enumerate(
        zip(BELIEF_RENDER_COLORS, BELIEF_RENDER_LABELS, strict=True)
    ):
        x, y = np.rint(means[view_index, anchor] * scale).astype(int)
        x = int(np.clip(x, 0, source_panel_size - 1))
        y = int(np.clip(y, 31, source_panel_size - 1))
        cv2.circle(image, (x, y), 6, color, 2, cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (min(x + 7, source_panel_size - 18), max(43, y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )


def _load_v177_home_state(path: Path, *, gripper_position: float) -> np.ndarray:
    """Load a HOME target inside the guarded deployment interval."""

    state = service_base.live_base._load_home_state(
        path,
        gripper_position=gripper_position,
    )
    if np.any(state[:5] < V177_DEPLOYMENT_POSITION_LOWER[:5]) or np.any(
        state[:5] > V177_DEPLOYMENT_POSITION_UPPER[:5]
    ):
        raise ValueError(
            "V17.7 HOME must remain inside the guarded deployment interval"
        )
    return state


@dataclass(frozen=True)
class V177ClearanceGate:
    """Fail-closed table-clearance checks in one public distance unit: mm."""

    kinematics: Any
    table_z_mm: float = TABLE_Z_MM
    planned_clearance_mm: float = PLANNED_CLEARANCE_MM
    readback_clearance_mm: float = READBACK_CLEARANCE_MM
    interpolation_step_position: float = PATH_INTERPOLATION_STEP_POSITION
    physical_floor_mm: float = MINIMUM_ALLOWED_CLEARANCE_MM

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.table_z_mm,
                self.planned_clearance_mm,
                self.readback_clearance_mm,
                self.interpolation_step_position,
                self.physical_floor_mm,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("V17.7 clearance contract must be finite")
        if not (
            LOWEST_PHYSICAL_FLOOR_MM
            <= self.physical_floor_mm
            <= MINIMUM_ALLOWED_CLEARANCE_MM
        ):
            raise ValueError(
                "the physical floor must lie between -10.0 and -6.0 mm"
            )
        if (
            self.planned_clearance_mm < self.physical_floor_mm
            or self.readback_clearance_mm < self.physical_floor_mm
            or self.interpolation_step_position <= 0.0
        ):
            raise ValueError(
                f"V17.7 clearance must remain at or above the "
                f"{self.physical_floor_mm:.1f} mm physical floor and "
                "interpolation step must be positive"
            )
        if self.planned_clearance_mm < self.readback_clearance_mm:
            raise ValueError(
                "planned clearance must not be below readback clearance"
            )

    def clearance_mm(self, state_position: np.ndarray) -> float:
        state = np.asarray(state_position, dtype=np.float64)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise RuntimeError("clearance state must be a finite six-vector")
        anchors_m = np.asarray(
            self.kinematics.forward_anchors(state), dtype=np.float64
        )
        if anchors_m.shape != (3, 3) or not np.all(np.isfinite(anchors_m)):
            raise RuntimeError(
                "SO-101 forward_anchors must return finite [3,3] XYZ"
            )
        return float(1000.0 * np.min(anchors_m[:, 2]) - self.table_z_mm)

    def require_readback(
        self, state_position: np.ndarray, *, label: str
    ) -> float:
        clearance = self.clearance_mm(state_position)
        if clearance < self.readback_clearance_mm - 1.0e-9:
            raise RuntimeError(
                f"{label} table clearance is {clearance:.2f} mm; "
                f"requires at least {self.readback_clearance_mm:.2f} mm"
            )
        return clearance

    def minimum_planned_clearance_mm(
        self, start_position: np.ndarray, commands_position: np.ndarray
    ) -> float:
        start = np.asarray(start_position, dtype=np.float64)
        commands = np.asarray(commands_position, dtype=np.float64)
        if start.shape != (6,) or commands.shape != (5, 6):
            raise RuntimeError("planned clearance path must be [6] plus [5,6]")
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(commands)):
            raise RuntimeError("planned clearance path must be finite")

        # The known SO-101 implementation accepts a batch and evaluates every
        # row in order under its FK lock. Custom per-state clearance methods
        # must retain the scalar dispatch that enforces their own geometry.
        # Pin the original method as well as the type: a class-level override
        # must not silently replace readback checks but bypass planned checks.
        batch_states = (
            [start]
            if (type(self) is V177ClearanceGate
                and type(self.kinematics) is SO101MuJoCoKinematics
                and "clearance_mm" not in vars(self)
                and getattr(self.clearance_mm, "__func__", None)
                    is _V177_STANDARD_CLEARANCE_MM)
            else None
        )
        minimum = self.clearance_mm(start) if batch_states is None else np.inf
        previous = start
        for target in commands:
            maximum_delta = float(np.max(np.abs(target - previous)))
            samples = max(
                1,
                int(
                    np.ceil(
                        maximum_delta / self.interpolation_step_position
                        - 1.0e-12
                    )
                ),
            )
            for alpha in np.linspace(0.0, 1.0, samples + 1)[1:]:
                state = previous + float(alpha) * (target - previous)
                if batch_states is None:
                    minimum = min(minimum, self.clearance_mm(state))
                else:
                    batch_states.append(state)
            previous = target
        if batch_states is not None:
            anchors_m = np.asarray(
                self.kinematics.forward_anchors(np.stack(batch_states)),
                dtype=np.float64,
            )
            if anchors_m.shape != (len(batch_states), 3, 3) or not np.all(
                np.isfinite(anchors_m)
            ):
                raise RuntimeError(
                    "SO-101 forward_anchors must return finite [N,3,3] XYZ"
                )
            minimum = np.min(
                1000.0 * np.min(anchors_m[:, :, 2], axis=1) - self.table_z_mm
            )
        return float(minimum)

    def require_planned(
        self, start_position: np.ndarray, commands_position: np.ndarray
    ) -> float:
        clearance = self.minimum_planned_clearance_mm(
            start_position, commands_position
        )
        if clearance < self.planned_clearance_mm - 1.0e-9:
            raise RuntimeError(
                f"planned table clearance is {clearance:.2f} mm; "
                f"requires at least {self.planned_clearance_mm:.2f} mm"
            )
        return clearance

    def require_monotonic_recovery(
        self, start_position: np.ndarray, commands_position: np.ndarray
    ) -> dict[str, float]:
        """Admit only an upward escape from between the two clearance gates.

        This is deliberately stricter than ordinary policy admission.  A
        stopped arm whose measured clearance is still above the 10 mm readback
        floor may regain the 25 mm planning envelope only through a path whose
        interpolated clearance never decreases.
        """

        start = np.asarray(start_position, dtype=np.float64)
        commands = np.asarray(commands_position, dtype=np.float64)
        if start.shape != (6,) or commands.shape != (5, 6):
            raise RuntimeError("clearance recovery path must be [6] plus [5,6]")
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(commands)):
            raise RuntimeError("clearance recovery path must be finite")

        start_clearance = self.require_readback(
            start, label="clearance recovery source"
        )
        if start_clearance >= self.planned_clearance_mm - 1.0e-9:
            minimum = self.require_planned(start, commands)
            return {
                "start_clearance_mm": start_clearance,
                "minimum_clearance_mm": minimum,
                "final_clearance_mm": self.clearance_mm(commands[-1]),
            }

        previous_state = start
        previous_clearance = start_clearance
        minimum_clearance = start_clearance
        recovered_planning_envelope = False
        for target in commands:
            maximum_delta = float(np.max(np.abs(target - previous_state)))
            samples = max(
                1,
                int(
                    np.ceil(
                        maximum_delta / self.interpolation_step_position
                        - 1.0e-12
                    )
                ),
            )
            for alpha in np.linspace(0.0, 1.0, samples + 1)[1:]:
                state = previous_state + float(alpha) * (target - previous_state)
                clearance = self.clearance_mm(state)
                if not recovered_planning_envelope:
                    if clearance + 1.0e-9 < previous_clearance:
                        raise RuntimeError(
                            "low-clearance recovery would move toward the table: "
                            f"{previous_clearance:.2f} -> {clearance:.2f} mm"
                        )
                    if clearance >= self.planned_clearance_mm - 1.0e-9:
                        recovered_planning_envelope = True
                elif clearance < self.planned_clearance_mm - 1.0e-9:
                    raise RuntimeError(
                        "low-clearance recovery would leave the regained "
                        "planning envelope: "
                        f"{clearance:.2f} mm < "
                        f"{self.planned_clearance_mm:.2f} mm"
                    )
                minimum_clearance = min(minimum_clearance, clearance)
                previous_clearance = clearance
            previous_state = target

        final_clearance = self.clearance_mm(commands[-1])
        if not recovered_planning_envelope:
            raise RuntimeError(
                "low-clearance recovery does not regain the planning gate: "
                f"{final_clearance:.2f} mm < {self.planned_clearance_mm:.2f} mm"
            )
        return {
            "start_clearance_mm": start_clearance,
            "minimum_clearance_mm": minimum_clearance,
            "final_clearance_mm": final_clearance,
        }

    def require_policy_path(
        self, start_position: np.ndarray, commands_position: np.ndarray
    ) -> dict[str, float | str]:
        """Audit one frozen-policy block without weakening either threshold.

        A source already inside the 25 mm planning buffer cannot satisfy the
        ordinary minimum-path test because the source itself is part of that
        path.  In that state, admit only the existing monotonic recovery
        contract.  The policy commands are never projected or edited here.
        """

        start_clearance = self.require_readback(
            start_position, label="policy source readback"
        )
        if start_clearance < self.planned_clearance_mm - 1.0e-9:
            evidence = self.require_monotonic_recovery(
                start_position, commands_position
            )
            return {**evidence, "mode": "monotonic_recovery"}

        minimum = self.require_planned(start_position, commands_position)
        return {
            "start_clearance_mm": start_clearance,
            "minimum_clearance_mm": minimum,
            "final_clearance_mm": self.clearance_mm(commands_position[-1]),
            "mode": "planned_envelope",
        }


_V177_STANDARD_CLEARANCE_MM = V177ClearanceGate.clearance_mm


def _audit_v177_chunk(
    commands: np.ndarray,
    state: np.ndarray,
    *,
    clearance_gate: V177ClearanceGate | None = None,
) -> dict[str, Any]:
    """Apply hard limits, guard recovery and optional millimetre clearance."""

    chunk = np.asarray(commands, dtype=np.float32)
    current = np.asarray(state, dtype=np.float32)
    reasons: set[str] = set()
    if (
        chunk.shape != (5, 6)
        or current.shape != (6,)
        or not np.all(np.isfinite(chunk))
        or not np.all(np.isfinite(current))
    ):
        return {
            "safe": False,
            "reasons": ["finite_shape"],
        }
    if np.any(chunk < SO101_POSITION_LOWER) or np.any(
        chunk > SO101_POSITION_UPPER
    ):
        reasons.add("hard_position_bounds")
    reversals = _nonmonotonic_recovery_joints(
        chunk,
        current,
        SO101_POSITION_LOWER,
        SO101_POSITION_UPPER,
    )
    if reversals:
        reasons.add("nonmonotonic_physical_limit_recovery")
    transitions = np.diff(
        np.concatenate(
            (current[None], chunk), axis=0
        ),
        axis=0,
    )
    audit = {
        "safe": not reasons,
        "reasons": sorted(reasons),
        "maximum_observed_step_per_joint_position": np.max(
            np.abs(transitions), axis=0
        ),
    }
    if clearance_gate is not None:
        try:
            clearance = clearance_gate.require_policy_path(current, chunk)
            audit["minimum_planned_clearance_mm"] = clearance[
                "minimum_clearance_mm"
            ]
            audit["clearance_mode"] = clearance["mode"]
            audit["final_planned_clearance_mm"] = clearance[
                "final_clearance_mm"
            ]
        except RuntimeError as exc:
            audit["safe"] = False
            audit["reasons"] = sorted(set(audit["reasons"]) | {"table_clearance"})
            audit["clearance_error"] = str(exc)
    return audit


@dataclass(frozen=True)
class V177HeldEndpoint:
    """Truthful endpoint evidence after cancelling pursuit at measured q5."""

    outcome: str
    commanded_state: np.ndarray
    actual_state: np.ndarray
    joint_residual: float
    step_delta: float
    reached_commanded_endpoint: bool


@dataclass(frozen=True)
class V177ClockedExecution:
    """One transport with a 0.1 s q-path and a held actual endpoint."""

    requested: np.ndarray
    sent: np.ndarray
    measured_q_path: np.ndarray
    transition_observation: LiveObservation
    settled_observation: LiveObservation
    transition_duration_s: float
    duration_s: float
    maximum_clock_lateness_s: float
    minimum_readback_clearance_mm: float
    command_journal: str | None
    settlement: V177HeldEndpoint

    @property
    def reached_commanded_endpoint(self) -> bool:
        return bool(self.settlement.reached_commanded_endpoint)


class V177HardwareSO101IO(service_base.live_base.HardwareSO101IO):
    """Established transport with V17.7's exact 10 Hz readback clock."""

    def __init__(
        self,
        *args: Any,
        clearance_gate: V177ClearanceGate,
        **kwargs: Any,
    ) -> None:
        self.clearance_gate = clearance_gate
        try:
            super().__init__(*args, **kwargs)
            self.command_gate = threading.Lock()
            serial_handle = self.follower.bus.port_handler.ser
            if serial_handle is None or not hasattr(serial_handle, "fileno"):
                raise RuntimeError("SO-101 serial handle is unavailable")
            fcntl.ioctl(serial_handle.fileno(), termios.TIOCEXCL)
        except BaseException:
            self._rollback_partial_connection()
            raise

    def observe(self) -> LiveObservation:
        observation = super().observe()
        self.clearance_gate.require_readback(
            observation.state, label="observed robot state"
        )
        return observation

    def execute_chunk(
        self,
        chunk: np.ndarray,
        *,
        period_s: float,
        stop_event: Any,
    ) -> V177ClockedExecution:
        """Give HOME/open-gripper the same clearance path as policy motion."""

        initial = self.observe()
        self.clearance_gate.require_planned(initial.state, chunk)
        return self.execute_v177_chunk(
            chunk,
            initial_observation=initial,
            period_s=period_s,
            stop_event=stop_event,
        )

    def execute_home_chunk(
        self,
        chunk: np.ndarray,
        *,
        period_s: float,
        stop_event: Any,
    ) -> Any:
        """Execute a HOME-only chunk without imposing the model q5 clock."""

        initial = self.observe()
        start_clearance = self.clearance_gate.clearance_mm(initial.state)
        low_clearance_recovery = bool(
            start_clearance < self.clearance_gate.planned_clearance_mm - 1.0e-9
        )
        if low_clearance_recovery:
            self.clearance_gate.require_monotonic_recovery(initial.state, chunk)
        else:
            self.clearance_gate.require_planned(initial.state, chunk)
        # HOME is not a model transition and is discarded before the fresh H0.
        # The established manual transport observes after every command; this
        # class's ``observe`` keeps the 10 mm readback gate active throughout.
        return super().execute_chunk(
            chunk, period_s=period_s, stop_event=stop_event
        )

    def _rollback_partial_connection(self) -> None:
        """Close every bus/camera that may exist after a partial constructor."""

        follower = getattr(self, "follower", None)
        if follower is None:
            return
        for camera in getattr(follower, "cameras", {}).values():
            try:
                if bool(getattr(camera, "is_connected", False)):
                    camera.disconnect()
            except Exception:
                pass
        bus = getattr(follower, "bus", None)
        try:
            if bus is not None and bool(getattr(bus, "is_connected", False)):
                bus.disconnect(disable_torque=False)
        except Exception:
            pass

    @staticmethod
    def _require_model_clock_deadline(
        *, phase: str, boundary_s: float, completed_s: float
    ) -> float:
        """Reject rather than compress the next interval of the 10 Hz clock."""

        lateness = max(0.0, float(completed_s) - float(boundary_s))
        if lateness > MAX_MODEL_CLOCK_LATENESS_S:
            raise V177ClockDeadlineMiss(
                f"V17.7 model clock {phase} missed its boundary by "
                f"{lateness:.4f}s"
            )
        return lateness

    def _flush_clock_records(
        self,
        journal_path: Path | None,
        records: list[dict[str, Any]],
    ) -> None:
        """Persist clock telemetry only after motion is physically held."""

        for record in records:
            self._append_command_journal(journal_path, record)

    def calibration_reconstructed_encoder_cells(
        self, state: np.ndarray
    ) -> np.ndarray:
        """Map calibrated positions back to integer motor cells for identity."""

        values = np.asarray(state, dtype=np.float32)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("endpoint state must be a finite six-vector")
        bus = self.follower.bus
        motor_order = service_base.live_base.SO101_MOTOR_ORDER
        by_id = {
            bus.motors[name].id: float(values[index])
            for index, name in enumerate(motor_order)
        }
        reconstructed = bus._unnormalize(by_id)
        return np.asarray(
            [int(reconstructed[motor_id]) for motor_id in by_id],
            dtype=np.int64,
        )

    def _endpoint_encoder_delta_ticks(
        self, reference: np.ndarray, measured: np.ndarray
    ) -> np.ndarray:
        """Compare calibrated positions in reconstructed integer motor cells."""

        target_cells = self.calibration_reconstructed_encoder_cells(reference)
        actual_cells = self.calibration_reconstructed_encoder_cells(measured)
        return np.abs(actual_cells - target_cells)

    def _same_model_endpoint(
        self, reference: np.ndarray, measured: np.ndarray
    ) -> tuple[bool, np.ndarray | None]:
        """Use encoder cells for identity while retaining the quiet-motion gate."""

        if not hasattr(self, "follower"):
            # Explicit unit-test doubles have no hardware calibration.
            delta = float(
                np.max(
                    np.abs(
                        np.asarray(reference, dtype=np.float32)
                        - np.asarray(measured, dtype=np.float32)
                    )
                )
            )
            return delta <= float(self.settle_step_tolerance), None
        ticks = self._endpoint_encoder_delta_ticks(reference, measured)
        return bool(np.all(ticks <= MAX_MODEL_ENDPOINT_ENCODER_TICKS)), ticks

    def _hold_at_model_endpoint(
        self,
        target_state: np.ndarray,
        *,
        reason: str,
    ) -> dict[str, Any]:
        """Write and verify a hold goal selected by the q5 lock controller."""

        target = np.asarray(target_state, dtype=np.float32)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise ValueError("V17.7 model endpoint hold must be a finite six-vector")
        self.clearance_gate.require_readback(
            target, label="model endpoint hold target"
        )
        attempts: list[dict[str, Any]] = []
        mismatched: dict[str, list[float]] = {}
        motor_order = service_base.live_base.SO101_MOTOR_ORDER
        for attempt in range(SAFE_HOLD_VERIFY_ATTEMPTS):
            written = self._write_position_command(target, target)
            if not np.array_equal(written, target):
                raise RuntimeError("model endpoint hold changed the q5 target")
            goal = {
                name: float(target[index])
                for index, name in enumerate(motor_order)
            }
            readback = self.follower.bus.sync_read(
                "Goal_Position", num_retry=STATE_READ_RETRIES
            )
            mismatched = {
                name: [goal[name], float(readback[name])]
                for name in goal
                if abs(float(readback[name]) - goal[name])
                > GOAL_READBACK_TOLERANCE
            }
            attempts.append(
                {
                    "attempt": attempt,
                    "target_state": target,
                    "goal_readback": {
                        name: float(readback[name]) for name in goal
                    },
                    "mismatched_joints": mismatched,
                }
            )
            if not mismatched:
                return {
                    "event": "model_endpoint_hold_verified",
                    "monotonic_s": time.perf_counter(),
                    "reason": reason,
                    "attempts": attempts,
                }
        raise RuntimeError(
            "V17.7 exact model endpoint hold was not confirmed after "
            f"{SAFE_HOLD_VERIFY_ATTEMPTS} attempts; mismatched joints "
            f"(requested, read-back): {mismatched}"
        )

    def _observe_held_model_endpoint(
        self,
        target_state: np.ndarray,
        *,
        period_s: float,
        stop_event: Any,
        records: list[dict[str, Any]] | None = None,
        clearances: list[float] | None = None,
    ) -> tuple[LiveObservation, list[dict[str, Any]], list[float]]:
        """Prove q5 is quiet before and throughout the dual-camera window."""

        target = np.asarray(target_state, dtype=np.float32)
        tolerance = float(self.settle_step_tolerance)
        deadline = time.perf_counter() + float(self.settle_timeout_s)
        previous = target.copy()
        hold_goal = target.copy()
        stable_samples = 0
        quiet_samples = 0
        correction_count = 0
        # A loaded joint can sit at a repeatable offset from its requested
        # Goal_Position.  Servo compensation is therefore a bounded root
        # search in command space, not an unconstrained integral controller.
        # Keep the most recent (goal, signed endpoint error) sample for each
        # joint so a sign change can be resolved inside the measured bracket.
        previous_correction_goal = np.full((6,), np.nan, dtype=np.float32)
        previous_correction_error = np.full((6,), np.nan, dtype=np.float32)
        awaiting_response_state = np.full((6,), np.nan, dtype=np.float32)
        awaiting_response_direction = np.zeros((6,), dtype=np.float32)
        awaiting_response_mask = np.zeros((6,), dtype=bool)
        correction_limit_wait_logged = False
        sample_records = [] if records is None else records
        clearance_values = [] if clearances is None else clearances
        while True:
            if stop_event.is_set():
                raise RuntimeError(
                    "execution stopped while holding the V17.7 model endpoint"
                )
            if time.perf_counter() >= deadline:
                raise RuntimeError(
                    "V17.7 model endpoint did not become frame-stable inside "
                    "the configured settle timeout"
                )
            time.sleep(period_s)
            sample_started = time.perf_counter()
            current = self._read_state_only()
            clearance = self.clearance_gate.require_readback(
                current, label="model endpoint settle readback"
            )
            sample_finished = time.perf_counter()
            clearance_values.append(clearance)
            target_error = float(np.max(np.abs(current - target)))
            step_delta = float(np.max(np.abs(current - previous)))
            motion_quiet = step_delta <= tolerance
            endpoint_matches, encoder_delta_ticks = self._same_model_endpoint(
                target, current
            )
            if bool(np.any(awaiting_response_mask)):
                if hasattr(self, "follower"):
                    response_ticks = self._endpoint_encoder_delta_ticks(
                        awaiting_response_state, current
                    )
                    response_changed = response_ticks > 0
                else:
                    response_ticks = None
                    response_changed = (
                        np.abs(current - awaiting_response_state) > 1.0e-6
                    )
                response_progress = (
                    (current - awaiting_response_state)
                    * awaiting_response_direction
                    > 0.0
                )
                if encoder_delta_ticks is None:
                    target_reached_by_joint = (
                        np.abs(current - target) <= tolerance
                    )
                else:
                    target_reached_by_joint = (
                        np.asarray(encoder_delta_ticks, dtype=np.int64)
                        <= MAX_MODEL_ENDPOINT_ENCODER_TICKS
                    )
                responded = target_reached_by_joint | (
                    response_changed & response_progress
                )
                awaiting_response_mask &= ~responded
            else:
                response_ticks = None
            endpoint_quiet = endpoint_matches and motion_quiet
            quiet_samples = quiet_samples + 1 if motion_quiet else 0
            stable_samples = stable_samples + 1 if endpoint_quiet else 0
            sample_records.append(
                {
                    "event": "v177_model_endpoint_settle_sample",
                    "sample_started_s": sample_started,
                    "sample_finished_s": sample_finished,
                    "state": current,
                    "target_error": target_error,
                    "encoder_delta_ticks": encoder_delta_ticks,
                    "awaiting_correction_response": awaiting_response_mask,
                    "correction_response_ticks": response_ticks,
                    "step_delta": step_delta,
                    "quiet_samples": quiet_samples,
                    "stable_samples": stable_samples,
                    "clearance_mm": clearance,
                }
            )
            previous = np.asarray(current, dtype=np.float32).copy()
            if (
                stable_samples == 0
                and quiet_samples >= int(self.settle_samples)
                and not endpoint_matches
            ):
                if target_error > float(self.settle_position_tolerance) + 1.0e-4:
                    raise RuntimeError(
                        "V17.7 q5 hold stalled outside the existing endpoint "
                        f"envelope: error={target_error:.3f}"
                    )
                # A quiet readback is not evidence that the most recent goal
                # change has taken effect.  Do not integrate the same stale
                # error repeatedly while a loaded servo is still responding.
                if bool(np.any(awaiting_response_mask)):
                    continue
                if correction_count >= MAX_MODEL_ENDPOINT_HOLD_CORRECTIONS:
                    if not correction_limit_wait_logged:
                        sample_records.append(
                            {
                                "event": (
                                    "v177_model_endpoint_correction_limit_wait"
                                ),
                                "correction_count": correction_count,
                                "target_state": target,
                                "measured_state": current,
                                "encoder_delta_ticks": encoder_delta_ticks,
                            }
                        )
                        correction_limit_wait_logged = True
                    # Keep the fourth bounded goal active and allow its delayed
                    # physical response until the original settle deadline.
                    continue
                signed_error = target - current
                if encoder_delta_ticks is None:
                    correction_mask = np.abs(signed_error) > tolerance
                else:
                    correction_mask = (
                        np.asarray(encoder_delta_ticks, dtype=np.int64)
                        > MAX_MODEL_ENDPOINT_ENCODER_TICKS
                    )
                if not bool(np.any(correction_mask)):
                    raise RuntimeError(
                        "V17.7 q5 hold endpoint classification was "
                        "inconsistent with its correction mask"
                    )

                next_goal = hold_goal.copy()
                bracketed = np.zeros((6,), dtype=bool)
                for joint in np.flatnonzero(correction_mask):
                    error = float(signed_error[joint])
                    old_error = float(previous_correction_error[joint])
                    old_goal = float(previous_correction_goal[joint])
                    current_goal = float(hold_goal[joint])
                    crossed = bool(
                        np.isfinite(old_error)
                        and np.isfinite(old_goal)
                        and old_error * error < 0.0
                    )
                    if crossed:
                        # Secant interpolation remains strictly inside the two
                        # observed command goals.  Fall back to their midpoint
                        # only for a numerically degenerate bracket.
                        denominator = error - old_error
                        candidate = (
                            current_goal
                            - error * (current_goal - old_goal) / denominator
                            if abs(denominator) > 1.0e-9
                            else 0.5 * (current_goal + old_goal)
                        )
                        lower = min(current_goal, old_goal)
                        upper = max(current_goal, old_goal)
                        if not np.isfinite(candidate) or not lower < candidate < upper:
                            candidate = 0.5 * (current_goal + old_goal)
                        next_goal[joint] = np.float32(candidate)
                        bracketed[joint] = True
                    else:
                        next_goal[joint] = np.float32(current_goal + error)

                correction = next_goal - hold_goal
                if (
                    np.any(next_goal < V177_DEPLOYMENT_POSITION_LOWER)
                    or np.any(next_goal > V177_DEPLOYMENT_POSITION_UPPER)
                ):
                    raise RuntimeError(
                        "V17.7 q5 hold compensation would leave deployment bounds"
                    )
                self.clearance_gate.require_planned(
                    current,
                    np.repeat(next_goal[None], 5, axis=0),
                )
                correction_count += 1
                sample_records.append(
                    {
                        "event": "v177_model_endpoint_hold_correction",
                        "correction_index": correction_count,
                        "measured_state": current,
                        "target_state": target,
                        "previous_hold_goal": hold_goal,
                        "correction": correction,
                        "corrected_joint_mask": correction_mask,
                        "bracketed_joint_mask": bracketed,
                        "next_hold_goal": next_goal,
                    }
                )
                with self.command_gate:
                    correction_record = self._hold_at_model_endpoint(
                        next_goal,
                        reason="v177_model_endpoint_servo_compensation",
                    )
                sample_records.append(correction_record)
                previous_correction_goal[correction_mask] = hold_goal[
                    correction_mask
                ]
                previous_correction_error[correction_mask] = signed_error[
                    correction_mask
                ]
                awaiting_response_state = np.asarray(
                    current, dtype=np.float32
                ).copy()
                awaiting_response_direction.fill(0.0)
                awaiting_response_direction[correction_mask] = np.sign(
                    correction[correction_mask]
                )
                awaiting_response_mask = correction_mask.copy()
                hold_goal = np.asarray(next_goal, dtype=np.float32).copy()
                quiet_samples = 0
                stable_samples = 0
                previous = np.asarray(current, dtype=np.float32).copy()
                continue
            if stable_samples < int(self.settle_samples):
                continue

            q_pre = self._read_state_only()
            pre_clearance = self.clearance_gate.require_readback(
                q_pre, label="pre-frame model endpoint readback"
            )
            frame_started = time.perf_counter()
            with self.camera_lock:
                frames = {
                    view: self._read_camera(view)
                    for view in ("wrist", "column")
                }
                raw = getattr(self, "_last_raw_frames", None)
                native = (
                    None
                    if not isinstance(raw, dict)
                    or not set(frames).issubset(raw)
                    else {
                        view: np.asarray(raw[view]).copy()
                        for view in frames
                    }
                )
            frame_finished = time.perf_counter()
            q_post = self._read_state_only()
            post_clearance = self.clearance_gate.require_readback(
                q_post, label="post-frame model endpoint readback"
            )
            clearance_values.extend((pre_clearance, post_clearance))
            pre_error = float(np.max(np.abs(q_pre - target)))
            post_error = float(np.max(np.abs(q_post - target)))
            pre_matches, pre_encoder_ticks = self._same_model_endpoint(
                target, q_pre
            )
            post_matches, post_encoder_ticks = self._same_model_endpoint(
                target, q_post
            )
            bracket_delta = float(np.max(np.abs(q_post - q_pre)))
            previous_delta = float(np.max(np.abs(q_pre - previous)))
            bracket_valid = bool(
                pre_matches
                and post_matches
                and max(bracket_delta, previous_delta) <= tolerance
            )
            sample_records.append(
                {
                    "event": "v177_model_endpoint_frame_bracket",
                    "frame_started_s": frame_started,
                    "frame_finished_s": frame_finished,
                    "q_pre": q_pre,
                    "q_post": q_post,
                    "pre_target_error": pre_error,
                    "post_target_error": post_error,
                    "pre_encoder_delta_ticks": pre_encoder_ticks,
                    "post_encoder_delta_ticks": post_encoder_ticks,
                    "bracket_delta": bracket_delta,
                    "previous_delta": previous_delta,
                    "valid": bracket_valid,
                    "pre_clearance_mm": pre_clearance,
                    "post_clearance_mm": post_clearance,
                }
            )
            if bracket_valid:
                return (
                    LiveObservation(
                        timestamp_s=0.5 * (frame_started + frame_finished),
                        state=np.asarray(q_post, dtype=np.float32),
                        frames=frames,
                        native_frames=native,
                    ),
                    sample_records,
                    clearance_values,
                )
            stable_samples = 0
            quiet_samples = 0
            previous = np.asarray(q_post, dtype=np.float32).copy()

    def _observe_settled_actual_endpoint(
        self,
        timed_q5: np.ndarray,
        *,
        period_s: float,
        stop_event: Any,
        records: list[dict[str, Any]] | None = None,
        clearances: list[float] | None = None,
    ) -> tuple[LiveObservation, list[dict[str, Any]], list[float]]:
        """Observe the real quiet endpoint without steering it back to q5.

        The frozen model was trained on actual readback at the endpoint.  The
        transport therefore lets the last Actor goal settle, writes exactly
        one Present->Goal=Present hold to cancel further pursuit, and brackets
        the two camera reads with fresh readbacks.  No corrective command is
        generated here and the timed 10 Hz q0..q5 path remains unchanged.
        """

        previous = np.asarray(timed_q5, dtype=np.float32).copy()
        if previous.shape != (6,) or not np.all(np.isfinite(previous)):
            raise ValueError("V17.7 timed q5 must be a finite six-vector")
        tolerance = float(self.settle_step_tolerance)
        deadline = time.perf_counter() + float(self.settle_timeout_s)
        quiet_samples = 0
        hold_armed = False
        sample_records = [] if records is None else records
        clearance_values = [] if clearances is None else clearances
        while True:
            if stop_event.is_set():
                raise RuntimeError(
                    "execution stopped while observing the actual V17.7 endpoint"
                )
            if time.perf_counter() >= deadline:
                raise RuntimeError(
                    "V17.7 actual endpoint did not become frame-stable inside "
                    "the configured settle timeout"
                )
            time.sleep(period_s)
            sample_started = time.perf_counter()
            current = self._read_state_only()
            clearance = self.clearance_gate.require_readback(
                current, label="actual endpoint settle readback"
            )
            sample_finished = time.perf_counter()
            clearance_values.append(clearance)
            step_delta = float(np.max(np.abs(current - previous)))
            motion_quiet = step_delta <= tolerance
            quiet_samples = quiet_samples + 1 if motion_quiet else 0
            sample_records.append(
                {
                    "event": "v177_actual_endpoint_settle_sample",
                    "sample_started_s": sample_started,
                    "sample_finished_s": sample_finished,
                    "state": current,
                    "step_delta": step_delta,
                    "quiet_samples": quiet_samples,
                    "hold_armed": hold_armed,
                    "clearance_mm": clearance,
                }
            )
            previous = np.asarray(current, dtype=np.float32).copy()
            if quiet_samples < int(self.settle_samples):
                continue
            if not hold_armed:
                with self.command_gate:
                    hold_record = self._hold_at_measured_pose(
                        None, reason="v177_present_endpoint_hold"
                    )
                sample_records.append(hold_record)
                attempts = hold_record.get("attempts", [])
                if not attempts:
                    raise RuntimeError(
                        "V17.7 Present endpoint hold returned no readback evidence"
                    )
                held_state = np.asarray(
                    attempts[-1].get("held_state"), dtype=np.float32
                )
                if held_state.shape != (6,) or not np.all(np.isfinite(held_state)):
                    raise RuntimeError(
                        "V17.7 Present endpoint hold returned an invalid state"
                    )
                self.clearance_gate.require_readback(
                    held_state, label="actual endpoint hold state"
                )
                previous = held_state.copy()
                quiet_samples = 0
                hold_armed = True
                continue

            q_pre = self._read_state_only()
            pre_clearance = self.clearance_gate.require_readback(
                q_pre, label="pre-frame actual endpoint readback"
            )
            frame_started = time.perf_counter()
            with self.camera_lock:
                frames = {
                    view: self._read_camera(view)
                    for view in ("wrist", "column")
                }
                raw = getattr(self, "_last_raw_frames", None)
                native = (
                    None
                    if not isinstance(raw, dict) or not set(frames).issubset(raw)
                    else {
                        view: np.asarray(raw[view]).copy()
                        for view in frames
                    }
                )
            frame_finished = time.perf_counter()
            q_post = self._read_state_only()
            post_clearance = self.clearance_gate.require_readback(
                q_post, label="post-frame actual endpoint readback"
            )
            clearance_values.extend((pre_clearance, post_clearance))
            bracket_matches, bracket_encoder_ticks = self._same_model_endpoint(
                q_pre, q_post
            )
            bracket_delta = float(np.max(np.abs(q_post - q_pre)))
            previous_delta = float(np.max(np.abs(q_pre - previous)))
            bracket_valid = bool(
                bracket_matches
                and max(bracket_delta, previous_delta) <= tolerance
            )
            sample_records.append(
                {
                    "event": "v177_actual_endpoint_frame_bracket",
                    "frame_started_s": frame_started,
                    "frame_finished_s": frame_finished,
                    "q_pre": q_pre,
                    "q_post": q_post,
                    "bracket_encoder_ticks": bracket_encoder_ticks,
                    "bracket_delta": bracket_delta,
                    "previous_delta": previous_delta,
                    "valid": bracket_valid,
                    "pre_clearance_mm": pre_clearance,
                    "post_clearance_mm": post_clearance,
                }
            )
            if bracket_valid:
                return (
                    LiveObservation(
                        timestamp_s=0.5 * (frame_started + frame_finished),
                        state=np.asarray(q_post, dtype=np.float32),
                        frames=frames,
                        native_frames=native,
                    ),
                    sample_records,
                    clearance_values,
                )
            quiet_samples = 0
            previous = np.asarray(q_post, dtype=np.float32).copy()

    def execute_v177_chunk(
        self,
        chunk: np.ndarray,
        *,
        initial_observation: LiveObservation,
        period_s: float,
        stop_event: Any,
    ) -> V177ClockedExecution:
        values = np.asarray(chunk, dtype=np.float32)
        if values.shape != (5, 6) or not np.all(np.isfinite(values)):
            raise ValueError("V17.7 hardware chunk must be finite [5,6]")
        if not np.isfinite(period_s) or float(period_s) <= 0.0:
            raise ValueError("V17.7 hardware period must be positive")
        self.clearance_gate.require_policy_path(
            initial_observation.state, values
        )

        journal_path = self._start_command_journal(values)
        self.last_command_journal = (
            None if journal_path is None else str(journal_path)
        )
        previous_command = np.asarray(
            initial_observation.state, dtype=np.float32
        ).copy()
        q_path = [previous_command.copy()]
        sent: list[np.ndarray] = []
        motion_attempted = False
        lateness: list[float] = []
        clock_records: list[dict[str, Any]] = []
        readback_clearances = [
            self.clearance_gate.clearance_mm(initial_observation.state)
        ]
        # The complete five-command intent is already durably journalled by
        # _start_command_journal.  Finish every other synchronous preamble
        # before choosing the 10 Hz epoch; fsync must never consume part of a
        # model interval.
        self._append_command_journal(
            journal_path,
            {
                "event": "v177_clock_schedule_armed",
                "model_period_s": period_s,
                "before_state": previous_command,
            },
        )
        started = time.perf_counter()
        clock_records.append(
            {
                "event": "v177_clocked_chunk_started",
                "monotonic_s": started,
                "before_state": previous_command,
                "model_period_s": period_s,
            }
        )
        try:
            for index, command in enumerate(values):
                if stop_event.is_set():
                    raise RuntimeError("execution stopped by operator")
                boundary = started + index * period_s
                delay = boundary - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)

                read_started: float | None = None
                read_finished: float | None = None
                clearance_finished: float | None = None
                pre_boundary_late = max(0.0, time.perf_counter() - boundary)
                lateness.append(pre_boundary_late)
                if pre_boundary_late > MAX_MODEL_CLOCK_LATENESS_S:
                    clock_records.append(
                        {
                            "event": "v177_command_deadline",
                            "command_index": index,
                            "boundary_s": boundary,
                            "pre_boundary_lateness_s": pre_boundary_late,
                        }
                    )
                    self._require_model_clock_deadline(
                        phase=f"q{index} read start",
                        boundary_s=boundary,
                        completed_s=boundary + pre_boundary_late,
                    )

                # q_i is sampled immediately before command i, matching the
                # fish archive's state/action row at t+[0,3,6,9,12]. q0 is the
                # fresh transport observation already captured above.
                if index > 0:
                    read_started = time.perf_counter()
                    measured = self._read_state_only()
                    read_finished = time.perf_counter()
                    acquisition_late = max(0.0, read_finished - boundary)
                    lateness.append(acquisition_late)
                    clearance = self.clearance_gate.require_readback(
                        measured, label=f"q{index} readback"
                    )
                    clearance_finished = time.perf_counter()
                    command_ready_late = max(
                        0.0, clearance_finished - boundary
                    )
                    lateness.append(command_ready_late)
                    readback_clearances.append(clearance)
                    q_path.append(np.asarray(measured, dtype=np.float32).copy())
                    clock_records.append(
                        {
                            "event": "v177_model_readback",
                            "path_index": index,
                            "boundary_s": boundary,
                            "read_started_s": read_started,
                            "read_finished_s": read_finished,
                            "clearance_finished_s": clearance_finished,
                            "pre_boundary_lateness_s": pre_boundary_late,
                            "acquisition_lateness_s": acquisition_late,
                            "command_ready_lateness_s": command_ready_late,
                            "state": measured,
                            "clearance_mm": clearance,
                        }
                    )
                    self._require_model_clock_deadline(
                        phase=f"q{index} acquisition",
                        boundary_s=boundary,
                        completed_s=read_finished,
                    )
                    self._require_model_clock_deadline(
                        phase=f"q{index} clearance",
                        boundary_s=boundary,
                        completed_s=clearance_finished,
                    )

                write_started = time.perf_counter()
                write_ready_late = max(0.0, write_started - boundary)
                lateness.append(write_ready_late)
                self._require_model_clock_deadline(
                    phase=f"command {index} write start",
                    boundary_s=boundary,
                    completed_s=write_started,
                )
                with self.command_gate:
                    if stop_event.is_set():
                        raise RuntimeError("execution stopped by operator")
                    motion_attempted = True
                    actual = self._write_position_command(command, previous_command)
                write_finished = time.perf_counter()
                write_finished_late = max(0.0, write_finished - boundary)
                lateness.append(write_finished_late)
                clock_records.append(
                    {
                        "event": "v177_command_intent",
                        "command_index": index,
                        "boundary_s": boundary,
                        "read_started_s": read_started,
                        "read_finished_s": read_finished,
                        "clearance_finished_s": clearance_finished,
                        "write_started_s": write_started,
                        "write_finished_s": write_finished,
                        "write_ready_lateness_s": write_ready_late,
                        "write_finished_lateness_s": write_finished_late,
                        "requested": command,
                        "previous_command": previous_command,
                    }
                )
                if not np.allclose(actual, command, atol=1.0e-4, rtol=0.0):
                    raise RuntimeError(
                        "LeRobot clipped or changed a V17.7 command"
                    )
                sent.append(np.asarray(actual, dtype=np.float32).copy())
                previous_command = np.asarray(actual, dtype=np.float32).copy()
                self._require_model_clock_deadline(
                    phase=f"command {index} write",
                    boundary_s=boundary,
                    completed_s=write_finished,
                )

            final_boundary = started + 5 * period_s
            delay = final_boundary - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
            final_read_started = time.perf_counter()
            final_pre_boundary_late = max(
                0.0, final_read_started - final_boundary
            )
            lateness.append(final_pre_boundary_late)
            self._require_model_clock_deadline(
                phase="q5 read start",
                boundary_s=final_boundary,
                completed_s=final_read_started,
            )
            q5 = self._read_state_only()
            final_read_finished = time.perf_counter()
            final_acquisition_late = max(
                0.0, final_read_finished - final_boundary
            )
            lateness.append(final_acquisition_late)
            q5_clearance = self.clearance_gate.require_readback(
                q5, label="q5 readback"
            )
            final_clearance_finished = time.perf_counter()
            final_ready_late = max(
                0.0, final_clearance_finished - final_boundary
            )
            lateness.append(final_ready_late)
            readback_clearances.append(q5_clearance)
            clock_records.append(
                {
                    "event": "v177_model_readback",
                    "path_index": 5,
                    "boundary_s": final_boundary,
                    "read_started_s": final_read_started,
                    "read_finished_s": final_read_finished,
                    "clearance_finished_s": final_clearance_finished,
                    "pre_boundary_lateness_s": final_pre_boundary_late,
                    "acquisition_lateness_s": final_acquisition_late,
                    "command_ready_lateness_s": final_ready_late,
                    "state": q5,
                    "clearance_mm": q5_clearance,
                }
            )
            if stop_event.is_set():
                raise RuntimeError("execution stopped by operator")
            self._require_model_clock_deadline(
                phase="q5 acquisition",
                boundary_s=final_boundary,
                completed_s=final_read_finished,
            )
            self._require_model_clock_deadline(
                phase="q5 clearance",
                boundary_s=final_boundary,
                completed_s=final_clearance_finished,
            )
            q_path.append(np.asarray(q5, dtype=np.float32).copy())
            transition_duration = final_read_finished - started

            # Preserve the exact 10 Hz q0..q5 path for LeWM's readback
            # residual, while pairing the current images and next Actor source
            # with the real quiet endpoint.  Do not servo the arm back to an
            # earlier sample: that would be an unmodelled hidden action.
            settle_records: list[dict[str, Any]] = []
            settle_clearances: list[float] = []
            try:
                settled_observation, _, _ = self._observe_settled_actual_endpoint(
                    q5,
                    period_s=period_s,
                    stop_event=stop_event,
                    records=settle_records,
                    clearances=settle_clearances,
                )
            finally:
                clock_records.extend(settle_records)
                readback_clearances.extend(settle_clearances)
            transition_observation = settled_observation
            settled_drift = float(
                np.max(
                    np.abs(
                        settled_observation.state - q5
                    )
                )
            )
            original_joint_residual = float(
                np.max(np.abs(settled_observation.state[:5] - sent[-1][:5]))
            )
            settlement = V177HeldEndpoint(
                outcome="settled_actual_endpoint",
                commanded_state=sent[-1].copy(),
                actual_state=settled_observation.state.copy(),
                joint_residual=original_joint_residual,
                step_delta=settled_drift,
                reached_commanded_endpoint=bool(
                    original_joint_residual
                    <= float(self.settle_position_tolerance) + 1.0e-4
                ),
            )
            measured_q_path = np.stack(q_path).astype(np.float32, copy=False)
            clock_records.append(
                {
                    "event": "v177_clocked_chunk_completed",
                    "monotonic_s": time.perf_counter(),
                    "measured_q_path": measured_q_path,
                    "timed_q5_state": q5,
                    "timed_q5_timestamp_s": final_read_finished,
                    "actual_endpoint_timestamp_s": settled_observation.timestamp_s,
                    "settled_state": settled_observation.state,
                    "maximum_clock_lateness_s": max(lateness, default=0.0),
                    "endpoint_outcome": settlement.outcome,
                    "minimum_readback_clearance_mm": min(readback_clearances),
                }
            )
            self._flush_clock_records(journal_path, clock_records)
            return V177ClockedExecution(
                requested=values.copy(),
                sent=np.stack(sent),
                measured_q_path=measured_q_path,
                transition_observation=transition_observation,
                settled_observation=settled_observation,
                transition_duration_s=float(transition_duration),
                duration_s=float(time.perf_counter() - started),
                maximum_clock_lateness_s=float(max(lateness, default=0.0)),
                minimum_readback_clearance_mm=float(min(readback_clearances)),
                command_journal=(
                    None if journal_path is None else str(journal_path)
                ),
                settlement=settlement,
            )
        except BaseException as exc:
            hold_error: str | None = None
            hold_record: dict[str, Any] | None = None
            if sent or motion_attempted:
                try:
                    with self.command_gate:
                        hold_record = self._hold_at_measured_pose(
                            None, reason="v177_clocked_chunk_fault"
                        )
                except BaseException as hold_exc:
                    hold_error = f"{type(hold_exc).__name__}: {hold_exc}"
            if hold_record is not None:
                clock_records.append(hold_record)
            clock_records.append(
                {
                    "event": "v177_clocked_chunk_fault",
                    "monotonic_s": time.perf_counter(),
                    "sent_commands_confirmed": len(sent),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "safe_hold_error": hold_error,
                }
            )
            journal_error: str | None = None
            try:
                self._flush_clock_records(journal_path, clock_records)
            except BaseException as journal_exc:
                journal_error = f"{type(journal_exc).__name__}: {journal_exc}"
            if hold_error is not None:
                raise RuntimeError(
                    f"V17.7 transport fault ({exc}); safe hold also failed "
                    f"({hold_error})"
                ) from exc
            if journal_error is not None:
                raise RuntimeError(
                    f"V17.7 transport fault ({exc}); post-hold journal flush "
                    f"also failed ({journal_error})"
                ) from exc
            raise


class V177PlanRefreshed(RuntimeError):
    """The requested plan was replaced safely and no command was sent."""


class V177UnsafePlan(RuntimeError):
    """A policy proposal failed deployment checks before transport."""


def _require_v177_pre_lewm_admission(
    state: np.ndarray,
    commands: np.ndarray,
    *,
    clearance_gate: V177ClearanceGate,
) -> None:
    """Reject a physically illegal Actor block before action-conditioned GDM."""

    audit = _audit_v177_chunk(
        commands,
        state,
        clearance_gate=clearance_gate,
    )
    if not bool(audit["safe"]):
        detail = audit.get("clearance_error")
        raise V177UnsafePlan(
            "V17.7 command rejected before LeWM: "
            + ", ".join(audit["reasons"])
            + ("; " + str(detail) if detail else "")
        )


class V177PlanRefreshRequired(RuntimeError):
    """A stale displayed plan was replaced without sending any command."""


class V177ClockDeadlineMiss(RuntimeError):
    """A read or command could not preserve the frozen 10 Hz model clock."""


class V177HardwareService(service_base.V175HardwareService):
    """V17.7 runtime on the existing single production hardware path."""

    def __init__(
        self,
        *args: Any,
        dino_target_profile: Path = DEFAULT_DINO_TARGET_PROFILE,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Session-only, display-only target references.  They are deliberately
        # outside the frozen policy/runtime objects and are never persisted to
        # disk or passed to a learned/control component.  Their fixed display
        # references survive replanning so a new round does not silently erase
        # the operator's PICKUP/DROP annotations.  Pixel coordinates do not:
        # each later round must localize the saved appearance in its own frame.
        self._operator_target_tracker = VisualTargetTracker(None)
        self._operator_targets: dict[tuple[str, str], dict[str, Any]] = {}
        self._operator_target_revision = 0
        # Automatic dashboard grounding uses the already registered reference
        # crops for this physical fish/box task and frozen DINOv2 matching.
        # It remains completely outside policy/runtime inputs.
        self._dino_target_profile = Path(dino_target_profile).expanduser().resolve()
        # Loading this display-only tracker must never prevent the hardware
        # controller from starting.  Construct it lazily on the first observed
        # frame and turn every failure into an unavailable overlay record.
        self._dino_target_tracker: VisualTargetTracker | None = None
        self._dino_targets: dict[str, Any] = {
            "version": "17.7.0",
            "source": "frozen_dinov2_profile_prototype",
            "display_only": True,
            "control_input": False,
            "targets": {"pickup": {}, "drop_zone": {}},
            "revision": 0,
            "source_timestamp_s": None,
        }
        self._dino_target_revision = 0

    def _refresh_dino_targets(self, observation: LiveObservation) -> None:
        """Automatically localize fish/box for the dashboard only."""

        tracker = getattr(self, "_dino_target_tracker", None)
        profile = getattr(self, "_dino_target_profile", None)
        if tracker is None and profile is None:
            # Unit fixtures may intentionally bypass ``__init__``.
            return
        try:
            if tracker is None:
                tracker = VisualTargetTracker(Path(profile))
                self._dino_target_tracker = tracker
            result = locate_dino_targets(
                observation.frames,
                tracker=tracker,
            )
        except Exception as exc:
            reason = f"localization_failed: {type(exc).__name__}: {exc}"
            result = {
                "version": "17.7.0",
                "source": "frozen_dinov2_profile_prototype",
                "display_only": True,
                "control_input": False,
                "targets": {
                    "pickup": {
                        view: {
                            "role": "pickup",
                            "view": view,
                            "visible": False,
                            "centroid_px": None,
                            "source": "frozen_dinov2_profile_prototype",
                            "reason": reason,
                        }
                        for view in ("wrist", "column")
                    },
                    "drop_zone": {
                        "column": {
                            "role": "drop_zone",
                            "view": "column",
                            "visible": False,
                            "centroid_px": None,
                            "source": "frozen_dinov2_profile_prototype",
                            "reason": reason,
                        }
                    },
                },
            }
        with self.state_lock:
            self._dino_target_revision = int(
                getattr(self, "_dino_target_revision", 0)
            ) + 1
            self._dino_targets = {
                **result,
                "revision": int(self._dino_target_revision),
                "source_timestamp_s": float(observation.timestamp_s),
                "profile": str(self._dino_target_profile),
            }

    def _dino_target_payload_locked(self) -> dict[str, Any]:
        payload = dict(getattr(self, "_dino_targets", {}) or {})
        targets = payload.get("targets")
        registrations: list[dict[str, Any]] = []
        if isinstance(targets, dict):
            for role in ("pickup", "drop_zone"):
                views = targets.get(role)
                if not isinstance(views, dict):
                    continue
                for view in ("wrist", "column"):
                    record = views.get(view)
                    if isinstance(record, dict):
                        registrations.append(dict(record))
        registrations.sort(key=lambda row: (str(row.get("view")), str(row.get("role"))))
        return {
            "source": str(
                payload.get("source", "frozen_dinov2_profile_prototype")
            ),
            "display_only": True,
            "control_input": False,
            "revision": int(payload.get("revision", 0)),
            "source_timestamp_s": payload.get("source_timestamp_s"),
            "profile": payload.get("profile"),
            "registrations": registrations,
        }

    def bootstrap(self) -> dict[str, Any]:
        """Keep HOME available when the initial Actor plan is rejected.

        An unsafe task proposal is not a service-start failure: it must remain
        unpublishable and unexecutable, while the separately audited HOME path
        stays available to recover the arm into the nominal planning envelope.
        """

        observed = self.io.observed_source_image_sizes()
        expected = tuple(
            int(value) for value in service_base.live_base.LIVE_SOURCE_IMAGE_SIZE
        )
        if any(tuple(size) != expected for size in observed.values()):
            raise RuntimeError(
                f"camera source size mismatch: {observed}, expected={expected}"
            )
        observation = self.io.observe()
        initial_plan_error: str | None = None
        try:
            self._plan_from(observation, replace_pending=False)
        except V177UnsafePlan as exc:
            initial_plan_error = f"{type(exc).__name__}: {exc}"
            with self.state_lock:
                self.last_observation = observation
                self.plan = None
                self.stopped = True
                self.stop_reason = "initial_plan_rejected"
                self.last_error = initial_plan_error
            self.stop_event.set()
        service_base.live_base._append_jsonl(
            self.events_path,
            {
                "event": "service_started",
                "wall_time_s": time.time(),
                "service_start_state_deg": observation.state,
                "home_state_deg": self.home_state,
                "motor_calibration": self.motor_calibration,
                "camera_source_sizes": observed,
                "initial_plan_error": initial_plan_error,
                "home_available_after_plan_rejection": True,
            },
        )
        return self.status()

    def _run_manual_target(
        self,
        target: np.ndarray,
        *,
        label: str,
        allow_gripper_recovery: bool = False,
    ) -> None:
        """Keep low-clearance recovery exclusive to the fixed HOME route."""

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
            if (
                label == "home"
                and self.io.clearance_gate.clearance_mm(live.state)
                < self.io.clearance_gate.planned_clearance_mm - 1.0e-9
            ):
                if not np.array_equal(goal, self.home_state):
                    raise RuntimeError(
                        "low-clearance recovery is restricted to the fixed HOME"
                    )
                recovery = np.repeat(live.state[None], 5, axis=0).astype(
                    np.float32
                )
                lift_end = max(
                    float(V177_DEPLOYMENT_POSITION_LOWER[1]),
                    float(live.state[1])
                    - 5.0 * HOME_RECOVERY_LIFT_STEP_POSITION,
                )
                recovery[:, 1] = np.linspace(
                    float(live.state[1]), lift_end, 6, dtype=np.float32
                )[1:]
                audit = _audit_v177_chunk(
                    recovery,
                    live.state,
                    clearance_gate=self.io.clearance_gate,
                )
                if not bool(audit["safe"]):
                    raise RuntimeError(
                        "home clearance-recovery command rejected: "
                        + ", ".join(audit["reasons"])
                    )
                self.io.execute_home_chunk(
                    recovery,
                    period_s=self.period_s,
                    stop_event=self.stop_event,
                )
                continue
            bounded_goal = goal.copy()
            near_lower = live.state <= SO101_POSITION_LOWER
            near_upper = live.state >= SO101_POSITION_UPPER
            bounded_goal[near_lower & (bounded_goal < live.state)] = live.state[
                near_lower & (bounded_goal < live.state)
            ]
            bounded_goal[near_upper & (bounded_goal > live.state)] = live.state[
                near_upper & (bounded_goal > live.state)
            ]
            commands = self._bounded_commands(live.state, bounded_goal)
            audit = _audit_v177_chunk(
                commands,
                live.state,
                clearance_gate=self.io.clearance_gate,
            )
            if (
                label == "home"
                and not bool(audit["safe"])
                and audit.get("reasons") == ["table_clearance"]
            ):
                commands, audit = self._select_safe_home_detour(
                    live.state, bounded_goal
                )
            if not bool(audit["safe"]):
                raise RuntimeError(
                    f"{label} command rejected: " + ", ".join(audit["reasons"])
                )
            if label == "home":
                self.io.execute_home_chunk(
                    commands,
                    period_s=self.period_s,
                    stop_event=self.stop_event,
                )
            else:
                self.io.execute_chunk(
                    commands,
                    period_s=self.period_s,
                    stop_event=self.stop_event,
                )
        raise RuntimeError(
            f"{label} did not converge inside its fixed budget: "
            f"residual={final_residual:.3f} position, "
            f"tolerance={tolerance:.3f} position"
        )

    def _select_safe_home_detour(
        self,
        current: np.ndarray,
        goal: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Choose a one-joint HOME step whose full path clears the table.

        A safe HOME endpoint does not imply that simultaneous joint-space
        interpolation to it is safe.  When the direct block is rejected only
        for table clearance, enumerate progress-toward-HOME moves for each arm
        joint, reject every candidate whose densely interpolated FK path
        violates the clearance contract, then prefer the candidate ending
        with the most clearance.  Nothing is sent until this filtering is
        complete.
        """

        source = np.asarray(current, dtype=np.float32)
        target = np.asarray(goal, dtype=np.float32)
        candidates: list[
            tuple[tuple[float, float, float], np.ndarray, dict[str, Any]]
        ] = []
        for joint in range(5):
            remaining = float(target[joint] - source[joint])
            if abs(remaining) <= 1.0e-6:
                continue
            endpoint = source.copy()
            maximum_change = 5.0 * HOME_RECOVERY_LIFT_STEP_POSITION
            endpoint[joint] = float(
                source[joint]
                + np.clip(remaining, -maximum_change, maximum_change)
            )
            chunk = np.repeat(source[None], 5, axis=0).astype(np.float32)
            chunk[:, joint] = np.linspace(
                float(source[joint]),
                float(endpoint[joint]),
                6,
                dtype=np.float32,
            )[1:]
            candidate_audit = _audit_v177_chunk(
                chunk,
                source,
                clearance_gate=self.io.clearance_gate,
            )
            if not bool(candidate_audit["safe"]):
                continue
            progress = abs(remaining) - abs(float(target[joint] - endpoint[joint]))
            score = (
                float(candidate_audit.get("final_planned_clearance_mm", -np.inf)),
                float(candidate_audit.get("minimum_planned_clearance_mm", -np.inf)),
                float(progress),
            )
            candidates.append((score, chunk, candidate_audit))

        if not candidates:
            raise RuntimeError(
                "home has no clearance-safe joint-order detour toward the fixed HOME"
            )
        candidates.sort(key=lambda item: item[0], reverse=True)
        _score, chunk, candidate_audit = candidates[0]
        return chunk, candidate_audit

    def _reset_operator_targets_locked(self) -> None:
        """Explicitly clear all session display annotations.

        Replanning must not call this helper.  The registered appearance may be
        reused for current-frame display localization, but a stale coordinate
        may never be copied into a later round.
        """

        self._operator_target_tracker = VisualTargetTracker(None)
        self._operator_targets = {}
        self._operator_target_revision = int(
            getattr(self, "_operator_target_revision", 0)
        ) + 1

    def _operator_target_payload_locked(self) -> dict[str, Any]:
        active_round = None
        if self.plan is not None:
            active_round = self.plan.get("round_id")
        registrations = []
        for record in getattr(self, "_operator_targets", {}).values():
            centroid = record.get("centroid_px")
            registered_centroid = np.asarray(
                record["registered_centroid_px"], dtype=np.float32
            )
            registrations.append(
                {
                    "role": str(record["role"]),
                    "view": str(record["view"]),
                    "centroid_px": (
                        None
                        if centroid is None
                        else [float(centroid[0]), float(centroid[1])]
                    ),
                    "radius_px": float(record["radius_px"]),
                    "visible": bool(record.get("visible", False)),
                    "localized_round_id": record.get("localized_round_id"),
                    "localization_source": record.get("localization_source"),
                    "localization_reason": record.get("localization_reason"),
                    "registered_round_id": str(record["round_id"]),
                    "registered_centroid_px": [
                        float(registered_centroid[0]),
                        float(registered_centroid[1]),
                    ],
                    "registered_source_timestamp_s": float(
                        record["source_timestamp_s"]
                    ),
                    "coordinate_mode": "current_frame_only",
                }
            )
        registrations.sort(key=lambda row: (row["view"], row["role"]))
        return {
            "round_id": active_round,
            "revision": int(getattr(self, "_operator_target_revision", 0)),
            "lifecycle": "session_reference_current_frame_relocalized",
            "control_input": False,
            "registrations": registrations,
        }

    def _refresh_operator_targets_locked(
        self,
        observation: LiveObservation,
        *,
        round_id: str,
    ) -> None:
        """Relocalize saved display references using only the new frame.

        This runs after the policy has produced and passed audit, so neither the
        reference nor the localization result can affect FCPG, LeWM, Actor, or
        the pending command.  A failed/held match clears the display coordinate
        for this round rather than copying a stale pixel forward.
        """

        records = getattr(self, "_operator_targets", {})
        if not records:
            return
        for record in records.values():
            role = str(record["role"])
            view = str(record["view"])
            # 10/31: the template tracker matched colour (VisualTargetTracker.track) and is gone from the code
            result = {
                "visible": False,
                "source": "current_frame_template_match",
                "reason": "template tracking removed (colour, 10/31)",
                "centroid_px": None,
            }
            source = str(result.get("source", "current_frame_template_match"))
            centroid = result.get("centroid_px")
            fresh = (
                bool(result.get("visible", False))
                and source == "appearance_template"
            )
            if fresh:
                point = np.asarray(centroid, dtype=np.float32)
                fresh = bool(point.shape == (2,) and np.all(np.isfinite(point)))
            if fresh:
                record["centroid_px"] = point.copy()
                record["visible"] = True
                record["localized_round_id"] = round_id
                record["localized_source_timestamp_s"] = float(
                    observation.timestamp_s
                )
                record["localization_source"] = "current_frame_template_match"
                record["localization_reason"] = None
            else:
                record["centroid_px"] = None
                record["visible"] = False
                record["localized_round_id"] = None
                record["localized_source_timestamp_s"] = None
                record["localization_source"] = source
                record["localization_reason"] = str(
                    result.get("reason") or "no_fresh_current_frame_match"
                )
        self._operator_target_revision = int(
            getattr(self, "_operator_target_revision", 0)
        ) + 1

    def register_operator_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Register a click on the exact current endpoint frame for display only.

        This method intentionally has no call into ``policy``, ``runtime`` or
        hardware I/O.  It cannot change the frozen model inputs or the pending
        command transaction.
        """

        round_id, role, view, centroid_px, radius_px = (
            _parse_operator_target_request(payload)
        )
        with self.state_lock:
            plan = self.plan
            observation = self.last_observation
            if plan is None or str(plan.get("round_id", "")) != round_id:
                raise RuntimeError("stale operator target planning round")
            if observation is None:
                raise RuntimeError("no endpoint observation is available")
            source_timestamp_s = float(plan.get("source_timestamp_s", float("nan")))
            if (
                not np.isfinite(source_timestamp_s)
                or abs(source_timestamp_s - float(observation.timestamp_s)) > 1.0e-6
            ):
                raise RuntimeError(
                    "operator target plan does not match the current endpoint frame"
                )
            frame = np.asarray(observation.frames[view]).copy()
            tracker = getattr(self, "_operator_target_tracker", None)
            if tracker is None:
                tracker = VisualTargetTracker(None)
                self._operator_target_tracker = tracker
            result = tracker.register(
                role,
                view,
                frame,
                centroid_px,
                radius_px=radius_px,
                persist=False,
                source="v177_current_plan_operator_annotation",
            )
            self._operator_targets[(role, view)] = {
                "role": role,
                "view": view,
                "centroid_px": np.asarray(result["centroid_px"], dtype=np.float32),
                "registered_centroid_px": np.asarray(
                    result["centroid_px"], dtype=np.float32
                ),
                "radius_px": radius_px,
                "round_id": round_id,
                "source_timestamp_s": source_timestamp_s,
                "visible": True,
                "localized_round_id": round_id,
                "localized_source_timestamp_s": source_timestamp_s,
                "localization_source": "operator_click_current_frame",
                "localization_reason": None,
            }
            self._operator_target_revision = int(
                getattr(self, "_operator_target_revision", 0)
            ) + 1
        return self.status()

    def _cancel_pending_display(self) -> None:
        """Withdraw runtime and display-only halves of an unexecuted plan."""

        self.policy.runtime.cancel_pending_proposal()
        self.policy.last_proposal = None
        self.policy._render_log_posterior = None

    def _invalidate_plan_transaction(self, error: str) -> None:
        """Atomically make both halves of an unpublished plan unexecutable."""

        self._cancel_pending_display()
        with self.state_lock:
            self.plan = None
            self.last_error = error

    def stop(self) -> dict[str, Any]:
        """Latch STOP and withdraw every unexecuted/displayed proposal."""

        self.stop_event.set()
        self._cancel_pending_display()
        with self.state_lock:
            self.plan = None
            self.stopped = True
            self.stop_reason = "operator_stop"
        service_base.live_base._append_jsonl(
            self.events_path,
            {"event": "operator_stop", "wall_time_s": time.time()},
        )
        return self.status()

    def home(self) -> dict[str, Any]:
        """Withdraw task state before moving, then publish one fresh HOME H0."""

        if self.home_state is None:
            raise RuntimeError("startup home state is unavailable")
        self._acquire("home")
        try:
            # A task proposal belongs to the pre-HOME endpoint and must become
            # unexecutable before the first manual HOME command is considered.
            self._cancel_pending_display()
            self.policy.reset()
            with self.state_lock:
                self.plan = None
                self.stopped = True
                self.stop_reason = "home_in_progress"
            self.stop_event.clear()
            try:
                self._run_manual_target(
                    self.home_state,
                    label="home",
                    allow_gripper_recovery=True,
                )
                observation = self.io.observe()
                self._plan_from(observation, replace_pending=False)
            except BaseException as exc:
                self.stop_event.set()
                self._cancel_pending_display()
                self.policy.reset()
                with self.state_lock:
                    self.plan = None
                    self.stopped = True
                    if self.stop_reason != "operator_stop":
                        self.stop_reason = "home_failed"
                    self.last_error = f"{type(exc).__name__}: {exc}"
                raise
            with self.state_lock:
                self.stopped = False
                self.stop_reason = None
                self.consecutive_endpoint_misses = 0
            return self.status()
        finally:
            self._release()

    def _plan_from(
        self, observation: LiveObservation, *, replace_pending: bool
    ) -> dict[str, Any]:
        """Run the control transaction and independently refresh display data."""

        try:
            return self._plan_from_control(
                observation, replace_pending=replace_pending
            )
        finally:
            # An unsafe Actor proposal must still expose the latest real camera
            # frame and its automatic TARGET/DROP overlay.  This state is not
            # model memory and cannot make a rejected plan executable.
            with self.state_lock:
                self.last_observation = observation
            self._refresh_dino_targets(observation)

    def _plan_from_control(
        self, observation: LiveObservation, *, replace_pending: bool
    ) -> dict[str, Any]:
        # Withdraw the old public plan before touching the policy transaction.
        # A proposal is published only after every deployment check succeeds.
        with self.state_lock:
            self.plan = None
        try:
            proposal = self.policy.propose(
                observation.frames,
                observation.state,
                replace_pending=replace_pending,
            )
            stop_event = getattr(self, "stop_event", None)
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError(
                    "V17.7 planning stopped by operator before publication"
                )
            commands = np.asarray(
                proposal["commands_position"], dtype=np.float32
            )
            audit = _audit_v177_chunk(
                commands,
                observation.state,
                clearance_gate=self.io.clearance_gate,
            )
            if not bool(audit["safe"]):
                detail = audit.get("clearance_error")
                raise V177UnsafePlan(
                    "V17.7 command rejected: "
                    + ", ".join(audit["reasons"])
                    + ("; " + str(detail) if detail else "")
                )
            generation = int(proposal["pending_generation"])
            pending_commands = self.policy.runtime.pending_commands
            pending_source = self.policy.runtime.pending_source_q
            if (
                self.policy.runtime.pending_generation != generation
                or pending_commands is None
                or pending_source is None
                or not np.array_equal(
                    pending_commands[0].float().cpu().numpy(), commands
                )
                or not np.array_equal(
                    pending_source[0].float().cpu().numpy(),
                    np.asarray(observation.state, dtype=np.float32),
                )
            ):
                raise RuntimeError(
                    "V17.7 pending prediction does not match the plan"
                )
            self._round_counter += 1
            plan = {
                "round_id": f"v177-{self._round_counter:06d}",
                "pending_generation": generation,
                "created_monotonic_s": time.monotonic(),
                "source_timestamp_s": observation.timestamp_s,
                "position_unit": POSITION_UNIT,
                "source_state_position": observation.state.copy(),
                "commands_position": commands,
                "safe": True,
                "reasons": [],
                "maximum_observed_step_per_joint_position": audit.get(
                    "maximum_observed_step_per_joint_position"
                ),
                "minimum_planned_clearance_mm": audit[
                    "minimum_planned_clearance_mm"
                ],
                "required_planned_clearance_mm": self.io.clearance_gate.planned_clearance_mm,
                "inference_s": proposal["inference_s"],
                "memory_chunks": proposal["memory_chunks"],
                "clock_phase": "H1" if proposal["memory_chunks"] else "H0",
                "innovation_used": bool(proposal["memory_chunks"]),
            }
        except BaseException as exc:
            self._invalidate_plan_transaction(f"{type(exc).__name__}: {exc}")
            raise
        with self.state_lock:
            stop_event = getattr(self, "stop_event", None)
            if stop_event is not None and stop_event.is_set():
                self._cancel_pending_display()
                self.plan = None
                raise RuntimeError(
                    "V17.7 planning stopped by operator before publication"
                )
            self.last_observation = observation
            self.plan = plan
            self._refresh_operator_targets_locked(
                observation, round_id=str(plan["round_id"])
            )
            self.last_error = None
        return plan

    def _pending_matches_plan(self, plan: dict[str, Any]) -> bool:
        pending_commands = self.policy.runtime.pending_commands
        pending_source = self.policy.runtime.pending_source_q
        if pending_commands is None or pending_source is None:
            return False
        return bool(
            self.policy.runtime.pending_generation
            == int(plan.get("pending_generation", -1))
            and np.array_equal(
                pending_commands[0].float().cpu().numpy(),
                np.asarray(plan.get("commands_position"), dtype=np.float32),
            )
            and np.array_equal(
                pending_source[0].float().cpu().numpy(),
                np.asarray(
                    plan.get("source_state_position"), dtype=np.float32
                ),
            )
        )

    def _same_endpoint(
        self, reference: np.ndarray, measured: np.ndarray
    ) -> tuple[bool, np.ndarray | None]:
        runtime = getattr(self.policy, "runtime", None)
        identity = getattr(runtime, "source_states_match", None)
        if callable(identity):
            matched = bool(identity(reference, measured))
            cell_delta_fn = getattr(runtime, "source_state_cell_delta", None)
            cell_delta = (
                cell_delta_fn(reference, measured)
                if callable(cell_delta_fn)
                else None
            )
            ticks = (
                None
                if cell_delta is None
                else cell_delta[0].detach().cpu().numpy().astype(np.int64)
            )
            return matched, ticks
        identity = getattr(self.io, "_same_model_endpoint", None)
        if callable(identity):
            return identity(reference, measured)
        drift = float(
            np.max(
                np.abs(
                    np.asarray(measured, dtype=np.float32)
                    - np.asarray(reference, dtype=np.float32)
                )
            )
        )
        return (
            drift <= float(self.io.settle_step_tolerance) + 1.0e-4,
            None,
        )

    def _fresh_plan_for_transport(self) -> tuple[LiveObservation, dict[str, Any]]:
        if self.policy.runtime.has_pending_chunk:
            with self.state_lock:
                plan = None if self.plan is None else dict(self.plan)
            if plan is None or not self._pending_matches_plan(plan):
                message = "V17.7 plan/pending transaction is inconsistent"
                self._invalidate_plan_transaction(message)
                raise RuntimeError(message)

            live = self.io.observe()
            now = time.monotonic()
            created = float(plan.get("created_monotonic_s", float("nan")))
            source_time = float(plan.get("source_timestamp_s", float("nan")))
            age = max(now - created, now - source_time)
            if not np.isfinite(age) or age < 0.0 or age > MAX_PLAN_AGE_S:
                self._plan_from(live, replace_pending=True)
                raise V177PlanRefreshRequired(
                    "displayed V17.7 image plan expired; a fresh plan is "
                    "displayed and was not executed"
                )

            source = np.asarray(
                plan["source_state_position"], dtype=np.float32
            )
            drift = float(np.max(np.abs(live.state - source)))
            source_matches, source_encoder_ticks = self._same_endpoint(
                source, live.state
            )
            if not source_matches:
                self._plan_from(live, replace_pending=True)
                raise V177PlanRefreshRequired(
                    "displayed V17.7 plan became stale because robot state "
                    "moved; a fresh plan is displayed and was not executed "
                    f"(drift={drift:.3f}, "
                    f"encoder_ticks={source_encoder_ticks})"
                )

            audit = _audit_v177_chunk(
                np.asarray(plan["commands_position"], dtype=np.float32),
                live.state,
                clearance_gate=self.io.clearance_gate,
            )
            reasons = set(audit["reasons"])
            if reasons:
                self._plan_from(live, replace_pending=True)
                raise RuntimeError(
                    "displayed V17.7 command no longer passes the current "
                    "audit; a fresh plan is displayed and was not executed"
                )
            return live, plan

        # H1+ planning must retain the exact completion endpoint consumed by
        # the policy, rather than attaching a newly captured image timestamp
        # to a proposal that never read that image.
        if self.policy.runtime.has_latest_actual:
            with self.state_lock:
                live = self.last_observation
            if live is None:
                raise RuntimeError(
                    "V17.7 has an actual belief without its source observation"
                )
        else:
            live = self.io.observe()
        plan = self._plan_from(live, replace_pending=False)
        # The H1 proposal consumes the actual q5 endpoint retained by the
        # runtime.  Read hardware once more immediately before transport and
        # admit only the same tolerance already enforced by that runtime.
        transport_live = self.io.observe()
        source = np.asarray(plan["source_state_position"], dtype=np.float32)
        drift = float(np.max(np.abs(transport_live.state - source)))
        source_matches, _source_encoder_ticks = self._same_endpoint(
            source, transport_live.state
        )
        audit = _audit_v177_chunk(
            np.asarray(plan["commands_position"], dtype=np.float32),
            transport_live.state,
            clearance_gate=self.io.clearance_gate,
        )
        if not source_matches or not bool(audit["safe"]):
            message = (
                "V17.7 retained endpoint changed before transport; no "
                "command was sent"
            )
            self._invalidate_plan_transaction(message)
            raise RuntimeError(message)
        return transport_live, plan

    def _execute_one(self) -> dict[str, Any]:
        """Execute with the trained q clock and keep settlement separate."""

        live, plan = self._fresh_plan_for_transport()
        commands = np.asarray(plan["commands_position"], dtype=np.float32)
        try:
            result = self.io.execute_v177_chunk(
                commands,
                initial_observation=live,
                period_s=self.period_s,
                stop_event=self.stop_event,
            )
        except BaseException as exc:
            service_base.live_base._append_jsonl(
                self.events_path,
                {
                    "event": "execution_error",
                    "wall_time_s": time.time(),
                    "round_id": plan["round_id"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "command_journal": self.io.last_command_journal,
                },
            )
            with self.state_lock:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.stopped = True
            self.stop_event.set()
            raise

        innovation: dict[str, Any] | None = None
        innovation_error: str | None = None
        feedback_retained = False
        settled_drift = float(
            np.max(
                np.abs(
                    result.settled_observation.state
                    - result.measured_q_path[-1]
                )
            )
        )
        settled_matches, settled_encoder_ticks = self._same_endpoint(
            result.measured_q_path[-1],
            result.settled_observation.state,
        )
        source = np.asarray(plan["source_state_position"], dtype=np.float32)
        measured_source = np.asarray(
            result.measured_q_path[0], dtype=np.float32
        )
        source_drift = float(np.max(np.abs(measured_source - source)))
        source_matches, source_encoder_ticks = self._same_endpoint(
            source, measured_source
        )
        clock_endpoint_compatible = bool(settled_matches)
        try:
            if not source_matches:
                innovation_error = (
                    "transport q0 changed from the verified proposal source; "
                    "next proposal must restart from a settled H0 observation"
                )
                self.policy.reset()
            elif not clock_endpoint_compatible:
                # The trained feedback transition ends at the timed 0.5 s q5.
                # The current image/q were captured later, after settle+hold.
                # Never silently pair those two clocks or stuff the settle
                # motion into q5's 0.1 s slot.  Restart from this truthful
                # settled observation as a fresh H0 instead.
                innovation_error = (
                    "timed q5 differs from the settled camera endpoint; "
                    "temporal feedback was discarded and restarted at H0"
                )
                self.policy.reset()
            else:
                innovation = self.policy.complete(
                    result.transition_observation.frames,
                    result.measured_q_path,
                    delta_t_s=result.transition_duration_s,
                    actual_endpoint_state_position=(
                        result.transition_observation.state
                    ),
                )
                feedback_retained = True
        except Exception as exc:
            innovation_error = f"{type(exc).__name__}: {exc}"
            self.policy.reset()

        tracking = self._observe_endpoint_outcome(
            reached=result.reached_commanded_endpoint
        )
        receipt = {
            "event": "chunk_completed",
            "wall_time_s": time.time(),
            "round_id": plan["round_id"],
            "position_unit": POSITION_UNIT,
            "clock_phase": plan["clock_phase"],
            "innovation_used": plan["innovation_used"],
            "requested_position": result.requested,
            "sent_position": result.sent,
            "measured_q_path_position": result.measured_q_path,
            "model_endpoint_state_position": result.transition_observation.state,
            "settled_state_position": result.settled_observation.state,
            "settled_vs_model_endpoint_drift_position": settled_drift,
            "settled_vs_model_endpoint_encoder_ticks": settled_encoder_ticks,
            "transport_q0_vs_plan_source_drift_position": source_drift,
            "transport_q0_vs_plan_source_encoder_ticks": source_encoder_ticks,
            "model_transition_duration_s": result.transition_duration_s,
            "total_duration_s": result.duration_s,
            "maximum_clock_lateness_s": result.maximum_clock_lateness_s,
            "minimum_planned_clearance_mm": plan[
                "minimum_planned_clearance_mm"
            ],
            "minimum_readback_clearance_mm": (
                result.minimum_readback_clearance_mm
            ),
            "feedback_state_retained": feedback_retained,
            "feedback_clock_endpoint_compatible": clock_endpoint_compatible,
            "feedback_clock_reset_to_h0": bool(
                source_matches and not clock_endpoint_compatible
            ),
            "settlement": {
                "outcome": result.settlement.outcome,
                "joint_residual": result.settlement.joint_residual,
                "actual_state": result.settlement.actual_state,
            },
            "innovation": innovation,
            **tracking,
            "command_journal": result.command_journal,
        }
        if innovation_error is not None:
            receipt["innovation_failure"] = innovation_error
        service_base.live_base._append_jsonl(self.events_path, receipt)
        with self.state_lock:
            self.completed_chunks += 1
            self.receipts.append(receipt)
            self.last_observation = (
                result.transition_observation
                if feedback_retained
                else result.settled_observation
            )
            self.last_error = innovation_error
        return receipt

    def execute(self, *, round_id: str, max_chunks: int) -> dict[str, Any]:
        """Execute a bounded batch after the mandatory H0-to-H1 canary."""

        if not isinstance(max_chunks, int) or isinstance(max_chunks, bool):
            raise ValueError("max_chunks must be an integer")
        if not 1 <= max_chunks <= MAX_CHUNKS_PER_REQUEST:
            raise ValueError(
                "V17.7 accepts at most two supervised chunks per request"
            )
        with self.state_lock:
            active = self.plan
            if self.completed_chunks == 0 and max_chunks != 2:
                raise RuntimeError(
                    "the first request must validate the complete H0-to-H1 pair"
                )
            if (
                self.completed_chunks > 0
                and not bool(self.continuous_feedback_clock_validated)
                and str((active or {}).get("clock_phase")) != "H0"
            ):
                raise RuntimeError(
                    "an unvalidated continuous-feedback plan cannot continue"
                )
            if self.stopped:
                raise RuntimeError("motion is stopped; restart the service")
            if active is None or round_id != active["round_id"]:
                raise RuntimeError("stale planning round")
        self._acquire("execute")
        completed = 0
        continuous_validated = bool(
            getattr(self, "continuous_feedback_clock_validated", False)
        )
        try:
            if self.completed_chunks == 0:
                with self.state_lock:
                    self.continuous_feedback_clock_validated = False
            self.stop_event.clear()
            batch_receipts: list[dict[str, Any]] = []
            feedback_clock_reset = False
            for _ in range(max_chunks):
                receipt = self._execute_one()
                batch_receipts.append(receipt)
                completed += 1
                if not bool(receipt["feedback_state_retained"]):
                    if bool(receipt.get("feedback_clock_reset_to_h0", False)):
                        feedback_clock_reset = True
                        break
                    raise RuntimeError("actual endpoint feedback could not be retained")
            if len(batch_receipts) >= 2:
                previous_receipt, latest_receipt = batch_receipts[-2:]
                continuous_validated = bool(
                    previous_receipt["clock_phase"] == "H0"
                    and latest_receipt["clock_phase"] == "H1"
                    and latest_receipt["innovation_used"]
                    and previous_receipt["feedback_state_retained"]
                    and latest_receipt["feedback_state_retained"]
                )
            if feedback_clock_reset:
                with self.state_lock:
                    next_observation = self.last_observation
                    self.continuous_feedback_clock_validated = False
                if next_observation is None:
                    raise RuntimeError(
                        "settled clock reset has no retained endpoint observation"
                    )
                self._plan_from(next_observation, replace_pending=False)
                with self.state_lock:
                    self.stopped = False
                    self.stop_reason = None
                return self.status()
            if not continuous_validated:
                raise RuntimeError(
                    "H0-to-H1 continuous feedback validation did not pass"
                )
            with self.state_lock:
                self.continuous_feedback_clock_validated = continuous_validated
                next_observation = self.last_observation
            if next_observation is None:
                raise RuntimeError(
                    "completed feedback has no retained endpoint observation"
                )
            self._plan_from(next_observation, replace_pending=False)
            with self.state_lock:
                self.stopped = False
                self.stop_reason = None
            return self.status()
        except V177PlanRefreshRequired:
            # No command was sent.  `_fresh_plan_for_transport` has already
            # installed a new, audited pending proposal, so retain both that
            # proposal and all completed belief/LeWM state for an explicit
            # retry using the new round id.
            with self.state_lock:
                self.stopped = False
                self.stop_reason = None
            raise
        except BaseException as exc:
            self.policy.reset()
            self._invalidate_plan_transaction(f"{type(exc).__name__}: {exc}")
            with self.state_lock:
                self.stopped = True
                if self.stop_reason != "operator_stop":
                    self.stop_reason = "v177_canary_failed"
            self.stop_event.set()
            raise
        finally:
            service_base.live_base._append_jsonl(
                self.events_path,
                {
                    "event": "v177_supervised_batch_finished",
                    "wall_time_s": time.time(),
                    "requested_chunks": max_chunks,
                    "completed_chunks": completed,
                    "continuous_feedback_clock_validated": continuous_validated,
                },
            )
            self._release()

    def status(self) -> dict[str, Any]:
        payload = super().status()
        payload["version"] = "17.7.0-live"
        payload["checkpoint_model_version"] = "17.7.0"
        payload["camera_aware_belief_grounding"] = True
        payload["fish_source_index"] = int(self.policy.source_index)
        payload["position_unit"] = POSITION_UNIT
        if "home_state_deg" in payload:
            payload["home_state_position"] = payload.pop("home_state_deg")
        payload["point_coordinates_present"] = False
        payload["diagnostic_belief_present"] = True
        payload["diagnostic_belief_mean_coordinates_present"] = True
        payload["diagnostic_anchor_coordinates_present"] = False
        payload["phase_state_present"] = False
        motor = dict(payload.get("motor_calibration") or {})
        motor.update(
            {
                "constraint_mode": "calibrated_hard_physical_only",
                "deployment_lower": V177_DEPLOYMENT_POSITION_LOWER,
                "deployment_upper": V177_DEPLOYMENT_POSITION_UPPER,
                "joint_limit_guard": 0.0,
                "gripper_operational_closed": float(
                    V177_DEPLOYMENT_POSITION_LOWER[-1]
                ),
                "gripper_operational_open": float(
                    V177_DEPLOYMENT_POSITION_UPPER[-1]
                ),
            }
        )
        payload["motor_calibration"] = motor
        payload["physical_constraint_mode"] = (
            "calibrated_joint_hard_limits_plus_table_z_zero"
        )
        payload["continuous_feedback_clock_validated"] = bool(
            getattr(self, "continuous_feedback_clock_validated", False)
        )
        payload["max_auto_chunks"] = MAX_CHUNKS_PER_REQUEST
        payload["execution_batch_max_chunks"] = MAX_CHUNKS_PER_REQUEST
        payload["supervised_task_chunk_budget"] = None
        payload["supervised_task_chunks_remaining"] = None
        payload["task_stop_condition"] = "operator_or_physical_fault_only"
        payload["functions"]["automatic_run"] = False
        payload["functions"]["execute_h0_h1"] = True
        payload["functions"]["operator_target_annotation"] = False
        payload["functions"]["automatic_dinov2_targets"] = True
        payload["dino_target_coordinates_present"] = True
        payload["dino_target_control_input"] = False
        lock = getattr(self, "state_lock", None)
        if lock is None:
            payload["operator_targets"] = {
                "round_id": None,
                "revision": 0,
                "lifecycle": "session_reference_current_frame_relocalized",
                "control_input": False,
                "registrations": [],
            }
        else:
            with lock:
                payload["operator_targets"] = self._operator_target_payload_locked()
                payload["dino_targets"] = self._dino_target_payload_locked()
        if "dino_targets" not in payload:
            payload["dino_targets"] = {
                "source": "frozen_dinov2_profile_prototype",
                "display_only": True,
                "control_input": False,
                "revision": 0,
                "source_timestamp_s": None,
                "profile": None,
                "registrations": [],
            }
        gate = getattr(getattr(self, "io", None), "clearance_gate", None)
        if gate is not None:
            payload["clearance_contract"] = {
                "distance_unit": "mm",
                "table_z_mm": gate.table_z_mm,
                "planned_minimum_mm": gate.planned_clearance_mm,
                "readback_minimum_mm": gate.readback_clearance_mm,
            }
        return payload

    def next_chunk_plan_image(self, view: str) -> np.ndarray:
        """Render current belief and image-space Actor/LeWM trajectories."""

        if view not in ("wrist", "column"):
            raise ValueError("plan view must be wrist or column")
        with self.state_lock:
            observation = self.last_observation
            plan = self.plan
            if observation is None:
                raise RuntimeError("no endpoint observation is available")
            frame = np.asarray(observation.frames[view]).copy()
            plan_copy = None if plan is None else dict(plan)
            active_round = None if plan is None else str(plan.get("round_id", ""))
            proposal = dict(self.policy.last_proposal or {})
            belief_reader = getattr(self.policy, "render_log_posterior", None)
            belief_log = (
                belief_reader()
                if plan is not None and callable(belief_reader)
                else None
            )
            target_records = [
                {
                    **record,
                    "centroid_px": (
                        None
                        if record.get("centroid_px") is None
                        else np.asarray(
                            record["centroid_px"], dtype=np.float32
                        ).copy()
                    ),
                }
                for record in getattr(self, "_operator_targets", {}).values()
                if record.get("view") == view
            ]
            dino_payload = self._dino_target_payload_locked()
            dino_records = [
                dict(record)
                for record in dino_payload["registrations"]
                if record.get("view") == view
            ]
            dino_timestamp = dino_payload.get("source_timestamp_s")
            if (
                dino_records
                and dino_timestamp is not None
                and abs(float(dino_timestamp) - float(observation.timestamp_s)) <= 1.0e-6
            ):
                # Automatic DINOv2 regions are authoritative for display.  The
                # legacy click records remain only as a hidden compatibility
                # path and are never mixed into a current automatic overlay.
                target_records = dino_records
        # The shared drawing helper predates the corrected unit names.  Keep
        # this translation private to rendering; no legacy key is returned by
        # the V17.7 service API.
        render_plan = None
        if plan_copy is not None:
            render_plan = dict(plan_copy)
            render_plan["source_state_deg"] = render_plan[
                "source_state_position"
            ]
            render_plan["commands_deg"] = render_plan["commands_position"]
        image = service_base.live_base._render_next_chunk_plan(
            frame,
            view=view,
            plan=render_plan,
            predicted_q_path=proposal.get("predicted_q_path_position"),
        )
        source_panel_size = service_base.live_base.PLAN_IMAGE_SIZE[1]
        scale = float(source_panel_size) / 224.0

        view_index = 0 if view == "wrist" else 1
        _draw_belief_overlay(
            image,
            belief_log,
            proposal.get("belief_mean_px"),
            view_index=view_index,
            source_panel_size=source_panel_size,
        )
        # Replace the inherited V18.1 title: this panel is no longer a raw
        # frame beside an unrelated joint chart.
        cv2.rectangle(
            image, (0, 0), (source_panel_size - 1, 31), (13, 18, 22), -1
        )
        cv2.putText(
            image,
            f"{view.upper()} | LF/LM BELIEF",
            (9, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (236, 242, 244),
            1,
            cv2.LINE_AA,
        )
        styles = {
            "pickup": ((57, 211, 152), "TARGET"),
            "drop_zone": ((255, 184, 72), "DROP"),
        }
        for record in target_records:
            if (
                not bool(record.get("visible", False))
                or record.get("centroid_px") is None
            ):
                continue
            if record.get("source") != "frozen_dinov2_profile_prototype" and (
                record.get("localized_round_id") != active_round
            ):
                continue
            color, label = styles[str(record["role"])]
            centroid = np.asarray(record["centroid_px"], dtype=np.float32)
            x, y = np.rint(centroid * scale).astype(int)
            x = int(np.clip(x, 0, source_panel_size - 1))
            y = int(np.clip(y, 0, source_panel_size - 1))
            axis = record.get("axis_px")
            length = record.get("length_px")
            width = record.get("width_px")
            if axis is not None and length is not None and width is not None:
                direction = np.asarray(axis, dtype=np.float32)
                major = max(8, int(round(0.5 * float(length) * scale)))
                minor = max(6, int(round(0.5 * float(width) * scale)))
                angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
                cv2.ellipse(
                    image,
                    (x, y),
                    (major, minor),
                    angle,
                    0.0,
                    360.0,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                radius = max(major, minor)
            else:
                radius = max(
                    7, int(round(float(record.get("radius_px", 18.0)) * scale))
                )
                cv2.circle(image, (x, y), radius, color, 2, cv2.LINE_AA)
            cv2.drawMarker(
                image,
                (x, y),
                color,
                cv2.MARKER_CROSS,
                14,
                2,
                cv2.LINE_AA,
            )
            text_x = min(x + 9, source_panel_size - 67)
            text_y = max(18, y - radius - 5)
            cv2.putText(
                image,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color,
                1,
                cv2.LINE_AA,
            )

        configured_roles = {str(record["role"]) for record in target_records}
        present_roles = {
            str(record["role"])
            for record in target_records
            if bool(record.get("visible", False))
            and record.get("centroid_px") is not None
            and (
                record.get("source") == "frozen_dinov2_profile_prototype"
                or record.get("localized_round_id") == active_round
            )
        }
        missing = []
        if "pickup" not in present_roles:
            missing.append(
                "TARGET NOT LOCALIZED"
                if "pickup" in configured_roles
                else "TARGET UNAVAILABLE"
            )
        if view == "column" and "drop_zone" not in present_roles:
            missing.append(
                "DROP NOT LOCALIZED"
                if "drop_zone" in configured_roles
                else "DROP UNAVAILABLE"
            )
        if missing:
            message = " / ".join(missing) + " (DINOv2 DISPLAY ONLY)"
            width = min(source_panel_size - 10, 7 * len(message) + 10)
            cv2.rectangle(
                image,
                (5, source_panel_size - 22),
                (5 + width, source_panel_size - 4),
                (13, 18, 22),
                -1,
            )
            cv2.putText(
                image,
                message,
                (9, source_panel_size - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.31,
                (210, 218, 222),
                1,
                cv2.LINE_AA,
            )
        return image


V177_HTML = (
    service_base.V175_HTML.replace("V17.5", "V17.7")
    .replace("双视角 Grounded-Anchor 闭环", "双视角 Predictive-Belief 闭环")
    .replace("ANCHOR-INNOVATION", "BELIEF-INNOVATION")
    .replace("视觉 innovation RMS", "信念 innovation RMS")
    .replace("completion.visual_innovation_rms", "completion.belief_innovation_rms")
    .replace("+' / 3'", "+' chunks'")
    .replace(
        "F/M/C 仅作结构化观测诊断，不进入控制",
        "空间信念仅作观测诊断，硬像素坐标不进入控制",
    )
    .replace("Grounded-Anchor", "Predictive-Belief")
    .replace("自动运行任务", "自动运行（时钟校准前禁用）")
    .replace("$('autoExecute').disabled=!ready", "$('autoExecute').disabled=true")
    .replace("commands_deg", "commands_position")
    .replace("source_state_deg", "source_state_position")
    .replace(
        "maximum_step_per_joint_deg",
        "maximum_observed_step_per_joint_position",
    )
    .replace("state_innovation_rms_deg", "state_innovation_rms_position")
    .replace("predicted_q_path_deg", "predicted_q_path_position")
    .replace("最大动作步长", "观测到的最大命令变化")
    .replace("执行 5 步", "执行 H0→H1（10 步）")
    .replace("$('execute').onclick=()=>execute(1)", "$('execute').onclick=()=>execute(2)")
    .replace(
        "五步命令已通过电机标定边界、任务夹爪边界、步长与 projector 检查",
        "命令已通过关节边界、毫米高度与 projector 检查",
    )
)

V177_HTML = (
    V177_HTML.replace(
        "</style>",
        """
.dino-target-tools{grid-column:1/-1;background:#fff;border:1px solid #c8cdd2;border-left:4px solid #235c82;border-radius:6px;padding:10px 12px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.dino-target-tools strong{font-size:13px}.dino-target-note{font-size:12px;color:#59616b}.dino-target-state{margin-left:auto;font:12px ui-monospace,monospace;color:#334155}
</style>""",
    )
    .replace(
        '<article class="panel"><h2>Wrist · 下一动作块</h2>',
        '<section class="dino-target-tools" id="dinoTargetTools"><strong>DINOv2 自动目标区域</strong><span class="dino-target-note">TARGET=红鱼，DROP=盒子；每次重新观测自动定位，仅用于 Dashboard 显示，不进入 FCPG / LeWM / Actor。</span><span class="dino-target-state" id="dinoTargetState">等待当前帧定位</span></section><article class="panel"><h2>Wrist · 下一动作块</h2>',
    )
    .replace("Wrist · 下一动作块", "Wrist · 当前画面 + 下一动作图像轨迹")
    .replace("Column · 下一动作块", "Column · 当前画面 + 下一动作图像轨迹")
    .replace(
        '<span>实线 ACTOR COMMAND</span><span>虚线 LEWM READBACK</span>',
        '<span>LF/LM=FCPG learned belief（青F/橙M）</span><span>TARGET/DROP=DINOv2自动区域（仅显示）</span>',
    )
    .replace(
        "function refreshPlanImages(plan){const round=plan&&plan.round_id;if(!round||round===renderedPlanRound)return;renderedPlanRound=round;const stamp=encodeURIComponent(round);$('planWrist').src='/api/plan-image/wrist.jpg?round='+stamp;$('planColumn').src='/api/plan-image/column.jpg?round='+stamp}",
        "function refreshPlanImages(plan,targets){const round=plan&&plan.round_id||'observation';const revision=Number(targets&&targets.revision||0),timestamp=targets&&targets.source_timestamp_s||0;const renderKey=round+':'+revision+':'+timestamp;if(renderKey===renderedPlanRound)return;renderedPlanRound=renderKey;const stamp=encodeURIComponent(round),annotation=encodeURIComponent(String(revision));const load=(id,view)=>{$(id).src='/api/plan-image/'+view+'.jpg?round='+stamp+'&annotation='+annotation+'&t='+encodeURIComponent(String(timestamp))};load('planWrist','wrist');load('planColumn','column')}",
    )
    .replace(
        "renderCommands(plan);refreshPlanImages(plan);const diagnostics=",
        "renderCommands(plan);renderDinoTargets(status);refreshPlanImages(plan,status.dino_targets);const diagnostics=",
    )
    .replace(
        "functions:status.functions}",
        "dino_targets:status.dino_targets,functions:status.functions}",
    )
    .replace(
        "function attachLiveStream(id,path)",
        """function renderDinoTargets(status){const targetStatus=status&&status.dino_targets||{};const rows=Array.isArray(targetStatus.registrations)?targetStatus.registrations:[],label=row=>row.view+(row.visible?'':' (未定位)');const target=rows.filter(row=>row.role==='pickup').map(label).join('+')||'-',dropRow=rows.find(row=>row.role==='drop_zone'&&row.view==='column'),drop=dropRow?label(dropRow):'-';$('dinoTargetState').textContent='TARGET '+target+' / DROP '+drop}
function attachLiveStream(id,path)""",
    )
    .replace(
        "$('execute').onclick=()=>execute(2)",
        "$('execute').onclick=()=>execute(2)",
    )
)


def handler_factory(controller: V177HardwareService) -> type:
    """Extend the established service handler with one non-control endpoint."""

    base_handler = service_base.live_base.handler_factory(controller)

    class Handler(base_handler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path.partition("?")[0] != "/api/operator-target":
                super().do_POST()
                return
            try:
                result = controller.register_operator_target(self._body())
                self._json(result)
            except Exception as exc:
                payload = controller.status()
                payload["request_error"] = f"{type(exc).__name__}: {exc}"
                self._json(payload, status=409)

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dino-repository", type=Path, default=DEFAULT_DINO_REPOSITORY)
    parser.add_argument("--dino-weights", type=Path, default=DEFAULT_DINO_WEIGHTS)
    parser.add_argument(
        "--dino-target-profile", type=Path, default=DEFAULT_DINO_TARGET_PROFILE
    )
    parser.add_argument(
        "--real-calibration", type=Path, default=DEFAULT_REAL_CALIBRATION
    )
    parser.add_argument(
        "--kinematics-model", type=Path, default=DEFAULT_KINEMATICS_MODEL
    )
    parser.add_argument("--table-z-mm", type=float, default=TABLE_Z_MM)
    parser.add_argument(
        "--planned-clearance-mm", type=float, default=PLANNED_CLEARANCE_MM
    )
    parser.add_argument(
        "--readback-clearance-mm", type=float, default=READBACK_CLEARANCE_MM
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "service")
    parser.add_argument("--robot-port", default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--robot-id", default="myfollower01")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--wrist-device", default=DEFAULT_WRIST_DEVICE)
    parser.add_argument("--column-device", default=DEFAULT_COLUMN_DEVICE)
    parser.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument("--settle-timeout-s", type=float, default=20.0)
    parser.add_argument("--settle-position-tolerance", type=float, default=3.0)
    parser.add_argument("--settle-step-tolerance", type=float, default=0.25)
    parser.add_argument("--settle-samples", type=int, default=3)
    parser.add_argument(
        "--open-gripper-position",
        type=float,
        default=DEFAULT_GRIPPER_OPEN_POSITION,
    )
    parser.add_argument(
        "--home-state-report",
        type=Path,
        default=REPO / "configs/prod_v17_7_legacy_v17_4_home.json",
    )
    parser.add_argument(
        "--home-gripper-position",
        type=float,
        default=31.340341567993164,
    )
    parser.add_argument(
        "--max-auto-chunks",
        type=int,
        default=MAX_CHUNKS_PER_REQUEST,
    )
    parser.add_argument(
        "--operator-present-workspace-clear",
        action="store_true",
        help="explicit per-start acknowledgement required before hardware opens",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--hardware-lock", type=Path, default=DEFAULT_HARDWARE_LOCK)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main() -> None:
    args = parse_args()
    if float(args.control_hz) != DEFAULT_CONTROL_HZ:
        raise ValueError("V17.7 command clock is fixed at 10 Hz")
    if int(args.max_auto_chunks) < MAX_CHUNKS_PER_REQUEST:
        raise ValueError(
            "V17.7 transport must permit the two-chunk H0-to-H1 request"
        )
    if not bool(args.operator_present_workspace_clear):
        raise ValueError(
            "hardware start requires --operator-present-workspace-clear for "
            "this live session"
        )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    home_state = _load_v177_home_state(
        args.home_state_report,
        gripper_position=args.home_gripper_position,
    )
    kinematics = SO101MuJoCoKinematics(args.kinematics_model)
    clearance_gate = V177ClearanceGate(
        kinematics=kinematics,
        table_z_mm=args.table_z_mm,
        planned_clearance_mm=args.planned_clearance_mm,
        readback_clearance_mm=args.readback_clearance_mm,
    )
    home_clearance_mm = clearance_gate.clearance_mm(home_state)
    if home_clearance_mm < clearance_gate.planned_clearance_mm - 1.0e-9:
        raise ValueError(
            "configured HOME fails V17.7 table clearance: "
            f"{home_clearance_mm:.2f} mm < "
            f"{clearance_gate.planned_clearance_mm:.2f} mm"
        )
    policy = V177LivePolicy(
        checkpoint=args.checkpoint,
        dino_repository=args.dino_repository,
        dino_weights=args.dino_weights,
        real_calibration=args.real_calibration,
        device=args.device,
        action_lower_position=V177_DEPLOYMENT_POSITION_LOWER,
        action_upper_position=V177_DEPLOYMENT_POSITION_UPPER,
    )
    policy.bind_command_admission_provider(
        lambda state, commands: _require_v177_pre_lewm_admission(
            state,
            commands,
            clearance_gate=clearance_gate,
        )
    )
    io_args = SimpleNamespace(
        wrist_device=args.wrist_device,
        column_device=args.column_device,
        output=output,
        robot_port=args.robot_port,
        robot_id=args.robot_id,
        calibration_dir=args.calibration_dir,
        # No V17.7 service-level waypoint delta cap.  This value only keeps
        # the inherited transport finite inside calibrated hard bounds; the
        # frozen Actor projector remains part of the checkpoint contract.
        max_command_jump=float(
            np.max(SO101_POSITION_UPPER - SO101_POSITION_LOWER) + 1.0e-4
        ),
        settle_timeout_s=args.settle_timeout_s,
        settle_position_tolerance=args.settle_position_tolerance,
        settle_step_tolerance=args.settle_step_tolerance,
        settle_samples=args.settle_samples,
        self_contact_calibration=False,
        i_accept_empty_jaw_self_contact=False,
        experiment_binding=None,
    )
    lease: SO101HardwareLease | None = None
    io: Any | None = None
    controller: V177HardwareService | None = None
    server: ThreadingHTTPServer | None = None
    original_html = service_base.live_base.HTML
    try:
        lease = acquire_so101_hardware_lease(
            lock_path=args.hardware_lock,
            serial_device=args.robot_port,
            wrist_device=args.wrist_device,
            column_device=args.column_device,
        )
        io = V177HardwareSO101IO(
            io_args,
            clearance_gate=clearance_gate,
            gripper_calibration_closed=DEFAULT_GRIPPER_CLOSED_POSITION,
            gripper_calibration_open=DEFAULT_GRIPPER_OPEN_POSITION,
            endpoint_envelope=EndpointEnvelope(
                max_joint_miss=float(args.settle_position_tolerance),
                max_tcp_miss_m=gripper_capture_radius_m(kinematics),
                forward=kinematics.forward,
            ),
        )
        policy.bind_source_state_encoder(
            io.calibration_reconstructed_encoder_cells
        )
        if not policy.runtime.has_bound_source_state_encoder:
            raise RuntimeError(
                "hardware source-state encoder was not bound before bootstrap"
            )
        if not policy.runtime.has_bound_command_admission_provider:
            raise RuntimeError(
                "physical command admission was not bound before bootstrap"
            )
        controller = V177HardwareService(
            policy,
            io,
            dino_target_profile=args.dino_target_profile,
            output=output,
            control_hz=args.control_hz,
            max_auto_chunks=args.max_auto_chunks,
            open_gripper_deg=args.open_gripper_position,
            home_state=home_state,
            motor_calibration=service_base.live_base._load_motor_calibration_contract(
                io.follower.calibration_fpath
            ),
        )
        controller.bootstrap()
        service_base.live_base.HTML = V177_HTML
        server = ThreadingHTTPServer(
            (args.host, args.port),
            handler_factory(controller),
        )
        server.daemon_threads = True
        print(
            f"SO-101 V17.7 direct hardware service: "
            f"http://{args.host}:{args.port}",
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
        service_base.live_base.HTML = original_html
        if server is not None:
            server.server_close()
        if controller is not None:
            controller.close()
        elif io is not None:
            io.close()
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    main()
