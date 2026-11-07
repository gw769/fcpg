"""Empty-jaw self-contact predicates and image-only knife-edge observation.

Gate 1 offline contract.  The live planner's default ``calibrate_gripper``
pulse is unchanged: ``closed_state`` remains Present_Position[5] argmin of
that pulse and is **not** a contact predicate.

Contact is true only from the joint predicate:

* A) independent image edge fits of the two triangular knife edges show a
  closed or intersecting gap for ``N`` consecutive frames
* B) Present_Position is a consecutive plateau / near-zero speed while Goal
  is still commanded inward

Vision-only, plateau-only, command-arrived, argmin, and Load/Current peaks
are all insufficient.  Optional Load/Current/Velocity/Moving readings are
safety witnesses: all-zero or stuck-constant **decoded** series cannot produce
``contact=true`` and cannot arm a load-based abort.  Load abort uses decoded
magnitude plus a low-speed persistence gate; ``abs(raw)`` is not a magnitude.

``u_hat`` is image-only: the intersection of the two fitted knife-edge
lines, or the midpoint of the two distal endpoints, on the **column**
frame.  Wrist is a contact witness only.  This module does not import or
call FK residual ranking, the fifth pulse-track geometry gate, metal
snapshots, M0/M1, or FCPG outputs.

True seek uses ``self_contact_seek_v3`` (motion profile still
``seek_profile_version=2``): ``approach_p1_ramp`` →
``approach_p1_settle`` → ``contact_search`` → ``restore_ramp``.  Each
command is bounded per frame.  Historical ``approach_p1`` one-shot writes
of session p1 are illegal.  v3 additionally persists per-view frames and
forbids reusing a column ROI on the wrist.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import cv2
import numpy as np

from fcpg.servo_register_decode import (
    decode_present_current,
    decode_present_load,
    decode_present_velocity,
    encoding_provenance,
    reinterpret_register_series,
)


SCHEMA_NAME = "self_contact_seek_v3"
SCHEMA_NAME_V2 = "self_contact_seek_v2"
SCHEMA_NAME_V1 = "self_contact_seek_v1"
ACCEPTED_SCHEMA_NAMES = (SCHEMA_NAME_V1, SCHEMA_NAME_V2, SCHEMA_NAME)
SEEK_PROFILE_VERSION = 2
EVIDENCE_SCHEMA_VERSION = 3
SEEK_PHASE_APPROACH_RAMP = "approach_p1_ramp"
SEEK_PHASE_APPROACH_SETTLE = "approach_p1_settle"
SEEK_PHASE_CONTACT_SEARCH = "contact_search"
SEEK_PHASE_RESTORE = "restore_ramp"
SEEK_PHASES = (
    SEEK_PHASE_APPROACH_RAMP,
    SEEK_PHASE_APPROACH_SETTLE,
    SEEK_PHASE_CONTACT_SEARCH,
    SEEK_PHASE_RESTORE,
)
LEGACY_PHASE_APPROACH = "approach_p1"
LEGACY_PHASE_SEEK = "seek"
LEGACY_PHASE_RESTORE = "restore"
PULSE_MAXIMUM_STEP = 1.5
# LeRobot gripper 0–100 over STS3215 0–4095 ≈ 0.0244 units/tick.
HARDWARE_POSITION_RESOLUTION = 100.0 / 4095.0
# Ordinary pulse / restore evidence from live_20260819:
# command step 1.5 @ 20 Hz; restore cruise Present_Velocity 150–200,
# Δpresent ≈ 0.66/frame.  The first true-seek slam wrote 31.87→2.23
# and reached velocity -1000 within 120 ms.
ORDINARY_PULSE_VELOCITY = 200.0
FORMAL_U_HAT_VIEW = "column"
PLANNING_CANVAS_HW = (224, 224)
PLANNING_CANVAS_NAME = "planning_224"
U_HAT_PROVENANCES = (
    "column_edge_intersection",
    "column_distal_midpoint",
)
DEBUG_FULL_IMAGE_ROI_PROVENANCE = "debug_full_image_explicit"
MOTION_GRIPPER_ROI_PROVENANCE = "open_close_motion_bbox"
MIN_GRIPPER_ROI_SIDE_PX = 16
MIN_GRIPPER_ROI_AREA_FRACTION = 0.008
MAX_GRIPPER_ROI_AREA_FRACTION = 0.50
DEFAULT_ROI_BORDER_MARGIN_PX = 8
DEFAULT_MIN_KNIFE_SPAN_PX = 40
LEFT_STRIP_MAX_WIDTH_PX = 56
FORBIDDEN_TRUTH_SOURCES = (
    "snapshot-metal",
    "world-fish-or-box",
    "fk-residual-ranking",
    "pulse-tracks-gate-5",
    "static-m0-m1",
    "fcpg-corrected-output",
    "moving-tip-track",
)

# Conservative defaults.  Starred values need a real-camera pilot before a
# physical Gate 1 claim; they are not a physical pass criterion.
DEFAULT_CONSECUTIVE_FRAMES = 6
DEFAULT_GAP_CLOSE_THRESHOLD_PX = 1.0  # PILOT @ planning 224
DEFAULT_PLATEAU_PRESENT_SPAN = 0.25
DEFAULT_GOAL_INWARD_MIN = 0.5
DEFAULT_VELOCITY_NEAR_ZERO = 1.0  # PILOT; ignored unless velocity is usable
DEFAULT_MAXIMUM_STEP = 0.5  # stricter than the existing pulse step 1.5
DEFAULT_APPROACH_RAMP_STEP = 0.35  # ~14 ticks; 32→p1 in ~86 frames @ 20 Hz
DEFAULT_CONTACT_SEARCH_STEP = 0.15  # smaller inward probe after settle
DEFAULT_RESTORE_STEP = 0.5
DEFAULT_MAXIMUM_ADDITIONAL_TRAVEL = 4.0  # PILOT; contact_search only
DEFAULT_APPROACH_RAMP_MAX_TRAVEL = 40.0
DEFAULT_RESTORE_MAX_TRAVEL = 40.0
DEFAULT_MAXIMUM_COMMAND_PRESENT_ERROR = 8.0  # PILOT; contact lead-in excluded
DEFAULT_MAXIMUM_APPROACH_LEAD = 2.0
DEFAULT_LOAD_ABORT_ABS = 400.0  # PILOT; decoded magnitude, armed only at low speed
DEFAULT_CURRENT_ABORT_ABS = 400.0  # PILOT; decoded magnitude, immediate
DEFAULT_LOAD_ABORT_CONSECUTIVE_FRAMES = 3
DEFAULT_LOAD_ABORT_ARM_SPEED = DEFAULT_VELOCITY_NEAR_ZERO
DEFAULT_VELOCITY_ABORT_ABS = 400.0  # 2× ordinary restore cruise; 0.4× slam
DEFAULT_VELOCITY_ABORT_CONSECUTIVE_FRAMES = 3
DEFAULT_VELOCITY_GLITCH_BUDGET = 2
DEFAULT_ACCEL_PROXY_ABORT_ABS = 250.0  # |Δvel|/frame; cruise ~50, slam ~900
DEFAULT_ACCEL_ABORT_CONSECUTIVE_FRAMES = 3
DEFAULT_APPROACH_TOLERANCE = 0.5
DEFAULT_APPROACH_SETTLE_STABLE_REQUIRED = 4
DEFAULT_CONTACT_SEARCH_SETTLE_FRAMES = 3
DEFAULT_MAXIMUM_PHASE_FRAMES = 160


PILOT_THRESHOLDS = {
    "gap_close_threshold_px": DEFAULT_GAP_CLOSE_THRESHOLD_PX,
    "velocity_near_zero": DEFAULT_VELOCITY_NEAR_ZERO,
    "maximum_additional_travel": DEFAULT_MAXIMUM_ADDITIONAL_TRAVEL,
    "maximum_command_present_error": DEFAULT_MAXIMUM_COMMAND_PRESENT_ERROR,
    "load_abort_abs": DEFAULT_LOAD_ABORT_ABS,
    "current_abort_abs": DEFAULT_CURRENT_ABORT_ABS,
    "load_abort_consecutive_frames": DEFAULT_LOAD_ABORT_CONSECUTIVE_FRAMES,
    "load_abort_arm_speed": DEFAULT_LOAD_ABORT_ARM_SPEED,
    "seek_schema": SCHEMA_NAME,
    "seek_profile_version": SEEK_PROFILE_VERSION,
    "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
    "approach_ramp_step": DEFAULT_APPROACH_RAMP_STEP,
    "contact_search_step": DEFAULT_CONTACT_SEARCH_STEP,
    "restore_step": DEFAULT_RESTORE_STEP,
    "velocity_abort_abs": DEFAULT_VELOCITY_ABORT_ABS,
    "velocity_abort_consecutive_frames": DEFAULT_VELOCITY_ABORT_CONSECUTIVE_FRAMES,
    "velocity_glitch_budget": DEFAULT_VELOCITY_GLITCH_BUDGET,
    "accel_proxy_abort_abs": DEFAULT_ACCEL_PROXY_ABORT_ABS,
    "sample_hz": 20.0,
    "note": (
        "Load/current ceilings are decoded STS3215 magnitudes, not raw "
        "direction-bit words (raw 1244 = 1024+220).  Load abort additionally "
        "requires N consecutive frames while |velocity| is near zero.  "
        "Current remains an immediate hard abort.  Velocity is a separate "
        "hard abort: decoded |Present_Velocity| above a conservative cruise "
        "ceiling for N consecutive frames, or more isolated glitches than "
        "the budget.  True-seek command steps are bounded per frame and "
        "must not one-shot session p1.  Historical session "
        "live_20260819/b_pilot_1119 used self_contact_seek_v1 approach_p1 "
        "and is not a legal motion profile.  A future session must emit "
        "self_contact_seek_v3 with per-view ROIs and complete frame "
        "evidence.  A real-camera pilot must confirm knife-edge contrast, "
        "contact-pixel floor, and safe extra travel before any physical "
        "Gate 1 claim."
    ),
}


class SelfContactFatalError(RuntimeError):
    """Hard safety abort after a best-effort restore.  Caller must not continue."""

    def __init__(self, reason: str, payload: Mapping[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = str(reason)
        self.payload = {} if payload is None else dict(payload)


@dataclass(frozen=True)
class SelfContactConfig:
    sample_hz: float = 20.0
    maximum_step: float = DEFAULT_MAXIMUM_STEP
    maximum_phase_frames: int = DEFAULT_MAXIMUM_PHASE_FRAMES
    maximum_arm_drift: float = 1.5
    maximum_additional_travel: float = DEFAULT_MAXIMUM_ADDITIONAL_TRAVEL
    maximum_command_present_error: float = DEFAULT_MAXIMUM_COMMAND_PRESENT_ERROR
    consecutive_frames: int = DEFAULT_CONSECUTIVE_FRAMES
    gap_close_threshold_px: float = DEFAULT_GAP_CLOSE_THRESHOLD_PX
    plateau_present_span: float = DEFAULT_PLATEAU_PRESENT_SPAN
    goal_inward_min: float = DEFAULT_GOAL_INWARD_MIN
    velocity_near_zero: float = DEFAULT_VELOCITY_NEAR_ZERO
    load_abort_abs: float = DEFAULT_LOAD_ABORT_ABS
    current_abort_abs: float = DEFAULT_CURRENT_ABORT_ABS
    load_abort_consecutive_frames: int = DEFAULT_LOAD_ABORT_CONSECUTIVE_FRAMES
    load_abort_arm_speed: float = DEFAULT_LOAD_ABORT_ARM_SPEED
    approach_tolerance: float = DEFAULT_APPROACH_TOLERANCE
    restore_step: float = DEFAULT_RESTORE_STEP
    restore_settle_tolerance: float = 2.0
    restore_stable_required: int = 6
    gripper_min: float = 0.0
    gripper_max: float = 100.0
    approach_ramp_step: float = DEFAULT_APPROACH_RAMP_STEP
    contact_search_step: float = DEFAULT_CONTACT_SEARCH_STEP
    approach_ramp_max_travel: float = DEFAULT_APPROACH_RAMP_MAX_TRAVEL
    restore_max_travel: float = DEFAULT_RESTORE_MAX_TRAVEL
    maximum_approach_lead: float = DEFAULT_MAXIMUM_APPROACH_LEAD
    velocity_abort_abs: float = DEFAULT_VELOCITY_ABORT_ABS
    velocity_abort_consecutive_frames: int = DEFAULT_VELOCITY_ABORT_CONSECUTIVE_FRAMES
    velocity_glitch_budget: int = DEFAULT_VELOCITY_GLITCH_BUDGET
    accel_proxy_abort_abs: float = DEFAULT_ACCEL_PROXY_ABORT_ABS
    accel_abort_consecutive_frames: int = DEFAULT_ACCEL_ABORT_CONSECUTIVE_FRAMES
    approach_settle_stable_required: int = DEFAULT_APPROACH_SETTLE_STABLE_REQUIRED
    contact_search_settle_frames: int = DEFAULT_CONTACT_SEARCH_SETTLE_FRAMES
    approach_ramp_max_frames: int | None = None
    approach_settle_max_frames: int | None = None
    contact_search_max_frames: int | None = None
    restore_max_frames: int | None = None
    seek_profile_version: int = SEEK_PROFILE_VERSION
    seek_schema: str = SCHEMA_NAME
    evidence_schema_version: int = EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "approach_ramp_max_frames",
            "approach_settle_max_frames",
            "contact_search_max_frames",
            "restore_max_frames",
        ):
            if getattr(self, name) is None:
                object.__setattr__(self, name, int(self.maximum_phase_frames))
        numeric = (
            self.sample_hz,
            self.maximum_step,
            self.maximum_arm_drift,
            self.maximum_additional_travel,
            self.maximum_command_present_error,
            self.gap_close_threshold_px,
            self.plateau_present_span,
            self.goal_inward_min,
            self.approach_tolerance,
            self.restore_step,
            self.restore_settle_tolerance,
            self.load_abort_arm_speed,
            self.load_abort_abs,
            self.current_abort_abs,
            self.approach_ramp_step,
            self.contact_search_step,
            self.approach_ramp_max_travel,
            self.restore_max_travel,
            self.maximum_approach_lead,
            self.velocity_abort_abs,
            self.accel_proxy_abort_abs,
        )
        if any(not np.isfinite(value) or float(value) <= 0.0 for value in numeric):
            raise ValueError("self-contact numeric limits must be finite and positive")
        if int(self.maximum_phase_frames) < 2:
            raise ValueError("maximum_phase_frames must be at least 2")
        if int(self.consecutive_frames) < 2:
            raise ValueError("consecutive_frames must be at least 2")
        if int(self.load_abort_consecutive_frames) < 1:
            raise ValueError("load_abort_consecutive_frames must be at least 1")
        if int(self.velocity_abort_consecutive_frames) < 1:
            raise ValueError("velocity_abort_consecutive_frames must be at least 1")
        if int(self.velocity_glitch_budget) < 0:
            raise ValueError("velocity_glitch_budget must be ≥ 0")
        if int(self.accel_abort_consecutive_frames) < 1:
            raise ValueError("accel_abort_consecutive_frames must be at least 1")
        if int(self.approach_settle_stable_required) < 1:
            raise ValueError("approach_settle_stable_required must be at least 1")
        if int(self.contact_search_settle_frames) < 1:
            raise ValueError("contact_search_settle_frames must be at least 1")
        if int(self.seek_profile_version) != SEEK_PROFILE_VERSION:
            raise ValueError(
                f"true-seek profile version must be {SEEK_PROFILE_VERSION}"
            )
        if str(self.seek_schema) != SCHEMA_NAME:
            raise ValueError(f"true-seek schema must be {SCHEMA_NAME}")
        for step_name in (
            "maximum_step",
            "approach_ramp_step",
            "contact_search_step",
            "restore_step",
        ):
            if float(getattr(self, step_name)) > PULSE_MAXIMUM_STEP + 1e-9:
                raise ValueError(
                    "self-contact step must not exceed the existing pulse step 1.5"
                )
        if self.approach_ramp_step > float(self.maximum_step) + 1e-9:
            raise ValueError("approach_ramp_step must not exceed maximum_step")
        if self.contact_search_step > float(self.approach_ramp_step) + 1e-9:
            raise ValueError("contact_search_step must be ≤ approach_ramp_step")
        if self.maximum_arm_drift > 1.5 + 1e-9:
            raise ValueError("self-contact must not loosen the existing 1.5 arm-drift gate")
        for name in (
            "approach_ramp_max_frames",
            "approach_settle_max_frames",
            "contact_search_max_frames",
            "restore_max_frames",
        ):
            if int(getattr(self, name)) < 2:
                raise ValueError(f"{name} must be at least 2")

    @property
    def sample_period_s(self) -> float:
        return 1.0 / float(self.sample_hz)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sample_period_s"] = self.sample_period_s
        payload["hardware_position_resolution"] = HARDWARE_POSITION_RESOLUTION
        payload["pulse_maximum_step"] = PULSE_MAXIMUM_STEP
        payload["ordinary_pulse_velocity"] = ORDINARY_PULSE_VELOCITY
        payload["seek_phases"] = list(SEEK_PHASES)
        return payload


@dataclass(frozen=True)
class KnifeEdgeFit:
    jaw: str
    line_point_px: tuple[float, float]
    line_direction: tuple[float, float]
    distal_endpoint_px: tuple[float, float]
    proximal_endpoint_px: tuple[float, float]
    inlier_count: int
    residual_px: float
    ok: bool
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualGapObservation:
    view: str
    role: str
    ok: bool
    closed_or_intersecting: bool
    signed_gap_px: float | None
    intersection_px: tuple[float, float] | None
    distal_midpoint_px: tuple[float, float] | None
    u_hat_px: tuple[float, float] | None
    u_hat_provenance: str | None
    fixed: KnifeEdgeFit | None
    moving: KnifeEdgeFit | None
    failure_reason: str | None
    polarity_inverted: bool = False
    split_method: str = "none"
    canvas_hw: tuple[int, int] | None = None
    canvas: str | None = None
    planning_224: bool = False
    gripper_roi_xyxy: tuple[int, int, int, int] | None = None
    gripper_roi_provenance: str | None = None
    gripper_roi_diagnostics: dict[str, Any] = field(default_factory=dict)
    allow_full_image: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(frozen=True)
class PlateauObservation:
    ok: bool
    plateau: bool
    goal_inward: bool
    command_arrived: bool
    present_span: float | None
    goal_minus_present: float | None
    velocity_near_zero: bool | None
    velocity_usable: bool
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JointContactDecision:
    visual_a: bool
    plateau_b: bool
    contact: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelfContactHardwareHooks:
    observe: Callable[[], Any]
    write_gripper: Callable[[float], float]
    read_telemetry: Callable[[], Mapping[str, Any]]
    sleep: Callable[[float], None] = field(default=lambda _delay: None)


def infer_close_direction(open_present: float, closed_present: float) -> float:
    """Return +1 or -1: the sign added to present to close further.

    The sign is inferred from an observed open→close pair.  It is never
    hard-coded from the convention that closed is numerically smaller.
    """

    values = np.asarray([open_present, closed_present], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("open/closed presents must be finite")
    delta = float(values[1] - values[0])
    if abs(delta) < 1e-6:
        raise ValueError(
            "open and closed presents are identical; cannot infer close direction"
        )
    return 1.0 if delta > 0.0 else -1.0


def infer_close_direction_from_pulse(
    presents: np.ndarray,
    phase_ranges: Sequence[Mapping[str, Any]],
) -> float:
    """Infer close direction from the named close phase of a pulse."""

    opening = np.asarray(presents, dtype=np.float64).reshape(-1)
    close_phase = _named_phase(phase_ranges, "close")
    start = int(close_phase["start"])
    stop = int(close_phase["stop"])
    if not 0 <= start < stop <= len(opening):
        raise ValueError("close phase range is outside the present series")
    return infer_close_direction(float(opening[start]), float(opening[stop - 1]))


def _named_phase(
    phase_ranges: Sequence[Mapping[str, Any]], name: str
) -> Mapping[str, Any]:
    for row in phase_ranges:
        if str(row.get("name")) == name:
            return row
    raise ValueError(f"pulse has no {name!r} phase")


def _as_rgb(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("frame must be uint8 RGB [H,W,3]")
    if image.shape[0] < 16 or image.shape[1] < 16:
        raise ValueError("frame is too small for knife-edge observation")
    return image


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(_as_rgb(frame), cv2.COLOR_RGB2GRAY)


def _component_candidates(
    binary: np.ndarray,
) -> list[tuple[int, int, np.ndarray]]:
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    height, width = binary.shape
    area_min = 40
    area_max = int(0.40 * height * width)
    rows: list[tuple[int, int, np.ndarray]] = []
    for index in range(1, num):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area_min <= area <= area_max:
            rows.append((area, index, np.asarray(centroids[index], dtype=np.float64)))
    rows.sort(key=lambda item: item[0], reverse=True)
    return rows


def _watershed_split(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    mask = binary.astype(bool)
    if int(mask.sum()) < 80:
        return None
    rows, cols = np.nonzero(mask)
    left_x = int(cols.min())
    right_x = int(cols.max())
    if right_x - left_x < 8:
        return None
    width = binary.shape[1]
    xs = np.arange(width, dtype=np.int32)[None, :]
    span = right_x - left_x
    seeds = np.zeros(binary.shape, dtype=np.int32)
    seeds[mask & (xs <= left_x + max(2, int(0.18 * span)))] = 1
    seeds[mask & (xs >= right_x - max(2, int(0.18 * span)))] = 2
    if not np.any(seeds == 1) or not np.any(seeds == 2):
        return None
    markers = seeds.copy()
    color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    watershed = cv2.watershed(color, markers)
    left = watershed == 1
    right = watershed == 2
    if int(left.sum()) < 20 or int(right.sum()) < 20:
        return None
    if float(np.mean(np.nonzero(left)[1])) > float(np.mean(np.nonzero(right)[1])):
        left, right = right, left
    return left, right


def _two_jaw_masks(
    gray: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    if float(np.ptp(gray.astype(np.float32))) < 20.0:
        return None
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    best: tuple[int, np.ndarray, np.ndarray, dict[str, Any]] | None = None
    for invert in (False, True):
        work = cv2.bitwise_not(blurred) if invert else blurred
        _threshold, binary = cv2.threshold(
            work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        occupied = float(np.mean(binary)) / 255.0
        if occupied < 0.02 or occupied > 0.80:
            continue
        candidates = _component_candidates(binary)
        split_method = "two_components"
        if len(candidates) >= 2:
            pair = sorted(candidates[:2], key=lambda item: float(item[2][0]))
            labels = (
                cv2.connectedComponentsWithStats(binary, 8)[1]
            )
            left = labels == pair[0][1]
            right = labels == pair[1][1]
            score = pair[0][0] + pair[1][0]
        else:
            split = _watershed_split(binary)
            if split is None:
                continue
            left, right = split
            split_method = "watershed"
            score = int(left.sum() + right.sum())
        meta = {"polarity_inverted": invert, "split_method": split_method}
        if split_method == "two_components":
            if (
                best is None
                or best[3]["split_method"] != "two_components"
                or score > best[0]
            ):
                best = (score, left, right, meta)
        elif best is None or best[3]["split_method"] != "two_components":
            if best is None or score > best[0]:
                best = (score, left, right, meta)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _facing_edge_points(mask: np.ndarray, *, side: str) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    points: list[tuple[float, float]] = []
    for row in np.unique(rows):
        xs = cols[rows == row]
        column = int(xs.max() if side == "left" else xs.min())
        points.append((float(column), float(row)))
    return np.asarray(points, dtype=np.float64)


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if len(points) < 5:
        raise ValueError("not enough edge points for a line fit")
    centroid = points.mean(axis=0)
    centered = points - centroid[None]
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("degenerate knife-edge direction")
    direction = direction / norm
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    residual = float(np.sqrt(np.mean((centered @ normal) ** 2)))
    return centroid, direction, residual


def _line_intersection(
    point_a: np.ndarray,
    direction_a: np.ndarray,
    point_b: np.ndarray,
    direction_b: np.ndarray,
) -> np.ndarray | None:
    matrix = np.column_stack((direction_a, -direction_b))
    if abs(float(np.linalg.det(matrix))) < 1e-8:
        return None
    try:
        parameters = np.linalg.solve(matrix, point_b - point_a)
    except np.linalg.LinAlgError:
        return None
    return point_a + float(parameters[0]) * direction_a


def _signed_gap_px(left_points: np.ndarray, right_points: np.ndarray) -> float:
    left_by_y = {int(round(y)): x for x, y in left_points}
    right_by_y = {int(round(y)): x for x, y in right_points}
    shared = sorted(set(left_by_y) & set(right_by_y))
    if shared:
        band = shared[int(0.65 * len(shared)) :] or shared
        return float(np.median([right_by_y[y] - left_by_y[y] for y in band]))
    return float(np.median(right_points[:, 0]) - np.median(left_points[:, 0]))


def _assign_fixed_moving(
    left: KnifeEdgeFit,
    right: KnifeEdgeFit,
    *,
    previous_centroids: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[KnifeEdgeFit, KnifeEdgeFit]:
    if previous_centroids is None:
        return (
            KnifeEdgeFit(**{**asdict(left), "jaw": "jaw_left"}),
            KnifeEdgeFit(**{**asdict(right), "jaw": "jaw_right"}),
        )
    previous_left, previous_right = previous_centroids
    left_shift = abs(float(left.line_point_px[0]) - float(previous_left[0]))
    right_shift = abs(float(right.line_point_px[0]) - float(previous_right[0]))
    if right_shift > left_shift + 0.25:
        moving, fixed = right, left
    else:
        moving, fixed = left, right
    return (
        KnifeEdgeFit(**{**asdict(fixed), "jaw": "fixed"}),
        KnifeEdgeFit(**{**asdict(moving), "jaw": "moving"}),
    )


def parse_gripper_roi_xyxy(raw: Any) -> tuple[int, int, int, int] | None:
    """Parse an explicit image-plane ROI.  None stays None."""

    if raw is None or raw == "":
        return None
    if isinstance(raw, Mapping):
        raw = raw.get("xyxy", raw.get("bbox_xyxy"))
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("gripper_roi_xyxy must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (int(round(float(v))) for v in raw)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("gripper_roi_xyxy must have positive area")
    return (x0, y0, x1, y1)


def assess_gripper_roi(
    raw: Any,
    frame_hw: tuple[int, int],
    *,
    allow_full_image: bool = False,
    provenance: str | None = None,
    min_side_px: int = MIN_GRIPPER_ROI_SIDE_PX,
    min_area_fraction: float = MIN_GRIPPER_ROI_AREA_FRACTION,
    max_area_fraction: float = MAX_GRIPPER_ROI_AREA_FRACTION,
    min_border_margin_px: int | None = None,
    min_knife_span_px: int | None = None,
    require_complete_in_frame: bool = False,
    view: str | None = None,
) -> dict[str, Any]:
    """Validate a motion-derived gripper ROI against one frame.

    Formal callers must pass ``allow_full_image=False``.  A missing, tiny,
    or near-full-image box is fail-closed.  An edge-clipped or left-strip
    box is refused when ``require_complete_in_frame`` (pilot / freeze).
    Debug may opt into the full canvas with an explicit provenance that
    formal paths must not emit.
    """

    height, width = int(frame_hw[0]), int(frame_hw[1])
    image_area = max(1, height * width)
    if allow_full_image and raw in {None, ""}:
        box = (0, 0, width, height)
        return {
            "valid": True,
            "xyxy": box,
            "provenance": DEBUG_FULL_IMAGE_ROI_PROVENANCE,
            "reason": None,
            "diagnostics": {
                "roi_xyxy": list(box),
                "roi_width": width,
                "roi_height": height,
                "roi_area_px": image_area,
                "image_area_px": image_area,
                "roi_area_fraction": 1.0,
                "allow_full_image": True,
                "debug_full_image": True,
            },
        }
    try:
        parsed = parse_gripper_roi_xyxy(raw)
    except ValueError as exc:
        return {
            "valid": False,
            "xyxy": None,
            "provenance": provenance,
            "reason": f"gripper_roi_invalid:{exc}",
            "diagnostics": {"allow_full_image": bool(allow_full_image)},
        }
    if parsed is None:
        return {
            "valid": False,
            "xyxy": None,
            "provenance": provenance,
            "reason": "gripper_roi_required",
            "diagnostics": {"allow_full_image": False},
        }
    x0, y0, x1, y1 = parsed
    x0 = max(0, min(width, x0))
    y0 = max(0, min(height, y0))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    roi_w = x1 - x0
    roi_h = y1 - y0
    roi_area = max(0, roi_w * roi_h)
    fraction = float(roi_area) / float(image_area)
    reasons: list[str] = []
    if roi_w < int(min_side_px) or roi_h < int(min_side_px):
        reasons.append("gripper_roi_coverage_insufficient:side")
    if fraction + 1e-12 < float(min_area_fraction):
        reasons.append("gripper_roi_coverage_insufficient:area")
    if fraction - 1e-12 > float(max_area_fraction):
        reasons.append("gripper_roi_coverage_insufficient:too_large")
    if str(provenance or "") == DEBUG_FULL_IMAGE_ROI_PROVENANCE and not allow_full_image:
        reasons.append("debug_full_image_not_allowed_on_formal_path")
    margin = 0 if min_border_margin_px is None else int(min_border_margin_px)
    if require_complete_in_frame and min_border_margin_px is None:
        margin = int(DEFAULT_ROI_BORDER_MARGIN_PX)
    knife_span = (
        int(DEFAULT_MIN_KNIFE_SPAN_PX)
        if require_complete_in_frame and min_knife_span_px is None
        else (None if min_knife_span_px is None else int(min_knife_span_px))
    )
    touches_border = x0 <= 0 or y0 <= 0 or x1 >= width or y1 >= height
    margin_ok = (
        x0 >= margin
        and y0 >= margin
        and x1 <= width - margin
        and y1 <= height - margin
    )
    if require_complete_in_frame or margin > 0:
        if touches_border:
            reasons.append("gripper_roi_clipped_at_border")
        elif not margin_ok:
            reasons.append("gripper_roi_border_margin_insufficient")
    if knife_span is not None and min(roi_w, roi_h) < int(knife_span):
        reasons.append("gripper_roi_knife_resolution_insufficient")
    if require_complete_in_frame and x0 <= 0 and roi_w <= int(LEFT_STRIP_MAX_WIDTH_PX):
        reasons.append("gripper_roi_left_strip_rejected")
    box = (int(x0), int(y0), int(x1), int(y1))
    diagnostics = {
        "roi_xyxy": list(box),
        "roi_width": int(roi_w),
        "roi_height": int(roi_h),
        "roi_area_px": int(roi_area),
        "image_area_px": int(image_area),
        "roi_area_fraction": fraction,
        "allow_full_image": bool(allow_full_image),
        "debug_full_image": False,
        "min_side_px": int(min_side_px),
        "min_area_fraction": float(min_area_fraction),
        "max_area_fraction": float(max_area_fraction),
        "min_border_margin_px": int(margin),
        "min_knife_span_px": knife_span,
        "touches_border": bool(touches_border),
        "complete_in_frame": bool(margin_ok and not touches_border),
        "view": view,
    }
    return {
        "valid": not reasons,
        "xyxy": box if not reasons else None,
        "provenance": provenance or MOTION_GRIPPER_ROI_PROVENANCE,
        "reason": None if not reasons else ";".join(reasons),
        "diagnostics": diagnostics,
        "view": view,
    }


def resolve_per_view_rois(
    *,
    gripper_roi_xyxy: Any = None,
    gripper_roi_by_view: Mapping[str, Any] | None = None,
    gripper_roi_provenance: str | None = None,
    frame_hw_by_view: Mapping[str, tuple[int, int]] | None = None,
    require_complete_in_frame: bool = False,
    allow_full_image: bool = False,
) -> dict[str, dict[str, Any]]:
    """Bind one ROI document per view.  Never copy column coordinates onto wrist."""

    default_hw = PLANNING_CANVAS_HW
    by_view = {} if gripper_roi_by_view is None else dict(gripper_roi_by_view)
    if "column" not in by_view and gripper_roi_xyxy is not None:
        by_view["column"] = gripper_roi_xyxy
    resolved: dict[str, dict[str, Any]] = {}
    for view in ("column", "wrist"):
        raw = by_view.get(view)
        if isinstance(raw, Mapping) and raw.get("xyxy") is None and "bbox_xyxy" not in raw:
            if raw.get("view") and str(raw.get("view")) != view:
                resolved[view] = {
                    "valid": False,
                    "xyxy": None,
                    "provenance": gripper_roi_provenance,
                    "reason": f"roi_view_mismatch:{raw.get('view')}->{view}",
                    "diagnostics": {"view": view},
                    "view": view,
                }
                continue
        if isinstance(raw, Mapping) and raw.get("view") not in {None, "", view}:
            resolved[view] = {
                "valid": False,
                "xyxy": None,
                "provenance": raw.get("provenance") or gripper_roi_provenance,
                "reason": f"roi_view_mismatch:{raw.get('view')}->{view}",
                "diagnostics": {"view": view, "source_view": raw.get("view")},
                "view": view,
            }
            continue
        hw = default_hw
        if frame_hw_by_view and view in frame_hw_by_view:
            hw = tuple(int(v) for v in frame_hw_by_view[view])  # type: ignore[assignment]
        xyxy = raw
        provenance = gripper_roi_provenance
        if isinstance(raw, Mapping):
            xyxy = raw.get("xyxy", raw.get("bbox_xyxy"))
            provenance = raw.get("provenance") or provenance
        assessed = assess_gripper_roi(
            xyxy,
            hw,
            allow_full_image=bool(allow_full_image),
            provenance=str(provenance or "") or None,
            require_complete_in_frame=bool(require_complete_in_frame),
            view=view,
        )
        assessed["view"] = view
        resolved[view] = assessed
    return resolved


def _visual_quality(observation: VisualGapObservation) -> dict[str, Any]:
    diagnostics = dict(observation.gripper_roi_diagnostics or {})
    residuals = [
        value
        for value in (
            diagnostics.get("left_residual_px"),
            diagnostics.get("right_residual_px"),
        )
        if value is not None
    ]
    inliers = int(diagnostics.get("left_inlier_count") or 0) + int(
        diagnostics.get("right_inlier_count") or 0
    )
    return {
        "ok": bool(observation.ok),
        "edge_in_roi": bool(diagnostics.get("edge_in_roi")),
        "inlier_count": inliers,
        "max_residual_px": None if not residuals else float(max(residuals)),
        "roi_area_fraction": diagnostics.get("roi_area_fraction"),
        "complete_in_frame": diagnostics.get("complete_in_frame"),
    }


def evaluate_visual_contact(
    column: VisualGapObservation,
    wrist: VisualGapObservation,
    *,
    gap_close_threshold_px: float = DEFAULT_GAP_CLOSE_THRESHOLD_PX,
    wrist_gap_contradict_px: float | None = None,
) -> dict[str, Any]:
    """Per-view A plus the joint rule used for formal B.

    Formal truth needs column A and a column ``u_hat``.  Wrist may
    strengthen or contradict.  A clear wrist gap forbids contact.
    A column/wrist conflict is INCONCLUSIVE.
    """

    contradict_px = (
        float(gap_close_threshold_px)
        if wrist_gap_contradict_px is None
        else float(wrist_gap_contradict_px)
    )
    column_closed = bool(column.ok and column.closed_or_intersecting)
    wrist_closed = bool(wrist.ok and wrist.closed_or_intersecting)
    wrist_gap = bool(
        wrist.ok
        and wrist.signed_gap_px is not None
        and float(wrist.signed_gap_px) > contradict_px
    )
    column_has_u_hat = bool(
        column.ok
        and column.u_hat_px is not None
        and column.u_hat_provenance in U_HAT_PROVENANCES
    )
    column_a = bool(column_closed and column_has_u_hat)
    conflict = bool(column_closed and wrist_gap)
    wrist_contradicts = bool(wrist_gap)
    formal_a = bool(column_a and not wrist_contradicts and not conflict)
    if conflict:
        reason = "view_conflict"
        adjudication = "INCONCLUSIVE"
    elif wrist_contradicts:
        reason = "wrist_gap_contradicts"
        adjudication = "INCONCLUSIVE"
    elif column_a:
        reason = "column_a_ready"
        adjudication = None
    elif column.ok and column.closed_or_intersecting and not column_has_u_hat:
        reason = "column_u_hat_unavailable"
        adjudication = "INCONCLUSIVE"
    elif column.ok:
        reason = "column_gap_open"
        adjudication = None
    else:
        reason = str(column.failure_reason or "column_visual_failed")
        adjudication = None
    return {
        "column": {
            "A": column_a,
            "closed_or_intersecting": column_closed,
            "gap_px": column.signed_gap_px,
            "u_hat_px": None
            if column.u_hat_px is None
            else [float(column.u_hat_px[0]), float(column.u_hat_px[1])],
            "u_hat_provenance": column.u_hat_provenance,
            "quality": _visual_quality(column),
            "failure": column.failure_reason,
            "roi_xyxy": None
            if column.gripper_roi_xyxy is None
            else list(column.gripper_roi_xyxy),
            "view": "column",
        },
        "wrist": {
            "A": False,
            "closed_or_intersecting": wrist_closed,
            "gap_px": wrist.signed_gap_px,
            "u_hat_px": None,
            "u_hat_provenance": None,
            "quality": _visual_quality(wrist),
            "failure": wrist.failure_reason,
            "roi_xyxy": None
            if wrist.gripper_roi_xyxy is None
            else list(wrist.gripper_roi_xyxy),
            "view": "wrist",
            "strengthens": bool(wrist_closed and column_a),
            "contradicts": wrist_contradicts,
        },
        "formal_a": formal_a,
        "column_a": column_a,
        "wrist_contradicts": wrist_contradicts,
        "view_conflict": conflict,
        "adjudication": adjudication,
        "reason": reason,
        "joint_rule": (
            "formal B needs column A and column u_hat; "
            "wrist may strengthen or contradict; "
            "a clear wrist gap forbids contact; "
            "view conflict is INCONCLUSIVE"
        ),
    }


def _point_in_roi(point: Sequence[float] | None, xyxy: tuple[int, int, int, int]) -> bool:
    if point is None:
        return False
    x, y = float(point[0]), float(point[1])
    if not (np.isfinite(x) and np.isfinite(y)):
        return False
    x0, y0, x1, y1 = xyxy
    return x0 <= x < x1 and y0 <= y < y1


def _crop_rgb(frame: np.ndarray, xyxy: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    x0, y0, x1, y1 = xyxy
    return frame[y0:y1, x0:x1], (x0, y0)


def _offset_xy(point: np.ndarray | Sequence[float], origin: tuple[int, int]) -> np.ndarray:
    array = np.asarray(point, dtype=np.float64)
    return array + np.asarray(origin, dtype=np.float64)


def _empty_visual(
    view: str,
    reason: str,
    *,
    canvas_hw: tuple[int, int] | None = None,
    gripper_roi_xyxy: tuple[int, int, int, int] | None = None,
    gripper_roi_provenance: str | None = None,
    gripper_roi_diagnostics: Mapping[str, Any] | None = None,
    allow_full_image: bool = False,
) -> VisualGapObservation:
    role = "formal_pixel_anchor" if view == FORMAL_U_HAT_VIEW else "contact_witness"
    planning = canvas_hw == PLANNING_CANVAS_HW
    return VisualGapObservation(
        view=view,
        role=role,
        ok=False,
        closed_or_intersecting=False,
        signed_gap_px=None,
        intersection_px=None,
        distal_midpoint_px=None,
        u_hat_px=None,
        u_hat_provenance=None,
        fixed=None,
        moving=None,
        failure_reason=reason,
        canvas_hw=canvas_hw,
        canvas=PLANNING_CANVAS_NAME if planning else None,
        planning_224=planning,
        gripper_roi_xyxy=gripper_roi_xyxy,
        gripper_roi_provenance=gripper_roi_provenance,
        gripper_roi_diagnostics=dict(gripper_roi_diagnostics or {}),
        allow_full_image=bool(allow_full_image),
    )


def observe_knife_edges(
    frame: np.ndarray,
    *,
    view: str,
    previous_frame: np.ndarray | None = None,
    gap_close_threshold_px: float = DEFAULT_GAP_CLOSE_THRESHOLD_PX,
    gripper_roi_xyxy: Any = None,
    gripper_roi_provenance: str | None = None,
    allow_full_image: bool = False,
) -> VisualGapObservation:
    """Fit the two facing knife edges in one RGB frame.

    Formal detection is restricted to an explicit gripper ROI derived from
    open-close image motion.  There is no full-image fallback, and no FK /
    moving-tip / metal-snapshot fallback.  ``allow_full_image=True`` is a
    debug-only escape for the pure function.

    ``u_hat`` is emitted only for the column view.  Wrist is witness-only.
    """

    if view not in {"column", "wrist"}:
        return _empty_visual(str(view), "unsupported_view")
    try:
        rgb = _as_rgb(frame)
    except ValueError as exc:
        return _empty_visual(view, str(exc))
    canvas_hw = (int(rgb.shape[0]), int(rgb.shape[1]))
    assessed = assess_gripper_roi(
        gripper_roi_xyxy,
        canvas_hw,
        allow_full_image=bool(allow_full_image),
        provenance=gripper_roi_provenance,
    )
    roi_kwargs = {
        "canvas_hw": canvas_hw,
        "gripper_roi_xyxy": assessed.get("xyxy"),
        "gripper_roi_provenance": assessed.get("provenance"),
        "gripper_roi_diagnostics": assessed.get("diagnostics"),
        "allow_full_image": bool(allow_full_image),
    }
    if not bool(assessed["valid"]):
        return _empty_visual(view, str(assessed["reason"]), **roi_kwargs)
    if view == FORMAL_U_HAT_VIEW and canvas_hw != PLANNING_CANVAS_HW:
        return _empty_visual(view, "frame_not_planning_224", **roi_kwargs)
    roi_box = tuple(int(v) for v in assessed["xyxy"])
    try:
        crop, origin = _crop_rgb(rgb, roi_box)
        gray = _gray(crop)
    except ValueError as exc:
        return _empty_visual(view, str(exc), **roi_kwargs)
    if gray.shape[0] < 16 or gray.shape[1] < 16:
        return _empty_visual(view, "gripper_roi_coverage_insufficient:crop", **roi_kwargs)
    split = _two_jaw_masks(gray)
    if split is None:
        return _empty_visual(view, "two_jaws_not_found", **roi_kwargs)
    left_mask, right_mask, meta = split
    previous_centroids = None
    if previous_frame is not None:
        try:
            previous_rgb = _as_rgb(previous_frame)
            previous_crop, _ignored = _crop_rgb(previous_rgb, roi_box)
            previous_split = _two_jaw_masks(_gray(previous_crop))
        except ValueError:
            previous_split = None
        if previous_split is not None:
            previous_centroids = (
                _offset_xy(_mask_centroid(previous_split[0]), origin),
                _offset_xy(_mask_centroid(previous_split[1]), origin),
            )
    try:
        left_points = _offset_xy(_facing_edge_points(left_mask, side="left"), origin)
        right_points = _offset_xy(_facing_edge_points(right_mask, side="right"), origin)
        left_point, left_dir, left_residual = _fit_line(left_points)
        right_point, right_dir, right_residual = _fit_line(right_points)
    except ValueError as exc:
        return _empty_visual(view, str(exc), **roi_kwargs)
    left_fit = _edge_fit("jaw_left", left_points, left_point, left_dir, left_residual)
    right_fit = _edge_fit("jaw_right", right_points, right_point, right_dir, right_residual)
    fixed, moving = _assign_fixed_moving(
        left_fit, right_fit, previous_centroids=previous_centroids
    )
    signed_gap = _signed_gap_px(left_points, right_points)
    intersection = _line_intersection(left_point, left_dir, right_point, right_dir)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if intersection is not None:
        if not (0.0 <= float(intersection[0]) < width and 0.0 <= float(intersection[1]) < height):
            intersection = None
    distal_mid = 0.5 * (
        np.asarray(left_fit.distal_endpoint_px, dtype=np.float64)
        + np.asarray(right_fit.distal_endpoint_px, dtype=np.float64)
    )
    intersection_tuple = (
        None
        if intersection is None
        else (float(intersection[0]), float(intersection[1]))
    )
    distal_tuple = (float(distal_mid[0]), float(distal_mid[1]))
    edge_in_roi = all(
        _point_in_roi(point, roi_box)
        for point in (
            left_fit.distal_endpoint_px,
            right_fit.distal_endpoint_px,
            left_fit.proximal_endpoint_px,
            right_fit.proximal_endpoint_px,
        )
    )
    intersection_in_roi = _point_in_roi(intersection_tuple, roi_box)
    distal_in_roi = _point_in_roi(distal_tuple, roi_box)
    diagnostics = dict(assessed["diagnostics"])
    diagnostics.update(
        {
            "edge_in_roi": bool(edge_in_roi),
            "intersection_in_roi": bool(intersection_in_roi),
            "distal_midpoint_in_roi": bool(distal_in_roi),
            "left_inlier_count": int(left_fit.inlier_count),
            "right_inlier_count": int(right_fit.inlier_count),
            "left_residual_px": float(left_fit.residual_px),
            "right_residual_px": float(right_fit.residual_px),
        }
    )
    roi_kwargs["gripper_roi_diagnostics"] = diagnostics
    if not edge_in_roi:
        return _empty_visual(view, "knife_edge_outside_roi", **roi_kwargs)
    closed = bool(np.isfinite(signed_gap) and signed_gap <= float(gap_close_threshold_px))
    u_hat = None
    provenance = None
    if view == FORMAL_U_HAT_VIEW and left_fit.ok and right_fit.ok:
        if intersection_tuple is not None and intersection_in_roi:
            u_hat = intersection_tuple
            provenance = "column_edge_intersection"
        elif distal_in_roi:
            u_hat = distal_tuple
            provenance = "column_distal_midpoint"
        else:
            return _empty_visual(view, "knife_edge_outside_roi", **roi_kwargs)
        if provenance not in U_HAT_PROVENANCES:
            u_hat = None
            provenance = None
    role = "formal_pixel_anchor" if view == FORMAL_U_HAT_VIEW else "contact_witness"
    return VisualGapObservation(
        view=view,
        role=role,
        ok=True,
        closed_or_intersecting=closed,
        signed_gap_px=float(signed_gap),
        intersection_px=intersection_tuple,
        distal_midpoint_px=distal_tuple,
        u_hat_px=u_hat,
        u_hat_provenance=provenance,
        fixed=fixed,
        moving=moving,
        failure_reason=None,
        polarity_inverted=bool(meta["polarity_inverted"]),
        split_method=str(meta["split_method"]),
        canvas_hw=canvas_hw,
        canvas=PLANNING_CANVAS_NAME if canvas_hw == PLANNING_CANVAS_HW else None,
        planning_224=canvas_hw == PLANNING_CANVAS_HW,
        gripper_roi_xyxy=roi_box,
        gripper_roi_provenance=str(assessed["provenance"]),
        gripper_roi_diagnostics=diagnostics,
        allow_full_image=bool(allow_full_image),
    )


def _mask_centroid(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    return np.asarray([float(cols.mean()), float(rows.mean())], dtype=np.float64)


def _edge_fit(
    jaw: str,
    points: np.ndarray,
    centroid: np.ndarray,
    direction: np.ndarray,
    residual: float,
) -> KnifeEdgeFit:
    order = np.argsort(points[:, 1])
    proximal = points[order[0]]
    distal = points[order[-1]]
    return KnifeEdgeFit(
        jaw=jaw,
        line_point_px=(float(centroid[0]), float(centroid[1])),
        line_direction=(float(direction[0]), float(direction[1])),
        distal_endpoint_px=(float(distal[0]), float(distal[1])),
        proximal_endpoint_px=(float(proximal[0]), float(proximal[1])),
        inlier_count=int(len(points)),
        residual_px=float(residual),
        ok=True,
        failure_reason=None,
    )


def evaluate_plateau(
    presents: Sequence[float],
    goals: Sequence[float],
    *,
    close_direction: float,
    consecutive_frames: int = DEFAULT_CONSECUTIVE_FRAMES,
    present_span_max: float = DEFAULT_PLATEAU_PRESENT_SPAN,
    goal_inward_min: float = DEFAULT_GOAL_INWARD_MIN,
    velocities: Sequence[float | None] | None = None,
    velocity_near_zero: float = DEFAULT_VELOCITY_NEAR_ZERO,
    velocity_usable: bool = False,
    phase: str | None = None,
) -> PlateauObservation:
    """B: Goal still inward and Present is a consecutive plateau.

    When ``phase`` is provided, B is legal only in ``contact_search``.
    Approach settle and command-arrived must not impersonate contact.
    """

    present = np.asarray(list(presents), dtype=np.float64)
    goal = np.asarray(list(goals), dtype=np.float64)
    needed = int(consecutive_frames)
    if len(present) < needed or len(goal) < needed:
        return PlateauObservation(
            ok=False,
            plateau=False,
            goal_inward=False,
            command_arrived=False,
            present_span=None,
            goal_minus_present=None,
            velocity_near_zero=None,
            velocity_usable=False,
            failure_reason="insufficient_frames",
        )
    present_w = present[-needed:]
    goal_w = goal[-needed:]
    if not np.all(np.isfinite(present_w)) or not np.all(np.isfinite(goal_w)):
        return PlateauObservation(
            ok=False,
            plateau=False,
            goal_inward=False,
            command_arrived=False,
            present_span=None,
            goal_minus_present=None,
            velocity_near_zero=None,
            velocity_usable=False,
            failure_reason="non_finite_series",
        )
    span = float(np.ptp(present_w))
    lead = float((goal_w[-1] - present_w[-1]) * float(close_direction))
    plateau = span <= float(present_span_max)
    goal_inward = lead >= float(goal_inward_min)
    command_arrived = abs(float(goal_w[-1] - present_w[-1])) <= float(present_span_max)
    velocity_flag = None
    if velocity_usable and velocities is not None:
        recent = list(velocities)[-needed:]
        if any(value is None or not np.isfinite(float(value)) for value in recent):
            velocity_flag = False
        else:
            velocity_flag = all(
                abs(float(value)) <= float(velocity_near_zero) for value in recent
            )
    ok = bool(plateau and goal_inward and not command_arrived)
    if velocity_usable:
        ok = bool(ok and velocity_flag)
    if phase is not None and str(phase) != SEEK_PHASE_CONTACT_SEARCH:
        ok = False
    reason = None
    if not ok:
        if phase is not None and str(phase) != SEEK_PHASE_CONTACT_SEARCH:
            reason = "phase_not_contact_search"
        elif command_arrived and not goal_inward:
            reason = "command_arrived"
        elif not plateau:
            reason = "present_still_moving"
        elif not goal_inward:
            reason = "goal_not_inward"
        elif velocity_usable and not velocity_flag:
            reason = "velocity_not_near_zero"
        else:
            reason = "plateau_predicate_false"
    return PlateauObservation(
        ok=ok,
        plateau=plateau,
        goal_inward=goal_inward,
        command_arrived=bool(command_arrived and not goal_inward),
        present_span=span,
        goal_minus_present=float(goal_w[-1] - present_w[-1]),
        velocity_near_zero=velocity_flag,
        velocity_usable=bool(velocity_usable),
        failure_reason=reason,
    )


def evaluate_joint_contact(
    visual_closed_run: int,
    plateau: PlateauObservation,
    *,
    consecutive_frames: int = DEFAULT_CONSECUTIVE_FRAMES,
    visual_contact: Mapping[str, Any] | None = None,
) -> JointContactDecision:
    """A ∧ B.  Either side alone is false.

    Formal A is column-only.  A clear wrist gap or a view conflict
    fail-closes contact even when the column run and plateau are true.
    """

    visual_a = int(visual_closed_run) >= int(consecutive_frames)
    plateau_b = bool(plateau.ok)
    if isinstance(visual_contact, Mapping):
        if visual_contact.get("view_conflict"):
            return JointContactDecision(
                visual_a, plateau_b, False, "view_conflict"
            )
        if visual_contact.get("wrist_contradicts"):
            return JointContactDecision(
                visual_a, plateau_b, False, "wrist_gap_contradicts"
            )
        if visual_contact.get("formal_a") is False:
            visual_a = False
    if visual_a and plateau_b:
        return JointContactDecision(True, True, True, "joint_visual_and_plateau")
    if visual_a and not plateau_b:
        return JointContactDecision(True, False, False, "visual_only")
    if plateau_b and not visual_a:
        return JointContactDecision(False, True, False, "plateau_only")
    return JointContactDecision(False, False, False, "neither_predicate")


def assess_optional_register_series(
    values: Sequence[float | None],
    *,
    errors: Sequence[str | None] | None = None,
) -> dict[str, Any]:
    """Availability / usability of an optional servo series.

    All-zero or constant successful reads are recorded but not usable as a
    contact trigger or as a load-based abort.  A hole in an otherwise live
    series no longer disables the witness for later frames; the seek window
    fail-closes contact B on a dropped sample.
    """

    parsed: list[float] = []
    missing = 0
    for value in values:
        if value is None or not np.isfinite(float(value)):
            missing += 1
            continue
        parsed.append(float(value))
    error_rows = [row for row in (errors or ()) if row]
    if error_rows and not parsed:
        return {
            "available": False,
            "usable": False,
            "reason": "read_error",
            "read_errors": list(error_rows),
            "n": len(values),
        }
    if missing == len(list(values)):
        return {
            "available": False,
            "usable": False,
            "reason": "unavailable",
            "read_errors": [],
            "n": len(values),
        }
    if parsed and all(value == 0.0 for value in parsed):
        return {
            "available": True,
            "usable": False,
            "reason": "all_zero",
            "read_errors": [],
            "n": len(parsed),
        }
    if len(parsed) >= 2 and all(value == parsed[0] for value in parsed):
        return {
            "available": True,
            "usable": False,
            "reason": "constant_register",
            "read_errors": [],
            "n": len(parsed),
        }
    return {
        "available": True,
        "usable": True,
        "reason": "ok",
        "read_errors": [],
        "n": len(parsed),
    }


@dataclass(frozen=True)
class LoadCurrentSafetyDecision:
    abort_reason: str | None
    load_run: int
    load_raw: float | None
    load_decoded: float | None
    load_magnitude: float | None
    current_raw: float | None
    current_decoded: float | None
    current_magnitude: float | None
    load_armed: bool
    used_as_contact_trigger: Literal[False] = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_decoded(
    raw: float | None,
    decoder: Callable[..., Any],
    **kwargs: Any,
) -> tuple[float | None, float | None, float | None]:
    if raw is None:
        return None, None, None
    try:
        decoded = decoder(raw, **kwargs)
    except (TypeError, ValueError):
        return float(raw), None, None
    return float(decoded.raw), float(decoded.decoded), float(decoded.magnitude)


def evaluate_load_current_safety(
    *,
    load_raw: float | None,
    current_raw: float | None,
    velocity: float | None,
    present: float | None,
    previous_present: float | None,
    load_run: int,
    config: SelfContactConfig,
    load_usable: bool,
    current_usable: bool,
    velocity_usable: bool,
) -> LoadCurrentSafetyDecision:
    """Hard-abort witness.  Never a contact trigger.

    Current: immediate abort on decoded magnitude.
    Load: decoded magnitude, armed only near zero speed / present plateau,
    and only after ``load_abort_consecutive_frames`` consecutive armed samples.
    """

    _raw_load, load_decoded, load_mag = _optional_decoded(load_raw, decode_present_load)
    _raw_cur, current_decoded, current_mag = _optional_decoded(
        current_raw, decode_present_current
    )
    if velocity is None:
        velocity_decoded = None
    else:
        try:
            velocity_decoded = float(
                decode_present_velocity(velocity, source="lerobot_sync_read").decoded
            )
        except (TypeError, ValueError):
            velocity_decoded = float(velocity) if np.isfinite(float(velocity)) else None

    if current_usable and current_mag is not None:
        if current_mag >= float(config.current_abort_abs):
            return LoadCurrentSafetyDecision(
                abort_reason="current_safety_witness",
                load_run=0,
                load_raw=None if load_raw is None else float(load_raw),
                load_decoded=load_decoded,
                load_magnitude=load_mag,
                current_raw=None if current_raw is None else float(current_raw),
                current_decoded=current_decoded,
                current_magnitude=current_mag,
                load_armed=False,
            )

    load_armed = False
    next_run = 0
    abort_reason = None
    if load_raw is None and load_usable:
        # A dropped sample must not reset a live run or arm a new one.
        next_run = int(load_run)
    elif load_usable and load_mag is not None and load_mag >= float(config.load_abort_abs):
        if velocity_usable and velocity_decoded is not None:
            load_armed = abs(velocity_decoded) <= float(config.load_abort_arm_speed)
        elif present is not None and previous_present is not None:
            load_armed = abs(float(present) - float(previous_present)) <= float(
                config.plateau_present_span
            )
        if load_armed:
            next_run = int(load_run) + 1
            if next_run >= int(config.load_abort_consecutive_frames):
                abort_reason = "load_safety_witness"
        else:
            next_run = 0

    return LoadCurrentSafetyDecision(
        abort_reason=abort_reason,
        load_run=next_run,
        load_raw=None if load_raw is None else float(load_raw),
        load_decoded=load_decoded,
        load_magnitude=load_mag,
        current_raw=None if current_raw is None else float(current_raw),
        current_decoded=current_decoded,
        current_magnitude=current_mag,
        load_armed=load_armed,
    )


def next_close_command(
    current: float,
    *,
    close_direction: float,
    step: float,
    start_present: float,
    max_additional_travel: float,
    lo: float = 0.0,
    hi: float = 100.0,
) -> tuple[float, str | None]:
    """One inward step from the observed close direction.  No hardcoded sign."""

    values = np.asarray(
        [current, close_direction, step, start_present, max_additional_travel, lo, hi],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("close command inputs must be finite")
    if step <= 0.0:
        raise ValueError("close step must be positive")
    if close_direction not in (-1.0, 1.0):
        raise ValueError("close_direction must be +1 or -1")
    traveled = (float(current) - float(start_present)) * float(close_direction)
    if traveled > float(max_additional_travel) + 1e-9:
        return float(current), "max_additional_travel"
    proposed = float(current) + float(close_direction) * float(step)
    clipped = float(np.clip(proposed, lo, hi))
    next_travel = (clipped - float(start_present)) * float(close_direction)
    if next_travel > float(max_additional_travel) + 1e-9:
        return float(current), "max_additional_travel"
    if abs(clipped - float(current)) < 1e-9:
        return clipped, "gripper_limit"
    return clipped, None


def bound_command_step(
    last_command: float,
    target: float,
    *,
    step: float,
    lo: float = 0.0,
    hi: float = 100.0,
) -> float:
    """One bounded command-to-command step.  Never writes ``target`` in one shot."""

    values = np.asarray([last_command, target, step, lo, hi], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("bounded command inputs must be finite")
    if step <= 0.0:
        raise ValueError("command step must be positive")
    if step > PULSE_MAXIMUM_STEP + 1e-9:
        raise ValueError("command step must not exceed the existing pulse step 1.5")
    delta = float(np.clip(float(target) - float(last_command), -float(step), float(step)))
    return float(np.clip(float(last_command) + delta, lo, hi))


def legacy_approach_p1_oneshot_command(session_p1: float) -> float:
    """v1 ``approach_p1`` wrote session p1 in one Goal write.  Replay only."""

    value = float(session_p1)
    if not np.isfinite(value):
        raise ValueError("session p1 must be finite")
    return value


def replay_seek_command_counterfactual(
    *,
    present0: float,
    session_p1: float,
    config: SelfContactConfig | None = None,
    historical_first_command: float | None = None,
) -> dict[str, Any]:
    """Offline old-vs-new first-command comparison.  No hardware."""

    cfg = SelfContactConfig() if config is None else config
    old_first = legacy_approach_p1_oneshot_command(session_p1)
    old_jump = abs(old_first - float(present0))
    new_first = bound_command_step(
        float(present0),
        float(session_p1),
        step=cfg.approach_ramp_step,
        lo=cfg.gripper_min,
        hi=cfg.gripper_max,
    )
    new_jump = abs(new_first - float(present0))
    ramp: list[float] = []
    command = float(present0)
    for _ in range(int(cfg.approach_ramp_max_frames or cfg.maximum_phase_frames)):
        command = bound_command_step(
            command,
            float(session_p1),
            step=cfg.approach_ramp_step,
            lo=cfg.gripper_min,
            hi=cfg.gripper_max,
        )
        ramp.append(command)
        if abs(command - float(session_p1)) <= float(cfg.approach_tolerance):
            break
    historical_jump = None
    if historical_first_command is not None:
        historical_jump = abs(float(historical_first_command) - float(present0))
    return {
        "present0": float(present0),
        "session_p1": float(session_p1),
        "old_first_command": old_first,
        "old_command_jump": old_jump,
        "old_triggers_slam": old_jump > PULSE_MAXIMUM_STEP + 1e-9,
        "new_first_command": new_first,
        "new_command_jump": new_jump,
        "new_avoids_slam": new_jump <= float(cfg.approach_ramp_step) + 1e-9,
        "new_ramp_commands": ramp,
        "max_new_command_delta": None
        if len(ramp) < 2
        else float(max(abs(b - a) for a, b in zip(ramp, ramp[1:]))),
        "historical_first_command": historical_first_command,
        "historical_command_jump": historical_jump,
        "historical_is_oneshot_p1": (
            historical_first_command is not None
            and abs(float(historical_first_command) - float(session_p1)) <= 1e-6
        ),
        "seek_schema": SCHEMA_NAME,
        "seek_profile_version": SEEK_PROFILE_VERSION,
    }


@dataclass(frozen=True)
class VelocitySafetyDecision:
    abort_reason: str | None
    consecutive_over: int
    glitch_count: int
    accel_consecutive_over: int
    velocity_decoded: float | None
    velocity_magnitude: float | None
    accel_proxy: float | None
    glitch: bool
    over_limit: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decoded_velocity(velocity: float | None) -> float | None:
    if velocity is None:
        return None
    try:
        return float(
            decode_present_velocity(velocity, source="lerobot_sync_read").decoded
        )
    except (TypeError, ValueError):
        return float(velocity) if np.isfinite(float(velocity)) else None


def evaluate_velocity_hard_safety(
    *,
    velocity: float | None,
    previous_velocity: float | None,
    consecutive_over: int,
    glitch_count: int,
    accel_consecutive_over: int,
    config: SelfContactConfig,
) -> VelocitySafetyDecision:
    """Hard abort on sustained over-speed.  Isolated glitches are budgeted."""

    decoded = _decoded_velocity(velocity)
    previous_decoded = _decoded_velocity(previous_velocity)
    if velocity is None or decoded is None:
        return VelocitySafetyDecision(
            abort_reason=None,
            consecutive_over=int(consecutive_over),
            glitch_count=int(glitch_count),
            accel_consecutive_over=int(accel_consecutive_over),
            velocity_decoded=decoded,
            velocity_magnitude=None,
            accel_proxy=None,
            glitch=False,
            over_limit=False,
        )
    magnitude = abs(float(decoded))
    accel = None
    if previous_decoded is not None:
        accel = abs(float(decoded) - float(previous_decoded))
    over = magnitude > float(config.velocity_abort_abs)
    next_consec = int(consecutive_over) + 1 if over else 0
    next_glitch = int(glitch_count)
    glitch = bool(over and int(consecutive_over) == 0)
    if glitch:
        next_glitch += 1
    accel_over = (
        accel is not None
        and over
        and accel > float(config.accel_proxy_abort_abs)
    )
    next_accel = int(accel_consecutive_over) + 1 if accel_over else 0
    abort_reason = None
    if next_consec >= int(config.velocity_abort_consecutive_frames):
        abort_reason = "velocity_safety_witness"
    elif next_glitch > int(config.velocity_glitch_budget):
        abort_reason = "velocity_safety_witness"
    elif next_accel >= int(config.accel_abort_consecutive_frames):
        abort_reason = "velocity_accel_safety_witness"
    return VelocitySafetyDecision(
        abort_reason=abort_reason,
        consecutive_over=next_consec,
        glitch_count=next_glitch,
        accel_consecutive_over=next_accel,
        velocity_decoded=decoded,
        velocity_magnitude=magnitude,
        accel_proxy=accel,
        glitch=glitch,
        over_limit=over,
    )


def phase_ranges_from_series(phases: Sequence[str]) -> list[dict[str, int | str]]:
    ranges: list[dict[str, int | str]] = []
    if not phases:
        return ranges
    start = 0
    current = str(phases[0])
    for index, name in enumerate(phases[1:], start=1):
        if str(name) != current:
            ranges.append({"name": current, "start": start, "stop": int(index)})
            start = int(index)
            current = str(name)
    ranges.append({"name": current, "start": start, "stop": len(phases)})
    return ranges


def render_synthetic_knife_jaws(
    height: int,
    width: int,
    *,
    gap_px: float,
    background: int = 40,
    jaw: int = 230,
) -> np.ndarray:
    """Deterministic two-wedge image used by offline tests and mocks."""

    frame = np.full((height, width, 3), int(background), dtype=np.uint8)
    mid = width / 2.0
    half = float(gap_px) / 2.0
    left_tip = int(round(mid - half))
    right_tip = int(round(mid + half))
    left = np.asarray(
        [
            [max(2, int(0.06 * width)), 4],
            [max(2, int(0.06 * width)), height - 4],
            [left_tip, int(0.58 * height)],
        ],
        dtype=np.int32,
    )
    right = np.asarray(
        [
            [min(width - 3, int(0.94 * width)), 4],
            [min(width - 3, int(0.94 * width)), height - 4],
            [right_tip, int(0.58 * height)],
        ],
        dtype=np.int32,
    )
    color = (int(jaw), int(jaw), int(jaw))
    cv2.fillConvexPoly(frame, left, color)
    cv2.fillConvexPoly(frame, right, color)
    return frame


def empty_telemetry() -> dict[str, Any]:
    keys = (
        "goal",
        "present_velocity",
        "moving",
        "present_load",
        "present_current",
    )
    return {
        "goal": None,
        "present_velocity": None,
        "moving": None,
        "present_load": None,
        "present_current": None,
        "availability": {key: False for key in keys},
        "read_errors": {key: "unavailable" for key in keys},
    }


def _gripper_from_mapping(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float, np.floating, np.integer)):
        value = float(raw)
        return value if np.isfinite(value) else None
    if isinstance(raw, Mapping):
        for key in ("gripper", "gripper.pos"):
            if key in raw:
                return _gripper_from_mapping(raw[key])
    return None


def _safe_frame(observation: Any, view: str) -> np.ndarray:
    frames = getattr(observation, "frames", None)
    if not isinstance(frames, Mapping) or view not in frames:
        raise RuntimeError(f"image_loss:{view}")
    frame = np.asarray(frames[view])
    if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
        raise RuntimeError(f"image_loss:{view}")
    if not frame.size:
        raise RuntimeError(f"image_loss:{view}")
    return frame


def _append_phase(
    phases: list[dict[str, int | str]], name: str, start: int, stop: int
) -> None:
    phases.append({"name": name, "start": int(start), "stop": int(stop)})


def run_self_contact_seek(
    hooks: SelfContactHardwareHooks,
    config: SelfContactConfig,
    *,
    pinned_arm: np.ndarray,
    start_present: float,
    close_direction: float,
    restore_target: float,
    stop_event: Any,
    gripper_roi_xyxy: Any = None,
    gripper_roi_by_view: Mapping[str, Any] | None = None,
    gripper_roi_provenance: str | None = None,
    allow_full_image: bool = False,
    identity_reference_column: Any = None,
    identity_reference_wrist: Any = None,
    pre_motion_identity_valid: bool | None = None,
    pre_motion_identity: Mapping[str, Any] | None = None,
    evidence_dir: Any = None,
    evidence_recorder: Any = None,
    require_evidence: bool = False,
    require_complete_in_frame: bool = False,
    session_identity: Any = None,
) -> dict[str, Any]:
    """Mockable low-speed seek.  Any hard abort stops motion and restores.

    Formal callers must pass per-view motion-derived gripper ROIs.  A
    single ``gripper_roi_xyxy`` is column-only and is never copied onto
    wrist.  ``allow_full_image`` is refused on this path.
    """

    if close_direction not in (-1.0, 1.0):
        raise ValueError("close_direction must be inferred as +1 or -1")
    if allow_full_image:
        raise ValueError(
            "run_self_contact_seek is the formal path and cannot use "
            "allow_full_image; pass a motion-derived gripper ROI"
        )
    from fcpg.pre_motion_identity import inspect_letterbox_canvas
    from fcpg.self_contact_evidence import EvidenceStoreError, SeekEvidenceRecorder

    recorder = evidence_recorder
    if recorder is None:
        recorder = SeekEvidenceRecorder(
            None if evidence_dir is None else evidence_dir,
            require_disk=bool(require_evidence),
        )
    if require_evidence or evidence_dir is not None or evidence_recorder is not None:
        preflight = recorder.run_preflight()
        if not preflight.get("ok"):
            reason = str(preflight.get("reason") or "disk_preflight_failed")
            payload = {
                "schema": SCHEMA_NAME,
                "seek_profile_version": SEEK_PROFILE_VERSION,
                "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
                "contact": False,
                "abort_reason": reason,
                "images_persisted_complete": False,
                "pre_motion_identity_valid": pre_motion_identity_valid,
                "before_first_command": True,
                "attempt_consumed": False,
                "evidence_preflight": preflight,
            }
            if session_identity is not None and hasattr(session_identity, "mark_broken"):
                session_identity.mark_broken(reason)
            return payload
    pinned = np.asarray(pinned_arm, dtype=np.float64).reshape(-1)
    if pinned.shape != (5,) or not np.all(np.isfinite(pinned)):
        raise ValueError("pinned_arm must be a finite five-vector")
    period_s = float(config.sample_period_s)
    session_p1 = float(start_present)
    timestamps: list[float] = []
    presents: list[float] = []
    goals: list[float] = []
    commands: list[float] = []
    targets: list[float] = []
    phases: list[str] = []
    arm_states: list[list[float]] = []
    velocities: list[float | None] = []
    velocities_decoded: list[float | None] = []
    movings: list[float | None] = []
    loads: list[float | None] = []
    currents: list[float | None] = []
    loads_decoded: list[float | None] = []
    currents_decoded: list[float | None] = []
    load_errors: list[str | None] = []
    current_errors: list[str | None] = []
    velocity_errors: list[str | None] = []
    velocity_glitches: list[bool] = []
    close_directions: list[float] = []
    high_load_run = 0
    velocity_over_run = 0
    velocity_glitch_count = 0
    accel_over_run = 0
    previous_velocity: float | None = None
    visual_column: list[dict[str, Any]] = []
    visual_wrist: list[dict[str, Any]] = []
    plateau_rows: list[dict[str, Any]] = []
    previous_column: np.ndarray | None = None
    last_command: float | None = None
    approach_origin: float | None = None
    restore_command: float | None = None
    abort_reason: str | None = None
    fatal = False
    contact = False
    contact_phase: str | None = None
    true_contact_position: float | None = None
    true_contact_uncertainty: float | None = None
    u_hat_px: tuple[float, float] | None = None
    u_hat_provenance: str | None = None
    u_hat_frame_hw: tuple[int, int] | None = None
    last_roi: dict[str, Any] | None = None
    last_roi_by_view: dict[str, Any] = {}
    visual_contact_rows: list[dict[str, Any]] = []
    last_visual_contact: dict[str, Any] | None = None
    visual_run = 0
    last_decision = evaluate_joint_contact(
        0,
        evaluate_plateau(
            [],
            [],
            close_direction=close_direction,
            consecutive_frames=config.consecutive_frames,
            phase=SEEK_PHASE_CONTACT_SEARCH,
        ),
        consecutive_frames=config.consecutive_frames,
    )
    seek_identity_break = False
    visual_truth = True
    seek_identity_verdict: dict[str, Any] | None = None

    def _append_telemetry(
        *,
        observation: Any,
        state: np.ndarray,
        present: float,
        telemetry: Mapping[str, Any],
        phase: str,
        unbounded_target: float,
        sent: float,
        velocity: float | None,
        velocity_decoded: float | None,
        glitch: bool,
    ) -> None:
        timestamps.append(float(observation.timestamp_s))
        presents.append(present)
        goals.append(sent)
        commands.append(sent)
        targets.append(float(unbounded_target))
        phases.append(str(phase))
        close_directions.append(float(close_direction))
        arm_states.append([float(value) for value in state[:5]])
        velocities.append(velocity)
        velocities_decoded.append(velocity_decoded)
        movings.append(_gripper_from_mapping(telemetry.get("moving")))
        load_raw = _gripper_from_mapping(telemetry.get("present_load"))
        current_raw = _gripper_from_mapping(telemetry.get("present_current"))
        loads.append(load_raw)
        currents.append(current_raw)
        try:
            loads_decoded.append(
                None if load_raw is None else float(decode_present_load(load_raw).decoded)
            )
        except (TypeError, ValueError):
            loads_decoded.append(None)
        try:
            currents_decoded.append(
                None
                if current_raw is None
                else float(decode_present_current(current_raw).decoded)
            )
        except (TypeError, ValueError):
            currents_decoded.append(None)
        load_errors.append((telemetry.get("read_errors") or {}).get("present_load"))
        current_errors.append((telemetry.get("read_errors") or {}).get("present_current"))
        velocity_errors.append(
            (telemetry.get("read_errors") or {}).get("present_velocity")
        )
        velocity_glitches.append(bool(glitch))
        try:
            column_rgb = _safe_frame(observation, "column")
            wrist_rgb = _safe_frame(observation, "wrist")
            native = getattr(observation, "native_frames", None)
            if not isinstance(native, Mapping):
                native = {}
            recorder.append(
                column_planning=column_rgb,
                wrist_planning=wrist_rgb,
                timestamp_s=float(observation.timestamp_s),
                phase=str(phase),
                command=sent,
                present=present,
                frame_id=len(recorder),
                column_native=native.get("column"),
                wrist_native=native.get("wrist"),
                column_canvas=inspect_letterbox_canvas(column_rgb),
                wrist_canvas=inspect_letterbox_canvas(wrist_rgb),
                column_letterbox=inspect_letterbox_canvas(column_rgb),
                wrist_letterbox=inspect_letterbox_canvas(wrist_rgb),
                telemetry={
                    "present_velocity": velocity,
                    "present_load": _gripper_from_mapping(telemetry.get("present_load")),
                    "present_current": _gripper_from_mapping(
                        telemetry.get("present_current")
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001 — evidence fail-closed
            if require_evidence:
                raise EvidenceStoreError(f"evidence_append:{exc}") from exc

    def _restore() -> dict[str, Any]:
        nonlocal restore_command
        stable = 0
        command_stable = 0
        final = None
        restored = False
        restore_error = None
        resynced = False
        last_present = presents[-1] if presents else None
        if restore_command is None and last_command is not None:
            restore_command = float(last_command)
        elif restore_command is None and last_present is not None:
            restore_command = float(last_present)
        try:
            for _ in range(int(config.restore_max_frames or config.maximum_phase_frames)):
                observation = None
                state = None
                present = last_present
                try:
                    observation = hooks.observe()
                    state = np.asarray(observation.state, dtype=np.float64)
                    if state.size >= 6 and np.isfinite(state[5]):
                        present = float(state[5])
                        last_present = present
                except Exception as exc:  # noqa: BLE001 — restore must keep ramping
                    if restore_error is None:
                        restore_error = f"observe:{type(exc).__name__}: {exc}"
                    observation = None
                    state = None

                if restore_command is None:
                    if present is not None:
                        restore_command = float(present)
                    else:
                        restore_error = restore_error or "restore_no_command_clock"
                        break

                if present is not None and not resynced:
                    if abs(float(restore_command) - float(present)) <= float(
                        config.restore_step
                    ) + 1e-9:
                        resynced = True
                        unbounded = float(restore_target)
                    else:
                        unbounded = float(present)
                else:
                    resynced = True
                    unbounded = float(restore_target)

                command = bound_command_step(
                    float(restore_command),
                    float(unbounded),
                    step=float(config.restore_step),
                    lo=config.gripper_min,
                    hi=config.gripper_max,
                )
                try:
                    sent = float(hooks.write_gripper(command))
                except Exception as exc:  # noqa: BLE001
                    restore_error = f"write:{type(exc).__name__}: {exc}"
                    break
                restore_command = sent

                telemetry = empty_telemetry()
                try:
                    telemetry = dict(hooks.read_telemetry() or empty_telemetry())
                except Exception as exc:  # noqa: BLE001
                    if restore_error is None:
                        restore_error = f"telemetry:{type(exc).__name__}: {exc}"
                    telemetry = empty_telemetry()

                if observation is not None and state is not None and present is not None:
                    velocity = _gripper_from_mapping(telemetry.get("present_velocity"))
                    try:
                        _append_telemetry(
                            observation=observation,
                            state=state,
                            present=present,
                            telemetry=telemetry,
                            phase=SEEK_PHASE_RESTORE,
                            unbounded_target=unbounded,
                            sent=sent,
                            velocity=velocity,
                            velocity_decoded=_decoded_velocity(velocity),
                            glitch=False,
                        )
                    except Exception as exc:  # noqa: BLE001
                        if restore_error is None:
                            restore_error = f"record:{type(exc).__name__}: {exc}"
                    if abs(present - float(restore_target)) <= float(
                        config.restore_settle_tolerance
                    ):
                        stable += 1
                    else:
                        stable = 0
                    if stable >= int(config.restore_stable_required):
                        restored = True
                        final = present
                        break
                if abs(float(restore_command) - float(restore_target)) <= float(
                    config.restore_settle_tolerance
                ):
                    command_stable += 1
                else:
                    command_stable = 0
                if (
                    not restored
                    and observation is None
                    and command_stable >= int(config.restore_stable_required)
                ):
                    restored = True
                    final = last_present if last_present is not None else float(
                        restore_command
                    )
                    break
                hooks.sleep(period_s)
            else:
                if not restored:
                    restore_error = restore_error or "restore_timeout"
                    final = last_present
        except Exception as exc:  # noqa: BLE001 — restore is best-effort
            restore_error = f"{type(exc).__name__}: {exc}"
            final = last_present
        return {
            "restored": bool(restored),
            "restore_target": float(restore_target),
            "restore_final": final,
            "restore_error": restore_error,
            "restore_resynced_to_present": bool(resynced),
        }

    def _frame_rows() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, phase_name in enumerate(phases):
            rows.append(
                {
                    "index": index,
                    "phase": phase_name,
                    "target": targets[index] if index < len(targets) else None,
                    "command": commands[index] if index < len(commands) else None,
                    "present": presents[index] if index < len(presents) else None,
                    "velocity": velocities[index] if index < len(velocities) else None,
                    "velocity_decoded": (
                        velocities_decoded[index]
                        if index < len(velocities_decoded)
                        else None
                    ),
                    "present_load": loads[index] if index < len(loads) else None,
                    "present_load_decoded": (
                        loads_decoded[index] if index < len(loads_decoded) else None
                    ),
                    "present_current": currents[index] if index < len(currents) else None,
                    "present_current_decoded": (
                        currents_decoded[index] if index < len(currents_decoded) else None
                    ),
                    "close_direction": float(close_direction),
                    "velocity_glitch": (
                        velocity_glitches[index] if index < len(velocity_glitches) else False
                    ),
                }
            )
        return rows

    def _record_payload(restore: Mapping[str, Any] | None = None) -> dict[str, Any]:
        load_assess = assess_optional_register_series(
            [None if item is None else abs(float(item)) for item in loads_decoded],
            errors=load_errors,
        )
        current_assess = assess_optional_register_series(
            [None if item is None else abs(float(item)) for item in currents_decoded],
            errors=current_errors,
        )
        velocity_assess = assess_optional_register_series(
            velocities, errors=velocity_errors
        )
        restore_row = {} if restore is None else dict(restore)
        command_deltas = [
            abs(float(commands[index]) - float(commands[index - 1]))
            for index in range(1, len(commands))
        ]
        evidence_summary = recorder.persist() if recorder is not None else {}
        images_complete = bool(evidence_summary.get("images_persisted_complete"))
        if require_evidence and not images_complete:
            nonlocal abort_reason, contact
            abort_reason = abort_reason or str(
                evidence_summary.get("persist_error") or "images_not_persisted_complete"
            )
            contact = False
        return {
            "schema": SCHEMA_NAME,
            "seek_profile_version": SEEK_PROFILE_VERSION,
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "contact": bool(contact),
            "contact_phase": contact_phase,
            "true_contact_position": true_contact_position,
            "true_contact_position_uncertainty": true_contact_uncertainty,
            "u_hat_px": None if u_hat_px is None else [float(u_hat_px[0]), float(u_hat_px[1])],
            "u_hat_view": None if u_hat_px is None else FORMAL_U_HAT_VIEW,
            "u_hat_provenance": u_hat_provenance,
            "u_hat_canvas": None if u_hat_px is None else PLANNING_CANVAS_NAME,
            "u_hat_frame_hw": None if u_hat_frame_hw is None else [int(u_hat_frame_hw[0]), int(u_hat_frame_hw[1])],
            "u_hat_planning_224": bool(
                u_hat_px is not None and u_hat_frame_hw == PLANNING_CANVAS_HW
            ),
            "gripper_roi_xyxy": None
            if last_roi is None
            else last_roi.get("xyxy"),
            "gripper_roi_provenance": None
            if last_roi is None
            else last_roi.get("provenance"),
            "gripper_roi_diagnostics": None
            if last_roi is None
            else last_roi.get("diagnostics"),
            "gripper_roi_by_view": last_roi_by_view,
            "visual_contact": last_visual_contact,
            "visual_contact_series": visual_contact_rows,
            "images_persisted_complete": images_complete,
            "evidence": evidence_summary,
            "evidence_paths": evidence_summary.get("paths"),
            "column_planning_224": None
            if not recorder.column_planning
            else np.stack(recorder.column_planning, axis=0),
            "wrist_planning_224": None
            if not recorder.wrist_planning
            else np.stack(recorder.wrist_planning, axis=0),
            "evidence_frame_ids": list(recorder.frame_ids),
            "evidence_phases": list(recorder.phases),
            "close_direction": float(close_direction),
            "close_directions": list(close_directions),
            "start_present": float(start_present),
            "session_p1": session_p1,
            "approach_origin": approach_origin,
            "abort_reason": abort_reason,
            "joint_predicate": last_decision.to_dict(),
            "visual_column": visual_column,
            "visual_wrist": visual_wrist,
            "plateau": plateau_rows,
            "phase_ranges": phase_ranges_from_series(phases),
            "phases": list(phases),
            "timestamps": list(timestamps),
            "presents": list(presents),
            "goals": list(goals),
            "targets": list(targets),
            "commands": list(commands),
            "command_deltas": command_deltas,
            "max_command_delta": None if not command_deltas else float(max(command_deltas)),
            "arm_states": arm_states,
            "present_velocity": list(velocities),
            "present_velocity_decoded": list(velocities_decoded),
            "velocity_glitches": list(velocity_glitches),
            "moving": list(movings),
            "present_load": list(loads),
            "present_current": list(currents),
            "present_load_decoded": list(loads_decoded),
            "present_current_decoded": list(currents_decoded),
            "frames": _frame_rows(),
            "register_encoding": encoding_provenance(),
            "telemetry_availability": {
                "present_load": load_assess,
                "present_current": current_assess,
                "present_velocity": velocity_assess,
            },
            "load_current_usable": bool(
                load_assess["usable"] and current_assess["usable"]
            ),
            "config": config.to_dict(),
            "pilot_thresholds": dict(PILOT_THRESHOLDS),
            "forbidden_truth_sources_used": [],
            "pre_motion_identity_valid": pre_motion_identity_valid,
            "pre_motion_identity": (
                None if pre_motion_identity is None else dict(pre_motion_identity)
            ),
            "pre_motion_handshake": (
                None if pre_motion_identity is None else dict(pre_motion_identity)
            ),
            "seek_identity_break": bool(seek_identity_break),
            "visual_truth": bool(visual_truth) and not bool(seek_identity_break),
            "seek_identity": seek_identity_verdict,
            **restore_row,
        }

    def _hold_command() -> float:
        if last_command is None:
            raise RuntimeError("command clock has not been synchronized to Present")
        return float(last_command)

    try:
        phase = SEEK_PHASE_APPROACH_RAMP
        phase_frames = 0
        settle_stable = 0
        search_settle_left = 0
        total_cap = (
            int(config.approach_ramp_max_frames or config.maximum_phase_frames)
            + int(config.approach_settle_max_frames or config.maximum_phase_frames)
            + int(config.contact_search_max_frames or config.maximum_phase_frames)
        )
        for _ in range(total_cap):
            if getattr(stop_event, "is_set", lambda: False)():
                abort_reason = "operator_stop"
                fatal = True
                break
            observation = hooks.observe()
            state = np.asarray(observation.state, dtype=np.float64)
            if state.shape != (6,) or not np.all(np.isfinite(state)):
                raise RuntimeError("image_loss:state")
            drift = float(np.max(np.abs(state[:5] - pinned)))
            if drift > float(config.maximum_arm_drift):
                abort_reason = (
                    f"arm_drift:{drift:.3f}>{config.maximum_arm_drift:.3f}"
                )
                fatal = True
                break
            column = _safe_frame(observation, "column")
            wrist = _safe_frame(observation, "wrist")
            u_hat_frame_hw = (int(column.shape[0]), int(column.shape[1]))
            if u_hat_frame_hw != PLANNING_CANVAS_HW:
                abort_reason = "image_loss:column_not_planning_224"
                fatal = True
                break
            if identity_reference_column is not None:
                from fcpg.pre_motion_identity import evaluate_seek_frame_identity

                seek_identity_verdict = evaluate_seek_frame_identity(
                    column,
                    wrist,
                    reference_column=np.asarray(identity_reference_column),
                    reference_wrist=(
                        None
                        if identity_reference_wrist is None
                        else np.asarray(identity_reference_wrist)
                    ),
                )
                if not bool(seek_identity_verdict.get("pre_motion_identity_valid")):
                    seek_identity_break = True
                    visual_truth = False
                    contact = False
                    u_hat_px = None
                    u_hat_provenance = None
                    abort_reason = (
                        "camera_identity_break:"
                        + str(seek_identity_verdict.get("reason") or "identity_break")
                    )
                    if last_command is None:
                        last_command = float(state[5])
                        approach_origin = float(state[5])
                    present = float(state[5])
                    telemetry = dict(hooks.read_telemetry() or empty_telemetry())
                    velocity = _gripper_from_mapping(telemetry.get("present_velocity"))
                    _append_telemetry(
                        observation=observation,
                        state=state,
                        present=present,
                        telemetry=telemetry,
                        phase=phase,
                        unbounded_target=_hold_command(),
                        sent=_hold_command(),
                        velocity=velocity,
                        velocity_decoded=_decoded_velocity(velocity),
                        glitch=False,
                    )
                    break
            roi_by_view = resolve_per_view_rois(
                gripper_roi_xyxy=gripper_roi_xyxy,
                gripper_roi_by_view=gripper_roi_by_view,
                gripper_roi_provenance=gripper_roi_provenance,
                frame_hw_by_view={
                    "column": u_hat_frame_hw,
                    "wrist": (int(wrist.shape[0]), int(wrist.shape[1])),
                },
                require_complete_in_frame=bool(require_complete_in_frame),
                allow_full_image=False,
            )
            last_roi_by_view = {
                view: {
                    "xyxy": row.get("xyxy"),
                    "provenance": row.get("provenance"),
                    "diagnostics": row.get("diagnostics"),
                    "valid": row.get("valid"),
                    "reason": row.get("reason"),
                    "view": view,
                }
                for view, row in roi_by_view.items()
            }
            roi_assessed = roi_by_view["column"]
            last_roi = {
                "xyxy": roi_assessed.get("xyxy"),
                "provenance": roi_assessed.get("provenance"),
                "diagnostics": roi_assessed.get("diagnostics"),
            }
            if not bool(roi_assessed["valid"]):
                abort_reason = str(roi_assessed["reason"] or "gripper_roi_required")
                break
            present = float(state[5])
            telemetry = dict(hooks.read_telemetry() or empty_telemetry())
            velocity = _gripper_from_mapping(telemetry.get("present_velocity"))
            load = _gripper_from_mapping(telemetry.get("present_load"))
            current = _gripper_from_mapping(telemetry.get("present_current"))
            if last_command is None:
                last_command = present
                approach_origin = present

            try:
                load_decoded_now = (
                    None if load is None else abs(float(decode_present_load(load).decoded))
                )
            except (TypeError, ValueError):
                load_decoded_now = None
            try:
                current_decoded_now = (
                    None if current is None else abs(float(decode_present_current(current).decoded))
                )
            except (TypeError, ValueError):
                current_decoded_now = None
            load_assess = assess_optional_register_series(
                [None if item is None else abs(float(item)) for item in (*loads_decoded, load_decoded_now)],
                errors=[*load_errors, (telemetry.get("read_errors") or {}).get("present_load")],
            )
            current_assess = assess_optional_register_series(
                [None if item is None else abs(float(item)) for item in (*currents_decoded, current_decoded_now)],
                errors=[*current_errors, (telemetry.get("read_errors") or {}).get("present_current")],
            )
            velocity_assess = assess_optional_register_series(
                [*velocities, velocity],
                errors=[*velocity_errors, (telemetry.get("read_errors") or {}).get("present_velocity")],
            )
            safety = evaluate_load_current_safety(
                load_raw=load,
                current_raw=current,
                velocity=velocity,
                present=present,
                previous_present=None if not presents else presents[-1],
                load_run=high_load_run,
                config=config,
                load_usable=bool(load_assess["usable"]),
                current_usable=bool(current_assess["usable"]),
                velocity_usable=bool(velocity_assess["usable"]),
            )
            high_load_run = int(safety.load_run)
            velocity_safety = evaluate_velocity_hard_safety(
                velocity=velocity,
                previous_velocity=previous_velocity,
                consecutive_over=velocity_over_run,
                glitch_count=velocity_glitch_count,
                accel_consecutive_over=accel_over_run,
                config=config,
            )
            velocity_over_run = int(velocity_safety.consecutive_over)
            velocity_glitch_count = int(velocity_safety.glitch_count)
            accel_over_run = int(velocity_safety.accel_consecutive_over)
            previous_velocity = velocity
            if safety.abort_reason is not None:
                abort_reason = safety.abort_reason
                fatal = True
                _append_telemetry(
                    observation=observation,
                    state=state,
                    present=present,
                    telemetry=telemetry,
                    phase=phase,
                    unbounded_target=_hold_command(),
                    sent=_hold_command(),
                    velocity=velocity,
                    velocity_decoded=velocity_safety.velocity_decoded,
                    glitch=velocity_safety.glitch,
                )
                break
            if velocity_safety.abort_reason is not None:
                abort_reason = velocity_safety.abort_reason
                fatal = True
                _append_telemetry(
                    observation=observation,
                    state=state,
                    present=present,
                    telemetry=telemetry,
                    phase=phase,
                    unbounded_target=_hold_command(),
                    sent=_hold_command(),
                    velocity=velocity,
                    velocity_decoded=velocity_safety.velocity_decoded,
                    glitch=velocity_safety.glitch,
                )
                break

            if (
                phase == SEEK_PHASE_APPROACH_RAMP
                and abs(present - session_p1) <= float(config.approach_tolerance)
                and abs(float(last_command) - session_p1) <= float(config.approach_tolerance)
            ):
                phase = SEEK_PHASE_APPROACH_SETTLE
                phase_frames = 0
                settle_stable = 0

            unbounded_target = float(last_command)
            step = float(config.approach_ramp_step)
            stop_after_record = False

            if phase == SEEK_PHASE_APPROACH_RAMP:
                unbounded_target = session_p1
                step = float(config.approach_ramp_step)
                if abs(float(last_command) - present) > float(config.maximum_approach_lead):
                    unbounded_target = float(last_command)
                traveled = abs(float(last_command) - float(approach_origin))
                if traveled > float(config.approach_ramp_max_travel) + 1e-9:
                    abort_reason = "approach_p1_ramp_max_travel"
                    stop_after_record = True
            elif phase == SEEK_PHASE_APPROACH_SETTLE:
                unbounded_target = float(last_command)
                step = float(config.approach_ramp_step)
                speed_ok = False
                decoded_vel = velocity_safety.velocity_decoded
                if decoded_vel is None:
                    speed_ok = True
                elif abs(float(decoded_vel)) <= float(config.velocity_near_zero):
                    speed_ok = True
                elif presents and abs(present - presents[-1]) <= float(
                    config.plateau_present_span
                ):
                    speed_ok = True
                if (
                    abs(present - session_p1) <= float(config.approach_tolerance)
                    and speed_ok
                ):
                    settle_stable += 1
                else:
                    settle_stable = 0
            elif phase == SEEK_PHASE_CONTACT_SEARCH:
                step = float(config.contact_search_step)
                column_roi = roi_by_view["column"]
                wrist_roi = roi_by_view["wrist"]
                column_obs = observe_knife_edges(
                    column,
                    view="column",
                    previous_frame=previous_column,
                    gap_close_threshold_px=config.gap_close_threshold_px,
                    gripper_roi_xyxy=column_roi.get("xyxy"),
                    gripper_roi_provenance=str(column_roi.get("provenance") or ""),
                    allow_full_image=False,
                )
                wrist_obs = observe_knife_edges(
                    wrist,
                    view="wrist",
                    gap_close_threshold_px=config.gap_close_threshold_px,
                    gripper_roi_xyxy=wrist_roi.get("xyxy"),
                    gripper_roi_provenance=str(wrist_roi.get("provenance") or ""),
                    allow_full_image=False,
                )
                last_roi = {
                    "xyxy": column_obs.gripper_roi_xyxy,
                    "provenance": column_obs.gripper_roi_provenance,
                    "diagnostics": column_obs.gripper_roi_diagnostics,
                }
                last_roi_by_view = {
                    "column": {
                        "xyxy": column_obs.gripper_roi_xyxy,
                        "provenance": column_obs.gripper_roi_provenance,
                        "diagnostics": column_obs.gripper_roi_diagnostics,
                        "valid": column_obs.ok,
                        "view": "column",
                    },
                    "wrist": {
                        "xyxy": wrist_obs.gripper_roi_xyxy,
                        "provenance": wrist_obs.gripper_roi_provenance,
                        "diagnostics": wrist_obs.gripper_roi_diagnostics,
                        "valid": wrist_obs.ok,
                        "view": "wrist",
                    },
                }
                previous_column = column.copy()
                visual_column.append(column_obs.to_dict())
                visual_wrist.append(wrist_obs.to_dict())
                last_visual_contact = evaluate_visual_contact(
                    column_obs,
                    wrist_obs,
                    gap_close_threshold_px=config.gap_close_threshold_px,
                )
                visual_contact_rows.append(last_visual_contact)
                if last_visual_contact["formal_a"]:
                    visual_run += 1
                else:
                    visual_run = 0
                goal_for_plateau = float(
                    last_command
                    if _gripper_from_mapping(telemetry.get("goal")) is None
                    else _gripper_from_mapping(telemetry.get("goal"))
                )
                velocity_window = [*velocities, velocity][-int(config.consecutive_frames) :]
                velocity_drop = any(
                    item is None or not np.isfinite(float(item)) for item in velocity_window
                )
                had_velocity = any(
                    item is not None and np.isfinite(float(item))
                    for item in (*velocities, velocity)
                )
                plateau = evaluate_plateau(
                    [*presents, present],
                    [*goals, goal_for_plateau],
                    close_direction=close_direction,
                    consecutive_frames=config.consecutive_frames,
                    present_span_max=config.plateau_present_span,
                    goal_inward_min=config.goal_inward_min,
                    velocities=[*velocities, velocity],
                    velocity_near_zero=config.velocity_near_zero,
                    velocity_usable=bool(velocity_assess["usable"])
                    or (had_velocity and velocity_drop),
                    phase=SEEK_PHASE_CONTACT_SEARCH,
                )
                plateau_rows.append(plateau.to_dict())
                last_decision = evaluate_joint_contact(
                    visual_run,
                    plateau,
                    consecutive_frames=config.consecutive_frames,
                    visual_contact=last_visual_contact,
                )
                if last_visual_contact.get("view_conflict"):
                    contact = False
                    abort_reason = abort_reason or "view_conflict"
                    stop_after_record = True
                tracking_error = abs(goal_for_plateau - present)
                if (
                    tracking_error > float(config.maximum_command_present_error)
                    and not (plateau.plateau and plateau.goal_inward)
                ):
                    abort_reason = "command_present_error"
                    fatal = True
                    stop_after_record = True
                elif last_decision.contact:
                    if (
                        column_obs.u_hat_px is None
                        or column_obs.u_hat_provenance not in U_HAT_PROVENANCES
                    ):
                        abort_reason = "u_hat_unavailable"
                        stop_after_record = True
                    else:
                        contact = True
                        contact_phase = SEEK_PHASE_CONTACT_SEARCH
                        window = [*presents, present][-int(config.consecutive_frames) :]
                        true_contact_position = float(np.median(window))
                        true_contact_uncertainty = float(np.ptp(window))
                        u_hat_px = column_obs.u_hat_px
                        u_hat_provenance = column_obs.u_hat_provenance
                        unbounded_target = float(last_command)
                        stop_after_record = True
                elif search_settle_left > 0:
                    unbounded_target = float(last_command)
                    search_settle_left -= 1
                else:
                    next_cmd, travel_abort = next_close_command(
                        float(last_command),
                        close_direction=close_direction,
                        step=float(config.contact_search_step),
                        start_present=session_p1,
                        max_additional_travel=config.maximum_additional_travel,
                        lo=config.gripper_min,
                        hi=config.gripper_max,
                    )
                    if travel_abort is not None:
                        abort_reason = travel_abort
                        stop_after_record = True
                    else:
                        unbounded_target = next_cmd
                        search_settle_left = int(config.contact_search_settle_frames)

            if not stop_after_record:
                phase_limit = {
                    SEEK_PHASE_APPROACH_RAMP: int(
                        config.approach_ramp_max_frames or config.maximum_phase_frames
                    ),
                    SEEK_PHASE_APPROACH_SETTLE: int(
                        config.approach_settle_max_frames or config.maximum_phase_frames
                    ),
                    SEEK_PHASE_CONTACT_SEARCH: int(
                        config.contact_search_max_frames or config.maximum_phase_frames
                    ),
                }[phase]
                phase_frames += 1
                if phase_frames > phase_limit:
                    abort_reason = (
                        "max_phase_frames"
                        if phase == SEEK_PHASE_CONTACT_SEARCH
                        else f"{phase}_timeout"
                    )
                    if phase != SEEK_PHASE_CONTACT_SEARCH:
                        fatal = True
                    stop_after_record = True

            if stop_after_record:
                sent = _hold_command()
            else:
                sent = bound_command_step(
                    float(last_command),
                    float(unbounded_target),
                    step=step,
                    lo=config.gripper_min,
                    hi=config.gripper_max,
                )
                sent = float(hooks.write_gripper(sent))
                last_command = sent
            _append_telemetry(
                observation=observation,
                state=state,
                present=present,
                telemetry=telemetry,
                phase=phase,
                unbounded_target=unbounded_target,
                sent=sent,
                velocity=velocity,
                velocity_decoded=velocity_safety.velocity_decoded,
                glitch=velocity_safety.glitch,
            )
            if stop_after_record or contact:
                break
            if (
                phase == SEEK_PHASE_APPROACH_SETTLE
                and settle_stable >= int(config.approach_settle_stable_required)
            ):
                phase = SEEK_PHASE_CONTACT_SEARCH
                phase_frames = 0
                search_settle_left = 0
            hooks.sleep(period_s)
        else:
            if abort_reason is None and not contact:
                abort_reason = "max_phase_frames"
    except Exception as exc:
        abort_reason = f"exception:{type(exc).__name__}: {exc}"
        fatal = True
        if str(exc).startswith("image_loss"):
            abort_reason = str(exc)
        payload = _record_payload(_restore())
        raise SelfContactFatalError(abort_reason, payload) from exc

    restore = _restore()
    payload = _record_payload(restore)
    if fatal:
        raise SelfContactFatalError(abort_reason or "fatal_abort", payload)
    return payload


def persistable_self_contact_arrays(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Arrays that can be written beside a historical pulse.npz."""

    def _float_series(key: str) -> np.ndarray:
        values = []
        for item in record.get(key) or []:
            if item is None:
                values.append(np.nan)
            else:
                values.append(float(item))
        return np.asarray(values, dtype=np.float32)

    arrays = {
        "timestamps": _float_series("timestamps"),
        "presents": _float_series("presents"),
        "goals": _float_series("goals"),
        "commands": _float_series("commands"),
        "present_velocity": _float_series("present_velocity"),
        "moving": _float_series("moving"),
        "present_load": _float_series("present_load"),
        "present_current": _float_series("present_current"),
    }
    if record.get("targets") is not None:
        arrays["targets"] = _float_series("targets")
    if record.get("present_velocity_decoded") is not None:
        arrays["present_velocity_decoded"] = _float_series("present_velocity_decoded")
    if record.get("phases"):
        arrays["phases"] = np.asarray(list(record.get("phases") or []), dtype=object)
    if record.get("present_load_decoded") is not None:
        arrays["present_load_decoded"] = _float_series("present_load_decoded")
    else:
        arrays["present_load_decoded"] = np.asarray(
            [
                np.nan if item is None else float(item)
                for item in reinterpret_register_series(
                    record.get("present_load"), "present_load"
                )
            ],
            dtype=np.float32,
        )
    if record.get("present_current_decoded") is not None:
        arrays["present_current_decoded"] = _float_series("present_current_decoded")
    else:
        arrays["present_current_decoded"] = np.asarray(
            [
                np.nan if item is None else float(item)
                for item in reinterpret_register_series(
                    record.get("present_current"), "present_current"
                )
            ],
            dtype=np.float32,
        )
    arm = np.asarray(record.get("arm_states") or [], dtype=np.float32)
    if arm.size:
        arrays["arm_states"] = arm
    for key in ("column_planning_224", "wrist_planning_224"):
        frames = record.get(key)
        if frames is None:
            continue
        stacked = np.asarray(frames)
        if stacked.ndim == 4 and stacked.shape[-1] == 3:
            arrays[key] = stacked.astype(np.uint8, copy=False)
    if record.get("evidence_frame_ids") is not None:
        arrays["evidence_frame_ids"] = np.asarray(
            record.get("evidence_frame_ids") or [], dtype=np.int32
        )
    if record.get("evidence_phases") is not None:
        arrays["evidence_phases"] = np.asarray(
            list(record.get("evidence_phases") or []), dtype=object
        )
    return arrays
