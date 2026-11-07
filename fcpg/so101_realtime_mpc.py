"""Pure control contracts for the SO-101 receding-horizon runtime.

This module deliberately has no LeRobot or web-server dependency.  It owns the
small state machine and fail-closed checks shared by replay tests and the real
hardware runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from fcpg.so101_retrieval_mpc import SO101_POSITION_LOWER, SO101_POSITION_UPPER


class ManipulationPhase(str, Enum):
    TRAJECTORY = "trajectory"
    APPROACH = "approach"
    GRASP = "grasp"
    TRANSPORT = "transport"
    RELEASE = "release"
    DONE = "done"
    FAULT = "fault"


@dataclass(frozen=True)
class PhaseThresholds:
    pickup_near_px: float
    drop_near_px: float
    pickup_near_m: float
    drop_near_m: float
    gripper_closed: float
    gripper_open: float
    grasp_roi_px: float
    confirmations: int = 2
    # Aperture at/below which a closed gripper is considered EMPTY (bottomed
    # out near the physical stop). A real grasp stalls against the object
    # well above this; attachment evidence requires the stall.
    gripper_empty_closed: float = 0.0

    def __post_init__(self) -> None:
        finite_positive = (
            self.pickup_near_px,
            self.drop_near_px,
            self.pickup_near_m,
            self.drop_near_m,
            self.grasp_roi_px,
        )
        if any(not np.isfinite(value) or value <= 0 for value in finite_positive):
            raise ValueError("phase thresholds must be finite and positive")
        if not np.isfinite(self.gripper_closed) or not np.isfinite(self.gripper_open):
            raise ValueError("gripper thresholds must be finite")
        if self.gripper_closed >= self.gripper_open:
            raise ValueError("gripper_closed must be lower than gripper_open")
        if (
            not np.isfinite(self.gripper_empty_closed)
            or self.gripper_empty_closed >= self.gripper_closed
        ):
            raise ValueError(
                "gripper_empty_closed must be finite and below gripper_closed"
            )
        if self.confirmations <= 0:
            raise ValueError("confirmations must be positive")


# Same degenerate-tip floor as wrist_tip_similarity_correction: closed or
# near-closed jaws make the two-fingertip similarity unusable.
JAW_COLLAPSED_TIP_SEPARATION_PX = 8.0
#: Servo settle tolerance around the calibrated empty-closed gripper position.
#: Live, a command of 1.774 settles at 1.84, so 1.0 is an order of magnitude
#: above the settle error while staying far below the aperture any graspable
#: object stalls the jaws at (the fish stalls them near 14-18).
EMPTY_CLOSED_GRIPPER_TOLERANCE = 1.0
# Cycle / planner share these. Projective "held" at the open stop is not a
# pinch; an empty-bite re-approach must open above the closed calibration stop.
HOLD_CONFIRM_MAX_JAWS_DEG = 20.0
JAWS_OPEN_MIN_DEG = 11.81


def jaws_are_pinched(gripper_position: float | None) -> bool:
    """True when the live aperture is at or below the hold-confirm ceiling."""

    if gripper_position is None or not np.isfinite(gripper_position):
        return False
    return float(gripper_position) <= HOLD_CONFIRM_MAX_JAWS_DEG


def jaws_in_holding_aperture(
    gripper_position: float | None,
    empty_closed: float | None,
) -> bool:
    """True for a pinch that is not the empty-closed stop and not still open."""

    if (
        gripper_position is None
        or empty_closed is None
        or not np.isfinite(gripper_position)
        or not np.isfinite(empty_closed)
    ):
        return False
    return (
        float(gripper_position)
        > float(empty_closed) + EMPTY_CLOSED_GRIPPER_TOLERANCE
        and float(gripper_position) <= HOLD_CONFIRM_MAX_JAWS_DEG
    )


def hold_filled_actor_allowed(
    *,
    allow_close: bool,
    current_gripper: float | None,
) -> bool:
    """Actor hold_filled_* may stay only on a permitted, already-pinched close."""

    return bool(allow_close) and jaws_are_pinched(current_gripper)


def empty_bite_reapproach_open_targets(
    current_gripper: float,
    *,
    gripper_open: float,
    calibration_open: float,
) -> tuple[float, float]:
    """Open-jaw command targets after an empty bite. Always above ~11°."""

    opened = min(
        max(
            float(gripper_open),
            float(current_gripper) + 2.0,
            JAWS_OPEN_MIN_DEG,
        ),
        float(calibration_open) + 1.0,
    )
    wide = min(opened + 4.0, float(calibration_open) + 2.0)
    return (float(opened), float(wide))


THREE_POINT_VERDICTS = (
    "held",
    "empty_bite",
    "in_corridor",
    "approach_more",
    "insufficient",
)


def half_extent_along(
    direction: np.ndarray | None,
    axis_px: np.ndarray | None,
    length_px: float | None,
    width_px: float | None,
) -> float | None:
    """How far the object reaches from its centre, measured along ``direction``.

    An object that is long one way and narrow the other does not present the
    same size to every approach, so the extent the jaws have to cover depends
    on which way they close. Returns ``None`` when the object's shape was not
    measured, leaving the caller to fall back on treating it as a point.
    """
    if direction is None or axis_px is None or length_px is None or width_px is None:
        return None
    axis = np.asarray(axis_px, dtype=np.float64).reshape(-1)
    if axis.shape != (2,) or not np.all(np.isfinite(axis)):
        return None
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-6:
        return None
    axis = axis / norm
    across = np.array([-axis[1], axis[0]])
    return 0.5 * (
        float(length_px) * abs(float(np.dot(axis, direction)))
        + float(width_px) * abs(float(np.dot(across, direction)))
    )


def closing_misalignment_deg(
    closing: np.ndarray, target_axis_px: np.ndarray | None
) -> float | None:
    """How far the jaws are from closing square across the object.

    Zero means the jaws come together straight across the object's length,
    which is the only way two fingers hold a long thing. Ninety means they are
    closing along it, where there is nothing to pinch and the object is free to
    squirt out either end.
    """
    if target_axis_px is None:
        return None
    axis = np.asarray(target_axis_px, dtype=np.float64).reshape(-1)
    if axis.shape != (2,) or not np.all(np.isfinite(axis)):
        return None
    axis_norm = float(np.linalg.norm(axis))
    closing_norm = float(np.linalg.norm(closing))
    if axis_norm <= 1.0e-6 or closing_norm <= 1.0e-6:
        return None
    axis = axis / axis_norm
    # Measured against the direction square to the object, because that is the
    # one the jaws are supposed to travel along.
    across = np.array([-axis[1], axis[0]])
    cosine = abs(float(np.dot(across, closing / closing_norm)))
    # The object's line has no head or tail as far as closing is concerned, so
    # the answer folds into a quarter turn: 0 is square across, 90 is along.
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def judge_three_point_closure(
    anchors_px: np.ndarray,
    target_px: np.ndarray | None,
    *,
    open_tip_separation_px: float | None = None,
    empty_close_tip_separation_px: float | None = None,
    gripper_position: float | None = None,
    empty_closed_gripper_position: float | None = None,
    target_axis_px: np.ndarray | None = None,
    target_length_px: float | None = None,
    target_width_px: float | None = None,
) -> dict[str, Any]:
    """Judge held / empty-bite / approach from FCPG three-anchor geometry.

    Anchors are ``[center, fixed_tip, moving_tip]`` in one camera. The closing
    trajectory is the moving tip traveling toward the fixed tip. The object is
    in the corridor when it sits between those tips; it is held once that
    corridor has closed far enough along the calibrated open→empty-close
    track. Collapsed tips with the target outside the corridor are an empty
    bite.

    ``held`` is additionally falsified by the gripper's own present position.
    A grasped object holds the jaws apart by its own width, so reaching the
    calibrated empty-closed position means the jaws met each other and nothing
    can be between them, whatever the pixels overlap. This matters because the
    span references are nominal-FK while ``anchors_px`` are detected: live, the
    shut jaws measure 10.5 px against a 4.5 px nominal empty-close, so
    ``collapsed`` never fires and any pickup centroid landing near the tip
    midpoint reads as ``held`` with empty jaws.
    """

    anchors = np.asarray(anchors_px, dtype=np.float64)
    if anchors.shape != (3, 2) or not np.all(np.isfinite(anchors)):
        return {
            "verdict": "insufficient",
            "target_between_tips": False,
            "jaws_collapsed": False,
            "tip_separation_px": None,
            "closing_progress": None,
        }
    fixed = anchors[1]
    moving = anchors[2]
    span = float(np.linalg.norm(moving - fixed))
    empty_span = (
        float(empty_close_tip_separation_px)
        if empty_close_tip_separation_px is not None
        and np.isfinite(empty_close_tip_separation_px)
        else None
    )
    open_span = (
        float(open_tip_separation_px)
        if open_tip_separation_px is not None and np.isfinite(open_tip_separation_px)
        else None
    )
    collapsed = span < JAW_COLLAPSED_TIP_SEPARATION_PX
    if empty_span is not None:
        collapsed = collapsed or span <= empty_span + 1.0
    closing_progress: float | None = None
    if (
        open_span is not None
        and empty_span is not None
        and (open_span - empty_span) > 1.0
    ):
        closing_progress = float(
            np.clip((open_span - span) / (open_span - empty_span), 0.0, 1.0)
        )
        closed_enough = closing_progress >= 0.5
    else:
        closed_enough = collapsed

    target_between = False
    target_seen = False
    if target_px is not None:
        target = np.asarray(target_px, dtype=np.float64).reshape(-1)
        if target.shape == (2,) and np.all(np.isfinite(target)):
            target_seen = True
            midpoint = 0.5 * (fixed + moving)
            delta = target - midpoint
            closing = moving - fixed
            norm = float(np.linalg.norm(closing))
            reach = 0.5 * max(span, 1.0e-6)
            half_extent = half_extent_along(
                closing / norm if norm > 1.0e-6 else None,
                target_axis_px,
                target_length_px,
                target_width_px,
            )
            if half_extent is None:
                target_between = float(np.linalg.norm(delta)) <= reach
            else:
                # Measured along the closing direction, and against the
                # object's own extent rather than its centre. A long object
                # laid across the jaws has its centre wherever its mass
                # happens to be -- for a fish, up in its body -- while the part
                # between the tips is somewhere else entirely. Asking only
                # about the centre stalled a grasp with the jaws straddling
                # the tail: 54 px from the midpoint against a 43 px span, so
                # "not between tips", with the object squarely in the gripper.
                axis = closing / norm
                along = abs(float(np.dot(delta, axis)))
                target_between = (along - half_extent) <= reach

    jaws_met_each_other = False
    if (
        gripper_position is not None
        and empty_closed_gripper_position is not None
        and np.isfinite(gripper_position)
        and np.isfinite(empty_closed_gripper_position)
    ):
        jaws_met_each_other = float(gripper_position) <= (
            float(empty_closed_gripper_position) + EMPTY_CLOSED_GRIPPER_TOLERANCE
        )

    if jaws_met_each_other:
        verdict = "empty_bite"
    elif target_between and closed_enough:
        verdict = "held"
    elif target_between:
        verdict = "in_corridor"
    elif collapsed:
        verdict = "empty_bite"
    elif (
        target_px is not None
        and not target_between
        and closed_enough
    ):
        # Jaws mostly closed but the pickup is outside the tip corridor:
        # empty bite even when tip span has not yet hit the collapsed floor
        # (live: close-from-9.72 reached sep~15px / progress~0.74 with
        # between=False and stuck in GRASP without regressing).
        verdict = "empty_bite"
    elif target_seen:
        verdict = "approach_more"
    else:
        # No target reached this judge, so it has nothing to say about where the
        # object sits. Saying "approach_more" would publish the absence of
        # evidence as evidence against, and the phase machine reads it that way:
        # "approach_more" is neither a corridor nor a gap, so it blocks the
        # entry to GRASP *and* the column-nearness fallback meant to cover a
        # blind wrist. Live that pinned APPROACH with the column reporting 6.6mm
        # to the pickup. The physical readings above still stand on their own --
        # jaws that met or collapsed are an empty bite whether or not anything
        # was seen -- and only the genuinely unknown case lands here.
        verdict = "insufficient"
    return {
        "verdict": verdict,
        "target_between_tips": target_between,
        "jaws_collapsed": collapsed,
        "tip_separation_px": span,
        "closing_progress": closing_progress,
        "closing_misalignment_deg": closing_misalignment_deg(
            moving - fixed, target_axis_px
        ),
    }


@dataclass(frozen=True)
class PhaseEvidence:
    gripper_position: float
    gripper_to_pickup_px: float | None
    gripper_to_drop_px: float | None
    gripper_to_pickup_m: float | None
    gripper_to_drop_m: float | None
    wrist_pickup_distance_px: float | None
    pickup_visible: bool
    drop_zone_visible: bool
    pickup_inside_drop_zone: bool
    three_point_verdict: str | None = None
    #: Planner-stamped drop reliability for TRANSPORT→RELEASE. Held occlusion
    #: and low-margin appearance must not open jaws mid-table; default True
    #: keeps unit fixtures that omit the field behavior-compatible.
    drop_phase_gate_reliable: bool = True
    along_close_sigma_m: float | None = None
    max_sigma_m: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.gripper_position):
            raise ValueError("gripper_position must be finite")
        for name in (
            "gripper_to_pickup_px",
            "gripper_to_drop_px",
            "gripper_to_pickup_m",
            "gripper_to_drop_m",
            "wrist_pickup_distance_px",
            "along_close_sigma_m",
            "max_sigma_m",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            self.three_point_verdict is not None
            and self.three_point_verdict not in THREE_POINT_VERDICTS
        ):
            raise ValueError("three_point_verdict is not a known closure verdict")


class PhaseTracker:
    """Causal manipulation phase tracker using only deployable evidence."""

    def __init__(
        self,
        thresholds: PhaseThresholds,
        *,
        initial: ManipulationPhase = ManipulationPhase.APPROACH,
    ) -> None:
        self.thresholds = thresholds
        self.phase = initial
        self._candidate_phase: ManipulationPhase | None = None
        self._candidate_count = 0
        self._three_point_verdict: str | None = None
        self._empty_bite_reapproach = False

    def _confirm(self, candidate: ManipulationPhase | None) -> None:
        if candidate is None:
            self._candidate_phase = None
            self._candidate_count = 0
            return
        if candidate == self._candidate_phase:
            self._candidate_count += 1
        else:
            self._candidate_phase = candidate
            self._candidate_count = 1
        if self._candidate_count >= self.thresholds.confirmations:
            self.phase = candidate
            if candidate != ManipulationPhase.APPROACH:
                self._empty_bite_reapproach = False
            self._candidate_phase = None
            self._candidate_count = 0

    def update(self, evidence: PhaseEvidence) -> ManipulationPhase:
        if self.phase in {ManipulationPhase.DONE, ManipulationPhase.FAULT}:
            return self.phase

        candidate: ManipulationPhase | None = None
        closure = evidence.three_point_verdict
        self._three_point_verdict = closure
        if self.phase == ManipulationPhase.APPROACH:
            in_corridor = closure in {"held", "in_corridor"}
            near_column = (
                evidence.gripper_to_pickup_px is not None
                and evidence.gripper_to_pickup_px
                <= self.thresholds.pickup_near_px
                and evidence.gripper_to_pickup_m is not None
                and evidence.gripper_to_pickup_m <= self.thresholds.pickup_near_m
            )
            # Three-point corridor/held is sufficient on its own. Live stuck
            # approach×16 with three_point=held while column template margin
            # fell below min_margin (pickup_visible=False / jaw occlusion).
            # Column nearness remains a fallback only when 3-point is missing.
            if in_corridor:
                candidate = ManipulationPhase.GRASP
            elif evidence.pickup_visible and closure in {
                None,
                "insufficient",
            } and near_column:
                candidate = ManipulationPhase.GRASP
        elif self.phase == ManipulationPhase.GRASP:
            if closure == "empty_bite":
                if not self._holding_aperture(evidence):
                    self.regress_to_approach()
                    return self.phase
            # Held is the 3-point closing corridor with the object between
            # the tips. Do not require pickup_visible: the appearance
            # template is often unreliable exactly when the fish is in-hand.
            # Projective held is not a pinch: close_allowed plus jaws <= 20°.
            elif closure == "held" and self._transport_entry_allowed(evidence):
                candidate = ManipulationPhase.TRANSPORT
        elif self.phase == ManipulationPhase.TRANSPORT:
            if closure == "empty_bite" and not self._holding_aperture(evidence):
                # Jaws pinched shut on air, or already open. A mid-closed
                # aperture is still a hold even when the far view reports
                # empty_bite.
                self.regress_to_approach()
                return self.phase
            # A confirmed hold stays in TRANSPORT and drives the drop even
            # when the fish is still the pickup blob and the box is briefly
            # unseen. Regressing here yo-yoed the arm back down onto the
            # table (live: held + pickup_visible + drop_zone_visible=False).
            # Empty-bite still regresses above. A pinch that is not held
            # and has no drop track still retreats.
            still_at_pick = (
                closure != "held"
                and not self._holding_aperture(evidence)
                and evidence.pickup_visible
                and evidence.gripper_to_pickup_m is not None
                and evidence.gripper_to_pickup_m
                <= max(0.04, 2.0 * self.thresholds.pickup_near_m)
                and not evidence.drop_zone_visible
                and self._jaws_empty_closed(evidence)
            )
            if still_at_pick:
                self.regress_to_approach()
                return self.phase
            # Require pixel AND metric nearness, plus planner-stamped drop
            # reliability (refuses held_gripper_occlusion mid-table).
            if (
                evidence.drop_zone_visible
                and evidence.drop_phase_gate_reliable
                and evidence.gripper_to_drop_px is not None
                and evidence.gripper_to_drop_px <= self.thresholds.drop_near_px
                and evidence.gripper_to_drop_m is not None
                and evidence.gripper_to_drop_m <= self.thresholds.drop_near_m
            ):
                candidate = ManipulationPhase.RELEASE
        elif self.phase == ManipulationPhase.RELEASE:
            deposited = (
                evidence.pickup_inside_drop_zone
                and evidence.gripper_position >= self.thresholds.gripper_open
            )
            # RELEASE is entered only under the strict near-box gate. Once
            # there, confirmed-open jaws latch DONE — do not re-require
            # drop_zone_visible / drop_near every tick (occlusion flaps after
            # open parked the overnight run forever in RELEASE).
            released_open = (
                evidence.gripper_position >= self.thresholds.gripper_open
            )
            if deposited or released_open:
                candidate = ManipulationPhase.DONE

        self._confirm(candidate)
        return self.phase

    def _jaws_empty_closed(self, evidence: PhaseEvidence) -> bool:
        return (
            evidence.gripper_position
            <= float(self.thresholds.gripper_empty_closed)
            + EMPTY_CLOSED_GRIPPER_TOLERANCE
        )

    def _holding_aperture(self, evidence: PhaseEvidence) -> bool:
        return jaws_in_holding_aperture(
            evidence.gripper_position,
            float(self.thresholds.gripper_empty_closed),
        )

    def _transport_entry_allowed(self, evidence: PhaseEvidence) -> bool:
        if not jaws_are_pinched(evidence.gripper_position):
            return False
        return close_allowed_for_pickup(
            three_point_verdict=evidence.three_point_verdict,
            gripper_to_pickup_px=evidence.gripper_to_pickup_px,
            gripper_to_pickup_m=evidence.gripper_to_pickup_m,
            pickup_near_px=self.thresholds.pickup_near_px,
            pickup_near_m=self.thresholds.pickup_near_m,
            along_close_sigma_m=evidence.along_close_sigma_m,
            max_sigma_m=evidence.max_sigma_m,
        )

    def regress_to_approach(self) -> None:
        """Retreat when the three-point track reports an empty bite.

        Collapsed / mid-closed tips with the target outside the closing
        corridor mean the jaws closed on air or pinched the object from
        outside. Re-aim from APPROACH (from GRASP or a false TRANSPORT held).
        """

        if self.phase in {
            ManipulationPhase.GRASP,
            ManipulationPhase.TRANSPORT,
        }:
            self.phase = ManipulationPhase.APPROACH
            self._candidate_phase = None
            self._candidate_count = 0
            self._empty_bite_reapproach = True

    @property
    def empty_bite_reapproach(self) -> bool:
        """True after an empty bite until APPROACH re-enters GRASP."""

        return self._empty_bite_reapproach

    def fault(self) -> None:
        self.phase = ManipulationPhase.FAULT
        self._candidate_phase = None
        self._candidate_count = 0

    def reset(self) -> None:
        self.phase = ManipulationPhase.APPROACH
        self._candidate_phase = None
        self._candidate_count = 0
        self._three_point_verdict = None
        self._empty_bite_reapproach = False

    def status(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "pending_phase": (
                None if self._candidate_phase is None else self._candidate_phase.value
            ),
            "pending_confirmations": self._candidate_count,
            "required_confirmations": self.thresholds.confirmations,
            "three_point_verdict": self._three_point_verdict,
            "empty_bite_reapproach": self._empty_bite_reapproach,
        }


def target_role_for_phase(phase: ManipulationPhase) -> str | None:
    if phase in {ManipulationPhase.APPROACH, ManipulationPhase.GRASP}:
        return "pickup"
    if phase in {ManipulationPhase.TRANSPORT, ManipulationPhase.RELEASE}:
        return "drop_zone"
    return None


# Retained as an API alias for older replay tools.  Values are generic roles,
# never color or object-class names.
target_kind_for_phase = target_role_for_phase


def exclusive_phase_gripper_open(
    phase: ManipulationPhase,
    *,
    current_gripper: float,
    gripper_empty_closed: float,
    empty_bite_reapproach: bool,
    three_point_verdict: str | None = None,
) -> bool:
    """True for APPROACH that must open in place before the arm moves.

    First approach with parked-empty-closed jaws, and empty-bite reapproach
    while the jaws are still below ``JAWS_OPEN_MIN_DEG``, take the arm-frozen
    open path. Mid-close first-approach apertures stay on IK/V11. GRASP
    never uses this path.
    """

    if phase != ManipulationPhase.APPROACH:
        return False
    if not np.isfinite(current_gripper) or not np.isfinite(gripper_empty_closed):
        return False
    if empty_bite_reapproach:
        return float(current_gripper) < JAWS_OPEN_MIN_DEG
    if three_point_verdict in {"held", "in_corridor"}:
        return False
    return float(current_gripper) <= float(gripper_empty_closed)


def grasp_aperture_candidates(
    aperture: float,
    *,
    steps: int,
    closed_floor: float,
    open_ceiling: float,
    rates: Sequence[float] = (-0.6, -0.25, 0.0, 0.6),
) -> np.ndarray | None:
    """Per-step gripper ramps to *propose* during GRASP, one row per rate.

    This replaces the verdict-driven widen/squeeze script. Nothing here
    decides whether to close: the rows span close-fast / close-slow / hold /
    open, and :func:`predicted_closure_cost` scores them on the LeWM rollout.
    ``rates`` are degrees per step and are a sampling grid, not thresholds.
    """

    if steps <= 0 or not np.isfinite(aperture):
        return None
    if not np.isfinite(closed_floor) or not np.isfinite(open_ceiling):
        return None
    if float(open_ceiling) <= float(closed_floor):
        return None
    idx = np.arange(1, int(steps) + 1, dtype=np.float64)
    ramps = float(aperture) + np.asarray(rates, dtype=np.float64)[:, None] * idx[None]
    return np.clip(ramps, float(closed_floor), float(open_ceiling)).astype(np.float32)


def detected_tip_separation_px(
    anchors_px: np.ndarray, valid: np.ndarray | None = None
) -> float:
    """Tip span the detector reports, in the pixels of the view it was read in.

    Callers pair this with the FK projection of the *same* view to build the
    corridor scale. Taking the two spans from different cameras produces a ratio
    with no geometric meaning.
    """

    anchors = np.asarray(anchors_px, dtype=np.float64)
    if anchors.shape != (3, 2) or not np.all(np.isfinite(anchors[1:])):
        return float("nan")
    if valid is not None:
        flags = np.asarray(valid, dtype=bool).reshape(-1)
        if flags.shape != (3,) or not bool(flags[1] and flags[2]):
            return float("nan")
    return float(np.linalg.norm(anchors[2] - anchors[1]))


def aperture_hold_cost(
    predicted_anchor_px: np.ndarray,
    *,
    current_tip_separation_px: float,
    horizon_index: int = 0,
) -> np.ndarray:
    """Prefer leaving the jaw where it is. For when the grasp is not yet reachable.

    While the pickup sits outside what the open jaws could contain, the closure
    objective has nothing to say -- neither closing nor opening brings the grasp
    closer, only the arm does. Left to score it anyway it does not stay quiet:
    opening widens the corridor, a wider corridor shrinks the exclusion penalty,
    and gaping becomes the cheapest answer to a pickup that is out of reach.
    Live that showed up as the jaw closing to 11 degrees and then flapping back
    open whenever the wrist detector produced an outlier. Holding is the honest
    preference until the arm has done its part.
    """

    pixels = np.asarray(predicted_anchor_px, dtype=np.float64)
    if pixels.ndim != 4 or pixels.shape[-2:] != (3, 2):
        raise ValueError("predicted_anchor_px must be [N, horizon, 3, 2]")
    if not -pixels.shape[1] <= int(horizon_index) < pixels.shape[1]:
        raise ValueError("horizon_index is outside the rollout")
    current = float(current_tip_separation_px)
    costs = np.full(int(pixels.shape[0]), np.nan, dtype=np.float64)
    if not np.isfinite(current) or current <= 0.0:
        return costs
    for index in range(int(pixels.shape[0])):
        terminal = pixels[index, int(horizon_index)]
        if not np.all(np.isfinite(terminal)):
            continue
        span = float(np.linalg.norm(terminal[2] - terminal[1]))
        costs[index] = abs(span - current) / current
    return costs


def predicted_closure_cost(
    predicted_anchor_px: np.ndarray,
    target_px: np.ndarray | None,
    *,
    current_tip_separation_px: float,
    horizon_index: int = -1,
    corridor_floor_px: float = JAW_COLLAPSED_TIP_SEPARATION_PX,
    outside_penalty: float = 2.0,
    target_axis_px: np.ndarray | None = None,
    target_length_px: float | None = None,
    target_width_px: float | None = None,
    alignment_penalty: float = 1.0,
) -> np.ndarray:
    """Score candidates on predicted jaw closure around the pickup.

    When the object's shape has been measured, candidates are also charged for
    closing at an angle to it. Two fingers hold a long object by pinching
    across its width; closing along its length has nothing to pinch, and the
    charge is what buys the wrist rotation that squares the jaws up. It is a
    predicted consequence of each candidate action, so which way to turn the
    wrist is ranked by the rollout rather than scripted. At ``1.0`` a fully
    misaligned jaw costs as much as a full close is worth, so the ranker will
    spend a round turning rather than close across the object's length.

    ``predicted_anchor_px`` is ``[N, horizon, 3, 2]``, anchors ordered
    ``[center, fixed_tip, moving_tip]``. Cheapest outcome is tips that close
    while the pickup stays inside them; closing so far that the jaws would push
    the pickup out is the pinch-from-outside / empty-bite failure, and
    ``outside_penalty`` sets what that costs relative to what closing saves. At
    2.0 a candidate that would exclude the pickup by half the open jaw gives up
    more than a full close is worth, so the jaws wait for the arm to centre
    them. Grasp progress is a scored objective, not a scripted aperture ramp.

    ``horizon_index`` selects which step of the rollout is scored. MPC executes
    only the first chunk and replans, so scoring the far end judges a trajectory
    that will be discarded, and it judges it against a stale picture: the pickup
    pixel is measured now, while the anchors have travelled a chunk of arm
    motion away from it. In the wrist view, which rides on the gripper, that is
    not even a valid projection. Live at the horizon end the arm's own advance
    put the jaws 20-40px from a pickup that was centred to within 1px, and every
    closing rate saturated the exclusion penalty.

    ``current_tip_separation_px`` and the anchors and ``target_px`` must all be
    read in one view. There used to be a ``corridor_scale`` here to reconcile
    the FK jaw span with the wider one the detector reports, and it looked
    necessary because it was being fed a wrist span over a column projection.
    Within a single view the two spans are the same number -- FCPG corrects the
    anchors per point, so projecting the current state through that correction
    returns the detected anchors exactly, and the ratio is 1 by construction.
    The knob could only ever be non-unity by mixing cameras, which is not a
    projection correction but the difference between two focal lengths.
    """

    pixels = np.asarray(predicted_anchor_px, dtype=np.float64)
    if pixels.ndim != 4 or pixels.shape[-2:] != (3, 2):
        raise ValueError("predicted_anchor_px must be [N, horizon, 3, 2]")
    count = int(pixels.shape[0])
    costs = np.full(count, np.nan, dtype=np.float64)
    current = float(current_tip_separation_px)
    if not np.isfinite(current) or current <= 0.0:
        return costs
    target: np.ndarray | None = None
    if target_px is not None:
        candidate = np.asarray(target_px, dtype=np.float64).reshape(-1)
        if candidate.shape == (2,) and np.all(np.isfinite(candidate)):
            target = candidate
    if not -pixels.shape[1] <= int(horizon_index) < pixels.shape[1]:
        raise ValueError("horizon_index is outside the rollout")
    for index in range(count):
        terminal = pixels[index, int(horizon_index)]
        if not np.all(np.isfinite(terminal)):
            continue
        fixed = terminal[1]
        moving = terminal[2]
        span = float(np.linalg.norm(moving - fixed))
        closure = float(np.clip(span / current, 0.0, 2.0))
        if target is None:
            costs[index] = closure
            continue
        # Split the pickup's offset along the jaw axis from its distance in
        # front of the jaw line. They are different failures and a plain
        # euclidean norm hides which one is happening: measured live the pickup
        # sat 19px along a 23px reach -- laterally perfect -- and 62px in front,
        # and the norm read 65px and scored the grasp as hopelessly off to one
        # side. Closing then cost more than doing nothing at every rate offered
        # (-0.60 -> 7.65, -0.25 -> 5.14, hold -> 1.81), so the ranker could
        # never finish the grasp, and the number gave no hint that what was
        # missing was approach depth.
        axis = (moving - fixed) / span if span > 0.0 else np.asarray([1.0, 0.0])
        delta = target - 0.5 * (fixed + moving)
        along = abs(float(delta @ axis))
        across = abs(float(delta[0] * axis[1] - delta[1] * axis[0]))
        # Zero for as long as the closing jaws would still contain the pickup,
        # growing once they would sweep past it. Because a contained pickup pays
        # nothing, scaling by the terminal reach is safe: it only ever scales a
        # real exclusion, and closing runs to the narrowest jaw that still
        # holds. The penalty this replaced grew with the offset at all times and
        # divided by the same shrinking reach, so closing worsened its own
        # denominator and the objective had no minimum at a closed jaw.
        reach = 0.5 * max(span, float(corridor_floor_px))
        # Only the along-axis offset: that is the one direction closing sweeps
        # through, so it is the only one a narrower jaw can leave behind. Across
        # was charged here too, and because the charge is divided by a reach that
        # shrinks with the jaw, any pickup off the jaw line made closing cost
        # more the further it went. Live, with the pickup 10px off the line, the
        # cost fell to a minimum at a 20px span and then climbed to 3.4 at 6px,
        # so the cheapest jaw was the widest one and the ranker held the gripper
        # at its calibrated full-open 31.4deg through every grasp round. Whether
        # the pickup is off the line at all is the reachability gate's question,
        # asked once against the jaw as it stands, and closing it is the arm's
        # job, not the jaw's.
        #
        # An object is only as wide as the jaws see it from the side they are
        # coming at it. A long object presents its length to jaws closing along
        # it and its width to jaws closing across it, and only the second can
        # be pinched -- close along a fish and it slides out the end. So the
        # part of the offset the jaws can still reach is set by the object's
        # own extent in the closing direction, and the alignment charge below
        # buys the wrist rotation that shrinks it.
        half_extent = half_extent_along(
            axis, target_axis_px, target_length_px, target_width_px
        )
        excluded = max(0.0, along - (half_extent or 0.0) - reach)
        misaligned = closing_misalignment_deg(moving - fixed, target_axis_px)
        alignment = 0.0
        if misaligned is not None:
            alignment = float(alignment_penalty) * (misaligned / 90.0)
        costs[index] = (
            closure
            + float(outside_penalty) * float(np.clip(excluded / reach, 0.0, 4.0))
            + alignment
        )
    return costs


GRASP_EMPTY_AIR_GAP_DEG = 6.0


def pickup_is_metrically_near(
    *,
    gripper_to_pickup_px: float | None,
    gripper_to_pickup_m: float | None,
    pickup_near_px: float,
    pickup_near_m: float,
) -> bool:
    """True only when the configured pickup is near in pixels AND metres."""

    if (
        gripper_to_pickup_px is None
        or gripper_to_pickup_m is None
        or not np.isfinite(gripper_to_pickup_px)
        or not np.isfinite(gripper_to_pickup_m)
    ):
        return False
    return (
        float(gripper_to_pickup_px) <= float(pickup_near_px)
        and float(gripper_to_pickup_m) <= float(pickup_near_m)
    )


def close_allowed_for_pickup(
    *,
    three_point_verdict: str | None,
    gripper_to_pickup_px: float | None,
    gripper_to_pickup_m: float | None,
    pickup_near_px: float,
    pickup_near_m: float,
    along_close_sigma_m: float | None = None,
    max_sigma_m: float | None = None,
) -> bool:
    """True only for a confirmed hold or a metrically near pickup.

    Pixel ``in_corridor`` is projective: a far object can sit between the
    image of the tips. That bit must not arm a close. ``empty_bite`` never
    closes. Missing range fails closed.

    ``along_close_sigma_m`` / ``max_sigma_m`` are the optional covariance
    gate.  When ``max_sigma_m`` is set, a missing, non-finite, or too-large
    posterior std fails closed -- including on a ``held`` verdict, because
    that verdict is itself produced by the same projective geometry.
    Omitted, the gate is inactive and the historical predicate is unchanged.
    """

    if max_sigma_m is not None:
        if (
            along_close_sigma_m is None
            or not np.isfinite(along_close_sigma_m)
            or float(along_close_sigma_m) > float(max_sigma_m)
        ):
            return False
    if three_point_verdict == "empty_bite":
        return False
    if three_point_verdict == "held":
        return True
    return pickup_is_metrically_near(
        gripper_to_pickup_px=gripper_to_pickup_px,
        gripper_to_pickup_m=gripper_to_pickup_m,
        pickup_near_px=pickup_near_px,
        pickup_near_m=pickup_near_m,
    )


def grasp_closed_floor(
    *,
    held: bool,
    empty_closed: float,
    air_gap_deg: float = GRASP_EMPTY_AIR_GAP_DEG,
) -> float:
    """GRASP may reach empty-closed only after a confirmed hold.

    On air, the floor stays above the empty-closed stop so one scaled jaw
    grid cannot park the gripper shut.
    """

    floor = float(empty_closed)
    if held:
        return floor
    return floor + float(air_gap_deg)


def keep_v11_candidates_phase_safe(
    phase_safe: np.ndarray,
    candidate_metadata: Sequence[Mapping[str, Any]],
    *,
    prefixes: np.ndarray | None = None,
    allow_close: bool = True,
    receding: np.ndarray | None = None,
    current_gripper: float | None = None,
    empty_bite_reapproach: bool = False,
) -> np.ndarray:
    """V11 prefixes stay eligible unless they close while receding or far.

    Actor ``hold_filled_*`` is dropped when close is not allowed or the
    jaws are still above the hold-confirm ceiling. After an empty bite,
    prefixes that stay below ``JAWS_OPEN_MIN_DEG`` are not force-kept.
    """

    out = np.asarray(phase_safe, dtype=bool).copy()
    if out.shape != (len(candidate_metadata),):
        raise ValueError("phase_safe must match candidate_metadata")
    receding_mask = None if receding is None else np.asarray(receding, dtype=bool)
    if receding_mask is not None and receding_mask.shape != out.shape:
        raise ValueError("receding must match candidate_metadata")
    values = None if prefixes is None else np.asarray(prefixes, dtype=np.float32)
    for index, metadata in enumerate(candidate_metadata):
        source = str(metadata.get("source", ""))
        if not source.startswith("v11_"):
            continue
        variant = str(metadata.get("variant") or "")
        if variant.startswith("hold_filled") and not hold_filled_actor_allowed(
            allow_close=allow_close,
            current_gripper=current_gripper,
        ):
            out[index] = False
            continue
        if receding_mask is not None and bool(receding_mask[index]):
            continue
        if values is not None and not allow_close:
            if float(values[index, -1, -1] - values[index, 0, -1]) < -0.5:
                continue
        if (
            values is not None
            and empty_bite_reapproach
            and float(values[index, -1, -1]) < JAWS_OPEN_MIN_DEG
        ):
            continue
        out[index] = True
    return out


def candidate_phase_mask(
    prefixes: np.ndarray,
    phase: ManipulationPhase,
    thresholds: PhaseThresholds,
    *,
    current_gripper: float | None = None,
    recovery_below: float | None = None,
    empty_bite_reapproach: bool = False,
    allow_close: bool | None = None,
    receding: np.ndarray | None = None,
) -> np.ndarray:
    """Return candidates whose gripper evolution matches the active phase."""

    values = np.asarray(prefixes, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 6 or values.shape[1] == 0:
        raise ValueError("prefixes must have shape [K,R,6]")
    if not np.all(np.isfinite(values)):
        raise ValueError("prefixes must be finite")
    initial = values[:, 0, -1]
    terminal = values[:, -1, -1]
    minimum = np.min(values[:, :, -1], axis=1)
    maximum = np.max(values[:, :, -1], axis=1)
    delta = terminal - initial
    if allow_close is None:
        allow_close = phase == ManipulationPhase.GRASP
    receding_mask = None if receding is None else np.asarray(receding, dtype=bool)
    if receding_mask is not None and receding_mask.shape != (len(values),):
        raise ValueError("receding must match prefixes")

    if phase == ManipulationPhase.APPROACH:
        # Never begin a close until the configured pickup is metrically near,
        # including after an empty bite. Re-aim first. After an empty bite
        # the prefix must actually open, not sit on the closed stop.
        jaw_ok = (delta >= -0.5) | bool(allow_close)
        if empty_bite_reapproach:
            jaw_ok = jaw_ok & (terminal >= np.float32(JAWS_OPEN_MIN_DEG))
        if receding_mask is None:
            return jaw_ok
        return jaw_ok & ~receding_mask
    if phase == ManipulationPhase.GRASP:
        # Close only when held or metrically near, and never while the first
        # chunk increases range to the current pickup.
        jaw_ok = (delta >= -0.5) | bool(allow_close)
        if empty_bite_reapproach:
            jaw_ok = jaw_ok & (terminal >= np.float32(JAWS_OPEN_MIN_DEG))
        if receding_mask is None:
            return jaw_ok
        return jaw_ok & ~(receding_mask & (delta < -0.5))
    if phase == ManipulationPhase.TRANSPORT:
        return (maximum <= thresholds.gripper_open) & (np.abs(delta) <= 5.0)
    if phase == ManipulationPhase.RELEASE:
        return (maximum >= thresholds.gripper_open) & (delta >= 0.5)
    return np.zeros(len(values), dtype=bool)


def phase_prediction_bounds_mask(
    predicted_states: np.ndarray,
    phase: ManipulationPhase,
    *,
    margin: float,
) -> np.ndarray:
    """Check LeWM forecasts without conflating approach-arm and gripper safety.

    During approach, the absolute command prefix already checks all six hard
    limits and the gripper is not the controlled motion. A small forecast error
    in the LeWM gripper head must not reject an otherwise bounded arm path.
    """

    values = np.asarray(predicted_states, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 6:
        raise ValueError("predicted_states must have shape [K,H,6]")
    if not np.all(np.isfinite(values)):
        return np.zeros(len(values), dtype=bool)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("prediction bound margin must be finite and non-negative")
    dimensions = 5 if phase == ManipulationPhase.APPROACH else 6
    lower = SO101_POSITION_LOWER[:dimensions] - margin
    upper = SO101_POSITION_UPPER[:dimensions] + margin
    checked = values[..., :dimensions]
    return np.all(
        (checked >= lower[None, None]) & (checked <= upper[None, None]),
        axis=(1, 2),
    )


@dataclass(frozen=True)
class CommandChunkCheck:
    safe: bool
    reasons: tuple[str, ...]
    first_jump: float
    maximum_step_jump: float


def validate_command_chunk(
    chunk: np.ndarray,
    current_state: np.ndarray,
    *,
    expected_commands: int = 5,
    max_first_jump: float = 8.0,
    max_step_jump: float = 8.0,
    limit_guard: float = 3.0,
    outward_tolerance: float = 0.0,
) -> CommandChunkCheck:
    values = np.asarray(chunk, dtype=np.float32)
    state = np.asarray(current_state, dtype=np.float32)
    if values.shape != (expected_commands, 6):
        raise ValueError(f"command chunk must have shape [{expected_commands},6]")
    if state.shape != (6,):
        raise ValueError("current_state must have shape [6]")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(state)):
        raise ValueError("command chunk and current state must be finite")
    if min(max_first_jump, max_step_jump, limit_guard) <= 0:
        raise ValueError("command safety limits must be positive")
    if not np.isfinite(outward_tolerance) or outward_tolerance < 0:
        raise ValueError("outward_tolerance must be finite and non-negative")

    previous = np.concatenate((state[None], values[:-1]), axis=0)
    jumps = np.max(np.abs(values - previous), axis=1)
    first_jump = float(jumps[0])
    maximum_step = float(np.max(jumps))
    reasons: list[str] = []
    if np.any(values < SO101_POSITION_LOWER) or np.any(values > SO101_POSITION_UPPER):
        reasons.append("hard_position_bounds")
    if first_jump > max_first_jump:
        reasons.append("first_command_jump")
    if maximum_step > max_step_jump:
        reasons.append("prefix_step_jump")

    lower_near = state <= SO101_POSITION_LOWER + limit_guard
    upper_near = state >= SO101_POSITION_UPPER - limit_guard
    first_delta = values[0] - state
    if np.any(lower_near & (first_delta < -outward_tolerance)) or np.any(
        upper_near & (first_delta > outward_tolerance)
    ):
        reasons.append("moving_outward_at_joint_limit")
    return CommandChunkCheck(
        safe=not reasons,
        reasons=tuple(reasons),
        first_jump=first_jump,
        maximum_step_jump=maximum_step,
    )


@dataclass(frozen=True)
class PlanningRoundGuard:
    round_id: str
    created_monotonic_s: float
    state: np.ndarray
    target_px: np.ndarray
    target_kind: str

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float32)
        target = np.asarray(self.target_px, dtype=np.float32)
        if not self.round_id:
            raise ValueError("round_id cannot be empty")
        if not np.isfinite(self.created_monotonic_s):
            raise ValueError("created_monotonic_s must be finite")
        if state.shape != (6,) or target.shape != (2,):
            raise ValueError("round guard state/target shapes are invalid")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(target)):
            raise ValueError("round guard state/target must be finite")

    def validate_live(
        self,
        *,
        round_id: str,
        now_monotonic_s: float,
        live_state: np.ndarray,
        live_target_px: np.ndarray | None,
        max_age_s: float,
        max_state_drift: float,
        max_target_drift_px: float,
    ) -> dict[str, float | None]:
        if round_id != self.round_id:
            raise RuntimeError("stale planning round")
        age = float(now_monotonic_s - self.created_monotonic_s)
        if age < 0 or (max_age_s > 0 and age > max_age_s):
            raise RuntimeError(f"planning round age {age:.3f}s exceeds limit")
        state = np.asarray(live_state, dtype=np.float32)
        if state.shape != (6,):
            raise ValueError("live state/target shapes are invalid")
        state_drift = float(np.max(np.abs(state - np.asarray(self.state))))
        if state_drift > max_state_drift:
            raise RuntimeError(
                f"live state drift {state_drift:.3f} exceeds {max_state_drift:.3f}"
            )
        if live_target_px is None:
            # No live target measurement exists for this plan; the caller must
            # record an explicit skip rather than compare the plan's own
            # target to itself.
            return {
                "age_s": age,
                "state_drift": state_drift,
                "target_drift_px": None,
            }
        target = np.asarray(live_target_px, dtype=np.float32)
        if target.shape != (2,):
            raise ValueError("live state/target shapes are invalid")
        target_drift = float(np.linalg.norm(target - np.asarray(self.target_px)))
        if target_drift > max_target_drift_px:
            raise RuntimeError(
                f"live target drift {target_drift:.3f}px exceeds "
                f"{max_target_drift_px:.3f}px"
            )
        return {
            "age_s": age,
            "state_drift": state_drift,
            "target_drift_px": target_drift,
        }


__all__ = [
    "CommandChunkCheck",
    "HOLD_CONFIRM_MAX_JAWS_DEG",
    "JAWS_OPEN_MIN_DEG",
    "JAW_COLLAPSED_TIP_SEPARATION_PX",
    "ManipulationPhase",
    "PhaseEvidence",
    "PhaseThresholds",
    "PhaseTracker",
    "PlanningRoundGuard",
    "THREE_POINT_VERDICTS",
    "candidate_phase_mask",
    "GRASP_EMPTY_AIR_GAP_DEG",
    "close_allowed_for_pickup",
    "empty_bite_reapproach_open_targets",
    "exclusive_phase_gripper_open",
    "grasp_aperture_candidates",
    "grasp_closed_floor",
    "hold_filled_actor_allowed",
    "jaws_are_pinched",
    "jaws_in_holding_aperture",
    "pickup_is_metrically_near",
    "predicted_closure_cost",
    "judge_three_point_closure",
    "keep_v11_candidates_phase_safe",
    "phase_prediction_bounds_mask",
    "target_kind_for_phase",
    "target_role_for_phase",
    "validate_command_chunk",
]
