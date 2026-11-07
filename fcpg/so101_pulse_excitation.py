"""Free-space hop that makes a gripper open-close pulse independently visible.

Pulse correspondence needs a moving-tip track *inside the gripper role box*
that is not explained by any other role. When the jaws already sit on an
object, every moving blob belongs to that object and the pulse must refuse.
This module does not change those gates. It only computes a short, generic
lift so the same pulse can see the gripper move by itself.

This is not ``observer_excitation`` (D-optimal micro-nudge for the shadow
SE(3) filter) and it is not tip authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


HOP_FRACTION_OF_SWEEP = 0.7
HOP_FRACTION_FALLBACKS = (0.70, 0.50, 0.35)
PLANAR_FRACTION_OF_HOP = 0.30
MOVING_TIP_INDEX = 2
APPROACH_PHASES = frozenset({"approach", "APPROACH"})
#: Same quantum the jaw pulse already uses to declare "settled at target".
JAW_OPEN_TOLERANCE = 2.0


class _IKResult(Protocol):
    state: np.ndarray
    achieved_xyz: np.ndarray
    error_m: float
    converged: bool


class _Kinematics(Protocol):
    def forward(self, states: np.ndarray) -> np.ndarray: ...

    def forward_anchors(self, states: np.ndarray) -> np.ndarray: ...

    def inverse_position(
        self,
        target_xyz: np.ndarray,
        initial_state: np.ndarray,
        **kwargs: Any,
    ) -> _IKResult: ...


@dataclass(frozen=True)
class ExcitationPreflight:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class ExcitationPose:
    start_state: np.ndarray
    target_state: np.ndarray
    start_xyz: np.ndarray
    target_xyz: np.ndarray
    delta_xyz: np.ndarray
    sweep_radius_m: float
    hop_m: float
    hop_fraction: float


def fk_jaw_sweep_radius_m(
    kinematics: _Kinematics,
    arm_state: np.ndarray,
    *,
    open_gripper: float,
    closed_gripper: float,
) -> float:
    """FK moving-tip travel of one empty open/close at the current arm pose.

    This is the same length ``set_live_moving_tip_track`` later records as
    ``support_radius_m``. Hopping a fraction of it keeps the return pose
    inside the *new* pulse support ball.
    """

    arm = np.asarray(arm_state, dtype=np.float64)
    if arm.shape != (6,) or not np.all(np.isfinite(arm)):
        raise ValueError("arm_state must be a finite six-vector")
    opened = float(open_gripper)
    closed = float(closed_gripper)
    if not np.isfinite(opened) or not np.isfinite(closed):
        raise ValueError("gripper stops must be finite")
    pair = np.stack((arm.copy(), arm.copy()))
    pair[0, 5] = opened
    pair[1, 5] = closed
    anchors = np.asarray(kinematics.forward_anchors(pair), dtype=np.float64)
    if anchors.shape != (2, 3, 3) or not np.all(np.isfinite(anchors)):
        raise ValueError("forward_anchors must return finite [2,3,3]")
    radius = float(
        np.linalg.norm(anchors[0, MOVING_TIP_INDEX] - anchors[1, MOVING_TIP_INDEX])
    )
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("FK jaw sweep radius must be finite and positive")
    return radius


def excitation_hop_delta_xyz(
    sweep_radius_m: float,
    *,
    hop_fraction: float = HOP_FRACTION_OF_SWEEP,
    planar_xy: np.ndarray | None = None,
    planar_fraction: float = PLANAR_FRACTION_OF_HOP,
) -> np.ndarray:
    """World-frame hop. +Z off the table; optional small planar retreat.

    ``planar_xy`` is a motion hint (away from a locked pickup in the task
    plane), not tip identity. Missing or degenerate planar input yields +Z
    only. The returned vector always has length ``hop_fraction * R``.
    """

    radius = float(sweep_radius_m)
    fraction = float(hop_fraction)
    planar_share = float(planar_fraction)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("sweep_radius_m must be finite and positive")
    if not np.isfinite(fraction) or not 0.0 < fraction <= HOP_FRACTION_OF_SWEEP:
        raise ValueError(
            f"hop_fraction must be in (0, {HOP_FRACTION_OF_SWEEP}]"
        )
    if not np.isfinite(planar_share) or not 0.0 <= planar_share < 1.0:
        raise ValueError("planar_fraction must be in [0, 1)")
    hop = fraction * radius
    lift = np.asarray((0.0, 0.0, hop), dtype=np.float64)
    if planar_xy is None:
        return lift
    planar = np.asarray(planar_xy, dtype=np.float64).reshape(-1)
    if planar.shape != (2,) or not np.all(np.isfinite(planar)):
        return lift
    planar_norm = float(np.linalg.norm(planar))
    if planar_norm <= 1.0e-9:
        return lift
    unit = planar / planar_norm
    combined = np.asarray(
        (
            unit[0] * planar_share * hop,
            unit[1] * planar_share * hop,
            (1.0 - planar_share) * hop,
        ),
        dtype=np.float64,
    )
    scale = hop / float(np.linalg.norm(combined))
    return combined * scale


def excitation_preflight(
    *,
    phase: str | None,
    current_gripper: float,
    open_gripper: float,
    closed_gripper: float,
    stopped: bool = False,
    jaw_tolerance: float = JAW_OPEN_TOLERANCE,
) -> ExcitationPreflight:
    """Refuse before the arm moves. Does not inspect tracks or residuals."""

    if stopped:
        return ExcitationPreflight(False, "controller is stopped")
    phase_name = "" if phase is None else str(phase)
    if phase_name not in APPROACH_PHASES and phase_name.lower() not in {
        "approach"
    }:
        return ExcitationPreflight(
            False, "excitation is only allowed during APPROACH"
        )
    opened = float(open_gripper)
    closed = float(closed_gripper)
    current = float(current_gripper)
    tolerance = float(jaw_tolerance)
    if not np.all(np.isfinite((opened, closed, current, tolerance))):
        return ExcitationPreflight(False, "gripper stops are not finite")
    if tolerance <= 0.0:
        return ExcitationPreflight(False, "jaw_tolerance must be positive")
    midpoint = 0.5 * (opened + closed)
    if current < midpoint:
        return ExcitationPreflight(
            False,
            "jaws are not at the calibrated open stop; refuse excitation while possibly holding",
        )
    if abs(current - opened) > tolerance:
        return ExcitationPreflight(
            False, "jaws are not at the calibrated open stop"
        )
    return ExcitationPreflight(True, "ok")


def solve_excitation_pose(
    kinematics: _Kinematics,
    start_state: np.ndarray,
    *,
    open_gripper: float,
    closed_gripper: float,
    planar_xy: np.ndarray | None = None,
    hop_fractions: tuple[float, ...] = HOP_FRACTION_FALLBACKS,
    tolerance_m: float = 1.0e-3,
) -> ExcitationPose:
    """IK a +Z-dominant hop. First convergent fraction wins. Else refuse.

    The gripper joint is pinned at the calibrated open stop. No in-place
    fallback is offered here; the caller must not pulse at the start pose.
    """

    start = np.asarray(start_state, dtype=np.float32)
    if start.shape != (6,) or not np.all(np.isfinite(start)):
        raise ValueError("start_state must be a finite six-vector")
    opened = float(open_gripper)
    if not np.isfinite(opened):
        raise ValueError("open_gripper must be finite")
    pinned = start.copy()
    pinned[5] = np.float32(opened)
    radius = fk_jaw_sweep_radius_m(
        kinematics,
        pinned,
        open_gripper=opened,
        closed_gripper=closed_gripper,
    )
    start_xyz = np.asarray(kinematics.forward(pinned), dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(start_xyz)):
        raise ValueError("forward kinematics at the start pose is not finite")
    last_error = "excitation IK did not converge"
    for fraction in hop_fractions:
        delta = excitation_hop_delta_xyz(
            radius, hop_fraction=float(fraction), planar_xy=planar_xy
        )
        if float(np.linalg.norm(delta)) > HOP_FRACTION_OF_SWEEP * radius + 1.0e-9:
            raise RuntimeError("excitation hop exceeded 0.7 × FK jaw sweep")
        target_xyz = start_xyz + delta
        result = kinematics.inverse_position(target_xyz, pinned)
        error = float(result.error_m)
        last_error = f"excitation IK error {error:.4f} m at hop_fraction={fraction}"
        if bool(result.converged) and error <= float(tolerance_m):
            target_state = np.asarray(result.state, dtype=np.float32).copy()
            if target_state.shape != (6,) or not np.all(np.isfinite(target_state)):
                raise ValueError("inverse_position must return a finite six-vector")
            target_state[5] = np.float32(opened)
            return ExcitationPose(
                start_state=pinned,
                target_state=target_state,
                start_xyz=start_xyz,
                target_xyz=np.asarray(result.achieved_xyz, dtype=np.float64).reshape(3),
                delta_xyz=delta,
                sweep_radius_m=radius,
                hop_m=float(np.linalg.norm(delta)),
                hop_fraction=float(fraction),
            )
    raise ValueError(last_error)
