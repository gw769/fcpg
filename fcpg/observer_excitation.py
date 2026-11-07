"""D-optimal micro-excitation during APPROACH (milestone M6).

A constant-parameter observer converges only as fast as the arm's motion
spans the information Gramian.  Task motion is whatever the pickup required;
it is not designed to light up the weakly observed directions (typically a
rotation about the optical axis, or ``t_z`` once ``f`` is held fixed).

This module picks a *small* joint-space step that maximises the predicted
information gain -- D-optimality, ``argmax det(Y + H^T R^{-1} H)`` -- and
returns it only in APPROACH, only when the jaws are empty.  A held object
turns the gripper into a moving landmark whose FK is no longer the FK of
the empty jaw; exciting then would identify a different plant.

The candidates are a fixed stencil of millimetre-scale joint nudges, not a
continuous optimiser.  A continuous search would need a model of collision
and of the object, which this module does not have; the stencil is small
enough that the existing supervisor still has the last word.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fcpg.observer_filter import InformationObserver, measurement_jacobian


__all__ = [
    "APPROACH_PHASES",
    "HOLDING_PHASES",
    "DEFAULT_STENCIL",
    "ExcitationDecision",
    "d_optimal_excitation",
]


APPROACH_PHASES = frozenset({"approach", "APPROACH"})
HOLDING_PHASES = frozenset(
    {"grasp", "GRASP", "transport", "TRANSPORT", "place", "PLACE", "release", "RELEASE"}
)

#: Normalized-joint stencil.  ±4 units is a few degrees on the arm joints
#: and is inside the live max-step clamp; the jaw is never moved here.
DEFAULT_STENCIL = (
    (4.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (-4.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 4.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, -4.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 4.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, -4.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 4.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, -4.0, 0.0, 0.0),
    (2.0, 2.0, -2.0, 0.0, 0.0, 0.0),
    (-2.0, 2.0, 2.0, 0.0, 0.0, 0.0),
)


@dataclass(frozen=True)
class ExcitationDecision:
    allowed: bool
    delta_q: np.ndarray | None
    predicted_logdet: float | None
    reason: str


def _logdet(matrix: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0.0:
        return float("-inf")
    return float(logdet)


def d_optimal_excitation(
    observer: InformationObserver,
    points_base_xyz: np.ndarray,
    joint_jacobian: np.ndarray,
    *,
    phase: str | None,
    jaws_holding: bool,
    stencil: Sequence[Sequence[float]] = DEFAULT_STENCIL,
    current_information: np.ndarray | None = None,
) -> ExcitationDecision:
    """Pick the stencil step with the largest predicted ``log det Y``.

    ``points_base_xyz`` / ``joint_jacobian`` are evaluated at the *current*
    configuration; the stencil is small enough that the same linearization
    is used for every candidate.  That is an approximation, and it is why
    the stencil stays small.
    """

    if phase in HOLDING_PHASES or jaws_holding:
        return ExcitationDecision(
            allowed=False,
            delta_q=None,
            predicted_logdet=None,
            reason="holding",
        )
    if phase not in APPROACH_PHASES:
        return ExcitationDecision(
            allowed=False,
            delta_q=None,
            predicted_logdet=None,
            reason="not_approach",
        )

    information = (
        observer.state.information
        if current_information is None
        else np.asarray(current_information, dtype=np.float64)
    )
    baseline = _logdet(information)
    weight = 1.0 / max(observer.sigma_px**2, 1e-12)
    best_delta = None
    best_logdet = baseline
    for raw in stencil:
        delta = np.asarray(raw, dtype=np.float64).reshape(6)
        # Predicted Jacobian at q+delta ≈ J(q) because the stencil is small;
        # the information gain then comes from the *change in predicted
        # pixels*, which we approximate by moving the 3-D points along J.
        shifted = points_base_xyz + joint_jacobian @ delta[:5]
        working = observer.state
        jacobian, _ = measurement_jacobian(working, shifted, joint_jacobian)
        predicted = information + weight * (jacobian.T @ jacobian)
        score = _logdet(predicted)
        if score > best_logdet:
            best_logdet = score
            best_delta = delta

    if best_delta is None:
        return ExcitationDecision(
            allowed=False,
            delta_q=None,
            predicted_logdet=baseline,
            reason="no_gain",
        )
    return ExcitationDecision(
        allowed=True,
        delta_q=best_delta,
        predicted_logdet=best_logdet,
        reason="d_optimal",
    )
