"""Grounding feedback protocol between a task planner and a grounding controller.

Defines the contract that lets a planner discover when its own grounding is
too uncertain to act on, and lets the grounding controller ask for the
observation it needs instead of simply refusing to continue.

The distinction from an ordinary perception pipeline is directional.  A planner
does not merely consume a target; it submits a *proposed action* and receives a
verdict about whether grounding supports it.  When it does not, the controller
returns an ObservationRequest describing the information it lacks, expressed as
a sensing goal rather than a motion command, so the planner remains the only
component that decides how to move.

Neither side imports the other.  The planner sees GroundingFeedback; the
controller sees ProposedAction.  Both are plain data.

Nothing here assumes a robot, sensor, model, or task.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


# ── verdicts ──────────────────────────────────────────────────────────────────

class GroundingVerdict(str, Enum):
    """What the grounding controller concludes about a proposed action."""

    PROCEED = "proceed"
    """Grounding supports the action within the task's positional tolerance."""

    CORRECT_TARGET = "correct_target"
    """Grounding is adequate but disagrees with the planner's target; a
    corrected target accompanies the feedback."""

    REQUEST_OBSERVATION = "request_observation"
    """Uncertainty exceeds tolerance, and an observation could reduce it.  The
    accompanying ObservationRequest states what is needed.  This is an
    information request, not a failure."""

    HALT = "halt"
    """Grounding cannot support the action and no observation would fix it
    (contradictory evidence, ambiguity, or a failed verification).  Fail-closed:
    the caller must stop rather than degrade to an external agent."""


# ── uncertainty ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UncertaintyCalibration:
    """Provenance for an uncertainty estimate.

    An uncertainty value is only meaningful if reported magnitudes match
    observed error.  This record travels with every estimate so a consumer can
    tell a validated number from a provisional one.  It exists to stop an
    uncalibrated estimate from being read as if it were validated.
    """
    is_calibrated: bool
    n_validation_samples: int = 0
    notes: str | None = None


@dataclass(frozen=True)
class GroundingUncertainty:
    """Decomposed uncertainty for one grounded target.

    Components are dimensionless in [0, 1] and name *sources* of doubt, so a
    consumer can distinguish "I see two plausible targets" from "I see one but
    it looks wrong".  ``position_sigma_m`` is the aggregate expressed in metres,
    which is the only component comparable against a physical tolerance.
    """
    position_sigma_m: float
    components: dict[str, float] = field(default_factory=dict)
    calibration: UncertaintyCalibration = field(
        default_factory=lambda: UncertaintyCalibration(is_calibrated=False)
    )

    def __post_init__(self) -> None:
        if not np.isfinite(self.position_sigma_m) or self.position_sigma_m < 0:
            raise ValueError("position_sigma_m must be finite and non-negative")
        for name, value in self.components.items():
            if not np.isfinite(value) or not (0.0 <= value <= 1.0):
                raise ValueError(f"uncertainty component {name!r} must lie in [0, 1]")


# ── requests and feedback ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ObservationRequest:
    """A sensing goal the planner is asked to satisfy.

    Deliberately expressed as a constraint on *sensing*, not as a motion
    command: the planner owns motion.  ``max_range_m`` is the distance from
    which the requested sensor can resolve the target, so the planner may reach
    it however it prefers.
    """
    sensor_role: str
    reason: str
    max_range_m: float | None = None
    expected_sigma_reduction_m: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProposedAction:
    """A planner's intent, submitted for grounding review before execution."""
    target_world_xyz: np.ndarray
    tolerance_m: float
    """Positional slack the action can absorb and still succeed.  Derived from
    task geometry by the caller — never a tuned constant."""
    phase: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.tolerance_m) or self.tolerance_m <= 0:
            raise ValueError("tolerance_m must be finite and positive")


@dataclass(frozen=True)
class GroundingFeedback:
    """The grounding controller's reply to a ProposedAction."""
    verdict: GroundingVerdict
    uncertainty: GroundingUncertainty
    corrected_target_xyz: np.ndarray | None = None
    observation_request: ObservationRequest | None = None
    reason: str | None = None
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def actionable(self) -> bool:
        """True when the planner may execute (possibly with a corrected target)."""
        return self.verdict in (
            GroundingVerdict.PROCEED,
            GroundingVerdict.CORRECT_TARGET,
        )

    def __post_init__(self) -> None:
        if (
            self.verdict is GroundingVerdict.CORRECT_TARGET
            and self.corrected_target_xyz is None
        ):
            raise ValueError("CORRECT_TARGET requires corrected_target_xyz")
        if (
            self.verdict is GroundingVerdict.REQUEST_OBSERVATION
            and self.observation_request is None
        ):
            raise ValueError("REQUEST_OBSERVATION requires an observation_request")


# ── interfaces ────────────────────────────────────────────────────────────────

class UncertaintyEstimatorInterface(ABC):
    """Turn perception evidence into a positional uncertainty estimate."""

    @abstractmethod
    def estimate(self, evidence: dict[str, Any]) -> GroundingUncertainty:
        """Return uncertainty for the grounded target described by ``evidence``.

        Implementations must report their calibration status honestly via
        GroundingUncertainty.calibration.
        """


class GroundingVerifierInterface(ABC):
    """Review a proposed action against current grounding evidence."""

    @abstractmethod
    def review(
        self,
        action: ProposedAction,
        evidence: dict[str, Any],
    ) -> GroundingFeedback:
        """Return a verdict for the proposed action.

        Must be fail-closed: when grounding cannot be established and no
        observation would help, return HALT rather than a permissive verdict.
        """
