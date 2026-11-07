"""Grounding review: decide whether a proposed action is supported by evidence.

Implements the loop that replaces an unconditional stop:

    planner proposes action
      -> verifier reviews grounding evidence
      -> PROCEED | CORRECT_TARGET | REQUEST_OBSERVATION | HALT
      -> planner acts, retargets, satisfies the observation request, or stops

The observation trigger is derived from geometry, not tuned.  An action carries
the positional slack it can absorb (``ProposedAction.tolerance_m``); an
observation is requested when the safety-scaled uncertainty exceeds that slack:

    safety_factor * position_sigma_m > tolerance_m

So the same code adapts to a wide-jaw grasp and a tight insertion without any
threshold being retuned, because tolerance comes from the task's own geometry.

Fail-closed rules:
  * an observation is requested only if one could plausibly help — a sensor must
    be available AND expected to reduce sigma below tolerance
  * if uncertainty exceeds tolerance and no observation would help, HALT
  * a repeated request for the same sensor escalates to HALT, so the loop
    cannot cycle forever
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from interfaces.grounding import (
    GroundingVerifierInterface,
    UncertaintyEstimatorInterface,
    ProposedAction,
    GroundingFeedback,
    GroundingVerdict,
    ObservationRequest,
)


def grasp_tolerance_m(
    aperture_m: float,
    object_width_m: float,
    pad_margin_m: float,
) -> float:
    """Lateral slack a grasp can absorb, from jaw geometry.

    Half the free space left in the jaw once the object and a pad margin are
    accounted for.  Provides ProposedAction.tolerance_m from measured geometry
    instead of a tuned constant.

    Raises:
        ValueError: if the object cannot fit the aperture with the margin.
    """
    for name, value in (
        ("aperture_m", aperture_m),
        ("object_width_m", object_width_m),
        ("pad_margin_m", pad_margin_m),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    slack = 0.5 * (float(aperture_m) - float(object_width_m)) - float(pad_margin_m)
    if slack <= 0.0:
        raise ValueError(
            "no grasp slack: object width plus pad margin exceeds the aperture"
        )
    return slack


@dataclass(frozen=True)
class SensorCapability:
    """What a sensor can contribute, for deciding if observing is worthwhile."""
    role: str
    achievable_sigma_m: float
    max_range_m: float | None = None


@dataclass(frozen=True)
class VerifierConfig:
    safety_factor: float
    """Multiplier on sigma before comparison against tolerance.  >1 demands
    headroom; supplied by config, not assumed."""
    max_observation_requests: int = 1
    """Requests permitted per sensor before escalating to HALT, bounding the
    loop."""
    max_correction_m: float | None = None
    """Largest target correction accepted.  A larger disagreement means the
    evidence contradicts the plan, which is a HALT, not a nudge."""

    def __post_init__(self) -> None:
        if not np.isfinite(self.safety_factor) or self.safety_factor <= 0:
            raise ValueError("safety_factor must be finite and positive")
        if self.max_observation_requests < 0:
            raise ValueError("max_observation_requests must be non-negative")


class GroundingVerifier(GroundingVerifierInterface):
    """Review proposed actions and request observations when warranted."""

    def __init__(
        self,
        estimator: UncertaintyEstimatorInterface,
        config: VerifierConfig,
        sensors: dict[str, SensorCapability] | None = None,
    ) -> None:
        self._estimator = estimator
        self._config = config
        self._sensors = dict(sensors or {})
        self._request_counts: dict[str, int] = {}

    def reset(self) -> None:
        """Clear observation-request history, e.g. for a new trial."""
        self._request_counts.clear()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _best_sensor(self, tolerance_m: float) -> SensorCapability | None:
        """Cheapest-sigma sensor expected to bring uncertainty under tolerance."""
        viable = [
            s for s in self._sensors.values()
            if self._config.safety_factor * s.achievable_sigma_m <= tolerance_m
            and self._request_counts.get(s.role, 0) < self._config.max_observation_requests
        ]
        if not viable:
            return None
        return min(viable, key=lambda s: s.achievable_sigma_m)

    # ── interface ────────────────────────────────────────────────────────────

    def review(
        self,
        action: ProposedAction,
        evidence: dict[str, Any],
    ) -> GroundingFeedback:
        uncertainty = self._estimator.estimate(evidence)
        sigma = uncertainty.position_sigma_m
        tolerance = action.tolerance_m
        effective = self._config.safety_factor * sigma

        checks: dict[str, Any] = {
            "position_sigma_m": round(sigma, 6),
            "tolerance_m": round(tolerance, 6),
            "safety_factor": self._config.safety_factor,
            "effective_sigma_m": round(effective, 6),
            "uncertainty_calibrated": uncertainty.calibration.is_calibrated,
        }

        # Contradictory evidence outranks uncertainty: a target that disagrees
        # with the plan beyond the accepted bound is not a small correction.
        observed = evidence.get("observed_target_xyz")
        correction_m: float | None = None
        if observed is not None:
            observed_xyz = np.asarray(observed, dtype=np.float64)
            proposed_xyz = np.asarray(action.target_world_xyz, dtype=np.float64)
            if observed_xyz.shape == proposed_xyz.shape:
                correction_m = float(np.linalg.norm(observed_xyz - proposed_xyz))
                checks["correction_m"] = round(correction_m, 6)
                bound = self._config.max_correction_m
                if bound is not None and correction_m > bound:
                    return GroundingFeedback(
                        verdict=GroundingVerdict.HALT,
                        uncertainty=uncertainty,
                        reason=(
                            f"observed target disagrees with proposal by "
                            f"{correction_m:.4f} m, exceeding the accepted "
                            f"correction bound {bound:.4f} m"
                        ),
                        checks=checks,
                    )

        if effective > tolerance:
            sensor = self._best_sensor(tolerance)
            if sensor is None:
                return GroundingFeedback(
                    verdict=GroundingVerdict.HALT,
                    uncertainty=uncertainty,
                    reason=(
                        f"effective uncertainty {effective:.4f} m exceeds tolerance "
                        f"{tolerance:.4f} m and no further observation is expected "
                        f"to close the gap"
                    ),
                    checks=checks,
                )
            self._request_counts[sensor.role] = (
                self._request_counts.get(sensor.role, 0) + 1
            )
            checks["requests_for_sensor"] = self._request_counts[sensor.role]
            return GroundingFeedback(
                verdict=GroundingVerdict.REQUEST_OBSERVATION,
                uncertainty=uncertainty,
                observation_request=ObservationRequest(
                    sensor_role=sensor.role,
                    reason=(
                        f"effective uncertainty {effective:.4f} m exceeds tolerance "
                        f"{tolerance:.4f} m"
                    ),
                    max_range_m=sensor.max_range_m,
                    expected_sigma_reduction_m=max(
                        0.0, sigma - sensor.achievable_sigma_m
                    ),
                ),
                reason="additional observation required before acting",
                checks=checks,
            )

        if correction_m is not None and correction_m > 0.0:
            return GroundingFeedback(
                verdict=GroundingVerdict.CORRECT_TARGET,
                uncertainty=uncertainty,
                corrected_target_xyz=np.asarray(observed, dtype=np.float64).copy(),
                reason=f"target corrected by {correction_m:.4f} m within tolerance",
                checks=checks,
            )

        return GroundingFeedback(
            verdict=GroundingVerdict.PROCEED,
            uncertainty=uncertainty,
            checks=checks,
        )
