"""Autonomous near-field target registration — fail-closed.

Supplies the stage that turns a blocking safety gate into a recoverable one:

    APPROACH
      -> NEAR_FIELD_OBSERVATION    capture from the near-field sensor
      -> NEAR_FIELD_DETECTION      detector proposes candidates (no prior)
      -> NEAR_FIELD_REGISTRATION   verify, then commit a reference
      -> APPROACH_RESUME

Every failure mode stops motion and reports which check failed.  There is no
degradation to an external agent: an unverifiable target is a halt, by design.
A caller that wants to proceed must fix perception, not bypass it.

This module holds no robot, sensor, or task knowledge.  It composes the
detector / registration interfaces and reads all thresholds from config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from interfaces.hardware import Frame
from interfaces.perception import (
    TargetDetectorInterface,
    TargetRegistrationInterface,
    DetectionCandidate,
    AppearanceDescriptor,
)


class RegistrationOutcome(str, Enum):
    REGISTERED           = "registered"
    NO_CANDIDATE         = "no_candidate"
    AMBIGUOUS_CANDIDATES = "ambiguous_candidates"
    SANITY_CHECK_FAILED  = "sanity_check_failed"
    ALREADY_REGISTERED   = "already_registered"
    WITHHELD             = "withheld"
    """Every check passed but the reference was deliberately not committed.

    Used by measurement runs that need to know whether registration WOULD have
    succeeded at a given pose without changing what the next pose observes.
    Deliberately not ``ok``: nothing was registered, so a caller that treated
    this as success would proceed on a reference that does not exist.
    """


@dataclass(frozen=True)
class RegistrationPolicyConfig:
    """Verification thresholds — all supplied by config, none hardcoded."""
    min_area_px: int
    max_area_px: int | None = None
    min_solidity: float | None = None
    # A second candidate within this fraction of the best score makes the
    # scene ambiguous, so registration refuses to guess.
    ambiguity_score_ratio: float = 0.5
    # Lower bound on descriptor correlation against a cross-view reference.
    # Deliberately loose: cross-view lighting and scale differ substantially.
    min_descriptor_correlation: float | None = None
    # Candidate must lie inside the frame inset by this margin.
    frame_margin_px: int = 0
    registration_radius_px: float = 28.0


@dataclass(frozen=True)
class RegistrationResult:
    outcome: RegistrationOutcome
    target_id: str | None = None
    centroid_px: np.ndarray | None = None
    reason: str | None = None
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.outcome in (
            RegistrationOutcome.REGISTERED,
            RegistrationOutcome.ALREADY_REGISTERED,
        )


def descriptor_correlation(
    a: AppearanceDescriptor, b: AppearanceDescriptor
) -> float:
    """Pearson correlation between two descriptors; 0.0 if undefined."""
    if a.histogram.shape != b.histogram.shape:
        raise ValueError("descriptors must share a layout to be comparable")
    x = np.asarray(a.histogram, dtype=np.float64)
    y = np.asarray(b.histogram, dtype=np.float64)
    if x.std() == 0.0 or y.std() == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


class AutonomousRegistrationPolicy:
    """Detect, verify, and register a near-field reference without an operator."""

    def __init__(
        self,
        detector: TargetDetectorInterface,
        registration: TargetRegistrationInterface,
        config: RegistrationPolicyConfig,
    ) -> None:
        self._detector = detector
        self._registration = registration
        self._config = config

    # ── verification ─────────────────────────────────────────────────────────

    def _sanity_check(
        self,
        candidate: DetectionCandidate,
        frame: Frame,
        reference_descriptor: AppearanceDescriptor | None,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        cfg = self._config
        checks: dict[str, Any] = {
            "area_px": candidate.area_px,
            "solidity": round(candidate.solidity, 4),
        }

        if candidate.area_px < cfg.min_area_px:
            return False, f"area {candidate.area_px} below min {cfg.min_area_px}", checks
        if cfg.max_area_px is not None and candidate.area_px > cfg.max_area_px:
            return False, f"area {candidate.area_px} above max {cfg.max_area_px}", checks
        if cfg.min_solidity is not None and candidate.solidity < cfg.min_solidity:
            return (
                False,
                f"solidity {candidate.solidity:.3f} below min {cfg.min_solidity}",
                checks,
            )

        # ROI sanity: centroid must sit inside the frame, honouring the margin.
        u, v = float(candidate.centroid_px[0]), float(candidate.centroid_px[1])
        m = cfg.frame_margin_px
        checks["centroid_px"] = [round(u, 2), round(v, 2)]
        if not (m <= u < frame.width - m and m <= v < frame.height - m):
            return False, f"centroid ({u:.1f},{v:.1f}) outside usable frame", checks

        if cfg.min_descriptor_correlation is not None:
            if reference_descriptor is None or candidate.descriptor is None:
                return False, "descriptor comparison required but unavailable", checks
            corr = descriptor_correlation(candidate.descriptor, reference_descriptor)
            checks["descriptor_correlation"] = round(corr, 4)
            if corr < cfg.min_descriptor_correlation:
                return (
                    False,
                    f"descriptor correlation {corr:.3f} below "
                    f"min {cfg.min_descriptor_correlation}",
                    checks,
                )
        return True, None, checks

    # ── main entry point ─────────────────────────────────────────────────────

    def acquire(
        self,
        frame: Frame,
        role: str,
        view: str,
        *,
        reference_descriptor: AppearanceDescriptor | None = None,
    ) -> RegistrationResult:
        """Run OBSERVATION -> DETECTION -> REGISTRATION for one role/view.

        Returns a RegistrationResult; callers must treat any non-ok outcome as
        a halt condition and must not fall back to an external registration
        path.
        """
        if self._registration.is_registered(role, view):
            return RegistrationResult(
                outcome=RegistrationOutcome.ALREADY_REGISTERED,
                reason="reference already exists",
            )

        candidates = self._detector.detect(frame)

        if not candidates:
            return RegistrationResult(
                outcome=RegistrationOutcome.NO_CANDIDATE,
                reason="detector returned no candidate in the near-field view",
                checks={"n_candidates": 0},
            )

        best = candidates[0]

        if len(candidates) > 1:
            runner_up = candidates[1]
            ratio = (
                runner_up.score / best.score if best.score > 0 else 1.0
            )
            if ratio >= self._config.ambiguity_score_ratio:
                return RegistrationResult(
                    outcome=RegistrationOutcome.AMBIGUOUS_CANDIDATES,
                    reason=(
                        f"{len(candidates)} candidates; runner-up score ratio "
                        f"{ratio:.3f} >= {self._config.ambiguity_score_ratio}"
                    ),
                    checks={
                        "n_candidates": len(candidates),
                        "best_score": best.score,
                        "runner_up_score": runner_up.score,
                        "score_ratio": round(ratio, 4),
                    },
                )

        passed, reason, checks = self._sanity_check(best, frame, reference_descriptor)
        checks["n_candidates"] = len(candidates)
        if not passed:
            return RegistrationResult(
                outcome=RegistrationOutcome.SANITY_CHECK_FAILED,
                reason=reason,
                checks=checks,
            )

        target_id = self._registration.register(
            frame,
            best.centroid_px,
            role,
            view,
            radius_px=self._config.registration_radius_px,
        )
        return RegistrationResult(
            outcome=RegistrationOutcome.REGISTERED,
            target_id=target_id,
            centroid_px=np.asarray(best.centroid_px, dtype=np.float64).copy(),
            checks=checks,
        )
