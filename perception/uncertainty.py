"""Uncertainty estimation from grounding evidence.

Aggregates independent doubt sources into a single positional sigma comparable
against a physical tolerance.  Every component is derived from a signal the
perception stack already produces, so no source is invented for the sake of a
tidy number:

  candidate_ambiguity     runner-up / best detector score
  appearance_inconsistency  1 - cross-view descriptor correlation
  projection_residual     calibration reprojection error, px -> m
  multiview_disagreement  distance between two views' estimates of one point
  temporal_jitter         spread of the estimate across recent frames

CALIBRATION STATUS
------------------
The combiner below is a documented modelling choice, NOT a validated error
model.  It reports is_calibrated=False until reported sigma has been checked
against measured grounding error over a sample of poses.  Reading its output as
a calibrated confidence would overstate what has been established.  Calibrating
it requires ground-truth target positions across many poses; that dataset does
not exist yet.

Pixel-to-metre conversion uses the calibration Jacobian's SMALLEST singular
value, so a pixel error is converted along the least-observable direction.  That
is the conservative choice: it cannot understate positional error.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from interfaces.grounding import (
    UncertaintyEstimatorInterface,
    GroundingUncertainty,
    UncertaintyCalibration,
)


def pixel_to_metre_scale(projection_coefficients: np.ndarray) -> float:
    """Metres of positional error per pixel, along the least-observable axis.

    Args:
        projection_coefficients: (3, 2) spatial block of a linear
            world->pixel fit, rows ordered (x, y, z).

    Returns:
        Metres per pixel using the smallest singular value of the Jacobian.
    """
    jacobian = np.asarray(projection_coefficients, dtype=np.float64)
    if jacobian.shape != (3, 2):
        raise ValueError("projection_coefficients must have shape (3, 2)")
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("projection_coefficients must be finite")
    singular = np.linalg.svd(jacobian.T, compute_uv=False)
    smallest = float(singular.min())
    if smallest <= 0.0:
        raise ValueError("degenerate projection: cannot convert pixels to metres")
    return 1.0 / smallest


class EvidenceUncertaintyEstimator(UncertaintyEstimatorInterface):
    """Combine evidence signals into a positional sigma.

    Config
    ------
    baseline_sigma_m      floor sigma from calibration residual alone
    component_weights     per-component multiplier, in metres, applied to the
                          dimensionless component value
    """

    def __init__(
        self,
        baseline_sigma_m: float,
        component_weights: dict[str, float],
    ) -> None:
        if not np.isfinite(baseline_sigma_m) or baseline_sigma_m < 0:
            raise ValueError("baseline_sigma_m must be finite and non-negative")
        if not component_weights:
            raise ValueError("at least one component weight is required")
        for name, weight in component_weights.items():
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(f"weight for {name!r} must be finite and non-negative")
        self._baseline = float(baseline_sigma_m)
        self._weights = dict(component_weights)

    # ── component extraction ─────────────────────────────────────────────────

    @staticmethod
    def _components(evidence: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}

        best = evidence.get("best_score")
        runner_up = evidence.get("runner_up_score")
        if best is not None and runner_up is not None and float(best) > 0:
            out["candidate_ambiguity"] = float(
                np.clip(float(runner_up) / float(best), 0.0, 1.0)
            )
        elif best is not None:
            out["candidate_ambiguity"] = 0.0

        correlation = evidence.get("descriptor_correlation")
        if correlation is not None:
            out["appearance_inconsistency"] = float(
                np.clip(1.0 - float(correlation), 0.0, 1.0)
            )

        residual_px = evidence.get("projection_residual_px")
        bound_px = evidence.get("projection_residual_bound_px")
        if residual_px is not None and bound_px is not None and float(bound_px) > 0:
            out["projection_residual"] = float(
                np.clip(float(residual_px) / float(bound_px), 0.0, 1.0)
            )

        disagreement_m = evidence.get("multiview_disagreement_m")
        tolerance_m = evidence.get("multiview_tolerance_m")
        if (
            disagreement_m is not None
            and tolerance_m is not None
            and float(tolerance_m) > 0
        ):
            out["multiview_disagreement"] = float(
                np.clip(float(disagreement_m) / float(tolerance_m), 0.0, 1.0)
            )

        history = evidence.get("recent_centroids_px")
        if history is not None:
            points = np.asarray(history, dtype=np.float64)
            if points.ndim == 2 and points.shape[0] >= 2:
                spread = float(np.linalg.norm(points.std(axis=0)))
                scale = float(evidence.get("temporal_jitter_bound_px", 10.0))
                if scale > 0:
                    out["temporal_jitter"] = float(np.clip(spread / scale, 0.0, 1.0))
        return out

    # ── interface ────────────────────────────────────────────────────────────

    def estimate(self, evidence: dict[str, Any]) -> GroundingUncertainty:
        components = self._components(evidence)

        # Independent sources combine in quadrature; unknown sources are
        # absent rather than assumed zero, so a missing signal cannot
        # silently lower the reported sigma.
        contributions = [self._baseline]
        for name, value in components.items():
            weight = self._weights.get(name)
            if weight is None:
                continue
            contributions.append(weight * value)
        sigma = float(np.sqrt(np.sum(np.square(contributions))))

        missing = sorted(set(self._weights) - set(components))
        note = "combiner is a modelling choice, not a validated error model"
        if missing:
            note += f"; no evidence for {', '.join(missing)}"

        return GroundingUncertainty(
            position_sigma_m=sigma,
            components=components,
            calibration=UncertaintyCalibration(
                is_calibrated=False,
                n_validation_samples=0,
                notes=note,
            ),
        )
