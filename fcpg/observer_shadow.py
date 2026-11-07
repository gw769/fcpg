"""Live shadow-mode recorder for the recursive observer (milestone M4).

The observer runs next to the existing affine + 2-D Kalman path and writes
what it would have done.  It does not change a commanded joint, a close
decision, or a drop registration.  That is the point of a shadow: the
comparison is against the live path on the same frames, not against a
replay that no longer matches what the robot saw.

A record that the observer never produces -- because shadow mode was off,
or because a frame had no usable measurement -- is stored as a skip with a
reason, not as a silent zero.  Plotting code that treated a missing update
as "zero error" would invent a convergence curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fcpg.observer_filter import InformationObserver, ObserverUpdate


__all__ = ["ShadowRecord", "ShadowLog"]


@dataclass(frozen=True)
class ShadowRecord:
    round_id: str
    phase: str | None
    accepted: bool
    residual_rms_px: float | None
    mahalanobis: float | None
    rejection_reason: str | None
    observer_tips_px: list[list[float]] | None
    affine_tips_px: list[list[float]] | None
    kf2d_offset_px: list[float] | None
    observer_sigma: list[float]
    trace_p: float
    along_close_sigma_m: float | None
    n_motion_tips: int
    tip_gate: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "phase": self.phase,
            "accepted": self.accepted,
            "residual_rms_px": self.residual_rms_px,
            "mahalanobis": self.mahalanobis,
            "rejection_reason": self.rejection_reason,
            "observer_tips_px": self.observer_tips_px,
            "affine_tips_px": self.affine_tips_px,
            "kf2d_offset_px": self.kf2d_offset_px,
            "observer_sigma": self.observer_sigma,
            "trace_p": self.trace_p,
            "along_close_sigma_m": self.along_close_sigma_m,
            "n_motion_tips": self.n_motion_tips,
            "tip_gate": self.tip_gate,
        }


@dataclass
class ShadowLog:
    """In-memory log the live planner can dump at the end of a session."""

    records: list[ShadowRecord] = field(default_factory=list)

    def append(
        self,
        *,
        round_id: str,
        phase: str | None,
        observer: InformationObserver,
        update: ObserverUpdate | None,
        observer_tips_px: np.ndarray | None,
        affine_tips_px: np.ndarray | None,
        kf2d_offset_px: np.ndarray | None,
        along_close_sigma_m: float | None,
        n_motion_tips: int = 0,
        tip_gate: dict[str, Any] | None = None,
    ) -> ShadowRecord:
        record = ShadowRecord(
            round_id=str(round_id),
            phase=phase,
            accepted=bool(update.accepted) if update is not None else False,
            residual_rms_px=None if update is None else float(update.residual_rms_px),
            mahalanobis=None if update is None else float(update.mahalanobis),
            rejection_reason=(
                "no_update" if update is None else update.rejection_reason
            ),
            observer_tips_px=(
                None
                if observer_tips_px is None
                else np.asarray(observer_tips_px, dtype=np.float64).tolist()
            ),
            affine_tips_px=(
                None
                if affine_tips_px is None
                else np.asarray(affine_tips_px, dtype=np.float64).tolist()
            ),
            kf2d_offset_px=(
                None
                if kf2d_offset_px is None
                else np.asarray(kf2d_offset_px, dtype=np.float64).reshape(-1).tolist()
            ),
            observer_sigma=observer.state.sigma().tolist(),
            trace_p=float(np.trace(observer.state.covariance())),
            along_close_sigma_m=along_close_sigma_m,
            n_motion_tips=int(n_motion_tips),
            tip_gate=None if tip_gate is None else dict(tip_gate),
        )
        self.records.append(record)
        return record

    def summary(self) -> dict[str, Any]:
        accepted = [r for r in self.records if r.accepted]
        residuals = [
            r.residual_rms_px for r in accepted if r.residual_rms_px is not None
        ]
        return {
            "frames": len(self.records),
            "accepted": len(accepted),
            "median_residual_px": (
                None if not residuals else float(np.median(residuals))
            ),
            "final_trace_p": (
                None if not self.records else self.records[-1].trace_p
            ),
        }
