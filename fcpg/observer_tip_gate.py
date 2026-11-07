"""Hard-reject fingertip measurements that have locked onto the wrong feature.

The column metal detector is bimodal, not noisy.  On archived
``snapshot:metal`` revisits (same ``epoch_key``, different rounds,
``|dq|_inf <= 1 deg``, 1038 per-anchor pairs):

* 78% of the time the lock repeats to 0.63 px median
* 21.7% of the time it jumps more than 10 px (p90 = 28.24 px)

Those jumps are a knuckle / finger-joint lock, not Gaussian jitter and
not motion blur (plan/snapshot frames are captured after the arm has
settled).  Averaging, Kalman, covariance fusion or EMA land between the
two hypotheses and are wrong in both cases.

The prediction this gate compares against is therefore **not** the raw
factory FK projection.  Against live ``nominal``, a 10 px cut
false-rejects 69% of the *good* (repeatable) detections, because the
configuration-dependent residual field is 15 px median / 62 px p90.
After transporting the last *accepted* lock by the FK-predicted pixel
displacement, the same pairs separate cleanly:

* good: p50 / p90 / p99 / max = 0.15 / 2.50 / 6.66 / 9.22 px
* mode-switch: p10 = 11.34 px

10 px is the unique integer in that gap.  It is also the scale of the
11 px median residual that remains after a rigid (M1) camera alignment:
a candidate farther from the transported lock is a different physical
feature and is dropped outright, never blended.  When there is no
accepted candidate, the last lock is held (with its age) rather than
replaced by the colour detector.  Once the hold is older than
``stale_age_frames`` or the arm has moved more than ``stale_dq_deg``,
the tip is reported unavailable so a consumer can tell "no evidence"
from "fresh".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


__all__ = [
    "DEFAULT_REJECT_RADIUS_PX",
    "DEFAULT_STALE_AGE_FRAMES",
    "DEFAULT_STALE_DQ_DEG",
    "DECISION_ACCEPT",
    "DECISION_HOLD",
    "DECISION_REJECT",
    "DECISION_UNAVAILABLE",
    "TipGateResult",
    "FingertipHardRejectGate",
    "candidate_distance_px",
    "transported_prediction_px",
]


#: Integer sitting in the same-config gap (good max 9.22 px, mode-switch
#: p10 11.34 px) on hold+FK-transported ``snapshot:metal`` revisits.  See
#: the module docstring for the full evidence.
DEFAULT_REJECT_RADIUS_PX = 10.0
#: Safety cap.  Same-config metal locks still agree after a median 88 s,
#: so frame count is not the binding constraint; configuration is.
DEFAULT_STALE_AGE_FRAMES = 30
#: 1–3 deg revisits still agree at 0.76 px median; the 3–10 deg band
#: jumps to 8.67 px, so a held pixel is no longer the same feature.
DEFAULT_STALE_DQ_DEG = 3.0

DECISION_ACCEPT = "accept"
DECISION_REJECT = "reject"
DECISION_HOLD = "hold"
DECISION_UNAVAILABLE = "unavailable"

_ANCHOR_COUNT = 3


def transported_prediction_px(
    last_accepted_px: np.ndarray,
    last_predicted_px: np.ndarray,
    predicted_px: np.ndarray,
) -> np.ndarray:
    """Move a previously accepted lock by the FK-predicted pixel displacement."""

    last = np.asarray(last_accepted_px, dtype=np.float64)
    prev = np.asarray(last_predicted_px, dtype=np.float64)
    now = np.asarray(predicted_px, dtype=np.float64)
    if last.shape != (_ANCHOR_COUNT, 2) or prev.shape != now.shape:
        raise ValueError("anchor pixel arrays must have shape [3, 2]")
    return last + (now - prev)


def candidate_distance_px(
    candidate_px: np.ndarray, predicted_px: np.ndarray
) -> np.ndarray:
    """Per-anchor Euclidean distance; NaN where either side is non-finite."""

    candidate = np.asarray(candidate_px, dtype=np.float64)
    predicted = np.asarray(predicted_px, dtype=np.float64)
    if candidate.shape != (_ANCHOR_COUNT, 2) or predicted.shape != (_ANCHOR_COUNT, 2):
        raise ValueError("candidate and predicted must have shape [3, 2]")
    finite = np.all(np.isfinite(candidate), axis=1) & np.all(
        np.isfinite(predicted), axis=1
    )
    distance = np.full(_ANCHOR_COUNT, np.nan, dtype=np.float64)
    if np.any(finite):
        delta = candidate[finite] - predicted[finite]
        distance[finite] = np.linalg.norm(delta, axis=-1)
    return distance


def _max_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


@dataclass(frozen=True)
class TipGateResult:
    """One frame's accept / reject / hold / unavailable decision."""

    points_px: np.ndarray
    valid: np.ndarray
    available: bool
    fresh: bool
    decision: str
    reason: str
    age_frames: int
    age_dq_deg: float
    candidate_distance_px: np.ndarray
    max_distance_px: float
    reject_radius_px: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "available": bool(self.available),
            "fresh": bool(self.fresh),
            "age_frames": int(self.age_frames),
            "age_dq_deg": float(self.age_dq_deg),
            "reject_radius_px": float(self.reject_radius_px),
            "max_distance_px": (
                None
                if not np.isfinite(self.max_distance_px)
                else float(self.max_distance_px)
            ),
            "candidate_distance_px": [
                None if not np.isfinite(value) else float(value)
                for value in self.candidate_distance_px
            ],
            "points_px": (
                None
                if not self.available
                else np.asarray(self.points_px, dtype=np.float64).tolist()
            ),
            "valid": np.asarray(self.valid, dtype=bool).tolist(),
        }


class FingertipHardRejectGate:
    """Stateful hard-reject / hold / stale gate for one camera's tip triple."""

    def __init__(
        self,
        *,
        reject_radius_px: float = DEFAULT_REJECT_RADIUS_PX,
        stale_age_frames: int = DEFAULT_STALE_AGE_FRAMES,
        stale_dq_deg: float = DEFAULT_STALE_DQ_DEG,
    ) -> None:
        if reject_radius_px <= 0.0:
            raise ValueError("reject_radius_px must be positive")
        if stale_age_frames < 1:
            raise ValueError("stale_age_frames must be >= 1")
        if stale_dq_deg <= 0.0:
            raise ValueError("stale_dq_deg must be positive")
        self.reject_radius_px = float(reject_radius_px)
        self.stale_age_frames = int(stale_age_frames)
        self.stale_dq_deg = float(stale_dq_deg)
        self._last_accepted_px: np.ndarray | None = None
        self._last_predicted_px: np.ndarray | None = None
        self._last_joints: np.ndarray | None = None
        self._age_frames = 0
        self.last_result: TipGateResult | None = None

    @property
    def last_accepted_px(self) -> np.ndarray | None:
        if self._last_accepted_px is None:
            return None
        return self._last_accepted_px.copy()

    def reset(self) -> None:
        self._last_accepted_px = None
        self._last_predicted_px = None
        self._last_joints = None
        self._age_frames = 0
        self.last_result = None

    def _age_dq_deg(self, joints: np.ndarray | None) -> float:
        if self._last_joints is None or joints is None:
            return 0.0
        delta = np.asarray(joints, dtype=np.float64) - self._last_joints
        if delta.size == 0 or not np.all(np.isfinite(delta)):
            return 0.0
        return float(np.max(np.abs(delta)))

    def _stale(self, age_frames: int, age_dq_deg: float) -> str | None:
        if self._last_accepted_px is None:
            return "no_lock"
        if age_frames > self.stale_age_frames:
            return "stale_age"
        if age_dq_deg > self.stale_dq_deg:
            return "stale_configuration"
        return None

    def _prediction(self, predicted_px: np.ndarray) -> np.ndarray:
        if self._last_accepted_px is None or self._last_predicted_px is None:
            return np.asarray(predicted_px, dtype=np.float64)
        return transported_prediction_px(
            self._last_accepted_px, self._last_predicted_px, predicted_px
        )

    def _emit(
        self,
        *,
        points_px: np.ndarray,
        valid: np.ndarray,
        available: bool,
        fresh: bool,
        decision: str,
        reason: str,
        age_frames: int,
        age_dq_deg: float,
        distances: np.ndarray,
        extra: dict[str, Any] | None = None,
    ) -> TipGateResult:
        diagnostics = {
            "decision": decision,
            "reason": reason,
            "available": bool(available),
            "fresh": bool(fresh),
            "age_frames": int(age_frames),
            "age_dq_deg": float(age_dq_deg),
            "reject_radius_px": float(self.reject_radius_px),
            "stale_age_frames": int(self.stale_age_frames),
            "stale_dq_deg": float(self.stale_dq_deg),
            "max_distance_px": (
                None
                if not np.isfinite(_max_finite(distances))
                else _max_finite(distances)
            ),
            "candidate_distance_px": [
                None if not np.isfinite(value) else float(value) for value in distances
            ],
        }
        if extra:
            diagnostics.update(extra)
        result = TipGateResult(
            points_px=np.asarray(points_px, dtype=np.float64),
            valid=np.asarray(valid, dtype=bool),
            available=bool(available),
            fresh=bool(fresh),
            decision=decision,
            reason=reason,
            age_frames=int(age_frames),
            age_dq_deg=float(age_dq_deg),
            candidate_distance_px=np.asarray(distances, dtype=np.float64),
            max_distance_px=_max_finite(distances),
            reject_radius_px=float(self.reject_radius_px),
            diagnostics=diagnostics,
        )
        self.last_result = result
        return result

    def _unavailable(
        self,
        *,
        reason: str,
        age_frames: int,
        age_dq_deg: float,
        distances: np.ndarray,
        extra: dict[str, Any] | None = None,
    ) -> TipGateResult:
        return self._emit(
            points_px=np.full((_ANCHOR_COUNT, 2), np.nan, dtype=np.float64),
            valid=np.zeros(_ANCHOR_COUNT, dtype=bool),
            available=False,
            fresh=False,
            decision=DECISION_UNAVAILABLE,
            reason=reason,
            age_frames=age_frames,
            age_dq_deg=age_dq_deg,
            distances=distances,
            extra=extra,
        )

    def _reject_keep_hold(
        self,
        *,
        age_frames: int,
        age_dq_deg: float,
        distances: np.ndarray,
        extra: dict[str, Any] | None = None,
    ) -> TipGateResult:
        """Drop the candidate; keep the last lock.  Never blend the two."""

        assert self._last_accepted_px is not None
        valid = np.all(np.isfinite(self._last_accepted_px), axis=1)
        return self._emit(
            points_px=self._last_accepted_px.copy(),
            valid=valid,
            available=True,
            fresh=False,
            decision=DECISION_REJECT,
            reason="fk_hard_reject",
            age_frames=age_frames,
            age_dq_deg=age_dq_deg,
            distances=distances,
            extra=extra,
        )

    def _hold(
        self,
        *,
        reason: str,
        age_frames: int,
        age_dq_deg: float,
        distances: np.ndarray,
        extra: dict[str, Any] | None = None,
    ) -> TipGateResult:
        assert self._last_accepted_px is not None
        valid = np.all(np.isfinite(self._last_accepted_px), axis=1)
        return self._emit(
            points_px=self._last_accepted_px.copy(),
            valid=valid,
            available=True,
            fresh=False,
            decision=DECISION_HOLD,
            reason=reason,
            age_frames=age_frames,
            age_dq_deg=age_dq_deg,
            distances=distances,
            extra=extra,
        )

    def step(
        self,
        *,
        predicted_px: np.ndarray,
        joints: np.ndarray | None = None,
        candidate_px: np.ndarray | None = None,
        candidate_reason: str | None = None,
    ) -> TipGateResult:
        """Accept, hard-reject, hold, or expire the tip triple for one frame.

        ``predicted_px`` is the current FK (or observer) projection.  Once a
        lock exists the compared prediction is that lock transported by
        ``predicted_now - predicted_at_accept``.  A candidate past
        ``reject_radius_px`` is dropped; it is never averaged with the
        prediction.  Colour is not consulted here: a missing candidate
        holds the last lock or reports unavailable.
        """

        predicted = np.asarray(predicted_px, dtype=np.float64)
        if predicted.shape != (_ANCHOR_COUNT, 2):
            raise ValueError("predicted_px must have shape [3, 2]")
        joints_arr = (
            None if joints is None else np.asarray(joints, dtype=np.float64)
        )
        age_dq = self._age_dq_deg(joints_arr)
        age_frames = self._age_frames + 1 if self._last_accepted_px is not None else 0
        stale = self._stale(age_frames, age_dq)
        transport = self._prediction(predicted)
        empty = np.full(_ANCHOR_COUNT, np.nan, dtype=np.float64)

        candidate = None if candidate_px is None else np.asarray(
            candidate_px, dtype=np.float64
        )
        if candidate is not None and candidate.shape != (_ANCHOR_COUNT, 2):
            raise ValueError("candidate_px must have shape [3, 2]")
        if candidate is not None and not np.any(np.isfinite(candidate)):
            candidate = None

        if candidate is None:
            if stale is not None:
                if stale != "no_lock":
                    self.reset()
                return self._unavailable(
                    reason=stale if stale != "no_lock" else (
                        candidate_reason or "no_lock"
                    ),
                    age_frames=age_frames if stale != "no_lock" else 0,
                    age_dq_deg=age_dq,
                    distances=empty,
                    extra={"candidate_reason": candidate_reason},
                )
            self._age_frames = age_frames
            return self._hold(
                reason=candidate_reason or "no_candidate_hold",
                age_frames=age_frames,
                age_dq_deg=age_dq,
                distances=empty,
                extra={"candidate_reason": candidate_reason},
            )

        distances = candidate_distance_px(candidate, transport)
        max_distance = _max_finite(distances)
        locked = self._last_accepted_px is not None
        # Cold start: there is no transported lock yet.  Motion-consistency
        # (or any other upstream test) has already chosen this candidate;
        # applying the 10 px cut against raw FK would false-reject the
        # configuration residual field.  Once locked, the cut is mandatory.
        if locked and (
            not np.isfinite(max_distance) or max_distance > self.reject_radius_px
        ):
            extra = {
                "candidate_reason": candidate_reason,
                "rejected_distance_px": (
                    None if not np.isfinite(max_distance) else float(max_distance)
                ),
            }
            if stale is not None:
                self.reset()
                return self._unavailable(
                    reason="fk_hard_reject_stale" if stale != "no_lock" else "fk_hard_reject",
                    age_frames=age_frames,
                    age_dq_deg=age_dq,
                    distances=distances,
                    extra=extra,
                )
            self._age_frames = age_frames
            return self._reject_keep_hold(
                age_frames=age_frames,
                age_dq_deg=age_dq,
                distances=distances,
                extra=extra,
            )

        accepted = np.asarray(candidate, dtype=np.float64).copy()
        # Never blend: rejected anchors stay empty rather than being
        # replaced by a midpoint between knuckle and fingertip.
        valid = np.all(np.isfinite(accepted), axis=1) & np.isfinite(distances)
        if locked:
            valid = valid & (distances <= self.reject_radius_px)
        accepted[~valid] = np.nan
        if not np.any(valid):
            if stale is not None and stale != "no_lock":
                self.reset()
                return self._unavailable(
                    reason=stale,
                    age_frames=age_frames,
                    age_dq_deg=age_dq,
                    distances=distances,
                )
            if self._last_accepted_px is not None:
                self._age_frames = age_frames
                return self._reject_keep_hold(
                    age_frames=age_frames,
                    age_dq_deg=age_dq,
                    distances=distances,
                )
            return self._unavailable(
                reason=candidate_reason or "no_lock",
                age_frames=0,
                age_dq_deg=age_dq,
                distances=distances,
            )

        self._last_accepted_px = accepted
        self._last_predicted_px = predicted.copy()
        self._last_joints = None if joints_arr is None else joints_arr.copy()
        self._age_frames = 0
        return self._emit(
            points_px=accepted,
            valid=valid,
            available=True,
            fresh=True,
            decision=DECISION_ACCEPT,
            reason="accepted",
            age_frames=0,
            age_dq_deg=0.0,
            distances=distances,
            extra={"candidate_reason": candidate_reason},
        )
