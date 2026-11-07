"""Reject a paper-covered lens without inventing a new visual observation.

The submitted ICRA result says a retained real frame stays useful for about
1.5--2.0 s after that camera stops. A sheet over the lens is not that frame.
This module only classifies the pixels and times the hold. It does not write
memory, move the arm, or decide that recognition succeeded.
"""
from __future__ import annotations

import numpy as np

PAPER_RETENTION_S = 2.0
VIEWS = ("wrist", "column")


def frame_is_covered(image: np.ndarray) -> bool:
    """A lens covered by paper is flat. A real workspace frame is not."""
    if getattr(image, "shape", None) != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError("lens-cover check requires uint8 RGB [224,224,3]")
    gray = image.astype(np.float32).mean(axis=2)
    spread = float(gray.std())
    step = max(float(np.abs(np.diff(gray, axis=1)).mean()),
               float(np.abs(np.diff(gray, axis=0)).mean()))
    return spread < 12.0 and step < 3.0


class PaperCoverHold:
    """Remember the last real frame and how long a cover has lasted."""

    def __init__(self, retention_s: float = PAPER_RETENTION_S):
        if not np.isfinite(retention_s) or retention_s <= 0:
            raise ValueError("paper retention must be a positive finite duration")
        self.retention_s = float(retention_s)
        self._real: dict[str, tuple[np.ndarray, float]] = {}

    def update(self, view: str, image: np.ndarray, covered: bool, now: float) -> dict:
        if view not in VIEWS:
            raise ValueError("lens cover view must be wrist or column")
        if not np.isfinite(now):
            raise ValueError("lens cover clock must be finite")
        if not covered:
            self._real[view] = (np.array(image, copy=True), float(now))
            return dict(covered=False, retained_real_frame=False, age_s=0.0,
                        within_paper_window=False)
        saved = self._real.get(view)
        if saved is None:
            return dict(covered=True, retained_real_frame=False, age_s=None,
                        within_paper_window=False)
        _rgb, captured = saved
        age = float(now) - captured
        if age < 0:
            raise ValueError("lens cover clock moved backwards")
        return dict(covered=True, retained_real_frame=True, age_s=age,
                    within_paper_window=bool(age <= self.retention_s))

    def retained_rgb(self, view: str) -> np.ndarray | None:
        saved = self._real.get(view)
        return None if saved is None else saved[0]
