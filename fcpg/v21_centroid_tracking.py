"""Small, pure gates for tracking the Actor's fish-body target.

This module does not generate motor commands.  It only makes the target
contract explicit: the same C-minus-M vector sent to the Actor decides when a
block may close and whether a completed block made usable progress.
"""
import numpy as np


def error_px(vector):
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if value.shape != (8,) or not np.isfinite(value).all() or value[0] <= 0:
        raise ValueError("current fish-body target is unavailable")
    return float(np.hypot(value[2], value[3]) * 224.)


def gate_closure(source_q, commands, before_error_px, *, close_error_px=32.):
    q = np.asarray(source_q, dtype=np.float64).reshape(-1)
    action = np.asarray(commands, dtype=np.float64)
    if q.shape != (6,) or action.shape != (5, 6):
        raise ValueError("expected q[6] and five command positions")
    if not np.isfinite(q).all() or not np.isfinite(action).all() or not np.isfinite(before_error_px):
        raise ValueError("centroid gate requires finite inputs")
    closing = bool((np.diff(np.r_[q[5], action[:, 5]]) < -1e-5).any())
    gated = closing and before_error_px > close_error_px
    if gated:
        action = action.copy()
        action[:, 5] = q[5]
    return action, dict(close_requested=closing, close_gated=gated,
                        close_error_px=float(close_error_px), before_error_px=float(before_error_px))


def progress(before_error_px, after_error_px, *, minimum_progress_px=2.):
    if not np.isfinite([before_error_px, after_error_px, minimum_progress_px]).all() or minimum_progress_px <= 0:
        raise ValueError("centroid progress values must be finite and positive")
    delta = float(before_error_px - after_error_px)
    return dict(before_error_px=float(before_error_px), after_error_px=float(after_error_px),
                improvement_px=delta, minimum_progress_px=float(minimum_progress_px),
                accepted=bool(delta >= minimum_progress_px),
                reason="centroid_error_reduced" if delta >= minimum_progress_px else "centroid_error_not_reduced")
