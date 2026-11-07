"""Fail-closed wrist and directed fingertip guards.

The wrist camera cannot tell the two jaws apart by pixel position alone.  The
SO-101 kinematic names are therefore part of this module's contract: ``F`` is
always ``fixed_finger_tip`` and ``G`` is always ``moving_finger_tip``.  The
short axis has two mathematical signs, but only the sign nearest the factual
directed F->G vector is selected at the source observation.  Forecasts carry
that choice continuously; they never solve the sign ambiguity again.
"""
import numpy as np

from .v21_execution_geometry import J5_WRAP_MIN, J5_WRAP_MAX


FINGER_ORDER = ('fixed_finger_tip', 'moving_finger_tip')
_ANGLE_TOL = 1e-9


def _unit_xy(value, name):
    """Return finite unit 2-vectors while retaining scalar/batch shape."""
    result = np.asarray(value, np.float64)
    if (result.shape[-1:] != (2,) or not np.isfinite(result).all()):
        raise ValueError(f'{name} must be finite (..., 2)')
    norm = np.linalg.norm(result, axis=-1)
    if (norm <= 1e-8).any():
        raise ValueError(f'{name} must be nondegenerate')
    return result / norm[..., None]


def directed_f_to_g(fixed_finger_tip, moving_finger_tip):
    """Return the physical directed unit vector F->G, without endpoint swaps."""
    fixed = np.asarray(fixed_finger_tip, np.float64)
    moving = np.asarray(moving_finger_tip, np.float64)
    if fixed.shape != moving.shape or fixed.shape[-1:] != (2,):
        raise ValueError('fixed_finger_tip and moving_finger_tip must share (..., 2) shape')
    if not np.isfinite(fixed).all() or not np.isfinite(moving).all():
        raise ValueError('fixed_finger_tip and moving_finger_tip must be finite')
    return _unit_xy(moving - fixed, 'directed F-to-G vector')


def _perpendicular(long_axis):
    unit = _unit_xy(long_axis, 'fish long axis')
    return np.stack((-unit[..., 1], unit[..., 0]), axis=-1)


def _reject_discontinuous_signed_angle(angles, name):
    """Reject a large signed step instead of wrapping it through a line angle."""
    angles = np.asarray(angles, np.float64)
    if len(angles) > 1 and (
            np.abs(np.diff(angles)) >= np.pi / 2.0 - _ANGLE_TOL).any():
        raise ValueError(
            f'{name} signed angle is not continuous; per-step change crosses 90 degrees')


def choose_source_blue_short_axis(fixed_finger_tip, moving_finger_tip, fish_long_axis,
                                  *, return_receipt=False):
    """Choose the nearest signed blue short-axis branch exactly once at source.

    ``fish_long_axis`` is an unoriented image measurement.  Reversing its sign
    must therefore select the same physical blue vector.  A tie is rejected:
    there is no factual nearest branch from which to authorize a forecast.
    """
    fg = directed_f_to_g(fixed_finger_tip, moving_finger_tip)
    base = _perpendicular(fish_long_axis)
    if base.ndim != 1 or fg.ndim != 1:
        raise ValueError('source F/G and fish axes must be single 2-vectors')
    dot = float(np.dot(fg, base))
    if abs(dot) <= _ANGLE_TOL:
        raise ValueError('source blue-short-axis branch is ambiguous')
    sign = 1.0 if dot > 0.0 else -1.0
    selected = sign * base
    if not return_receipt:
        return selected
    return dict(
        fixed_finger_tip=np.asarray(fixed_finger_tip, np.float64).tolist(),
        moving_finger_tip=np.asarray(moving_finger_tip, np.float64).tolist(),
        directed_f_to_g=fg.tolist(),
        source_blue_short_axis=selected.tolist(),
        source_branch_sign=int(sign),
        jaw_endpoint_order=list(FINGER_ORDER),
        angle_branch='nearest_source_F_to_G',
        half_turn_allowed=False,
    )


def require_directed_fg_forecast(source_fixed_finger_tip, source_moving_finger_tip,
                                 source_fish_long_axis, forecast_fixed_finger_tip,
                                 forecast_moving_finger_tip, forecast_fish_long_axis):
    """Validate ordered F/G and the source-selected blue branch over a forecast.

    The forecast arrays contain one row per predicted observation.  We reject
    an endpoint reversal, a long-axis sign/branch change, or an angle crossing
    the +/-90 degree branch boundary.  In particular, an apparent 180-degree
    improvement cannot be made admissible by taking an unoriented modulo-pi
    angle.
    """
    source_fg = directed_f_to_g(source_fixed_finger_tip, source_moving_finger_tip)
    source_long = _unit_xy(source_fish_long_axis, 'source fish long axis')
    if source_fg.ndim != 1 or source_long.ndim != 1:
        raise ValueError('source F/G and fish axes must be single 2-vectors')
    source_blue = choose_source_blue_short_axis(
        source_fixed_finger_tip, source_moving_finger_tip, source_long)

    fixed = np.asarray(forecast_fixed_finger_tip, np.float64)
    moving = np.asarray(forecast_moving_finger_tip, np.float64)
    long = np.asarray(forecast_fish_long_axis, np.float64)
    if (fixed.ndim != 2 or moving.ndim != 2 or long.ndim != 2
            or fixed.shape != moving.shape or fixed.shape != long.shape
            or fixed.shape[1:] != (2,) or len(fixed) == 0):
        raise ValueError('forecast F/G and fish axes must be nonempty [steps, 2] arrays')
    forecast_fg = directed_f_to_g(fixed, moving)
    forecast_long = _unit_xy(long, 'forecast fish long axis')
    expected_blue = np.stack((-forecast_long[:, 1], forecast_long[:, 0]), axis=-1)
    source_base = _perpendicular(source_long)
    branch_sign = 1.0 if float(np.dot(source_blue, source_base)) > 0.0 else -1.0
    expected_blue *= branch_sign

    # Compare directed angles to the factual source.  A dot product alone
    # would accept a gradual route through a branch boundary; the continuous
    # signed angle makes that route fail closed as soon as it crosses +/-90.
    fg_angles = np.arctan2(
        source_fg[0] * forecast_fg[:, 1] - source_fg[1] * forecast_fg[:, 0],
        source_fg[0] * forecast_fg[:, 0] + source_fg[1] * forecast_fg[:, 1])
    if (np.abs(fg_angles) >= np.pi / 2.0 - _ANGLE_TOL).any():
        raise ValueError('forecast reverses the directed fixed F-to-G finger order')
    _reject_discontinuous_signed_angle(fg_angles, 'directed F-to-G')
    branch_dots = np.sum(forecast_fg * expected_blue, axis=-1)
    if (branch_dots <= _ANGLE_TOL).any():
        raise ValueError('forecast changes the source blue-short-axis branch')
    branch_angles = np.arctan2(
        source_blue[0] * expected_blue[:, 1] - source_blue[1] * expected_blue[:, 0],
        source_blue[0] * expected_blue[:, 0] + source_blue[1] * expected_blue[:, 1])
    if (np.abs(branch_angles) >= np.pi / 2.0 - _ANGLE_TOL).any():
        raise ValueError('forecast requires a blue-short-axis half-turn')
    _reject_discontinuous_signed_angle(branch_angles, 'source blue branch')
    alignment_angles = np.arctan2(
        forecast_fg[:, 0] * expected_blue[:, 1]
        - forecast_fg[:, 1] * expected_blue[:, 0],
        np.sum(forecast_fg * expected_blue, axis=-1))
    if (np.abs(alignment_angles) >= np.pi / 2.0 - _ANGLE_TOL).any():
        raise ValueError('forecast white/blue angle crosses the directed 90 degree boundary')
    source_alignment = np.arctan2(
        source_fg[0] * source_blue[1] - source_fg[1] * source_blue[0],
        np.dot(source_fg, source_blue))
    _reject_discontinuous_signed_angle(
        np.r_[source_alignment, alignment_angles], 'directed white/blue')
    return dict(
        jaw_endpoint_order=list(FINGER_ORDER),
        angle_branch='nearest_source_F_to_G',
        half_turn_allowed=False,
        source_directed_f_to_g=source_fg.tolist(),
        source_blue_short_axis=source_blue.tolist(),
        forecast_directed_f_to_g=forecast_fg.tolist(),
        forecast_blue_short_axis=expected_blue.tolist(),
        directed_angle_deg=np.degrees(fg_angles).tolist(),
        branch_angle_deg=np.degrees(branch_angles).tolist(),
        directed_white_blue_angle_deg=np.degrees(alignment_angles).tolist(),
        directed_white_blue_step_deg=np.degrees(
            np.diff(np.r_[source_alignment, alignment_angles])).tolist(),
    )


# Descriptive aliases keep callers from inventing a second ordering primitive.
guard_finger_order = require_directed_fg_forecast
require_finger_order = require_directed_fg_forecast


def require_wrist_path(q, path, *, max_delta=2., max_step=1.):
    """Reject, never clip, an unsafe roll path in calibrated position units."""
    q = np.asarray(q, np.float64)
    path = np.asarray(path, np.float64)
    if (q.shape != (6,) or path.ndim != 2 or path.shape[1] != 6 or not len(path)
            or not np.isfinite(q).all() or not np.isfinite(path).all()):
        raise ValueError('wrist protection requires finite actual q and a joint path')
    for name, limit in (('max_delta', max_delta), ('max_step', max_step)):
        if limit is not None and (not np.isfinite(limit) or limit < 0.):
            raise ValueError(f'{name} must be finite and nonnegative or None')
    roll = np.r_[q[4], path[:, 4]]
    if (roll < J5_WRAP_MIN - 1e-6).any() or (roll > J5_WRAP_MAX + 1e-6).any():
        raise ValueError('wrist path leaves the cable-safe J5 interval')
    if max_delta is not None and (np.abs(roll - q[4]) > max_delta + 1e-6).any():
        raise ValueError('wrist path exceeds the local roll displacement')
    if max_step is not None and (np.abs(np.diff(roll)) > max_step + 1e-6).any():
        raise ValueError('wrist path exceeds the per-command roll step')


def require_wrist_readback(q):
    require_wrist_path(q, np.asarray(q)[None], max_delta=None, max_step=None)


def wrist_guard_receipt(q, path):
    """Return auditable limits after applying the five-step wrist guard."""
    require_wrist_path(q, path)
    q = np.asarray(q, np.float64)
    path = np.asarray(path, np.float64)
    roll = np.r_[q[4], path[:, 4]]
    return dict(
        guarded_steps=int(len(path)),
        j5_band=[float(J5_WRAP_MIN), float(J5_WRAP_MAX)],
        total_roll_delta=float(np.max(np.abs(roll - q[4]))),
        max_command_roll_step=float(np.max(np.abs(np.diff(roll)))),
        total_roll_delta_limit=2.0,
        per_command_roll_step_limit=1.0,
        modulo_wrap=False,
        jaw_endpoint_order=list(FINGER_ORDER),
    )
