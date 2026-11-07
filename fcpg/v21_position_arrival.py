"""Pure position-arrival assessment against one frozen original target."""
from numbers import Integral, Real

import numpy as np


def _finite_array(value, name):
    try:
        array = np.asarray(value)
        if array.dtype.kind not in 'iuf':
            raise ValueError('not real numeric data')
        array = np.array(array, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must contain real numeric values') from error
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite values')
    return array


def evaluate_position_arrival(original_target, actual_positions, relative_times_s, *,
                              position_tolerance, step_tolerance, quiet_samples):
    """Judge the last K actual samples, without changing a target or a command.

    Tolerances are explicit calibrated-position units; no global default or
    encoder allowance is inserted. K = quiet_samples must be at least two:
    all K tail positions must be close to original_target, and all K-1
    adjacent changes must be small. Earlier successful samples do not count.

    The caller supplies fresh actual readbacks and the original target frozen
    before execution, never a replacement HOLD target. This pure function can
    validate relative clocks but cannot establish freshness or provenance.
    It performs no waiting, dispatch, replanning, or register access.
    """
    for name, value in (('position_tolerance', position_tolerance),
                        ('step_tolerance', step_tolerance)):
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be a finite nonnegative scalar')
    if isinstance(quiet_samples, bool) or not isinstance(quiet_samples, Integral) or quiet_samples < 2:
        raise ValueError('quiet_samples must be an integer of at least two')
    count = int(quiet_samples)
    target = _finite_array(original_target, 'original_target')
    samples = _finite_array(actual_positions, 'actual_positions')
    times = _finite_array(relative_times_s, 'relative_times_s')
    if target.shape != (6,):
        raise ValueError('original_target must have shape [6]')
    if samples.ndim != 2 or samples.shape[1] != 6 or len(samples) == 0:
        raise ValueError('actual_positions must have nonempty shape [N,6]')
    if times.shape != (len(samples),):
        raise ValueError('relative_times_s must have shape [N] matching actual_positions')
    if (times < 0).any() or not (np.diff(times) > 0).all():
        raise ValueError('relative_times_s must be nonnegative and strictly increasing')

    start = max(0, len(samples) - count)
    tail = samples[start:]
    with np.errstate(over='ignore', invalid='ignore'):
        residuals = tail - target
        steps = np.diff(tail, axis=0)
    if not np.isfinite(residuals).all() or not np.isfinite(steps).all():
        raise ValueError('position residual or step arithmetic is nonfinite')
    max_error = float(np.abs(residuals[-1]).max())
    tail_max_error = float(np.abs(residuals).max())
    tail_max_step = float(np.abs(steps).max()) if len(steps) else None
    close = tail_max_error <= position_tolerance
    stable = tail_max_step is not None and tail_max_step <= step_tolerance
    enough = len(samples) >= count
    if not enough:
        reason = 'insufficient_samples'
    elif not close:
        reason = 'outside_position_tolerance'
    elif not stable:
        reason = 'outside_step_tolerance'
    else:
        reason = 'within_position_and_step_tolerances'
    return dict(schema='v21-position-arrival-v1',
        original_target=target.tolist(), actual_position=samples[-1].tolist(),
        residual_by_axis=residuals[-1].tolist(), max_abs_residual=max_error,
        residual_convention='actual_position minus frozen original_target',
        reached=bool(enough and close and stable), reason=reason,
        sample_count=len(samples), quiet_samples=count,
        position_tolerance=float(position_tolerance), step_tolerance=float(step_tolerance),
        tail_sample_indices=list(range(start, len(samples))),
        tail_relative_times_s=times[start:].tolist(),
        tail_max_abs_residual=tail_max_error, tail_max_abs_step=tail_max_step,
        tail_position_within_tolerance=bool(close), tail_stable=bool(stable))


def evaluate_selected_position_arrival(original_target, actual_positions, relative_times_s, *,
                                       selected_axes, position_tolerance, step_tolerance,
                                       quiet_samples):
    """Assess a stated subset from real six-axis samples without replacing coordinates.

    The ordinary all-axis result remains separate. In particular, reaching the
    selected targets cannot turn an unrelated joint's residual into an arrival.
    """
    result = evaluate_position_arrival(original_target, actual_positions, relative_times_s,
        position_tolerance=position_tolerance, step_tolerance=step_tolerance,
        quiet_samples=quiet_samples)
    try:
        axes = list(selected_axes)
    except TypeError as error:
        raise ValueError('selected_axes must be a nonempty sequence of unique joint indices') from error
    if (not axes or any(isinstance(axis, bool) or not isinstance(axis, Integral)
                        or not 0 <= axis < 6 for axis in axes)
            or len(set(axes)) != len(axes)):
        raise ValueError('selected_axes must be a nonempty sequence of unique joint indices')
    axes = [int(axis) for axis in axes]
    target = np.asarray(original_target, dtype=np.float64)[axes]
    samples = np.asarray(actual_positions, dtype=np.float64)[:, axes]
    tail = samples[result['tail_sample_indices']]
    residuals = tail - target
    steps = np.diff(tail, axis=0)
    max_step = float(np.abs(steps).max()) if len(steps) else None
    close = float(np.abs(residuals).max()) <= position_tolerance
    stable = max_step is not None and max_step <= step_tolerance
    enough = len(samples) >= quiet_samples
    reason = ('insufficient_samples' if not enough else
              'outside_position_tolerance' if not close else
              'outside_step_tolerance' if not stable else
              'within_position_and_step_tolerances')
    result.update(schema='v21-selected-position-arrival-v1', axes=axes,
        scope='selected_axes_only', all_axes_reached=result['reached'],
        original_target=target.tolist(), actual_position=samples[-1].tolist(),
        residual_by_axis=residuals[-1].tolist(), max_abs_residual=float(np.abs(residuals[-1]).max()),
        reached=bool(enough and close and stable), reason=reason,
        tail_max_abs_residual=float(np.abs(residuals).max()), tail_max_abs_step=max_step,
        tail_position_within_tolerance=bool(close), tail_stable=bool(stable))
    return result
