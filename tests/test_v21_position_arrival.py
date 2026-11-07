"""Pure saved-value/fixture tests; no motor or service operations."""
import json

import numpy as np
import pytest

from fcpg.v21_position_arrival import evaluate_position_arrival


def evaluate(j3, *, target=None, times=None, **kwargs):
    target = np.zeros(6) if target is None else target
    samples = np.zeros((len(j3), 6))
    samples[:, 2] = j3
    times = np.arange(len(j3)) * .05 if times is None else times
    return evaluate_position_arrival(target, samples, times,
        position_tolerance=kwargs.pop('position_tolerance', .2),
        step_tolerance=kwargs.pop('step_tolerance', .1),
        quiet_samples=kwargs.pop('quiet_samples', 3), **kwargs)


def test_saved_j3_range10_plateau_is_not_arrival_at_original_target():
    target = np.zeros(6)
    target[2] = 84.67613220214844
    result = evaluate([87.04524993896484] * 8, target=target)
    assert result['reached'] is False
    assert result['reason'] == 'outside_position_tolerance'
    assert result['tail_stable'] is True
    assert result['residual_by_axis'][2] == pytest.approx(2.3691177368164062)
    assert result['max_abs_residual'] == pytest.approx(2.3691177368164062)
    assert result['original_target'] == target.tolist()


def test_tail_requires_both_target_accuracy_and_stability():
    reached = evaluate([.18, .13, .08])
    assert reached['reached'] is True
    assert reached['max_abs_residual'] == pytest.approx(.08)
    assert reached['tail_max_abs_step'] == pytest.approx(.05)
    assert reached['reason'] == 'within_position_and_step_tolerances'
    moving = evaluate([.15, -.15, .15])
    assert moving['tail_position_within_tolerance'] is True
    assert moving['reached'] is False
    assert moving['reason'] == 'outside_step_tolerance'
    assert moving['tail_max_abs_step'] == pytest.approx(.3)


def test_only_contiguous_tail_samples_count():
    assert evaluate([10., 2., .15, .10, .05])['reached'] is True
    lost = evaluate([0., 0., 0., 1.])
    assert lost['reached'] is False
    assert lost['tail_sample_indices'] == [1, 2, 3]
    # The last sample is exact, but its recent predecessors were off target.
    last_only = evaluate([1., 1., 0.], step_tolerance=2.)
    assert last_only['max_abs_residual'] == 0.
    assert last_only['reached'] is False
    assert last_only['reason'] == 'outside_position_tolerance'


@pytest.mark.parametrize('samples', [[0.], [0., 0.]])
def test_insufficient_actual_samples_do_not_prove_arrival(samples):
    result = evaluate(samples)
    assert result['reached'] is False
    assert result['reason'] == 'insufficient_samples'
    json.dumps(result, allow_nan=False)


def test_all_axes_are_checked_without_mutating_inputs():
    target = np.arange(6, dtype=float)
    samples = np.tile(target, (3, 1))
    samples[:, 5] -= .3
    times = np.array([0., .05, .1])
    originals = [x.copy() for x in (target, samples, times)]
    result = evaluate_position_arrival(target, samples, times,
        position_tolerance=.2, step_tolerance=.1, quiet_samples=3)
    assert result['reached'] is False
    assert result['residual_by_axis'][5] == pytest.approx(-.3)
    assert result['max_abs_residual'] == pytest.approx(.3)
    for value, before in zip((target, samples, times), originals):
        np.testing.assert_array_equal(value, before)
    target[:] = 999.
    assert result['original_target'] == originals[0].tolist()


@pytest.mark.parametrize('field', ['target', 'samples', 'times'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_actual_data_is_rejected(field, bad):
    data = dict(target=np.zeros(6), samples=np.zeros((3, 6)), times=np.array([0., .05, .1]))
    data[field].flat[0] = bad
    with pytest.raises(ValueError, match='finite'):
        evaluate_position_arrival(data['target'], data['samples'], data['times'],
            position_tolerance=.2, step_tolerance=.1, quiet_samples=3)


@pytest.mark.parametrize('times', [[0., 0., .1], [0., .2, .1], [-.1, 0., .1]])
def test_relative_sample_clocks_must_be_nonnegative_and_strictly_increasing(times):
    with pytest.raises(ValueError, match='strictly increasing'):
        evaluate([0., 0., 0.], times=times)


@pytest.mark.parametrize('target,samples,times', [
    ([0.] * 5, [[0.] * 6] * 3, [0., .1, .2]),
    ([0.] * 6, [[0.] * 5] * 3, [0., .1, .2]),
    ([0.] * 6, [0.] * 6, [0.]),
    ([0.] * 6, np.empty((0, 6)), []),
    ([0.] * 6, [[0.] * 6] * 3, [0., .1]),
])
def test_shape_or_missing_sample_errors_are_rejected(target, samples, times):
    with pytest.raises(ValueError, match='shape'):
        evaluate_position_arrival(target, samples, times,
            position_tolerance=.2, step_tolerance=.1, quiet_samples=3)


@pytest.mark.parametrize('kwargs', [
    {'position_tolerance': -.1}, {'position_tolerance': float('nan')},
    {'step_tolerance': float('inf')}, {'step_tolerance': True},
    {'quiet_samples': 1}, {'quiet_samples': True}, {'quiet_samples': 2.5},
])
def test_invalid_acceptance_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        evaluate([0., 0., 0.], **kwargs)


def test_tolerances_are_explicit_and_never_silently_widened():
    assert evaluate([0., 0., 0.], position_tolerance=0., step_tolerance=0.)['reached']
    assert not evaluate([.201] * 3)['reached']
    with pytest.raises(TypeError):
        evaluate_position_arrival(np.zeros(6), np.zeros((3, 6)), [0., .05, .1])


def test_finite_inputs_cannot_produce_nonfinite_residuals():
    with pytest.raises(ValueError, match='arithmetic is nonfinite'):
        evaluate_position_arrival(np.full(6, -1e308), np.full((3, 6), 1e308), [0., .05, .1],
            position_tolerance=.2, step_tolerance=.1, quiet_samples=3)
