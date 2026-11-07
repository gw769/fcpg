import numpy as np
import pytest

from fcpg.v21_position_arrival import evaluate_position_arrival, evaluate_selected_position_arrival


def assess(target, samples, **kwargs):
    return evaluate_selected_position_arrival(target, samples, np.arange(len(samples)) * .05,
        selected_axes=kwargs.get('selected_axes', (2, 3, 1)), position_tolerance=.2,
        step_tolerance=.1, quiet_samples=22)


def test_stationary_j1_residual_is_not_erased_by_selected_arrival():
    target = np.array([-6.6414709091, -52.3390808105, 71.8829345703,
                       60.0954055786, -62.9026565552, 26.2756099701])
    samples = np.tile(target, (22, 1))
    samples[:, 0] += .210835070968
    samples[:, 2] += .08
    original = samples.copy()
    selected = assess(target, samples)
    complete = evaluate_position_arrival(target, samples, np.arange(22) * .05,
        position_tolerance=.2, step_tolerance=.1, quiet_samples=22)
    assert selected['reached'] is True
    assert selected['all_axes_reached'] is complete['reached'] is False
    assert selected['axes'] == [2, 3, 1]
    assert selected['original_target'] == target[[2, 3, 1]].tolist()
    assert selected['actual_position'] == samples[-1, [2, 3, 1]].tolist()
    np.testing.assert_array_equal(samples, original)


@pytest.mark.parametrize('axis', (2, 3, 1))
def test_every_selected_axis_must_hold_for_entire_window(axis):
    samples = np.zeros((22, 6))
    samples[0, axis] = .21
    result = assess(np.zeros(6), samples)
    assert result['reached'] is False
    assert result['reason'] == 'outside_position_tolerance'


def test_selected_motion_is_not_called_quiet_within_position_tolerance():
    samples = np.zeros((22, 6))
    samples[-1, 3] = .11
    assert assess(np.zeros(6), samples)['reason'] == 'outside_step_tolerance'


def test_unselected_corrupt_measurement_is_still_rejected():
    samples = np.zeros((22, 6))
    samples[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        assess(np.zeros(6), samples)


@pytest.mark.parametrize('axes', ((), (2, 2), (True,), (-1,), (6,), (2.,), None))
def test_invalid_axis_selection_is_rejected(axes):
    with pytest.raises(ValueError, match='selected_axes'):
        assess(np.zeros(6), np.zeros((22, 6)), selected_axes=axes)


def test_short_selected_window_does_not_claim_full_hold():
    assert assess(np.zeros(6), np.zeros((3, 6)))['reason'] == 'insufficient_samples'
