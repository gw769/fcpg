from copy import deepcopy
import numpy as np
import pytest

from fcpg.v21_column_dino_transport import (
    CEMActualResponseError, require_actual_cem_response, short_window_keeps_actual_response)


def trace(j1):
    command = [.403, 1.155, -3., 1.251, -.22, -.334]
    actual = np.zeros((6, 6))
    actual[1:, 0] = j1
    actual[1:, 1] = np.linspace(.2, .99, 5)
    actual[1:, 2] = -np.linspace(.1, .89, 5)
    actual[1:, 3] = np.linspace(.2, 1.09, 5)
    return dict(source_q_position=[0.]*6, commands_position=[command]*5,
                measured_q_path=actual.tolist())


def test_failed_axis_is_reported_without_changing_trace_or_threshold():
    value = trace(0.)
    before = deepcopy(value)
    with pytest.raises(CEMActualResponseError, match='J1:') as caught:
        require_actual_cem_response(value, minimum_motion=.25)
    diag = caught.value.diagnostic
    assert diag['blocked_arm_joint_indices'] == [0]
    assert diag['commanded_peak_delta'][0] == .403
    assert diag['achieved_directional_motion'][0] == 0
    assert diag['minimum_motion'] == .25
    assert not diag['predictions_used_as_evidence']
    assert value == before


def test_exact_response_threshold_still_fails():
    with pytest.raises(CEMActualResponseError):
        require_actual_cem_response(trace(.25), minimum_motion=.25)


def test_above_threshold_retains_actual_return_value():
    value = trace(.251)
    np.testing.assert_array_equal(require_actual_cem_response(value, minimum_motion=.25),
                                  np.array(value['measured_q_path'])[-1, :5])


def test_peak_response_is_not_replaced_by_final_delta():
    value = trace(0.)
    value['measured_q_path'][2][0] = .3
    actual = require_actual_cem_response(value, minimum_motion=.25)
    assert actual[0] == 0.  # The same existing peak-based guard still accepts it.


def test_short_follow_keeps_the_j2_j3_shortfall_instead_of_failing_the_request():
    value = trace(0.)
    value['commands_position'] = [[0., -1.06586, -0.955917, 0., 0., 0.]] * 5
    value['measured_q_path'] = np.zeros((6, 6)).tolist()
    value['measured_q_path'][-1][2] = 0.0887375
    value.update(completed=True, policy_commands_returned=5,
                 normal_follow=dict(budget_s=0.4))
    with pytest.raises(CEMActualResponseError) as caught:
        require_actual_cem_response(value, minimum_motion=.25)
    assert short_window_keeps_actual_response(value, caught.value)
    longer = deepcopy(value)
    longer['normal_follow'] = dict(budget_s=2.0)
    assert not short_window_keeps_actual_response(longer, caught.value)


def test_invalid_trace_is_not_given_a_physical_failure_diagnosis():
    value = trace(0.)
    value['measured_q_path'][2][0] = float('nan')
    with pytest.raises(ValueError) as caught:
        require_actual_cem_response(value, minimum_motion=.25)
    assert not isinstance(caught.value, CEMActualResponseError)
