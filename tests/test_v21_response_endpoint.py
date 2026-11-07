import numpy as np
import pytest

from fcpg.v21_column_dino_transport import require_actual_cem_response, require_actual_probe_response


def test_probe_uses_actual_image_endpoint_without_replacing_q5():
    measured = [[0.] * 6] * 6
    result = dict(source_q_position=[0.] * 6, measured_q_path=measured,
                  readback_times_s=[0., .1, .2, .3, .4, .5],
                  endpoint_time_s=.9, endpoint_q_position=[0., 0., -1., 0., 0., 0.])
    delta = require_actual_probe_response(result, 2, -1, minimum_motion=.25)
    np.testing.assert_array_equal(delta, [0., 0., -1., 0., 0., 0.])
    assert result['measured_q_path'][-1] == [0.] * 6


def test_old_endpoint_cannot_certify_a_later_probe():
    result = dict(source_q_position=[0.] * 6, measured_q_path=[[0.] * 6] * 6,
                  readback_times_s=[0., .1, .2, .3, .4, .5],
                  endpoint_time_s=.4, endpoint_q_position=[0., 0., -1., 0., 0., 0.])
    with pytest.raises(ValueError, match='endpoint must follow'):
        require_actual_probe_response(result, 2, -1, minimum_motion=.25)


def test_actual_cem_out_and_back_and_gripper_only_are_valid():
    commands = np.zeros((5, 6))
    commands[0, 0] = 1.
    measured = np.vstack((np.zeros(6), commands))
    result = dict(source_q_position=[0.] * 6, commands_position=commands,
                  measured_q_path=measured)
    np.testing.assert_array_equal(require_actual_cem_response(result, minimum_motion=.25), np.zeros(5))
    commands[:, 0] = 0.
    commands[:, 5] = 1.
    result['measured_q_path'] = np.vstack((np.zeros(6), commands))
    np.testing.assert_array_equal(require_actual_cem_response(result, minimum_motion=.25), np.zeros(5))
