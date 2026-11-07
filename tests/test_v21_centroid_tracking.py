import numpy as np
import pytest

from fcpg.v21_centroid_tracking import error_px, gate_closure, progress


def test_actor_vector_is_the_centroid_error():
    assert error_px([1., .5, 3 / 224, 4 / 224, 0., 0., 1., 0.]) == pytest.approx(5.)
    with pytest.raises(ValueError, match="unavailable"):
        error_px([0.] * 8)


def test_far_centroid_gates_only_closure():
    commands = np.zeros((5, 6)); commands[:, 5] = [30., 25., 20., 15., 10.]
    action, report = gate_closure([0., 0., 0., 0., 0., 31.], commands, 80.)
    np.testing.assert_allclose(action[:, 5], 31.)
    assert report["close_gated"]
    near, near_report = gate_closure([0., 0., 0., 0., 0., 31.], commands, 20.)
    np.testing.assert_allclose(near, commands)
    assert not near_report["close_gated"]


def test_next_block_requires_real_centroid_progress():
    assert progress(100., 96.)["accepted"]
    failed = progress(100., 99.)
    assert not failed["accepted"] and failed["reason"] == "centroid_error_not_reduced"
