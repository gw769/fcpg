"""Offline contracts for directed F/G and the local J5 wrist guard.

These tests use only synthetic image vectors and numpy arrays.  They do not
authorize motion and provide no physical grasp evidence.
"""
import numpy as np
import pytest

from fcpg.v21_grasp_cost import WristGraspCost
from fcpg.v21_visual_alignment import AlignmentObservation, MeasuredImageResponse
from fcpg.v21_wrist_guard import (
    choose_source_blue_short_axis,
    directed_f_to_g,
    require_directed_fg_forecast,
    require_wrist_path,
    wrist_guard_receipt,
)


Q = np.array([-5., -25., 35., 10., -65., 30.])
FIXED = np.array([0., 0.])
MOVING = np.array([1., 0.])
LONG = np.array([0., 1.])


def repeated(value, steps=5):
    return np.repeat(np.asarray(value, np.float64)[None], steps, axis=0)


def test_source_selection_is_physical_and_invariant_to_unoriented_long_axis_sign():
    expected = np.array([1., 0.])
    np.testing.assert_allclose(choose_source_blue_short_axis(FIXED, MOVING, LONG), expected)
    np.testing.assert_allclose(choose_source_blue_short_axis(FIXED, MOVING, -LONG), expected)
    np.testing.assert_allclose(directed_f_to_g(FIXED, MOVING), expected)
    np.testing.assert_allclose(directed_f_to_g(MOVING, FIXED), -expected)


def test_old_modulo_pi_180_degree_bug_is_rejected():
    # An unoriented line metric scores this opposite long-axis branch as zero.
    # The source-selected physical branch must instead reject it as a half-turn.
    with pytest.raises(ValueError, match='half-turn|branch'):
        require_directed_fg_forecast(
            FIXED, MOVING, LONG,
            repeated(FIXED), repeated(MOVING), repeated(-LONG))


def test_swapping_f_and_g_in_a_forecast_is_rejected_without_silent_relabeling():
    forecast_fixed = repeated(FIXED)
    forecast_moving = repeated(MOVING)
    forecast_moving[-1], forecast_fixed[-1] = FIXED, MOVING
    with pytest.raises(ValueError, match='directed fixed F-to-G|branch'):
        require_directed_fg_forecast(
            FIXED, MOVING, LONG,
            forecast_fixed, forecast_moving, repeated(LONG))


def test_branch_crossing_at_ninety_degrees_is_rejected():
    # F/G remains ordered, but the forecasted fish long axis would force the
    # source-selected blue branch through the +/-90 degree ambiguity.
    angle = np.radians(90.)
    crossing = np.array([[-np.sin(angle), np.cos(angle)]])
    with pytest.raises(ValueError, match='branch|half-turn'):
        require_directed_fg_forecast(
            FIXED, MOVING, LONG,
            repeated(FIXED, 1), repeated(MOVING, 1), crossing)


def test_signed_forecast_angle_cannot_jump_between_opposite_near_boundary_sides():
    angles = np.radians([80., -80.])
    forecast_moving = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    forecast_long = np.stack((-np.sin(angles), np.cos(angles)), axis=-1)
    with pytest.raises(ValueError, match='continuous|angle step'):
        require_directed_fg_forecast(
            FIXED, MOVING, LONG,
            repeated(FIXED, 2), forecast_moving, forecast_long)


def test_safe_forecast_receipt_keeps_order_and_branch_contract():
    angle = np.radians(8.)
    forecast_long = np.array([[-np.sin(angle), np.cos(angle)]] * 5)
    receipt = require_directed_fg_forecast(
        FIXED, MOVING, LONG,
        repeated(FIXED), repeated(MOVING), forecast_long)
    assert receipt['jaw_endpoint_order'] == ['fixed_finger_tip', 'moving_finger_tip']
    assert receipt['angle_branch'] == 'nearest_source_F_to_G'
    assert receipt['half_turn_allowed'] is False
    assert max(abs(value) for value in receipt['branch_angle_deg']) == pytest.approx(8.)


def test_wrist_guard_preserves_band_total_delta_and_per_command_step():
    path = np.repeat(Q[None], 5, axis=0)
    path[:, 4] = Q[4] + np.arange(1., 6.) * .4
    require_wrist_path(Q, path)
    receipt = wrist_guard_receipt(Q, path)
    assert receipt['j5_band'] == [-95., -45.]
    assert receipt['guarded_steps'] == 5
    assert receipt['total_roll_delta'] == pytest.approx(2.)
    assert receipt['max_command_roll_step'] == pytest.approx(.4)
    assert receipt['total_roll_delta_limit'] == 2.
    assert receipt['per_command_roll_step_limit'] == 1.
    assert receipt['modulo_wrap'] is False


@pytest.mark.parametrize('mutator, message', [
    (lambda p: p.__setitem__((0, 4), -95.1), 'cable-safe'),
    (lambda p: p.__setitem__((4, 4), -62.9), 'local roll'),
    (lambda p: p.__setitem__((1, 4), -63.5), 'per-command'),
])
def test_wrist_guard_rejects_band_total_or_step_violation(mutator, message):
    path = np.repeat(Q[None], 5, axis=0)
    path[:, 4] = Q[4] + np.arange(1., 6.) * .4
    mutator(path)
    with pytest.raises(ValueError, match=message):
        require_wrist_path(Q, path)


def test_receipt_contract_matches_existing_wrist_grasp_cost_without_integration_edits():
    observation = AlignmentObservation(
        Q.copy(), np.array([120., 112.]), LONG.copy(),
        np.array([[90., 112.], [130., 112.]]), 1., 1.01)
    response = MeasuredImageResponse(
        np.array([[1., 0., 0., 0., .1], [0., 1., 0., 0., .1],
                  [0., 0., 0., 0., .1]]),
        Q.copy(), np.ones(5), np.ones(5), np.zeros(3), 1., 10)
    receipt = WristGraspCost(observation, response).receipt()
    assert receipt['jaw_endpoint_order'] == ['fixed_finger_tip', 'moving_finger_tip']
    assert receipt['angle_branch'] == 'nearest_source_F_to_G'
    assert receipt['half_turn_allowed'] is False
