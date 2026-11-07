"""Independent simulated regression for the 2 s response observation contract.

No service, serial, cameras, checkpoint loading, or model forward calls. The
generic fake-clock runtime is reused; all response trajectories are explicit
fixtures independent of the follow implementation.
"""
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from fcpg.v21_models import CommandTrace
from fcpg.v21_position_arrival import evaluate_position_arrival
from test_v21_hardware_bridge import setup
from test_v21_normal_completion_hold import fixture as hold_fixture


@dataclass
class Pending:
    source_q: object
    source_time: object
    trace: object
    response: object


@dataclass
class State:
    q: object
    time: object


def full_window_bridge(tail):
    """q5 is 2.6 units short but already passes the .25 response gate."""
    bridge, runtime, writes, holds = setup(normal_follow_budget_s=2.)
    commands = torch.zeros(1, 5, 6)
    commands[0, :, 2] = torch.tensor([-1.5, -3., -4.5, -5.8, -5.8])
    original = runtime.pending
    runtime.pending = Pending(original.source_q, torch.tensor([10.], dtype=torch.float64),
        CommandTrace.policy(commands, torch.full((1, 5), .1)), original.response)
    runtime.state = State(original.source_q, torch.tensor([10.], dtype=torch.float64))
    runtime.pending_action_source = 'v21_lewm_cem'

    def actual():
        stamp = bridge.clock()
        q = np.zeros(6, np.float32)
        if stamp < 10.5 - 1e-8:
            index = int(round((stamp - 10.) / .1))
            q[2] = [0., -.4, -1., -2., -3.2][min(index, 4)]
        else:
            q[2] = -3.2 if stamp <= 10.5 + 1e-8 else tail(stamp - 10.5)
        return q

    bridge.read_q = actual
    bridge.capture_endpoint = lambda: State(torch.from_numpy(actual())[None],
        torch.tensor([bridge.clock()], dtype=torch.float64))
    return bridge, runtime, writes, holds


def assert_original_y(result, runtime):
    q = np.asarray(result['measured_q_path'])
    np.testing.assert_allclose(q[:, 2], [0., -.4, -1., -2., -3.2, -3.2], atol=1e-6, rtol=0)
    assert q.shape == (6, 6)
    np.testing.assert_allclose(result['readback_times_s'], np.arange(6) * .1 + 10., atol=1e-8, rtol=0)
    np.testing.assert_array_equal(runtime.actual[1][0].numpy(), q)
    assert result['model_feedback_path'] == 'original_five_step_readback'
    assert runtime.actual[0].positions.shape == (1, 5, 6)


def assert_arrival(result, reached):
    follow = result['normal_follow']
    assert follow['fixed_observation_window'] is True
    assert follow['reason'] == 'fixed_window_complete'
    assert follow['goal_reached_scope'] == 'original_command_position_arrival'
    assert follow['goal_reached'] is reached
    assert follow['hold_eligible'] is True
    arrival = follow['original_target_arrival']
    assert arrival['reached'] is reached
    assert arrival['position_tolerance'] == .2
    assert arrival['step_tolerance'] == .1
    assert arrival['quiet_samples'] == 3
    assert arrival['original_target'] == result['commands_position'][-1]
    assert result['original_target_arrival'] == arrival
    assert result['original_target_reached'] is reached


def test_stable_2p6_error_cannot_shorten_the_explicit_two_second_window():
    bridge, runtime, writes, holds = full_window_bridge(lambda elapsed: -3.2)
    result = bridge.execute()
    follow = result['normal_follow']
    assert len(follow['positions']) == 40
    np.testing.assert_allclose(follow['times_s'], 10.5 + np.arange(1, 41) * .05, atol=1e-8, rtol=0)
    assert follow['final_position_error'] == pytest.approx(2.6, abs=1e-6)
    assert follow['cem_response_satisfied'] is True
    assert len(writes) == 5 and holds == [pytest.approx(12.5)]
    assert_original_y(result, runtime)
    assert_arrival(result, False)


def test_motion_that_starts_after_old_early_exit_is_observed_without_replacing_y5():
    def delayed(elapsed):
        if elapsed < .9:
            return -3.2
        return -min(5.75, 3.2 + (elapsed - .9) * 8.)

    bridge, runtime, writes, holds = full_window_bridge(delayed)
    result = bridge.execute()
    follow = result['normal_follow']
    assert len(follow['positions']) == 40
    assert follow['times_s'][-1] == pytest.approx(12.5)
    assert follow['positions'][2][2] == pytest.approx(-3.2, abs=1e-6)
    assert follow['positions'][-1][2] == pytest.approx(-5.75)
    assert result['endpoint_q_position'][2] == pytest.approx(-5.75)
    assert len(writes) == 5 and len(holds) == 1
    assert_original_y(result, runtime)
    assert_arrival(result, True)


def test_a_temporarily_reached_goal_does_not_hide_a_late_departure():
    bridge, runtime, writes, holds = full_window_bridge(
        lambda elapsed: -5.75 if elapsed < 1.2 else -3.2)
    result = bridge.execute()
    follow = result['normal_follow']
    assert len(follow['positions']) == 40
    assert follow['positions'][3][2] == pytest.approx(-5.75)
    assert follow['positions'][-1][2] == pytest.approx(-3.2, abs=1e-6)
    assert follow['final_position_error'] == pytest.approx(2.6, abs=1e-6)
    assert len(writes) == 5 and holds[-1] == pytest.approx(12.5)
    assert_original_y(result, runtime)
    assert_arrival(result, False)


@pytest.mark.parametrize('stop_at', [10.75, 11.65, 12.5])
def test_stop_interrupts_even_when_response_and_coarse_stability_already_pass(stop_at):
    bridge, runtime, writes, holds = full_window_bridge(lambda elapsed: -3.2)
    bridge.cancelled = lambda: bridge.clock() >= stop_at - 1e-8
    with pytest.raises(RuntimeError, match='cancelled'):
        bridge.execute()
    assert len(writes) == 5 and runtime.stopped
    assert holds == [pytest.approx(stop_at)]
    assert not bridge.last_attempt['completed'] and not hasattr(runtime, 'actual')
    assert bridge.last_attempt['readback_times_s'][-1] == pytest.approx(10.5)
    assert bridge.last_attempt['measured_q_path'][-1][2] == pytest.approx(-3.2, abs=1e-6)
    assert max(bridge.last_attempt['normal_follow']['times_s']) < stop_at


def test_late_read_timeout_freezes_and_keeps_partial_actual_evidence():
    bridge, runtime, writes, holds = full_window_bridge(lambda elapsed: -3.2)
    actual = bridge.read_q

    def slow_read():
        q = actual()
        if bridge.clock() >= 11.65 - 1e-8:
            bridge.wait(.031)
        return q

    bridge.read_q = slow_read
    with pytest.raises(RuntimeError, match='deadline'):
        bridge.execute()
    assert len(writes) == 5 and runtime.stopped and len(holds) == 1
    assert holds[-1] == pytest.approx(11.681)
    assert not bridge.last_attempt['completed'] and not hasattr(runtime, 'actual')
    assert bridge.last_attempt['deadline_miss']['allowed_lateness_s'] == .03
    assert bridge.last_attempt['deadline_miss']['excess_s'] == pytest.approx(.001)
    assert bridge.last_attempt['readback_times_s'][-1] == pytest.approx(10.5)
    assert bridge.last_attempt['normal_follow']['times_s'][-1] == pytest.approx(11.681)


def test_late_geometry_failure_interrupts_full_window_without_new_commands():
    bridge, runtime, writes, holds = full_window_bridge(lambda elapsed: -3.2)

    def geometry(q, commands):
        if bridge.clock() >= 11.15 - 1e-8:
            raise RuntimeError('simulated late geometry failure')

    bridge.physical_check = geometry
    with pytest.raises(RuntimeError, match='late geometry'):
        bridge.execute()
    assert len(writes) == 5 and runtime.stopped and holds == [pytest.approx(11.15)]
    assert not bridge.last_attempt['completed'] and not hasattr(runtime, 'actual')
    assert bridge.last_attempt['readback_times_s'][-1] == pytest.approx(10.5)


def stable_miss_hold_receipt():
    owner, bus = hold_fixture()
    bus.bias[:] = 0.
    bus.bias[2] = 2.6
    target = bus.goal.copy()
    q = bus.goal + bus.bias
    positions = np.repeat(q[None], 40, axis=0)
    arrival = evaluate_position_arrival(target, positions, np.arange(1, 41) * .05,
        position_tolerance=.2, step_tolerance=.1, quiet_samples=3)
    follow = dict(budget_s=2., goal_reached=False, hold_eligible=True,
        goal_reached_scope='original_command_position_arrival', positions=positions.tolist(),
        times_s=(10.5 + np.arange(1, 41) * .05).tolist(), reason='fixed_window_complete',
        fixed_observation_window=True, original_target_arrival=arrival)
    return owner, bus, target, q, follow


def test_coarse_safe_hold_may_retain_goal_without_claiming_precise_arrival():
    owner, bus, target, q5, follow = stable_miss_hold_receipt()
    result = owner._completion_hold(target, follow, q5)
    assert result['reused_existing_hold'] is True
    assert result['motor_register_writes'] == 0
    assert 'goal_write' not in bus.events
    assert follow['goal_reached'] is False
    assert follow['original_target_arrival']['reached'] is False
    assert follow['original_target_arrival']['max_abs_residual'] == pytest.approx(2.6)
    np.testing.assert_array_equal(bus.goal, target)


def test_explicit_ineligible_hold_is_not_overridden_by_a_goal_reached_boolean():
    owner, bus, target, q5, follow = stable_miss_hold_receipt()
    follow.update(hold_eligible=False, goal_reached=True)
    result = owner._completion_hold(target, follow, q5)
    assert result['reused_existing_hold'] is False
    assert result['motor_register_writes'] == 1
    assert bus.events.count('goal_write') == 1


def test_fresh_present_must_still_pass_old_hold_safety_after_full_window():
    owner, bus, target, q5, follow = stable_miss_hold_receipt()
    bus.bias[2] = 3.2
    result = owner._completion_hold(target, follow, q5)
    assert result['reused_existing_hold'] is False
    assert result['motor_register_writes'] == 1
    assert bus.events.count('goal_write') == 1
    assert result['normal_validation_error']
    assert follow['original_target_arrival']['max_abs_residual'] == pytest.approx(2.6)
