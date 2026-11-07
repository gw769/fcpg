"""Independent fake-bus checks for waiting budgets and baseline restoration."""
from copy import deepcopy

import pytest

from test_v21_arm_position_servo import J3, change_pose, fixture, rejected, run


def stopped_cleanup_fixture(fault=None):
    owner, execution = fixture()
    q = owner.bus.q()
    q[2], q[3] = .8, .1
    change_pose(owner, execution, q)
    owner.bus.steps[J3] = 0
    state = dict(injected=False, low_cap_reads=0, restored=False, post_reads=0)
    if fault == 'high_load':
        owner.quiet_load = 33

    def write(register, motor, value):
        if ((register, motor, value) == ('I_Coefficient', J3, 1)
                and not state['injected']):
            state['injected'] = True
            raise RuntimeError('injected execution failure for cleanup test')
        if ((register, motor, value) == ('Torque_Limit', J3, 1000)
                and owner.bus.after_halt):
            state['restored'] = True
            # The saved HOLD becomes effective once the original cap returns.
            owner.bus.values[J3]['Present_Position'] = owner.bus.values[J3]['Goal_Position']

    def read(register, motor):
        if register != 'Present_Position' or not owner.bus.after_halt:
            return
        values = owner.bus.values[J3]
        if values['Torque_Limit'] == 200 and not state['restored']:
            state['low_cap_reads'] += 1
            values['Present_Position'] += 5
            if fault == 'changed_goal':
                values['Goal_Position'] += 1
        elif values['Torque_Limit'] == 1000:
            state['post_reads'] += 1
            if fault == 'post_restore_motion' and state['post_reads'] == 2:
                values['Present_Position'] += 11

    owner.bus.on_write, owner.bus.on_read = write, read
    return owner, execution, state


def test_heavy_arm_returns_to_original_cap_before_low_cap_sag_accumulates():
    owner, execution, state = stopped_cleanup_fixture()
    before = deepcopy(execution)
    result = rejected(owner, execution, 'injected execution failure')
    cleanup = result['cleanup']
    assert state['injected'] and state['restored']
    assert state['low_cap_reads'] == 1
    assert cleanup['restored'] and cleanup['profile_state'] == 'baseline1000'
    assert cleanup['post_baseline_restore']['sample_count'] == 22
    assert cleanup['post_baseline_restore']['actual_span_s'] >= 1.
    assert cleanup['hold_verified'] and cleanup['recovery_by_motor'][J3]['integral_zero_verified']
    assert owner.bus.values[J3]['I_Coefficient'] == 0
    assert owner.bus.values[J3]['Torque_Limit'] == 1000
    assert not result['original_target_reached'] and not result['original_goal_held']
    assert execution == before and 'completion_follow' not in result


@pytest.mark.parametrize('fault', ['high_load', 'changed_goal', 'post_restore_motion'])
def test_cleanup_restoration_requires_verified_hold_low_load_and_post_restore_stability(fault):
    owner, execution, state = stopped_cleanup_fixture(fault)
    result = rejected(owner, execution, 'injected execution failure')
    cleanup = result['cleanup']
    assert state['injected'] and not cleanup['restored']
    assert cleanup['baseline_restore_errors']
    assert cleanup['profile_state'] == 'cleanup_cap200'
    assert owner.bus.values[J3]['I_Coefficient'] == 0
    assert owner.bus.values[J3]['Torque_Limit'] == 200
    assert state['restored'] is (fault == 'post_restore_motion')
    assert not result['original_target_reached'] and not result['selected_axes_completed']


@pytest.mark.parametrize(('after_i0_position', 'i0_delay', 'reached'), [
    (.10, 0., True),
    (.21, 0., False),
    (-.01, 0., False),  # Landing near target cannot excuse a >.75 actual step.
    (.10, .2, False),  # Verified I0 itself must finish inside the total budget.
])
def test_twenty_second_wait_is_followed_by_actual_i0_arrival_not_elapsed_time(
        after_i0_position, i0_delay, reached):
    from fcpg.v21_hardware_bridge import validate_arm_finishing_restoration, validate_arm_original_goal_hold

    owner, execution = fixture()
    q = owner.bus.q()
    q[2], q[3] = .8, .1
    change_pose(owner, execution, q)
    owner.bus.steps[J3] = 0
    before = deepcopy(execution)
    state = dict(pulses=0, cleared=False)

    def write(register, motor, value):
        if motor != J3 or register != 'I_Coefficient':
            return
        if value == 1:
            state['pulses'] += 1
        elif state['pulses'] == 1 and not state['cleared'] and not owner.stop_event.is_set():
            state['cleared'] = True
            owner.clock.now += i0_delay
            owner.bus.values[J3]['Present_Position'] = round(10000 + 100 * after_i0_position)

    owner.bus.on_write = write
    result = run(owner, execution) if reached else rejected(owner, execution)
    axis = result['axes'][J3]
    candidate = axis['stages']['candidate']
    exposure = axis['integral_exposure']
    assert candidate['sample_count'] > 200 and candidate['actual_span_s'] > 19.
    assert candidate['stop_reason'] == 'i1_observation_budget_exhausted'
    assert exposure['maximum_s'] == 20. and exposure['all_pulses_i0_verified']
    assert exposure['within_limit'] is (i0_delay == 0.)
    assert state == dict(pulses=1, cleared=True)
    assert execution == before and owner.bus.values[J3]['I_Coefficient'] == 0
    assert result['original_goal_held'] is reached
    if reached:
        settled = axis['stages']['integral_off_settling']
        assert settled['axis_arrival']['reached']
        assert all(row['started_s'] > candidate['integral_zero_verified_s']
                   and row['parameters']['I_Coefficient'] == 0 for row in settled['readbacks'])
        assert result['original_target_arrival']['residual_by_axis'][2] == pytest.approx(.1)
        validate_arm_finishing_restoration(result)
        validate_arm_original_goal_hold(result, result['completion_follow'],
                                        execution['commands_position'][-1])
    else:
        assert result['cleanup']['hold_performed'] and 'completion_follow' not in result
