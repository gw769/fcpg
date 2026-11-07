"""Active bridge finishing over fake registers only; no hardware construction."""
from copy import deepcopy
import json
import threading

import numpy as np
import pytest

from fcpg.v21_j3_position_servo import finish_original_j3_goal, J3PositionServoError
from fcpg.v21_position_arrival import evaluate_position_arrival
from test_v21_same_goal_integral import Owner, MOTOR, ORDER, AXIS, source_execution, writes


def fixture():
    owner = Owner()
    # The standalone diagnostic fake only lowers load after HOLD. Normal
    # finishing never HOLDs; simulate a quiet servo independently of HOLD.
    owner.bus.values[MOTOR]['Present_Load'] = 0
    execution = source_execution()
    execution.update(completed=False, action_source='v21_lewm_cem', source_time_s=execution['readback_times_s'][0])
    for receipt in execution['policy_goal_readbacks']:
        receipt.update(read_attempted=True, read_returned=True)
    owner.runtime.stopped = False
    owner.runtime.executing = True
    owner.runtime.pending = object()
    owner.bridge.last_attempt = execution
    owner.bridge.lock = threading.Lock()
    owner.bridge.lock.acquire()
    owner.io.settle_samples = 3
    owner.io.settle_position_tolerance = 3.
    owner.io.settle_step_tolerance = .25
    def no_position(*args, **kwargs):
        pytest.fail('position servo must never issue or replay a Goal')
    owner._write_q = no_position
    return owner, execution


def run(owner, execution):
    return finish_original_j3_goal(owner, deepcopy(execution), enabled=True,
                                  clock=owner.clock, wait=owner.clock.wait)


def rejected(owner, execution, match=None):
    with pytest.raises(J3PositionServoError, match=match) as caught:
        run(owner, execution)
    json.dumps(caught.value.report, allow_nan=False)
    return caught.value.report


def test_default_disabled_touches_no_owner_or_clock():
    class Forbidden:
        def __getattr__(self, name):
            pytest.fail('disabled helper touched owner '+name)
    def forbidden():
        pytest.fail('disabled helper touched clock')
    with pytest.raises(J3PositionServoError, match='disabled; no IO'):
        finish_original_j3_goal(Forbidden(), clock=forbidden)


@pytest.mark.parametrize('damage', ['idle', 'stopped', 'no_pending', 'unlocked', 'foreign_runtime',
    'completed', 'goal_count', 'goal_mismatch', 'short_follow', 'wrong_target', 'diagnostic'])
def test_only_the_current_active_five_command_attempt_is_admitted(damage):
    owner, execution = fixture()
    if damage == 'idle':
        owner.runtime.executing = False
    elif damage == 'stopped':
        owner.runtime.stopped = True
    elif damage == 'no_pending':
        owner.runtime.pending = None
    elif damage == 'unlocked':
        owner.bridge.lock.release()
    elif damage == 'foreign_runtime':
        owner.bridge.runtime = object()
    elif damage == 'completed':
        execution['completed'] = True
    elif damage == 'goal_count':
        execution['policy_goal_registers_verified'] = 4
    elif damage == 'goal_mismatch':
        execution['policy_goal_readbacks'][0]['goal_raw'][MOTOR] += 1
    elif damage == 'short_follow':
        execution['normal_follow']['positions'].pop()
    elif damage == 'wrong_target':
        execution['original_target_arrival']['original_target'][0] += .01
    elif damage == 'diagnostic':
        execution['action_source'] = 'j3_native_response_diagnostic'
    report = rejected(owner, execution)
    assert not owner.bus.events and not report['applied']
    assert not report['cleanup']['restored']


def test_old_execution_cannot_be_substituted_for_the_active_attempt():
    owner, execution = fixture(); supplied = deepcopy(execution)
    supplied['normal_follow']['positions'][5][0] += .01
    report = rejected(owner, supplied, 'not the bridge current execution')
    assert not owner.bus.events and not report['applied']


def test_success_finishes_original_goal_without_stopping_holding_or_changing_five_y():
    owner, execution = fixture(); saved = deepcopy(execution); pending = owner.runtime.pending
    report = run(owner, execution)
    json.dumps(report, allow_nan=False)
    assert report['schema'] == 'v21-j3-position-servo-v1'
    assert report['completed'] and report['applied'] and not report['skipped']
    assert report['original_target_reached'] and report['cleanup']['restored']
    assert report['original_goal_preserved'] and not report['cleanup']['hold_performed']
    assert not owner.stop_event.is_set() and not owner.halts
    assert owner.runtime.pending is pending and owner.runtime.executing and not owner.runtime.stopped
    assert owner.runtime.history == ['untouched'] and execution == saved
    assert report['policy_commands_added'] == 0 and not report['model_feedback_label']
    assert not [event for event in owner.bus.events if event['kind'] in ('position', 'halt')]
    assert writes(owner) == [('Torque_Limit', 200), ('D_Coefficient', 32), ('I_Coefficient', 1),
        ('I_Coefficient', 0), ('Torque_Limit', 200), ('D_Coefficient', 0), ('Torque_Limit', 1000)]
    assert all(owner.bus.values[name]['Goal_Position'] == report['original_target_raw'][name] for name in ORDER)
    follow = report['completion_follow']
    assert follow['schema'] == 'v21-position-servo-completion-follow-v1'
    assert follow['baseline_restored'] and follow['hold_eligible'] and follow['goal_reached']
    assert len(follow['positions']) == len(follow['times_s']) == 9
    assert 'budget_s' not in follow
    assert np.all(np.diff(follow['times_s']) > 0)
    expected = evaluate_position_arrival(report['original_target'], follow['positions'],
        np.asarray(follow['times_s'])-execution['source_time_s'], position_tolerance=.2, step_tolerance=.1, quiet_samples=3)
    for key, value in expected.items():
        assert report['original_target_arrival'][key] == value
    retained = report['stages']['original_goal_retention']['readbacks']
    final = report['stages']['restored_original_goal']['readbacks']
    assert len(retained) == 6 and len(final) == 3
    assert all(row['parameters']['Torque_Limit'] == 200 for row in retained)
    assert all(row['parameters']['Torque_Limit'] == 1000 for row in final)
    assert report['integral_exposure']['within_limit']
    assert report['finished_s'] >= report['started_s']


def test_final_arrival_comes_from_post_cap1000_present_not_i1_exact_goal():
    owner, execution = fixture()
    def change_after_restore(register, value):
        if (register, value) == ('Torque_Limit', 1000):
            owner.bus.values[MOTOR]['Present_Position'] = 10013
    owner.bus.on_write = change_after_restore
    result = run(owner, execution)
    assert result['stages']['candidate']['original_target_arrival']['actual_position'][AXIS] == 0.
    assert result['original_target_arrival']['actual_position'][AXIS] == .13
    assert result['original_target_reached'] and result['original_goal_preserved']
    assert owner.bus.values[MOTOR]['Goal_Position'] == 10000 and not owner.halts


@pytest.mark.parametrize('kind', ['already_reached', 'non_j3', 'too_far', 'unstable'])
def test_nonapplicable_pose_returns_true_fresh_evidence_without_parameter_changes(kind):
    owner, execution = fixture()
    tail = execution['normal_follow']['positions']
    q = np.asarray(tail[-1]).copy()
    if kind == 'already_reached':
        q[AXIS] = .1
    elif kind == 'non_j3':
        q[0] = .4
    elif kind == 'too_far':
        q[AXIS] = 3.2
    for row in tail:
        row[:] = q.tolist()
    if kind == 'unstable':
        tail[-2][AXIS] += .12
    owner.bus.set_q(q)
    result = run(owner, execution)
    assert result['completed'] and not result['applied'] and result['skipped']
    assert result['cleanup']['restored'] and result['cleanup']['no_mutation_attempted']
    assert result['original_goal_preserved'] and not writes(owner) and not owner.halts
    assert not owner.stop_event.is_set() and owner.runtime.executing
    assert result['original_target_reached'] is (kind == 'already_reached')
    assert len(result['completion_follow']['positions']) == 4
    assert result['completion_follow']['positions'][-1] == owner.bus.q().tolist()
    assert result['reason']
    assert result['finished_s'] >= result['started_s']


@pytest.mark.parametrize('register,value', [('Max_Torque_Limit', 900), ('Firmware_Minor_Version', 11)])
def test_geometry_cannot_leave_stale_identity_or_maximum_torque_evidence(register, value):
    owner, execution = fixture()
    def changed():
        if len(owner.paths) == 10:
            owner.bus.values[MOTOR][register] = value
    owner.on_geometry = changed
    result = rejected(owner, execution)
    assert not writes(owner) and owner.stop_event.is_set() and owner.halts
    assert not result['cleanup']['restored']


@pytest.mark.parametrize('damage', ['wrong_goal', 'wrong_i', 'wrong_cap', 'fresh_motion', 'geometry'])
def test_fresh_admission_state_failures_do_not_silently_skip(damage):
    owner, execution = fixture()
    if damage == 'wrong_goal':
        owner.bus.values[MOTOR]['Goal_Position'] += 1
    elif damage == 'wrong_i':
        owner.bus.values[MOTOR]['I_Coefficient'] = 1
    elif damage == 'wrong_cap':
        owner.bus.values[MOTOR]['Torque_Limit'] = 200
    elif damage == 'fresh_motion':
        owner.bus.values[MOTOR]['Present_Position'] += 26
    else:
        owner.on_geometry = lambda: (_ for _ in ()).throw(RuntimeError('fake geometry rejected'))
    result = rejected(owner, execution)
    assert not writes(owner) and not result['cleanup']['restored']
    assert owner.stop_event.is_set() and owner.halts


@pytest.mark.parametrize('phase', ['candidate', 'retention', 'restored'])
def test_stop_at_each_live_phase_clears_i_and_holds_without_releasing_cap(phase):
    owner, execution = fixture(); cleared = []; injected = []
    def written(register, value):
        if (register, value) == ('I_Coefficient', 0):
            cleared.append(True)
    def stop(register, motor):
        if register != 'Present_Position' or injected:
            return
        values = owner.bus.values[MOTOR]
        active = ((phase == 'candidate' and values['I_Coefficient'] == 1)
            or (phase == 'retention' and cleared and values['D_Coefficient'] == 0 and values['Torque_Limit'] == 200)
            or (phase == 'restored' and cleared and values['Torque_Limit'] == 1000))
        if active:
            injected.append(True)
            owner.stop_event.set()
    owner.bus.on_write = written
    owner.bus.on_read = stop
    result = rejected(owner, execution, 'STOP')
    assert owner.stop_event.is_set() and result['cleanup']['hold_performed']
    assert result['cleanup']['integral_zero_verified'] and not result['cleanup']['restored']
    assert owner.bus.values[MOTOR]['Torque_Limit'] == 200
    assert 'completion_follow' not in result
    assert result['finished_s'] >= result['started_s']


@pytest.mark.parametrize('failure', ['i1_ack', 'i0_ack', 'cap_ack', 'd0_ack', 'not_reached', 'final_loss', 'final_load', 'retention_loss', 'late'])
def test_mutated_failure_restores_i_independently_and_leaves_low_cap_with_hold(failure):
    owner, execution = fixture(); cleared = []; injected = []
    if failure == 'not_reached':
        owner.bus.motion_step = 0
    def written(register, value):
        if (register, value) == ('I_Coefficient', 0):
            cleared.append(True)
        if failure == 'i1_ack' and (register, value) == ('I_Coefficient', 1):
            raise ConnectionError('I1 ACK lost')
        if failure == 'i0_ack' and (register, value) == ('I_Coefficient', 0):
            raise ConnectionError('I0 ACK lost')
        if failure == 'cap_ack' and cleared and (register, value) == ('Torque_Limit', 200):
            raise ConnectionError('cap ACK lost')
        if failure == 'd0_ack' and (register, value) == ('D_Coefficient', 0):
            raise ConnectionError('D0 ACK lost')
        if (register, value) == ('Torque_Limit', 1000):
            if failure == 'final_loss':
                owner.bus.values[MOTOR]['Present_Position'] = 10024
            elif failure == 'final_load':
                owner.bus.values[MOTOR]['Present_Load'] = 33
    def read(register, motor):
        if register == 'Present_Position' and not injected:
            if failure == 'retention_loss' and cleared and owner.bus.values[MOTOR]['D_Coefficient'] == 0:
                injected.append(True)
                owner.bus.values[MOTOR]['Present_Position'] = 10024
            elif failure == 'late' and owner.bus.values[MOTOR]['I_Coefficient'] == 1:
                injected.append(True)
                owner.clock.now += .031
    owner.bus.on_write = written
    owner.bus.on_read = read
    result = rejected(owner, execution)
    assert not result['cleanup']['restored'] and result['cleanup']['hold_performed']
    assert owner.bus.values[MOTOR]['Torque_Limit'] == 200
    assert owner.stop_event.is_set() and owner.halts and not result['original_goal_preserved']
    assert 'completion_follow' not in result
    assert result['finished_s'] >= result['started_s']
    if failure in ('final_loss', 'final_load'):
        assert result['stages']['original_goal_retention']['completed']
        assert ('Torque_Limit', 1000) in writes(owner)
        assert result['stages']['restored_original_goal']['sample_count'] >= 1
    elif failure == 'retention_loss':
        assert result['stages']['candidate']['original_target_reached']
        assert result['stages']['original_goal_retention']['completed']
        assert not result['original_goal_retained']
        assert ('Torque_Limit', 1000) not in writes(owner)
    if failure == 'i0_ack':
        assert not result['cleanup']['integral_zero_verified']
        assert owner.bus.values[MOTOR]['D_Coefficient'] == 32
        assert ('D_Coefficient', 0) not in writes(owner)
    else:
        assert result['cleanup']['integral_zero_verified']
