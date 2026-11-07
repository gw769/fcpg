"""HOME configuration recovery uses only fake registers and an advancing clock."""
from copy import deepcopy
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from fcpg.v21_position_servo_recovery import PositionServoRecoveryError, restore_failed_position_servo_baseline
from fcpg.v21_same_goal_integral import ORDER
from test_v21_arm_position_servo import fixture, rejected


J3 = ORDER[2]


def setup(tmp_path, *, final_profile='cleanup_cap200'):
    assert final_profile in ('cleanup_cap200', 'baseline1000')
    owner, source = fixture()
    def fail_i1(register, motor, value):
        if register == 'I_Coefficient' and value == 1:
            raise RuntimeError('injected diagnostic readback failure')
        if (final_profile == 'cleanup_cap200'
                and (register, motor, value) == ('Torque_Limit', J3, 1000)):
            # This suite exercises the separate HOME recovery path. Force the
            # controller cleanup's post-restore proof to fail so its verified
            # final profile is cap200 rather than the ordinary baseline1000.
            owner.bus.values[J3]['Present_Load'] = 33
    owner.bus.on_write = fail_i1
    source['position_servo'] = rejected(owner, source)
    source['requires_home_after_incomplete_execution'] = True
    owner.bus.on_write = None
    owner.bus.events.clear()
    original_halt = owner._halt
    def evidenced_halt():
        value = original_halt()
        value.update(event='safe_hold_verified', reason='v21_emergency_stop',
            reused_existing_hold=False, motor_register_writes=1)
        return value
    owner._halt = evidenced_halt
    for name in ORDER:
        owner.bus.values[name].update(Status=0, Present_Velocity=0, Present_Load=0)
    owner.runtime.stopped = True
    owner.runtime.executing = False
    owner.runtime.pending = None
    owner.runtime.requires_home_after_incomplete_execution = True
    owner.lease = object()
    service = SimpleNamespace(owner=owner, clock=owner.clock, output=tmp_path, last_attempt=source,
        phase='home', operation_lock=threading.Lock(), epoch=0, _trial_stop_generation=0)
    service.operation_lock.acquire()
    service.ros_control_admission_epoch = lambda:service.epoch
    service._require_ros_control_admission = lambda:None
    return service


def run(service):
    return restore_failed_position_servo_baseline(service, wait=service.owner.clock.wait)


def fail(service, match=None):
    with pytest.raises(PositionServoRecoveryError, match=match) as caught:
        run(service)
    result = caught.value.report
    json.dumps(result, allow_nan=False)
    assert result['completed'] is False
    assert result['configuration_restored'] is False
    assert service.owner.stop_event.is_set()
    assert service.owner.runtime.requires_home_after_incomplete_execution is True
    return result


def writes(service):
    return [(row['motor'], row['register'], row['value']) for row in service.owner.bus.events
            if row['kind'] == 'write']


def test_exact_source_caps_only_and_full_actual_windows_keep_model_and_home_interlock(tmp_path):
    service = setup(tmp_path)
    before = deepcopy(service.last_attempt)
    history = service.owner.runtime.history
    old_halts = service.owner.halts
    result = run(service)
    assert result['completed'] and result['configuration_restored']
    assert writes(service) == [(J3, 'Torque_Limit', 1000)]
    assert service.last_attempt == before
    assert service.owner.runtime.history is history
    assert service.owner.halts == old_halts
    assert all(row['kind'] != 'halt' for row in service.owner.bus.events)
    assert all(row['stopped'] for row in service.owner.bus.events)
    assert service.owner.runtime.requires_home_after_incomplete_execution is True
    for phase in ('pre_restore', 'post_restore'):
        assert result[phase]['sample_count'] == 22
        assert result[phase]['actual_span_s'] >= 1.
        assert result[phase]['completed'] is True
    assert result['final_parameters_by_motor'][ORDER[-1]]['Torque_Limit'] == 500
    assert result['final_parameters_by_motor'][J3]['Torque_Limit'] == 1000
    assert json.loads(next(tmp_path.glob('position-servo-recovery-*.json')).read_text()) == result


def test_already_verified_baseline_cleanup_has_no_home_parameter_write(tmp_path):
    service = setup(tmp_path, final_profile='baseline1000')
    assert service.last_attempt['position_servo']['cleanup']['profile_state'] == 'baseline1000'
    assert run(service) is None
    assert writes(service) == []


def test_same_output_file_source_is_read_without_overwriting_failed_evidence(tmp_path):
    service = setup(tmp_path)
    path = tmp_path/'last-attempt.json'
    path.write_text(json.dumps(service.last_attempt))
    before = path.read_bytes()
    service.last_attempt = None
    assert run(service)['source_kind'] == 'same_output/last-attempt.json'
    assert path.read_bytes() == before


@pytest.mark.parametrize('error_cells', [1, 3, 6])
def test_stable_deadband_present_error_keeps_original_goal(tmp_path, error_cells):
    service = setup(tmp_path)
    bus = service.owner.bus
    goal = bus.values[J3]['Goal_Position']
    bus.values[J3]['Present_Position'] = goal+error_cells
    result = run(service)
    assert result['completed']
    assert bus.values[J3]['Goal_Position'] == goal
    assert all(row['max_abs_raw_goal_error'] == error_cells for row in result['readbacks'])


def test_one_pre_restore_packet_drop_reacquires_complete_snapshot_before_any_write(tmp_path):
    service = setup(tmp_path)
    bus = service.owner.bus
    dropped = []
    def on_read(register, motor):
        if register == 'Present_Position' and motor == 'all' and not dropped:
            dropped.append(True)
            assert writes(service) == []
            raise ConnectionError('injected pre-restore Present_Position packet timeout')
    bus.on_read = on_read

    result = run(service)

    assert result['completed'] and dropped
    stage = result['pre_restore']
    assert stage['maximum_snapshot_attempts'] == 4
    assert stage['retry_policy'] == 'complete_snapshot_reacquisition_after_transport_connection_error'
    assert len(stage['transport_retries']) == 1
    retry = stage['transport_retries'][0]
    assert retry['failed_register'] == 'Present_Position' and retry['retry_started'] is True
    attempts = [row for row in result['readbacks']
        if row['phase'] == 'pre_restore' and row['sample_index'] == 1]
    assert [row['sample_attempt'] for row in attempts] == [1, 2]
    assert attempts[0]['discarded'] and not attempts[0]['completed']
    assert attempts[1]['completed'] and not attempts[1]['discarded']
    assert list(attempts[0]['register_reads']) == ['Present_Position']
    assert attempts[1]['scheduled_start_s'] == pytest.approx(attempts[0]['failed_s']+.05)
    assert all(event['num_retry'] == 0 for row in attempts
               for event in row['register_reads'].values())
    assert writes(service) == [(J3, 'Torque_Limit', 1000)]


def test_two_pre_restore_packet_drops_reacquire_until_complete(tmp_path):
    service = setup(tmp_path)
    bus = service.owner.bus
    dropped = 0
    def on_read(register, motor):
        nonlocal dropped
        if register == 'Present_Position' and motor == 'all' and dropped < 2:
            dropped += 1
            assert writes(service) == []
            raise ConnectionError('injected repeated Present_Position packet timeout')
    bus.on_read = on_read

    result = run(service)

    assert dropped == 2
    stage = result['pre_restore']
    assert result['completed'] is True
    assert stage['sample_count'] == 22 and len(stage['transport_retries']) == 2
    assert stage['transport_retries'][0]['retry_started'] is True
    assert stage['transport_retries'][1]['retry_started'] is True
    attempts = [row for row in result['readbacks']
                if row['phase'] == 'pre_restore' and row['sample_index'] == 1]
    assert [row['sample_attempt'] for row in attempts] == [1, 2, 3]
    assert all(row['failed'] and row['discarded']
               and row['failed_register'] == 'Present_Position' for row in attempts[:2])
    assert attempts[2]['completed'] and not attempts[2]['discarded']
    assert writes(service) == [(J3, 'Torque_Limit', 1000)]


def test_four_pre_restore_packet_drops_exhaust_retry_and_fail_closed(tmp_path):
    service = setup(tmp_path)
    bus = service.owner.bus
    dropped = 0
    def on_read(register, motor):
        nonlocal dropped
        if register == 'Present_Position' and motor == 'all' and dropped < 4:
            dropped += 1
            assert writes(service) == []
            raise ConnectionError('injected repeated Present_Position packet timeout')
    bus.on_read = on_read

    result = fail(service, 'packet timeout')

    assert dropped == 4
    stage = result['pre_restore']
    assert stage['sample_count'] == 0 and len(stage['transport_retries']) == 4
    assert [row['retry_started'] for row in stage['transport_retries']] == [True, True, True, False]
    attempts = [row for row in result['readbacks'] if row['phase'] == 'pre_restore']
    assert [row['sample_attempt'] for row in attempts] == [1, 2, 3, 4]
    assert all(row['failed'] and row['discarded']
               and row['failed_register'] == 'Present_Position' for row in attempts)
    assert writes(service) == [(J3, 'Torque_Limit', 200)]
    assert result['cleanup']['cap200_verified'][J3] is True
    assert result['cleanup']['hold_verified'] is True


@pytest.mark.parametrize('error_cells,accepted', [(6, True), (7, False)])
def test_real_j3_scale_accepts_six_cells_but_rejects_seven(tmp_path, error_cells, accepted):
    service = setup(tmp_path)
    bus = service.owner.bus
    original_normalize, original_unnormalize = bus._normalize, bus._unnormalize
    bus._normalize = lambda values:{
        key:(value-10000)/11.27 if key == 3 else original_normalize({key:value})[key]
        for key,value in values.items()}
    bus._unnormalize = lambda values:{
        key:round(10000+11.27*value) if key == 3 else original_unnormalize({key:value})[key]
        for key,value in values.items()}
    goal = bus.values[J3]['Goal_Position']
    bus.values[J3]['Present_Position'] = goal+error_cells
    if accepted:
        result = run(service)
        assert result['completed']
        assert result['readbacks'][0]['max_abs_goal_error'] == pytest.approx(error_cells/11.27)
    else:
        fail(service, 'six-tick/.6')


def test_retry_uses_latest_verified_failed_recovery_hold(tmp_path):
    service = setup(tmp_path)
    bus = service.owner.bus
    original_goal = bus.values[J3]['Goal_Position']
    bus.values[J3]['Present_Position'] = original_goal+7
    first = fail(service, 'six-tick/.6')
    latest_goal = first['cleanup']['hold']['attempts'][-1]['goal_readback_raw'][J3]
    assert latest_goal == original_goal+7

    result = run(service)

    assert result['completed'] and result['frozen_hold_goal_raw'][J3] == latest_goal
    assert result['original_failed_cleanup_hold_goal_raw'][J3] == original_goal
    assert result['recovery_hold_source'].endswith('.json')
    assert bus.values[J3]['Goal_Position'] == latest_goal


def test_old_final_parameters_are_never_used_as_current_readback(tmp_path):
    service = setup(tmp_path)
    service.last_attempt['position_servo']['final_parameters_by_motor'] = {'fake':'ignored'}
    assert run(service)['completed']
    assert writes(service) == [(J3, 'Torque_Limit', 1000)]


@pytest.mark.parametrize('change', [
    lambda s:s['position_servo']['touched_motors'].append(ORDER[5]),
    lambda s:s['position_servo']['cleanup']['recovery_by_motor'][J3].update(integral_zero_verified=False),
    lambda s:s['position_servo']['baseline_parameters_by_motor'][ORDER[5]].update(Torque_Limit=1000),
    lambda s:s['position_servo']['axes'][J3]['identity'].update(Lock=0),
    lambda s:s['position_servo']['parameter_writes'][-1].update(verified=False),
    lambda s:s['position_servo']['cleanup']['hold']['attempts'][-1].update(goal_register_verified=False),
])
def test_bad_provenance_rejects_without_register_write(tmp_path, change):
    service = setup(tmp_path)
    change(service.last_attempt)
    fail(service)
    assert writes(service) == []


@pytest.mark.parametrize('motor, register, value', [
    (J3, 'Present_Position', None), (J3, 'I_Coefficient', 1),
    (J3, 'Present_Load', 33), (ORDER[1], 'Present_Load', 33),
    (ORDER[0], 'Moving', 1), (J3, 'Status', 1),
    (J3, 'Present_Velocity', 1), (ORDER[5], 'Torque_Limit', 1000),
    (ORDER[0], 'P_Coefficient', 32),
])
def test_unsafe_live_state_never_increases_cap(tmp_path, motor, register, value):
    service = setup(tmp_path)
    service.owner.bus.values[motor][register] = (service.owner.bus.values[motor][register]+7
                                                if value is None else value)
    result = fail(service)
    assert writes(service) == [(J3, 'Torque_Limit', 200)]
    assert result['cleanup']['cap200_verified'][J3] is True
    assert result['cleanup']['hold_verified'] is True


@pytest.mark.parametrize('stop_kind', ['epoch', 'generation'])
def test_stop_during_pre_window_does_not_clear_existing_stop(tmp_path, stop_kind):
    service = setup(tmp_path)
    def stop():
        service.owner.clock.on_wait = None
        if stop_kind == 'epoch':
            service.epoch += 1
        else:
            service._trial_stop_generation += 1
    service.owner.clock.on_wait = stop
    result = fail(service, 'cancelled')
    assert writes(service) == [(J3, 'Torque_Limit', 200)]
    assert 'post_restore' not in result


@pytest.mark.parametrize('when', ['write_return', 'readback_return'])
def test_stop_while_cap_write_or_readback_returns_immediately_rolls_back(tmp_path, when):
    service = setup(tmp_path)
    bus = service.owner.bus
    def on_write(register, motor, value):
        if value == 1000 and when == 'write_return':
            service.epoch += 1
    def on_read(register, motor):
        if (when == 'readback_return' and register == 'Torque_Limit' and motor == J3
                and bus.values[J3]['Torque_Limit'] == 1000):
            service.epoch += 1
    bus.on_write, bus.on_read = on_write, on_read
    result = fail(service, 'cancelled')
    assert writes(service) == [(J3, 'Torque_Limit', 1000), (J3, 'Torque_Limit', 200)]
    assert 'post_restore' not in result
    assert result['cleanup']['hold_verified']


def test_load_rise_after_cap_write_rolls_back_and_holds(tmp_path):
    service = setup(tmp_path)
    def on_write(register, motor, value):
        if value == 1000:
            service.owner.bus.values[J3]['Present_Load'] = 80
    service.owner.bus.on_write = on_write
    result = fail(service, 'low-load')
    assert writes(service) == [(J3, 'Torque_Limit', 1000), (J3, 'Torque_Limit', 200)]
    assert result['post_restore']['completed'] is False


def test_delayed_real_sample_fails_without_repeating_window(tmp_path):
    service = setup(tmp_path)
    def delay():
        service.owner.clock.on_wait = None
        service.owner.clock.now += .09
    service.owner.clock.on_wait = delay
    result = fail(service, 'deadline')
    assert result['pre_restore']['sample_count'] == 0
    assert writes(service) == [(J3, 'Torque_Limit', 200)]


@pytest.mark.parametrize('boundary', ['within_window', 'across_cap'])
def test_alternating_one_tick_error_cannot_hide_two_tick_actual_motion(tmp_path, boundary):
    service = setup(tmp_path)
    bus = service.owner.bus
    # Match the real J3 scale: one tick is .088731..., two ticks exceeds .1.
    original_normalize, original_unnormalize = bus._normalize, bus._unnormalize
    bus._normalize = lambda values:{key:(value-10000)/11.27 if key == 3 else original_normalize({key:value})[key]
                                   for key,value in values.items()}
    bus._unnormalize = lambda values:{key:round(10000+11.27*value) if key == 3 else original_unnormalize({key:value})[key]
                                     for key,value in values.items()}
    goal = bus.values[J3]['Goal_Position']
    bus.values[J3]['Present_Position'] = goal-1
    seen = 0
    def on_read(register, motor):
        nonlocal seen
        if register == 'Present_Position':
            seen += 1
            if boundary == 'within_window' and seen == 2:
                bus.values[J3]['Present_Position'] = goal+1
    def on_write(register, motor, value):
        if boundary == 'across_cap' and value == 1000:
            bus.values[J3]['Present_Position'] = goal+1
    bus.on_read, bus.on_write = on_read, on_write
    fail(service, 'step exceeds')
    expected = [(J3, 'Torque_Limit', 200)]
    if boundary == 'across_cap':
        expected.insert(0, (J3, 'Torque_Limit', 1000))
    assert writes(service) == expected


def test_source_changed_during_window_rejects_cap_increase(tmp_path):
    service = setup(tmp_path)
    def change():
        service.owner.clock.on_wait = None
        service.last_attempt['source_time_s'] = -1.
    service.owner.clock.on_wait = change
    fail(service, 'source changed')
    assert writes(service) == [(J3, 'Torque_Limit', 200)]


def test_no_relevant_failure_is_a_noop(tmp_path):
    service = setup(tmp_path)
    service.last_attempt.pop('position_servo')
    assert run(service) is None
    assert not service.owner.bus.events
