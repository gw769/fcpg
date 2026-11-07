"""The prepared diagnostic shares transport but cannot fabricate model history."""
import numpy as np
import pytest

from test_v21_hardware_bridge import setup
from test_v21_policy_goal_readback import Bus, attach


def diagnostic(*, slow_write=False, cancel_at=None):
    bridge, runtime, writes, holds = setup(slow_write=slow_write, cancel_at=cancel_at)
    runtime.stop()
    runtime.executing = False
    runtime.state = object()
    bridge.normal_follow_budget_s = .4
    bus = Bus()
    attach(bridge, bus)
    source = np.zeros(6, np.float32)
    commands = np.repeat(source[None], 5, axis=0)
    commands[:, 2] = [-1.5, -3, -3, -3, -3]
    return bridge, runtime, writes, holds, bus, source, commands


def test_default_diagnostic_invokes_no_callbacks_even_with_invalid_input():
    bridge, runtime, writes, holds, bus, _, _ = diagnostic()
    def forbidden(*args, **kwargs):
        pytest.fail('default diagnostic must not inspect hardware')
    bridge.clock = bridge.read_q = bridge.cancelled = bridge.physical_check = forbidden
    record = bridge.execute_fixed_diagnostic(None, None)
    assert record['dry_run'] and record['callbacks_invoked'] == 0
    assert not writes and not holds and not bus.reads


def test_diagnostic_records_actual_five_samples_without_model_or_camera_calls():
    bridge, runtime, writes, holds, bus, source, commands = diagnostic()
    state = runtime.state
    def forbidden(*args, **kwargs):
        pytest.fail('diagnostic must not invoke model or camera')
    runtime.begin_execution = runtime.complete = bridge.capture_endpoint = forbidden
    bridge._sample_dynamic_actuators = forbidden
    record = bridge.execute_fixed_diagnostic(commands, source, enable=True)
    assert record['completed'] and len(writes) == len(bus.reads) == 5
    assert record['action_source'] == 'j3_pid_diagnostic'
    assert record['not_model_transition'] and not record['model_feedback_label']
    assert 'predicted_q_path' not in record and 'history_chunks' not in record
    assert record['measured_q_path'] == [source.tolist()] * 6
    assert record['readback_times_s'] == pytest.approx([10 + i * .1 for i in range(6)])
    assert record['command_times_s'] == pytest.approx([10 + i * .1 for i in range(5)])
    assert record['write_lateness_s'] == pytest.approx([.002] * 5)
    assert record['policy_goal_registers_verified'] == 5
    # Weak response cannot exit following early simply because target error <=3.
    assert len(record['normal_follow']['positions']) == 8
    assert not record['normal_follow']['cem_response_satisfied']
    assert record['protective_hold_after_q5'] and holds
    assert runtime.state is state and runtime.stopped and runtime.pending is None
    from scripts.prepare_v21_j3_pid_protocol import J3PIDExperiment
    J3PIDExperiment._validate_drive(record, {'q_position': source.tolist(), 'q_time_s': 10.}, commands.tolist(),
        callback_started_s=10., callback_returned_s=bridge.clock())


@pytest.mark.parametrize('column', range(6))
def test_fixed_sequence_refuses_any_other_motion_before_io(column):
    bridge, _, writes, holds, bus, source, commands = diagnostic()
    commands[2, column] += .1
    with pytest.raises(ValueError, match='fixed J3'):
        bridge.execute_fixed_diagnostic(commands, source, enable=True)
    assert not writes and not holds and not bus.reads


@pytest.mark.parametrize('condition', ('pending', 'executing', 'not_stopped', 'no_goals', 'no_follow'))
def test_invalid_admission_never_sends_or_freezes_someone_elses_operation(condition):
    bridge, runtime, writes, holds, bus, source, commands = diagnostic()
    if condition == 'pending': runtime.pending = object()
    if condition == 'executing': runtime.executing = True
    if condition == 'not_stopped': runtime.stopped = False
    if condition == 'no_goals': bridge.verify_policy_goal = None
    if condition == 'no_follow': bridge.normal_follow_budget_s = 0
    with pytest.raises(RuntimeError):
        bridge.execute_fixed_diagnostic(commands, source, enable=True)
    assert not writes and not holds and not bus.reads


def test_stall_geometry_failure_refuses_motion():
    bridge, _, writes, holds, _, source, commands = diagnostic()
    def gate(q, candidate):
        array = np.asarray(candidate)
        if np.all(array[..., 2] == 0):
            raise RuntimeError('J3 stall geometry collision')
    bridge.physical_check = gate
    with pytest.raises(RuntimeError, match='single-axis-stall') as failure:
        bridge.execute_fixed_diagnostic(commands, source, enable=True)
    assert not writes and holds
    assert failure.value.last_attempt['policy_commands_attempted'] == 0


def test_source_change_between_geometry_and_epoch_aborts_without_command():
    bridge, _, writes, holds, _, source, commands = diagnostic()
    reads = []
    def read():
        reads.append(None)
        q = source.copy()
        if len(reads) > 1: q[2] += .08
        return q
    bridge.read_q = read
    with pytest.raises(RuntimeError, match='source changed before first'):
        bridge.execute_fixed_diagnostic(commands, source, enable=True)
    assert not writes and holds


def test_partial_goal_mismatch_preserves_three_writes_and_original_error():
    bridge, runtime, writes, holds, bus, source, commands = diagnostic()
    bus.read_hook = lambda: setattr(bus, 'mismatch', len(bus.reads) == 3)
    with pytest.raises(RuntimeError, match='raw readback mismatch') as failure:
        bridge.execute_fixed_diagnostic(commands, source, enable=True)
    record = failure.value.last_attempt
    assert len(writes) == len(bus.reads) == 3 and holds
    assert record['policy_commands_attempted'] == record['policy_commands_returned'] == 3
    assert record['policy_goal_registers_verified'] == 2
    assert not record['completed'] and runtime.requires_home_after_incomplete_execution


def test_diagnostic_observation_window_is_full_point_four_even_for_strong_response():
    bridge, _, _, _, _, source, commands = diagnostic()
    current = source.copy()
    original_write = bridge.write_q
    def write(command, previous):
        value = original_write(command, previous)
        current[:] = value
        return value
    bridge.write_q = write
    bridge.read_q = lambda: current.copy()
    record = bridge.execute_fixed_diagnostic(commands, source, enable=True)
    assert record['completed'] and record['normal_follow']['cem_response_satisfied']
    assert len(record['normal_follow']['positions']) == 8
    assert record['normal_follow']['times_s'][-1] - record['readback_times_s'][0] == pytest.approx(.9)


def test_original_thirty_ms_deadline_still_aborts_after_one_slow_write():
    bridge, _, writes, holds, bus, source, commands = diagnostic(slow_write=True)
    with pytest.raises(RuntimeError, match='motor write missed') as failure:
        bridge.execute_fixed_diagnostic(commands, source, enable=True)
    assert len(writes) == 1 and not bus.reads and holds
    assert failure.value.last_attempt['deadline_miss']['allowed_lateness_s'] == .03


def test_operator_stop_keeps_partial_evidence_and_holds():
    bridge, _, writes, holds, bus, source, commands = diagnostic(cancel_at=3)
    with pytest.raises(RuntimeError, match='cancelled') as failure:
        bridge.execute_fixed_diagnostic(commands, source, enable=True)
    assert len(writes) == 3 and holds
    assert not failure.value.last_attempt['completed']


def range_diagnostic(*, amplitude=5, **kwargs):
    bridge, runtime, writes, holds, bus, source, _ = diagnostic(**kwargs)
    commands = np.repeat(source[None], 10, axis=0)
    commands[:, 2] = ([-2.5] + [-5.] * 9 if amplitude == 5
                      else [-2.5, -5., -7.5] + [-10.] * 7)
    return bridge, runtime, writes, holds, bus, source, commands


def test_default_range_diagnostic_has_no_hardware_or_runtime_callbacks():
    bridge, runtime, writes, holds, bus, _, _ = range_diagnostic()
    def forbidden(*args, **kwargs):
        pytest.fail('disabled range diagnostic touched runtime or hardware')
    bridge.clock = bridge.read_q = bridge.cancelled = bridge.physical_check = forbidden
    runtime.stop = forbidden
    record = bridge.execute_j3_range_diagnostic(None, None, amplitude=10, telemetry=forbidden)
    assert record['dry_run'] and record['callbacks_invoked'] == 0
    assert record['action_source'] == 'j3_range_diagnostic'
    assert not writes and not holds and not bus.reads
    with pytest.raises(TypeError):
        bridge.execute_j3_range_diagnostic(None, None, enable=1)


def test_range_diagnostic_records_ten_commands_goals_and_eleven_actual_samples():
    bridge, runtime, writes, holds, bus, source, commands = range_diagnostic()
    state = runtime.state
    def forbidden(*args, **kwargs):
        pytest.fail('range diagnostic must not invoke model or cameras')
    runtime.begin_execution = runtime.complete = bridge.capture_endpoint = forbidden
    record = bridge.execute_j3_range_diagnostic(commands, source, enable=True)
    assert record['completed'] and len(writes) == len(bus.reads) == 10
    assert record['action_source'] == 'j3_range_diagnostic'
    assert record['not_model_transition'] and not record['model_feedback_label']
    assert record['diagnostic_command_count'] == 10 and record['gain_writes'] == 0
    assert record['diagnostic_total_amplitude'] == 5.
    assert 'dynamic_actuator_readbacks' not in record
    assert 'predicted_q_path' not in record and 'history_chunks' not in record
    assert record['measured_q_path'] == [source.tolist()] * 11
    assert record['commands_position'] == commands.tolist()
    assert record['readback_times_s'] == pytest.approx([10 + i * .1 for i in range(11)])
    assert record['command_times_s'] == pytest.approx([10 + i * .1 for i in range(10)])
    assert record['write_lateness_s'] == pytest.approx([.002] * 10)
    assert record['policy_goal_reads_attempted'] == record['policy_goal_reads_returned'] == record['policy_goal_registers_verified'] == 10
    assert [entry['command_index'] for entry in record['policy_goal_readbacks']] == list(range(10))
    follow = record['normal_follow']
    assert len(follow['positions']) == 8 and follow['fixed_observation_window']
    assert follow['times_s'] == pytest.approx([11 + i * .05 for i in range(1, 9)])
    assert record['protective_hold_after_q10'] and 'protective_hold_after_q5' not in record
    assert holds == pytest.approx([11.4])
    assert record['endpoint_time_s'] >= follow['times_s'][-1]
    assert runtime.state is state and runtime.stopped and runtime.pending is None


@pytest.mark.parametrize('column', range(6))
def test_range_diagnostic_rejects_every_other_axis_or_j3_profile_before_io(column):
    bridge, _, writes, holds, bus, source, commands = range_diagnostic()
    commands[6, column] += .1
    with pytest.raises(ValueError, match='fixed ten-command J3'):
        bridge.execute_j3_range_diagnostic(commands, source, enable=True)
    assert not writes and not holds and not bus.reads


@pytest.mark.parametrize('fault', ('nine_commands', 'eleven_commands', 'nonfinite_command',
                                 'nonfinite_source', 'wrong_source_shape'))
def test_range_diagnostic_rejects_invalid_shape_or_nonfinite_inputs(fault):
    bridge, _, writes, holds, bus, source, commands = range_diagnostic()
    if fault == 'nine_commands': commands = commands[:9]
    elif fault == 'eleven_commands': commands = np.concatenate((commands, commands[-1:]))
    elif fault == 'nonfinite_command': commands[0, 2] = np.nan
    elif fault == 'nonfinite_source': source[1] = np.inf
    elif fault == 'wrong_source_shape': source = source[:5]
    with pytest.raises(ValueError, match='finite source'):
        bridge.execute_j3_range_diagnostic(commands, source, enable=True)
    assert not writes and not holds and not bus.reads


@pytest.mark.parametrize('condition', ('pending', 'executing', 'not_stopped', 'no_goals',
                                     'no_follow', 'changed_deadline', 'locked'))
def test_range_diagnostic_keeps_original_admission_before_motor_io(condition):
    bridge, runtime, writes, holds, bus, source, commands = range_diagnostic()
    if condition == 'pending': runtime.pending = object()
    elif condition == 'executing': runtime.executing = True
    elif condition == 'not_stopped': runtime.stopped = False
    elif condition == 'no_goals': bridge.verify_policy_goal = None
    elif condition == 'no_follow': bridge.normal_follow_budget_s = 0.
    elif condition == 'changed_deadline': bridge.maximum_lateness_s = .04
    elif condition == 'locked': bridge.lock.acquire()
    with pytest.raises(RuntimeError):
        bridge.execute_j3_range_diagnostic(commands, source, enable=True)
    assert not writes and not holds and not bus.reads
    if condition == 'locked': bridge.lock.release()


@pytest.mark.parametrize('amplitude', (5, 10))
def test_range_geometry_checks_both_five_command_blocks_from_original_source(amplitude):
    bridge, _, _, _, _, source, commands = range_diagnostic(amplitude=amplitude)
    calls = []
    def gate(q, sequence):
        q, sequence = np.asarray(q), np.asarray(sequence)
        calls.append((q.copy(), sequence.copy()))
        assert sequence.shape in ((1, 5, 6), (5, 6))
    bridge.physical_check = gate
    bridge.execute_j3_range_diagnostic(commands, source, enable=True, amplitude=amplitude)
    # Nominal + five independent arm-axis stalls for each block, before the
    # eight separate checks against the measured follow positions.
    assert len(calls) == 12 + 8
    for index in range(12):
        np.testing.assert_array_equal(calls[index][0], source[None])
    for block in range(2):
        base = block * 6
        expected = commands[block*5:(block+1)*5]
        np.testing.assert_array_equal(calls[base][1], expected[None])
        for joint in range(5):
            stalled = expected.copy()
            stalled[:, joint] = source[joint]
            np.testing.assert_array_equal(calls[base+1+joint][1], stalled[None])


@pytest.mark.parametrize('blocked', ('second_block', 'j3_stalled'))
def test_range_geometry_refusal_cannot_send_any_command(blocked):
    bridge, _, writes, holds, _, source, commands = range_diagnostic()
    def gate(q, sequence):
        candidate = np.asarray(sequence)
        if ((blocked == 'second_block' and np.all(candidate[..., 2] == -5.))
                or (blocked == 'j3_stalled' and np.all(candidate[..., 2] == 0.))):
            raise RuntimeError('physical path rejected')
    bridge.physical_check = gate
    with pytest.raises(RuntimeError, match='single-axis-stall') as caught:
        bridge.execute_j3_range_diagnostic(commands, source, enable=True)
    assert not writes and holds
    assert caught.value.last_attempt['policy_commands_attempted'] == 0


def test_range_source_drift_before_first_write_is_rejected():
    bridge, _, writes, holds, _, source, commands = range_diagnostic()
    calls = []
    def read():
        calls.append(None)
        value = source.copy()
        if len(calls) > 1: value[2] -= .08
        return value
    bridge.read_q = read
    with pytest.raises(RuntimeError, match='source changed before first command'):
        bridge.execute_j3_range_diagnostic(commands, source, enable=True)
    assert not writes and holds


def test_range_full_follow_window_remains_eight_samples_after_strong_response():
    bridge, _, _, _, _, source, commands = range_diagnostic()
    current = source.copy()
    original_write = bridge.write_q
    def write(command, previous):
        returned = original_write(command, previous)
        current[:] = returned
        return returned
    bridge.write_q = write
    bridge.read_q = lambda: current.copy()
    record = bridge.execute_j3_range_diagnostic(commands, source, enable=True)
    assert record['completed'] and record['normal_follow']['goal_reached']
    assert len(record['normal_follow']['positions']) == 8
    assert record['normal_follow']['times_s'][-1] - record['readback_times_s'][0] == pytest.approx(1.4)
    assert record['measured_q_path'][-1][2] == -5.
    assert record['original_target_reached'] is True
    assert record['original_target_arrival']['position_tolerance'] == .2


def test_range_plateau_is_not_arrival_even_when_completion_hold_reports_close():
    bridge, _, writes, _, _, source, commands = range_diagnostic(amplitude=10)
    original_write = bridge.write_q
    current = source.copy()
    def write(command, previous):
        actual = original_write(command, previous)
        current[:] = command
        current[2] += 2.3691177368164062
        return actual
    bridge.write_q = write
    bridge.read_q = lambda: current.copy()
    def hold(goal, follow, q10):
        assert follow['goal_reached'] is True
        # A later hold/endpoint reading must not erase the original error.
        current[:] = goal
    bridge.completion_hold = hold
    record = bridge.execute_j3_range_diagnostic(commands, source, enable=True, amplitude=10)
    assert record['completed'] and len(writes) == 10
    assert record['normal_follow']['goal_reached'] is True
    assert record['original_target_reached'] is False
    arrival = record['original_target_arrival']
    assert arrival['max_abs_residual'] == pytest.approx(2.3691177368164062)
    assert arrival['original_target'] == commands[-1].tolist()
    assert record['endpoint_q_position'] == commands[-1].tolist()
    assert arrival['excludes_post_hold_readbacks'] is True


def test_range_stop_during_seventh_write_preserves_counts_without_remaining_actions():
    bridge, runtime, writes, holds, bus, source, commands = range_diagnostic(cancel_at=7)
    with pytest.raises(RuntimeError, match='cancelled') as caught:
        bridge.execute_j3_range_diagnostic(commands, source, enable=True)
    record = caught.value.last_attempt
    assert len(writes) == 7 and len(bus.reads) == 6 and holds
    assert record['policy_commands_attempted'] == record['policy_commands_returned'] == 7
    assert record['policy_goal_registers_verified'] == 6
    assert len(record['measured_q_path']) == 7 and not record['completed']
    assert runtime.requires_home_after_incomplete_execution and runtime.stopped


def test_range_partial_write_exception_retains_actual_attempted_returned_split():
    bridge, runtime, writes, holds, bus, source, commands = range_diagnostic()
    original_write = bridge.write_q
    failure = ConnectionError('seventh write acknowledgement missing')
    def write(command, previous):
        returned = original_write(command, previous)
        if len(writes) == 7: raise failure
        return returned
    bridge.write_q = write
    with pytest.raises(ConnectionError) as caught:
        bridge.execute_j3_range_diagnostic(commands, source, enable=True)
    assert caught.value is failure and holds and runtime.requires_home_after_incomplete_execution
    assert len(writes) == 7 and len(bus.reads) == 6
    assert failure.last_attempt['policy_commands_attempted'] == 7
    assert failure.last_attempt['policy_commands_returned'] == 6
    assert failure.last_attempt == bridge.last_attempt and not failure.last_attempt['completed']


def test_range_eighth_goal_mismatch_retains_partial_evidence():
    bridge, runtime, writes, holds, bus, source, commands = range_diagnostic()
    bus.read_hook = lambda: setattr(bus, 'mismatch', len(bus.reads) == 8)
    with pytest.raises(RuntimeError, match='raw readback mismatch') as caught:
        bridge.execute_j3_range_diagnostic(commands, source, enable=True)
    record = caught.value.last_attempt
    assert len(writes) == len(bus.reads) == 8 and holds
    assert record['policy_goal_registers_verified'] == 7
    assert not record['completed'] and runtime.requires_home_after_incomplete_execution


def test_range_original_thirty_ms_write_deadline_remains_enforced():
    bridge, _, writes, holds, bus, source, commands = range_diagnostic(slow_write=True)
    with pytest.raises(RuntimeError, match='motor write missed') as caught:
        bridge.execute_j3_range_diagnostic(commands, source, enable=True)
    assert len(writes) == 1 and not bus.reads and holds
    assert caught.value.last_attempt['deadline_miss']['allowed_lateness_s'] == .03


def test_range_q10_read_is_not_retimed_after_deadline_miss():
    bridge, runtime, writes, holds, _, source, commands = range_diagnostic()
    original_read = bridge.read_q
    def read():
        if len(writes) == 10 and bridge.clock() >= 11.:
            bridge.wait(.04)
        return original_read()
    bridge.read_q = read
    with pytest.raises(RuntimeError, match='final readback missed') as caught:
        bridge.execute_j3_range_diagnostic(commands, source, enable=True)
    record = caught.value.last_attempt
    assert len(writes) == 10 and len(record['readback_times_s']) == 10 and holds
    assert record['read_calls'][-1]['returned_s'] == pytest.approx(11.04)
    assert not record['completed'] and runtime.requires_home_after_incomplete_execution


def test_range_stop_during_follow_keeps_full_q0_through_q10_and_real_follow_clock():
    bridge, runtime, writes, holds, _, source, commands = range_diagnostic()
    bridge.cancelled = lambda: bridge.clock() >= 11.1
    with pytest.raises(RuntimeError, match='cancelled') as caught:
        bridge.execute_j3_range_diagnostic(commands, source, enable=True)
    record = caught.value.last_attempt
    assert len(writes) == 10 and len(record['readback_times_s']) == 11 and holds
    assert record['normal_follow']['times_s'] == pytest.approx([11.05])
    assert not record['completed'] and runtime.requires_home_after_incomplete_execution


def test_range_amplitude_ten_is_four_step_ramp_then_fixed_total_offset():
    bridge, runtime, writes, holds, bus, source, commands = range_diagnostic(amplitude=10)
    state = runtime.state
    record = bridge.execute_j3_range_diagnostic(commands, source, enable=True, amplitude=10)
    assert record['diagnostic_total_amplitude'] == 10.
    assert np.asarray(writes)[:, 2].tolist() == [-2.5, -5., -7.5, -10., -10., -10., -10., -10., -10., -10.]
    assert len(bus.reads) == 10 and len(record['measured_q_path']) == 11
    assert len(record['normal_follow']['positions']) == 8 and holds == pytest.approx([11.4])
    assert record['completed'] and runtime.state is state
    assert record['not_model_transition'] and not record['model_feedback_label']


@pytest.mark.parametrize('command_amplitude,declared_amplitude', ((5, 10), (10, 5), (5, 7)))
def test_range_amplitude_mismatch_is_rejected_before_any_io(command_amplitude, declared_amplitude):
    bridge, _, writes, holds, bus, source, commands = range_diagnostic(amplitude=command_amplitude)
    with pytest.raises(ValueError):
        bridge.execute_j3_range_diagnostic(commands, source, enable=True, amplitude=declared_amplitude)
    assert not writes and not holds and not bus.reads


def test_range_invalid_telemetry_is_rejected_before_any_io():
    bridge, _, writes, holds, bus, source, commands = range_diagnostic()
    with pytest.raises(TypeError, match='explicit callable'):
        bridge.execute_j3_range_diagnostic(commands, source, enable=True, telemetry=True)
    assert not writes and not holds and not bus.reads


def test_range_telemetry_follows_exact_goal_verification_without_retiming_commands():
    bridge, runtime, writes, holds, bus, source, commands = range_diagnostic(amplitude=10)
    state = runtime.state
    bus.read_hook = lambda: bridge.wait(.001)
    def telemetry(*, command_index, deadline_s):
        assert len(writes) == len(bus.reads) == command_index + 1
        assert bridge.last_attempt['policy_goal_registers_verified'] == command_index + 1
        assert bridge.clock() < deadline_s
        started = bridge.clock()
        bridge.wait(.005)
        return dict(command_index=command_index, started_s=started, finished_s=bridge.clock(),
                    deadline_missed=False, read_budget_exceeded=False)
    record = bridge.execute_j3_range_diagnostic(commands, source, enable=True,
                                                 amplitude=10, telemetry=telemetry)
    events = record['dynamic_actuator_readbacks']
    assert len(events) == 10
    for index, event in enumerate(events):
        assert event['command_index'] == event['snapshot']['command_index'] == index
        assert event['deadline_s'] == pytest.approx(10 + (index + 1) * .1)
        assert event['callback_started_s'] >= record['policy_goal_readbacks'][index]['callback_returned_s']
        assert event['callback_returned_s'] < record['readback_times_s'][index + 1]
        assert event['callback_invoked'] and event['callback_returned'] and not event['gap_overrun_s']
    assert record['command_times_s'] == pytest.approx([10 + i*.1 for i in range(10)])
    assert record['readback_times_s'] == pytest.approx([10 + i*.1 for i in range(11)])
    assert record['normal_follow']['times_s'][-1] == pytest.approx(11.4)
    assert holds == pytest.approx([11.4]) and runtime.state is state


def test_range_telemetry_exception_keeps_partial_snapshot_and_existing_stop():
    bridge, runtime, writes, holds, bus, source, commands = range_diagnostic(amplitude=10)
    failure = ConnectionError('telemetry failed after a partial read')
    failure.dynamic_actuator_snapshot = dict(readings=[dict(raw_value=28)], partial=True)
    def telemetry(*, command_index, deadline_s):
        if command_index == 2:
            raise failure
        return dict(command_index=command_index)
    with pytest.raises(ConnectionError) as caught:
        bridge.execute_j3_range_diagnostic(commands, source, enable=True,
                                             amplitude=10, telemetry=telemetry)
    assert caught.value is failure and len(writes) == len(bus.reads) == 3 and holds
    record = failure.last_attempt
    event = record['dynamic_actuator_readbacks'][-1]
    assert event['snapshot'] == failure.dynamic_actuator_snapshot
    assert event['callback_invoked'] and not event['callback_returned'] and event['error']
    assert event['callback_failed_s'] <= event['callback_returned_s']
    assert not record['completed'] and runtime.requires_home_after_incomplete_execution


@pytest.mark.parametrize('overrun', ('actual_clock', 'snapshot_deadline', 'snapshot_read_budget'))
def test_range_telemetry_overrun_cannot_delay_the_next_command(overrun):
    bridge, runtime, writes, holds, bus, source, commands = range_diagnostic(amplitude=10)
    def telemetry(*, command_index, deadline_s):
        if overrun == 'actual_clock':
            bridge.wait(deadline_s + .031 - bridge.clock())
        return dict(deadline_missed=overrun == 'snapshot_deadline',
                    read_budget_exceeded=overrun == 'snapshot_read_budget')
    with pytest.raises(RuntimeError, match='telemetry') as caught:
        bridge.execute_j3_range_diagnostic(commands, source, enable=True,
                                             amplitude=10, telemetry=telemetry)
    record = caught.value.last_attempt
    assert len(writes) == len(bus.reads) == len(record['dynamic_actuator_readbacks']) == 1
    assert len(record['measured_q_path']) == 1 and holds and not record['completed']
    assert record['maximum_lateness_s'] == .03 and runtime.requires_home_after_incomplete_execution
    if overrun == 'actual_clock':
        miss = record['dynamic_actuator_deadline_miss']
        assert miss['deadline_s'] == pytest.approx(10.1)
        assert miss['observed_s'] == pytest.approx(10.131)
        assert record['dynamic_actuator_readbacks'][0]['snapshot'] is not None


def test_range_goal_failure_never_runs_telemetry_before_a_verified_command():
    bridge, _, writes, holds, bus, source, commands = range_diagnostic(amplitude=10)
    bus.mismatch = True
    def forbidden(**kwargs):
        pytest.fail('telemetry preceded Goal verification')
    with pytest.raises(RuntimeError, match='raw readback mismatch') as caught:
        bridge.execute_j3_range_diagnostic(commands, source, enable=True,
                                             amplitude=10, telemetry=forbidden)
    assert caught.value.last_attempt['dynamic_actuator_readbacks'] == []
    assert len(writes) == len(bus.reads) == 1 and holds
