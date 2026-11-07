"""Hardware-free coordinator tests for the single ten-command J3 profile."""
from copy import deepcopy
import json
import os
import stat
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import fcpg.v21_j3_range_service as module
from fcpg.v21_j3_range_service import J3RangeDiagnostic, OFFSETS
from fcpg.v21_j3_range_profile import range_offsets, range_result_name
from test_v21_j3_pid_service import Service


def make(tmp_path, *, enabled=True, amplitude=5):
    service = Service()
    runtime = service.owner.runtime
    runtime.state = object()
    runtime.requires_home_after_incomplete_execution = False
    def stop_runtime():
        runtime.stopped, runtime.pending, runtime.executing = True, None, False
    runtime.stop = stop_runtime
    service.ros_control_admission_epoch = lambda: service.epoch
    def drive(commands, source_q, *, enable=False):
        assert enable is True and service.operation_lock.locked()
        assert not service.owner.stop_event.is_set()
        service.driver_calls.append(commands.copy())
        q = np.asarray(source_q, dtype=np.float32)
        epoch = service.clock.now
        result = dict(completed=True, action_source='j3_range_diagnostic',
            not_model_transition=True, model_feedback_label=False, gain_writes=0,
            original_geometry_admitted=True, protective_hold_after_q10=True,
            maximum_lateness_s=.03, normal_follow={'budget_s': .4},
            policy_commands_attempted=10, policy_commands_returned=10,
            policy_goal_reads_attempted=10, policy_goal_reads_returned=10,
            policy_goal_registers_verified=10, source_q_position=q.tolist(),
            commands_position=np.asarray(commands, dtype=np.float32).tolist(),
            measured_q_path=[q.tolist() for _ in range(11)],
            readback_times_s=[epoch + i*.1 for i in range(11)],
            command_times_s=[epoch + i*.1 + .001 for i in range(10)],
            write_completed_times_s=[epoch + i*.1 + .002 for i in range(10)],
            policy_goal_readbacks=[{'command_index': i} for i in range(10)])
        service.owner.bridge.last_attempt = result
        service.clock.now += 1.5
        return result
    service.owner.bridge.execute_j3_range_diagnostic = drive
    return service, J3RangeDiagnostic(service, enabled=enabled, output=tmp_path,
                                     amplitude=amplitude)


def handle(service, coordinator):
    with service.operation_lock:
        return coordinator.handle()


def saved_range_report(amplitude, *, finished_s=12., residual=2.4):
    source = np.array([0., -97., 95., 63., -54., 31.])
    commands = np.repeat(source[None], 10, axis=0)
    commands[:, 2] += np.asarray(range_offsets(amplitude))
    actual = commands[-1].copy()
    actual[2] += residual
    drive = dict(completed=True, action_source='j3_range_diagnostic',
        not_model_transition=True, model_feedback_label=False, gain_writes=0,
        original_geometry_admitted=True, protective_hold_after_q10=True,
        maximum_lateness_s=.03,
        policy_commands_attempted=10, policy_commands_returned=10,
        policy_goal_reads_attempted=10, policy_goal_reads_returned=10,
        policy_goal_registers_verified=10,
        source_q_position=source.tolist(), commands_position=commands.tolist(),
        measured_q_path=[source.tolist()] * 10 + [actual.tolist()],
        readback_times_s=[10.+index*.1 for index in range(11)],
        command_times_s=[10.+index*.1+.001 for index in range(10)],
        write_completed_times_s=[10.+index*.1+.002 for index in range(10)],
        policy_goal_readbacks=[dict(command_index=index) for index in range(10)],
        normal_follow=dict(budget_s=.4, positions=[actual.tolist()] * 8,
                           times_s=[11.+index*.05 for index in range(1, 9)]),
        endpoint_q_position=commands[-1].tolist(),
        completion_hold=dict(held_position=commands[-1].tolist()))
    return dict(schema='v21-j3-range-diagnostic-v1', completed=True,
        amplitude=amplitude, finished_s=finished_s, commands_position=commands.tolist(),
        drive_result=drive, final_source=dict(q_position=commands[-1].tolist()),
        original_target_reached=True)


@pytest.mark.parametrize('latest_amplitude', [5, 10])
def test_saved_results_choose_latest_finished_without_owner_or_writes(tmp_path, latest_amplitude):
    class Forbidden:
        def __getattr__(self, name):
            pytest.fail('historical display touched owner: ' + name)
    contents = {}
    for amplitude in (5, 10):
        report = saved_range_report(amplitude, finished_s=30. if amplitude == latest_amplitude else 20.)
        path = tmp_path/range_result_name(amplitude)
        path.write_text(json.dumps(report))
        contents[path] = path.read_bytes()
    diagnostic = J3RangeDiagnostic(Forbidden(), output=tmp_path)
    status = diagnostic.status()
    previous = status['previous_result']
    assert previous['amplitude'] == latest_amplitude and previous['finished_s'] == 30.
    assert previous['evidence_source'] == 'saved_actual_receipt'
    assert previous['display_only'] and previous['model_history_imported'] is False
    assert diagnostic.last_result is None and diagnostic.consumed is False
    assert status['last_result'] is None and status['display_error'] is None
    assert previous['original_target_reached'] is False
    assert previous['original_target_arrival']['max_abs_residual'] == pytest.approx(2.4)
    assert all(path.read_bytes() == value for path, value in contents.items())
    previous['original_target_arrival']['original_target'][2] = 1000.
    assert diagnostic.status()['previous_result']['original_target_arrival']['original_target'][2] != 1000.


def test_saved_evaluation_ignores_stop_goal_final_source_and_old_reached_flag(tmp_path):
    report = saved_range_report(5)
    # Old range-5 receipts predate amplitude and arrival metadata.
    report.pop('amplitude')
    path = tmp_path/range_result_name(5)
    path.write_text(json.dumps(report))
    original = path.read_bytes()
    diagnostic = J3RangeDiagnostic(object(), output=tmp_path, amplitude=10)
    arrival = diagnostic.status()['previous_result']['original_target_arrival']
    assert arrival['sample_count'] == 9
    assert arrival['original_target'] == report['commands_position'][-1]
    assert arrival['actual_position'] == report['drive_result']['normal_follow']['positions'][-1]
    assert arrival['reached'] is False and arrival['max_abs_residual'] == pytest.approx(2.4)
    assert arrival['position_tolerance'] == .2 and arrival['step_tolerance'] == .1
    assert arrival['quiet_samples'] == 3 and arrival['excludes_post_hold_readbacks'] is True
    assert not diagnostic.status()['consumed'] and diagnostic.last_result is None
    assert path.read_bytes() == original


@pytest.mark.parametrize('damage', ['json', 'directory', 'finished_nan', 'duplicate_time',
                                  'missing_follow', 'post_hold_only', 'changed_commands',
                                  'short_count', 'missing_goal', 'invalid_goal'])
def test_saved_result_errors_do_not_block_startup_or_discard_other_valid_file(tmp_path, damage):
    (tmp_path/range_result_name(5)).write_text(json.dumps(saved_range_report(5)))
    path = tmp_path/range_result_name(10)
    report = saved_range_report(10, finished_s=20.)
    if damage == 'json': path.write_text('{broken')
    elif damage == 'directory': path.mkdir()
    else:
        if damage == 'finished_nan': report['finished_s'] = float('nan')
        elif damage == 'duplicate_time':
            report['drive_result']['normal_follow']['times_s'][1] = 11.05
        elif damage == 'missing_follow': report['drive_result'].pop('normal_follow')
        elif damage == 'post_hold_only': report['drive_result'].pop('measured_q_path')
        elif damage == 'short_count': report['drive_result']['policy_commands_returned'] = 9
        elif damage == 'missing_goal': report['drive_result']['policy_goal_readbacks'].pop()
        elif damage == 'invalid_goal': report['drive_result']['policy_goal_readbacks'][-1] = None
        else: report['drive_result']['commands_position'][-1][2] -= 3.
        path.write_text(json.dumps(report))
    diagnostic = J3RangeDiagnostic(object(), output=tmp_path, amplitude=10)
    status = diagnostic.status()
    assert status['previous_result']['amplitude'] == 5
    assert path.name in status['display_error']
    assert diagnostic.last_result is None and diagnostic.consumed is False


def test_saved_result_is_display_only_when_new_live_result_arrives(tmp_path):
    (tmp_path/range_result_name(5)).write_text(json.dumps(saved_range_report(5, residual=.1)))
    incomplete = saved_range_report(10, finished_s=100.)
    incomplete['completed'] = False
    (tmp_path/range_result_name(10)).write_text(json.dumps(incomplete))
    diagnostic = J3RangeDiagnostic(object(), output=tmp_path)
    prior = diagnostic.status()['previous_result']
    assert prior['amplitude'] == 5 and prior['original_target_reached'] is True
    live = dict(completed=False, original_target_reached=False, finished_s=200.)
    diagnostic.last_result = deepcopy(live)
    status = diagnostic.status()
    assert status['last_result'] == live and status['previous_result'] == prior
    assert status['display_error'] is None


def test_default_disabled_does_not_touch_owner_or_disk(tmp_path):
    class Forbidden:
        def __getattr__(self, name):
            pytest.fail('disabled range diagnostic touched service: ' + name)
    diagnostic = J3RangeDiagnostic(Forbidden(), output=tmp_path)
    with pytest.raises(RuntimeError, match='disabled'):
        diagnostic.handle()
    assert not diagnostic.status()['consumed'] and not list(tmp_path.iterdir())


@pytest.mark.parametrize('enabled', [None, 1, 'yes'])
def test_enable_is_explicit_boolean(tmp_path, enabled):
    with pytest.raises(TypeError):
        J3RangeDiagnostic(object(), enabled=enabled, output=tmp_path)


@pytest.mark.parametrize('fault', ['unlocked', 'owner', 'lease', 'not_stopped',
                                  'pending', 'executing', 'unlatched', 'incomplete'])
def test_bad_entry_never_reads_clears_stop_or_consumes(tmp_path, fault):
    service, diagnostic = make(tmp_path)
    owner = service.owner
    if fault == 'owner': service.owner = None
    elif fault == 'lease': owner.lease = None
    elif fault == 'not_stopped': owner.runtime.stopped = False
    elif fault == 'pending': owner.runtime.pending = object()
    elif fault == 'executing': owner.runtime.executing = True
    elif fault == 'unlatched': owner.stop_event.clear()
    elif fault == 'incomplete': owner.runtime.requires_home_after_incomplete_execution = True
    with pytest.raises(RuntimeError):
        diagnostic.handle() if fault == 'unlocked' else handle(service, diagnostic)
    assert not owner.bus.events and not service.driver_calls
    assert service.clears == service.stops == 0 and not diagnostic.path.exists()


def test_exact_ten_commands_keep_p16_and_model_history_then_cannot_replay(tmp_path):
    service, diagnostic = make(tmp_path)
    state = service.owner.runtime.state
    result = handle(service, diagnostic)
    assert result['completed'] and result['configuration_unchanged_verified'] and result['stopped']
    assert service.clears == service.stops == 1 and len(service.driver_calls) == 1
    delta = service.driver_calls[0] - np.array(result['source']['q_position'])
    np.testing.assert_allclose(delta[:, 2], OFFSETS)
    np.testing.assert_array_equal(delta[:, [0, 1, 3, 4, 5]], np.zeros((10, 5)))
    assert not [e for e in service.owner.bus.events if e[0] == 'write']
    assert service.owner.runtime.state is state and service.owner.stop_event.is_set()
    assert not result['model_history_imported'] and not result['gain_writes']
    assert result['amplitude'] == 5 and result['dynamic_telemetry_enabled'] is False
    assert diagnostic.status()['amplitude'] == 5
    assert diagnostic.status()['dynamic_telemetry_enabled'] is False
    assert json.loads(diagnostic.path.read_text()) == result
    count = len(service.owner.bus.events)
    for instance in (diagnostic, J3RangeDiagnostic(service, enabled=True, output=tmp_path)):
        with pytest.raises(RuntimeError, match='consumed'):
            handle(service, instance)
    assert len(service.owner.bus.events) == count and len(service.driver_calls) == 1


@pytest.mark.parametrize('amplitude, offsets, name', [
    (5, (-2.5,) + (-5.,) * 9, 'j3-range-diagnostic.json'),
    (10, (-2.5, -5., -7.5) + (-10.,) * 7, 'j3-range-diagnostic-10.json'),
])
def test_only_fixed_ten_command_profiles_exist(amplitude, offsets, name):
    assert range_offsets(amplitude) == offsets
    assert range_offsets(float(amplitude)) == offsets
    assert range_result_name(amplitude) == name
    assert range_offsets() == OFFSETS
    assert range_result_name() == 'j3-range-diagnostic.json'


@pytest.mark.parametrize('amplitude', [None, True, False, '5', '10', 0, -5, 7.5, 20,
                                     float('nan'), float('inf'), -float('inf')])
def test_unsupported_amplitude_rejected_before_service_or_disk(tmp_path, amplitude):
    class Forbidden:
        def __getattr__(self, name):
            pytest.fail('invalid profile touched service: ' + name)
    for function in (range_offsets, range_result_name):
        with pytest.raises(ValueError, match='exactly 5 or 10'):
            function(amplitude)
    with pytest.raises(ValueError, match='exactly 5 or 10'):
        J3RangeDiagnostic(Forbidden(), enabled=True, output=tmp_path, amplitude=amplitude)
    assert not list(tmp_path.iterdir())


def test_amplitude10_uses_same_thread_locked_bus_and_preserves_old_receipt(tmp_path, monkeypatch):
    old_path = tmp_path/'j3-range-diagnostic.json'
    previous = b'{"previous_physical_record": true}\n'
    old_path.write_bytes(previous)
    service, diagnostic = make(tmp_path, amplitude=10)
    owner, thread = service.owner, threading.get_ident()
    owner.bus.motors['elbow_flex'].model = 'sts3215'
    owner.bus.values['elbow_flex'].update(
        Present_Load=36, Present_Current=1, Present_Voltage=120, Status=0)
    original_reader = module.read_j3_dynamic_snapshot
    callbacks, snapshots = [], []
    def reader(bus, **kwargs):
        assert bus is owner.bus
        assert owner.bus_lock._is_owned() and service.operation_lock.locked()
        assert threading.get_ident() == thread
        assert kwargs['enable'] is True and kwargs['phase'] == 'after_goal_readback'
        assert kwargs['clock'] is service.clock and not kwargs['cancelled']()
        callbacks.append((kwargs['command_index'], kwargs['deadline_s']))
        result = original_reader(bus, **kwargs)
        snapshots.append(result)
        return result
    monkeypatch.setattr(module, 'read_j3_dynamic_snapshot', reader)
    original_drive = owner.bridge.execute_j3_range_diagnostic
    expected_deadlines = []
    def drive(commands, source_q, *, enable, amplitude, telemetry):
        assert amplitude == 10 and callable(telemetry)
        for command_index in range(10):
            deadline = service.clock() + .1
            expected_deadlines.append(deadline)
            telemetry(command_index, deadline)
        return original_drive(commands, source_q, enable=enable)
    owner.bridge.execute_j3_range_diagnostic = drive
    result = handle(service, diagnostic)
    assert result['completed'] and result['amplitude'] == 10
    assert result['dynamic_telemetry_enabled'] is True
    assert diagnostic.status()['amplitude'] == 10
    assert diagnostic.status()['dynamic_telemetry_enabled'] is True
    assert result['offsets'] == list(range_offsets(10))
    delta = service.driver_calls[0] - np.array(result['source']['q_position'])
    np.testing.assert_allclose(delta[:, 2], range_offsets(10))
    np.testing.assert_array_equal(delta[:, [0, 1, 3, 4, 5]], np.zeros((10, 5)))
    assert callbacks == list(enumerate(expected_deadlines))
    assert all(snapshot['all_registers_valid'] for snapshot in snapshots)
    assert all(snapshot['bus_reads_returned'] == 4 for snapshot in snapshots)
    assert all(snapshot['motion_verified'] is False for snapshot in snapshots)
    assert not [event for event in owner.bus.events if event[0] == 'write']
    assert diagnostic.path.name == 'j3-range-diagnostic-10.json'
    assert json.loads(diagnostic.path.read_text()) == result
    assert old_path.read_bytes() == previous


@pytest.mark.parametrize('change', ['STOP', 'stop_event', 'owner', 'lease', 'runtime',
                                  'bridge', 'operation_lock'])
def test_amplitude10_callback_refuses_changed_admission_before_bus_reads(tmp_path, monkeypatch, change):
    service, diagnostic = make(tmp_path, amplitude=10)
    owner = service.owner
    def forbidden(*args, **kwargs):
        pytest.fail('inadmissible telemetry touched bus reader')
    monkeypatch.setattr(module, 'read_j3_dynamic_snapshot', forbidden)
    def drive(commands, source_q, *, enable, amplitude, telemetry):
        assert enable and amplitude == 10
        if change == 'STOP': service.stop()
        elif change == 'stop_event': owner.stop_event.set()
        elif change == 'owner': service.owner = object()
        elif change == 'lease': owner.lease = object()
        elif change == 'runtime': owner.runtime = object()
        elif change == 'bridge': owner.bridge = object()
        else:
            service.operation_lock = threading.Lock()
            service.operation_lock.acquire()
        before = len(owner.bus.events)
        try:
            telemetry(0, service.clock() + .1)
        finally:
            assert len(owner.bus.events) == before
    owner.bridge.execute_j3_range_diagnostic = drive
    with pytest.raises(RuntimeError, match='range diagnostic'):
        handle(service, diagnostic)
    assert diagnostic.last_result['drive_attempted']
    assert not diagnostic.last_result['completed']
    assert owner.stop_event.is_set()


def test_amplitude10_stop_during_telemetry_skips_remaining_reads(tmp_path):
    service, diagnostic = make(tmp_path, amplitude=10)
    owner, snapshots = service.owner, []
    owner.bus.motors['elbow_flex'].model = 'sts3215'
    owner.bus.values['elbow_flex'].update(
        Present_Load=36, Present_Current=1, Present_Voltage=120, Status=0)
    def stop_after_load(register, motor):
        if register == 'Present_Load':
            service.stop()
    owner.bus.on_read = stop_after_load
    def drive(commands, source_q, *, enable, amplitude, telemetry):
        before = len(owner.bus.events)
        snapshots.append(telemetry(0, service.clock() + .1))
        assert owner.bus.events[before:] == [('read', 'Present_Load', 'elbow_flex')]
        raise RuntimeError('existing bridge sees STOP')
    owner.bridge.execute_j3_range_diagnostic = drive
    with pytest.raises(RuntimeError, match='existing bridge sees STOP'):
        handle(service, diagnostic)
    assert snapshots[0]['bus_reads_attempted'] == 1
    assert snapshots[0]['bus_reads_returned'] == 1
    assert [item['skip_reason'] for item in snapshots[0]['readings']] == [
        None, 'cancelled', 'cancelled', 'cancelled']
    assert not diagnostic.last_result['completed'] and owner.stop_event.is_set()


@pytest.mark.parametrize('kind', ['malformed', 'directory', 'dangling_symlink'])
def test_any_existing_result_path_prevents_replay(tmp_path, kind):
    service, diagnostic = make(tmp_path)
    if kind == 'malformed': diagnostic.path.write_text('partial previous result')
    elif kind == 'directory': diagnostic.path.mkdir()
    else: diagnostic.path.symlink_to(tmp_path/'absent')
    with pytest.raises((RuntimeError, FileExistsError)):
        handle(service, diagnostic)
    assert not service.owner.bus.events and service.clears == 0


def test_initial_file_and_directory_are_synced_before_any_register_read(tmp_path, monkeypatch):
    service, diagnostic = make(tmp_path)
    syncs = []
    original = os.fsync
    def sync(fd):
        assert not service.owner.bus.events
        syncs.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        original(fd)
    monkeypatch.setattr(module.os, 'fsync', sync)
    handle(service, diagnostic)
    assert syncs == [False, True]


def test_initial_sync_failure_never_touches_bus_and_remains_consumed(tmp_path, monkeypatch):
    service, diagnostic = make(tmp_path)
    def fail(fd):
        raise OSError('initial disk failure')
    monkeypatch.setattr(module.os, 'fsync', fail)
    with pytest.raises(OSError, match='initial disk failure'):
        handle(service, diagnostic)
    assert diagnostic.status()['consumed'] and not service.owner.bus.events
    assert service.clears == service.stops == 0


def test_slow_intent_log_is_followed_by_a_new_actual_read(tmp_path, monkeypatch):
    service, diagnostic = make(tmp_path)
    original = module.write_json
    def slow(path, value):
        original(path, value)
        service.clock.now += 3.
    monkeypatch.setattr(module, 'write_json', slow)
    result = handle(service, diagnostic)
    assert result['dispatch_source']['q_time_s'] - result['source']['q_time_s'] > 3.
    assert result['drive_started_s'] >= result['dispatch_source']['q_time_s']
    assert result['drive_started_s'] - result['dispatch_source']['q_time_s'] < .1
    assert result['source']['q_position'] == result['dispatch_source']['q_position']
    assert len(service.driver_calls) == 1


@pytest.mark.parametrize('mutation', ['pose', 'J3_config', 'other_axis_config', 'STOP'])
def test_change_during_logging_prevents_stop_clear_and_drive(tmp_path, monkeypatch, mutation):
    service, diagnostic = make(tmp_path)
    original = module.write_json
    def change(path, value):
        original(path, value)
        if 'dispatch_source' in value or 'finished_s' in value:
            return
        if mutation == 'pose': service.owner.bus.q[2] += .01
        elif mutation == 'J3_config': service.owner.bus.values['elbow_flex']['P_Coefficient'] = 32
        elif mutation == 'other_axis_config': service.owner.bus.values['gripper']['P_Coefficient'] = 17
        else: service.stop()
    monkeypatch.setattr(module, 'write_json', change)
    with pytest.raises(RuntimeError):
        handle(service, diagnostic)
    assert service.clears == 0 and not service.driver_calls
    assert diagnostic.last_result['drive_attempted'] is False
    assert service.owner.stop_event.is_set()


@pytest.mark.parametrize('changed', ['owner', 'lease', 'runtime', 'bridge', 'lock', 'STOP'])
def test_binding_is_rechecked_after_stop_clear(tmp_path, changed):
    service, diagnostic = make(tmp_path)
    original_owner = service.owner
    clear = service._clear_stop_for_current_operation
    def change():
        clear()
        if changed == 'owner': service.owner = make(tmp_path/'other')[0].owner
        elif changed == 'lease': service.owner.lease = object()
        elif changed == 'runtime': service.owner.runtime = SimpleNamespace(**vars(service.owner.runtime))
        elif changed == 'bridge': service.owner.bridge = SimpleNamespace(runtime=service.owner.runtime)
        elif changed == 'lock':
            service.operation_lock = threading.Lock()
            service.operation_lock.acquire()
        else: service.stop()
    service._clear_stop_for_current_operation = change
    with pytest.raises(RuntimeError):
        handle(service, diagnostic)
    assert not service.driver_calls and original_owner.stop_event.is_set()
    assert diagnostic.last_result['drive_attempted'] is False


@pytest.mark.parametrize('damage', ['goal_count', 'q_count', 'commands', 'source', 'clock',
                                  'follow_budget', 'hold', 'model_label', 'nan', 'history'])
def test_malformed_success_cannot_be_reported_as_completed(tmp_path, damage):
    service, diagnostic = make(tmp_path)
    original = service.owner.bridge.execute_j3_range_diagnostic
    def damaged(*args, **kwargs):
        result = original(*args, **kwargs)
        if damage == 'goal_count': result['policy_goal_reads_returned'] = 9
        elif damage == 'q_count': result['measured_q_path'].pop()
        elif damage == 'commands': result['commands_position'][0][2] += .1
        elif damage == 'source': result['source_q_position'][2] += .1
        elif damage == 'clock': result['command_times_s'][4] = result['readback_times_s'][0]
        elif damage == 'follow_budget': result['normal_follow']['budget_s'] = 4.
        elif damage == 'hold': result['protective_hold_after_q10'] = False
        elif damage == 'model_label': result['model_feedback_label'] = True
        elif damage == 'nan': result['measured_q_path'][4][2] = float('nan')
        else: service.owner.runtime.state = object()
        return result
    service.owner.bridge.execute_j3_range_diagnostic = damaged
    with pytest.raises(RuntimeError):
        handle(service, diagnostic)
    assert not diagnostic.last_result['completed'] and diagnostic.last_result['drive_attempted']
    assert service.owner.runtime.requires_home_after_incomplete_execution
    assert service.stops == 1 and service.owner.stop_event.is_set()


def test_original_drive_exception_and_partial_survive_final_disk_failure(tmp_path, monkeypatch):
    service, diagnostic = make(tmp_path)
    error = RuntimeError('real transport failure')
    partial = dict(policy_commands_attempted=3, policy_commands_returned=2,
                   action_source='j3_range_diagnostic', measured_q_path=[[1., 2., 3., 4., 5., 6.]])
    def failed(*args, **kwargs):
        service.owner.bridge.last_attempt = deepcopy(partial)
        error.last_attempt = deepcopy(partial)
        raise error
    service.owner.bridge.execute_j3_range_diagnostic = failed
    original = module.write_json
    def save(path, report):
        if 'finished_s' in report: raise OSError('final disk failure')
        original(path, report)
    monkeypatch.setattr(module, 'write_json', save)
    with pytest.raises(RuntimeError) as caught:
        handle(service, diagnostic)
    assert caught.value is error
    result = diagnostic.last_result
    assert result['failed_drive_attempt'] == partial and 'final disk failure' in result['recording_error']
    assert 'real transport failure' in result['error']
    assert service.stops == 1 and service.owner.stop_event.is_set()
    assert service.owner.runtime.requires_home_after_incomplete_execution and diagnostic.status()['consumed']


def test_final_recording_failure_preserves_completed_physical_receipt(tmp_path, monkeypatch):
    service, diagnostic = make(tmp_path)
    original = module.write_json
    def save(path, report):
        if 'finished_s' in report: raise OSError('final disk failure')
        original(path, report)
    monkeypatch.setattr(module, 'write_json', save)
    with pytest.raises(OSError, match='final disk failure'):
        handle(service, diagnostic)
    result = diagnostic.last_result
    assert result['completed'] and result['drive_result']['policy_commands_returned'] == 10
    assert result['recording_error'] and result['stopped'] and diagnostic.status()['consumed']


def test_failure_before_new_receipt_does_not_reuse_old_bridge_attempt(tmp_path):
    service, diagnostic = make(tmp_path)
    service.owner.bridge.last_attempt = {'policy_commands_attempted': 5, 'old_record': True}
    def fail(*args, **kwargs):
        raise RuntimeError('new driver refused before starting')
    service.owner.bridge.execute_j3_range_diagnostic = fail
    with pytest.raises(RuntimeError, match='new driver refused'):
        handle(service, diagnostic)
    assert 'failed_drive_attempt' not in diagnostic.last_result
    assert not service.owner.runtime.requires_home_after_incomplete_execution
    assert service.stops == 1


def test_stop_failure_keeps_actual_driver_result_and_memory_stop(tmp_path):
    service, diagnostic = make(tmp_path)
    service.stop_error = RuntimeError('STOP transport failure')
    with pytest.raises(RuntimeError, match='STOP transport failure'):
        handle(service, diagnostic)
    result = diagnostic.last_result
    assert result['drive_result']['completed'] and not result['completed']
    assert 'STOP transport failure' in result['cleanup_error']
    assert service.owner.runtime.stopped and service.owner.stop_event.is_set()
