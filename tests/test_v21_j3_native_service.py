"""Offline tests of native-five admission, evidence and fault containment."""
from copy import deepcopy
from dataclasses import dataclass
import json
import os
import stat
import threading

import numpy as np
import pytest
import torch

import fcpg.v21_j3_native_service as module
from fcpg.v21_j3_native_service import ACTION_SOURCE, J3NativeResponseDiagnostic, OFFSETS
from scripts.prepare_v21_j3_pid_protocol import FakeJ3Bus, MOTORS
from test_v21_j3_pid_service import Service


@dataclass
class StoredSourceFixture:
    q: torch.Tensor
    time: torch.Tensor


def make(tmp_path, *, enabled=True):
    service = Service()
    owner = service.owner
    # A source far outside the old PID HOME-near requirement is intentional.
    owner.bus.q = [0., -97.282829, 84.738243, 63.255035, -54.233765, 30.749014]
    runtime = owner.runtime
    runtime.state = object()
    runtime.requires_home_after_incomplete_execution = False
    def stop_runtime():
        runtime.stopped, runtime.pending, runtime.executing = True, None, False
    runtime.stop = stop_runtime
    service.ros_control_admission_epoch = lambda: service.epoch
    def drive(commands, source_q, *, enable=False):
        assert enable is True and service.operation_lock.locked()
        assert not owner.stop_event.is_set()
        service.driver_calls.append(commands.copy())
        fixture = FakeJ3Bus()
        fixture.q = np.asarray(source_q, np.float32).tolist()
        fixture.now = service.clock.now
        epoch = fixture.now
        result = fixture.execute_five(commands.tolist(), 'offline_native_fixture')
        result.update(action_source=ACTION_SOURCE, not_model_transition=True,
                      model_feedback_label=False, gain_writes=0,
                      source_q_position=fixture.q, source_time_s=epoch,
                      source_q_readback_time_s=epoch)
        result['normal_follow'].update(positions=[fixture.q.copy() for _ in range(8)],
            times_s=[epoch + .5 + index * .05 for index in range(1, 9)])
        service.clock.now = fixture.now
        owner.bridge.last_attempt = result
        return result
    owner.bridge.execute_j3_native_response_diagnostic = drive
    return service, J3NativeResponseDiagnostic(service, enabled=enabled, output=tmp_path)


def handle(service, coordinator):
    with service.operation_lock:
        return coordinator.handle()


def test_disabled_does_not_touch_owner_clock_or_disk(tmp_path):
    class Forbidden:
        def __getattr__(self, name):
            pytest.fail('disabled coordinator touched service: ' + name)
    coordinator = J3NativeResponseDiagnostic(Forbidden(), output=tmp_path)
    with pytest.raises(RuntimeError, match='disabled'):
        coordinator.handle()
    assert not coordinator.status()['consumed'] and not list(tmp_path.iterdir())


@pytest.mark.parametrize('value', [None, 0, 1, 'yes'])
def test_enable_is_explicit_bool(tmp_path, value):
    with pytest.raises(TypeError):
        J3NativeResponseDiagnostic(object(), enabled=value, output=tmp_path)


@pytest.mark.parametrize('fault', ['unlocked', 'owner', 'lease', 'not_stopped',
                                  'pending', 'executing', 'unlatched', 'incomplete'])
def test_bad_entry_does_not_consume_or_read(tmp_path, fault):
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


def test_current_pose_native5_preserves_p16_history_and_requires_explicit_reset(tmp_path):
    service, diagnostic = make(tmp_path)
    state = service.owner.runtime.state
    report = handle(service, diagnostic)
    assert report['completed'] and report['configuration_unchanged_verified'] and report['stopped']
    assert service.clears == service.stops == 1 and len(service.driver_calls) == 1
    q = np.asarray(report['source']['q_position'], np.float32)
    expected = np.repeat(q[None], 5, axis=0)
    expected[:, 2] += np.asarray(OFFSETS, np.float32)
    np.testing.assert_array_equal(service.driver_calls[0], expected)
    assert report['source']['q_position'][2] == pytest.approx(84.738243)
    assert service.owner.runtime.state is state
    assert service.owner.runtime.requires_home_after_incomplete_execution
    assert report['requires_home_before_policy_resume'] and report['model_transition'] is False
    assert not report['automatic_home'] and not report['automatic_retry'] and report['gain_writes'] == 0
    assert not [event for event in service.owner.bus.events if event[0] == 'write']
    assert report['native_response']['directional_j3_response'] == 0
    assert report['native_response']['excludes_follow_and_hold']
    assert report['original_target_reached'] is False
    assert not report['training_ready'] and not report['initial_context']['missing_values_filled']
    assert 'pre_dispatch_actuator_state' in report['initial_context']['missing_fields']
    assert json.loads(diagnostic.path.read_text()) == report
    # Resetting a test fixture's HOME latch cannot reset the durable one-shot.
    service.owner.runtime.requires_home_after_incomplete_execution = False
    for instance in (diagnostic, J3NativeResponseDiagnostic(service, enabled=True, output=tmp_path)):
        with pytest.raises(RuntimeError, match='consumed'):
            handle(service, instance)
    assert len(service.driver_calls) == 1


@pytest.mark.parametrize('kind', ['file', 'directory', 'dangling_symlink'])
def test_any_existing_intent_prevents_replay_without_owner_reads(tmp_path, kind):
    service, diagnostic = make(tmp_path)
    if kind == 'file': diagnostic.path.write_text('partial prior result')
    elif kind == 'directory': diagnostic.path.mkdir()
    else: diagnostic.path.symlink_to(tmp_path/'missing')
    assert diagnostic.status()['consumed']
    with pytest.raises(RuntimeError, match='consumed'):
        handle(service, diagnostic)
    assert not service.owner.bus.events and service.clears == 0


def test_initial_intent_file_and_directory_fsync_precede_reads(tmp_path, monkeypatch):
    service, diagnostic = make(tmp_path)
    syncs = []
    original = os.fsync
    def synced(fd):
        if len(syncs) < 2:
            assert not service.owner.bus.events
        syncs.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        original(fd)
    monkeypatch.setattr(module.os, 'fsync', synced)
    handle(service, diagnostic)
    assert syncs[:2] == [False, True]


def test_initial_sync_failure_consumes_without_io(tmp_path, monkeypatch):
    service, diagnostic = make(tmp_path)
    def broken(_): raise OSError('intent fsync failure')
    monkeypatch.setattr(module.os, 'fsync', broken)
    with pytest.raises(OSError, match='intent fsync'):
        handle(service, diagnostic)
    assert diagnostic.status()['consumed'] and not service.owner.bus.events
    assert service.clears == service.stops == 0


def test_slow_intent_persistence_uses_fresh_matching_second_snapshot(tmp_path, monkeypatch):
    service, diagnostic = make(tmp_path)
    original = module._durable_json
    def delayed(path, report):
        result = original(path, report)
        service.clock.now += 3.
        return result
    monkeypatch.setattr(module, '_durable_json', delayed)
    report = handle(service, diagnostic)
    assert report['dispatch_source']['q_time_s'] - report['source']['q_time_s'] > 3.
    assert report['drive_started_s'] - report['dispatch_source']['q_time_s'] < .1
    assert report['dispatch_source']['present_raw'] == report['source']['present_raw']


@pytest.mark.parametrize('mutation', ['pose', 'raw', 'P32', 'other_configuration', 'STOP', 'lease', 'lock'])
def test_changed_source_or_admission_never_clears_stop(tmp_path, monkeypatch, mutation):
    service, diagnostic = make(tmp_path)
    owner = service.owner
    original = module._durable_json
    def change(path, report):
        value = original(path, report)
        if 'finished_s' in report:
            return value
        if mutation == 'pose': owner.bus.q[2] += .01
        elif mutation == 'raw':
            owner.bus.values['gripper']['Present_Position'] += 1
            owner.bus.inverse_raw[6] += 1
        elif mutation == 'P32': owner.bus.values['elbow_flex']['P_Coefficient'] = 32
        elif mutation == 'other_configuration': owner.bus.values['gripper']['P_Coefficient'] = 17
        elif mutation == 'STOP': service.stop()
        elif mutation == 'lease': owner.lease = object()
        else:
            service.operation_lock = threading.Lock()
            service.operation_lock.acquire()
        return value
    monkeypatch.setattr(module, '_durable_json', change)
    with pytest.raises(RuntimeError):
        handle(service, diagnostic)
    assert service.clears == 0 and not service.driver_calls and owner.stop_event.is_set()
    assert diagnostic.status()['consumed'] and not diagnostic.last_result['completed']


@pytest.mark.parametrize('fault', ['moving', 'unheld', 'wrong_gain', 'slow_read', 'slow_lease_after_clear'])
def test_unqualified_or_stale_source_cannot_drive(tmp_path, fault):
    service, diagnostic = make(tmp_path)
    owner = service.owner
    if fault == 'moving': owner.bus.values['elbow_flex']['Moving'] = 1
    elif fault == 'unheld': owner.bus.values['elbow_flex']['Goal_Position'] += 2
    elif fault == 'wrong_gain': owner.bus.values['elbow_flex']['P_Coefficient'] = 32
    elif fault == 'slow_read':
        owner.bus.on_read = lambda *args: setattr(service.clock, 'now', service.clock.now + .01)
    else:
        original = owner._lease_check
        def delayed():
            original()
            if service.clears: service.clock.now += .2
        owner._lease_check = delayed
    with pytest.raises(RuntimeError):
        handle(service, diagnostic)
    assert not service.driver_calls and owner.stop_event.is_set()


@pytest.mark.parametrize('damage', ['action', 'count', 'y_count', 'goal_raw', 'goal_requested',
                                  'goal_late', 'q_late', 'follow_short', 'follow_late',
                                  'source_epoch', 'history', 'gain_writes'])
def test_invalid_receipt_is_saved_incomplete_and_never_model_history(tmp_path, damage):
    service, diagnostic = make(tmp_path)
    original = service.owner.bridge.execute_j3_native_response_diagnostic
    def damaged(*args, **kwargs):
        drive = original(*args, **kwargs)
        if damage == 'action': drive['action_source'] = 'j3_pid_diagnostic'
        elif damage == 'count': drive['policy_commands_returned'] = 4
        elif damage == 'y_count': drive['measured_q_path'].pop()
        elif damage == 'goal_raw': drive['policy_goal_readbacks'][0]['goal_raw']['gripper'] += 1
        elif damage == 'goal_requested': drive['policy_goal_readbacks'][0]['requested_goal_position']['gripper'] += 1
        elif damage == 'goal_late': drive['policy_goal_readbacks'][0]['callback_returned_s'] += .1
        elif damage == 'q_late': drive['readback_times_s'][-1] += .031
        elif damage == 'follow_short': drive['normal_follow']['positions'].pop()
        elif damage == 'follow_late': drive['normal_follow']['times_s'][-1] += .031
        elif damage == 'source_epoch': drive['source_time_s'] -= .1
        elif damage == 'history': service.owner.runtime.state = object()
        else: drive['gain_writes'] = 1
        return drive
    service.owner.bridge.execute_j3_native_response_diagnostic = damaged
    with pytest.raises((ValueError, RuntimeError)):
        handle(service, diagnostic)
    report = diagnostic.last_result
    assert report['drive_attempted'] and not report['completed'] and report['drive_result']
    assert report['requires_home_after_incomplete_execution'] and service.owner.stop_event.is_set()
    assert json.loads(diagnostic.path.read_text()) == report


@pytest.mark.parametrize('with_partial', [True, False])
def test_drive_failure_keeps_partial_or_unknown_evidence_without_retry(tmp_path, with_partial):
    service, diagnostic = make(tmp_path)
    partial = dict(completed=False, policy_commands_attempted=2, policy_commands_returned=1,
                   measured_q_path=[[0., 0., 0., 0., 0., 0.]])
    def fail(*args, **kwargs):
        if with_partial: service.owner.bridge.last_attempt = partial
        raise RuntimeError('native write interrupted')
    service.owner.bridge.execute_j3_native_response_diagnostic = fail
    with pytest.raises(RuntimeError, match='native write interrupted'):
        handle(service, diagnostic)
    report = diagnostic.last_result
    if with_partial: assert report['failed_drive_attempt'] == partial
    else: assert 'failed_drive_attempt' not in report
    assert report['requires_home_after_incomplete_execution'] and not report['completed']
    assert service.owner.stop_event.is_set() and service.stops == 1


def test_native_y_excludes_follow_and_hold_and_keeps_missing_context_null(tmp_path):
    service, diagnostic = make(tmp_path)
    original = service.owner.bridge.execute_j3_native_response_diagnostic
    def result(*args, **kwargs):
        drive = original(*args, **kwargs)
        target = drive['commands_position'][-1]
        drive['normal_follow']['positions'] = [target.copy() for _ in range(8)]
        drive['endpoint_q_position'] = target.copy()
        drive['endpoint_time_s'] = service.clock()
        drive['pre_dispatch_actuator_state'] = dict(registers={'Present_Load': None},
            owner_observation_history={'complete_since_connection': False},
            optional_register_errors={'Present_Load': 'not available'})
        return drive
    service.owner.bridge.execute_j3_native_response_diagnostic = result
    report = handle(service, diagnostic)
    assert report['native_response']['directional_j3_response'] == 0
    assert report['original_target_reached']
    assert 'Present_Load' in report['initial_context']['missing_fields']
    assert report['drive_result']['pre_dispatch_actuator_state']['registers']['Present_Load'] is None
    assert not report['training_ready']


def test_one_cell_gripper_goal_delta_is_reported_without_modifying_profile(tmp_path):
    service, diagnostic = make(tmp_path)
    original = service.owner.bridge.execute_j3_native_response_diagnostic
    def quantized(*args, **kwargs):
        drive = original(*args, **kwargs)
        source = {name: service.owner.bus.values[name]['Present_Position'] for name in MOTORS}
        for event in drive['policy_goal_readbacks']:
            cells = {**source, 'elbow_flex': source['elbow_flex'] - 17, 'gripper': source['gripper'] - 1}
            event['goal_raw'], event['expected_goal_raw'] = cells.copy(), cells.copy()
        return drive
    service.owner.bridge.execute_j3_native_response_diagnostic = quantized
    report = handle(service, diagnostic)
    quantization = report['raw_goal_quantization']
    assert not quantization['raw_pure_j3']
    assert all(row['gripper'] == -1 for row in quantization['non_j3_raw_delta'])
    assert all(row[name] == 0 for row in quantization['non_j3_raw_delta']
               for name in ('shoulder_pan', 'shoulder_lift', 'wrist_flex', 'wrist_roll'))


@pytest.mark.parametrize('fault', ['stop', 'final_read', 'final_persist'])
def test_cleanup_or_final_persistence_failure_latches_recovery_and_preserves_receipt(tmp_path, monkeypatch, fault):
    service, diagnostic = make(tmp_path)
    if fault == 'stop': service.stop_error = RuntimeError('STOP failed')
    elif fault == 'final_read':
        def outage(register, motor):
            if service.stops: raise RuntimeError('final source unavailable')
        service.owner.bus.on_read = outage
    else:
        original = module._durable_json
        def fail(path, report):
            if 'finished_s' in report: raise OSError('final disk failure')
            return original(path, report)
        monkeypatch.setattr(module, '_durable_json', fail)
    with pytest.raises((RuntimeError, OSError)):
        handle(service, diagnostic)
    report = diagnostic.last_result
    assert not report['completed'] and report['drive_result']['completed']
    assert report['requires_home_after_incomplete_execution'] and service.owner.stop_event.is_set()
    assert diagnostic.status()['consumed'] and not report['training_ready']


def enable_source_recording(service, tmp_path):
    service.record_policy_inputs = True
    service.root = tmp_path/'model-output'
    service.lewm_controller = {'candidate_sha256': 'offline-fixture'}
    service.temporal_contract = {'profile': 'actual-sample-index-fixture'}
    state = StoredSourceFixture(q=torch.tensor([[1., -90., 99., 64., -51., 31.]]),
                                time=torch.tensor([3.125], dtype=torch.float64))
    service.owner.runtime.state = state
    service.owner.runtime.text_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    return state


def test_model_source_is_saved_before_any_read_or_motion_with_original_time(tmp_path, monkeypatch):
    service, diagnostic = make(tmp_path)
    state = enable_source_recording(service, tmp_path)
    original_q, original_time = state.q.clone(), state.time.clone()
    save = module.torch.save
    saves = []
    def checked(payload, stream):
        assert not service.owner.bus.events and not service.driver_calls and service.clears == 0
        assert payload['state'].q.data_ptr() != state.q.data_ptr()
        assert payload['state'].time.data_ptr() != state.time.data_ptr()
        saves.append(None)
        save(payload, stream)
        service.clock.now += 2.
    monkeypatch.setattr(module.torch, 'save', checked)
    report = handle(service, diagnostic)
    reference = report['source_state_snapshot']
    assert len(saves) == 1 and reference['saved'] and not reference['state_time_rewritten']
    assert 'sha256' not in reference and 'capture_started_s' not in reference
    assert reference['snapshot_started_s'] <= reference['saved_s']
    payload = torch.load(reference['path'], map_location='cpu', weights_only=False)
    assert payload['kind'] == 'native-response-source' and payload['executable'] is False
    assert torch.equal(payload['state'].q, original_q) and torch.equal(payload['state'].time, original_time)
    assert torch.equal(state.q, original_q) and torch.equal(state.time, original_time)
    assert torch.equal(payload['text_ids'], service.owner.runtime.text_ids)
    assert payload['lewm_controller'] == service.lewm_controller
    assert payload['temporal_contract'] == service.temporal_contract
    assert payload['model_output'] == str(service.root.resolve())
    assert reference['stored_state_time_s'] == [3.125]
    assert reference['dispatch_q_max_abs_difference'] > 10.
    assert report['source']['started_s'] >= reference['saved_s']
    assert report['dispatch_source']['q_time_s'] >= report['source']['q_time_s']


@pytest.mark.parametrize('failure', ['disk', 'missing_state', 'missing_text_ids', 'existing_snapshot'])
def test_requested_source_record_failure_prevents_policy_motion_and_consumes(tmp_path, monkeypatch, failure):
    service, diagnostic = make(tmp_path)
    enable_source_recording(service, tmp_path)
    if failure == 'disk':
        def broken(*args, **kwargs): raise OSError('source save failed')
        monkeypatch.setattr(module.torch, 'save', broken)
    elif failure == 'missing_state': service.owner.runtime.state = None
    elif failure == 'missing_text_ids': service.owner.runtime.text_ids = None
    else: (tmp_path/'native-response-source-3.pt').write_bytes(b'old reviewed source')
    with pytest.raises((OSError, RuntimeError)):
        handle(service, diagnostic)
    assert not service.driver_calls and service.clears == 0
    assert diagnostic.status()['consumed'] and not diagnostic.last_result['completed']
    if failure == 'existing_snapshot':
        assert (tmp_path/'native-response-source-3.pt').read_bytes() == b'old reviewed source'


@pytest.mark.parametrize('primary_failure', [False, True])
def test_final_latch_error_preserves_primary_and_durable_report(tmp_path, primary_failure):
    service, diagnostic = make(tmp_path)
    if primary_failure:
        def interrupted(*args, **kwargs): raise RuntimeError('primary native drive error')
        service.owner.bridge.execute_j3_native_response_diagnostic = interrupted
    def broken_latch(): raise RuntimeError('runtime latch failed')
    service.owner.runtime.stop = broken_latch
    with pytest.raises(RuntimeError, match='primary native drive error' if primary_failure else 'runtime latch failed'):
        handle(service, diagnostic)
    report = diagnostic.last_result
    assert report['stop_latch_error'] == 'RuntimeError: runtime latch failed'
    assert report['completed'] is False and report['requires_home_after_incomplete_execution']
    assert json.loads(diagnostic.path.read_text()) == report
    if primary_failure: assert report['error'] == 'RuntimeError: primary native drive error'


@pytest.mark.parametrize('cancellation', ['stop', 'epoch_only'])
def test_stop_in_drive_return_gap_preserves_receipt_but_rejects_outer_completion(tmp_path, cancellation):
    service, diagnostic = make(tmp_path)
    original = service.owner.bridge.execute_j3_native_response_diagnostic
    actual = []
    def returning(*args, **kwargs):
        drive = original(*args, **kwargs)
        actual.append(deepcopy(drive))
        if cancellation == 'stop': service.stop()
        else: service.epoch += 1
        return drive
    service.owner.bridge.execute_j3_native_response_diagnostic = returning
    with pytest.raises(RuntimeError, match='cancelled by STOP'):
        handle(service, diagnostic)
    report = diagnostic.last_result
    assert not report['completed'] and report['drive_result'] == actual[0]
    assert report['drive_result']['completed'] is True
    assert report['requires_home_after_incomplete_execution'] and service.owner.stop_event.is_set()
    assert json.loads(diagnostic.path.read_text()) == report


def test_restart_reassesses_saved_actual_result_without_owner_or_control_changes(tmp_path):
    service, diagnostic = make(tmp_path)
    report = handle(service, diagnostic)
    original_bytes = diagnostic.path.read_bytes()
    class Forbidden:
        def __getattr__(self, name):
            pytest.fail('display restoration touched service: ' + name)
    restored = J3NativeResponseDiagnostic(Forbidden(), output=tmp_path)
    status = restored.status()
    assert not status['enabled'] and status['consumed'] and status['last_result'] is None
    assert status['display_error'] is None
    previous = status['previous_result']
    assert previous['completed'] and previous['command_budget'] == 5
    assert previous['finished_s'] == report['finished_s']
    assert previous['original_target_reached'] is report['original_target_reached']
    for key, value in report['original_target_arrival'].items():
        assert previous['original_target_arrival'][key] == value
    assert previous['original_target_arrival']['position_unit'] == 'lerobot_calibrated_position'
    assert previous['display_only'] and not previous['model_history_imported']
    assert not previous['model_transition'] and not previous['training_ready']
    assert diagnostic.path.read_bytes() == original_bytes
    with pytest.raises(RuntimeError, match='disabled'):
        restored.handle()
    previous['original_target_arrival']['max_abs_residual'] = 99
    assert restored.status()['previous_result']['original_target_arrival']['max_abs_residual'] != 99


@pytest.mark.parametrize('damage', ['arrival', 'goal', 'commands', 'clock', 'source', 'configuration'])
def test_restart_rejects_corrupt_success_instead_of_showing_old_arrival(tmp_path, damage):
    service, diagnostic = make(tmp_path)
    report = handle(service, diagnostic)
    if damage == 'arrival': report['original_target_reached'] = True
    elif damage == 'goal': report['drive_result']['policy_goal_readbacks'][0]['goal_raw']['gripper'] += 1
    elif damage == 'commands': report['commands_position'][-1][2] -= 1
    elif damage == 'clock': report['finished_s'] = True
    elif damage == 'source': report['dispatch_source']['registers']['P_Coefficient'] = 32
    else: report['configuration_unchanged_verified'] = False
    diagnostic.path.write_text(json.dumps(report))
    restored = J3NativeResponseDiagnostic(object(), output=tmp_path)
    status = restored.status()
    assert status['consumed'] and status['previous_result'] is None and status['display_error']
    assert not status['enabled']


@pytest.mark.parametrize('failed', [False, True])
def test_restart_keeps_incomplete_or_failed_attempt_without_inventing_arrival(tmp_path, failed):
    service, diagnostic = make(tmp_path)
    report = dict(schema='v21-j3-native-response-diagnostic-v1', command_budget=5,
                  completed=False, started_s=12)
    if failed:
        report.update(finished_s=13, error='recorded transport failure')
    diagnostic.path.write_text(json.dumps(report))
    restored = J3NativeResponseDiagnostic(object(), output=tmp_path)
    status = restored.status()
    previous = status['previous_result']
    assert status['consumed'] and status['display_error'] is None
    assert not previous['completed'] and previous['started_s'] == 12
    assert 'original_target_reached' not in previous and 'original_target_arrival' not in previous
    if failed:
        assert previous['error'] == 'recorded transport failure'
    assert not service.owner.bus.events and service.clears == service.stops == 0
