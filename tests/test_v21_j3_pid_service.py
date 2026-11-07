import json
import threading

import pytest

from fcpg.v21_j3_pid_service import J3PIDServiceCoordinator, J3PIDServiceError
from scripts.prepare_v21_j3_pid_protocol import FakeJ3Bus, J3PIDCancelled
from test_v21_j3_pid_adapter import Owner


class Service:
    def __init__(self):
        self.owner = Owner()
        self.owner.lease = object()
        self.owner.stop_event.set()
        self.operation_lock = threading.Lock()
        self.clock = self.owner.clock
        self.clears = self.stops = self.epoch = 0
        self.clear_error = self.stop_error = None
        self.driver_calls = []
        self.owner.bridge.execute_fixed_diagnostic = self._drive

    def _clear_stop_for_current_operation(self):
        self.clears += 1
        if self.clear_error:
            raise self.clear_error
        self.owner.stop_event.clear()

    def stop(self):
        self.stops += 1
        self.epoch += 1
        self.owner.stop_event.set()
        self.owner.runtime.stopped = True
        if self.stop_error:
            raise self.stop_error

    def _drive(self, commands, source_q, *, enable=False):
        assert enable is True and self.operation_lock.locked()
        assert not self.owner.stop_event.is_set()
        self.driver_calls.append(commands.copy())
        fixture = FakeJ3Bus()
        fixture.q = source_q.tolist()
        fixture.now = self.clock.now
        record = fixture.execute_five(commands.tolist(), 'fixture')
        self.clock.now = fixture.now
        return record


def make_service(tmp_path, *, enabled=True):
    service = Service()
    coordinator = J3PIDServiceCoordinator(service, enabled=enabled,
        seal_path=tmp_path/'stable'/'seal.json', reports_directory=tmp_path/'reports')
    return service, coordinator


def handle(service, coordinator, phase):
    with service.operation_lock:
        return coordinator.handle(phase)


def test_disabled_handle_and_status_do_not_touch_owner_clock_factory_or_write_disk(tmp_path, monkeypatch):
    class Forbidden:
        def __getattr__(self, name):
            pytest.fail('disabled coordinator touched service: ' + name)
    def forbidden(*args, **kwargs):
        pytest.fail('disabled coordinator constructed callbacks')
    monkeypatch.setattr('fcpg.v21_j3_pid_service.create_existing_owner_j3_pid_callbacks', forbidden)
    coordinator = J3PIDServiceCoordinator(Forbidden(), seal_path=tmp_path/'seal.json',
        reports_directory=tmp_path/'reports')
    for phase in ('baseline', 'variant'):
        result = coordinator.handle(phase)
        assert result['dry_run'] and result['callbacks_invoked'] == 0
    assert not coordinator.status()['motion_sealed']
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('kind', ('json', 'malformed', 'directory', 'dangling_symlink'))
def test_disabled_guard_still_blocks_existing_seals_without_reading_owner(tmp_path, kind):
    class Forbidden:
        def __getattr__(self, name):
            pytest.fail('seal guard touched service')
    path = tmp_path/'seal.json'
    if kind == 'json': path.write_text(json.dumps(dict(motion_sealed=True, session_id='old')))
    elif kind == 'malformed': path.write_text('not json')
    elif kind == 'directory': path.mkdir()
    else: path.symlink_to(tmp_path/'absent')
    coordinator = J3PIDServiceCoordinator(Forbidden(), seal_path=path, reports_directory=tmp_path/'reports')
    for action in ('startup', 'home', 'execute', 'plan', 'trial', 'preview', 'j3-pid-baseline', 'j3-pid-variant'):
        with pytest.raises(J3PIDServiceError): coordinator.require_motion_unsealed(action)
    for action in ('observe', 'actuator-observe', 'stop', 'status'):
        coordinator.require_motion_unsealed(action)
    assert coordinator.status()['motion_sealed'] and not (tmp_path/'reports').exists()


def test_unheld_operation_lock_refuses_before_any_callback_or_disk_creation(tmp_path):
    service, coordinator = make_service(tmp_path)
    with pytest.raises(J3PIDServiceError, match='operation_lock'):
        coordinator.handle('baseline')
    assert not service.owner.bus.events and service.clears == service.stops == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('fault', ('pending', 'executing', 'not_stopped', 'unlatched', 'no_lease'))
def test_unsafe_entry_never_arms_or_writes(tmp_path, fault):
    service, coordinator = make_service(tmp_path)
    if fault == 'pending': service.owner.runtime.pending = object()
    elif fault == 'executing': service.owner.runtime.executing = True
    elif fault == 'not_stopped': service.owner.runtime.stopped = False
    elif fault == 'unlatched': service.owner.stop_event.clear()
    else: service.owner.lease = None
    with pytest.raises(J3PIDServiceError): handle(service, coordinator, 'baseline')
    assert not service.owner.bus.events and service.clears == service.stops == 0
    assert list(tmp_path.iterdir()) == []


def test_pending_baseline_blocks_motion_and_verified_terminal_variant_releases_block(tmp_path):
    service, coordinator = make_service(tmp_path)
    baseline = handle(service, coordinator, 'baseline')
    assert baseline['completed'] and baseline['p_restored_verified']
    source = baseline['protocol_report']['source']
    assert source['started_s'] < source['q_time_s'] <= source['returned_s']
    assert source['q_provenance'] == 'same raw Present_Position sample normalized by existing bus'
    assert source['returned_s'] - source['started_s'] < .1
    assert service.clears == service.stops == 1
    assert coordinator.experiment.phase == 'baseline_recorded'
    marker = json.loads(coordinator.seal_path.read_text())
    assert marker['motion_sealed'] and marker['session_id'] == coordinator.experiment.session_id
    for action in ('startup', 'execute', 'plan', 'home', 'j3-pid-baseline'):
        with pytest.raises(J3PIDServiceError): coordinator.require_motion_unsealed(action)
    coordinator.require_motion_unsealed('j3-pid-variant')
    result = handle(service, coordinator, 'variant')
    assert result['completed'] and result['p_restored_verified'] and result['configuration_restored_verified']
    assert not result['actual_response_verified']
    assert coordinator.experiment.phase == 'finished'
    assert coordinator.experiment.reserved_commands == 10
    assert service.clears == service.stops == 2 and len(service.driver_calls) == 2
    assert [event[-1] for event in service.owner.bus.events if event[0] == 'write'] == [32, 16]
    assert service.owner.stop_event.is_set() and service.owner.runtime.stopped
    assert not coordinator.status()['motion_sealed'] and coordinator.status()['last_result']['stopped']
    assert result['diagnostic_block_released'] and not coordinator.seal_path.exists()
    coordinator.require_motion_unsealed('plan')
    for phase in ('baseline', 'variant'):
        with pytest.raises(J3PIDServiceError): handle(service, coordinator, phase)
    assert service.clears == 2


def test_restart_never_rehydrates_baseline_or_replaces_persistent_marker(tmp_path):
    service, original = make_service(tmp_path)
    handle(service, original, 'baseline')
    contents = original.seal_path.read_bytes()
    restarted = J3PIDServiceCoordinator(service, enabled=True, seal_path=original.seal_path,
        reports_directory=original.reports_directory)
    for action in ('startup', 'j3-pid-baseline', 'j3-pid-variant'):
        with pytest.raises(J3PIDServiceError): restarted.require_motion_unsealed(action)
    for phase in ('baseline', 'variant'):
        with pytest.raises(J3PIDServiceError): handle(service, restarted, phase)
    assert original.seal_path.read_bytes() == contents
    assert restarted.experiment is None and service.clears == 1


@pytest.mark.parametrize('changed', ('owner', 'lease'))
def test_variant_cannot_reuse_callbacks_after_owner_or_lease_replacement(tmp_path, changed):
    service, coordinator = make_service(tmp_path)
    handle(service, coordinator, 'baseline')
    if changed == 'owner':
        service.owner = Owner()
        service.owner.lease = object()
        service.owner.stop_event.set()
    else:
        service.owner.lease = object()
    with pytest.raises(J3PIDServiceError, match='original baseline owner'):
        handle(service, coordinator, 'variant')
    assert service.clears == 1 and len(service.driver_calls) == 1


@pytest.mark.parametrize('mutation', ('missing', 'foreign', 'invalid'))
def test_variant_requires_exact_existing_session_marker_and_ordinary_motion_stays_sealed(tmp_path, mutation):
    service, coordinator = make_service(tmp_path)
    handle(service, coordinator, 'baseline')
    if mutation == 'missing': coordinator.seal_path.unlink()
    elif mutation == 'foreign': coordinator.seal_path.write_text(json.dumps(dict(motion_sealed=True, session_id='foreign')))
    else: coordinator.seal_path.write_text('invalid')
    with pytest.raises(J3PIDServiceError): handle(service, coordinator, 'variant')
    with pytest.raises(J3PIDServiceError): coordinator.require_motion_unsealed('execute')
    assert service.clears == 1 and len(service.driver_calls) == 1
    assert coordinator.status()['motion_sealed']


def test_stop_cancellation_after_p32_does_not_cancel_restore_admission(tmp_path):
    service, coordinator = make_service(tmp_path)
    handle(service, coordinator, 'baseline')
    original_write = service.owner.bus.write
    restoration_admitted = []
    def write(register, motor, value, **kwargs):
        original_write(register, motor, value, **kwargs)
        if value == 32:
            service.stop()  # External STOP invalidates epochs during this phase.
            restoration_admitted.append(coordinator._admission_check('gain_restore'))
    service.owner.bus.write = write
    with pytest.raises(J3PIDCancelled) as caught: handle(service, coordinator, 'variant')
    assert restoration_admitted == [True]
    result = caught.value.j3_pid_service_result
    assert result['p_restored_verified'] and result['configuration_restored_verified']
    assert result['stopped'] and not result['motion_sealed']
    assert result['diagnostic_block_released'] and not coordinator.seal_path.exists()
    assert service.owner.bus.values['elbow_flex']['P_Coefficient'] == 16
    assert len(service.driver_calls) == 1


def test_admission_is_bound_to_active_phase_thread_lock_owner_and_lease(tmp_path):
    service, coordinator = make_service(tmp_path)
    observed = []
    original_read = service.owner.bus.read
    def read(*args, **kwargs):
        if not observed:
            observed.append(coordinator._admission_check('restore_read'))
            answers = []
            thread = threading.Thread(target=lambda: answers.append(coordinator._admission_check('gain_restore')))
            thread.start(); thread.join(timeout=1.)
            assert not thread.is_alive()
            observed.extend(answers)
            lease = service.owner.lease
            service.owner.lease = object()
            observed.append(coordinator._admission_check('gain_restore'))
            service.owner.lease = lease
        return original_read(*args, **kwargs)
    service.owner.bus.read = read
    handle(service, coordinator, 'baseline')
    assert observed == [True, False, False]
    assert coordinator._admission_check('gain_restore') is False


def test_clear_stop_failure_never_invokes_register_callback_and_phase_cannot_retry(tmp_path):
    service, coordinator = make_service(tmp_path)
    service.clear_error = RuntimeError('stale ROS epoch')
    with pytest.raises(RuntimeError, match='stale ROS epoch'):
        handle(service, coordinator, 'baseline')
    assert service.clears == service.stops == 1 and not service.owner.bus.events
    assert coordinator.status()['motion_sealed'] and service.owner.stop_event.is_set()
    with pytest.raises(J3PIDServiceError): handle(service, coordinator, 'baseline')


def test_stop_failure_is_retained_without_retry_and_p16_restore_still_completes(tmp_path):
    service, coordinator = make_service(tmp_path)
    handle(service, coordinator, 'baseline')
    service.stop_error = RuntimeError('STOP transport fault')
    with pytest.raises(RuntimeError, match='STOP transport fault') as caught:
        handle(service, coordinator, 'variant')
    result = caught.value.j3_pid_service_result
    assert result['stop_error'] and result['p_restored_verified']
    assert service.stops == 2 and service.clears == 2
    assert result['stopped'] and result['motion_sealed']


def test_persist_failure_after_gain_write_cannot_block_recovery_admission(tmp_path, monkeypatch):
    service, coordinator = make_service(tmp_path)
    handle(service, coordinator, 'baseline')
    import fcpg.v21_j3_pid_adapter as adapter
    original = adapter._durable_json
    def broken(path, value):
        if value.get('phase') == 'variant' and value.get('events', [{}])[-1].get('event') == 'five_command_intent':
            assert value['events'][-2]['event'] == 'P32_readback_verified'
            assert value['source']['registers']['P_Coefficient'] == 32
            raise OSError('phase report disk fault')
        return original(path, value)
    monkeypatch.setattr(adapter, '_durable_json', broken)
    with pytest.raises(OSError, match='phase report disk fault') as caught:
        handle(service, coordinator, 'variant')
    result = caught.value.j3_pid_service_result
    assert result['p_restored_verified'] and result['configuration_restored_verified']
    assert result['stopped'] and not coordinator.seal_path.exists()
    assert result['diagnostic_block_released']
    assert service.owner.bus.values['elbow_flex']['P_Coefficient'] == 16


def test_new_stable_directory_is_durably_created_before_rearm(tmp_path, monkeypatch):
    service, coordinator = make_service(tmp_path)
    import fcpg.v21_j3_pid_adapter as adapter
    original = adapter._sync_directory
    synchronized = []
    def sync(path):
        synchronized.append((path, service.clears))
        original(path)
    monkeypatch.setattr(adapter, '_sync_directory', sync)
    handle(service, coordinator, 'baseline')
    assert (tmp_path, 0) in synchronized
    assert (coordinator.seal_path.parent, 1) in synchronized


def test_directory_durability_failure_never_rearms_or_reads_hardware(tmp_path, monkeypatch):
    service, coordinator = make_service(tmp_path)
    def broken(path):
        raise OSError('directory fsync unavailable')
    monkeypatch.setattr('fcpg.v21_j3_pid_service._durable_directory', broken)
    with pytest.raises(OSError, match='directory fsync unavailable'):
        handle(service, coordinator, 'baseline')
    assert service.clears == service.stops == 0 and not service.owner.bus.events
    assert service.owner.stop_event.is_set()


def test_inconsistent_runtime_after_stop_is_not_reported_as_complete(tmp_path):
    service, coordinator = make_service(tmp_path)
    original = service.stop
    def inconsistent_stop():
        original()
        service.owner.runtime.executing = True
    service.stop = inconsistent_stop
    with pytest.raises(Exception) as caught:
        handle(service, coordinator, 'baseline')
    result = caught.value.j3_pid_service_result
    assert not result['completed'] and not result['stopped']
    assert result['motion_sealed'] and service.owner.stop_event.is_set()
    assert service.stops == 1 and coordinator.status()['failed']


def completed_baseline_and_unexecuted_variant(tmp_path):
    """Synthetic service fixtures; no physical owner or model is constructed."""
    service, coordinator = make_service(tmp_path)
    handle(service, coordinator, 'baseline')
    original_write = service.owner.bus.write
    def stop_after_candidate(register, motor, value, **kwargs):
        original_write(register, motor, value, **kwargs)
        if value == 32:
            service.stop()
    service.owner.bus.write = stop_after_candidate
    with pytest.raises(J3PIDCancelled):
        handle(service, coordinator, 'variant')
    assert len(service.driver_calls) == 1 and not coordinator.seal_path.exists()
    directory = coordinator.reports_directory / coordinator.experiment.session_id
    final = {}
    for path in directory.glob('*.json'):
        record = json.loads(path.read_text())
        if record['events'][-1]['event'] == 'phase_finished_still_sealed':
            final[record['phase']] = path
    return service, coordinator, final


def continuing_service(original, final):
    service = Service()
    service.clock.now = original.service.clock.now + 5.
    coordinator = J3PIDServiceCoordinator(service, enabled=True,
        seal_path=original.seal_path, reports_directory=original.reports_directory,
        continuation_baseline_report=final['baseline'], continuation_variant_report=final['variant'])
    return service, coordinator


def test_explicit_continuation_uses_old_baseline_and_only_remaining_five_commands(tmp_path):
    prior_service, prior, final = completed_baseline_and_unexecuted_variant(tmp_path)
    service, coordinator = continuing_service(prior, final)
    state = service.owner.runtime.state = object()
    with pytest.raises(J3PIDServiceError, match='only the remaining variant'):
        handle(service, coordinator, 'baseline')
    assert not service.owner.bus.events and service.clears == 0
    result = handle(service, coordinator, 'variant')
    assert result['completed'] and result['diagnostic_block_released']
    assert result['stopped'] and not result['motion_sealed'] and not result['policy_resume']
    assert service.owner.runtime.state is state
    assert coordinator.experiment.session_id != prior.experiment.session_id
    assert coordinator.experiment.reserved_commands == 10
    assert len(prior_service.driver_calls) == len(service.driver_calls) == 1
    assert sum(len(commands) for commands in prior_service.driver_calls + service.driver_calls) == 10
    assert len(coordinator.experiment.reports) == 1 and coordinator.experiment.reports[0]['phase'] == 'variant'
    provenance = result['protocol_report']['comparison_continuation']
    assert provenance['baseline_report'] == str(final['baseline'])
    assert provenance['prior_actual_commands'] == provenance['remaining_actual_commands'] == 5
    assert provenance['model_history_imported'] is False
    for path in (coordinator.reports_directory / coordinator.experiment.session_id).glob('*.json'):
        assert json.loads(path.read_text())['comparison_continuation'] == provenance
    for phase in ('baseline', 'variant'):
        with pytest.raises(J3PIDServiceError):
            handle(service, coordinator, phase)
    assert len(service.driver_calls) == 1
    restarted_service = Service()
    with pytest.raises(J3PIDServiceError, match='already has a continuation attempt'):
        J3PIDServiceCoordinator(restarted_service, enabled=True,
            seal_path=prior.seal_path, reports_directory=prior.reports_directory,
            continuation_baseline_report=final['baseline'], continuation_variant_report=final['variant'])
    assert not restarted_service.owner.bus.events and restarted_service.clears == 0


@pytest.mark.parametrize('fault', ('baseline_incomplete', 'baseline_goal_mismatch', 'baseline_missing_q5',
    'baseline_wrong_commands', 'different_session', 'variant_unfinished', 'variant_drive_attempted',
    'variant_drive_returned', 'variant_drive_result', 'variant_failed_attempt',
    'variant_p_unverified', 'variant_configuration_unverified', 'variant_stop_error'))
def test_invalid_continuation_rejected_before_owner_callbacks(tmp_path, fault):
    _, prior, final = completed_baseline_and_unexecuted_variant(tmp_path)
    key = 'baseline' if fault.startswith('baseline_') else 'variant'
    record = json.loads(final[key].read_text())
    if fault == 'baseline_incomplete': record['five_command_receipt_validated'] = False
    elif fault == 'baseline_goal_mismatch': record['drive_result']['policy_goal_readbacks'][0]['goal_raw']['elbow_flex'] += 1
    elif fault == 'baseline_missing_q5': record['drive_result']['measured_q_path'].pop()
    elif fault == 'baseline_wrong_commands': record['drive_result']['commands_position'][0][0] += 1.
    elif fault == 'different_session': record['session_id'] = 'f' * 32
    elif fault == 'variant_unfinished': record['events'].pop()
    elif fault == 'variant_drive_attempted': record['drive_attempted'] = True
    elif fault == 'variant_drive_returned': record['drive_returned'] = True
    elif fault == 'variant_drive_result': record['drive_result'] = {}
    elif fault == 'variant_failed_attempt': record['failed_drive_attempt'] = {}
    elif fault == 'variant_p_unverified': record['p_restored_verified'] = False
    elif fault == 'variant_configuration_unverified': record['configuration_restored_verified'] = False
    elif fault == 'variant_stop_error': record['stop_error'] = 'unknown HOLD'
    final[key].write_text(json.dumps(record))
    class Forbidden:
        def __getattr__(self, name):
            pytest.fail('invalid continuation touched service: ' + name)
    with pytest.raises(J3PIDServiceError):
        J3PIDServiceCoordinator(Forbidden(), enabled=True,
            seal_path=prior.seal_path, reports_directory=prior.reports_directory,
            continuation_baseline_report=final['baseline'], continuation_variant_report=final['variant'])


def test_continuation_requires_both_reports_and_original_report_directory(tmp_path):
    _, prior, final = completed_baseline_and_unexecuted_variant(tmp_path)
    with pytest.raises(ValueError, match='both final'):
        J3PIDServiceCoordinator(None, enabled=True, seal_path=prior.seal_path,
            reports_directory=prior.reports_directory, continuation_baseline_report=final['baseline'])
    with pytest.raises(J3PIDServiceError, match='canonical diagnostic report directory'):
        J3PIDServiceCoordinator(None, enabled=True, seal_path=prior.seal_path,
            reports_directory=tmp_path/'new-reports', continuation_baseline_report=final['baseline'],
            continuation_variant_report=final['variant'])


def test_partial_variant_keeps_recovery_block_even_after_verified_restoration(tmp_path):
    service, coordinator = make_service(tmp_path)
    handle(service, coordinator, 'baseline')
    def partial(*args, **kwargs):
        service.owner.runtime.requires_home_after_incomplete_execution = True
        service.owner.bridge.last_attempt = dict(completed=False, policy_commands_attempted=1,
                                                  policy_commands_returned=0)
        raise RuntimeError('partial motion')
    service.owner.bridge.execute_fixed_diagnostic = partial
    with pytest.raises(RuntimeError, match='partial motion') as caught:
        handle(service, coordinator, 'variant')
    result = caught.value.j3_pid_service_result
    assert result['p_restored_verified'] and result['configuration_restored_verified']
    assert result['stopped'] and result['motion_sealed'] and not result['diagnostic_block_released']
    assert coordinator.seal_path.exists() and service.owner.runtime.requires_home_after_incomplete_execution


def test_existing_incomplete_execution_flag_prevents_terminal_unblock(tmp_path):
    service, coordinator = make_service(tmp_path)
    handle(service, coordinator, 'baseline')
    service.owner.runtime.requires_home_after_incomplete_execution = True
    result = handle(service, coordinator, 'variant')
    assert result['completed'] and result['p_restored_verified']
    assert result['motion_sealed'] and not result['diagnostic_block_released']
    assert coordinator.seal_path.exists()
