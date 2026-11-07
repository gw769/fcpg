"""Service admission for the prepared diagnostic, without constructing IO."""
import json
import threading
from types import SimpleNamespace

import pytest

from fcpg.v21_live_service import V21LiveService
from fcpg import v21_j3_pid_service as coordinator_module


def test_persistent_seal_blocks_startup_before_owner_or_model_construction(tmp_path, monkeypatch):
    seal = tmp_path/'seal.json'
    seal.write_text(json.dumps({'motion_sealed': True, 'session_id': '1'*32}))
    monkeypatch.setattr(coordinator_module, 'DEFAULT_SEAL_PATH', seal)
    calls = []
    def owner(*args, **kwargs):
        calls.append(None)
        pytest.fail('sealed startup must not construct a hardware owner')
    with pytest.raises(RuntimeError):
        V21LiveService(tmp_path/'no-model', tmp_path/'new-output', owner_factory=owner)
    assert not calls and not (tmp_path/'new-output').exists()


def service_stub():
    service = V21LiveService.__new__(V21LiveService)
    service.operation_lock = threading.Lock()
    service._trial_admission_lock = threading.Lock()
    service._trial_stop_generation = 0
    service.trial = SimpleNamespace(running=False)
    service.auxiliary_budget = SimpleNamespace(active=False)
    service._assistance = SimpleNamespace(abort=lambda: None)
    service._refresh_status = lambda: None
    service.status = lambda: {'phase': service.phase}
    return service


@pytest.mark.parametrize('phase', ['baseline', 'variant'])
def test_live_route_holds_existing_operation_lock(phase):
    service = service_stub()
    calls = []
    def handle(value):
        assert service.operation_lock.locked()
        calls.append(value)
    service.j3_pid_diagnostic = SimpleNamespace(require_motion_unsealed=lambda action: None,
                                                handle=handle)
    assert service.handle('j3-pid-'+phase, {}) == {'phase': 'idle'}
    assert calls == [phase] and not service.operation_lock.locked()


def test_live_route_rejects_custom_parameters_before_diagnostic_callback():
    service = service_stub()
    service.j3_pid_diagnostic = SimpleNamespace(require_motion_unsealed=lambda action: None,
        handle=lambda _: pytest.fail('custom diagnostic must not run'))
    with pytest.raises(ValueError, match='no parameters'):
        service.handle('j3-pid-variant', {'p': 64})
    assert not service.operation_lock.locked()


@pytest.mark.parametrize('phase', ['baseline', 'variant'])
def test_partial_diagnostic_readback_and_recovery_survive_restart(tmp_path, phase):
    service = service_stub()
    service.output = tmp_path
    service.owner = SimpleNamespace(runtime=SimpleNamespace(
        requires_home_after_incomplete_execution=False))
    attempt = {'completed': False, 'sent_commands': 2,
               'actual_q': [[0, -97, 96, 63, -54, 31]]}
    diagnostic = SimpleNamespace(require_motion_unsealed=lambda action: None,
                                 last_result=None)

    def fail_after_motion(_phase):
        service.owner.runtime.requires_home_after_incomplete_execution = True
        diagnostic.last_result = {'protocol_report': {'failed_drive_attempt': attempt}}
        raise RuntimeError('diagnostic readback interrupted')

    diagnostic.handle = fail_after_motion
    service.j3_pid_diagnostic = diagnostic
    with pytest.raises(RuntimeError, match='readback interrupted'):
        service.handle('j3-pid-'+phase, {})
    assert not service.operation_lock.locked()
    assert json.loads((tmp_path/'last-attempt.json').read_text()) == attempt
    assert service.last_attempt == attempt
    assert service.last_attempt is not attempt
    assert json.loads((tmp_path/'incomplete-execution.json').read_text())['requires_home'] is True
    restarted = service_stub()
    restarted.output = tmp_path
    restarted.owner = SimpleNamespace(runtime=SimpleNamespace(
        requires_home_after_incomplete_execution=False))
    restarted._restore_incomplete_execution()
    assert restarted.owner.runtime.requires_home_after_incomplete_execution is True


def test_completed_diagnostic_saves_actual_receipt_without_incomplete_latch(tmp_path):
    service = service_stub()
    service.output = tmp_path
    service.owner = SimpleNamespace(runtime=SimpleNamespace(
        requires_home_after_incomplete_execution=False))
    attempt = {'completed': True, 'sent_commands': 5,
               'actual_q': [[0, -97, 96, 63, -54, 31]]}
    service.j3_pid_diagnostic = SimpleNamespace(
        require_motion_unsealed=lambda action: None, handle=lambda phase: None,
        last_result={'protocol_report': {'drive_result': attempt}})
    service.handle('j3-pid-variant', {})
    assert json.loads((tmp_path/'last-attempt.json').read_text()) == attempt
    assert not (tmp_path/'incomplete-execution.json').exists()
    assert not service.operation_lock.locked()


def test_range_route_keeps_partial_actual_receipt_for_restart(tmp_path):
    service = service_stub()
    service.output = tmp_path
    service.owner = SimpleNamespace(runtime=SimpleNamespace(
        requires_home_after_incomplete_execution=False))
    service.j3_pid_diagnostic = SimpleNamespace(require_motion_unsealed=lambda action: None)
    attempt = {'completed': False, 'policy_commands_attempted': 7,
               'policy_commands_returned': 6, 'measured_q_path': [[0, -97, 95, 63, -54, 31]]}
    diagnostic = SimpleNamespace(last_result=None)
    def interrupt_range():
        assert service.operation_lock.locked()
        service.owner.runtime.requires_home_after_incomplete_execution = True
        diagnostic.last_result = {'failed_drive_attempt': attempt}
        raise RuntimeError('range action interrupted')
    diagnostic.handle = interrupt_range
    service.j3_range_diagnostic = diagnostic
    with pytest.raises(RuntimeError, match='range action interrupted'):
        service.handle('j3-range-diagnostic', {})
    assert json.loads((tmp_path/'last-attempt.json').read_text()) == attempt
    assert json.loads((tmp_path/'incomplete-execution.json').read_text())['requires_home'] is True
    assert not service.operation_lock.locked()


def test_range_route_refuses_custom_amplitude():
    service = service_stub()
    service.j3_pid_diagnostic = SimpleNamespace(require_motion_unsealed=lambda action: None)
    service.j3_range_diagnostic = SimpleNamespace(
        handle=lambda: pytest.fail('unexpected amplitude must not reach hardware'))
    with pytest.raises(ValueError, match='no parameters'):
        service.handle('j3-range-diagnostic', {'amplitude': 20})
    assert not service.operation_lock.locked()


def test_ordinary_home_cannot_skip_persistent_motion_seal():
    service = service_stub()
    def guard(_):
        raise RuntimeError('persistently sealed')
    service.j3_pid_diagnostic = SimpleNamespace(require_motion_unsealed=guard)
    service._home = lambda _: pytest.fail('HOME must respect the diagnostic seal')
    with pytest.raises(RuntimeError, match='sealed'):
        service.handle('home', {})


@pytest.mark.parametrize('action', ['home', 'plan', 'trial'])
def test_seal_created_after_ingress_is_rechecked_under_operation_lock(action):
    service = service_stub()
    inspected, resume = threading.Event(), threading.Event()
    sealed = [False]
    calls = []
    errors = []
    def guard(_):
        stale = sealed[0]
        calls.append(None)
        if len(calls) == 1:
            inspected.set()
            assert resume.wait(2)
        if stale:
            raise RuntimeError('new diagnostic seal')
    service.j3_pid_diagnostic = SimpleNamespace(require_motion_unsealed=guard)
    service._home = lambda _: pytest.fail('stale ingress cannot authorize motion')
    service._clear_stop_for_current_operation = lambda: pytest.fail('stale ingress cannot re-arm')
    def ordinary_request():
        try: service.handle(action, {})
        except BaseException as error: errors.append(error)
    worker = threading.Thread(target=ordinary_request)
    worker.start()
    assert inspected.wait(2)
    # Another admitted operation completes its durable seal before this older
    # request obtains the same service operation lock.
    with service.operation_lock:
        sealed[0] = True
    resume.set()
    worker.join(2)
    assert not worker.is_alive() and len(calls) == 2
    assert len(errors) == 1 and str(errors[0]) == 'new diagnostic seal'
    assert not service.operation_lock.locked()
