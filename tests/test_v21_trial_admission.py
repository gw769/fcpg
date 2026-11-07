"""STOP during trial admission never reaches fake policy writes or real IO."""
import threading

import numpy as np
import pytest

import fcpg.v21_live_service as live
from fcpg.v21_camera_condition import V21CameraCondition
from test_v21_live_service import service  # noqa: F401 -- fake IO only


@pytest.fixture
def admitted(service, monkeypatch):
    service.owner.read_checked = service.owner._read_q
    service.owner.io.settle_position_tolerance = 3.
    monkeypatch.setattr(live.established, '_load_v177_home_state',
                        lambda *args, **kwargs: service.owner.q.copy())
    return service


def assert_no_dispatch(service):
    service.trial.join()
    assert service.owner.writes == 0
    assert service.owner.stop_event.is_set()
    assert service.owner.runtime.stopped
    assert not service.trial.running
    assert not service.operation_lock.locked()


@pytest.mark.parametrize('stage', ['home_load', 'home_read'])
def test_stop_during_home_admission_does_not_start_a_worker(admitted, monkeypatch, stage):
    service = admitted
    original = (live.established._load_v177_home_state if stage == 'home_load'
                else service.owner.read_checked)

    def stop_during_precheck(*args, **kwargs):
        service.handle('stop')
        assert service.trial.cancelled.is_set() and service.owner.stop_event.is_set()
        return original(*args, **kwargs)

    if stage == 'home_load':
        monkeypatch.setattr(live.established, '_load_v177_home_state', stop_during_precheck)
    else:
        service.owner.read_checked = stop_during_precheck
    with pytest.raises(RuntimeError, match='admission cancelled by STOP'):
        service.handle('trial')
    assert service.trial.thread is None
    assert not list(service.output.glob('capped-trial-*.json'))
    assert_no_dispatch(service)


@pytest.mark.parametrize('stage', ['before_start', 'after_start', 'start_persistence'])
def test_stop_around_paused_start_survives_trial_latch_reset(admitted, monkeypatch, stage):
    service = admitted
    original_start = service.trial.start

    def start(*, paused=False):
        assert paused
        if stage == 'before_start':
            service.stop()
        original_start(paused=paused)
        if stage == 'after_start':
            service.stop()

    monkeypatch.setattr(service.trial, 'start', start)
    if stage == 'start_persistence':
        original_save = service.trial.save
        stopped = []

        def save(data):
            if not stopped:
                stopped.append(True)
                service.stop()
            original_save(data)

        monkeypatch.setattr(service.trial, 'save', save)
    with pytest.raises(RuntimeError, match='admission cancelled by STOP'):
        service.handle('trial')
    assert_no_dispatch(service)
    assert service.trial.status()['outcome'] == 'operator_stopped'
    assert service.trial.status()['blocks_started'] == 0


def test_stop_at_operation_unlock_is_checked_before_release(admitted, monkeypatch):
    service = admitted
    original = service.operation_lock

    class StopOnRelease:
        def acquire(self, *args, **kwargs):
            return original.acquire(*args, **kwargs)

        def locked(self):
            return original.locked()

        def release(self):
            original.release()
            service.stop()

    monkeypatch.setattr(service, 'operation_lock', StopOnRelease())
    with pytest.raises(RuntimeError, match='admission cancelled by STOP'):
        service.handle('trial')
    assert_no_dispatch(service)


def test_stop_during_operation_acquire_cannot_be_lost_before_token_publication(admitted, monkeypatch):
    service = admitted
    original = service.operation_lock

    class StopOnAcquire:
        def acquire(self, *args, **kwargs):
            acquired = original.acquire(*args, **kwargs)
            if acquired:
                service.stop()
            return acquired

        def locked(self):
            return original.locked()

        def release(self):
            original.release()

    monkeypatch.setattr(service, 'operation_lock', StopOnAcquire())
    with pytest.raises(RuntimeError, match='admission cancelled by STOP'):
        service.handle('trial')
    assert service.trial.thread is None
    assert_no_dispatch(service)


def test_rejected_concurrent_request_does_not_replace_an_admission_token(admitted, monkeypatch):
    service = admitted
    original_read = service.owner.read_checked

    def read():
        token = service._trial_admission_cancelled
        with pytest.raises(RuntimeError, match='another operation'):
            service.handle('trial')
        assert service._trial_admission_cancelled is token
        service.stop()
        return original_read()

    service.owner.read_checked = read
    with pytest.raises(RuntimeError, match='admission cancelled by STOP'):
        service.handle('trial')
    assert_no_dispatch(service)


def test_rejected_request_does_not_replace_a_started_trial_token(admitted, monkeypatch):
    service = admitted
    original = service.trial.start

    def start(*, paused=False):
        original(paused=paused)
        token = service._trial_admission_cancelled
        with pytest.raises(RuntimeError, match='owns motion'):
            service.handle('trial')
        assert service._trial_admission_cancelled is token
        service.stop()

    monkeypatch.setattr(service.trial, 'start', start)
    with pytest.raises(RuntimeError, match='admission cancelled by STOP'):
        service.handle('trial')
    assert_no_dispatch(service)


def test_cancelled_worker_cannot_admit_another_request_before_admission_returns(admitted, monkeypatch):
    service = admitted
    original = service.operation_lock
    checked = []

    class RetryOnRelease:
        def acquire(self, *args, **kwargs):
            return original.acquire(*args, **kwargs)

        def locked(self):
            return original.locked()

        def release(self):
            original.release()
            if not checked:
                checked.append(True)
                service.stop()
                service.trial.join()
                token = service._trial_admission_cancelled
                with pytest.raises(RuntimeError, match='another trial admission'):
                    service.handle('trial')
                assert service._trial_admission_cancelled is token

    monkeypatch.setattr(service, 'operation_lock', RetryOnRelease())
    with pytest.raises(RuntimeError, match='admission cancelled by STOP'):
        service.handle('trial')
    assert checked
    assert_no_dispatch(service)


def test_stop_after_final_check_before_gate_release_still_cancels_worker(admitted, monkeypatch):
    service = admitted
    original = service.trial.release_start

    def release():
        service.stop()
        original()

    monkeypatch.setattr(service.trial, 'release_start', release)
    service.handle('trial')
    assert_no_dispatch(service)
    assert service.trial.status()['outcome'] == 'operator_stopped'


def test_stop_does_not_wait_for_blocked_admission_operation(admitted):
    service = admitted
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def read():
        entered.set()
        assert release.wait(3.)
        return service.owner.q.copy()

    def admit():
        try:
            service.handle('trial')
        except BaseException as exc:
            errors.append(exc)

    def stop():
        service.handle('stop')
        stopped.set()

    service.owner.read_checked = read
    admitting = threading.Thread(target=admit)
    stopper = threading.Thread(target=stop)
    admitting.start()
    try:
        assert entered.wait(1.)
        assert service.operation_lock.locked()
        stopper.start()
        assert stopped.wait(1.), 'STOP waited for the trial admission operation'
        assert service.owner.stop_event.is_set()
    finally:
        release.set()
        admitting.join(3.)
        if stopper.ident is not None:
            stopper.join(3.)
    assert not admitting.is_alive() and not stopper.is_alive()
    assert len(errors) == 1 and 'admission cancelled by STOP' in str(errors[0])
    assert service.trial.thread is None
    assert_no_dispatch(service)


def test_stop_does_not_wait_for_start_record_persistence(admitted, monkeypatch):
    service = admitted
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    errors = []
    original_save = service.trial.save

    def save(data):
        if data['running']:
            entered.set()
            assert release.wait(3.)
        original_save(data)

    def admit():
        try:
            service.handle('trial')
        except BaseException as exc:
            errors.append(exc)

    def stop():
        service.stop()
        stopped.set()

    monkeypatch.setattr(service.trial, 'save', save)
    admitting = threading.Thread(target=admit)
    stopper = threading.Thread(target=stop)
    admitting.start()
    try:
        assert entered.wait(1.)
        stopper.start()
        assert stopped.wait(1.), 'STOP waited for the start record to finish writing'
        assert service.owner.stop_event.is_set()
    finally:
        release.set()
        admitting.join(3.)
        if stopper.ident is not None:
            stopper.join(3.)
    assert not admitting.is_alive() and not stopper.is_alive()
    assert len(errors) == 1 and 'admission cancelled by STOP' in str(errors[0])
    assert_no_dispatch(service)


def test_closed_loop_admission_is_home_empty_history_and_no_assist(admitted):
    """CLOSED_LOOP start: HOME + empty history + no pending + unmodified Actor.

    Camera-condition cells bind at start(); address must stay factual.
    """
    service = admitted
    assert not service.grounding_correction and not service.head_grasp
    assert service.owner.runtime.pending is None and not service.owner.runtime.executing
    assert service.owner.runtime.state is not None
    assert not bool(service.owner.runtime.state.history.valid.any())

    service.owner.runtime.state.history.valid[0, 0] = True
    with pytest.raises(RuntimeError, match='HOME and object reset'):
        service.handle('trial')
    service.owner.runtime.state.history.valid[:] = False

    service.owner.runtime.pending = object()
    with pytest.raises(RuntimeError, match='pending transaction'):
        service.handle('trial')
    service.owner.runtime.pending = None

    service.grounding_correction = True
    with pytest.raises(RuntimeError, match='unmodified Actor'):
        service.handle('trial')
    service.grounding_correction = False

    service.alignment_motion_enabled = True
    with pytest.raises(RuntimeError, match='unmodified Actor'):
        service.handle('trial')
    service.alignment_motion_enabled = False

    service.camera_condition = V21CameraCondition('both_fresh', 'zero')
    with pytest.raises(RuntimeError, match='factual address'):
        service.handle('trial')
    service.camera_condition = V21CameraCondition('both_fresh', 'factual')

    assert service.trial.thread is None
    assert not list(service.output.glob('capped-trial-*.json'))
    assert_no_dispatch(service)


@pytest.mark.parametrize('failure', ['home_configuration', 'outside_home'])
def test_failed_preflight_does_not_create_a_worker(admitted, monkeypatch, failure):
    service = admitted
    if failure == 'home_configuration':
        def missing(*args, **kwargs):
            raise ValueError('missing HOME configuration')
        monkeypatch.setattr(live.established, '_load_v177_home_state', missing)
    else:
        service.owner.read_checked = lambda: np.full(6, 20.)
    with pytest.raises((RuntimeError, ValueError)):
        service.handle('trial')
    assert service.trial.thread is None
    assert not list(service.output.glob('capped-trial-*.json'))
    assert_no_dispatch(service)


def test_a_later_explicit_trial_gets_its_own_admission_token(admitted, monkeypatch):
    service = admitted
    original = service.owner.read_checked

    def cancelled_read():
        service.stop()
        return original()

    service.owner.read_checked = cancelled_read
    with pytest.raises(RuntimeError, match='admission cancelled by STOP'):
        service.handle('trial')
    cancelled = service._trial_admission_cancelled
    service.owner.read_checked = original
    service.handle('trial')
    service.trial.join(15.)
    assert cancelled.is_set()
    assert service._trial_admission_cancelled is not cancelled
    assert service.owner.writes == 50
    assert service.trial.status()['outcome'] == 'budget_exhausted'
    assert not service.trial.status()['task_success_verified']
