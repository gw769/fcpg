import threading

import pytest

from fcpg.v21_live_service import V21LiveService


class _Owner:
    def __init__(self):
        self.stop_event = threading.Event()
        self.bridge = type('Bridge', (), {'stop': lambda self: None})()


def _bare_service():
    service = object.__new__(V21LiveService)
    service._ros_control_epoch_lock = threading.RLock()
    service._ros_control_epoch = 0
    service._ros_control_local = threading.local()
    service.owner = _Owner()
    service._trial_admission_lock = threading.RLock()
    service._trial_stop_generation = 0
    service._trial_admission_cancelled = None
    service.trial = type('Trial', (), {'cancel': lambda self: None})()
    service._assistance = type('Assistance', (), {'abort': lambda self: None})()
    service._assistance_proposal_id = None
    service.operation_lock = threading.Lock()
    service._refresh_status = lambda: None
    service.stop_error = None
    return service


def test_ros_stop_invalidates_admission_before_a_stale_request_can_rearm():
    service = _bare_service()
    epoch = service.ros_control_admission_epoch()
    service._ros_control_local.admission_epoch = epoch
    service._clear_stop_for_current_operation()
    assert not service.owner.stop_event.is_set()

    service.stop()
    assert service.owner.stop_event.is_set()
    with pytest.raises(RuntimeError, match='cancelled by STOP'):
        service._clear_stop_for_current_operation()


def test_ros_ingress_rejects_a_request_invalidated_before_handle_entry():
    service = _bare_service()
    calls = []
    service.handle = lambda action, payload: calls.append((action, payload))
    epoch = service.ros_control_admission_epoch()
    service.stop()

    with pytest.raises(RuntimeError, match='cancelled by STOP'):
        service.handle_ros_control('home', {}, admission_epoch=epoch)
    assert calls == []
