import numpy as np
import pytest

from test_v21_live_service import service  # noqa: F401
from fcpg.v21_live_service import V21LiveService

MONITOR_LOOP=V21LiveService._monitor_q


def test_idle_readback_uses_clearance_and_stationary_joint_checks(service):
    calls=[]
    service.owner.read_checked=lambda:(calls.append('clearance') or np.zeros(6))
    service.owner._physical_check=lambda q,u:calls.append(('physical',u.shape))
    service._monitor_readback_once()
    assert calls==['clearance',('physical',(5,6))]
    assert service.current_q==[0.]*6 and service.owner.writes==0


def test_invalid_idle_readback_never_publishes_a_valid_position(service):
    service.owner.read_checked=lambda:np.full(6,np.nan)
    service.current_q=None
    with pytest.raises(ValueError):service._monitor_readback_once()
    assert service.current_q is None and service.owner.writes==0


def test_idle_loop_latches_stop_on_existing_physical_guard_failure(service):
    class Once:
        count=0
        def wait(self,_):self.count+=1;return self.count>1
    old=service.closing
    service.closing=Once()
    def bad():raise RuntimeError('actual clearance below existing floor')
    service.owner.read_checked=bad
    try:MONITOR_LOOP(service)
    finally:service.closing=old
    assert service.owner.stop_event.is_set() and service.owner.runtime.stopped
    assert service.owner.holds==1 and 'clearance' in service.error
    assert not service.operation_lock.locked()
