import threading
import time
from types import SimpleNamespace
import pytest

from fcpg.v21_hardware_state import ORDER
from test_v21_hardware_state import Bus
from test_v21_live_service import service  # noqa: F401


@pytest.mark.parametrize('phase_error', [False, True])
def test_actuator_query_is_read_only_and_preserves_trial_and_model_state(service, phase_error):
    phase_calls = []
    phase_values = {name: 32 + index for index, name in enumerate(ORDER)}
    class PhaseBus(Bus):
        def sync_read(self, register, normalize=True, num_retry=0):
            if register == 'Phase':
                phase_calls.append((normalize, num_retry))
                if phase_error:
                    raise RuntimeError('optional Phase unavailable')
                return phase_values.copy()
            return super().sync_read(register, normalize=normalize, num_retry=num_retry)
    bus=PhaseBus();service.owner.io.follower=SimpleNamespace(bus=bus)
    service.owner.bus_lock=threading.RLock();service.owner._lease_check=lambda:None
    state=service.owner.runtime.state
    budget=service.auxiliary_budget.status()
    result=service.handle('actuator-observe')
    assert result['actuator_state']['motor_register_writes']==0
    assert result['actuator_state']['present_minus_goal']['shoulder_lift']==1.
    telemetry=result['actuator_state']['servo_load_telemetry']
    assert telemetry['control_influence'] is False
    assert telemetry['physical_witness_valid'] is False  # fake bus has no optional load registers
    diagnostics=result['actuator_state']['diagnostics']
    assert diagnostics['motor_register_writes']==0
    assert phase_calls == [(False, 0)]
    if phase_error:
        assert diagnostics['registers']['Phase'] is None
        assert diagnostics['errors']['Phase'] == 'RuntimeError: optional Phase unavailable'
    else:
        assert diagnostics['registers']['Phase'] == phase_values
        assert 'Phase' not in diagnostics['errors']
    assert diagnostics['registers']['CW_Dead_Zone'] is None
    assert 'CW_Dead_Zone' in diagnostics['errors']
    assert diagnostics['registers']['P_Coefficient']['elbow_flex'] == 16
    assert diagnostics['registers']['I_Coefficient']['elbow_flex'] == 0
    assert diagnostics['registers']['D_Coefficient']['elbow_flex'] == 0
    assert diagnostics['registers']['Lock'] is None
    assert diagnostics['registers']['Model_Number'] is None
    assert 'Model_Number' in diagnostics['errors']
    assert diagnostics['registers']['Goal_Velocity'] is None
    assert 'Goal_Velocity' in diagnostics['errors']
    # Unsupported optional speed reads must not hide later acceleration evidence.
    assert diagnostics['registers']['Acceleration']['elbow_flex'] == 254
    assert diagnostics['registers']['Maximum_Acceleration']['elbow_flex'] == 254
    assert service.owner.runtime.state is state and service.auxiliary_budget.status()==budget
    assert service.owner.writes==0 and bus.events==[]


def test_actuator_query_does_not_run_beside_a_pending_plan(service):
    service.handle('plan')
    with pytest.raises(RuntimeError,match='requires STOP'):service.handle('actuator-observe')
    assert service.owner.writes==0


def attach_observation_bus(service):
    bus = Bus()
    bus.read_calls = []
    original = bus.sync_read
    def observed(register, **kwargs):
        bus.read_calls.append(register)
        return original(register, **kwargs)
    bus.sync_read = observed
    service.owner.io.follower = SimpleNamespace(bus=bus)
    service.owner.bus_lock = threading.RLock()
    service.owner._lease_check = lambda: None
    return bus


def start_call(function):
    call = SimpleNamespace(started=threading.Event(), done=threading.Event(), result=None,
        error=None, elapsed=None)
    def run():
        started = time.monotonic()
        call.started.set()
        try:
            call.result = function()
        except BaseException as exc:
            call.error = exc
        finally:
            call.elapsed = time.monotonic()-started
            call.done.set()
    call.thread = threading.Thread(target=run, daemon=True)
    call.thread.start()
    assert call.started.wait(1.)
    return call


def test_actuator_query_waits_for_short_monitor_readback(service, monkeypatch):
    bus = attach_observation_bus(service)
    state = service.owner.runtime.state
    budget = service.auxiliary_budget.status()
    in_read, release_read = threading.Event(), threading.Event()
    def slow_readback():
        in_read.set()
        assert release_read.wait(2.)
        return service.owner.q.copy()
    monkeypatch.setattr(service.owner, 'read_checked', slow_readback)
    def monitor_once():
        with service.operation_lock:
            service._monitor_readback_once()
    monitor = start_call(monitor_once)
    observation = None
    try:
        assert in_read.wait(1.)
        observation = start_call(lambda: service.handle('actuator-observe'))
        assert not observation.done.wait(.04)
        assert not bus.read_calls
        release_read.set()
        assert observation.done.wait(1.)
        assert observation.error is None
        assert observation.result['actuator_state']['motor_register_writes'] == 0
        assert bus.read_calls
        assert service.owner.runtime.state is state
        assert service.auxiliary_budget.status() == budget
        assert service.owner.writes == 0 and bus.events == []
    finally:
        release_read.set()
        monitor.thread.join(2.)
        if observation is not None:
            observation.thread.join(2.)
    assert monitor.error is None and not service.operation_lock.locked()


def test_visual_observe_waits_for_short_idle_monitor_lock(service):
    previous = service.owner.runtime.state
    service.operation_lock.acquire()
    observation = start_call(lambda: service.handle('observe'))
    try:
        assert not observation.done.wait(.04)
    finally:
        service.operation_lock.release()
    assert observation.done.wait(1.)
    observation.thread.join(2.)
    assert observation.error is None
    assert observation.result['phase'] == 'idle'
    assert service.owner.runtime.state is not previous
    assert service.owner.writes == 0 and not service.operation_lock.locked()


def test_visual_observe_wait_is_bounded_while_motion_remains_unqueued(service):
    service.operation_lock.acquire()
    observation = start_call(lambda: service.handle('observe'))
    try:
        motion = start_call(lambda: service.handle('execute'))
        assert motion.done.wait(3.)
        assert isinstance(motion.error, RuntimeError)
        assert 1.5 <= motion.elapsed < 3.
        assert observation.done.wait(1.)
        assert isinstance(observation.error, RuntimeError)
        assert 'another operation is active' in str(observation.error)
        assert .20 <= observation.elapsed < 1.
    finally:
        service.operation_lock.release()
        observation.thread.join(2.)
        motion.thread.join(2.)
    assert service.owner.writes == 0 and not service.operation_lock.locked()


def test_actuator_query_rejects_long_operation_without_register_reads(service):
    bus = attach_observation_bus(service)
    state = service.owner.runtime.state
    phase, error = service.phase, service.error
    service.operation_lock.acquire()
    observation = None
    try:
        observation = start_call(lambda: service.handle('actuator-observe'))
        assert observation.done.wait(1.)
        assert isinstance(observation.error, RuntimeError)
        assert 'another operation is active' in str(observation.error)
        assert .20 <= observation.elapsed < 1.
        assert service.operation_lock.locked()
        assert (service.phase, service.error) == (phase, error)
        assert service.owner.runtime.state is state
        assert service.owner.writes == 0 and bus.events == [] and not bus.read_calls
    finally:
        service.operation_lock.release()
        if observation is not None:
            observation.thread.join(2.)


def test_stop_and_motion_admission_remain_available_while_query_waits(service):
    bus = attach_observation_bus(service)
    service.operation_lock.acquire()
    observation = None
    try:
        observation = start_call(lambda: service.handle('actuator-observe'))
        assert not observation.done.wait(.03)
        motion = start_call(lambda: service.handle('execute'))
        assert motion.done.wait(3.)
        assert isinstance(motion.error, RuntimeError)
        assert 'another operation is active' in str(motion.error)
        stop = start_call(lambda: service.handle('stop'))
        assert stop.done.wait(.1)
        assert stop.error is None and service.owner.stop_event.is_set()
        assert service.owner.runtime.stopped
        assert not bus.read_calls and service.owner.writes == 0
    finally:
        service.operation_lock.release()
        if observation is not None:
            observation.thread.join(2.)
    assert observation.error is None
    assert observation.result['actuator_state']['motor_register_writes'] == 0


def test_waiting_actuator_query_rechecks_runtime_before_reading_registers(service):
    bus = attach_observation_bus(service)
    service.operation_lock.acquire()
    observation = None
    try:
        observation = start_call(lambda: service.handle('actuator-observe'))
        assert not observation.done.wait(.04)
        service.owner.runtime.stopped = False
    finally:
        service.operation_lock.release()
        if observation is not None:
            observation.thread.join(2.)
    assert observation.done.is_set()
    assert isinstance(observation.error, RuntimeError)
    assert 'requires STOP' in str(observation.error)
    assert not bus.read_calls and not bus.events and service.owner.writes == 0
    assert not service.operation_lock.locked()
