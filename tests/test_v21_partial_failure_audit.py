"""Failure admission with synthetic clocks/IO only; never opens hardware."""
from dataclasses import replace

import pytest
import torch

from fcpg.v21_hardware_bridge import V21TransactionBridge
from test_v21_live_service import service  # noqa: F401


def install_bridge(service, monkeypatch, *, fail_after=None, stall_j3=False):
    owner = service.owner
    now = [service.clock()]
    writes = []
    failure = [fail_after]
    original_capture = owner.capture

    def capture():
        endpoint = original_capture()
        now[0] += .01
        stamp = torch.tensor([now[0]], dtype=torch.float64)
        endpoint = replace(endpoint, time=stamp, observation=replace(
            endpoint.observation, image_times=stamp[:, None].expand(-1, 2)))
        owner.last_endpoint = endpoint
        return endpoint

    def read():
        if failure[0] is not None and len(writes) >= failure[0]:
            raise RuntimeError('synthetic actual clearance failure')
        return owner.q.copy()

    def write(command, previous):
        writes.append(command.copy())
        owner.writes += 1
        owner.q = command.copy()
        if stall_j3:
            owner.q[2] = 0.
        return command.copy()

    def hold():
        owner.holds += 1
        return dict(event='safe_hold_verified', attempts=[dict(
            held_state=owner.q.copy(), mismatched_joints={})])

    def commands(state):
        result = state.q[:, None].repeat(1, 5, 1)
        result[:, :, 1:3] += torch.arange(1, 6)[None, :, None] * .2
        return result

    monkeypatch.setattr(service, 'clock', lambda: now[0])
    monkeypatch.setattr(owner.runtime, 'clock', lambda: now[0])
    monkeypatch.setattr(owner, 'capture', capture)
    bridge = V21TransactionBridge(owner.runtime, read_q=read, write_q=write,
        capture_endpoint=capture, physical_check=lambda q, u: None,
        halt=hold, request_stop=owner.stop_event.set,
        cancelled=owner.stop_event.is_set, clock=lambda: now[0],
        wait=lambda delay: now.__setitem__(0, now[0] + delay))
    owner.bridge = bridge
    service.lewm_cem = True
    monkeypatch.setattr(service, '_plan_with_policy', lambda **kwargs: bridge.plan(
        command_builder=commands, action_source='v21_lewm_cem'))
    return bridge, writes, failure


def test_partial_write_failure_latches_runtime_until_explicit_home(service, monkeypatch):
    bridge, writes, failure = install_bridge(service, monkeypatch, fail_after=2)
    service.handle('plan')
    with pytest.raises(RuntimeError, match='synthetic actual clearance'):
        service.handle('execute')
    assert len(writes) == 2
    assert bridge.last_attempt['completed'] is False
    assert bridge.last_attempt['policy_commands_returned'] == 2
    assert bridge.last_attempt['emergency_hold']['goal_register_verified'] is True
    assert not bool(service.owner.runtime.state.history.valid.any())
    assert service.owner.runtime.pending is None and service.owner.runtime.stopped

    # Finite safe telemetry alone must not erase the unmodelled partial motion.
    failure[0] = None
    service.owner.runtime.stop()
    service.owner.runtime.observe(service.owner.capture())
    assert service.owner.runtime.requires_home_after_incomplete_execution
    with pytest.raises(RuntimeError, match='(?i)home|incomplete|partial'):
        service.handle('plan')
    assert len(writes) == 2 and service.owner.runtime.pending is None


def test_rejected_cem_response_is_distinct_from_completed_transport(service, monkeypatch):
    bridge, writes, _ = install_bridge(service, monkeypatch, stall_j3=True)
    service.handle('plan')
    with pytest.raises(ValueError, match='actual.*response|respond'):
        service.handle('execute')
    assert len(writes) == 5
    # The complete, measured transition belongs in actual history, while the
    # failed response receipt must not authorize a continuation as a success.
    assert bool(service.owner.runtime.state.history.valid.any())
    assert bridge.last_attempt['completed'] is True
    assert service.last_attempt['response_verified'] is False
    assert service.last_attempt.get('response_error')
    assert not service.status()['ready']
    assert service.owner.runtime.stopped and service.owner.stop_event.is_set()


def test_attempted_write_without_return_still_requires_home(service, monkeypatch):
    bridge, writes, _ = install_bridge(service, monkeypatch)
    write = bridge.write_q

    def interrupted_write(command, previous):
        write(command, previous)  # hardware may change before its callback fails
        raise RuntimeError('synthetic partial register write')

    monkeypatch.setattr(bridge, 'write_q', interrupted_write)
    service.handle('plan')
    with pytest.raises(RuntimeError, match='partial register write'):
        service.handle('execute')
    assert len(writes) == 1
    assert bridge.last_attempt['policy_commands_attempted'] == 1
    assert bridge.last_attempt['policy_commands_returned'] == 0
    assert service.owner.runtime.requires_home_after_incomplete_execution


def test_failure_before_any_attempt_does_not_claim_partial_motion(service, monkeypatch):
    bridge, writes, failure = install_bridge(service, monkeypatch, fail_after=0)
    service.handle('plan')
    with pytest.raises(RuntimeError, match='synthetic actual clearance'):
        service.handle('execute')
    assert not writes
    assert bridge.last_attempt['policy_commands_attempted'] == 0
    assert bridge.last_attempt['policy_commands_returned'] == 0
    assert not service.owner.runtime.requires_home_after_incomplete_execution
    failure[0] = None
    service.handle('plan')
    assert service.status()['ready']


@pytest.mark.parametrize('outcome', ['accepted', 'response_rejected', 'dispatch_failed'])
def test_cem_chunk_gain_raise_is_written_back(service, monkeypatch, outcome):
    # Since the 2026-09-23 check-in a LeWM CEM chunk runs with P 32 on these
    # three joints; P 16 must be written back whatever the chunk ends with.
    install_bridge(service, monkeypatch, stall_j3=outcome == 'response_rejected',
                   fail_after=2 if outcome == 'dispatch_failed' else None)
    bus = service.owner.io.follower.bus
    service.handle('plan')
    if outcome == 'accepted':
        service.handle('execute')
    else:
        with pytest.raises((RuntimeError, ValueError)):
            service.handle('execute')
    raised = ('elbow_flex', 'wrist_flex', 'wrist_roll')
    assert sorted(motor for _, motor, value in bus.writes if value == 32) == sorted(raised)
    assert {motor for _, motor, _ in bus.writes} == set(raised)
    assert all(bus.values[('P_Coefficient', motor)] == 16 for motor in raised)
    assert [value for _, motor, value in bus.writes if motor == 'elbow_flex'] == [32, 16]
