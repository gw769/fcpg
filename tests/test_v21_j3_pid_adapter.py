from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from fcpg.v21_j3_pid_adapter import (
    ExistingOwnerJ3PIDAdapter, J3PIDAdapterError,
    create_existing_owner_j3_pid_callbacks,
)
from scripts.prepare_v21_j3_pid_protocol import (
    ACTUAL_BASELINE, ACTUAL_IDENTITY, HOME_PATH, IDENTITY, MOTOR, MOTORS,
    PID, PROTECTED, j3_targets,
)


class FakeClock:
    def __init__(self):
        self.now = 10.

    def __call__(self):
        return self.now


class Bus:
    def __init__(self, lock, clock):
        self.lock, self.clock = lock, clock
        self.motors = {name: SimpleNamespace(id=i+1) for i, name in enumerate(MOTORS)}
        self.values = {name: dict(ACTUAL_BASELINE, **ACTUAL_IDENTITY) for name in MOTORS}
        self.q = json.loads(HOME_PATH.read_text())['current_state']
        for i, name in enumerate(MOTORS):
            self.values[name].update(ID=i+1, Present_Position=2000+i, Goal_Position=2000+i, Moving=0)
        self.events = []
        self.failure = None
        self.on_read = None
        self.write_error = None
        self.normalization_inputs = []
        self.reconstruction_inputs = []
        self.inverse_raw = {i+1: 2000+i for i in range(6)}

    def _read_event(self, register, motor, normalize, num_retry):
        assert self.lock._is_owned()
        assert normalize is False and num_retry == 0
        self.events.append(('read', register, motor))
        self.clock.now += .0005
        if self.on_read:
            self.on_read(register, motor)
        if self.failure == (register, motor):
            raise ConnectionError('fake read outage')

    def read(self, register, motor, *, normalize, num_retry):
        self._read_event(register, motor, normalize, num_retry)
        return self.values[motor][register]

    def sync_read(self, register, *, normalize, num_retry):
        self._read_event(register, 'all', normalize, num_retry)
        return {name: self.values[name][register] for name in MOTORS}

    def _normalize(self, values):
        assert self.lock._is_owned()
        self.normalization_inputs.append(values.copy())
        return {i+1: value for i, value in enumerate(self.q)}

    def _unnormalize(self, values):
        assert self.lock._is_owned()
        self.reconstruction_inputs.append(values.copy())
        return self.inverse_raw.copy()

    def write(self, register, motor, value, *, normalize, num_retry):
        assert self.lock._is_owned()
        assert register == 'P_Coefficient' and motor == MOTOR
        assert normalize is False and num_retry == 0
        self.events.append(('write', register, motor, value))
        self.values[motor][register] = value
        if self.write_error:
            raise self.write_error


class Bridge:
    def __init__(self, runtime):
        self.runtime = runtime
        self.calls = []
        self.last_attempt = None
        self.error = None

    def execute_fixed_diagnostic(self, commands, source_q, *, enable=False):
        assert enable is True
        self.calls.append((commands.copy(), source_q.copy()))
        if self.error:
            self.last_attempt = dict(policy_commands_attempted=2, actual_only=True)
            raise self.error
        return dict(completed=True, fixed_diagnostic=True)


class Owner:
    def __init__(self):
        self.clock = FakeClock()
        self.bus_lock = threading.RLock()
        self.bus = Bus(self.bus_lock, self.clock)
        self.io = SimpleNamespace(follower=SimpleNamespace(bus=self.bus))
        self.runtime = SimpleNamespace(stopped=True, pending=None, executing=False)
        self.bridge = Bridge(self.runtime)
        self.stop_event = threading.Event()
        self.geometry = []
        self.clearance = SimpleNamespace(require_readback=self._check_geometry)
        self.lease_calls = 0
        self.stops = 0

    def _lease_check(self):
        self.lease_calls += 1

    def _check_geometry(self, q, *, label):
        self.geometry.append(q.copy())

    def stop(self):
        assert self.stop_event.is_set()
        self.stops += 1


def adapter_for(tmp_path, *, admission=None):
    owner = Owner()
    admitted = []
    def check(operation):
        admitted.append(operation)
        return True if admission is None else admission(operation)
    adapter = ExistingOwnerJ3PIDAdapter(owner, enabled=True, admission_check=check,
        stop_hold=owner.stop, seal_path=tmp_path/'stable-seal.json',
        reports_directory=tmp_path/'evidence', clock_fn=owner.clock)
    return owner, adapter, admitted


def sealed_report(**updates):
    value = dict(session_id='a'*32, motion_sealed=True, phase='baseline', events=[{'event': 'session_sealed'}])
    value.update(updates)
    return value


def prepare(tmp_path):
    owner, adapter, admitted = adapter_for(tmp_path)
    adapter.persist(sealed_report())
    source = adapter.read_snapshot()
    return owner, adapter, admitted, source


def test_default_callbacks_never_touch_foreign_owner_clock_or_files(tmp_path):
    class Forbidden:
        def __getattr__(self, name):
            pytest.fail('disabled callback touched owner: ' + name)
    def forbidden():
        pytest.fail('disabled callback touched clock')
    callbacks = create_existing_owner_j3_pid_callbacks(Forbidden(), clock_fn=forbidden)
    invocations = dict(read_snapshot=(), read_restore_guard=(), write_p=(32,),
        execute_five=([], 'baseline'), stop_hold=(), cancelled=(), persist=({},))
    for field in fields(callbacks):
        with pytest.raises(J3PIDAdapterError, match='disabled'):
            getattr(callbacks, field.name)(*invocations[field.name])
    assert not list(tmp_path.iterdir())


def test_enabled_factory_requires_outer_admission_and_stable_absolute_paths(tmp_path):
    with pytest.raises(ValueError, match='independent admission'):
        ExistingOwnerJ3PIDAdapter(enabled=True)
    with pytest.raises(ValueError, match='absolute stable'):
        ExistingOwnerJ3PIDAdapter(enabled=True, admission_check=lambda _: True,
            stop_hold=lambda: None, seal_path='relative', reports_directory=tmp_path)


def test_fresh_snapshot_uses_same_raw_present_sample_for_normalization_and_original_geometry(tmp_path):
    owner, adapter, _, source = prepare(tmp_path)
    assert source['q_position'] == owner.bus.q
    assert source['registers']['Present_Position'] == source['present_raw'][MOTOR]
    assert owner.bus.normalization_inputs == [{i+1: 2000+i for i in range(6)}]
    assert owner.bus.reconstruction_inputs == [{i+1: value for i, value in enumerate(owner.bus.q)}]
    assert source['present_reconstructed_raw'] == source['present_raw']
    assert source['present_normalization_verified'] and source['present_normalization_max_error_ticks'] == 0
    assert source['started_s'] <= source['q_time_s'] <= source['returned_s']
    assert source['source_geometry_validated'] and source['exclusive_owner'] and source['owner_idle']
    assert set(source['other_motor_configuration']) == set(MOTORS)-{MOTOR}
    assert all(set(values) == set((*PID, *PROTECTED)) for values in source['other_motor_configuration'].values())
    assert len(owner.geometry) == 1 and not [e for e in owner.bus.events if e[0] == 'write']


@pytest.mark.parametrize('motor', [MOTOR, 'gripper'])
def test_clipped_normalized_source_is_rejected_against_same_raw_sample(tmp_path, motor):
    owner, adapter, _ = adapter_for(tmp_path)
    owner.bus.values[motor]['Present_Position'] += 2
    with pytest.raises(J3PIDAdapterError, match='hides raw Present_Position'):
        adapter.read_snapshot()
    assert adapter.last_source is None and not owner.geometry
    assert not [event for event in owner.bus.events if event[0] == 'write']


@pytest.mark.parametrize('rounding', [-1, 1])
def test_one_raw_tick_reconstruction_rounding_preserves_existing_tolerance(tmp_path, rounding):
    owner, adapter, _ = adapter_for(tmp_path)
    owner.bus.inverse_raw[3] += rounding
    source = adapter.read_snapshot()
    assert source['present_normalization_verified']
    assert source['present_normalization_max_error_ticks'] == 1
    assert source['present_raw'][MOTOR] == 2002
    assert source['present_reconstructed_raw'][MOTOR] == 2002 + rounding


def test_failed_raw_reconstruction_cannot_reuse_an_earlier_source(tmp_path):
    owner, adapter, _, _ = prepare(tmp_path)
    owner.bus.values[MOTOR]['Present_Position'] += 10
    with pytest.raises(J3PIDAdapterError, match='hides raw Present_Position'):
        adapter.read_snapshot()
    assert adapter.last_source is None


def test_only_p_changes_with_guard_under_same_bus_lock_and_stop_does_not_block_restoration(tmp_path):
    owner, adapter, _, _ = prepare(tmp_path)
    initial = deepcopy(owner.bus.values)
    adapter.write_p(32)
    owner.stop_event.set()
    adapter.write_p(16)
    assert owner.bus.values == initial
    assert [e for e in owner.bus.events if e[0] == 'write'] == [
        ('write', 'P_Coefficient', MOTOR, 32), ('write', 'P_Coefficient', MOTOR, 16)]
    for i, event in enumerate(owner.bus.events):
        if event[0] == 'write':
            assert owner.bus.events[i-7:i] == [('read', name, MOTOR) for name in (*IDENTITY, 'Torque_Enable', 'Lock', 'Operating_Mode')]


def test_stop_during_candidate_guard_refuses_candidate_without_unlock_fallback(tmp_path):
    owner, adapter, _, _ = prepare(tmp_path)
    owner.bus.on_read = lambda register, motor: owner.stop_event.set() if register == 'Lock' else None
    with pytest.raises(J3PIDAdapterError, match='STOP during'):
        adapter.write_p(32)
    assert not [e for e in owner.bus.events if e[0] == 'write']


@pytest.mark.parametrize('field,value', [('ID', 4), ('Model_Number', 778), ('Lock', 0), ('Torque_Enable', 0), ('Operating_Mode', 1)])
def test_lost_identity_or_locked_torque_state_forbids_even_restore(tmp_path, field, value):
    owner, adapter, _, _ = prepare(tmp_path)
    adapter.write_p(32)
    owner.bus.values[MOTOR][field] = value
    with pytest.raises(J3PIDAdapterError):
        adapter.write_p(16)
    assert [e[-1] for e in owner.bus.events if e[0] == 'write'] == [32]


def test_other_motor_telemetry_outage_cannot_prevent_independent_p_restore_read(tmp_path):
    owner, adapter, _, _ = prepare(tmp_path)
    adapter.write_p(32)
    owner.stop_event.set()
    owner.bus.failure = ('P_Coefficient', 'all')
    with pytest.raises(ConnectionError):
        adapter.read_snapshot()
    guard = adapter.read_restore_guard()
    assert 'P_Coefficient' not in guard['registers'] and guard['exclusive_owner']
    adapter.write_p(16)
    assert adapter.read_restore_guard(include_p=True)['registers']['P_Coefficient'] == 16
    assert owner.stop_event.is_set()


def test_failed_ack_keeps_actual_attempt_record_for_later_recovery(tmp_path):
    owner, adapter, _, _ = prepare(tmp_path)
    owner.bus.write_error = ConnectionError('write may have reached motor')
    with pytest.raises(ConnectionError):
        adapter.write_p(32)
    assert owner.bus.values[MOTOR]['P_Coefficient'] == 32
    assert adapter.events[-1]['returned'] is False
    assert 'write may have reached motor' in adapter.events[-1]['error']
    owner.bus.write_error = None
    adapter.write_p(16)


def test_dispatch_calls_only_existing_fixed_bridge_with_exact_source_and_commands(tmp_path):
    owner, adapter, _, source = prepare(tmp_path)
    commands = j3_targets(source['q_position'])
    result = adapter.execute_five(commands, 'baseline')
    assert result == {'completed': True, 'fixed_diagnostic': True}
    assert len(owner.bridge.calls) == 1
    sent, q = owner.bridge.calls[0]
    np.testing.assert_array_equal(sent, np.array(commands, dtype=np.float32))
    np.testing.assert_array_equal(q, np.array(source['q_position'], dtype=np.float32))
    assert owner.runtime.stopped and owner.runtime.pending is None


def test_partial_dispatch_receipt_attaches_to_original_exception(tmp_path):
    owner, adapter, _, source = prepare(tmp_path)
    error = RuntimeError('physical dispatch failure')
    owner.bridge.error = error
    with pytest.raises(RuntimeError) as caught:
        adapter.execute_five(j3_targets(source['q_position']), 'variant')
    assert caught.value is error
    assert error.last_attempt == owner.bridge.last_attempt
    assert error.last_attempt is not owner.bridge.last_attempt


@pytest.mark.parametrize('fault', ['stale', 'stop', 'pending', 'command', 'seal'])
def test_dispatch_rejects_stale_cancelled_or_mismatched_inputs_before_bridge(tmp_path, fault):
    owner, adapter, _, source = prepare(tmp_path)
    commands = j3_targets(source['q_position'])
    if fault == 'stale':
        owner.clock.now += .11
    elif fault == 'stop':
        owner.stop_event.set()
    elif fault == 'pending':
        owner.runtime.pending = object()
    elif fault == 'command':
        commands[0][0] += .01
    elif fault == 'seal':
        adapter.seal_path.unlink()
    with pytest.raises((J3PIDAdapterError, FileNotFoundError)):
        adapter.execute_five(commands, 'baseline')
    assert not owner.bridge.calls


def test_rejected_outer_admission_precedes_any_hardware_read(tmp_path):
    owner, adapter, admitted = adapter_for(tmp_path, admission=lambda _: False)
    with pytest.raises(J3PIDAdapterError, match='admission refused'):
        adapter.read_snapshot()
    assert not owner.bus.events and owner.lease_calls == 0
    assert admitted == ['read']


def test_stop_cleanup_always_asserts_event_and_calls_existing_stop(tmp_path):
    owner, adapter, _ = adapter_for(tmp_path, admission=lambda _: False)
    adapter.stop_hold()
    assert owner.stop_event.is_set() and owner.stops == 1


def test_seal_and_evidence_are_durable_and_existing_session_cannot_be_replaced(tmp_path, monkeypatch):
    owner, adapter, _ = adapter_for(tmp_path)
    import fcpg.v21_j3_pid_adapter as module
    original = module.os.fsync
    syncs = []
    def tracked(fd):
        syncs.append(fd)
        return original(fd)
    monkeypatch.setattr(module.os, 'fsync', tracked)
    receipt = adapter.persist(sealed_report())
    assert receipt['saved'] and len(syncs) >= 5
    marker = json.loads(adapter.seal_path.read_text())
    assert marker['motion_sealed'] and marker['ordinary_policy_resume'] is False
    assert Path(receipt['report_path']).exists()
    with pytest.raises(J3PIDAdapterError, match='another diagnostic session'):
        adapter.persist(sealed_report(session_id='b'*32))
    assert json.loads(adapter.seal_path.read_text()) == marker
    assert not owner.bus.events


def test_each_intent_is_durable_without_rewriting_the_existing_seal(tmp_path, monkeypatch):
    owner, adapter, _, source = prepare(tmp_path)
    import fcpg.v21_j3_pid_adapter as module
    original = module._durable_json
    initial_marker = adapter.seal_path.read_bytes()
    calls = []
    def tracked(path, value):
        assert not adapter._durability_acknowledged
        # Even the preceding successful intent cannot authorize another write
        # while the next report has not finished its durable commit.
        with pytest.raises(J3PIDAdapterError, match='not acknowledged'):
            adapter.write_p(32)
        calls.append((path, deepcopy(value)))
        owner.clock.now += .07 if path == adapter.seal_path else .05
        return original(path, value)
    monkeypatch.setattr(module, '_durable_json', tracked)
    intent = sealed_report(source=source, commands_position=j3_targets(source['q_position']),
        events=[{'event': 'session_sealed'}, {'event': 'five_command_intent'}])
    started = owner.clock()
    receipt = adapter.persist(intent)
    assert owner.clock()-started == pytest.approx(.05)
    assert receipt['saved'] and adapter._durability_acknowledged
    assert adapter.seal_path.read_bytes() == initial_marker
    assert len(calls) == 1 and calls[0][0] == Path(receipt['report_path'])
    stored = json.loads(Path(receipt['report_path']).read_text())
    assert stored['source'] == source and stored['commands_position'] == intent['commands_position']
    final = deepcopy(intent)
    final['events'].append({'event': 'phase_finished_still_sealed'})
    finished = adapter.persist(final)
    assert [path for path, _ in calls[1:]] == [Path(finished['report_path']), adapter.seal_path]
    assert json.loads(adapter.seal_path.read_text())['report_path'] == finished['report_path']
    assert not [event for event in owner.bus.events if event[0] == 'write']


@pytest.mark.parametrize('fault', ['missing', 'foreign', 'corrupt', 'symlink'])
def test_later_persist_never_reconstructs_or_replaces_an_untrusted_seal(tmp_path, fault):
    owner, adapter, _, _ = prepare(tmp_path)
    if fault == 'missing':
        adapter.seal_path.unlink()
    elif fault == 'foreign':
        adapter.seal_path.write_text(json.dumps(dict(session_id='b'*32, motion_sealed=True)))
    elif fault == 'corrupt':
        adapter.seal_path.write_text('{')
    else:
        saved = tmp_path/'prior-seal.json'
        adapter.seal_path.rename(saved)
        adapter.seal_path.symlink_to(saved)
    before = set((adapter.reports_directory/('a'*32)).iterdir())
    with pytest.raises((J3PIDAdapterError, ValueError)):
        adapter.persist(sealed_report(events=[{'event': 'session_sealed'}, {'event': 'five_command_intent'}]))
    assert not adapter._durability_acknowledged
    assert set((adapter.reports_directory/('a'*32)).iterdir()) == before
    if fault == 'missing':
        assert not adapter.seal_path.exists()
    assert not [event for event in owner.bus.events if event[0] == 'write']


def test_a_new_adapter_cannot_adopt_an_existing_matching_seal(tmp_path):
    owner, adapter, _, _ = prepare(tmp_path)
    _, another, _ = adapter_for(tmp_path)
    with pytest.raises(J3PIDAdapterError, match='cannot be adopted'):
        another.persist(sealed_report())
    assert another.session_id is None and not another._durability_acknowledged


def test_final_pointer_failure_keeps_durable_report_and_blocks_new_writes(tmp_path, monkeypatch):
    owner, adapter, _, _ = prepare(tmp_path)
    import fcpg.v21_j3_pid_adapter as module
    original = module._durable_json
    initial_marker = adapter.seal_path.read_bytes()
    reports = []
    def fail_pointer(path, value):
        if path == adapter.seal_path:
            raise OSError('final pointer storage failed')
        result = original(path, value)
        reports.append(path)
        return result
    monkeypatch.setattr(module, '_durable_json', fail_pointer)
    final = sealed_report(events=[{'event': 'session_sealed'}, {'event': 'phase_finished_still_sealed'}])
    with pytest.raises(OSError, match='final pointer'):
        adapter.persist(final)
    assert len(reports) == 1 and json.loads(reports[0].read_text())['events'] == final['events']
    assert adapter.seal_path.read_bytes() == initial_marker
    assert not adapter._durability_acknowledged
    with pytest.raises(J3PIDAdapterError, match='not acknowledged'):
        adapter.write_p(32)
    assert not [event for event in owner.bus.events if event[0] == 'write']


def test_failed_report_write_leaves_seal_and_cannot_acknowledge_durability(tmp_path, monkeypatch):
    owner, adapter, _ = adapter_for(tmp_path)
    import fcpg.v21_j3_pid_adapter as module
    original = module._durable_json
    def fail_report(path, value):
        if path != adapter.seal_path:
            raise OSError('evidence disk failure')
        return original(path, value)
    monkeypatch.setattr(module, '_durable_json', fail_report)
    with pytest.raises(OSError, match='evidence disk failure'):
        adapter.persist(sealed_report())
    assert json.loads(adapter.seal_path.read_text())['motion_sealed']
    assert not owner.bus.events
    with pytest.raises(J3PIDAdapterError, match='not acknowledged'):
        adapter.write_p(32)


def test_cannot_write_before_first_durable_protocol_intent(tmp_path):
    owner, adapter, _ = adapter_for(tmp_path)
    with pytest.raises(J3PIDAdapterError, match='seal has not been established'):
        adapter.write_p(32)
    assert not owner.bus.events


def test_filesystem_seal_loss_after_candidate_does_not_block_original_p_restoration(tmp_path):
    owner, adapter, _, _ = prepare(tmp_path)
    adapter.write_p(32)
    adapter.seal_path.unlink()
    owner.stop_event.set()
    adapter.write_p(16)
    assert adapter.read_restore_guard(include_p=True)['registers']['P_Coefficient'] == 16
    assert owner.stop_event.is_set()


def test_seal_read_failure_after_failed_ack_does_not_block_p_restore(tmp_path, monkeypatch):
    owner, adapter, _, _ = prepare(tmp_path)
    owner.bus.write_error = ConnectionError('candidate applied but ACK lost')
    with pytest.raises(ConnectionError):
        adapter.write_p(32)
    original = Path.read_text
    def fail_seal(path, *args, **kwargs):
        if path == adapter.seal_path:
            raise OSError('seal storage unavailable')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', fail_seal)
    owner.bus.write_error = None
    owner.stop_event.set()
    adapter.write_p(16)
    assert adapter.read_restore_guard(include_p=True)['registers']['P_Coefficient'] == 16


def test_sealed_but_unused_adapter_has_no_parameter_recovery_authority(tmp_path):
    owner, adapter, _, _ = prepare(tmp_path)
    with pytest.raises(J3PIDAdapterError, match='possibly applied candidate'):
        adapter.write_p(16)
    assert not [e for e in owner.bus.events if e[0] == 'write']
