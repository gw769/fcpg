"""Disabled-by-default callbacks for a J3 experiment on an existing owner.

This module creates no controller, bus, camera, ROS route, or hardware lease.
The caller must own the service operation lock and enforce the stable seal on
every ordinary motion route, including after a restart. ``admission_check``
must return exactly True for the named operation only while that exclusive
admission is held. Reads, persistence, and recovery admission must survive
STOP; candidate writes and diagnostic motion must not. No callback clears STOP.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid

import numpy as np

from scripts.prepare_v21_j3_pid_protocol import (
    ACTUAL_IDENTITY, FRESHNESS_S, IDENTITY, J3PIDCallbacks, MOTOR, MOTORS,
    PID, PROTECTED, j3_targets,
)


class J3PIDAdapterError(RuntimeError):
    pass


def _raw_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise J3PIDAdapterError(label + ' must be a raw integer')
    return int(value)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_json(path, value):
    """Commit before returning; a failed write never authorizes hardware IO."""
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def _durable_directory(path):
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        _sync_directory(directory.parent)


class ExistingOwnerJ3PIDAdapter:
    """One protocol session, explicit existing-owner admission, no live default.

    ``stop_hold`` must be the existing service STOP/HOLD operation. The
    admission callback receives ``read``, ``restore_read``, ``gain_candidate``,
    ``gain_restore``, ``execute``, or ``persist``. It must check operation
    ownership and ordinary-motion sealing without acquiring another bus owner.
    Gain restoration deliberately ignores the diagnostic cancellation event;
    it still verifies the current lease and J3 identity/Torque1/Lock1/mode0.
    """

    def __init__(self, owner=None, *, enabled=False, admission_check=None,
                 stop_hold=None, seal_path=None, reports_directory=None,
                 clock_fn=time.monotonic):
        if type(enabled) is not bool:
            raise TypeError('enabled must be an explicit bool')
        if enabled and (not callable(admission_check) or not callable(stop_hold)):
            raise ValueError('enabled adapter requires independent admission and existing STOP callbacks')
        self.enabled = enabled
        self.owner = owner
        self.admission_check = admission_check
        self.stop_callback = stop_hold
        self.seal_path = None if seal_path is None else Path(seal_path)
        self.reports_directory = None if reports_directory is None else Path(reports_directory)
        if enabled and (self.seal_path is None or self.reports_directory is None
                        or not self.seal_path.is_absolute() or not self.reports_directory.is_absolute()):
            raise ValueError('enabled adapter requires explicit absolute stable seal and evidence paths')
        self.clock = clock_fn
        self.session_id = None
        self.last_source = None
        self.events = []
        self._durability_acknowledged = False
        self._candidate_write_may_have_applied = False
        self._persist_lock = threading.Lock()

    def callbacks(self):
        return J3PIDCallbacks(read_snapshot=self.read_snapshot, write_p=self.write_p,
            stop_hold=self.stop_hold, execute_five=self.execute_five,
            cancelled=self.cancelled, persist=self.persist,
            read_restore_guard=self.read_restore_guard)

    def _enabled(self):
        if not self.enabled:
            raise J3PIDAdapterError('J3 adapter disabled; no owner callback was invoked')

    def _admit(self, operation):
        self._enabled()
        if self.admission_check(operation) is not True:
            raise J3PIDAdapterError('existing-owner diagnostic admission refused: ' + operation)

    def _idle(self):
        runtime = self.owner.runtime
        if not runtime.stopped or runtime.pending is not None or runtime.executing:
            raise J3PIDAdapterError('diagnostic requires stopped runtime with no pending/executing transaction')
        if self.owner.bridge.runtime is not runtime:
            raise J3PIDAdapterError('diagnostic bridge must belong to this same owner runtime')

    def _seal(self):
        if self.session_id is None or self.seal_path is None:
            raise J3PIDAdapterError('durable diagnostic seal has not been established')
        if not self._durability_acknowledged:
            raise J3PIDAdapterError('last diagnostic evidence persistence was not acknowledged')
        record = json.loads(self.seal_path.read_text())
        if record.get('session_id') != self.session_id or record.get('motion_sealed') is not True:
            raise J3PIDAdapterError('stable diagnostic seal is missing or belongs to another session')

    def _single_read(self, bus, register):
        return _raw_integer(bus.read(register, MOTOR, normalize=False, num_retry=0), register)

    def _guard_locked(self, bus, *, include_p=False):
        registers = {name: self._single_read(bus, name)
                     for name in (*IDENTITY, 'Torque_Enable', 'Lock', 'Operating_Mode')}
        if any(registers[name] != value for name, value in ACTUAL_IDENTITY.items()):
            raise J3PIDAdapterError('J3 ID/model/firmware differs from pinned identity')
        if [registers[name] for name in ('Torque_Enable', 'Lock', 'Operating_Mode')] != [1, 1, 0]:
            raise J3PIDAdapterError('J3 recovery requires Torque1/Lock1/position mode')
        if include_p:
            registers['P_Coefficient'] = self._single_read(bus, 'P_Coefficient')
        return registers

    def read_restore_guard(self, *, include_p=False):
        self._admit('restore_read')
        if type(include_p) is not bool:
            raise TypeError('include_p must be bool')
        started = float(self.clock())
        self.owner._lease_check()
        with self.owner.bus_lock:
            registers = self._guard_locked(self.owner.io.follower.bus, include_p=include_p)
        return dict(started_s=started, returned_s=float(self.clock()), registers=registers,
                    exclusive_owner=True, motor_register_writes=0, num_retry=0)

    def read_snapshot(self):
        self._admit('read')
        self._idle()
        self.last_source = None
        started = float(self.clock())
        self.owner._lease_check()
        with self.owner.bus_lock:
            bus = self.owner.io.follower.bus
            registers = self._guard_locked(bus)
            configuration = {}
            for register in (*PID, *PROTECTED):
                values = bus.sync_read(register, normalize=False, num_retry=0)
                if not isinstance(values, dict) or set(values) != set(MOTORS):
                    raise J3PIDAdapterError('incomplete six-motor configuration: ' + register)
                configuration[register] = {name: _raw_integer(values[name], register + '/' + name)
                                           for name in MOTORS}
                registers[register] = configuration[register][MOTOR]
            registers['Goal_Position'] = self._single_read(bus, 'Goal_Position')
            registers['Moving'] = self._single_read(bus, 'Moving')
            raw = bus.sync_read('Present_Position', normalize=False, num_retry=0)
            q_time = float(self.clock())
            if not isinstance(raw, dict) or set(raw) != set(MOTORS):
                raise J3PIDAdapterError('incomplete actual Present_Position readback')
            raw = {name: _raw_integer(raw[name], 'Present_Position/' + name) for name in MOTORS}
            ids = {name: bus.motors[name].id for name in MOTORS}
            if len(set(ids.values())) != 6 or ids[MOTOR] != ACTUAL_IDENTITY['ID']:
                raise J3PIDAdapterError('bus motor identity mapping changed')
            normalized = bus._normalize({ids[name]: raw[name] for name in MOTORS})
            q = [float(normalized[ids[name]]) for name in MOTORS]
            if not all(math.isfinite(value) for value in q):
                raise J3PIDAdapterError('nonfinite actual position after bus normalization')
            reconstructed = bus._unnormalize({ids[name]: q[index] for index, name in enumerate(MOTORS)})
            if not isinstance(reconstructed, dict) or set(reconstructed) != set(ids.values()):
                raise J3PIDAdapterError('incomplete raw reconstruction of actual position')
            reconstructed = {name: _raw_integer(reconstructed[ids[name]], 'reconstructed/' + name)
                             for name in MOTORS}
            raw_error = max(abs(reconstructed[name] - raw[name]) for name in MOTORS)
            if raw_error > 1:
                raise J3PIDAdapterError('position normalization hides raw Present_Position by more than one tick')
            registers['Present_Position'] = raw[MOTOR]
            self.owner.clearance.require_readback(np.array(q, dtype=np.float32),
                                                  label='J3 PID diagnostic actual source')
        result = dict(started_s=started, q_time_s=q_time, returned_s=float(self.clock()),
            q_position=q, registers=registers,
            other_motor_configuration={name: {key: configuration[key][name] for key in (*PID, *PROTECTED)}
                                       for name in MOTORS if name != MOTOR},
            owner_idle=True, exclusive_owner=True, source_geometry_validated=True,
            present_raw=raw, q_provenance='same raw Present_Position sample normalized by existing bus',
            present_reconstructed_raw=reconstructed, present_normalization_max_error_ticks=raw_error,
            present_normalization_verified=True,
            motor_register_writes=0, num_retry=0)
        self.last_source = deepcopy(result)
        return result

    def _fresh_source(self):
        if self.last_source is None:
            raise J3PIDAdapterError('no fresh existing-owner source snapshot')
        age = float(self.clock()) - self.last_source['q_time_s']
        if not 0 <= age <= FRESHNESS_S:
            raise J3PIDAdapterError('diagnostic source snapshot is stale; no automatic reread')

    def write_p(self, value):
        self._enabled()
        if type(value) is not int or value not in (16, 32):
            raise ValueError('only J3 P32 candidate and P16 restoration are allowed')
        candidate = value == 32
        self._admit('gain_candidate' if candidate else 'gain_restore')
        if candidate:
            self._seal()
        elif self.session_id is None or not self._candidate_write_may_have_applied:
            raise J3PIDAdapterError('P restoration requires this admitted session with a possibly applied candidate')
        # Restoring the original P is independent of filesystem health. A
        # failed report/seal read after P32 must not leave P32 applied.
        self.owner._lease_check()
        with self.owner.bus_lock:
            if candidate and self.cancelled():
                raise J3PIDAdapterError('STOP before candidate P write')
            bus = self.owner.io.follower.bus
            guard = self._guard_locked(bus)
            if candidate:
                self._idle()
                self._fresh_source()
                if self.cancelled():
                    raise J3PIDAdapterError('STOP during candidate P guard')
            event = dict(kind='P_write', value=value, motor=MOTOR, register='P_Coefficient',
                normalize=False, num_retry=0, started_s=float(self.clock()), guard=guard,
                returned=False)
            self.events.append(event)
            if candidate:
                # A transport exception is not evidence that the write failed
                # to reach the device. Recovery authority precedes this call.
                self._candidate_write_may_have_applied = True
            try:
                bus.write('P_Coefficient', MOTOR, value, normalize=False, num_retry=0)
                event.update(returned=True, returned_s=float(self.clock()))
            except BaseException as error:
                event.update(error=f'{type(error).__name__}: {error}', failed_s=float(self.clock()))
                raise

    def execute_five(self, commands, phase):
        self._admit('execute')
        self._seal()
        self._idle()
        self._fresh_source()
        if phase not in ('baseline', 'variant'):
            raise ValueError('unknown J3 diagnostic phase')
        expected = np.array(j3_targets(self.last_source['q_position']), dtype=np.float32)
        actual = np.array(commands, dtype=np.float32)
        if actual.shape != (5, 6) or not np.isfinite(actual).all() or not np.array_equal(actual, expected):
            raise J3PIDAdapterError('diagnostic commands differ from the sealed source-relative J3 protocol')
        if self.cancelled():
            raise J3PIDAdapterError('STOP before fixed diagnostic dispatch')
        try:
            return self.owner.bridge.execute_fixed_diagnostic(actual,
                np.array(self.last_source['q_position'], dtype=np.float32), enable=True)
        except BaseException as error:
            attempt = getattr(self.owner.bridge, 'last_attempt', None)
            if isinstance(attempt, dict):
                error.last_attempt = deepcopy(attempt)
            raise

    def stop_hold(self):
        self._enabled()
        # STOP is cleanup: admission cancellation must never suppress it.
        self.owner.stop_event.set()
        return self.stop_callback()

    def cancelled(self):
        self._enabled()
        return self.owner.stop_event.is_set()

    def persist(self, report):
        self._admit('persist')
        if not isinstance(report, dict) or report.get('motion_sealed') is not True:
            raise ValueError('only a sealed diagnostic report may be persisted')
        session_id = report.get('session_id')
        if not isinstance(session_id, str) or re.fullmatch('[0-9a-f]{32}', session_id) is None:
            raise ValueError('diagnostic session ID must be a canonical UUID hex value')
        phase = report.get('phase')
        if phase not in ('baseline', 'variant') or not isinstance(report.get('events'), list):
            raise ValueError('diagnostic report lacks phase/event binding')
        with self._persist_lock:
            self._durability_acknowledged = False
            if self.session_id is not None and self.session_id != session_id:
                raise J3PIDAdapterError('adapter is already bound to another diagnostic session')
            if not self.seal_path.parent.is_dir():
                raise J3PIDAdapterError('stable seal parent must already exist before admission')
            seal_exists = self.seal_path.exists()
            if self.seal_path.is_symlink() or (seal_exists and not self.seal_path.is_file()):
                raise J3PIDAdapterError('stable seal must be a regular file')
            if self.session_id is not None and not seal_exists:
                raise J3PIDAdapterError('bound stable seal disappeared; no automatic reconstruction')
            if seal_exists:
                sealed = json.loads(self.seal_path.read_text())
                if sealed.get('session_id') != session_id or sealed.get('motion_sealed') is not True:
                    raise J3PIDAdapterError('existing stable seal cannot be replaced by a new session')
                if self.session_id is None:
                    raise J3PIDAdapterError('existing stable seal cannot be adopted by a new adapter')
            evidence = self.reports_directory / session_id
            _durable_directory(evidence)
            path = evidence / f'{phase}-{len(report["events"]):04d}.json'
            if path.exists():
                raise J3PIDAdapterError('diagnostic evidence event already exists; no overwrite')
            record = deepcopy(report)
            record['existing_owner_adapter_events'] = deepcopy(self.events)
            marker = dict(schema='v21-j3-pid-stable-seal-v1', motion_sealed=True,
                          session_id=session_id, report_path=str(path),
                          ordinary_policy_resume=False, phase=phase)
            # Establish the durable seal once. Every later event still checks
            # that same seal and durably commits its complete report before IO.
            # Rewriting an unchanged seal here spends the source freshness
            # budget on a second filesystem transaction without adding a guard.
            if not seal_exists:
                _durable_json(self.seal_path, marker)
                self.session_id = session_id
            digest = _durable_json(path, record)
            # Update the discovery pointer after the final report is durable,
            # outside the window between reading a source and dispatching it.
            if report['events'] and report['events'][-1].get('event') == 'phase_finished_still_sealed':
                _durable_json(self.seal_path, marker)
            self._durability_acknowledged = True
            return dict(saved=True, report_path=str(path), report_sha256=digest,
                        seal_path=str(self.seal_path), motion_sealed=True)


def create_existing_owner_j3_pid_callbacks(owner=None, **kwargs):
    """Return disabled callbacks unless the caller explicitly enables them."""
    return ExistingOwnerJ3PIDAdapter(owner, **kwargs).callbacks()
