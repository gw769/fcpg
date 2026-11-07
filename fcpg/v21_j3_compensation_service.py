"""Disabled-by-default, one-use J3 compensation on the existing leased owner."""
from copy import deepcopy
import json
import os
from pathlib import Path
import threading

import numpy as np

from .v21_j3_compensation import (prepare_j3_compensation, bind_j3_compensation_wire,
                                  verify_j3_compensation_wire, validate_j3_compensation_recipe)
from .v21_j3_pid_adapter import ExistingOwnerJ3PIDAdapter
from .v21_j3_pid_service import DEFAULT_SEAL_PATH
from .v21_j3_range_service import J3RangeDiagnostic
from .v21_position_arrival import evaluate_position_arrival
from .v21_training import write_json
from scripts.prepare_v21_j3_pid_protocol import _config, MOTORS


class J3CompensationDiagnostic:
    def __init__(self, service, *, enabled=False, output):
        if type(enabled) is not bool:
            raise TypeError('compensation diagnostic enable must be a bool')
        self.service, self.enabled = service, enabled
        directory = Path(output).resolve()
        self.source_report = directory/'j3-range-diagnostic-10.json'
        self.path = directory/'j3-compensation-diagnostic.json'
        self.last_result = None
        self.consumed = False
        self.previous_result, self.display_error = self._load_previous_result()

    def status(self):
        return dict(enabled=self.enabled, consumed=self.consumed or self.path.exists(),
            source_report=str(self.source_report), compensated_amplitude=12.3,
            requires_expanded_amplitude_authorization=True, automatic_retry=False,
            last_result=deepcopy(self.last_result),
            previous_result=deepcopy(self.previous_result), display_error=self.display_error)

    def _load_previous_result(self):
        """Keep the saved actual outcome visible across restart, without restoring control state."""
        try:
            report = json.loads(self.path.read_text())
            if not isinstance(report, dict) or report.get('schema') != 'v21-j3-compensation-diagnostic-v1':
                raise ValueError('saved result is not a compensation diagnostic')
            if report.get('completed') is not True:
                return None, None
            if (report.get('configuration_unchanged_verified') is not True
                    or report.get('stopped') is not True
                    or report.get('model_history_imported') is not False):
                raise ValueError('saved compensation did not finish stopped with unchanged configuration')
            finished = report['finished_s']
            if type(finished) not in (int, float) or not np.isfinite(finished) or finished < 0:
                raise ValueError('saved compensation completion time is invalid')
            recipe = report['recipe']
            fixed = validate_j3_compensation_recipe(recipe)
            expected = prepare_j3_compensation(report['source_range10_report'], report['source']['q_position'])
            if fixed != expected:
                raise ValueError('saved compensation target differs from the original range target')
            drive = report['drive_result']
            self._check_receipt(drive, recipe)
            return dict(completed=True, finished_s=finished,
                original_target_arrival=deepcopy(drive['original_target_arrival']),
                original_target_reached=drive['original_target_reached'],
                evidence_source='saved_actual_receipt', saved_result_path=str(self.path),
                display_only=True, model_history_imported=False), None
        except FileNotFoundError:
            return None, None
        except (OSError, ValueError, TypeError, KeyError, IndexError, RuntimeError, OverflowError) as error:
            return None, f'{type(error).__name__}: {error}'

    @staticmethod
    def _check_receipt(drive, recipe):
        if not isinstance(drive, dict):
            raise RuntimeError('compensation returned no actual receipt')
        if (any(drive.get(key) is not True for key in ('completed', 'not_model_transition',
                'original_geometry_admitted', 'protective_hold_after_q5'))
                or drive.get('action_source') != 'j3_compensation_diagnostic'
                or drive.get('model_feedback_label') is not False
                or drive.get('gain_writes') != 0 or drive.get('maximum_lateness_s') != .03
                or drive.get('diagnostic_total_amplitude') != 12.3
                or drive.get('normal_follow', {}).get('budget_s') != .4):
            raise RuntimeError('compensation returned an incomplete fixed receipt')
        for key in ('policy_commands_attempted', 'policy_commands_returned',
                    'policy_goal_reads_attempted', 'policy_goal_reads_returned', 'policy_goal_registers_verified'):
            if type(drive.get(key)) is not int or drive[key] != 5:
                raise RuntimeError('compensation lacks five actual ' + key)
        values = {}
        for key, shape in (('commands_position', (5, 6)), ('source_q_position', (6,)),
                           ('measured_q_path', (6, 6)), ('readback_times_s', (6,)),
                           ('command_times_s', (5,)), ('write_completed_times_s', (5,))):
            values[key] = np.asarray(drive.get(key), dtype=np.float64)
            if values[key].shape != shape or not np.isfinite(values[key]).all():
                raise RuntimeError('compensation lacks finite actual ' + key)
        if (not np.allclose(values['commands_position'], recipe['wire']['commands'], atol=1e-4, rtol=0)
                or not np.allclose(values['source_q_position'], recipe['source_q'], atol=1e-4, rtol=0)
                or not np.array_equal(values['source_q_position'], values['measured_q_path'][0])
                or drive.get('original_target') != recipe['original_target']
                or drive.get('compensated_goal') != recipe['compensated_goal']):
            raise RuntimeError('compensation receipt belongs to different commands or target')
        qt, ct, wt = (values[key] for key in ('readback_times_s', 'command_times_s', 'write_completed_times_s'))
        if (not np.all(np.diff(qt) > 0) or not np.all(np.diff(ct) > 0)
                or not np.all((qt[:-1] <= ct) & (ct <= wt) & (wt <= qt[1:]))):
            raise RuntimeError('compensation command/readback clocks do not correspond')
        goals = drive.get('policy_goal_readbacks', [])
        if len(goals) != 5 or any(type(goal.get('command_index')) is not int
                                 or goal['command_index'] != i for i, goal in enumerate(goals)):
            raise RuntimeError('compensation lacks five ordered Goal readbacks')
        follow = drive['normal_follow']
        positions = np.asarray(follow.get('positions'), dtype=np.float64)
        times = np.asarray(follow.get('times_s'), dtype=np.float64)
        if (positions.shape != (8, 6) or times.shape != (8,)
                or not np.isfinite(positions).all() or not np.isfinite(times).all()
                or not np.all(np.diff(times) > 0) or times[0] <= qt[-1]
                or drive.get('completion_hold', {}).get('started_s', -1) < times[-1]):
            raise RuntimeError('compensation lacks full actual pre-HOLD following')
        arrival = evaluate_position_arrival(recipe['original_target'],
            [values['measured_q_path'][-1].tolist(), *positions.tolist()],
            [qt[-1] - qt[0], *(times - qt[0]).tolist()],
            position_tolerance=.2, step_tolerance=.1, quiet_samples=3)
        saved = drive.get('original_target_arrival', {})
        if (any(saved.get(key) != value for key, value in arrival.items())
                or drive.get('original_target_reached') is not arrival['reached']):
            raise RuntimeError('compensation arrival does not match actual original-target evidence')

    def handle(self):
        if not self.enabled:
            raise RuntimeError('J3 compensation diagnostic is disabled')
        service = self.service
        operation_lock = service.operation_lock
        if not operation_lock.locked():
            raise RuntimeError('compensation requires the existing operation lock')
        owner = service.owner
        if owner is None or getattr(owner, 'lease', None) is None:
            raise RuntimeError('compensation requires the existing leased owner')
        runtime, bridge = owner.runtime, owner.bridge
        if (not runtime.stopped or runtime.pending is not None or runtime.executing
                or not owner.stop_event.is_set()
                or getattr(runtime, 'requires_home_after_incomplete_execution', False)):
            raise RuntimeError('compensation requires STOP and no unfinished execution')
        if self.consumed or self.path.exists():
            raise RuntimeError('this compensation diagnostic has already been consumed')
        old_report = json.loads(self.source_report.read_text())
        # Reject missing/incomplete evidence before consuming or reading hardware.
        previous_path = old_report.get('drive_result', {}).get('measured_q_path', [])
        prepare_j3_compensation(old_report, previous_path[-1] if previous_path else None)
        thread, lease = threading.get_ident(), owner.lease
        original_state = getattr(runtime, 'state', None)
        entry_epoch = service.ros_control_admission_epoch()

        def admitted(operation):
            return (operation == 'read' and threading.get_ident() == thread
                and service.operation_lock is operation_lock and operation_lock.locked()
                and service.owner is owner and owner.lease is lease
                and owner.runtime is runtime and owner.bridge is bridge)

        def require_drive_admission(*, stop_latched):
            if not admitted('read'):
                raise RuntimeError('compensation original owner/lease/lock/runtime/bridge changed')
            if service.ros_control_admission_epoch() != entry_epoch:
                raise RuntimeError('compensation cancelled by STOP before drive')
            if (not runtime.stopped or runtime.pending is not None or runtime.executing
                    or owner.stop_event.is_set() is not stop_latched
                    or getattr(runtime, 'requires_home_after_incomplete_execution', False)):
                raise RuntimeError('compensation admission state changed')
            owner._lease_check()
            if not admitted('read') or service.ros_control_admission_epoch() != entry_epoch:
                raise RuntimeError('compensation admission changed during lease verification')

        reader = ExistingOwnerJ3PIDAdapter(owner, enabled=True, admission_check=admitted,
            stop_hold=service.stop, seal_path=DEFAULT_SEAL_PATH,
            reports_directory=self.path.parent, clock_fn=service.clock)

        def conversions():
            bus = owner.io.follower.bus
            ids = [bus.motors[name].id for name in MOTORS]
            def normalize(cells):
                values = bus._normalize(dict(zip(ids, map(int, cells))))
                return [values[key] for key in ids]
            def encode(position):
                values = bus._unnormalize(dict(zip(ids, map(float, position))))
                return [values[key] for key in ids]
            return normalize, encode

        def verify_wire(recipe):
            with owner.bus_lock:
                require_drive_admission(stop_latched=False)
                normalize, encode = conversions()
                return verify_j3_compensation_wire(recipe,
                    normalize_raw=normalize, encode_positions=encode)

        report = dict(schema='v21-j3-compensation-diagnostic-v1', completed=False,
            command_budget=5, period_s=.1, p=16, compensated_amplitude=12.3,
            requires_expanded_amplitude_authorization=True, gain_writes=0,
            source_report=str(self.source_report), source_range10_report=deepcopy(old_report),
            model_history_imported=False, automatic_retry=False, drive_attempted=False,
            started_s=float(service.clock()))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('x') as handle:
            self.consumed = True
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.last_result = report
        failure = None
        previous_attempt = getattr(bridge, 'last_attempt', None)
        try:
            source = reader.read_snapshot()
            q = J3RangeDiagnostic._check_source(source)
            if _config(source) != _config(old_report['source']):
                raise RuntimeError('configuration differs from the actual range10 source')
            recipe = prepare_j3_compensation(old_report, q)
            with owner.bus_lock:
                require_drive_admission(stop_latched=True)
                normalize, encode = conversions()
                recipe = bind_j3_compensation_wire(recipe, [source['present_raw'][name] for name in MOTORS],
                    normalize_raw=normalize, encode_positions=encode)
            report.update(source=deepcopy(source), recipe=deepcopy(recipe),
                original_target=recipe['original_target'], compensated_goal=recipe['compensated_goal'],
                commands_position=recipe['wire']['commands'])
            write_json(self.path, report)
            dispatch = reader.read_snapshot()
            current = J3RangeDiagnostic._check_source(dispatch)
            if (not np.allclose(current, q, atol=1e-4, rtol=0)
                    or dispatch['present_raw'] != source['present_raw']
                    or _config(dispatch) != _config(source)):
                raise RuntimeError('actual source or configuration changed before compensation')
            report['dispatch_source'] = deepcopy(dispatch)
            require_drive_admission(stop_latched=True)
            service._clear_stop_for_current_operation()
            require_drive_admission(stop_latched=False)
            report.update(drive_attempted=True, drive_started_s=float(service.clock()))
            drive = bridge.execute_j3_compensation_diagnostic(recipe, enable=True, verify_wire=verify_wire)
            report.update(drive_result=deepcopy(drive), drive_returned_s=float(service.clock()))
            self._check_receipt(drive, recipe)
            report['original_target_arrival'] = deepcopy(drive['original_target_arrival'])
            report['original_target_reached'] = drive['original_target_reached']
            if getattr(runtime, 'state', None) is not original_state:
                raise RuntimeError('compensation changed model history state')
            report['completed'] = True
        except BaseException as error:
            failure = error
            report['error'] = f'{type(error).__name__}: {error}'
            if report['drive_attempted']:
                attempt = getattr(error, 'last_attempt', None)
                if attempt is None and 'drive_result' not in report:
                    current_attempt = getattr(bridge, 'last_attempt', None)
                    if current_attempt is not previous_attempt:
                        attempt = current_attempt
                if isinstance(attempt, dict):
                    report['failed_drive_attempt'] = deepcopy(attempt)
                actual = report.get('drive_result') or report.get('failed_drive_attempt') or {}
                if actual.get('policy_commands_attempted', 0):
                    runtime.requires_home_after_incomplete_execution = True
                    report['requires_home_after_incomplete_execution'] = True
        finally:
            try:
                service.stop()
                final = reader.read_snapshot()
                report['final_source'] = deepcopy(final)
                source = report.get('source')
                if source is not None and _config(final) != _config(source):
                    raise RuntimeError('configuration changed during compensation')
                if final['registers']['P_Coefficient'] != 16:
                    raise RuntimeError('final compensation P16 readback failed')
                report['configuration_unchanged_verified'] = source is not None
                report['stopped'] = (owner.stop_event.is_set() and runtime.stopped
                    and runtime.pending is None and not runtime.executing)
                if not report['stopped']:
                    raise RuntimeError('compensation did not finish stopped and idle')
            except BaseException as error:
                report.update(cleanup_error=f'{type(error).__name__}: {error}', completed=False)
                if failure is None:
                    failure = error
            finally:
                try:
                    owner.stop_event.set()
                    runtime.stop()
                except BaseException as error:
                    report.update(stop_latch_error=f'{type(error).__name__}: {error}', completed=False)
                    if failure is None:
                        failure = error
                report['finished_s'] = float(service.clock())
                self.last_result = deepcopy(report)
                try:
                    write_json(self.path, report)
                except BaseException as error:
                    report['recording_error'] = f'{type(error).__name__}: {error}'
                    self.last_result = deepcopy(report)
                    if failure is None:
                        failure = error
        if failure is not None:
            raise failure
        return deepcopy(report)
