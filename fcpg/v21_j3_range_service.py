"""One explicitly enabled ten-command J3 range diagnostic on the existing owner."""
from copy import deepcopy
import json
import os
from pathlib import Path
import threading

import numpy as np

from .v21_j3_pid_adapter import ExistingOwnerJ3PIDAdapter
from .v21_j3_pid_service import DEFAULT_SEAL_PATH
from .v21_j3_dynamic_telemetry import read_j3_dynamic_snapshot
from .v21_j3_range_profile import range_offsets, range_result_name
from .v21_position_arrival import evaluate_position_arrival
from .v21_training import write_json
from scripts.prepare_v21_j3_pid_protocol import ACTUAL_BASELINE, _config


OFFSETS = range_offsets()


class J3RangeDiagnostic:
    def __init__(self, service, *, enabled=False, output, amplitude=5):
        if type(enabled) is not bool:
            raise TypeError('range diagnostic enable must be a bool')
        self.offsets = range_offsets(amplitude)
        self.amplitude = int(amplitude)
        self.service, self.enabled = service, enabled
        self.path = Path(output).resolve()/range_result_name(amplitude)
        self.last_result = None
        self.consumed = False
        self.previous_result, self.display_error = self._load_previous_result()

    def status(self):
        return dict(enabled=self.enabled, consumed=self.consumed or self.path.exists(),
                    amplitude=self.amplitude, dynamic_telemetry_enabled=self.amplitude == 10,
                    last_result=deepcopy(self.last_result),
                    previous_result=deepcopy(self.previous_result), display_error=self.display_error)

    def _load_previous_result(self):
        """Reassess saved actual readbacks for display only; never restore control state."""
        previous, errors = None, []
        for amplitude in (5, 10):
            path = self.path.parent/range_result_name(amplitude)
            try:
                report = json.loads(path.read_text())
                if not isinstance(report, dict):
                    raise ValueError('saved diagnostic must be an object')
                if report.get('completed') is not True:
                    continue
                drive = report.get('drive_result')
                if (report.get('schema') != 'v21-j3-range-diagnostic-v1'
                        or report.get('amplitude', amplitude) != amplitude
                        or not isinstance(drive, dict) or drive.get('completed') is not True
                        or drive.get('action_source') != 'j3_range_diagnostic'):
                    raise ValueError('saved result is not a completed range diagnostic')
                finished = report['finished_s']
                if type(finished) not in (int, float) or not np.isfinite(finished) or finished < 0:
                    raise ValueError('saved finished_s must be finite and nonnegative')
                commands = np.asarray(report['commands_position'], dtype=np.float64)
                sent = np.asarray(drive['commands_position'], dtype=np.float64)
                source = np.asarray(drive['source_q_position'], dtype=np.float64)
                samples = np.asarray(drive['measured_q_path'], dtype=np.float64)
                times = np.asarray(drive['readback_times_s'], dtype=np.float64)
                if (commands.shape != (10, 6) or source.shape != (6,)
                        or samples.shape != (11, 6) or times.shape != (11,)
                        or not all(np.isfinite(value).all() for value in
                                   (commands, source, samples, times))
                        or not np.all(np.diff(times) > 0)):
                    raise ValueError('saved diagnostic lacks finite ordered ten-command readbacks')
                expected = np.repeat(source[None], 10, axis=0)
                expected[:, 2] += np.asarray(range_offsets(amplitude))
                if (not np.allclose(commands, expected, atol=1e-4, rtol=0)
                        or sent.shape != commands.shape
                        or not np.allclose(sent, commands, atol=1e-4, rtol=0)):
                    raise ValueError('saved original targets differ from the fixed range profile')
                self._check_receipt(drive, commands, source)
                follow = drive['normal_follow']
                if not isinstance(follow, dict):
                    raise ValueError('saved diagnostic lacks follow readbacks')
                arrival = evaluate_position_arrival(
                    commands[-1], [samples[-1].tolist(), *follow['positions']],
                    [times[-1]-times[0], *(stamp-times[0] for stamp in follow['times_s'])],
                    position_tolerance=.2, step_tolerance=.1, quiet_samples=3)
                arrival.update(position_unit='lerobot_calibrated_position',
                    source='actual_q10_and_follow_before_completion_hold',
                    source_time_s=float(times[0]), excludes_post_hold_readbacks=True,
                    control_influence=False)
                candidate = dict(completed=True, amplitude=amplitude, finished_s=finished,
                    original_target_arrival=arrival, original_target_reached=arrival['reached'],
                    evidence_source='saved_actual_receipt', saved_result_path=str(path),
                    display_only=True, model_history_imported=False)
                if previous is None or finished > previous['finished_s']:
                    previous = candidate
            except FileNotFoundError:
                continue
            except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError,
                    RuntimeError, AttributeError) as error:
                errors.append(f'{path.name}: {type(error).__name__}: {error}')
        return previous, '; '.join(errors) or None

    @staticmethod
    def _check_source(snapshot):
        q = np.asarray(snapshot['q_position'], dtype=np.float64)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('range diagnostic requires a finite actual six-axis source')
        if any(snapshot['registers'].get(k) != value for k, value in ACTUAL_BASELINE.items()):
            raise RuntimeError('range diagnostic requires the established P16 configuration')
        if snapshot['registers']['Moving'] != 0:
            raise RuntimeError('range diagnostic requires J3 to be stationary at entry')
        if snapshot.get('present_normalization_verified') is not True:
            raise RuntimeError('range diagnostic requires actual raw position normalization')
        return q

    @staticmethod
    def _check_receipt(drive, commands, source):
        if not isinstance(drive, dict):
            raise RuntimeError('range diagnostic returned no actual receipt')
        true_flags = ('completed', 'not_model_transition', 'original_geometry_admitted',
                      'protective_hold_after_q10')
        counters = ('policy_commands_attempted', 'policy_commands_returned',
                    'policy_goal_reads_attempted', 'policy_goal_reads_returned',
                    'policy_goal_registers_verified')
        if (any(drive.get(key) is not True for key in true_flags)
                or drive.get('action_source') != 'j3_range_diagnostic'
                or drive.get('model_feedback_label') is not False
                or type(drive.get('gain_writes')) is not int or drive['gain_writes'] != 0
                or drive.get('maximum_lateness_s') != .03
                or drive.get('normal_follow', {}).get('budget_s') != .4
                or any(type(drive.get(key)) is not int or drive[key] != 10 for key in counters)):
            raise RuntimeError('range diagnostic returned an incomplete ten-command receipt')
        values = {}
        for key, shape in (('measured_q_path', (11, 6)), ('commands_position', (10, 6)),
                           ('source_q_position', (6,)), ('readback_times_s', (11,)),
                           ('command_times_s', (10,)), ('write_completed_times_s', (10,))):
            values[key] = np.asarray(drive.get(key), dtype=np.float64)
            if values[key].shape != shape or not np.isfinite(values[key]).all():
                raise RuntimeError('range diagnostic lacks finite actual ' + key)
        if (not np.allclose(values['commands_position'], commands, atol=1e-4, rtol=0)
                or not np.allclose(values['source_q_position'], source, atol=1e-4, rtol=0)
                or not np.array_equal(values['source_q_position'], values['measured_q_path'][0])):
            raise RuntimeError('range diagnostic receipt belongs to different commands or actual q0')
        qt, ct, wt = (values[key] for key in
                      ('readback_times_s', 'command_times_s', 'write_completed_times_s'))
        if (not np.all(np.diff(qt) > 0) or not np.all(np.diff(ct) > 0)
                or not np.all((qt[:-1] <= ct) & (ct <= wt) & (wt <= qt[1:]))):
            raise RuntimeError('range diagnostic command/readback clocks do not correspond')
        goals = drive.get('policy_goal_readbacks', [])
        if len(goals) != 10 or any(type(goal.get('command_index')) is not int
                                  or goal['command_index'] != i for i, goal in enumerate(goals)):
            raise RuntimeError('range diagnostic lacks ten ordered Goal readbacks')

    def handle(self):
        if not self.enabled:
            raise RuntimeError('ten-command J3 diagnostic is disabled')
        service = self.service
        operation_lock = service.operation_lock
        if not operation_lock.locked():
            raise RuntimeError('range diagnostic requires the existing operation lock')
        owner = service.owner
        if owner is None or getattr(owner, 'lease', None) is None:
            raise RuntimeError('range diagnostic requires the existing leased owner')
        runtime, bridge = owner.runtime, owner.bridge
        if (not runtime.stopped or runtime.pending is not None or runtime.executing
                or not owner.stop_event.is_set()
                or getattr(runtime, 'requires_home_after_incomplete_execution', False)):
            raise RuntimeError('range diagnostic requires STOP and no unfinished execution')
        if self.consumed or self.path.exists():
            raise RuntimeError('this ten-command diagnostic has already been consumed')
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
                raise RuntimeError('range diagnostic original lock/owner/lease/runtime/bridge changed')
            if service.ros_control_admission_epoch() != entry_epoch:
                raise RuntimeError('range diagnostic cancelled by STOP before its drive')
            if (not runtime.stopped or runtime.pending is not None or runtime.executing
                    or owner.stop_event.is_set() is not stop_latched
                    or getattr(runtime, 'requires_home_after_incomplete_execution', False)):
                raise RuntimeError('range diagnostic operation state changed before its drive')
            owner._lease_check()
            if not admitted('read') or service.ros_control_admission_epoch() != entry_epoch:
                raise RuntimeError('range diagnostic admission changed during lease verification')
        reader = ExistingOwnerJ3PIDAdapter(owner, enabled=True, admission_check=admitted,
            stop_hold=service.stop, seal_path=DEFAULT_SEAL_PATH,
            reports_directory=self.path.parent, clock_fn=service.clock)
        report = dict(schema='v21-j3-range-diagnostic-v1', completed=False,
            command_budget=10, period_s=.1, offsets=list(self.offsets), p=16,
            amplitude=self.amplitude, dynamic_telemetry_enabled=self.amplitude == 10,
            gain_writes=0, model_history_imported=False, automatic_retry=False,
            drive_attempted=False, started_s=float(service.clock()))
        # This ordinary result file also records consumption across a restart.
        # No hardware callback or STOP clear precedes its creation.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('x') as handle:
            self.consumed = True
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self.last_result = report
        failure = None
        previous_attempt = getattr(bridge, 'last_attempt', None)
        try:
            source = reader.read_snapshot()
            q = self._check_source(source)
            commands = np.repeat(q[None], 10, axis=0)
            commands[:, 2] += np.asarray(self.offsets)
            report.update(source=deepcopy(source), commands_position=commands.tolist())
            write_json(self.path, report)
            # A new actual read after disk work binds these fixed commands.
            dispatch = reader.read_snapshot()
            current = self._check_source(dispatch)
            if not np.allclose(current, q, atol=1e-4, rtol=0) or _config(dispatch) != _config(source):
                raise RuntimeError('actual pose or configuration changed before fixed range commands')
            report['dispatch_source'] = deepcopy(dispatch)
            require_drive_admission(stop_latched=True)
            service._clear_stop_for_current_operation()
            require_drive_admission(stop_latched=False)
            report['drive_attempted'] = True
            report['drive_started_s'] = float(service.clock())
            if self.amplitude == 10:
                def telemetry(command_index, deadline_s):
                    with owner.bus_lock:
                        require_drive_admission(stop_latched=False)
                        return read_j3_dynamic_snapshot(owner.io.follower.bus, enable=True,
                            phase='after_goal_readback', command_index=command_index,
                            deadline_s=deadline_s, clock=service.clock,
                            cancelled=owner.stop_event.is_set)
                drive = bridge.execute_j3_range_diagnostic(commands, current, enable=True,
                    amplitude=10, telemetry=telemetry)
            else:
                drive = bridge.execute_j3_range_diagnostic(commands, current, enable=True)
            report.update(drive_result=deepcopy(drive), drive_returned_s=float(service.clock()))
            self._check_receipt(drive, commands, current)
            # Completing the fixed diagnostic is separate from reaching its
            # original target. Never replace this target with the STOP goal.
            report['original_target_arrival'] = deepcopy(drive.get('original_target_arrival'))
            report['original_target_reached'] = drive.get('original_target_reached')
            if getattr(runtime, 'state', None) is not original_state:
                raise RuntimeError('range diagnostic changed the model history state')
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
                actual_attempt = report.get('drive_result') or report.get('failed_drive_attempt') or {}
                if actual_attempt.get('policy_commands_attempted', 0):
                    runtime.requires_home_after_incomplete_execution = True
                    report['requires_home_after_incomplete_execution'] = True
        finally:
            # The existing STOP path holds the measured pose and preserves its
            # raw Goal readback. No gain or controller parameter is changed.
            try:
                service.stop()
                final = reader.read_snapshot()
                report['final_source'] = deepcopy(final)
                original = report.get('source')
                if original is not None and _config(final) != _config(original):
                    raise RuntimeError('actuator configuration changed during range diagnostic')
                if final['registers']['P_Coefficient'] != 16:
                    raise RuntimeError('final J3 P16 readback failed')
                report['configuration_unchanged_verified'] = original is not None
                report['stopped'] = (owner.stop_event.is_set() and runtime.stopped
                                     and runtime.pending is None and not runtime.executing)
                if not report['stopped']:
                    raise RuntimeError('range diagnostic did not finish stopped and idle')
            except BaseException as cleanup_error:
                report['cleanup_error'] = f'{type(cleanup_error).__name__}: {cleanup_error}'
                report['completed'] = False
                if failure is None:
                    failure = cleanup_error
            finally:
                try:
                    owner.stop_event.set()
                    runtime.stop()
                except BaseException as latch_error:
                    report['stop_latch_error'] = f'{type(latch_error).__name__}: {latch_error}'
                    report['completed'] = False
                    if failure is None:
                        failure = latch_error
                report['finished_s'] = float(service.clock())
                self.last_result = deepcopy(report)
                try:
                    write_json(self.path, report)
                except BaseException as recording_error:
                    report['recording_error'] = f'{type(recording_error).__name__}: {recording_error}'
                    self.last_result = deepcopy(report)
                    if failure is None:
                        failure = recording_error
        if failure is not None:
            raise failure
        return deepcopy(report)
