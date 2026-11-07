"""Once-only same-Goal gain diagnostics on the service's existing owner.

The ordinary result file reserves the request before hardware access. This
module performs no planning, target compensation, camera/model inference or
model completion; a diagnostic invalidates old model history and ends at STOP.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

from .v21_j3_pid_adapter import _durable_directory, _durable_json, _sync_directory


READ_ONLY_ACTIONS = frozenset(('status', 'observe', 'actuator-observe', 'stop'))
SCHEMA = 'v21-same-goal-service-result-v1'
MODES = ('p32', 'i1')
DIAGNOSTIC_SCHEMA = 'v21-same-goal-gain-diagnostic-v1'


class SameGoalServiceError(RuntimeError):
    pass


def _run_diagnostic(*args, **kwargs):
    # Importing the service never constructs or arms the low-level diagnostic.
    from .v21_same_goal_integral import run_same_goal_integral
    return run_same_goal_integral(*args, **kwargs)


class SameGoalServiceCoordinator:
    def __init__(self, service, *, enabled=False, source_execution_path=None,
                 mode='p32', integral_max_seconds=2, output=None):
        if type(enabled) is not bool:
            raise TypeError('same-goal diagnostic enabled must be a bool')
        if mode not in MODES:
            raise ValueError('same-goal service supports only explicit p32 or i1')
        if (isinstance(integral_max_seconds, bool)
                or not isinstance(integral_max_seconds, (int, float))
                or integral_max_seconds not in ((2, 10) if mode == 'i1' else (2,))):
            raise ValueError('integral_max_seconds must be 2 for p32 or 2/10 for i1')
        if enabled and source_execution_path is None:
            raise ValueError('same-goal diagnostic requires an explicit source_execution_path')
        self.service, self.enabled, self.mode = service, enabled, mode
        self.integral_max_seconds = float(integral_max_seconds)
        self.source_execution_path = (None if source_execution_path is None else
                                      Path(source_execution_path).resolve())
        directory = Path(service.output if output is None else output).resolve()
        self.result_paths = {item: directory/f'same-goal-{item}-result.json' for item in MODES}
        self.path = self.result_paths[mode]
        if self.source_execution_path == self.path:
            raise ValueError('source execution must be separate from the diagnostic result')
        self.last_result = None
        self.consumed = False
        self._handling = threading.Lock()
        self._motion_blocked = False
        self.previous_results = {}
        self.previous_result, self.display_error = self._load_previous()

    def _load_previous(self):
        errors = []
        # A different selected mode must not erase unfinished restoration from
        # the previous process. Consumption, however, remains per result file.
        for mode, path in self.result_paths.items():
            if not os.path.lexists(path):
                continue
            try:
                if path.is_symlink() or not path.is_file():
                    raise ValueError('prior diagnostic receipt is not a regular file')
                record = json.loads(path.read_text())
                if not isinstance(record, dict) or record.get('schema') != SCHEMA:
                    raise ValueError('prior diagnostic receipt has an invalid schema')
                if record.get('mode') != mode:
                    raise ValueError('prior diagnostic receipt has an invalid mode')
                self.previous_results[mode] = record
                # A started/interrupted receipt cannot prove restoration.
                restored = record.get('cleanup', {}).get('restored') is True
                no_attempt = (record.get('diagnostic_attempted') is False
                              and record.get('state') == 'failed')
                safe = ((restored or no_attempt) and record.get('state') in ('completed', 'failed')
                    and record.get('motion_blocked') is False and record.get('stopped') is True)
                self._motion_blocked = self._motion_blocked or not safe
            except (OSError, ValueError, TypeError, AttributeError) as error:
                self._motion_blocked = True
                errors.append(f'{path.name}: {type(error).__name__}: {error}')
        return self.previous_results.get(self.mode), '; '.join(errors) or None

    def status(self):
        return dict(enabled=self.enabled, mode=self.mode, integral_max_seconds=self.integral_max_seconds,
            source_execution_path=None if self.source_execution_path is None else str(self.source_execution_path),
            result_path=str(self.path), consumed=self.consumed or os.path.lexists(self.path),
            running=self._handling.locked(), motion_blocked=self._motion_blocked,
            automatic_retry=False, automatic_home=False, model_transition=False,
            last_result=deepcopy(self.last_result), previous_result=deepcopy(self.previous_result),
            previous_results=deepcopy(self.previous_results),
            display_error=self.display_error)

    def require_motion_unsealed(self, action):
        if action in READ_ONLY_ACTIONS:
            return
        if self._motion_blocked:
            raise SameGoalServiceError('same-goal diagnostic restoration or actual-state verification '
                                      'is incomplete; ordinary motion is blocked')

    def _require_entry(self):
        service = self.service
        if not service.operation_lock.locked():
            raise SameGoalServiceError('same-goal diagnostic requires the held service operation_lock')
        owner = service.owner
        if owner is None or getattr(owner, 'lease', None) is None:
            raise SameGoalServiceError('same-goal diagnostic requires the existing leased owner')
        runtime, bridge = owner.runtime, owner.bridge
        if (runtime.stopped is not True or runtime.pending is not None or runtime.executing
                or not owner.stop_event.is_set()
                or getattr(runtime, 'requires_home_after_incomplete_execution', False)):
            raise SameGoalServiceError('same-goal diagnostic requires STOP and no unfinished model execution')
        if bridge.runtime is not runtime:
            raise SameGoalServiceError('same-goal diagnostic requires the original owner runtime')
        if (getattr(getattr(service, 'trial', None), 'running', False)
                or getattr(service, '_trial_admission_in_progress', False)
                or getattr(getattr(service, 'auxiliary_budget', None), 'active', False)):
            raise SameGoalServiceError('same-goal diagnostic requires no active trial or auxiliary experiment')
        if self.consumed or os.path.lexists(self.path):
            raise SameGoalServiceError('this same-goal diagnostic result is already consumed; no replay')
        self.require_motion_unsealed('same-goal-' + self.mode)
        return owner, runtime, bridge

    def _reserve(self, report):
        _durable_directory(self.path.parent)
        with self.path.open('x') as stream:
            self.consumed = True
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(self.path.parent)

    def _reset_actual_state(self, owner, runtime, report):
        """Drop unmodelled motion history; publish only one fresh checked q."""
        runtime.state = None
        self.service.plan_record = None
        self.service.preview_record = None
        self.service.candidate_preview_record = None
        self.service.last_execution = None
        self.service._last_head_observations = {}
        self.service._assistance_proposal_id = None
        self.service._carry = None
        self.service._geom_place = None
        assistance = getattr(self.service, '_assistance', None)
        if assistance is not None:
            assistance.reset()
        reset_condition = getattr(self.service, '_reset_camera_condition', None)
        if callable(reset_condition):
            reset_condition()
        self.service.memory_reset_reason = (f'same-goal {self.mode.upper()} diagnostic: old model history cleared; '
            'fresh actual q only, next explicit observation seeds a new H0')
        report.update(model_history_cleared=True, model_history_imported=False,
                      diagnostic_samples_added_to_model_y=False, model_transition=False,
                      next_explicit_observation_seeds_h0=True)
        owner._lease_check()
        q = np.asarray(owner.read_checked(), dtype=np.float64)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise SameGoalServiceError('same-goal cleanup requires a finite fresh actual q')
        owner._physical_check(q, np.repeat(q[None], 5, axis=0))
        stamp = float(self.service.clock())
        report['fresh_actual_state'] = dict(q_position=q.tolist(), time_s=stamp,
            source='fresh_checked_present_after_diagnostic', geometry_validated=True,
            model_history_imported=False, model_transition=False)
        with self.service.display_lock:
            self.service.current_q, self.service.current_q_time = q.tolist(), stamp

    def handle(self):
        if not self.enabled:
            raise SameGoalServiceError(f'same-goal {self.mode.upper()} diagnostic is disabled')
        if not self._handling.acquire(blocking=False):
            raise SameGoalServiceError('same-goal diagnostic is already running')
        try:
            return self._handle()
        finally:
            self._handling.release()

    def _handle(self):
        owner, runtime, bridge = self._require_entry()
        service = self.service
        source_bytes = self.source_execution_path.read_bytes()
        source = json.loads(source_bytes)
        if not isinstance(source, dict):
            raise SameGoalServiceError('source diagnostic input must be a saved JSON object')
        source_before = deepcopy(source)
        operation_lock, lease, thread = service.operation_lock, owner.lease, threading.get_ident()
        entry_epoch = service.ros_control_admission_epoch()

        def identity_matches():
            return (threading.get_ident() == thread and service.operation_lock is operation_lock
                and operation_lock.locked() and service.owner is owner and owner.lease is lease
                and owner.runtime is runtime and owner.bridge is bridge and bridge.runtime is runtime)

        def cancelled():
            return (not identity_matches() or owner.stop_event.is_set()
                    or service.ros_control_admission_epoch() != entry_epoch)

        def require_admission(stop_latched):
            if (not identity_matches() or service.ros_control_admission_epoch() != entry_epoch
                    or owner.stop_event.is_set() is not stop_latched
                    or runtime.stopped is not True or runtime.pending is not None or runtime.executing
                    or getattr(runtime, 'requires_home_after_incomplete_execution', False)):
                raise SameGoalServiceError('same-goal operation was cancelled or its owner admission changed')
            owner._lease_check()
            if not identity_matches() or service.ros_control_admission_epoch() != entry_epoch:
                raise SameGoalServiceError('same-goal owner admission changed during lease verification')

        report = dict(schema=SCHEMA, state='started', completed=False, mode=self.mode,
            integral_max_seconds=self.integral_max_seconds,
            source_execution_path=str(self.source_execution_path),
            diagnostic_attempted=False, cleanup={}, started_s=float(service.clock()),
            model_transition=False, model_history_imported=False, diagnostic_samples_added_to_model_y=False,
            automatic_retry=False, automatic_home=False, physical_task_success_verified=False,
            stopped=True, motion_blocked=True)
        self.last_result = report
        try:
            self._reserve(report)  # Disk evidence precedes every owner callback and STOP clear.
        except BaseException as error:
            self._motion_blocked = self.consumed
            report['persistence_error'] = f'{type(error).__name__}: {error}'
            raise
        failure = None
        try:
            with owner.bus_lock:
                require_admission(True)
                service._clear_stop_for_current_operation()
                require_admission(False)
                report['diagnostic_attempted'] = True
                report['diagnostic_started_s'] = float(service.clock())
                result = _run_diagnostic(owner, source, enabled=True, mode=self.mode,
                    integral_max_seconds=self.integral_max_seconds,
                    cancelled=cancelled, clock=service.clock, wait=getattr(service, 'same_goal_wait', time.sleep))
                if not isinstance(result, dict):
                    raise SameGoalServiceError('same-goal diagnostic returned an invalid report')
                report['diagnostic_result'] = deepcopy(result)
                if (result.get('completed') is not True
                        or result.get('schema') != DIAGNOSTIC_SCHEMA
                        or result.get('mode') != self.mode):
                    raise SameGoalServiceError('same-goal diagnostic returned without complete stage evidence')
                if not identity_matches() or service.ros_control_admission_epoch() != entry_epoch:
                    raise SameGoalServiceError('same-goal diagnostic was cancelled before service completion')
                if source != source_before:
                    raise SameGoalServiceError('same-goal helper changed its frozen source JSON')
                if (not isinstance(result.get('cleanup'), dict)
                        or result['cleanup'].get('restored') is not True):
                    raise SameGoalServiceError('same-goal diagnostic parameter restoration is incomplete')
                report['completed'] = True
        except BaseException as error:
            failure = error
            report['error'] = f'{type(error).__name__}: {error}'
            partial = getattr(error, 'report', None)
            if isinstance(partial, dict):
                report['diagnostic_result'] = deepcopy(partial)
        finally:
            result = report.get('diagnostic_result') or {}
            cleanup = result.get('cleanup')
            report['cleanup'] = deepcopy(cleanup) if isinstance(cleanup, dict) else {}
            for key in ('original_target', 'original_target_raw', 'original_target_arrival', 'original_target_reached'):
                if key in result:
                    report[key] = deepcopy(result[key])
            restored = (result.get('schema') == DIAGNOSTIC_SCHEMA
                and result.get('mode') == self.mode and report['cleanup'].get('restored') is True)
            self._motion_blocked = report['diagnostic_attempted'] and not restored
            try:
                owner.stop_event.set()
                if report['diagnostic_attempted']:
                    runtime.state = None
                runtime.stop()
                if report['diagnostic_attempted']:
                    if not identity_matches():
                        runtime.state = None
                        raise SameGoalServiceError('diagnostic owner changed before actual-state reset')
                    with owner.bus_lock:
                        self._reset_actual_state(owner, runtime, report)
                report['stopped'] = bool(owner.stop_event.is_set() and runtime.stopped
                                        and runtime.pending is None and not runtime.executing)
                if not report['stopped']:
                    raise SameGoalServiceError('same-goal diagnostic did not finish at STOP')
            except BaseException as error:
                self._motion_blocked = True
                report['actual_state_cleanup_error'] = f'{type(error).__name__}: {error}'
                if failure is None:
                    failure = error
            report.update(completed=bool(report['completed'] and failure is None),
                state='completed' if failure is None else 'failed', motion_blocked=self._motion_blocked,
                finished_s=float(service.clock()))
            if failure is not None and 'error' not in report:
                report['error'] = f'{type(failure).__name__}: {failure}'
            try:
                _durable_json(self.path, report)
            except BaseException as error:
                self._motion_blocked = True
                report.update(motion_blocked=True, persistence_error=f'{type(error).__name__}: {error}')
                if failure is None:
                    failure = error
            self.last_result = deepcopy(report)
        if failure is not None:
            raise failure
        return deepcopy(report)
