"""One fixed-P16 native-five response sample through the existing owner.

This coordinator has no controller, gains, arbitrary targets, HOME, retry,
training, or model-history path. Its durable result file consumes the request
before the first read. Incomplete response evidence remains incomplete.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import threading

import numpy as np
import torch

from .v21_j3_pid_adapter import ExistingOwnerJ3PIDAdapter, _durable_json
from .v21_j3_pid_service import DEFAULT_SEAL_PATH
from .v21_position_arrival import evaluate_position_arrival
from .v21_response_sampling import response_features
from scripts.prepare_v21_j3_pid_protocol import (
    ACTUAL_BASELINE, FRESHNESS_S, J3PIDExperiment, MOTORS, _config,
)


OFFSETS = (-1.5, -3., -3., -3., -3.)
ACTION_SOURCE = 'j3_native_response_diagnostic'


class J3NativeResponseDiagnostic:
    def __init__(self, service, *, enabled=False, output):
        if type(enabled) is not bool:
            raise TypeError('native response enable must be a bool')
        self.service, self.enabled = service, enabled
        self.path = Path(output).resolve() / 'native-response-diagnostic-3.json'
        self.last_result = None
        self.consumed = False
        self.previous_result, self.display_error = self._load_previous_result()

    def status(self):
        return dict(enabled=self.enabled, consumed=self.consumed or os.path.lexists(self.path),
                    amplitude=3, command_budget=5, automatic_retry=False,
                    last_result=deepcopy(self.last_result),
                    previous_result=deepcopy(self.previous_result), display_error=self.display_error)

    def _load_previous_result(self):
        """Reassess a saved receipt for display without touching the owner or admitting motion."""
        try:
            report = json.loads(self.path.read_text())
            if (not isinstance(report, dict)
                    or report.get('schema') != 'v21-j3-native-response-diagnostic-v1'
                    or type(report.get('completed')) is not bool
                    or type(report.get('command_budget')) is not int or report['command_budget'] != 5):
                raise ValueError('saved result is not a native five-command diagnostic')
            clocks = {}
            for key in ('started_s', 'finished_s'):
                value = report.get(key)
                if key == 'finished_s' and value is None and not report['completed']:
                    continue
                if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                    raise ValueError('saved native diagnostic has invalid ' + key)
                clocks[key] = value
            if clocks.get('finished_s', clocks['started_s']) < clocks['started_s']:
                raise ValueError('saved native diagnostic clocks are reversed')
            errors = {key: report[key] for key in ('error', 'cleanup_error', 'stop_latch_error',
                'recording_error', 'final_recording_error') if report.get(key)}
            previous = dict(completed=report['completed'], command_budget=5, **clocks, **errors,
                evidence_source='saved_actual_receipt', saved_result_path=str(self.path),
                display_only=True, model_history_imported=False, model_transition=False,
                training_ready=False)
            if not report['completed']:
                # An interrupted/failed attempt remains visible, but partial
                # evidence must not inherit an arrival claim from another run.
                return previous, None
            if (errors or report.get('configuration_unchanged_verified') is not True
                    or report.get('stopped') is not True
                    or report.get('model_history_imported') is not False
                    or report.get('model_transition') is not False
                    or report.get('gain_writes') != 0):
                raise ValueError('saved native diagnostic did not finish stopped with unchanged configuration')
            started, returned = report['drive_started_s'], report['drive_returned_s']
            if (any(type(value) not in (int, float) or not np.isfinite(value)
                    for value in (started, returned))
                    or not clocks['started_s'] <= started <= returned <= clocks['finished_s']):
                raise ValueError('saved native callback clocks are outside the diagnostic')
            dispatch = report['dispatch_source']
            source = self._check_source(dispatch, started)
            commands = np.asarray(report['commands_position'], dtype=np.float32)
            expected = np.repeat(np.asarray(source, np.float32)[None], 5, axis=0)
            expected[:, 2] += np.asarray(OFFSETS, np.float32)
            if (commands.shape != (5, 6) or not np.isfinite(commands).all()
                    or not np.allclose(commands, expected, atol=1e-4, rtol=0)):
                raise ValueError('saved native commands differ from the fixed five-step profile')
            drive = report['drive_result']
            self._check_receipt(drive, commands, dispatch, started_s=started, returned_s=returned)
            qt, follow = drive['readback_times_s'], drive['normal_follow']
            arrival = evaluate_position_arrival(commands[-1],
                [drive['measured_q_path'][-1], *follow['positions']],
                [qt[-1] - qt[0], *(stamp - qt[0] for stamp in follow['times_s'])],
                position_tolerance=.2, step_tolerance=.1, quiet_samples=3)
            saved = report.get('original_target_arrival', {})
            if (not isinstance(saved, dict) or any(saved.get(key) != value for key, value in arrival.items())
                    or report.get('original_target_reached') is not arrival['reached']):
                raise ValueError('saved native arrival differs from actual pre-HOLD readbacks')
            arrival.update(position_unit='lerobot_calibrated_position',
                source='actual_q5_and_follow_before_completion_hold', excludes_post_hold_readbacks=True,
                control_influence=False)
            previous.update(original_target_arrival=arrival, original_target_reached=arrival['reached'])
            return previous, None
        except FileNotFoundError:
            return None, None
        except (OSError, ValueError, TypeError, KeyError, IndexError, RuntimeError,
                OverflowError, AttributeError) as error:
            return None, f'{type(error).__name__}: {error}'

    @staticmethod
    def _check_source(snapshot, now):
        if not isinstance(snapshot, dict):
            raise ValueError('native response requires an actual source snapshot')
        q = np.asarray(snapshot.get('q_position'), dtype=np.float64)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('native response requires a finite actual six-axis source')
        times = np.asarray([snapshot.get(key) for key in ('started_s', 'q_time_s', 'returned_s')], dtype=np.float64)
        if (not np.isfinite(times).all() or not times[0] <= times[1] <= times[2] <= now
                or now - times[1] > FRESHNESS_S or times[2] - times[0] > FRESHNESS_S):
            raise RuntimeError('native response actual source is stale or has inconsistent clocks')
        if any(snapshot.get(key) is not True for key in (
                'owner_idle', 'exclusive_owner', 'source_geometry_validated', 'present_normalization_verified')):
            raise RuntimeError('native response requires original geometry and raw source normalization')
        regs = snapshot.get('registers', {})
        if any(regs.get(key) != value for key, value in ACTUAL_BASELINE.items()):
            raise RuntimeError('native response requires unchanged established P16 configuration')
        if regs.get('Moving') != 0 or abs(regs['Goal_Position'] - regs['Present_Position']) > 1:
            raise RuntimeError('native response requires stationary J3 held within one raw cell')
        raw = snapshot.get('present_raw')
        if (not isinstance(raw, dict) or set(raw) != set(MOTORS)
                or any(type(value) is not int for value in raw.values())):
            raise RuntimeError('native response requires all six actual raw Present cells')
        return q

    @staticmethod
    def _check_receipt(drive, commands, source, *, started_s, returned_s):
        if not isinstance(drive, dict):
            raise RuntimeError('native response returned no actual receipt')
        if (drive.get('action_source') != ACTION_SOURCE
                or drive.get('not_model_transition') is not True
                or drive.get('model_feedback_label') is not False
                or type(drive.get('gain_writes')) is not int or drive['gain_writes'] != 0):
            raise RuntimeError('native response receipt changed action or model/gain semantics')
        # Pure shared validation only: this does not construct/run a PID phase,
        # write a gain, inspect HOME, or create/release a PID motion seal.
        J3PIDExperiment._validate_drive(drive, source, commands.tolist(),
            callback_started_s=started_s, callback_returned_s=returned_s)
        qt = np.asarray(drive['readback_times_s'], dtype=np.float64)
        q = np.asarray(drive.get('source_q_position'), dtype=np.float64)
        if (q.shape != (6,) or not np.isfinite(q).all()
                or not np.array_equal(q, np.asarray(drive['measured_q_path'][0]))
                or drive.get('source_time_s') != qt[0]
                or drive.get('source_q_readback_time_s') != qt[0]):
            raise RuntimeError('native response lacks the actual q0/epoch binding')
        follow = drive['normal_follow']
        positions = np.asarray(follow.get('positions'), dtype=np.float64)
        times = np.asarray(follow.get('times_s'), dtype=np.float64)
        if (positions.shape != (8, 6) or times.shape != (8,)
                or not np.isfinite(positions).all() or not np.isfinite(times).all()
                or not np.all(np.diff(times) > 0) or times[0] <= qt[-1] or times[-1] > returned_s):
            raise RuntimeError('native response requires eight actual readbacks in the .4s follow window')
        deadlines = qt[0] + .5 + np.arange(1, 9) * .05
        if np.any(times < deadlines - 1e-9) or np.any(times - deadlines > .03 + 1e-9):
            raise RuntimeError('native response follow violated the original 30ms deadlines')

    @staticmethod
    def _context_summary(drive):
        context = drive.get('pre_dispatch_actuator_state')
        required = ('Present_Velocity', 'Present_Load', 'Present_Current',
                    'P_Coefficient', 'I_Coefficient', 'D_Coefficient', 'Operating_Mode')
        regs = context.get('registers', {}) if isinstance(context, dict) else {}
        missing = [name for name in required if regs.get(name) is None]
        if not isinstance(context, dict):
            missing.insert(0, 'pre_dispatch_actuator_state')
        elif not isinstance(context.get('owner_observation_history'), dict):
            missing.append('owner_observation_history')
        return dict(pre_dispatch_context_present=isinstance(context, dict), missing_fields=missing,
                    context_is_model_input=False, training_ready=False,
                    training_readiness_reason='diagnostic receipt only; no fit split or training admission',
                    missing_values_filled=False)

    def _save_source_state(self, runtime):
        """Freeze existing model inputs before reads/dispatch without inference."""
        if not getattr(self.service, 'record_policy_inputs', False):
            return dict(saved=False, executable=False,
                        missing_reason='record_policy_inputs is disabled or unavailable')
        from scripts.canary_so101_v21_research import cpu_actual_state
        source = getattr(runtime, 'state', None)
        text_ids = getattr(runtime, 'text_ids', None)
        if source is None or not isinstance(text_ids, torch.Tensor):
            raise RuntimeError('native response recording requires existing actual state and text_ids')
        if not isinstance(getattr(source, 'q', None), torch.Tensor) or not isinstance(getattr(source, 'time', None), torch.Tensor):
            raise RuntimeError('native response source lacks factual stored q/time tensors')
        started = float(self.service.clock())
        frozen = cpu_actual_state(source)
        path = self.path.with_name('native-response-source-3.pt')
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        reference = None
        object_provenance = None
        # The real ActualState owns visual memory. Small legacy/offline source
        # fixtures may only contain q/time; never invent images for either case.
        if hasattr(frozen, 'memory'):
            from .v21_source_evidence import snapshot_source_evidence, save_source_evidence
            bundle = snapshot_source_evidence(frozen,
                getattr(self.service.owner, 'source_evidence_store', None))
            reference = save_source_evidence(path.parent, path.stem, bundle)
            object_provenance = deepcopy(bundle.manifest['object_token'])
        payload = dict(kind='native-response-source', schema=1, executable=False,
            actual_future_labels_available=False, model_transition=False,
            state=frozen, text_ids=text_ids.detach().cpu().clone(),
            model_output=str(Path(self.service.root).resolve()),
            lewm_controller=deepcopy(getattr(self.service, 'lewm_controller', None)),
            temporal_contract=deepcopy(getattr(self.service, 'temporal_contract', None)),
            source_visual_evidence=deepcopy(reference), object_token_provenance=object_provenance,
            object_target_source=(None if object_provenance is None else object_provenance['source']),
            source_visual_evidence_missing_reason=(None if reference is not None
                else 'stored_state_has_no_visual_memory'),
            state_time_semantics='unchanged factual stored state time; not refreshed to diagnostic dispatch',
            planned_action_source=ACTION_SOURCE)
        with path.open('xb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return dict(saved=True, path=str(path),
                    executable=False, snapshot_started_s=started, saved_s=float(self.service.clock()),
                    stored_q_position=frozen.q.tolist(), stored_state_time_s=frozen.time.tolist(),
                    source_visual_evidence=deepcopy(reference),
                    state_time_rewritten=False, model_or_camera_called=False,
                    missing_reason=None)

    def handle(self):
        if not self.enabled:
            raise RuntimeError('native five-command J3 response diagnostic is disabled')
        service = self.service
        operation_lock = service.operation_lock
        if not operation_lock.locked():
            raise RuntimeError('native response requires the existing operation lock')
        owner = service.owner
        if owner is None or getattr(owner, 'lease', None) is None:
            raise RuntimeError('native response requires the existing leased owner')
        runtime, bridge = owner.runtime, owner.bridge
        if (not runtime.stopped or runtime.pending is not None or runtime.executing
                or not owner.stop_event.is_set()
                or getattr(runtime, 'requires_home_after_incomplete_execution', False)):
            raise RuntimeError('native response requires STOP and no unfinished execution')
        if self.consumed or os.path.lexists(self.path):
            raise RuntimeError('this native response diagnostic has already been consumed')
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
                raise RuntimeError('native response original lock/owner/lease/runtime/bridge changed')
            if service.ros_control_admission_epoch() != entry_epoch:
                raise RuntimeError('native response cancelled by STOP before drive')
            if (not runtime.stopped or runtime.pending is not None or runtime.executing
                    or owner.stop_event.is_set() is not stop_latched
                    or getattr(runtime, 'requires_home_after_incomplete_execution', False)):
                raise RuntimeError('native response operation state changed before drive')
            owner._lease_check()
            if not admitted('read') or service.ros_control_admission_epoch() != entry_epoch:
                raise RuntimeError('native response admission changed during lease verification')

        reader = ExistingOwnerJ3PIDAdapter(owner, enabled=True, admission_check=admitted,
            stop_hold=service.stop, seal_path=DEFAULT_SEAL_PATH,
            reports_directory=self.path.parent, clock_fn=service.clock)
        report = dict(schema='v21-j3-native-response-diagnostic-v1', completed=False,
            command_budget=5, period_s=.1, offsets=list(OFFSETS), p=16, amplitude=3,
            gain_writes=0, model_history_imported=False, model_transition=False,
            automatic_retry=False, automatic_home=False, drive_attempted=False,
            training_ready=False, started_s=float(service.clock()))
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
        self.last_result = deepcopy(report)
        failure = None
        previous_attempt = getattr(bridge, 'last_attempt', None)
        try:
            report['source_state_snapshot'] = self._save_source_state(runtime)
            source = reader.read_snapshot()
            q = self._check_source(source, float(service.clock()))
            commands = np.repeat(np.asarray(q, np.float32)[None], 5, axis=0)
            commands[:, 2] += np.asarray(OFFSETS, np.float32)
            report.update(source=deepcopy(source), commands_position=commands.tolist())
            _durable_json(self.path, report)
            # Disk latency does not use the old source. A second actual read
            # must match the exact raw pose and all protected configuration.
            dispatch = reader.read_snapshot()
            current = self._check_source(dispatch, float(service.clock()))
            if (not np.allclose(current, q, atol=1e-4, rtol=0)
                    or dispatch['present_raw'] != source['present_raw']
                    or _config(dispatch) != _config(source)):
                raise RuntimeError('actual pose or configuration changed before native commands')
            report['dispatch_source'] = deepcopy(dispatch)
            saved_source = report['source_state_snapshot']
            if saved_source.get('saved'):
                stored_q = np.asarray(saved_source['stored_q_position'], dtype=np.float64)
                saved_source['dispatch_q_max_abs_difference'] = (
                    float(np.abs(stored_q[0] - current).max()) if stored_q.shape == (1, 6) else None)
                saved_source['dispatch_state_time_s_is_separate'] = True
            require_drive_admission(stop_latched=True)
            service._clear_stop_for_current_operation()
            require_drive_admission(stop_latched=False)
            self._check_source(dispatch, float(service.clock()))
            report.update(drive_attempted=True, drive_started_s=float(service.clock()))
            drive = bridge.execute_j3_native_response_diagnostic(commands, current, enable=True)
            report.update(drive_result=deepcopy(drive), drive_returned_s=float(service.clock()))
            # STOP also wins in the return gap after the bridge's final read.
            # Preserve its actual receipt while rejecting outer completion.
            require_drive_admission(stop_latched=False)
            self._check_receipt(drive, commands, dispatch,
                started_s=report['drive_started_s'], returned_s=report['drive_returned_s'])
            report['initial_context'] = self._context_summary(drive)
            report['native_response'] = response_features(drive)
            report['native_response'].update(source='actual_q1_through_q5', excludes_follow_and_hold=True)
            goal_cells = [item['goal_raw'] for item in drive['policy_goal_readbacks']]
            report['raw_goal_quantization'] = dict(
                actual_goal_raw=deepcopy(goal_cells), source_present_raw=deepcopy(dispatch['present_raw']),
                non_j3_raw_delta=[{name: row[name] - dispatch['present_raw'][name]
                                  for name in MOTORS if name != 'elbow_flex'} for row in goal_cells],
                raw_pure_j3=all(row[name] == dispatch['present_raw'][name]
                               for row in goal_cells for name in MOTORS if name != 'elbow_flex'))
            follow = drive['normal_follow']
            qt = drive['readback_times_s']
            arrival = evaluate_position_arrival(commands[-1],
                [drive['measured_q_path'][-1], *follow['positions']],
                [qt[-1] - qt[0], *(stamp - qt[0] for stamp in follow['times_s'])],
                position_tolerance=.2, step_tolerance=.1, quiet_samples=3)
            arrival.update(source='actual_q5_and_follow_before_completion_hold', control_influence=False,
                           excludes_post_hold_readbacks=True)
            report['original_target_arrival'] = arrival
            report['original_target_reached'] = arrival['reached']
            if getattr(runtime, 'state', None) is not original_state:
                raise RuntimeError('native response changed the model history state')
            # The diagnostic deliberately creates no model transition. Even
            # a complete receipt must not resume policy from the old state.
            runtime.requires_home_after_incomplete_execution = True
            report.update(requires_home_after_incomplete_execution=True,
                          requires_home_before_policy_resume=True,
                          policy_resume_block_reason='completed diagnostic left model history unchanged; explicit HOME and fresh observation required')
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
                # A missing receipt after entering drive cannot prove zero
                # attempted writes. Preserve the incomplete-execution latch.
                runtime.requires_home_after_incomplete_execution = True
                report['requires_home_after_incomplete_execution'] = True
        finally:
            try:
                service.stop()
                final = reader.read_snapshot()
                report['final_source'] = deepcopy(final)
                original = report.get('source')
                if original is not None and _config(final) != _config(original):
                    raise RuntimeError('actuator configuration changed during native response')
                if final['registers']['P_Coefficient'] != 16:
                    raise RuntimeError('final native response P16 readback failed')
                report['configuration_unchanged_verified'] = original is not None
                report['stopped'] = (owner.stop_event.is_set() and runtime.stopped
                                     and runtime.pending is None and not runtime.executing)
                if not report['stopped']:
                    raise RuntimeError('native response did not finish stopped and idle')
            except BaseException as cleanup_error:
                report['cleanup_error'] = f'{type(cleanup_error).__name__}: {cleanup_error}'
                report['completed'] = False
                if report['drive_attempted']:
                    runtime.requires_home_after_incomplete_execution = True
                    report['requires_home_after_incomplete_execution'] = True
                if failure is None:
                    failure = cleanup_error
            finally:
                try:
                    owner.stop_event.set()
                    runtime.stop()
                except BaseException as latch_error:
                    report['stop_latch_error'] = f'{type(latch_error).__name__}: {latch_error}'
                    report['completed'] = False
                    if report['drive_attempted']:
                        runtime.requires_home_after_incomplete_execution = True
                        report['requires_home_after_incomplete_execution'] = True
                    if failure is None:
                        failure = latch_error
                report['finished_s'] = float(service.clock())
                self.last_result = deepcopy(report)
                try:
                    _durable_json(self.path, report)
                except BaseException as recording_error:
                    report['recording_error'] = f'{type(recording_error).__name__}: {recording_error}'
                    report['completed'] = False
                    if report['drive_attempted']:
                        runtime.requires_home_after_incomplete_execution = True
                        report['requires_home_after_incomplete_execution'] = True
                    self.last_result = deepcopy(report)
                    if failure is None:
                        failure = recording_error
        if failure is not None:
            raise failure
        return deepcopy(report)
