"""Default-disabled service admission for one bounded J3 gain experiment.

The service owns operation_lock before calling handle(). This coordinator never
creates a hardware owner or acquires a bus. Explicit continuation imports only
completed comparison metadata, never model history or an unfinished command.
Verified terminal restoration ends the diagnostic block while retaining STOP.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import threading

from .v21_j3_pid_adapter import create_existing_owner_j3_pid_callbacks, _durable_directory, _sync_directory
from scripts.prepare_v21_j3_pid_protocol import ACTUAL_IDENTITY, J3PIDExperiment, MANUAL_SHA, j3_targets


DEFAULT_SEAL_PATH = Path('/home/james/.local/state/so101/v21-j3-pid-motion-seal.json')
READ_ONLY_ACTIONS = frozenset(('observe', 'actuator-observe', 'status', 'stop'))
CALLBACK_OPERATIONS = frozenset(('read', 'restore_read', 'gain_candidate', 'gain_restore', 'execute', 'persist'))
FIRMWARE_APPLICABILITY = dict(
    reviewed=True,
    manual_sha256=MANUAL_SHA,
    basis=('Reviewed Feetech manufacturer family manual pages 12-13: P controls static '
           'holding and Lock1 allows temporary parameter edits. Fresh ID3/model777/'
           'firmware3.10 and P16/I0/D0 must match every protocol admission. '
           'Energized P32 behavior on this firmware is not verified.'),
    energized_P32_verified=False,
)


class J3PIDServiceError(RuntimeError):
    pass


def _final_comparison_reports(baseline_path, variant_path, reports_directory):
    """Read the completed baseline and a conclusively unexecuted variant."""
    paths = [Path(value).resolve() for value in (baseline_path, variant_path)]
    try:
        reports = [json.loads(path.read_text()) for path in paths]
        baseline, variant = reports
        session = baseline['session_id']
        if not isinstance(session, str) or len(session) != 32 or variant['session_id'] != session:
            raise ValueError('comparison reports belong to different sessions')
        if any(path.parent.name != session or path.parent.parent != reports_directory.resolve()
               for path in paths):
            raise ValueError('continuation must retain the canonical diagnostic report directory')
        for phase, path, report, state in zip(('baseline', 'variant'), paths, reports,
                                            ('baseline_recorded', 'failed_sealed')):
            if (report.get('schema') != 'v21-j3-pid-phase-result-v2'
                    or report.get('dry_run') is not False or report.get('phase') != phase
                    or report.get('session_state') != state
                    or report.get('expected_identity') != ACTUAL_IDENTITY
                    or report.get('p_restored_verified') is not True
                    or report.get('configuration_restored_verified') is not True
                    or report.get('stop_error') is not None
                    or report.get('persistence_error') is not None
                    or report['events'][-1]['event'] != 'phase_finished_still_sealed'
                    or path.name != f'{phase}-{len(report["events"]):04d}.json'):
                raise ValueError('continuation requires canonical final restored phase reports')
            start, stop, final = (report['stop_started_s'], report['stop_returned_s'],
                                  report['events'][-1]['time_s'])
            if not all(type(value) in (int, float) and math.isfinite(value) for value in (start, stop, final)) or not start <= stop <= final:
                raise ValueError('phase STOP completion is not established')
            checks = {check['label']: check.get('snapshot') for check in report['read_checks']}
            final_read = checks['baseline_final' if phase == 'baseline' else 'after_restore']
            J3PIDExperiment._same_config(final_read, baseline['before_configuration'], 16)
            if any(final_read['registers'].get(key) != value for key, value in ACTUAL_IDENTITY.items()):
                raise ValueError('restored hardware identity differs')
        if (baseline.get('drive_attempted') is not True or baseline.get('drive_returned') is not True
                or baseline.get('five_command_receipt_validated') is not True
                or baseline.get('primary_error') is not None):
            raise ValueError('baseline is not a completed five-command transaction')
        source = baseline.get('dispatch_source', baseline['source'])
        J3PIDExperiment._same_config(source, baseline['before_configuration'], 16)
        J3PIDExperiment._validate_drive(baseline['drive_result'], source,
            j3_targets(source['q_position']), callback_started_s=baseline['drive_started_s'],
            callback_returned_s=baseline['drive_returned_s'])
        if (variant.get('drive_attempted') is not False or variant.get('drive_returned') is not False
                or 'drive_result' in variant or 'failed_drive_attempt' in variant
                or variant.get('five_command_receipt_validated') is not False
                or variant.get('primary_error') is None
                or variant['before_configuration'] != baseline['before_configuration']):
            raise ValueError('only a final variant that never entered its driver may continue')
        return dict(prior_session_id=session, baseline_report=str(paths[0]),
            unexecuted_variant_report=str(paths[1]), baseline_source=deepcopy(source['q_position']),
            baseline_configuration=deepcopy(baseline['before_configuration']),
            prior_actual_commands=5, remaining_actual_commands=5,
            model_history_imported=False, automatic_retry=False)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        raise J3PIDServiceError('invalid explicit J3 comparison continuation: ' + str(error)) from error


class J3PIDServiceCoordinator:
    def __init__(self, service, *, enabled=False, seal_path=DEFAULT_SEAL_PATH,
                 reports_directory, continuation_baseline_report=None,
                 continuation_variant_report=None):
        if type(enabled) is not bool:
            raise TypeError('enabled must be an explicit bool')
        self.service = service
        self.enabled = enabled
        self.seal_path = Path(seal_path)
        self.reports_directory = Path(reports_directory)
        if not self.seal_path.is_absolute() or not self.reports_directory.is_absolute():
            raise ValueError('diagnostic seal and report paths must be absolute')
        if self.seal_path == self.reports_directory:
            raise ValueError('diagnostic seal must be separate from report directory')
        self.experiment = None
        self.last_result = None
        self._sticky_sealed = False
        self._failed = False
        self._handling = threading.Lock()
        self._active_thread = self._active_owner = self._active_lease = self._active_lock = None
        self._active_phase = None
        self._session_owner = self._session_lease = None
        self._stop_attempted = False
        self._stop_error = None
        continuation = (continuation_baseline_report, continuation_variant_report)
        if any(value is not None for value in continuation) and not all(value is not None for value in continuation):
            raise ValueError('explicit continuation requires both final comparison reports')
        self._continuation = (None if continuation[0] is None else
            _final_comparison_reports(*continuation, self.reports_directory))
        if self._continuation is not None:
            self._require_unused_continuation()

    def _seal_exists(self):
        # lstat treats dangling links and directories as seals, not as absence.
        try:
            self.seal_path.lstat()
            return True
        except FileNotFoundError:
            return False
        except OSError as error:
            raise J3PIDServiceError('cannot establish that persistent J3 motion seal is absent') from error

    def _read_seal(self):
        if self.seal_path.is_symlink() or not self.seal_path.is_file():
            raise J3PIDServiceError('persistent J3 motion seal is not a regular file')
        try:
            marker = json.loads(self.seal_path.read_text())
        except (OSError, ValueError) as error:
            raise J3PIDServiceError('persistent J3 motion seal is unreadable') from error
        if (not isinstance(marker, dict) or marker.get('motion_sealed') is not True
                or not isinstance(marker.get('session_id'), str)):
            raise J3PIDServiceError('persistent J3 motion seal is invalid')
        return marker

    def _require_same_session_variant(self, *, require_marker=True):
        if (not self.enabled or self._failed or self.experiment is None
                or self.experiment.phase != 'baseline_recorded'):
            raise J3PIDServiceError('variant requires this uninterrupted in-memory baseline session; no disk resume')
        if not require_marker:
            return
        if not self._seal_exists():
            raise J3PIDServiceError('same-session variant requires its existing persistent motion seal')
        if self._read_seal()['session_id'] != self.experiment.session_id:
            raise J3PIDServiceError('persistent J3 seal belongs to another session')

    def require_motion_unsealed(self, action):
        """Read-only guard used before startup and every ordinary motion route."""
        if action in READ_ONLY_ACTIONS:
            return
        sealed = self._sticky_sealed or self._seal_exists()
        if action == 'j3-pid-variant' and sealed:
            self._require_same_session_variant()
            return
        if sealed:
            raise J3PIDServiceError('persistent J3 diagnostic seal blocks ' + str(action) + '; independent recovery required')

    @staticmethod
    def _lock_held(lock):
        owned = getattr(lock, '_is_owned', None)
        if callable(owned):
            return owned() is True
        locked = getattr(lock, 'locked', None)
        return callable(locked) and locked() is True

    def _admission_check(self, operation):
        # Recovery deliberately ignores STOP, ROS epochs and current disk IO.
        # The adapter checks the actual hardware lease and same-lock J3 guard.
        return (operation in CALLBACK_OPERATIONS and self._active_phase in ('baseline', 'variant')
                and self._active_thread == threading.get_ident()
                and self.service.owner is self._active_owner
                and getattr(self._active_owner, 'lease', None) is self._active_lease
                and self._active_lease is not None
                and self.service.operation_lock is self._active_lock
                and self._lock_held(self._active_lock))

    def _stop_for_phase(self):
        # One STOP attempt per phase; failures are retained, never retried here.
        if self._stop_attempted:
            raise J3PIDServiceError('diagnostic phase already attempted STOP')
        self._stop_attempted = True
        try:
            return self.service.stop()
        except BaseException as error:
            self._stop_error = f'{type(error).__name__}: {error}'
            raise

    def _make_experiment(self, owner):
        callbacks = create_existing_owner_j3_pid_callbacks(owner, enabled=True,
            admission_check=self._admission_check, stop_hold=self._stop_for_phase,
            seal_path=self.seal_path, reports_directory=self.reports_directory,
            clock_fn=self.service.clock)
        if self._continuation is not None:
            persist = callbacks.persist
            def with_comparison_provenance(report):
                report['comparison_continuation'] = deepcopy(self._continuation)
                return persist(report)
            callbacks = replace(callbacks, persist=with_comparison_provenance)
        return J3PIDExperiment(callbacks, expected_identity=ACTUAL_IDENTITY,
            firmware_applicability=FIRMWARE_APPLICABILITY, clock_fn=self.service.clock)

    def _require_unused_continuation(self):
        # Existing receipts also prevent CLI options from replaying a completed
        # continuation after restart. No extra marker or cryptographic token.
        for path in self.reports_directory.glob('*/variant-*.json'):
            try:
                previous = json.loads(path.read_text()).get('comparison_continuation')
            except (OSError, ValueError, TypeError) as error:
                raise J3PIDServiceError('cannot establish prior continuation outcome: ' + str(path)) from error
            if previous and previous.get('prior_session_id') == self._continuation['prior_session_id']:
                raise J3PIDServiceError('this baseline already has a continuation attempt; no replay')

    def _initialize_continuation(self, owner):
        """Reserve only the remaining phase on a fresh current-owner adapter."""
        if self._seal_exists() or self._sticky_sealed or self._failed:
            raise J3PIDServiceError('explicit continuation requires no unresolved diagnostic block')
        self._require_unused_continuation()
        self.experiment = self._make_experiment(owner)
        self.experiment.baseline_source = deepcopy(self._continuation['baseline_source'])
        self.experiment.baseline_configuration = deepcopy(self._continuation['baseline_configuration'])
        self.experiment.reserved_commands = 5
        self.experiment.phase = 'baseline_recorded'
        self._session_owner, self._session_lease = owner, owner.lease

    def _release_terminal_block(self, phase, report, owner, stopped):
        if phase != 'variant' or not isinstance(report, dict):
            return False
        no_drive = (report.get('drive_attempted') is False and report.get('drive_returned') is False
                    and 'drive_result' not in report and 'failed_drive_attempt' not in report)
        complete = (report.get('drive_returned') is True
                    and report.get('five_command_receipt_validated') is True
                    and report.get('drive_result', {}).get('completed') is True)
        if (not (no_drive or complete) or report.get('p_restored_verified') is not True
                or report.get('configuration_restored_verified') is not True
                or self._stop_error is not None or report.get('stop_error') is not None
                or not self._stop_attempted or not stopped or not owner.stop_event.is_set()
                or getattr(owner.runtime, 'requires_home_after_incomplete_execution', False)):
            return False
        if self._seal_exists():
            if self._read_seal()['session_id'] != self.experiment.session_id:
                raise J3PIDServiceError('terminal cleanup cannot remove another session marker')
            self.seal_path.unlink()
            _sync_directory(self.seal_path.parent)
        self._sticky_sealed = False
        return True

    def _require_entry(self, phase):
        lock = self.service.operation_lock
        if not self._lock_held(lock):
            raise J3PIDServiceError('service operation_lock must already be held by the caller')
        owner = self.service.owner
        if owner is None or getattr(owner, 'lease', None) is None:
            raise J3PIDServiceError('diagnostic requires the existing leased owner')
        runtime = owner.runtime
        if runtime.stopped is not True or runtime.pending is not None or runtime.executing:
            raise J3PIDServiceError('diagnostic requires stopped runtime with no pending/executing command')
        if not owner.stop_event.is_set():
            raise J3PIDServiceError('diagnostic phase admission requires latched STOP')
        continued = False
        if self._continuation is not None and self.experiment is None:
            if phase != 'variant':
                raise J3PIDServiceError('explicit continuation permits only the remaining variant')
            self._initialize_continuation(owner)
            continued = True
        if phase == 'variant':
            self._require_same_session_variant(require_marker=not continued)
            if owner is not self._session_owner or owner.lease is not self._session_lease:
                raise J3PIDServiceError('variant requires the original baseline owner and hardware lease')
        elif self.experiment is not None or self._failed or self._sticky_sealed or self._seal_exists():
            raise J3PIDServiceError('baseline requires a new unsealed session; no retries or disk resume')
        return owner, lock

    def handle(self, phase):
        if phase not in ('baseline', 'variant'):
            raise ValueError('J3 diagnostic phase must be baseline or variant')
        if not self.enabled:
            return dict(schema='v21-j3-pid-service-dry-run-v1', enabled=False, phase=phase,
                        dry_run=True, callbacks_invoked=0, hardware_accessed=False)
        if not self._handling.acquire(blocking=False):
            raise J3PIDServiceError('another J3 diagnostic phase is active')
        try:
            owner, lock = self._require_entry(phase)
            _durable_directory(self.seal_path.parent)
            _durable_directory(self.reports_directory)
            if not self.seal_path.parent.is_dir() or not self.reports_directory.is_dir():
                raise J3PIDServiceError('diagnostic seal and report parents must be directories')
            self._active_thread = threading.get_ident()
            self._active_owner, self._active_lease, self._active_lock = owner, owner.lease, lock
            self._active_phase = phase
            self._stop_attempted = False
            self._stop_error = None
            self._sticky_sealed = True
            result = None
            failure = None
            try:
                if self.experiment is None:
                    self.experiment = self._make_experiment(owner)
                    self._session_owner, self._session_lease = owner, owner.lease
                # This is the only explicit re-arm, after phase admission.
                # A stale ROS epoch is rejected by this existing service method.
                self.service._clear_stop_for_current_operation()
                result = getattr(self.experiment, phase)(enable=True)
            except BaseException as error:
                failure = error
                self._failed = True
            finally:
                if not self._stop_attempted:
                    try:
                        self._stop_for_phase()
                    except BaseException as error:
                        if failure is None:
                            failure = error
                        self._failed = True
                # Preserve the local STOP latch even if the HOLD transport fails.
                owner.stop_event.set()
                stopped = (owner.runtime.stopped is True and owner.runtime.pending is None
                           and not owner.runtime.executing)
                if not stopped:
                    self._failed = True
                    if failure is None:
                        failure = J3PIDServiceError('phase cleanup did not establish a stopped idle runtime')
                report = result if result is not None else getattr(failure, 'j3_pid_report', None)
                if report is not None and self._continuation is not None:
                    report['comparison_continuation'] = deepcopy(self._continuation)
                released = False
                cleanup_error = None
                try:
                    released = self._release_terminal_block(phase, report, owner, stopped)
                except (OSError, J3PIDServiceError) as error:
                    cleanup_error = f'{type(error).__name__}: {error}'
                    if failure is None:
                        failure = error
                    self._failed = True
                self.last_result = dict(schema='v21-j3-pid-service-result-v1', phase=phase,
                    session_id=None if self.experiment is None else self.experiment.session_id,
                    completed=failure is None, motion_sealed=not released, policy_resume=False,
                    diagnostic_block_released=released, cleanup_error=cleanup_error,
                    stop_attempted=self._stop_attempted, stop_error=self._stop_error,
                    stopped=owner.stop_event.is_set() and stopped,
                    p_restored_verified=bool(report and report.get('p_restored_verified')),
                    configuration_restored_verified=bool(report and report.get('configuration_restored_verified')),
                    actual_response_verified=bool(report and report.get('actual_response_verified')),
                    error=None if failure is None else f'{type(failure).__name__}: {failure}',
                    protocol_report=deepcopy(report))
                self._active_thread = self._active_owner = self._active_lease = self._active_lock = None
                self._active_phase = None
            if failure is not None:
                failure.j3_pid_service_result = deepcopy(self.last_result)
                raise failure
            return deepcopy(self.last_result)
        finally:
            self._handling.release()

    def status(self):
        """No owner, clock, register or service callback is invoked."""
        error = None
        try:
            disk_sealed = self._seal_exists()
        except J3PIDServiceError as caught:
            disk_sealed = True
            error = str(caught)
        return dict(enabled=self.enabled, motion_sealed=self._sticky_sealed or disk_sealed,
            persistent_seal_present=disk_sealed, seal_path=str(self.seal_path),
            session_id=None if self.experiment is None else self.experiment.session_id,
            session_phase=None if self.experiment is None else self.experiment.phase,
            active_phase=self._active_phase, failed=self._failed, seal_error=error,
            comparison_continuation=deepcopy(self._continuation),
            last_result=deepcopy(self.last_result), policy_resume=False)
