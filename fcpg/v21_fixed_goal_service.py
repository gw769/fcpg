"""Bind fixed-Goal compensation to the existing stopped service operation.

The ordinary JSON receipt keeps the original failed execution separate from
the compensation readbacks. It is evidence for fresh HOME admission, never a
model transition or permission to clear STOP.
"""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import threading
import time

from .v21_j3_pid_adapter import _durable_json


SCHEMA = 'v21-fixed-goal-service-result-v1'
HELPER_SCHEMA = 'v21-fixed-goal-compensation-v1'
VERIFICATION_SCHEMA = 'v21-fixed-goal-compensation-verification-v1'
FILENAME = 'fixed-goal-compensation.json'
QUIET_SAMPLES = 22
HOLD_MAX_RAW_ERROR = 6
HOLD_MAX_STEP = .1


class FixedGoalServiceError(RuntimeError):
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = deepcopy(report or {})


def _compensate(*args, **kwargs):
    from .v21_fixed_goal_compensation import compensate_failed_hold
    return compensate_failed_hold(*args, **kwargs)


def _verify(*args, **kwargs):
    from .v21_fixed_goal_compensation import verify_compensated_hold
    return verify_compensated_hold(*args, **kwargs)


def _json_object(value, label):
    if not isinstance(value, dict):
        raise FixedGoalServiceError(label+' must be a JSON object')
    # Reject nonfinite evidence instead of silently turning it into null.
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise FixedGoalServiceError(label+' is not finite plain JSON') from error


def _read_object(path, label):
    if path.is_symlink() or not path.is_file():
        raise FixedGoalServiceError(label+' must be a regular same-output file')
    try:
        return _json_object(json.loads(path.read_text()), label)
    except (OSError, ValueError) as error:
        raise FixedGoalServiceError(label+' cannot be read as ordinary JSON') from error


def _admitted_source(source):
    from .v21_position_servo_recovery import _failed_source
    admitted = _failed_source(source)
    if admitted is None or admitted[0] != ['elbow_flex']:
        raise FixedGoalServiceError('fixed-Goal compensation requires the actual failed J3-only ARM source')
    stamp = source.get('source_time_s')
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
        raise FixedGoalServiceError('failed execution lacks its finite actual source time')
    return admitted


def load_fixed_goal_compensation(output, execution):
    """Load same-source evidence for HOME or warm connection, without any IO writes.

    Missing or another execution's well-formed receipt is not a success receipt.
    A failed receipt remains ordinary failure evidence; callers must inspect
    its actual cleanup, not interpret the state string as a permanent seal.
    """
    directory = Path(output).resolve()
    path = directory/FILENAME
    if not os.path.lexists(path):
        return None
    value = _read_object(path, 'fixed-Goal receipt')
    if (value.get('schema') != SCHEMA
            or value.get('source_execution_path') != str(directory/'last-attempt.json')
            or value.get('state') not in ('started', 'completed', 'failed')):
        raise FixedGoalServiceError('fixed-Goal receipt has an invalid source binding or state')
    saved_source = _json_object(value.get('source_execution'), 'receipt source')
    if saved_source != _json_object(execution, 'current failed execution'):
        return None
    admitted = _admitted_source(saved_source)
    report = _json_object(value.get('report'), 'compensation report')
    if report.get('schema') != HELPER_SCHEMA:
        raise FixedGoalServiceError('unrecognized fixed-Goal compensation report')
    if value['state'] == 'completed':
        if (value.get('completed') is not True or value.get('configuration_restored') is not True
                or report.get('completed') is not True or report.get('configuration_restored') is not True
                or report.get('source_time_s') != saved_source['source_time_s']
                or report.get('frozen_hold_goal_raw') != admitted[2]
                or report.get('baseline_parameters_by_motor') != admitted[1]):
            raise FixedGoalServiceError('completed compensation receipt lacks same-source restoration evidence')
    elif (value.get('completed') is not False or report.get('completed') is not False
            or type(value.get('configuration_restored')) is not bool
            or type(report.get('configuration_restored')) is not bool
            or value.get('configuration_restored') != report.get('configuration_restored')):
        raise FixedGoalServiceError('unfinished compensation receipt claims inconsistent completion')
    if value['state'] == 'failed' and report.get('configuration_restored') is True:
        _validated_failed_cleanup(value, admitted)
    return value


class _Source:
    def __init__(self, service):
        self.service = service
        self.directory = Path(service.output).resolve()
        self.path = self.directory/'last-attempt.json'
        self.original = getattr(service, 'last_attempt', None)
        self.saved = os.path.lexists(self.path)
        disk = _read_object(self.path, 'failed execution') if self.saved else None
        memory = None if self.original is None else _json_object(self.original, 'service.last_attempt')
        if disk is None and memory is None:
            raise FixedGoalServiceError('no actual same-output failed execution is available')
        if disk is not None and memory is not None and disk != memory:
            raise FixedGoalServiceError('service.last_attempt differs from its same-output failed execution')
        self.execution = memory if memory is not None else disk
        self.kind = 'service.last_attempt' if memory is not None else 'same_output/last-attempt.json'
        _admitted_source(self.execution)

    def unchanged(self):
        if Path(self.service.output).resolve() != self.directory:
            raise FixedGoalServiceError('fixed-Goal output changed during the operation')
        if getattr(self.service, 'last_attempt', None) is not self.original:
            raise FixedGoalServiceError('the failed execution object changed during the operation')
        if self.original is not None and _json_object(self.original, 'service.last_attempt') != self.execution:
            raise FixedGoalServiceError('the original failed execution contents changed')
        if self.saved != os.path.lexists(self.path):
            raise FixedGoalServiceError('the same-output failed execution file changed during the operation')
        if self.saved and _read_object(self.path, 'failed execution') != self.execution:
            raise FixedGoalServiceError('the same-output failed execution contents changed')


class _Admission:
    def __init__(self, service, source, phase):
        self.service, self.source, self.phase = service, source, phase
        self.owner = service.owner
        self.runtime, self.bridge = self.owner.runtime, self.owner.bridge
        self.lease = self.owner.lease
        self.io, self.follower = self.owner.io, self.owner.io.follower
        self.bus, self.bus_lock = self.follower.bus, self.owner.bus_lock
        self.stop_event = self.owner.stop_event
        self.operation_lock, self.thread = service.operation_lock, threading.get_ident()
        self.output = service.output
        self.owner_output = getattr(self.owner, 'output', None)
        self.epoch = service.ros_control_admission_epoch()
        self.generation = service._trial_stop_generation
        if self.lease is None:
            raise FixedGoalServiceError('fixed-Goal compensation requires the existing hardware lease')
        if self.owner_output is not None and Path(self.owner_output).resolve() != source.directory:
            raise FixedGoalServiceError('the existing owner and failed source have different output directories')

    def _identity(self):
        service, owner = self.service, self.owner
        return (threading.get_ident() == self.thread and service.owner is owner
            and service.operation_lock is self.operation_lock and self.operation_lock.locked()
            and owner.runtime is self.runtime and owner.bridge is self.bridge
            and self.bridge.runtime is self.runtime and owner.lease is self.lease
            and owner.io is self.io and self.io.follower is self.follower
            and self.follower.bus is self.bus and owner.bus_lock is self.bus_lock
            and owner.stop_event is self.stop_event and service.output == self.output
            and getattr(owner, 'output', None) == self.owner_output
            and getattr(service, 'last_attempt', None) is self.source.original)

    def _state(self, recovery):
        service = self.service
        if (getattr(self.owner, 'arm_position_servo', False) is not True
                or not self.stop_event.is_set() or self.runtime.stopped is not True
                or self.runtime.pending is not None or self.runtime.executing
                or self.runtime.requires_home_after_incomplete_execution is not True
                or getattr(service, 'phase', None) != self.phase
                or getattr(getattr(service, 'trial', None), 'running', False)
                or getattr(service, '_trial_admission_in_progress', False)
                or getattr(getattr(service, 'auxiliary_budget', None), 'active', False)):
            raise FixedGoalServiceError('fixed-Goal operation requires STOP, HOME-required and an idle existing owner')
        if not recovery:
            if (service.ros_control_admission_epoch() != self.epoch
                    or service._trial_stop_generation != self.generation):
                raise FixedGoalServiceError('fixed-Goal operation cancelled by a newer STOP')
            service._require_ros_control_admission()

    def check(self, *, recovery=False):
        if not self._identity():
            raise FixedGoalServiceError('fixed-Goal owner/lease/bus/operation/thread identity changed')
        self._state(recovery)
        self.owner._lease_check()
        if not self._identity():
            raise FixedGoalServiceError('fixed-Goal ownership changed during lease verification')
        self._state(recovery)


def _publish(service, report, path):
    service.fixed_goal_compensation = deepcopy(report)
    service.fixed_goal_compensation_path = str(path)


def _clear_model_state(service, runtime):
    runtime.state = None
    for name in ('plan_record', 'preview_record', 'candidate_preview_record', 'last_execution',
                 '_assistance_proposal_id', '_carry', '_geom_place'):
        if hasattr(service, name):
            setattr(service, name, None)
    service._last_head_observations = {}
    service.memory_reset_reason = 'fixed-Goal compensation: unmodelled motion; explicit HOME remains required'


def _verify_receipt(service, source, admission, envelope):
    report = envelope['report']
    with admission.bus_lock:
        admission.check()
        source.unchanged()
        execution = deepcopy(source.execution)
        result = _json_object(_verify(admission.owner, execution, deepcopy(report),
            admission_check=admission.check), 'fresh compensation verification')
        admission.check()
        source.unchanged()
        if execution != source.execution:
            raise FixedGoalServiceError('HOME verification changed the original source copy')
    if (result.get('schema') != VERIFICATION_SCHEMA
            or result.get('configuration_restored') is not True
            or result.get('requires_home') is not True
            or result.get('goal_raw') != report['frozen_hold_goal_raw']
            or result.get('parameters_by_motor') != report['baseline_parameters_by_motor']):
        raise FixedGoalServiceError('fresh HOME verification did not prove the same restored profile and Goal')
    result.update(completed=True, model_transition=False, model_feedback_labels_added=0,
        source_time_s=source.execution['source_time_s'], compensation_receipt_path=str(source.directory/FILENAME))
    _publish(service, report, source.directory/FILENAME)
    return result


def handle_fixed_goal_compensation(service):
    """Explicit empty-payload route; its caller already holds operation_lock."""
    owner = getattr(service, 'owner', None)
    if owner is None or getattr(owner, 'arm_position_servo', False) is not True:
        raise FixedGoalServiceError('fixed-Goal compensation is disabled without ARM position servo')
    source = _Source(service)
    admission = _Admission(service, source, 'fixed-goal-compensate')
    with admission.bus_lock:
        admission.check()
        source.unchanged()
    path = source.directory/FILENAME
    previous = load_fixed_goal_compensation(source.directory, source.execution)
    if previous is not None and previous['state'] == 'completed':
        _verify_receipt(service, source, admission, previous)
        return deepcopy(previous['report'])
    report = dict(schema=HELPER_SCHEMA, completed=False, configuration_restored=False,
        requires_home=True, model_transition=False, model_feedback_labels_added=0,
        physical_task_success_verified=False, source_time_s=source.execution['source_time_s'],
        readbacks=[], parameter_writes=[], cleanup={})
    envelope = dict(schema=SCHEMA, state='started', completed=False, configuration_restored=False,
        source_execution_path=str(source.path), source_execution=deepcopy(source.execution),
        source_kind=source.kind, report=report, started_s=float(service.clock()))
    _publish(service, report, path)
    # This intent survives a process interruption. It is not a once-only token.
    _durable_json(path, envelope)
    try:
        with admission.bus_lock:
            admission.check()
            source.unchanged()
            _clear_model_state(service, admission.runtime)
            execution = deepcopy(source.execution)
            result = _compensate(owner, execution, admission_check=admission.check,
                enabled=True, clock=service.clock, wait=getattr(service, 'fixed_goal_wait', time.sleep))
            report = _json_object(result, 'fixed-Goal helper result')
            admission.check()
            source.unchanged()
            if execution != source.execution:
                raise FixedGoalServiceError('fixed-Goal helper changed the original source copy')
            admitted = _admitted_source(source.execution)
            if (report.get('schema') != HELPER_SCHEMA or report.get('completed') is not True
                    or report.get('configuration_restored') is not True
                    or report.get('source_time_s') != source.execution['source_time_s']
                    or report.get('frozen_hold_goal_raw') != admitted[2]
                    or report.get('baseline_parameters_by_motor') != admitted[1]):
                raise FixedGoalServiceError('fixed-Goal helper returned without same-source restoration evidence')
        envelope.update(state='completed', completed=True, configuration_restored=True,
            report=report, finished_s=float(service.clock()))
        _durable_json(path, envelope)
        with admission.bus_lock:
            admission.check()
            source.unchanged()
    except BaseException as error:
        partial = getattr(error, 'report', None)
        if isinstance(partial, dict) and partial.get('schema') == HELPER_SCHEMA:
            report = _json_object(partial, 'failed fixed-Goal helper report')
        else:
            report = deepcopy(report)
            report.update(completed=False, error=f'{type(error).__name__}: {error}')
        envelope.update(state='failed', completed=False,
            configuration_restored=report.get('configuration_restored') is True,
            report=report, error=f'{type(error).__name__}: {error}', finished_s=float(service.clock()))
        try:
            _durable_json(path, envelope)
        except BaseException as persistence_error:
            report['persistence_error'] = f'{type(persistence_error).__name__}: {persistence_error}'
        _publish(service, report, path)
        raise FixedGoalServiceError(str(error), report) from error
    _publish(service, report, path)
    return deepcopy(report)


def _fresh_reused_original_hold(hold, original_goal, envelope):
    """Recognize the zero-write HOLD producer's fresh, same-operation readback.

    This permits ordinary HOME recovery to perform its own live checks; it
    never grants the completed-compensation cap-restoration shortcut.
    """
    report = envelope['report']
    sample = hold.get('actual_readback')
    if (hold.get('event') != 'safe_hold_verified'
            or hold.get('reason') != 'existing_hold_reverified_no_new_goal'
            or hold.get('reused_existing_hold') is not True or not isinstance(sample, dict)):
        return False
    if any(type(value) is not int or value != 0 for value in (
            hold.get('motor_register_writes'), sample.get('motor_register_writes'), report.get('goal_writes'))):
        return False
    if (sample.get('goal_raw') != original_goal
            or sample.get('normalised_position_source') != 'same_raw_register_read'):
        return False
    for key in ('present_raw', 'moving', 'torque', 'present_position_from_raw'):
        values = sample.get(key)
        if not isinstance(values, dict) or set(values) != set(original_goal):return False
        if key == 'present_position_from_raw':
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in values.values()):
                return False
        elif any(type(value) is not int for value in values.values()):return False
    if (any(value < 0 for value in sample['present_raw'].values())
            or any(sample['moving'].values()) or any(value != 1 for value in sample['torque'].values())):
        return False
    intervals = sample.get('position_read_intervals_s')
    if not isinstance(intervals, dict):return False
    present, goal = intervals.get('Present_Position'), intervals.get('Goal_Position')
    if not isinstance(present, dict) or not isinstance(goal, dict):return False
    stamps = [envelope.get('started_s'), report.get('started_s'), sample.get('started_s'),
        present.get('started_s'), present.get('returned_s'), goal.get('started_s'), goal.get('returned_s'),
        sample.get('finished_s'), hold.get('monotonic_s'), report.get('finished_s'), envelope.get('finished_s')]
    if any(type(stamp) not in (int, float) or not math.isfinite(stamp) for stamp in stamps):return False
    return all(left <= right for left, right in zip(stamps, stamps[1:]))


def _current_home_execution(service):
    """Read the current failed execution without applying ARM admission."""
    output = getattr(service, 'output', None)
    if output is None:
        return None
    directory = Path(output).resolve()
    path = directory/'last-attempt.json'
    disk = _read_object(path, 'failed execution') if os.path.lexists(path) else None
    original = getattr(service, 'last_attempt', None)
    memory = None if original is None else _json_object(original, 'service.last_attempt')
    if disk is not None and memory is not None and disk != memory:
        raise FixedGoalServiceError('service.last_attempt differs from its same-output failed execution')
    return memory if memory is not None else disk


def _verified_final_cap_write(report, value):
    writes = report.get('parameter_writes')
    if not isinstance(writes, list):return False
    events = [row for row in writes if isinstance(row, dict)
              and row.get('motor') == 'elbow_flex' and row.get('register') == 'Torque_Limit'
              and row.get('may_have_applied') is True]
    event = events[-1] if events else {}
    return (event.get('recovery') is True and event.get('value') == value
            and event.get('verified') is True and event.get('readback') == value)


def _validated_quiet_window(report, name, goal, profile):
    stage = (report.get('stages') or {}).get(name) or {}
    rows = [row for row in report.get('readbacks', []) if isinstance(row, dict)
            and row.get('phase') == name and row.get('completed') is True
            and row.get('discarded') is False]
    if (stage.get('completed') is not True or stage.get('sample_count') != QUIET_SAMPLES
            or len(rows) != QUIET_SAMPLES or stage.get('actual_span_s', 0) < 1.):
        raise FixedGoalServiceError('failed compensation lacks its full '+name+' quiet window')
    previous = None
    for row in rows:
        present = row.get('present_raw')
        q = row.get('q_position')
        loads = row.get('load_decoded')
        if (not isinstance(present, dict) or set(present) != set(goal)
                or any(type(value) is not int for value in present.values())
                or max(abs(present[key]-goal[key]) for key in goal) > HOLD_MAX_RAW_ERROR
                or not isinstance(q, list) or len(q) != 6
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in q)
                or row.get('goal_raw') != goal
                or any(any(row.get(key, {}).values()) for key in ('moving', 'status', 'velocity'))
                or any(value != 1 for value in row.get('torque', {}).values())
                or not isinstance(loads, dict) or set(loads) != set(goal)
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or abs(value) > 32 for value in loads.values())):
            raise FixedGoalServiceError('failed compensation '+name+' is not a quiet effective HOLD')
        if previous is not None and max(abs(a-b) for a,b in zip(q,previous)) > HOLD_MAX_STEP+1e-12:
            raise FixedGoalServiceError('failed compensation '+name+' exceeded its HOLD step bound')
        previous = q
    if (rows[0].get('parameters_by_motor') != profile
            or rows[-1].get('parameters_by_motor') != profile):
        raise FixedGoalServiceError('failed compensation '+name+' lacks endpoint profile readback')


def _validated_failed_cleanup(envelope, admitted):
    """Return the effective HOLD and evidenced final profile of a failed run."""
    report = envelope['report']
    cleanup = report.get('cleanup') or {}
    if (any(cleanup.get(key) is not True for key in ('integral_zero_verified',
            'low_cap_verified', 'damping_zero_verified', 'hold_verified'))
            or cleanup.get('errors') != []):
        raise FixedGoalServiceError('failed compensation lacks verified I0/cap200/D0 and HOLD cleanup')
    hold = cleanup.get('hold') or {}
    attempts = hold.get('attempts') or []
    last = attempts[-1] if attempts else {}
    goal = cleanup.get('effective_hold_goal_raw', last.get('goal_readback_raw'))
    if (not isinstance(goal, dict) or set(goal) != set(admitted[2])
            or any(type(value) is not int for value in goal.values())
            or last.get('goal_register_verified') is not True
            or last.get('requested_goal_raw') != goal or last.get('goal_readback_raw') != goal):
        raise FixedGoalServiceError('latest compensation cleanup HOLD is not a verified raw freeze')
    reused = (hold.get('reused_existing_hold') is True
              or hold.get('reason') == 'existing_hold_reverified_no_new_goal')
    pose_verified = (_fresh_reused_original_hold(hold, goal, envelope) if reused
                     else last.get('pose_validated') is True)
    if not pose_verified:
        raise FixedGoalServiceError('latest compensation cleanup HOLD lacks fresh pose evidence')

    state = cleanup.get('profile_state')
    if state is None and report.get('configuration_restored') is False:
        state = 'cleanup_cap200'  # Legacy failed receipts predate explicit profile_state.
    if report.get('configuration_restored') is True:
        baseline = admitted[1]
        cap = baseline['elbow_flex']['Torque_Limit']
        if (state != 'baseline1000' or cleanup.get('restored') is not True
                or cleanup.get('baseline_restored') is not True
                or cleanup.get('torque_restored') is not True
                or cleanup.get('pre_restore_quiet_verified') is not True
                or cleanup.get('baseline_cap_verified') is not True
                or cleanup.get('post_restore_quiet_verified') is not True
                or cleanup.get('baseline_restore_errors') != []
                or cleanup.get('final_parameters_by_motor') != baseline
                or report.get('final_parameters_by_motor') != baseline
                or not _verified_final_cap_write(report, cap)):
            raise FixedGoalServiceError('failed compensation lacks verified saved-baseline restoration')
        _validated_quiet_window(report, 'failed_hold_retention', goal, {
            name:dict(values, **({'Torque_Limit':200} if name == 'elbow_flex' else {}))
            for name,values in baseline.items()})
        _validated_quiet_window(report, 'failed_baseline_restored', goal, baseline)
        return goal, state
    if (state != 'cleanup_cap200' or cleanup.get('baseline_restored') is True
            or cleanup.get('restored') is True):
        raise FixedGoalServiceError('failed compensation has no trustworthy final profile')
    if cleanup.get('profile_state') is not None and not _verified_final_cap_write(report, 200):
        raise FixedGoalServiceError('failed compensation does not prove its final cap200 fallback')
    return goal, state


def verify_fixed_goal_compensation_before_home(service):
    """Return a fresh success receipt, or None when ordinary HOME recovery applies."""
    owner = getattr(service, 'owner', None)
    runtime = getattr(owner, 'runtime', None)
    if getattr(runtime, 'requires_home_after_incomplete_execution', False) is not True:
        return None
    output = getattr(service, 'output', None)
    if output is None or not os.path.lexists(Path(output)/FILENAME):
        return None
    current = _current_home_execution(service)
    # Fixed-Goal compensation is the narrow single-axis ARM recovery path. A
    # generic failed transaction, a multi-axis ARM finishing record, or an
    # unrelated/stale position-servo record must continue through the ordinary
    # HOME/baseline-recovery path. In particular, merely having a
    # ``position_servo`` key is insufficient evidence: incomplete CEM dispatch
    # can retain a partial record while no servo parameters were touched.
    servo = None if current is None else current.get('position_servo')
    if not isinstance(servo, dict):
        return None
    from .v21_arm_position_servo import SCHEMA as ARM_POSITION_SERVO_SCHEMA
    if (servo.get('schema') != ARM_POSITION_SERVO_SCHEMA
            or servo.get('touched_motors') != ['elbow_flex']):
        return None
    source = _Source(service)
    envelope = load_fixed_goal_compensation(source.directory, source.execution)
    if envelope is None:
        return None
    if getattr(owner, 'arm_position_servo', False) is not True:
        raise FixedGoalServiceError('fixed-Goal HOME verification requires the enabled ARM owner')
    admission = _Admission(service, source, 'home')
    if envelope['state'] == 'started':
        raise FixedGoalServiceError('interrupted compensation has no verified cleanup for HOME')
    if envelope['state'] != 'completed':
        _validated_failed_cleanup(envelope, _admitted_source(source.execution))
        with admission.bus_lock:
            admission.check()
            source.unchanged()
        _publish(service, envelope['report'], source.directory/FILENAME)
        return None
    return _verify_receipt(service, source, admission, envelope)
