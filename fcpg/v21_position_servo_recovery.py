"""Restore failed ARM finishing caps only inside the existing explicit HOME.

STOP and requires_home stay latched. The failed execution and model history are
never edited; success restores configuration only and issues no Goal write.
"""
from copy import deepcopy
import json
from pathlib import Path
import threading
import time

import numpy as np

from .servo_register_decode import decode_present_load
from .v21_arm_position_servo import AXES, BASELINE, PARAMETERS, SCHEMA as ARM_SCHEMA
from .v21_j3_pid_adapter import _durable_json
from .v21_same_goal_integral import (
    IDENTITY, LATENESS_S, ORDER, PERIOD_S, _Run, _array, _integer, _map, _plain,
)


SCHEMA = 'v21-position-servo-baseline-recovery-v1'
SAMPLES = 22
SAMPLE_MAX_ATTEMPTS = 4
SYNC_READ_NUM_RETRY = 0
HOLD_MAX_RESIDUAL = .6
HOLD_MAX_RAW_ERROR = 6
HOLD_MAX_STEP = .1


class PositionServoRecoveryError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = _plain(report)


class _RegisterReadConnectionError(ConnectionError):
    """A transport failure raised only by this helper's sync_read call."""
    def __init__(self, register, error):
        super().__init__(str(error))
        self.register = register


def _failed_source(source):
    servo = source.get('position_servo')
    if servo is None:
        return None
    if not isinstance(servo, dict) or servo.get('schema') != ARM_SCHEMA:
        raise ValueError('unrecognized failed position-servo evidence')
    cleanup = servo.get('cleanup') or {}
    if servo.get('completed') is True:
        return None
    if (source.get('completed') is not False
            or source.get('requires_home_after_incomplete_execution') is not True
            or source.get('action_source') not in ('v21_actor', 'v21_lewm_cem')
            or servo.get('completed') is not False or servo.get('enabled') is not True
            or servo.get('mode') != 'i1'):
        raise ValueError('recovery requires the same failed live ARM execution')
    writes = servo.get('parameter_writes')
    if not isinstance(writes, list):
        raise ValueError('failed ARM source lacks actual parameter write intents')
    touched = list(dict.fromkeys(row['motor'] for row in writes if row.get('may_have_applied') is True))
    if touched != servo.get('touched_motors') or not set(touched).issubset({ORDER[i] for i in AXES}):
        raise ValueError('source touched motors differ from actual J3/J4/J2 writes')
    if not touched:
        return None
    baseline = servo.get('baseline_parameters_by_motor')
    expected = {name:dict(BASELINE) for name in ORDER}
    expected[ORDER[-1]].update(Torque_Limit=500, Max_Torque_Limit=500)
    if baseline != expected:
        raise ValueError('original six-motor baseline including gripper500 is missing')
    if cleanup.get('hold_performed') is not True or cleanup.get('hold_verified') is not True:
        raise ValueError('failed ARM cleanup lacks verified HOLD')
    hold = cleanup.get('hold') or {}
    attempts = hold.get('attempts') or []
    last = attempts[-1] if attempts else {}
    goal = _map(last.get('goal_readback_raw'), 'recorded cleanup HOLD Goal')
    if (last.get('goal_register_verified') is not True or last.get('pose_validated') is not True
            or goal != _map(last.get('requested_goal_raw'), 'recorded frozen Present')):
        raise ValueError('recorded HOLD does not prove raw Present-to-Goal freeze')
    for name in touched:
        identity = dict(IDENTITY, ID=ORDER.index(name)+1)
        if (servo.get('axes', {}).get(name, {}).get('identity') != identity
                or servo['axes'][name].get('baseline_parameters') != baseline[name]):
            raise ValueError(name+' lacks original identity and baseline evidence')
        flags = cleanup.get('recovery_by_motor', {}).get(name, {})
        if any(flags.get(key) is not True for key in
               ('integral_zero_verified', 'low_cap_verified', 'damping_zero_verified')):
            raise ValueError(name+' lacks verified I0/cap200/D0 cleanup')
        for register, value in (('I_Coefficient', 0), ('Torque_Limit', 200), ('D_Coefficient', 0)):
            applied = [row for row in writes if row.get('motor') == name
                       and row.get('register') == register and row.get('value') == value
                       and row.get('may_have_applied') is True]
            event = applied[-1] if applied else {}
            if (event.get('recovery') is not True or event.get('verified') is not True
                    or event.get('value') != value or event.get('readback') != value):
                raise ValueError(name+' cleanup flags lack actual verified '+register+' receipt')
    _failed_cleanup_profile(servo, touched, baseline, goal)
    return touched, deepcopy(baseline), goal


def _failed_cleanup_profile(servo, touched, baseline, goal):
    """Validate which parameter profile a failed ARM cleanup actually left."""
    cleanup = servo.get('cleanup') or {}
    writes = servo.get('parameter_writes') or []
    state = cleanup.get('profile_state')
    if state is None:
        if cleanup.get('errors') != [] or cleanup.get('restored') is True:
            raise ValueError('legacy failed ARM cleanup has no trustworthy final profile')
        state = 'cleanup_cap200'
    if state not in ('cleanup_cap200', 'baseline1000'):
        raise ValueError('failed ARM cleanup has an unknown final profile')

    if state == 'cleanup_cap200':
        if cleanup.get('baseline_restored') is True or cleanup.get('restored') is True:
            raise ValueError('failed ARM cleanup contradicts its cap200 profile')
        for name in touched:
            events = [row for row in writes if isinstance(row, dict)
                      and row.get('motor') == name and row.get('register') == 'Torque_Limit'
                      and row.get('may_have_applied') is True]
            event = events[-1] if events else {}
            if (event.get('value') != 200 or event.get('verified') is not True
                    or event.get('readback') != 200):
                raise ValueError(name+' failed cleanup does not prove the final cap200 profile')
        return state

    if (cleanup.get('errors') != [] or cleanup.get('restored') is not True
            or cleanup.get('baseline_restored') is not True
            or cleanup.get('torque_restored') is not True
            or cleanup.get('pre_restore_quiet_verified') is not True
            or cleanup.get('post_restore_quiet_verified') is not True
            or cleanup.get('parameters_restored_by_motor') != {name:True for name in touched}
            or servo.get('final_parameters_by_motor') != baseline):
        raise ValueError('failed ARM cleanup lacks verified baseline1000 restoration')
    for name in touched:
        flags = cleanup.get('recovery_by_motor', {}).get(name, {})
        events = [row for row in writes if isinstance(row, dict)
                  and row.get('motor') == name and row.get('register') == 'Torque_Limit'
                  and row.get('may_have_applied') is True]
        event = events[-1] if events else {}
        expected = baseline[name]['Torque_Limit']
        if (flags.get('baseline_cap_verified') is not True or event.get('value') != expected
                or event.get('verified') is not True or event.get('readback') != expected):
            raise ValueError(name+' failed cleanup baseline cap receipt is incomplete')
    stage = cleanup.get('post_baseline_restore') or {}
    rows = stage.get('readbacks')
    if (stage.get('completed') is not True or not isinstance(rows, list) or len(rows) != SAMPLES
            or stage.get('sample_count') != len(rows) or stage.get('actual_span_s', 0) < 1.
            or stage.get('final_parameters_by_motor') != baseline):
        raise ValueError('failed ARM cleanup lacks its full baseline1000 quiet window')
    for row in rows:
        if (not isinstance(row, dict) or row.get('goal_raw') != goal
                or row.get('max_abs_raw_goal_error', HOLD_MAX_RAW_ERROR+1) > HOLD_MAX_RAW_ERROR
                or row.get('max_abs_goal_error', HOLD_MAX_RESIDUAL+1) > HOLD_MAX_RESIDUAL+1e-12
                or row.get('max_abs_step') is not None
                and row.get('max_abs_step') > HOLD_MAX_STEP+1e-12
                or any(row.get('moving', {}).values())
                or any(value != 1 for value in row.get('torque', {}).values())
                or any(abs(value) > 32 for value in row.get('all_present_load_decoded', {}).values())):
            raise ValueError('failed ARM cleanup baseline1000 window is not a quiet effective HOLD')
    return state


def failed_position_servo_profile(source, admitted=None):
    """Return the strictly evidenced final profile of a failed ARM execution."""
    admitted = _failed_source(source) if admitted is None else admitted
    if admitted is None:
        return None
    servo = source['position_servo']
    return _failed_cleanup_profile(servo, admitted[0], admitted[1], admitted[2])


def _read_recovery_report(path):
    """Read one ordinary same-output recovery receipt without following links."""
    if path.is_symlink() or not path.is_file():
        raise ValueError('position-servo recovery evidence must be a regular file')
    try:
        value = json.loads(path.read_text())
    except Exception as error:
        raise ValueError('position-servo recovery evidence is unreadable') from error
    if not isinstance(value, dict) or value.get('schema') != SCHEMA:
        raise ValueError('unrecognized position-servo recovery evidence')
    return value


def _recovery_hold_from_report(report, touched, baseline, *, failed, admission_goal, admission_held):
    """Validate a report's actual HOLD and return its effective pose/profile."""
    cleanup = report.get('cleanup') or {}
    if failed:
        if (report.get('completed') is not False or report.get('configuration_restored') is not False
                or cleanup.get('errors') != [] or cleanup.get('hold_verified') is not True
                or cleanup.get('cap200_verified') != {name:True for name in touched}):
            raise ValueError('latest failed position-servo recovery lacks verified cap200 and HOLD cleanup')
        writes = report.get('parameter_writes')
        if not isinstance(writes, list):
            raise ValueError('latest failed position-servo recovery lacks parameter receipts')
        for name in touched:
            events = [row for row in writes if isinstance(row, dict)
                      and row.get('motor') == name and row.get('register') == 'Torque_Limit']
            event = events[-1] if events else {}
            if (event.get('fallback') is not True or event.get('may_have_applied') is not True
                    or event.get('verified') is not True or event.get('value') != 200
                    or event.get('readback') != 200):
                raise ValueError(name+' latest recovery cap200 receipt is incomplete')
        hold = cleanup.get('hold') or {}
        attempts = hold.get('attempts') or []
        last = attempts[-1] if attempts else {}
        goal = _map(last.get('goal_readback_raw'), 'latest recovery cleanup HOLD Goal')
        requested = _map(last.get('requested_goal_raw'), 'latest recovery frozen Present')
        held = _array(last.get('held_state'), (6,), 'latest recovery held position')
        if (hold.get('event') != 'safe_hold_verified'
                or hold.get('reason') != 'v21_emergency_stop'
                or hold.get('reused_existing_hold') is not False
                or hold.get('motor_register_writes') != 1
                or last.get('goal_register_verified') is not True
                or last.get('pose_validated') is not True
                or requested != goal):
            raise ValueError('latest position-servo recovery cleanup HOLD is incomplete')
        if _map(report.get('frozen_hold_goal_raw'), 'failed recovery admission Goal') != admission_goal:
            raise ValueError('failed position-servo recovery changed its admission Goal')
        return dict(goal_raw=goal, held_position=held.tolist(), profile_state='cleanup_cap200',
                    baseline_restored=False, hold_attempt=deepcopy(last))

    # A completed report contains two actual windows. The final post-restore
    # row is the only acceptable source for a warm reconnect cache.
    if (report.get('completed') is not True or report.get('configuration_restored') is not True
            or cleanup.get('errors') != []):
        raise ValueError('completed position-servo recovery lacks baseline restoration evidence')
    final = report.get('final_parameters_by_motor')
    expected = deepcopy(baseline)
    if final != expected:
        raise ValueError('completed position-servo recovery does not prove baseline1000')
    rows = report.get('readbacks')
    if not isinstance(rows, list):
        raise ValueError('completed position-servo recovery lacks actual readbacks')
    post = [row for row in rows if isinstance(row, dict) and row.get('phase') == 'post_restore'
            and row.get('discarded') is not True and row.get('failed') is not True]
    if not post:
        raise ValueError('completed position-servo recovery lacks post-restore readbacks')
    window = report.get('post_restore') or {}
    if (window.get('completed') is not True or window.get('sample_count') != SAMPLES
            or window.get('actual_span_s', 0) < 1. or len(post) != SAMPLES):
        raise ValueError('completed recovery lacks a full post-restore quiet window')
    if _map(report.get('frozen_hold_goal_raw'), 'completed recovery admission Goal') != admission_goal:
        raise ValueError('completed position-servo recovery changed its admission Goal')
    for row in post:
        goal = _map(row.get('goal_raw'), 'completed recovery Goal')
        present = _map(row.get('present_raw'), 'completed recovery Present')
        position = _array(row.get('q_position'), (6,), 'completed recovery actual position')
        if (goal != admission_goal or row.get('parameters_by_motor') != expected
                or max(abs(present[name]-goal[name]) for name in ORDER) > HOLD_MAX_RAW_ERROR
                or row.get('max_abs_goal_error', HOLD_MAX_RESIDUAL+1) > HOLD_MAX_RESIDUAL+1e-12
                or row.get('max_abs_step') is not None
                and row.get('max_abs_step') > HOLD_MAX_STEP+1e-12
                or any(row.get(key, {}).get(name) for key in ('moving', 'status', 'velocity')
                       for name in ORDER)
                or not np.isfinite(position).all()):
            raise ValueError('completed recovery post-restore HOLD is not a quiet baseline readback')
    last = post[-1]
    goal = _map(last.get('goal_raw'), 'completed recovery Goal')
    present = _map(last.get('present_raw'), 'completed recovery Present')
    if (goal != admission_goal or last.get('parameters_by_motor') != expected):
        raise ValueError('completed recovery post-restore HOLD is not a quiet baseline readback')
    held = _array(admission_held, (6,), 'completed recovery effective held position')
    return dict(goal_raw=goal, held_position=held.tolist(), profile_state='baseline1000',
                baseline_restored=True,
                hold_attempt=dict(goal_readback_raw=goal, requested_goal_raw=present,
                                  pose_validated=True, held_state=held.tolist()))


def latest_failed_position_servo_hold(directory, source, admitted):
    """Return the newest source-bound effective HOLD, including recovery lineage.

    ``admitted[2]`` is the immutable HOLD from ``last-attempt.json``. Later
    recovery files may admit the previous effective Goal and produce a new one;
    the predecessor link prevents a newer report from silently replacing that
    chain. The returned report snapshot is used to detect post-parse changes.
    """
    directory = Path(directory).resolve()
    touched, baseline, original_goal = admitted
    original_attempt = source['position_servo']['cleanup']['hold']['attempts'][-1]
    original_held = _array(original_attempt.get('held_state'), (6,),
                           'original failed cleanup held position').tolist()
    paths = []
    for path in directory.glob('position-servo-recovery-*.json'):
        suffix = path.stem.rsplit('-', 1)[-1]
        if suffix.isdigit() and path.parent.resolve() == directory:
            paths.append((int(suffix), path))
    reports = {}
    for _, path in sorted(paths):
        report = _read_recovery_report(path)
        if report.get('source_time_s') != source.get('source_time_s'):
            continue
        if (report.get('touched_motors') != touched
                or report.get('original_baseline_by_motor') != baseline):
            raise ValueError('position-servo recovery is not bound to the failed ARM source')
        immutable = report.get('original_failed_cleanup_hold_goal_raw')
        frozen = _map(report.get('frozen_hold_goal_raw'), 'recovery admission Goal')
        if immutable is None:
            if frozen != original_goal:
                raise ValueError('position-servo recovery admission Goal has no valid source lineage')
        elif _map(immutable, 'original failed cleanup HOLD Goal') != original_goal:
            raise ValueError('position-servo recovery immutable source Goal changed')
        reports[path.resolve()] = (int(path.stem.rsplit('-', 1)[-1]), report, frozen)

    if not reports:
        return None

    validating = set()
    validated = {}

    def validate(path):
        path = path.resolve()
        if path in validated:
            return validated[path]
        if path in validating:
            raise ValueError('position-servo recovery lineage contains a cycle')
        item = reports.get(path)
        if item is None:
            raise ValueError('position-servo recovery predecessor is not a same-source report')
        stamp, report, frozen = item
        validating.add(path)
        parent_name = report.get('recovery_hold_source')
        if parent_name is None:
            if frozen != original_goal:
                raise ValueError('position-servo recovery first admission Goal changed')
            admission_held = original_held
        else:
            parent = Path(parent_name)
            if parent.is_symlink() or parent.resolve() not in reports:
                raise ValueError('position-servo recovery predecessor is not bound')
            parent_path = parent.resolve()
            if reports[parent_path][0] >= stamp:
                raise ValueError('position-servo recovery predecessor is not older')
            parent_state = validate(parent_path)
            if parent_state['goal_raw'] != frozen:
                raise ValueError('position-servo recovery admission Goal skips its predecessor')
            admission_held = parent_state['held_position']
        failed = report.get('completed') is not True
        state = _recovery_hold_from_report(report, touched, baseline, failed=failed,
            admission_goal=frozen, admission_held=admission_held)
        state.update(path=str(path), report=deepcopy(report), admission_goal_raw=deepcopy(frozen),
                     predecessor_path=None if parent_name is None else str(parent_path),
                     report_snapshot=deepcopy(report))
        validating.remove(path)
        validated[path] = state
        return state

    # The highest numbered same-source report is authoritative. A malformed
    # newest report must fail closed instead of silently falling back to an old
    # effective Goal that may no longer be installed on the arm.
    latest = max(reports, key=lambda path: reports[path][0])
    return validate(latest)


class _Recovery(_Run):
    def __init__(self, service, source, source_object, source_path, admitted, report, wait):
        super().__init__(service.owner, report, lambda:False, service.clock, wait)
        self.service, self.source, self.source_object, self.source_path = service, source, source_object, source_path
        self.touched, self.baseline, self.target_raw = admitted
        self.expected = deepcopy(self.baseline)
        for name in self.touched:
            self.expected[name]['Torque_Limit'] = 200
        self.runtime, self.bridge, self.lease = self.owner.runtime, self.owner.bridge, self.owner.lease
        self.operation_lock, self.thread = service.operation_lock, threading.get_ident()
        self.stop_epoch = service.ros_control_admission_epoch()
        self.stop_generation = service._trial_stop_generation
        self.path = Path(service.output)/('position-servo-recovery-'+str(time.time_ns())+'.json')
        self.armed = False
        self.cleanup_mode = False
        self.last_actual_q = None

    def check(self, *, recovery=False):
        service, owner = self.service, self.owner
        def owns():
            return not (threading.get_ident() != self.thread or service.owner is not owner
                or service.operation_lock is not self.operation_lock or not self.operation_lock.locked()
                or owner.runtime is not self.runtime or owner.bridge is not self.bridge
                or self.bridge.runtime is not self.runtime or owner.lease is not self.lease
                or owner.io.follower.bus is not self.bus)
        if not owns():
            raise RuntimeError('HOME recovery owner/lease/bus/operation identity changed')
        owner._lease_check()
        if not owns():
            raise RuntimeError('HOME recovery ownership changed during lease verification')
        if not recovery and not self.cleanup_mode:
            if (service.phase != 'home' or not owner.stop_event.is_set()
                    or self.runtime.stopped is not True or self.runtime.pending is not None
                    or self.runtime.executing or self.runtime.requires_home_after_incomplete_execution is not True
                    or service.ros_control_admission_epoch() != self.stop_epoch
                    or service._trial_stop_generation != self.stop_generation
                    or self.source_object is not None and service.last_attempt is not self.source_object):
                raise RuntimeError('HOME recovery cancelled or its stopped admission changed')
            service._require_ros_control_admission()

    def source_unchanged(self):
        current = (self.service.last_attempt if self.source_object is not None
                   else json.loads(self.source_path.read_text()))
        if current != self.source:
            raise RuntimeError('failed source changed during HOME recovery')

    def read_all(self, register):
        self.check()
        try:
            values = self.bus.sync_read(register, normalize=False, num_retry=SYNC_READ_NUM_RETRY)
        except ConnectionError as error:
            raise _RegisterReadConnectionError(register, error) from error
        result = _map(values, register)
        self.check()
        return result

    def snapshot_read(self, row, register):
        event = row['register_reads'][register] = dict(
            started_s=float(self.clock()), num_retry=SYNC_READ_NUM_RETRY)
        try:
            values = self.read_all(register)
        except BaseException as error:
            event.update(failed_s=float(self.clock()), error=f'{type(error).__name__}: {error}')
            raise
        event.update(returned_s=float(self.clock()), values=deepcopy(values))
        return values

    def profile(self, row=None):
        columns = {key:(self.read_all(key) if row is None else self.snapshot_read(row, key))
                   for key in (*IDENTITY, *PARAMETERS)}
        for axis, name in enumerate(ORDER):
            if {key:columns[key][name] for key in IDENTITY} != dict(IDENTITY, ID=axis+1):
                raise RuntimeError(name+' actual ID/model/firmware/Lock/mode/torque changed')
        actual = {name:{key:columns[key][name] for key in PARAMETERS} for name in ORDER}
        if actual != self.expected:
            raise RuntimeError('fresh six-motor parameters differ from recorded baseline and touched cap200')
        return actual

    def sample(self, phase, *, sample_index=None, sample_attempt=None):
        row = dict(phase=phase, started_s=float(self.clock()), model_feedback_label=False,
                   register_reads={})
        if sample_index is not None:row['sample_index'] = int(sample_index)
        if sample_attempt is not None:row['sample_attempt'] = int(sample_attempt)
        self.report['readbacks'].append(row)
        with self.owner.bus_lock:
            self.check()
            raw = self.snapshot_read(row, 'Present_Position')
            row.update(present_raw=raw, readback_time_s=float(self.clock()))
            q = self.normalized(raw)
            goal = self.snapshot_read(row, 'Goal_Position')
            row.update(goal_raw=goal, q_position=q.tolist(), parameters_by_motor=self.profile(row),
                moving=self.snapshot_read(row, 'Moving'), status=self.snapshot_read(row, 'Status'),
                velocity=self.snapshot_read(row, 'Present_Velocity'),
                load_raw=self.snapshot_read(row, 'Present_Load'))
            goal_q = self.normalized(goal)
            raw_error = max(abs(raw[name]-goal[name]) for name in ORDER)
            error = float(np.abs(q-goal_q).max())
            step = None if self.last_actual_q is None else float(np.abs(q-self.last_actual_q).max())
            row.update(max_abs_raw_goal_error=raw_error, max_abs_goal_error=error, max_abs_step=step)
            # A stable six-tick J3 residual is 0.532 degrees at the installed
            # calibration. Accept that fixed raw HOLD, while
            # retaining the stricter cross-sample motion check.
            if (goal != self.target_raw or raw_error > HOLD_MAX_RAW_ERROR
                    or error > HOLD_MAX_RESIDUAL+1e-12):
                raise RuntimeError('fresh Present exceeds the six-tick/.6 recorded HOLD tolerance or Goal changed')
            if step is not None and step > HOLD_MAX_STEP+1e-12:
                raise RuntimeError('actual HOLD step exceeds .1 across recovery samples')
            self.last_actual_q = q.copy()
            if any(value for key in ('moving','status','velocity') for value in row[key].values()):
                raise RuntimeError('fresh readback is moving or has a status fault')
            loads = row['load_decoded'] = {}
            for name, value in row['load_raw'].items():
                if not 0 <= value <= 2047:
                    raise RuntimeError('invalid raw load word')
                loads[name] = decode_present_load(value).decoded
            if any(abs(value) > 32 for value in loads.values()):
                raise RuntimeError('fresh HOLD lacks low-load proof on every motor')
        self.geometry(q, np.repeat(q[None], 5, axis=0))
        self.check()
        row['returned_s'] = float(self.clock())
        return row

    def complete_sample(self, stage, phase, sample_index, previous_completed, *, immediate=False):
        """Retry bounded transport failures by reacquiring every register from the start."""
        attempt_base = previous_completed
        for attempt in range(1, SAMPLE_MAX_ATTEMPTS+1):
            scheduled = float(self.clock()) if immediate and attempt == 1 else (
                stage['epoch_s'] if attempt_base is None else attempt_base) + PERIOD_S
            if not (immediate and attempt == 1):
                self.boundary(scheduled)
            try:
                row = self.sample(phase, sample_index=sample_index, sample_attempt=attempt)
            except BaseException as error:
                failed = float(self.clock())
                partial = self.report['readbacks'][-1]
                partial.update(scheduled_start_s=scheduled,
                    start_lateness_s=partial['started_s']-scheduled,
                    schedule_basis='previous_attempt_completion_plus_0.05s', completed=False,
                    failed=True, discarded=isinstance(error, _RegisterReadConnectionError),
                    failed_s=failed, error=f'{type(error).__name__}: {error}')
                if isinstance(error, _RegisterReadConnectionError):
                    partial['failed_register'] = error.register
                if not isinstance(error, _RegisterReadConnectionError):
                    raise
                retry = dict(sample_index=int(sample_index), sample_attempt=attempt,
                    failed_s=failed, error=f'{type(error).__name__}: {error}',
                    failed_register=error.register, retry_started=False)
                if attempt == SAMPLE_MAX_ATTEMPTS:
                    stage['transport_retries'].append(retry)
                    raise
                try:
                    self.check()
                except BaseException as admission_error:
                    retry['retry_blocked_error'] = f'{type(admission_error).__name__}: {admission_error}'
                    stage['transport_retries'].append(retry)
                    raise
                retry['retry_started'] = True
                stage['transport_retries'].append(retry)
                attempt_base = failed
                immediate = False
                continue
            row.update(scheduled_start_s=scheduled,
                start_lateness_s=row['started_s']-scheduled,
                schedule_basis='previous_attempt_completion_plus_0.05s',
                completed=True, discarded=False)
            if row['start_lateness_s'] > LATENESS_S:
                raise RuntimeError('position-servo recovery exceeded the sample-start deadline')
            return row

    def quiet(self, phase):
        epoch = float(self.clock())
        rows = []
        window = self.report[phase] = dict(epoch_s=epoch, completed=False, sample_count=0,
            actual_samples_only=True, minimum_span_s=1., period_s=PERIOD_S,
            deadline_lateness_limit_s=LATENESS_S, maximum_snapshot_attempts=SAMPLE_MAX_ATTEMPTS,
            retry_policy='complete_snapshot_reacquisition_after_transport_connection_error',
            transport_retries=[], schedule_basis='previous_attempt_completion_plus_0.05s')
        previous_completed = None
        for index in range(1, SAMPLES+1):
            row = self.complete_sample(window, phase, index, previous_completed)
            rows.append(row)
            previous_completed = row['returned_s']
            window['sample_count'] = len(rows)
        times = np.array([row['readback_time_s'] for row in rows])
        span = float(times[-1]-times[0])
        if not np.isfinite(times).all() or not (np.diff(times) > 0).all() or span < 1.:
            raise RuntimeError('HOLD proof lacks a full second of consecutive fresh actual samples')
        window.update(completed=True, actual_span_s=span, first_readback_s=float(times[0]),
            last_readback_s=float(times[-1]))

    def write_cap(self, name, value, *, fallback=False):
        if name not in self.touched or value not in (200, 1000) or fallback != (value == 200):
            raise RuntimeError('only recorded touched cap restoration and cap200 fallback are allowed')
        with self.owner.bus_lock:
            self.check(recovery=fallback)
            identity = {}
            for key in IDENTITY:
                self.check(recovery=fallback)
                identity[key] = _integer(self.bus.read(key, name, normalize=False, num_retry=0), key)
                self.check(recovery=fallback)
            if identity != dict(IDENTITY, ID=ORDER.index(name)+1):
                raise RuntimeError(name+' identity changed before cap write')
            event = dict(motor=name, register='Torque_Limit', value=value, fallback=fallback,
                started_s=float(self.clock()), may_have_applied=False, verified=False)
            self.report['parameter_writes'].append(event)
            if not fallback:
                # The source-bound intent is durable before the potentially applied write.
                event['may_have_applied'] = True
                _durable_json(self.path, _plain(self.report))
                self.source_unchanged()
                stage = self.report.setdefault('cap_restore_snapshot', dict(
                    epoch_s=float(self.clock()), maximum_snapshot_attempts=SAMPLE_MAX_ATTEMPTS,
                    retry_policy='complete_snapshot_reacquisition_after_transport_connection_error',
                    transport_retries=[]))
                self.complete_sample(stage, 'immediately_before_cap_restore', 1, None, immediate=True)
            self.check(recovery=fallback)
            event['may_have_applied'] = True
            self.bus.write('Torque_Limit', name, value, normalize=False, num_retry=0)
            event['write_returned_s'] = float(self.clock())
            self.check(recovery=fallback)
            readback = _integer(self.bus.read('Torque_Limit', name, normalize=False, num_retry=0), 'Torque_Limit')
            event.update(readback=readback, verified=readback == value, returned_s=float(self.clock()))
            self.check(recovery=fallback)
            if readback != value:
                raise RuntimeError(name+' cap write was not verified by actual readback')
            self.expected[name]['Torque_Limit'] = value

    def fail_closed(self):
        self.cleanup_mode = True
        self.owner.stop_event.set()
        cleanup = self.report['cleanup']
        for name in self.touched:
            try:
                self.write_cap(name, 200, fallback=True)
                cleanup['cap200_verified'][name] = True
            except BaseException as error:
                cleanup['errors'].append(name+': '+str(error))
                cleanup['cap200_verified'][name] = False
        try:
            with self.owner.bus_lock:
                self.check(recovery=True)
                self.halt()
        except BaseException as error:
            cleanup['errors'].append('existing HOLD: '+str(error))


def restore_failed_position_servo_baseline(service, *, wait=time.sleep):
    """Called only by explicit HOME, with service.operation_lock already held.

    Return None when there is no failed ARM cap to restore. Any relevant missing
    evidence rejects HOME. This function never clears STOP or requires_home.
    """
    if getattr(service.owner.runtime, 'requires_home_after_incomplete_execution', False) is not True:
        return None
    report = dict(schema=SCHEMA, completed=False, configuration_restored=False,
        requires_home=True, model_transition=False, model_feedback_labels_added=0,
        physical_task_success_verified=False, readbacks=[], parameter_writes=[],
        cleanup=dict(cap200_verified={}, errors=[]))
    run = None
    try:
        source_object = getattr(service, 'last_attempt', None)
        source_path = Path(service.output)/'last-attempt.json'
        if source_object is None and not source_path.exists():
            return None
        source = deepcopy(source_object) if source_object is not None else json.loads(source_path.read_text())
        if not isinstance(source, dict):
            raise ValueError('latest failed attempt is not a JSON object')
        admitted = _failed_source(source)
        if admitted is None:
            return None
        effective_profile = failed_position_servo_profile(source, admitted)
        latest_hold = latest_failed_position_servo_hold(service.output, source, admitted)
        original_hold_goal = deepcopy(admitted[2])
        if latest_hold is not None:
            admitted = (admitted[0], admitted[1], latest_hold['goal_raw'])
            effective_profile = latest_hold['profile_state']
        if effective_profile == 'baseline1000':
            # The failed controller or an earlier recovery already restored and
            # verified the saved baseline. HOME owns subsequent motion; there is
            # no parameter write left for this helper to perform.
            return None
        report.update(source_time_s=source.get('source_time_s'), source_kind='service.last_attempt' if source_object is not None
            else 'same_output/last-attempt.json', touched_motors=admitted[0], frozen_hold_goal_raw=admitted[2],
            original_failed_cleanup_hold_goal_raw=original_hold_goal,
            recovery_hold_source=None if latest_hold is None else latest_hold['path'],
            original_baseline_by_motor=admitted[1], started_s=float(service.clock()))
        run = _Recovery(service, source, source_object, source_path, admitted, report, wait)
        run.check()
        run.source_unchanged()
        _durable_json(run.path, _plain(report))
        run.armed = True
        run.quiet('pre_restore')
        for name in run.touched:
            run.write_cap(name, run.baseline[name]['Torque_Limit'])
        run.quiet('post_restore')
        run.check()
        run.source_unchanged()
        report.update(completed=True, configuration_restored=True, stopped=True,
            final_parameters_by_motor=deepcopy(report['readbacks'][-1]['parameters_by_motor']),
            finished_s=float(service.clock()))
        _durable_json(run.path, _plain(report))
        run.check()
        return _plain(report)
    except BaseException as error:
        report.update(completed=False, configuration_restored=False, error=f'{type(error).__name__}: {error}')
        if run is not None and run.armed:
            run.fail_closed()
            report['finished_s'] = float(service.clock())
            try:
                _durable_json(run.path, _plain(report))
            except BaseException as persistence_error:
                report['persistence_error'] = str(persistence_error)
        raise PositionServoRecoveryError(str(error), report) from error
