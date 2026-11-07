"""Bounded J3 output compensation at an already recorded failed HOLD Goal.

This control layer owns no service, file, model, serial port or policy command.
The caller supplies the existing stopped owner and its operation admission.
"""
from copy import deepcopy
import time

import numpy as np

from .servo_register_decode import decode_present_load
from .v21_arm_position_servo import PARAMETERS, _ExactGeometryCache, _geometry_cache_binding
from .v21_position_servo_recovery import HOLD_MAX_RAW_ERROR, HOLD_MAX_RESIDUAL, _failed_source
from .v21_same_goal_integral import IDENTITY, MOTOR, AXIS, ORDER, PERIOD_S, _Run, _integer, _map, _plain


SCHEMA = 'v21-fixed-goal-compensation-v1'
MAX_RESIDUAL = HOLD_MAX_RESIDUAL
ACTIVE_CAP = 300
CLEANUP_CAP = 200
MAX_INTEGRAL_S = 10.
STATIONARY_ADMISSION_BUDGET_S = .5
SAMPLE_MAX_ATTEMPTS = 2
SYNC_READ_NUM_RETRY = 1


class FixedGoalCompensationError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = _plain(report)


class _RegisterReadConnectionError(ConnectionError):
    """Only a transport failure inside this helper's actual sync_read call."""
    def __init__(self, register, error):
        super().__init__(str(error))
        self.register = register


class _Compensation(_Run):
    def __init__(self, owner, execution, report, admission_check, clock, wait):
        super().__init__(owner, report, lambda:False, clock, wait)
        self.original_source, self.source = execution, deepcopy(execution)
        self.admission_check = admission_check
        self.runtime, self.bridge, self.lease = owner.runtime, owner.bridge, owner.lease
        self.bus_lock, self.stop_event = owner.bus_lock, owner.stop_event
        self.admitted = False
        self.cleanup_mode = False
        self.anchor = self.last_q = None
        self.bounds = None
        self.stationary_budget = None
        self.geometry_cache = _ExactGeometryCache(owner, _geometry_cache_binding(owner))
        report['geometry_cache'] = self.geometry_cache.receipt

    def check(self, *, recovery=False):
        recovery = recovery or self.cleanup_mode
        def owns():
            return (self.owner.runtime is self.runtime and self.owner.bridge is self.bridge
                and self.bridge.runtime is self.runtime and self.owner.lease is self.lease
                and self.owner.io.follower.bus is self.bus and self.owner.bus_lock is self.bus_lock
                and self.owner.stop_event is self.stop_event)
        if not owns():raise RuntimeError('fixed-Goal compensation owner identity changed')
        self.admission_check(recovery=recovery)
        self.owner._lease_check()
        if not owns():raise RuntimeError('fixed-Goal ownership changed during lease verification')
        if not recovery:
            if (not self.stop_event.is_set() or self.runtime.stopped is not True
                    or self.runtime.pending is not None or self.runtime.executing
                    or self.runtime.requires_home_after_incomplete_execution is not True):
                raise RuntimeError('fixed-Goal compensation requires the existing stopped HOME interlock')
            self.geometry_cache.validate()
            if self.stationary_budget is not None:
                now = float(self.clock())
                if not np.isfinite(now) or not self.stationary_budget[0] <= now <= self.stationary_budget[1]:
                    raise RuntimeError('stationary admission exceeded its absolute 0.5s budget')
            exposure = self.report.get('integral_exposure')
            if exposure and 'i0_verified_s' not in exposure:
                elapsed = float(self.clock())-exposure['started_s']
                if not np.isfinite(elapsed) or not 0 <= elapsed < MAX_INTEGRAL_S:
                    raise RuntimeError('fixed-Goal I1 exceeded its ten-second budget')

    def read(self, register):
        self.check()
        result = super().read(register)
        self.check()
        return result

    def read_all(self, register):
        self.check()
        try:
            values = self.bus.sync_read(register, normalize=False, num_retry=SYNC_READ_NUM_RETRY)
        except ConnectionError as error:
            raise _RegisterReadConnectionError(register, error) from error
        result = _map(values, register)
        self.check()
        return result

    def profile(self, row=None):
        actual = {name:{} for name in ORDER}
        if row is not None:row['parameters_by_motor'] = actual
        for key in PARAMETERS:
            column = self.read_all(key) if row is None else self.snapshot_read(row, key)
            for name in ORDER:actual[name][key] = column[name]
            if any(column[name] != self.expected[name][key] for name in ORDER):
                raise RuntimeError('actual six-motor profile changed during fixed-Goal compensation')
        return actual

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

    def geometry(self, q, commands=None, *, stalled=False):
        self.owner.clearance.require_readback(q, label='fixed HOLD compensation actual readback')
        self.geometry_cache(q, np.repeat(self.target[None], 5, axis=0))

    def near_goal(self, row):
        return (max(abs(row['present_raw'][name]-self.target_raw[name]) for name in ORDER)
                <= HOLD_MAX_RAW_ERROR
                and np.max(np.abs(np.asarray(row['q_position'])-self.target)) <= MAX_RESIDUAL+1e-12)

    def snapshot(self, phase, *, quiet=False, deadline=None, admission_attempt=None,
                 verify_profile=True, sample_index=None, sample_attempt=None):
        row = dict(phase=phase, started_s=float(self.clock()), model_feedback_label=False)
        row['register_reads'] = {}
        if admission_attempt is not None:row['admission_attempt'] = admission_attempt
        if sample_index is not None:row['sample_index'] = int(sample_index)
        if sample_attempt is not None:row['sample_attempt'] = int(sample_attempt)
        self.report['readbacks'].append(row)
        with self.bus_lock:
            self.check()
            raw = self.snapshot_read(row, 'Present_Position')
            row.update(present_raw=raw, readback_time_s=float(self.clock()))
            q = self.normalized(raw)
            row['q_position'] = q.tolist()
            other = [i for i in range(6) if i != AXIS]
            if (np.max(np.abs(q[other]-self.target[other])) > .1+1e-12
                    or any(abs(raw[ORDER[i]]-self.target_raw[ORDER[i]]) > 1 for i in other)):
                raise RuntimeError('an untouched axis left its stationary one-tick HOLD')
            if self.anchor is not None and np.max(np.abs(q[other]-self.anchor[other])) > .1+1e-12:
                raise RuntimeError('untouched axis drifted from fixed compensation admission')
            if self.bounds is not None and not self.bounds[0]-1e-12 <= q[AXIS] <= self.bounds[1]+1e-12:
                raise RuntimeError('J3 left its admitted fixed-Goal compensation corridor')
            if self.last_q is not None:
                delta = np.abs(q-self.last_q)
                if delta[AXIS] > (.1 if quiet or admission_attempt is not None else .75)+1e-12 or delta[other].max() > .1+1e-12:
                    raise RuntimeError('actual compensation step exceeded its original bound')
            if admission_attempt is not None and abs(q[AXIS]-self.anchor[AXIS]) > .1+1e-12:
                raise RuntimeError('compensation admission did not remain stationary')
            row['goal_raw'] = self.snapshot_read(row, 'Goal_Position')
            if row['goal_raw'] != self.target_raw:
                raise RuntimeError('the original raw HOLD Goal changed')
            row['moving'] = self.snapshot_read(row, 'Moving')
            if (any(row['moving'][ORDER[i]] for i in other)
                    or admission_attempt is not None and any(row['moving'].values())):
                raise RuntimeError('compensation admission or untouched axis did not remain stationary')
            row['torque'] = self.snapshot_read(row, 'Torque_Enable')
            if any(value != 1 for value in row['torque'].values()):
                raise RuntimeError('actual torque or status fault during fixed-Goal compensation')
            row['status'] = self.snapshot_read(row, 'Status')
            if any(row['status'].values()):
                raise RuntimeError('actual torque or status fault during fixed-Goal compensation')
            row['velocity'] = self.snapshot_read(row, 'Present_Velocity')
            if any(row['velocity'][ORDER[i]] for i in other):
                raise RuntimeError('an untouched axis left its stationary one-tick HOLD')
            row['load_raw'] = self.snapshot_read(row, 'Present_Load')
            row['load_decoded'] = {}
            for name, value in row['load_raw'].items():
                if not 0 <= value <= 2047:raise RuntimeError('invalid actual load register')
                row['load_decoded'][name] = decode_present_load(value).decoded
            if any(abs(row['load_decoded'][ORDER[i]]) > 32 for i in other):
                raise RuntimeError('an untouched axis left its stationary one-tick HOLD')
            if abs(row['load_decoded'][MOTOR]) > (32 if quiet else self.expected[MOTOR]['Torque_Limit']):
                raise RuntimeError('actual J3 load exceeds the active compensation limit')
            if verify_profile:self.profile(row)
            if quiet and (not self.near_goal(row) or any(row['moving'].values()) or any(row['velocity'].values())):
                raise RuntimeError('original HOLD was not retained after removing compensation')
            self.last_q = q.copy()
        self.geometry(q)
        self.check()
        row['returned_s'] = float(self.clock())
        if deadline is not None:self.boundary(deadline)
        return row

    def profile_boundary(self, stage, label, *, wall_deadline=None):
        """Verify the six-motor profile with one explicit transport reacquisition."""
        boundary = dict(label=label, completed=False, maximum_attempts=SAMPLE_MAX_ATTEMPTS,
            retry_policy='complete_profile_reacquisition_after_transport_connection_error', attempts=[])
        stage.setdefault('parameter_boundaries', []).append(boundary)
        previous_completed = None
        for attempt in range(1,SAMPLE_MAX_ATTEMPTS+1):
            if previous_completed is not None:
                scheduled = previous_completed+PERIOD_S
                if wall_deadline is not None and scheduled >= wall_deadline:
                    boundary.update(retry_not_started_due_to_wall_deadline=True,
                        wall_deadline_s=wall_deadline,finished_s=float(self.clock()))
                    raise RuntimeError('fixed-Goal profile retry cannot fit before the I1 wall deadline')
                self.boundary(scheduled)
            row = dict(attempt=attempt,started_s=float(self.clock()),register_reads={})
            boundary['attempts'].append(row)
            try:
                with self.bus_lock:
                    self.check()
                    actual = self.profile(row)
            except BaseException as error:
                failed = float(self.clock())
                row.update(completed=False,failed_s=failed,error=f'{type(error).__name__}: {error}')
                if isinstance(error,_RegisterReadConnectionError):row['failed_register'] = error.register
                if not isinstance(error,_RegisterReadConnectionError) or attempt == SAMPLE_MAX_ATTEMPTS:raise
                try:self.check()
                except BaseException as admission_error:
                    row.update(retry_started=False,
                        retry_blocked_error=f'{type(admission_error).__name__}: {admission_error}')
                    raise
                row['retry_started'] = True
                previous_completed = failed
                continue
            row.update(completed=True,returned_s=float(self.clock()))
            boundary.update(completed=True,accepted_attempt=attempt,returned_s=row['returned_s'],
                parameters_by_motor=deepcopy(actual))
            return actual

    def scheduled_snapshot(self, stage, phase, sample_index, previous_completed, *, quiet=False,
                           verify_profile=False, wall_deadline=None):
        """Acquire one complete sample, advancing retries from actual completion."""
        attempt_base = previous_completed
        for attempt in range(1,SAMPLE_MAX_ATTEMPTS+1):
            scheduled = (stage['epoch_s'] if attempt_base is None else attempt_base)+PERIOD_S
            if wall_deadline is not None and scheduled >= wall_deadline:
                stage.update(wall_deadline_reached_s=float(self.clock()),wall_deadline_s=wall_deadline)
                return None
            try:
                self.boundary(scheduled)
            except BaseException:
                now = float(self.clock())
                if np.isfinite(now) and now-scheduled > .03:
                    stage['deadline_miss'] = dict(phase=phase,sample_index=int(sample_index),
                        sample_attempt=attempt,scheduled_start_s=scheduled,observed_s=now,
                        observed_lateness_s=now-scheduled,allowed_lateness_s=.03)
                raise
            try:
                row = self.snapshot(phase,quiet=quiet,verify_profile=verify_profile,
                    sample_index=sample_index,sample_attempt=attempt)
            except BaseException as error:
                failed = float(self.clock())
                partial = self.report['readbacks'][-1]
                partial.update(scheduled_start_s=scheduled,start_lateness_s=partial['started_s']-scheduled,
                    schedule_basis='previous_attempt_completion_plus_0.05s',completed=False,
                    failed=True,discarded=isinstance(error,_RegisterReadConnectionError),failed_s=failed,
                    error=f'{type(error).__name__}: {error}')
                if isinstance(error,_RegisterReadConnectionError):partial['failed_register'] = error.register
                if not isinstance(error,_RegisterReadConnectionError):
                    raise
                retry = dict(sample_index=int(sample_index),sample_attempt=attempt,
                    failed_s=failed,error=f'{type(error).__name__}: {error}',failed_register=error.register,
                    retry_started=False)
                if attempt == SAMPLE_MAX_ATTEMPTS:
                    stage['transport_retries'].append(retry)
                    raise
                # The wrapper is created only around the actual bus sync_read.
                # Re-check STOP, lease, owner and the I1 wall clock before retrying.
                try:self.check()
                except BaseException as admission_error:
                    retry['retry_blocked_error'] = f'{type(admission_error).__name__}: {admission_error}'
                    stage['transport_retries'].append(retry)
                    raise
                retry['retry_started'] = True
                stage['transport_retries'].append(retry)
                attempt_base = failed
                continue
            row.update(scheduled_start_s=scheduled,start_lateness_s=row['started_s']-scheduled,
                schedule_basis='previous_attempt_completion_plus_0.05s',completed=True,discarded=False)
            if row['start_lateness_s'] > .03:
                stage['deadline_miss'] = dict(phase=phase,sample_index=int(sample_index),
                    sample_attempt=attempt,scheduled_start_s=scheduled,observed_s=row['started_s'],
                    observed_lateness_s=row['start_lateness_s'],allowed_lateness_s=.03)
                raise RuntimeError('fixed-Goal compensation exceeded the 0.03s sample-start deadline')
            return row

    def preflight(self):
        self.check()
        admitted = _failed_source(self.source)
        if admitted is None or admitted[0] != [MOTOR]:
            raise ValueError('fixed-Goal compensation supports only this failure\'s actual J3 writes')
        _, self.baseline, self.target_raw = admitted
        self.expected = deepcopy(self.baseline)
        self.expected[MOTOR]['Torque_Limit'] = 200
        with self.bus_lock:
            self.check()
            self.target = self.normalized(self.target_raw)
            identity = self.identity()
            self.coefficient_layout()
            columns = {key:self.read_all(key) for key in IDENTITY}
            for index,name in enumerate(ORDER):
                if {key:values[name] for key,values in columns.items()} != dict(IDENTITY, ID=index+1):
                    raise RuntimeError('actual six-motor identity changed before compensation')
            self.profile()
        self.report.update(source_time_s=self.source.get('source_time_s'), frozen_hold_goal_raw=deepcopy(self.target_raw),
            frozen_hold_goal_position=self.target.tolist(), baseline_parameters_by_motor=deepcopy(self.baseline),
            source_identity=identity)
        self.admitted = True
        row = self.snapshot('admission')
        q = np.asarray(row['q_position'])
        error = abs(float(q[AXIS]-self.target[AXIS]))
        if error > MAX_RESIDUAL+1e-12 or any(row['moving'].values()) or any(row['velocity'].values()):
            raise RuntimeError('admission requires a quiet J3 residual no larger than .6')
        self.anchor = q.copy()
        self.bounds = (max(min(q[AXIS], self.target[AXIS])-.1,
                           q[AXIS]-MAX_RESIDUAL, self.target[AXIS]-MAX_RESIDUAL),
                       min(max(q[AXIS], self.target[AXIS])+.1,
                           q[AXIS]+MAX_RESIDUAL, self.target[AXIS]+MAX_RESIDUAL))
        self.report.update(admission_q=q.tolist(), initial_j3_residual=float(q[AXIS]-self.target[AXIS]),
            actual_j3_corridor=list(self.bounds))
        # Warm exact raw J3 poses before I1. A different actual pose still gets
        # its full physical check; no rounding or model-based clearance shortcut.
        with self.bus_lock:
            low, high = sorted((row['present_raw'][MOTOR], self.target_raw[MOTOR]))
            for cell in range(low-2, high+3):
                self.check()
                raw = dict(row['present_raw'], **{MOTOR:cell})
                probe = self.normalized(raw)
                if self.bounds[0] <= probe[AXIS] <= self.bounds[1]:self.geometry(probe)
        try:
            row = self.stationary_admission()
        finally:
            self.stationary_budget = None
            if 'stationary_admission' in self.report:
                self.report['stationary_admission']['finished_s'] = float(self.clock())
        if self.original_source != self.source:raise RuntimeError('failed source changed before compensation')
        return row

    def stationary_admission(self):
        started = float(self.clock())
        deadline = started+STATIONARY_ADMISSION_BUDGET_S
        window = self.report['stationary_admission'] = dict(started_s=started, deadline_s=deadline,
            budget_s=STATIONARY_ADMISSION_BUDGET_S, maximum_attempts=2, completed=False, attempts=[])
        self.stationary_budget = (started, deadline)
        def within_budget():
            now = float(self.clock())
            if not np.isfinite(now) or not started <= now <= deadline:
                raise RuntimeError('stationary admission exceeded its absolute 0.5s budget')
        for attempt in range(1,3):
            self.check()
            within_budget()
            if self.report['parameter_writes'] or 'integral_exposure' in self.report:
                raise RuntimeError('stationary admission reacquisition is read-only and before I1')
            epoch = float(self.clock())
            current = dict(attempt=attempt, started_s=epoch, completed=False, discarded=False, sample_count=0)
            window['attempts'].append(current)
            try:
                for index in range(1,4):
                    sample_deadline = epoch+index*PERIOD_S
                    if sample_deadline > deadline:
                        raise RuntimeError('stationary admission cannot fit its absolute 0.5s budget')
                    self.boundary(sample_deadline)
                    within_budget()
                    row = self.snapshot('stationary_admission', deadline=sample_deadline,
                        admission_attempt=attempt)
                    within_budget()
                    current['sample_count'] += 1
            except BaseException as error:
                current.update(discarded=True, finished_s=float(self.clock()), error=f'{type(error).__name__}: {error}')
                if isinstance(error, _RegisterReadConnectionError):current['failed_register'] = error.register
                if not isinstance(error, _RegisterReadConnectionError) or attempt == 2:raise
                # Reacquire a complete pre-activation window only. The source,
                # anchor, last valid actual pose and absolute budget never reset.
                self.check()
                within_budget()
                if self.original_source != self.source:raise RuntimeError('failed source changed before compensation')
                continue
            current.update(completed=True, finished_s=float(self.clock()))
            window.update(completed=True, finished_s=float(self.clock()), accepted_attempt=attempt)
            return row

    def write_register(self, register, value, *, recovery=False):
        baseline_cap = self.baseline[MOTOR]['Torque_Limit']
        allowed = {'I_Coefficient':(0,1), 'D_Coefficient':(0,32),
            'Torque_Limit':tuple(dict.fromkeys((CLEANUP_CAP,ACTIVE_CAP,baseline_cap)))}
        if not self.admitted or register not in allowed or value not in allowed[register]:
            raise RuntimeError('write outside the fixed J3 compensation recipe')
        with self.bus_lock:
            self.check(recovery=recovery)
            self.identity()
            if register in ('I_Coefficient','D_Coefficient'):self.coefficient_layout(register)
            independent = recovery and (register,value) in (
                ('I_Coefficient',0),('Torque_Limit',CLEANUP_CAP))
            if not independent:
                actual = {key:self.read(key) for key in ('P_Coefficient','I_Coefficient','D_Coefficient','Torque_Limit')}
                required = {('D_Coefficient',32):(16,0,0,CLEANUP_CAP),
                    ('Torque_Limit',ACTIVE_CAP):(16,0,32,CLEANUP_CAP),
                    ('I_Coefficient',1):(16,0,32,ACTIVE_CAP),
                    ('D_Coefficient',0):(16,0,32,CLEANUP_CAP),
                    ('Torque_Limit',baseline_cap):(16,0,0,CLEANUP_CAP)}.get((register,value))
                already_zero = (recovery and (register,value)==('D_Coefficient',0)
                    and tuple(actual.values())==(16,0,0,CLEANUP_CAP))
                if required is not None and tuple(actual.values()) != required and not already_zero:
                    raise RuntimeError('actual P/I/D/cap does not permit the requested fixed recipe step')
                retained = any(self.report['stages'].get(name,{}).get('completed') is True
                    for name in ('retention','failed_hold_retention'))
                if register == 'Torque_Limit' and value == baseline_cap and not retained:
                    raise RuntimeError('baseline cap restoration requires full actual original-Goal retention')
            event = dict(motor=MOTOR,register=register,value=value,recovery=recovery,
                started_s=float(self.clock()),may_have_applied=False,verified=False)
            self.report['parameter_writes'].append(event)
            if register == 'I_Coefficient' and value == 1:
                self.report['integral_exposure'] = dict(started_s=float(self.clock()),maximum_s=MAX_INTEGRAL_S,
                    within_limit=False,clock_origin='before_I1_write_and_verification')
            self.check(recovery=recovery)
            event['may_have_applied'] = True
            self.bus.write(register,MOTOR,value,normalize=False,num_retry=0)
            event['write_returned_s'] = float(self.clock())
            self.check(recovery=recovery)
            actual = _integer(self.bus.read(register,MOTOR,normalize=False,num_retry=0),register)
            event.update(readback=actual,verified=actual==value,returned_s=float(self.clock()))
            if actual != value:raise RuntimeError('compensation parameter ACK/readback did not verify '+register)
            self.expected[MOTOR][register] = value
            if register == 'I_Coefficient' and value == 0 and 'integral_exposure' in self.report:
                exposure = self.report['integral_exposure']
                exposure.setdefault('i0_verified_s',float(self.clock()))
                exposure['duration_s'] = exposure['i0_verified_s']-exposure['started_s']
                exposure['within_limit'] = exposure['duration_s'] <= MAX_INTEGRAL_S
            self.check(recovery=recovery)

    def quiet_window(self, name):
        epoch = float(self.clock())
        rows = []
        stage = self.report['stages'][name] = dict(epoch_s=epoch,completed=False,sample_count=0,
            actual_samples_only=True,period_s=PERIOD_S,maximum_snapshot_attempts=SAMPLE_MAX_ATTEMPTS,
            retry_policy='complete_snapshot_reacquisition_after_transport_connection_error',
            schedule_basis='previous_attempt_completion_plus_0.05s',transport_retries=[])
        previous_completed = None
        for index in range(1,23):
            row = self.scheduled_snapshot(stage,name,index,previous_completed,quiet=True,
                verify_profile=index in (1,22))
            rows.append(row)
            previous_completed = row['returned_s']
            stage['sample_count'] = len(rows)
        span = rows[-1]['readback_time_s']-rows[0]['readback_time_s']
        if span < 1.:raise RuntimeError('actual retention window is shorter than one second')
        stage.update(completed=True,actual_span_s=span)
        return rows[-1]

    def compensate(self, admission):
        if not self.near_goal(admission):
            self.write_register('D_Coefficient',32)
            self.write_register('Torque_Limit',ACTIVE_CAP)
            self.write_register('I_Coefficient',1)
            wall_deadline = self.report['integral_exposure']['started_s']+MAX_INTEGRAL_S
            stage = self.report['stages']['candidate'] = dict(epoch_s=float(self.clock()),completed=False,
                sample_count=0,actual_samples_only=True,period_s=PERIOD_S,
                maximum_snapshot_attempts=SAMPLE_MAX_ATTEMPTS,
                retry_policy='complete_snapshot_reacquisition_after_transport_connection_error',
                schedule_basis='previous_attempt_completion_plus_0.05s',transport_retries=[],
                wall_deadline_s=wall_deadline)
            self.profile_boundary(stage,'after_i1_verified',wall_deadline=wall_deadline)
            stage['epoch_s'] = float(self.clock())
            tail = []
            previous_completed = None
            for index in range(1,201):
                row = self.scheduled_snapshot(stage,'candidate',index,previous_completed,
                    wall_deadline=wall_deadline)
                if row is None:break
                stage['sample_count'] += 1
                previous_completed = row['returned_s']
                tail = (tail+[row])[-3:]
                if (len(tail)==3 and all(self.near_goal(item) and not any(item['moving'].values()) for item in tail)
                        and np.max(np.abs(np.diff(np.array([item['q_position'] for item in tail]),axis=0))) <= .1+1e-12):
                    stage['completed'] = True
                    break
            if stage['completed'] is not True:raise RuntimeError('fixed original HOLD not reached within ten seconds')
            self.profile_boundary(stage,'before_i0',wall_deadline=wall_deadline)
            self.write_register('I_Coefficient',0,recovery=True)
            self.check()
            self.write_register('Torque_Limit',CLEANUP_CAP,recovery=True)
            self.check()
            self.write_register('D_Coefficient',0,recovery=True)
            self.check()
            if not self.report['integral_exposure']['within_limit']:
                raise RuntimeError('actual I1 exposure exceeded ten seconds')
        self.quiet_window('retention')
        self.write_register('Torque_Limit',self.baseline[MOTOR]['Torque_Limit'])
        endpoint = self.quiet_window('restored')
        self.check()
        if self.original_source != self.source:raise RuntimeError('failed source changed during compensation')
        self.report.update(completed=True,configuration_restored=True,original_goal_preserved=True,
            final_parameters_by_motor=deepcopy(endpoint['parameters_by_motor']),endpoint=deepcopy(endpoint))

    def fail_closed(self):
        self.cleanup_mode = True
        self.stop_event.set()
        cleanup = self.report['cleanup']
        cleanup.update(restored=False,baseline_restored=False,torque_restored=False,
            pre_restore_quiet_verified=False,baseline_cap_verified=False,
            post_restore_quiet_verified=False,profile_state='unknown',
            baseline_restore_errors=[])
        def attempt(label, callback):
            try:callback();return True
            except BaseException as error:
                cleanup['errors'].append(label+': '+str(error));return False
        cleanup['integral_zero_verified'] = attempt('I0',lambda:self.write_register('I_Coefficient',0,recovery=True))
        cleanup['low_cap_verified'] = attempt('cap200',
            lambda:self.write_register('Torque_Limit',CLEANUP_CAP,recovery=True))
        if cleanup['integral_zero_verified'] and cleanup['low_cap_verified']:
            cleanup['damping_zero_verified'] = attempt('D0',lambda:self.write_register('D_Coefficient',0,recovery=True))
        def hold():
            with self.bus_lock:
                self.check(recovery=True)
                self.report['goal_writes'] = None  # Existing emergency HOLD may issue a real freeze.
                result = self.owner._halt()
                cleanup['hold'] = deepcopy(result)
                if type(result.get('motor_register_writes')) is int:
                    self.report['goal_writes'] = result['motor_register_writes']
                rows = result.get('attempts') or []
                last = rows[-1] if rows else {}
                if (last.get('goal_register_verified') is not True
                        or _map(last.get('goal_readback_raw'),'cleanup HOLD') != _map(last.get('requested_goal_raw'),'frozen Present')):
                    raise RuntimeError('existing HOLD did not return verified raw freeze evidence')
                self.target_raw = _map(last.get('goal_readback_raw'),'cleanup HOLD')
                self.target = self.normalized(self.target_raw)
                self.anchor = self.target.copy()
                self.bounds = (self.target[AXIS]-MAX_RESIDUAL,self.target[AXIS]+MAX_RESIDUAL)
                self.last_q = None
                cleanup['effective_hold_goal_raw'] = deepcopy(self.target_raw)
        cleanup['hold_verified'] = attempt('existing HOLD',hold)
        ready = all(cleanup.get(key) is True for key in
            ('integral_zero_verified','low_cap_verified','damping_zero_verified','hold_verified'))
        if not ready:
            return
        try:
            self.quiet_window('failed_hold_retention')
            cleanup['pre_restore_quiet_verified'] = True
            self.write_register('Torque_Limit',self.baseline[MOTOR]['Torque_Limit'],recovery=True)
            cleanup['baseline_cap_verified'] = True
            endpoint = self.quiet_window('failed_baseline_restored')
            cleanup.update(restored=True,baseline_restored=True,torque_restored=True,
                post_restore_quiet_verified=True,profile_state='baseline1000',
                final_parameters_by_motor=deepcopy(endpoint['parameters_by_motor']))
        except BaseException as error:
            cleanup['baseline_restore_errors'].append(
                f'baseline restore: {type(error).__name__}: {error}')
            cleanup.update(restored=False,baseline_restored=False,torque_restored=False,
                post_restore_quiet_verified=False,profile_state='unknown')
            fallback = attempt('fallback cap200',
                lambda:self.write_register('Torque_Limit',CLEANUP_CAP,recovery=True))
            cleanup['fallback_low_cap_verified'] = fallback
            cleanup['low_cap_verified'] = fallback
            if fallback:
                cleanup['profile_state'] = 'cleanup_cap200'


def compensate_failed_hold(owner=None, execution=None, *, admission_check=None, enabled=False,
                           clock=time.monotonic, wait=time.sleep):
    """Run one fixed bounded recipe; new STOP must invalidate admission_check.

    The callback receives recovery=True for independent I0/cap200/HOLD cleanup;
    it must still verify the same owner and operation lock in that case.
    """
    report = dict(schema=SCHEMA,enabled=enabled,completed=False,configuration_restored=False,
        mode='i1',motor=MOTOR,maximum_residual=MAX_RESIDUAL,active_cap=ACTIVE_CAP,
        cleanup_cap=CLEANUP_CAP,maximum_integral_s=MAX_INTEGRAL_S,
        stopped=True,requires_home=True,model_transition=False,model_feedback_labels_added=0,
        physical_task_success_verified=False,original_goal_preserved=False,goal_writes=0,
        readbacks=[],parameter_writes=[],stages={},cleanup=dict(errors=[]))
    if enabled is False:return report
    run = None
    try:
        if enabled is not True or not callable(admission_check):
            raise ValueError('compensation needs explicit enablement and the existing operation admission')
        report['started_s'] = float(clock())
        run = _Compensation(owner,execution,report,admission_check,clock,wait)
        run.compensate(run.preflight())
        run.check()
    except BaseException as error:
        report.update(completed=False,configuration_restored=False,error=f'{type(error).__name__}: {error}')
        report.pop('final_parameters_by_motor',None)
        if run is not None and run.admitted:
            run.fail_closed()
            cleanup = report['cleanup']
            if cleanup.get('baseline_restored') is True:
                report.update(configuration_restored=True,
                    final_parameters_by_motor=deepcopy(cleanup['final_parameters_by_motor']))
        report['finished_s'] = float(clock())
        raise FixedGoalCompensationError(str(error),report) from error
    finally:
        report['finished_s'] = float(clock())
    return _plain(report)


def verify_compensated_hold(owner, execution, receipt, *, admission_check):
    """Fresh read-only HOME admission after this same source's completed compensation."""
    if (not isinstance(receipt,dict) or receipt.get('schema')!=SCHEMA
            or receipt.get('completed') is not True or receipt.get('configuration_restored') is not True):
        raise ValueError('completed fixed-Goal compensation receipt required')
    admitted = _failed_source(execution)
    if (admitted is None or admitted[0]!=[MOTOR] or receipt.get('frozen_hold_goal_raw')!=admitted[2]
            or receipt.get('source_time_s')!=execution.get('source_time_s')
            or receipt.get('baseline_parameters_by_motor')!=admitted[1]):
        raise ValueError('compensation receipt does not bind the current failed source')
    report = dict(readbacks=[],parameter_writes=[],cleanup=dict(errors=[]))
    run = _Compensation(owner,execution,report,admission_check,time.monotonic,time.sleep)
    run.baseline,run.target_raw = admitted[1:]
    run.expected = deepcopy(run.baseline)
    with owner.bus_lock:
        run.check()
        run.target = run.normalized(run.target_raw)
        identities = {key:run.read_all(key) for key in IDENTITY}
        for index,name in enumerate(ORDER):
            if {key:values[name] for key,values in identities.items()} != dict(IDENTITY,ID=index+1):
                raise RuntimeError('fresh compensated HOME motor identity changed: '+name)
    result = run.snapshot('fresh_compensated_HOME_admission',quiet=True)
    result.update(schema='v21-fixed-goal-compensation-verification-v1',configuration_restored=True,requires_home=True)
    return result
