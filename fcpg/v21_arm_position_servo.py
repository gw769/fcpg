"""Bounded J3, J4, then J2 finishing of one existing live five-command Goal.

Only controller parameters on the three named arm motors may change. The original
six Goal cells, the five policy commands, and the five model y labels stay frozen.
"""
from copy import deepcopy
from itertools import product
import time

import numpy as np

from .servo_register_decode import decode_present_load
from .v21_j3_position_servo import _PositionRun, _execution_source, SOURCE_FIELDS
from .v21_position_arrival import evaluate_position_arrival, evaluate_selected_position_arrival
from .v21_same_goal_integral import (
    IDENTITY, MAX_STEP, ORDER, PERIOD_S, _array, _integer, _map, _plain,
)


SCHEMA = 'v21-arm-position-servo-v1'
AXES = (2, 3, 1)
PARAMETERS = ('P_Coefficient', 'I_Coefficient', 'D_Coefficient', 'Torque_Limit', 'Max_Torque_Limit')
BASELINE = dict(zip(PARAMETERS, (16, 0, 0, 1000, 1000)))
MAX_CONTROL_S = 45.
MAX_INTEGRAL_S = 10.
J3_MAX_INTEGRAL_S = 20.
CLEANUP_CAP = 200
ACTIVE_CAP = 300
J3_MAX_PULSES = 16
SAMPLE_MAX_ATTEMPTS = 2
SCHEDULER_LATENESS_NOTICE_S = .03
FAILED_HOLD_SAMPLES = 22
FAILED_HOLD_MAX_RAW_ERROR = 6
FAILED_HOLD_MAX_RESIDUAL = .6
FAILED_HOLD_MAX_STEP = .1
PENDING_COUPLING_MAX_STEP = .25
DYNAMIC_SNAPSHOT_REGISTERS = ('Present_Position','Goal_Position','Moving','Torque_Enable',
    'Present_Load','P_Coefficient','I_Coefficient','D_Coefficient','Torque_Limit')


class _DynamicSyncReadConnectionError(ConnectionError):
    """A ConnectionError raised by an actual dynamic-register sync_read."""
    def __init__(self, register, error):
        super().__init__(str(error))
        self.register = register


def _array_key(value):
    array = np.asarray(value)
    return array.dtype.str, array.shape, array.tobytes(order='C')


def _callable_key(value):
    function = getattr(value,'__func__',value)
    return (id(getattr(value,'__self__',None)),id(function),id(getattr(function,'__code__',None)),
            repr(getattr(function,'__defaults__',None)),repr(getattr(function,'__kwdefaults__',None)))


def _geometry_cache_binding(owner):
    """Recognize the exact static FK gate; unknown checks stay uncached.

    Runtime anchors consume only body transforms, six overwritten qpos cells,
    joint ranges, and the two initialized tip offsets. Meshes are not read by
    forward_anchors. Models with other qpos or mocap state are ineligible.
    """
    checker = getattr(owner._physical_check,'__func__',None)
    if getattr(checker,'__module__',None) != 'fcpg.prod_v20_3_hardware_service':
        return None
    from . import prod_v20_3_hardware_service as hardware
    from . import so101_kinematics as fk
    import scripts.serve_so101_prod_v17_7 as gate_module
    gate = owner.clearance
    if (checker is not hardware.V203HardwareService._physical_check
            or getattr(owner._physical_check,'__self__',None) is not owner
            or type(gate) is not gate_module.V177ClearanceGate
            or type(gate.kinematics) is not fk.SO101MuJoCoKinematics):
        return None
    kin = gate.kinematics
    if (getattr(owner,'kinematics',None) is not kin or type(kin.model) is not kin._mujoco.MjModel
            or (kin.model.nq,kin.model.njnt,kin.model.nmocap) != (6,6,0)
            or not np.array_equal(np.sort(kin.all_qpos_addresses),np.arange(6))):
        return None
    gate_methods = ('clearance_mm','require_readback','minimum_planned_clearance_mm',
                    'require_planned','require_monotonic_recovery','require_policy_path')
    kin_methods = ('forward_anchors','_set_kinematic_state')
    if (any(name in vars(gate) for name in gate_methods)
            or any(name in vars(kin) for name in kin_methods)):
        return None
    audit = checker.__globals__.get('_audit_v177_chunk')
    if audit is not gate_module._audit_v177_chunk:
        return None
    known_functions = [(getattr(gate,name).__func__,'scripts.serve_so101_prod_v17_7',
                        'V177ClearanceGate.'+name) for name in gate_methods]
    known_functions += [(getattr(kin,name).__func__,'fcpg.so101_kinematics',
                         'SO101MuJoCoKinematics.'+name) for name in kin_methods]
    known_functions += [(audit,'scripts.serve_so101_prod_v17_7','_audit_v177_chunk'),
        (audit.__globals__.get('_nonmonotonic_recovery_joints'),'fcpg.prod_v17_7_live','_nonmonotonic_recovery_joints'),
        (fk.normalized_positions_to_mujoco_qpos,'fcpg.so101_kinematics','normalized_positions_to_mujoco_qpos'),
        (fk.normalized_arm_positions_to_radians,'fcpg.so101_kinematics','normalized_arm_positions_to_radians')]
    if any((getattr(function,'__module__',None),getattr(function,'__qualname__',None)) != (module,name)
           for function,module,name in known_functions):
        return None
    model_arrays = tuple(name for name in dir(kin.model)
        if name.startswith(('body_','jnt_')) and isinstance(getattr(kin.model,name),np.ndarray))+('qpos0',)

    def binding():
        current_gate = owner.clearance
        current_kin = current_gate.kinematics
        model = current_kin.model
        current_checker = owner._physical_check
        audit = current_checker.__func__.__globals__['_audit_v177_chunk']
        audit_globals = audit.__globals__
        recovery = audit_globals['_nonmonotonic_recovery_joints']
        forward = current_kin.forward_anchors.__func__
        normalize = forward.__globals__['normalized_positions_to_mujoco_qpos']
        arm_normalize = normalize.__globals__['normalized_arm_positions_to_radians']
        return (
            tuple(id(value) for value in (current_gate,type(current_gate),current_kin,type(current_kin),
                owner.kinematics,model,current_kin.data,current_kin._forward_lock,current_kin._mujoco)),
            tuple(_callable_key(value) for value in (current_checker,audit,recovery,normalize,arm_normalize,
                current_kin._mujoco.mj_kinematics)),
            tuple(_callable_key(getattr(current_gate,name)) for name in gate_methods),
            tuple(_callable_key(getattr(current_kin,name)) for name in kin_methods),
            tuple(getattr(current_gate,name) for name in ('table_z_mm','planned_clearance_mm',
                'readback_clearance_mm','interpolation_step_position')),
            _array_key(audit_globals['SO101_POSITION_LOWER']),_array_key(audit_globals['SO101_POSITION_UPPER']),
            recovery.__globals__['COMMAND_STEPS'],recovery.__globals__['STATE_DIM'],
            tuple(forward.__globals__['SO101_ANCHOR_NAMES']),
            (model.nq,model.njnt,model.nmocap,model.nbody),
            tuple((name,_array_key(getattr(model,name))) for name in model_arrays),
            tuple(_array_key(getattr(current_kin,name)) for name in
                ('all_qpos_addresses','all_joint_ranges','fixed_tip_local','moving_tip_local')),
            current_kin.gripper_body_id,current_kin.moving_jaw_body_id,
        )
    return binding


class _ExactGeometryCache:
    """Operation-local successful five-row checks, with exact inputs and FK binding."""
    def __init__(self, owner, binding):
        self.owner, self.binding = owner, binding
        self.bound = binding() if binding is not None else None
        self.successful = set()
        self.receipt = dict(enabled=binding is not None,success_hits=0,raw_checks=0,unique_successes=0,
            scope='single_arm_operation_exact_q_and_command_bytes',live_clearance_checks_preserved=True,
            configuration_change='abort',unknown_checker='uncached')

    def validate(self):
        if self.binding is None:
            return
        try:
            unchanged = self.binding() == self.bound
        except Exception as error:
            raise RuntimeError('arm geometry cache checker or FK configuration changed') from error
        if not unchanged:
            raise RuntimeError('arm geometry cache checker or FK configuration changed')

    def __call__(self, q, commands):
        if self.binding is None:
            self.receipt['raw_checks'] += 1
            return self.owner._physical_check(q,commands)
        self.validate()
        key = _array_key(q),_array_key(commands)
        if key in self.successful:
            self.receipt['success_hits'] += 1
            return None
        self.receipt['raw_checks'] += 1
        result = self.owner._physical_check(q,commands)
        self.validate()
        if key != (_array_key(q),_array_key(commands)):
            raise RuntimeError('arm physical checker changed its original geometry inputs')
        if result is None:
            self.successful.add(key)
            self.receipt['unique_successes'] = len(self.successful)
        return result


class ArmPositionServoError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = _plain(report)


def _path(source, target):
    """Geometry only: cover the full path even near a command-box edge."""
    count = max(5, int(np.ceil(np.max(np.abs(target-source)/MAX_STEP))))
    return np.linspace(source, target, count+1)[1:]


class _ArmRun(_PositionRun):
    def __init__(self, owner, execution, report, cancelled, clock, wait):
        super().__init__(owner, execution, report, cancelled, clock, wait)
        self.control_started = None
        self.completed_axes = set()
        self.pending_axes = set()
        self.pending_band_entries = set()
        self.bounds = None
        self.current_axis_started = False
        self.constraints_active = False
        self.final_holding = False
        self.last_actual_q = None
        self.pending_anchor_q = None
        self.deferred_axes = set()
        self.deferred_anchor_q = None
        self.axis_reports = report['axes']
        self.geometry_cache = _ExactGeometryCache(owner,_geometry_cache_binding(owner))
        self.report['geometry_cache'] = self.geometry_cache.receipt
        self._snapshot_register_reads = None
        self._snapshot_all_present_load = False
        self._snapshot_all_present_load_values = None
        self.j3_settled_release_progress = None

    def read(self, register):
        events = self._snapshot_register_reads
        if events is None:
            return super().read(register)
        event = events[register] = dict(started_s=float(self.clock()), num_retry=1,
                                        operation='read', motor=self.motor)
        try:
            if register == 'Present_Load' and self._snapshot_all_present_load:
                event.update(operation='sync_read',motor='all')
                try:
                    values = _map(self.bus.sync_read(register,normalize=False,num_retry=1),register)
                except ConnectionError as error:
                    raise _DynamicSyncReadConnectionError(register,error) from error
                self._snapshot_all_present_load_values = deepcopy(values)
                value = values[self.motor]
                event['values'] = deepcopy(values)
            else:
                value = _integer(self.bus.read(register,self.motor,normalize=False,num_retry=1),register)
        except BaseException as error:
            event.update(failed_s=float(self.clock()),error=f'{type(error).__name__}: {error}')
            raise
        event.update(returned_s=float(self.clock()),value=value)
        return value

    def read_all(self, register):
        events = self._snapshot_register_reads
        if events is None:
            return super().read_all(register)
        event = events[register] = dict(started_s=float(self.clock()), num_retry=1,
                                        operation='sync_read')
        try:
            try:
                raw = self.bus.sync_read(register,normalize=False,num_retry=1)
            except ConnectionError as error:
                raise _DynamicSyncReadConnectionError(register,error) from error
            values = _map(raw,register)
        except BaseException as error:
            event.update(failed_s=float(self.clock()),error=f'{type(error).__name__}: {error}')
            raise
        event.update(returned_s=float(self.clock()),values=deepcopy(values))
        return values

    def _write_active_parameter(self, register, value, expected):
        """Write an arm-only active setting after proving its exact predecessor."""
        if not self.baseline_verified:
            raise RuntimeError('no parameter write before complete baseline verification')
        self.check()
        event = dict(register=register, motor=self.motor, value=value, recovery=False,
                     started_s=float(self.clock()), may_have_applied=False, verified=False)
        self.report['parameter_writes'].append(event)
        with self.owner.bus_lock:
            self.check()
            self.identity()
            if register == 'I_Coefficient':
                event.update(self.coefficient_layout(register)[register])
            parameters = {key:self.read(key) for key in
                          ('P_Coefficient','I_Coefficient','D_Coefficient','Torque_Limit')}
            if parameters != expected:
                raise RuntimeError(register+' active transition lacks its exact verified predecessor')
            self.check()
            event['may_have_applied'] = True
            if register == 'Torque_Limit':
                self.cap_attempted = True
            else:
                self.integral_attempted = True
            self.bus.write(register, self.motor, value, normalize=False, num_retry=0)
            event['write_returned_s'] = float(self.clock())
            readback = self.read(register)
            event.update(readback=readback, verified=readback == value, returned_s=float(self.clock()))
            if readback != value:
                raise RuntimeError(register+' write was not verified')
        self.check()
        if register == 'Torque_Limit':
            self.cap = value
        else:
            self.i = value
        return event

    def write_register(self, register, value, *, recovery=False):
        active_cap = ACTIVE_CAP
        if not recovery and (register, value) == ('Torque_Limit', active_cap):
            return self._write_active_parameter(register, value,
                dict(P_Coefficient=16,I_Coefficient=0,D_Coefficient=32,Torque_Limit=CLEANUP_CAP))
        if not recovery and (register, value) == ('I_Coefficient', 1) and self.cap == active_cap:
            return self._write_active_parameter(register, value,
                dict(P_Coefficient=16,I_Coefficient=0,D_Coefficient=32,Torque_Limit=active_cap))
        exposure = self.axis_reports.get(self.motor,{}).get('integral_exposure')
        if recovery and (register,value) == ('I_Coefficient',0) and exposure:
            # The inherited writer closes exactly one pulse on a verified I0.
            # Give it the current pulse, including after an uncertain I1 write;
            # a previous pulse's I0 must never hide a later open integrator.
            self.report['integral_exposure'] = exposure['pulses'][-1]
            try:
                return super().write_register(register,value,recovery=True)
            finally:
                self.update_integral_exposure(exposure)
                self.report['integral_exposure'] = exposure
        return super().write_register(register, value, recovery=recovery)

    def update_integral_exposure(self, exposure):
        pulses = exposure['pulses']
        durations = [pulse.get('i0_verified_s',float(self.clock()))-pulse['started_s'] for pulse in pulses]
        closed = all('i0_verified_s' in pulse for pulse in pulses)
        duration = float(sum(durations))
        exposure.update(duration_s=duration,pulse_count=len(pulses),
            all_pulses_i0_verified=closed,
            within_limit=bool(closed and all(np.isfinite(value) and value >= 0 for value in durations)
                              and duration <= exposure['maximum_s']
                              and all(value<=pulse['maximum_s'] for value,pulse in zip(durations,pulses))))
        if closed:
            exposure['i0_verified_s'] = pulses[-1]['i0_verified_s']
        else:
            exposure.pop('i0_verified_s',None)

    def integral_pulse(self, name, *, landing_tolerance=.2, correction=None):
        report = self.axis_reports[self.motor]
        exposure = report.get('integral_exposure')
        if exposure is None:
            exposure = dict(started_s=float(self.clock()),
                maximum_s=J3_MAX_INTEGRAL_S if self.axis==2 else MAX_INTEGRAL_S,
                within_limit=False,pulses=[],duration_s=0.,
                clock_origin='sum_of_pulses_before_I1_write_through_verified_I0')
            report['integral_exposure'] = exposure
        if exposure['pulses']:
            self.update_integral_exposure(exposure)
            if not exposure['within_limit'] or len(exposure['pulses']) >= (J3_MAX_PULSES if self.axis==2 else 2):
                raise RuntimeError(self.motor+' lacks a verified bounded I1 correction budget')
        remaining = exposure['maximum_s']-exposure['duration_s']
        if remaining <= PERIOD_S:
            raise RuntimeError(self.motor+' has no remaining cumulative I1 observation budget')
        progress = (self.j3_settled_release_progress
                    if self.axis == 2 and correction is None else None)
        # Time is only a bounded observation budget, never an arrival test.
        # Even the first release can be slow; every pulse spends the remaining
        # cumulative budget and still cuts I1 at first actual motion/arrival.
        maximum = remaining
        epoch = float(self.clock())
        pulse = dict(started_s=epoch,maximum_s=maximum,within_limit=False,
            clock_origin='before_I1_write_and_verification',stage=name)
        if self.axis == 2:
            pulse['budget_basis'] = ('verified_same_side_settled_release_progress'
                                     if progress is not None else 'remaining_cumulative_budget')
            if progress is not None:
                pulse['preceding_settled_progress'] = deepcopy(progress)
        exposure['pulses'].append(pulse)
        self.update_integral_exposure(exposure)
        self.report['integral_exposure'] = exposure
        self.write_register('I_Coefficient',1)
        candidate = self.observe_axis(name,epoch=float(self.clock()),integral=True,
            landing_tolerance=landing_tolerance,correction=correction,wall_deadline=epoch+maximum,
            stop_after_stationary_release=self.axis==2 and correction is None)
        self.check(); self.write_register('I_Coefficient',0,recovery=True); self.check()
        candidate['integral_zero_verified_s'] = pulse['i0_verified_s']
        if candidate.get('correction_rejected'):
            raise RuntimeError(self.motor+' bounded I0 rebound correction '+candidate['stop_reason'])
        if ('target_band_entry' not in candidate
                and candidate.get('stop_reason') not in
                    ('first_actual_motion_after_stationary_integral','i1_observation_budget_exhausted')):
            raise RuntimeError(self.motor+' original target or bounded motion not reached within the I1 pulse budget')
        if not exposure['within_limit']:
            raise RuntimeError(self.motor+' cumulative I1 exposure exceeded its bounded budget')
        return candidate

    @staticmethod
    def _initialize_scheduler_receipt(stage):
        stage.setdefault('scheduler_lateness_notice_s',SCHEDULER_LATENESS_NOTICE_S)
        stage.setdefault('scheduler_lateness_policy','record_and_continue_with_fresh_actual_checks')
        stage.setdefault('scheduler_lateness_events',[])
        stage.setdefault('maximum_observed_start_lateness_s',0.)

    def _record_scheduler_lateness(self, stage, phase, sample_index, attempt,
                                   scheduled, observed, *, snapshot_started, **details):
        if not np.isfinite(observed) or observed < scheduled-1e-8:
            raise RuntimeError('arm position servo sampling clock is invalid')
        lateness = max(0.,float(observed-scheduled))
        stage['maximum_observed_start_lateness_s'] = max(
            float(stage['maximum_observed_start_lateness_s']),lateness)
        if lateness > SCHEDULER_LATENESS_NOTICE_S:
            event = dict(phase=phase,sample_index=int(sample_index),snapshot_attempt=int(attempt),
                scheduled_start_s=float(scheduled),observed_start_s=float(observed),
                observed_lateness_s=lateness,notice_threshold_s=SCHEDULER_LATENESS_NOTICE_S,
                snapshot_started=bool(snapshot_started),scheduler_lateness_only=True)
            event.update(details)
            stage['scheduler_lateness_events'].append(event)
        return lateness

    def _wait_for_scheduled_snapshot(self, stage, phase, sample_index, attempt, scheduled,
                                     *, wall_deadline=None):
        """Wait best-effort for one sample without treating Linux jitter as arm motion."""
        self.check()
        now = float(self.clock())
        while np.isfinite(now) and now < scheduled:
            self.wait(min(.01,scheduled-now))
            self.check()
            now = float(self.clock())
        if not np.isfinite(now):
            raise RuntimeError('arm position servo sampling clock is not finite')
        if wall_deadline is not None and now >= wall_deadline:
            self._record_scheduler_lateness(stage,phase,sample_index,attempt,scheduled,now,
                snapshot_started=False,blocked_by_i1_wall_deadline=True)
            stage.update(wall_deadline_reached_s=now,wall_deadline_s=wall_deadline)
            raise RuntimeError('fixed-target snapshot cannot start after the I1 wall deadline')

    def scheduled_snapshot(self, stage, phase, sample_index, previous_completed, *, expected_goal,
                           wall_deadline=None, all_present_load=False):
        """Acquire one complete sample, retrying one exhausted sync_read only."""
        attempt_base = previous_completed
        attempts = stage.setdefault('snapshot_attempts',[])
        stage.setdefault('maximum_snapshot_attempts',SAMPLE_MAX_ATTEMPTS)
        stage.setdefault('retry_policy','complete_snapshot_reacquisition_after_sync_read_connection_error')
        self._initialize_scheduler_receipt(stage)
        for attempt in range(1,SAMPLE_MAX_ATTEMPTS+1):
            scheduled = (stage['epoch_s'] if attempt_base is None else attempt_base)+PERIOD_S
            schedule_basis = ('previous_sample_completion_plus_0.05s' if attempt == 1
                              else 'previous_attempt_completion_plus_0.05s')
            if wall_deadline is not None and scheduled >= wall_deadline:
                stage.update(wall_deadline_reached_s=float(self.clock()),wall_deadline_s=wall_deadline)
                raise RuntimeError('fixed-target snapshot retry cannot fit before the I1 wall deadline')
            self._wait_for_scheduled_snapshot(stage,phase,sample_index,attempt,scheduled,
                                              wall_deadline=wall_deadline)
            before = len(self.report['readbacks'])
            try:
                row = self.snapshot(phase,expected_goal=expected_goal,
                                    all_present_load=all_present_load)
            except BaseException as error:
                finished = float(self.clock())
                if len(self.report['readbacks']) > before:
                    row = self.report['readbacks'][-1]
                    row.update(sample_index=int(sample_index),snapshot_attempt=attempt,
                        scheduled_start_s=scheduled,start_lateness_s=row['started_s']-scheduled,
                        schedule_basis=schedule_basis,completed=False,
                        failed=True,discarded=True,finished_s=finished,
                        error=f'{type(error).__name__}: {error}')
                    if isinstance(error,_DynamicSyncReadConnectionError):
                        row['failed_register'] = error.register
                    self._record_scheduler_lateness(stage,phase,sample_index,attempt,scheduled,
                                                    row['started_s'],snapshot_started=True)
                    attempts.append(row)
                if not isinstance(error,_DynamicSyncReadConnectionError) or attempt == SAMPLE_MAX_ATTEMPTS:
                    raise
                try:
                    self.check()
                except BaseException as admission_error:
                    row.update(retry_started=False,
                        retry_blocked_error=f'{type(admission_error).__name__}: {admission_error}')
                    raise
                next_scheduled = finished+PERIOD_S
                if wall_deadline is not None and next_scheduled >= wall_deadline:
                    row.update(retry_started=False,retry_not_started_due_to_wall_deadline=True,
                               wall_deadline_s=wall_deadline)
                    raise
                row['retry_started'] = True
                attempt_base = finished
                continue
            row.update(sample_index=int(sample_index),snapshot_attempt=attempt,
                scheduled_start_s=scheduled,start_lateness_s=row['started_s']-scheduled,
                schedule_basis=schedule_basis,completed=True,discarded=False)
            self._record_scheduler_lateness(stage,phase,sample_index,attempt,scheduled,
                                            row['started_s'],snapshot_started=True)
            attempts.append(row)
            if wall_deadline is not None and row['started_s'] >= wall_deadline:
                stage.update(wall_deadline_reached_s=float(self.clock()),wall_deadline_s=wall_deadline)
                raise RuntimeError('fixed-target snapshot started after the I1 wall deadline')
            return row

    def check(self, *, recovery=False):
        super().check(recovery=recovery)
        if self.owner.io.follower.bus is not self.bus:
            raise RuntimeError('arm servo bus ownership changed')
        if not recovery and self.control_started is not None:
            now = float(self.clock())
            elapsed = now-self.control_started
            if not np.isfinite(elapsed) or elapsed < 0 or now >= self.control_started+MAX_CONTROL_S:
                raise RuntimeError('arm position servo exceeded the 45-second control deadline')

    def bind_axis(self, axis):
        if axis not in AXES:
            raise ValueError('only J3, J4, and J2 may be touched')
        self.axis, self.motor = axis, ORDER[axis]
        self.identity_expected = dict(IDENTITY, ID=axis+1)
        self.p, self.i, self.d, self.cap = 16, 0, 0, 1000
        self.source_q = self.previous_q = None
        self.current_axis_started = False
        # The shared writer updates exposure on verified recovery I0 only.
        self.report.pop('integral_exposure', None)
        if self.motor in self.axis_reports and 'integral_exposure' in self.axis_reports[self.motor]:
            self.report['integral_exposure'] = self.axis_reports[self.motor]['integral_exposure']

    def all_parameters(self):
        with self.owner.bus_lock:
            self.check()
            columns = {key:self.read_all(key) for key in PARAMETERS}
        return {name:{key:columns[key][name] for key in PARAMETERS} for name in ORDER}

    def geometry(self, q, commands, *, stalled=False):
        def audit(source, path, *, stalled):
            # The existing owner accepts exactly five rows per audit. Every
            # segment starts at the preceding segment's actual geometric end.
            cursor = np.asarray(source).copy()
            for offset in range(0,len(path),5):
                segment = np.asarray(path[offset:offset+5])
                if len(segment) < 5:
                    segment = np.vstack((segment,np.repeat(segment[-1:,:],5-len(segment),axis=0)))
                self.geometry_cache.validate()
                self.owner.clearance.require_readback(cursor,label='fixed original target diagnostic actual/corridor')
                self.geometry_cache(cursor,segment)
                if stalled:
                    for axis in range(5):
                        frozen = segment.copy(); frozen[:,axis] = cursor[axis]
                        self.geometry_cache(cursor,frozen)
                cursor = segment[-1].copy()
        audit(q, commands, stalled=stalled)
        # The J3-first mixed pose is a different path from the original
        # simultaneous command path. Check it and every stalled-arm variant.
        if self.target is not None:
            cursor = np.asarray(q).copy()
            for axis in AXES:
                endpoint = cursor.copy(); endpoint[axis] = self.target[axis]
                audit(cursor, _path(cursor, endpoint), stalled=True)
                cursor = endpoint

    def geometry_path(self, source, target):
        return _path(source,target)

    def snapshot(self, phase, *, expected_goal=None, recovery=False, deadline=None,
                 all_present_load=False):
        if self._snapshot_register_reads is not None:
            raise RuntimeError('nested arm register snapshot is forbidden')
        register_reads = {}
        before = len(self.report['readbacks'])
        self._snapshot_register_reads = register_reads
        self._snapshot_all_present_load = all_present_load
        self._snapshot_all_present_load_values = None
        try:
            row = super().snapshot(phase,expected_goal=expected_goal,recovery=recovery,deadline=deadline)
        except BaseException as error:
            if len(self.report['readbacks']) > before:
                row = self.report['readbacks'][-1]
                complete = (set(register_reads) == set(DYNAMIC_SNAPSHOT_REGISTERS)
                            and all('returned_s' in event for event in register_reads.values()))
                row.update(register_reads=register_reads,register_snapshot_complete=complete,
                           finished_s=float(self.clock()),error=f'{type(error).__name__}: {error}')
            raise
        finally:
            self._snapshot_register_reads = None
            self._snapshot_all_present_load = False
        row.update(register_reads=register_reads,register_snapshot_complete=True)
        if all_present_load:
            row['all_present_load_raw'] = deepcopy(self._snapshot_all_present_load_values)
        self._snapshot_all_present_load_values = None
        q = np.asarray(row['q_position'])
        previous = self.last_actual_q
        self.last_actual_q = q.copy()
        if recovery:
            return row
        if previous is not None and np.abs(q-previous).max() > .75:
            raise RuntimeError('actual per-sample displacement exceeded 0.75 across servo phases')
        strict_axes = sorted(set(range(6))-self.deferred_axes)
        if self.final_holding and (np.abs(q[strict_axes]-self.target[strict_axes]).max() > .2
                or previous is None or np.abs(q-previous).max() > .1):
            raise RuntimeError('full-second selected Goal hold exceeded position or per-sample step tolerance')
        if not self.constraints_active:
            return row
        for axis in set(range(6))-self.pending_axes-self.deferred_axes:
            if abs(q[axis]-self.target[axis]) > .2:
                raise RuntimeError('a completed or untouched axis left its original target tolerance')
        active = {self.axis} if self.current_axis_started else set()
        for axis in self.pending_axes-active:
            low, high = self.bounds[axis]
            if (not low-1e-5 <= q[axis] <= high+1e-5
                    and abs(q[axis]-self.target[axis]) > .2):
                raise RuntimeError('a pending axis left its original verified command hull')
            if abs(q[axis]-self.pending_anchor_q[axis]) > .25:
                raise RuntimeError('a pending axis drifted beyond its fixed admission corridor')
            previous_error = None if previous is None else abs(previous[axis]-self.target[axis])
            current_error = abs(q[axis]-self.target[axis])
            first_entry = (axis not in self.pending_band_entries and previous_error is not None
                and previous_error > .2 and current_error <= .2 and current_error < previous_error)
            step = None if previous is None else abs(q[axis]-previous[axis])
            # The untouched motor still owns its original Goal. Its first
            # natural entry into that Goal's band can span two encoder cells,
            # while remaining inside the already checked fixed corridor.
            # This is progress only; finish_axis still requires fresh quiet
            # samples before declaring the motor naturally reached.
            accepted_entry_step = first_entry and low-1e-5 <= q[axis] <= high+1e-5
            bounded_coupling_step = (step is not None and .1 < step <= PENDING_COUPLING_MAX_STEP
                and not row['moving'][ORDER[axis]])
            if (row['moving'][ORDER[axis]] or step is not None
                    and step > .1 and not accepted_entry_step and not bounded_coupling_step):
                raise RuntimeError('a pending axis moved during single-axis finishing')
            if bounded_coupling_step and not accepted_entry_step:
                self.report.setdefault('pending_axis_coupling_events',[]).append(dict(
                    motor=ORDER[axis],active_motor=self.motor,phase=phase,
                    readback_time_s=row['readback_time_s'],previous_position=float(previous[axis]),
                    actual_position=float(q[axis]),original_target=float(self.target[axis]),
                    previous_abs_error=float(previous_error),abs_error=float(current_error),
                    abs_step=float(step),maximum_step=float(PENDING_COUPLING_MAX_STEP),
                    fixed_anchor_position=float(self.pending_anchor_q[axis]),
                    fixed_anchor_offset=float(q[axis]-self.pending_anchor_q[axis]),
                    command_hull=[float(low),float(high)],moving=False,
                    scheduled_for_later_finishing=True,
                    geometry_basis='prechecked fixed cross-axis corridor'))
            if first_entry:
                self.pending_band_entries.add(axis)
                self.report.setdefault('pending_natural_target_band_entries',[]).append(dict(
                    motor=ORDER[axis],active_motor=self.motor,phase=phase,
                    readback_time_s=row['readback_time_s'],previous_position=float(previous[axis]),
                    actual_position=float(q[axis]),original_target=float(self.target[axis]),
                    previous_abs_error=float(previous_error),abs_error=float(current_error),
                    abs_step=float(step),fixed_anchor_position=float(self.pending_anchor_q[axis]),
                    step_guard_exception=bool(step > .1),stationary_arrival_claimed=False))
        for axis in self.deferred_axes:
            low, high = self.bounds[axis]
            if (abs(q[axis]-self.deferred_anchor_q[axis]) > .25
                    or previous is not None and abs(q[axis]-previous[axis]) > .1
                    or row['moving'][ORDER[axis]]):
                raise RuntimeError('a deferred axis left its fixed stationary admission corridor')
            if (not low-1e-5 <= q[axis] <= high+1e-5
                    and abs(q[axis]-self.target[axis]) > .2):
                raise RuntimeError('a deferred axis left its original verified command hull and target band')
        return row

    def publish_completion(self, positions, times, *, source):
        super().publish_completion(positions,times,source=source)
        hold = self.report.get('original_goal_hold',{})
        if hold.get('completed') and hold.get('selected_axes_arrival') is not None:
            self.report['completion_follow'].update(selected_axes=list(AXES),
                selected_axes_arrival=hold['selected_axes_arrival'],
                deferred_axes=sorted(self.deferred_axes),deferred_anchor_q=self.deferred_anchor_q.tolist(),
                deferred_position_tolerance=.25,source_q0=self.report['source_q0'])

    def preflight(self):
        self.check()
        commands, positions, times, target, target_raw, epoch = _execution_source(self.execution)
        if any(self.live_attempt.get(key) != self.execution.get(key) for key in SOURCE_FIELDS):
            raise RuntimeError('arm servo source is not the bridge current execution evidence')
        self.admitted = True
        self.target, self.target_raw, self.source_epoch = target, target_raw, epoch
        q0 = _array(self.execution['measured_q_path'], (6, 6), 'live actual path')[0]
        hull = np.vstack((q0, commands))
        # The envelope establishes only that a delayed actual readback is
        # explained by a Goal already sent in this transaction. Geometry and
        # drift checks below still admit every recovery path independently.
        self.bounds = {axis:(float(hull[:,axis].min()), float(hull[:,axis].max())) for axis in range(6)}
        self.report.update(original_target=target.tolist(), original_target_raw=target_raw,
            source_time_s=epoch, source_q0=q0.tolist(), source_follow_samples=40, source_follow_budget_s=2.,
            command_envelope=dict(lower=hull.min(0).tolist(), upper=hull.max(0).tolist(), includes_original_q0=True),
            finishing_intervals={ORDER[axis]:list(self.bounds[axis]) for axis in AXES},
            finishing_interval_basis='q0_and_verified_policy_goal_hull')
        baseline = self.all_parameters()
        self.report['baseline_parameters_by_motor'] = baseline
        for axis in AXES:
            self.bind_axis(axis)
            with self.owner.bus_lock:
                self.check()
                identity = self.identity()
                layout = self.coefficient_layout()
            if baseline[self.motor] != BASELINE:
                raise RuntimeError(self.motor+' original P16/I0/D0/Torque1000/Max1000 baseline is unverified')
            self.axis_reports[self.motor] = dict(axis=axis, identity=identity, coefficient_layout=layout,
                original_target=float(target[axis]), baseline_parameters=deepcopy(baseline[self.motor]),
                stages={}, applied=False, completed=False)
        self.bind_axis(AXES[0])
        with self.owner.bus_lock:
            self.check()
            for command, receipt in zip(commands, self.execution['policy_goal_readbacks']):
                if self.encoded(command) != receipt['goal_raw']:
                    raise RuntimeError('current calibration does not reconstruct original Goal cells')
        row = self.snapshot('admission', expected_goal=target_raw)
        q = np.asarray(row['q_position'])
        if np.abs(q-positions[-1]).max() > .25:
            raise RuntimeError('fresh actual position changed from the original follow tail')
        self.geometry(q, np.repeat(q[None], 5, axis=0))
        error = q-target
        self.deferred_axes = {axis for axis in set(range(6))-set(AXES) if abs(error[axis]) > .2}
        self.deferred_anchor_q = q.copy()
        self.report.update(deferred_axes=sorted(self.deferred_axes),deferred_anchor_q=q.tolist(),
            deferred_position_tolerance=.25)
        self.baseline_verified = True
        self.report['original_goal_preserved'] = True
        self.report['admission'] = dict(current_q=q.tolist(), actual_goal_raw=row['goal_raw'])
        actual = [*positions[-3:].tolist(), row['q_position']]
        actual_times = [*times[-3:].tolist(), row['readback_time_s']]
        self.publish_completion(actual, actual_times, source='live_follow_tail_and_fresh_admission')
        if self.report['original_target_reached']:
            return 'already_reached'
        for axis in self.deferred_axes:
            low,high = self.bounds[axis]
            if not low-1e-5 <= q[axis] <= high+1e-5:
                return 'deferred_axis_outside_original_verified_command_hull'
        if np.abs(np.diff(np.asarray(actual), axis=0)).max() > .1 or any(row['moving'].values()):
            return 'actual_follow_tail_not_stationary'
        self.pending_axes = {axis for axis in AXES if abs(error[axis]) > .2}
        for axis in self.pending_axes:
            low, high = self.bounds[axis]
            if not low-1e-5 <= q[axis] <= high+1e-5:
                return 'residual_outside_original_verified_command_hull'
        self.completed_axes = set(AXES)-self.pending_axes
        self.pending_anchor_q = q.copy()
        # Geometry may take long enough for an untouched motor to settle one
        # encoder cell away. Roles are provisional until fresh baseline reads
        # confirm them; no parameter write is permitted during this process.
        classification = self.preflight_roles(q)
        self.pending_axes = set(classification['pending_axes'])
        self.completed_axes = set(classification['completed_axes'])
        receipt = self.report['preflight_classification'] = dict(
            provisional=deepcopy(classification),rebuilds=[],verification_windows=[],
            maximum_rebuilds=1,committed=False)
        for geometry_pass in range(2):
            self.require_unwritten_preflight()
            self.precheck_corridors(q)
            stable = self.verify_preflight_positions(q,geometry_pass)
            fresh_roles = self.preflight_roles(stable)
            refresh_anchor = receipt['verification_windows'][-1]['requires_corridor_rebuild']
            if fresh_roles == classification and not refresh_anchor:
                self.require_unwritten_preflight()
                self.constraints_active = True
                receipt.update(committed=True,final=deepcopy(classification),
                    committed_q=stable.tolist(),committed_s=float(self.clock()))
                return None
            if geometry_pass:
                if refresh_anchor:
                    raise RuntimeError('arm preflight anchor changed a second time')
                raise RuntimeError('arm preflight classification changed a second time')
            self.require_unwritten_preflight()
            receipt['rebuilds'].append(dict(previous=deepcopy(classification),
                updated=deepcopy(fresh_roles),anchor_q=stable.tolist(),
                previous_anchor_q=q.tolist(),
                reason='fresh_stationary_anchor_changed' if refresh_anchor else 'classification_changed',
                before_any_parameter_write=True,stationary_arrival_claimed=False))
            q = stable.copy()
            classification = fresh_roles
            self.pending_axes = set(classification['pending_axes'])
            self.deferred_axes = set(classification['deferred_axes'])
            self.completed_axes = set(classification['completed_axes'])
            self.pending_anchor_q = self.deferred_anchor_q = q.copy()
            self.report.update(deferred_axes=sorted(self.deferred_axes),deferred_anchor_q=q.tolist())
            self.report.setdefault('initial_admission',deepcopy(self.report['admission']))
            anchor_row = receipt['verification_windows'][-1]['readbacks'][-1]
            self.report['admission'] = dict(current_q=q.tolist(),
                actual_goal_raw=deepcopy(anchor_row['goal_raw']),
                readback_time_s=anchor_row['readback_time_s'],
                source='verified_preflight_corridor_rebuild',
                verification_window_index=len(receipt['verification_windows'])-1)

    def require_unwritten_preflight(self):
        self.check()
        if self.control_started is not None or self.report['parameter_writes']:
            raise RuntimeError('arm preflight classification cannot change after a parameter write')

    def preflight_roles(self, q):
        residual = np.abs(np.asarray(q)-self.target)
        pending = {axis for axis in AXES if residual[axis] > .2}
        needs_finishing = sorted(pending)
        deferred = {axis for axis in set(range(6))-set(AXES) if residual[axis] > .2}
        for axis in pending|deferred:
            low,high = self.bounds[axis]
            if not low-1e-5 <= q[axis] <= high+1e-5:
                raise RuntimeError('fresh preflight residual is outside the original verified command hull')
        # An in-band joint scheduled after an unresolved joint has not yet
        # completed its own stage. Earlier arm motion can displace it by a
        # bounded encoder cell. Queue it under the existing pending corridor,
        # precheck that corridor now, and require fresh arrival at its turn.
        # Joints preceding the first unresolved stage remain strictly held.
        # A command hull wholly inside the target band cannot authorize an
        # outside-band residual, so such a joint retains its strict hold role.
        if pending:
            first = min(AXES.index(axis) for axis in pending)
            pending.update(axis for axis in AXES[first:]
                if max(abs(bound-self.target[axis]) for bound in self.bounds[axis]) > .2)
        return dict(pending_axes=sorted(pending),deferred_axes=sorted(deferred),
                    completed_axes=sorted(set(AXES)-pending),needs_finishing_axes=needs_finishing)

    def verify_preflight_positions(self, anchor, geometry_pass):
        stage = dict(epoch_s=float(self.clock()),geometry_pass=geometry_pass,readbacks=[],
            sample_count=0,completed=False,required_fresh_samples=3,actual_samples_only=True,
            model_feedback_label=False,position_tolerance=.2,step_tolerance=.1,
            precheck_anchor_q=np.asarray(anchor).tolist(),requires_corridor_rebuild=False,
            preceding_readback_time_s=self.report['readbacks'][-1]['readback_time_s'])
        self.report['preflight_classification']['verification_windows'].append(stage)
        previous_completed = None
        previous = fresh_anchor = None
        classifications = []
        self.verify_all_baseline()
        for index in range(1,4):
            self.require_unwritten_preflight()
            row = self.scheduled_snapshot(stage,'admission_after_geometry',index,previous_completed,
                                          expected_goal=self.target_raw)
            q = np.asarray(row['q_position'])
            # The first sample follows an expensive geometry precheck, not
            # a 50 ms stationary interval. It starts a new observation window;
            # snapshot still enforces the unchanged 0.75 actual-read gate.
            # A changed anchor cannot authorize writes: all corridors must be
            # rebuilt and a second fresh window must pass before commitment.
            if fresh_anchor is None:
                fresh_anchor = q.copy()
                stage.update(fresh_anchor_q=q.tolist(),
                    precheck_anchor_abs_change=np.abs(q-anchor).tolist(),
                    preceding_readback_gap_s=row['readback_time_s']-stage['preceding_readback_time_s'])
            if (any(row['moving'].values())
                    or previous is not None and np.abs(q-previous).max() > .1
                    or np.abs(q-fresh_anchor).max() > .1):
                raise RuntimeError('actual baseline changed during the sequential corridor precheck')
            if np.abs(q-anchor).max() > .1:
                stage['requires_corridor_rebuild'] = True
            self.geometry(q,_path(q,self.target),stalled=True)
            row['preflight_parameters_by_motor'] = self.verify_all_baseline()
            self.require_unwritten_preflight()
            classifications.append(self.preflight_roles(q))
            row.update(snapshot_returned_s=row['returned_s'],returned_s=float(self.clock()),
                       preflight_identity_verified=True,live_geometry_verified=True)
            stage['readbacks'].append(row)
            stage['sample_count'] = len(stage['readbacks'])
            previous = q
            previous_completed = row['returned_s']
        if any(roles != classifications[0] for roles in classifications[1:]):
            raise RuntimeError('arm preflight classification changed during fresh verification')
        stage.update(completed=True,classification=classifications[-1],
            actual_span_s=stage['readbacks'][-1]['readback_time_s']-stage['readbacks'][0]['readback_time_s'])
        return q.copy()

    def precheck_corridors(self, q):
        target = self.target
        for axis_report in self.axis_reports.values():
            axis_report.pop('corridor_precheck',None)
        self.report.pop('hold_corridor_precheck',None)
        # Expand the same existing precheck corridor around each sequential
        # stage; explicitly include pending/completed joint drift cross-sections.
        mixed = q.copy()
        for stage_index, axis in enumerate(AXES):
            if axis not in self.pending_axes:
                continue
            others = [other for other in AXES if other != axis]+sorted(self.deferred_axes)
            low = min(mixed[axis], target[axis])-.75
            high = max(mixed[axis], target[axis])+.75
            grid = np.linspace(low, high, int(np.ceil((high-low)/.1))+1)
            margins = [.25 if other in self.deferred_axes or other in self.pending_axes and AXES.index(other) > stage_index else .2
                       for other in others]
            centers = [q[other] if other in self.deferred_axes or other in self.pending_axes and AXES.index(other) > stage_index else target[other]
                       for other in others]
            offsets = [(-margin,0.,margin) for margin in margins]
            combinations = list(product(*offsets))
            for drifts in combinations:
                for value in grid:
                    self.check()
                    probe = mixed.copy(); probe[axis] = value
                    for other,center,drift in zip(others,centers,drifts):
                        probe[other] = center+drift
                    self.geometry(probe, _path(probe, target), stalled=True)
            self.axis_reports[ORDER[axis]]['corridor_precheck'] = dict(
                active_interval=[low, high], active_grid_points=len(grid),
                other_axes=others, other_axis_offsets={ORDER[other]:list(values) for other,values in zip(others,offsets)},
                other_axis_centers={ORDER[other]:float(center) for other,center in zip(others,centers)},
                other_axis_offset_combinations=[list(values) for values in combinations],
                sequential_path_and_stalls_checked=True)
            mixed[axis] = target[axis]
        if not self.pending_axes and self.deferred_axes:
            # No parameter change is needed, but partial completion still
            # requires joint coverage of the selected and deferred hold box.
            axes = list(AXES)+sorted(self.deferred_axes)
            offsets = [(-.25,0.,.25) if axis in self.deferred_axes else (-.2,0.,.2) for axis in axes]
            centers = [q[axis] if axis in self.deferred_axes else target[axis] for axis in axes]
            for drifts in product(*offsets):
                self.check()
                probe = q.copy()
                for axis,center,drift in zip(axes,centers,drifts):
                    probe[axis] = center+drift
                self.geometry(probe,_path(probe,target),stalled=True)
            self.report['hold_corridor_precheck'] = dict(axes=axes,centers=centers,
                offsets=[list(values) for values in offsets],combinations=int(np.prod([len(values) for values in offsets])))
        self.report.setdefault('admission_geometry_passes',[]).append(dict(anchor_q=q.tolist(),
            pending_axes=sorted(self.pending_axes),deferred_axes=sorted(self.deferred_axes),
            corridor_prechecks={name:deepcopy(axis['corridor_precheck']) for name,axis in self.axis_reports.items()
                               if 'corridor_precheck' in axis},
            hold_corridor_precheck=deepcopy(self.report.get('hold_corridor_precheck'))))

    def verify_all_baseline(self):
        actual = self.all_parameters()
        if actual != self.report['baseline_parameters_by_motor']:
            raise RuntimeError('actual parameters differ from the original per-motor baseline')
        original_axis = self.axis
        with self.owner.bus_lock:
            for axis in AXES:
                self.axis, self.motor = axis, ORDER[axis]
                self.identity_expected = dict(IDENTITY, ID=axis+1)
                self.identity(); self.coefficient_layout()
        self.axis, self.motor = original_axis, ORDER[original_axis]
        self.identity_expected = dict(IDENTITY, ID=original_axis+1)
        self.report.update(final_parameters_by_motor=actual, final_equals_baseline=True)
        return actual

    def axis_arrival(self, rows, count=3):
        q = np.array([row['q_position'] for row in rows])[:,self.axis]
        tail = q[-count:]
        errors = tail-self.target[self.axis]
        max_step = float(np.abs(np.diff(tail)).max()) if len(tail) > 1 else None
        reached = len(tail) == count and np.abs(errors).max() <= .2 and max_step is not None and max_step <= .1
        return dict(motor=self.motor, original_target=float(self.target[self.axis]),
            actual_position=float(q[-1]), residual=float(errors[-1]), reached=bool(reached),
            sample_count=len(rows), quiet_samples=count, max_abs_residual=float(np.abs(errors).max()),
            max_abs_step=max_step, position_tolerance=.2, step_tolerance=.1,
            actual_samples_only=True, other_axes_arrival_not_claimed=True)

    def observe_axis(self, name, *, count=40, epoch=None, integral=False, quiet=False, stationary=False,
                     settling=False, landing_tolerance=.2, correction=None,
                     wall_deadline=None,stop_after_stationary_release=False,settle_progress=False):
        epoch = float(self.clock()) if epoch is None else epoch
        stage = dict(epoch_s=epoch, readbacks=[], completed=False, sample_count=0,
                     actual_samples_only=True, model_feedback_label=False, period_s=PERIOD_S,
                     scheduler_lateness_notice_s=SCHEDULER_LATENESS_NOTICE_S,
                     scheduler_lateness_policy='record_and_continue_with_fresh_actual_checks',
                     scheduler_lateness_events=[],maximum_observed_start_lateness_s=0.,
                     schedule_basis='previous_sample_completion_plus_0.05s')
        self.axis_reports[self.motor]['stages'][name] = stage
        indices = (range(1,int(np.ceil((wall_deadline-epoch)/PERIOD_S))+1)
                   if integral and wall_deadline is not None else range(1, count+1))
        previous_completed = None
        previous_raw = None if correction is None else correction['settled_raw']
        release_anchor_raw = None
        stationary_raw = None
        stationary_count = 0
        for index in indices:
            scheduled = (epoch if previous_completed is None else previous_completed)+PERIOD_S
            if wall_deadline is not None and scheduled > wall_deadline:
                stage['wall_deadline_reached_s'] = float(self.clock())
                break
            row = self.scheduled_snapshot(stage,self.motor+'/'+name,index,previous_completed,
                                          expected_goal=self.target_raw,wall_deadline=wall_deadline)
            stage['readbacks'].append(row); stage['sample_count'] += 1
            previous_completed = row['returned_s']
            if quiet and (any(row['moving'].values()) or abs(row['present_load_decoded']) > 32):
                raise RuntimeError(self.motor+' lacks stationary low-output restoration proof')
            raw = row['present_raw'][self.motor]
            in_landing_band = (abs(row['q_position'][self.axis]-self.target[self.axis]) <= landing_tolerance
                               and (correction is None or raw in correction['inner_raws']))
            if integral and in_landing_band:
                # I1 supplies the effort needed to leave a static residual. It
                # must stop on first entry, before waiting for quiet samples:
                # keeping it active during that wait can drive through the
                # entire target band. Entry alone is never arrival evidence.
                stage['target_band_entry'] = dict(sample_index=index,
                    readback_time_s=row['readback_time_s'],actual_position=row['q_position'][self.axis],
                    actual_raw=row['present_raw'][self.motor],position_tolerance=.2,
                    integral_cutoff_tolerance=landing_tolerance,
                    stationary_arrival_claimed=False)
                stage['stop_reason'] = ('first_actual_target_band_entry' if correction is None
                                        else 'first_actual_inner_band_entry')
                break
            if integral and stop_after_stationary_release:
                if release_anchor_raw is not None and raw!=release_anchor_raw:
                    direction=np.sign(self.target_raw[self.motor]-release_anchor_raw)
                    if direction*(raw-release_anchor_raw)<=0:
                        raise RuntimeError(self.motor+' stationary integral released away from the original Goal')
                    stage.update(stop_reason='first_actual_motion_after_stationary_integral',
                        motion_release=dict(stationary_raw=release_anchor_raw,actual_raw=raw,
                            actual_position=row['q_position'][self.axis],
                            readback_time_s=row['readback_time_s'],stationary_arrival_claimed=False))
                    break
                stationary_count=stationary_count+1 if stationary_raw==raw else 1
                stationary_raw=raw
                if stationary_count>=3:
                    release_anchor_raw=raw
            if integral and correction is not None:
                direction = correction['raw_approach_side']
                # Compare with the frozen encoder Goal, not the unquantized
                # float target: the Goal cell itself can straddle that float.
                # A valid inner-band landing wins before either crossing or
                # reversal is considered; the opposite approach can legally
                # cross the Goal cell and settle on its inner neighbour.
                if direction*(raw-self.target_raw[self.motor]) < 0:
                    stage.update(stop_reason='crossed_original_goal_cell',correction_rejected=True)
                    break
                if direction*(raw-previous_raw) > 0:
                    stage.update(stop_reason='reversed_away_from_original_goal',correction_rejected=True)
                    break
                previous_raw = raw
            if settling and len(stage['readbacks']) >= 3:
                tail = stage['readbacks'][-3:]
                if ((self.axis_arrival(tail)['reached'] or settle_progress)
                        and not any(any(item['moving'].values()) for item in tail)
                        and np.abs(np.diff(np.array([item['q_position'] for item in tail]),axis=0)).max() <= .1):
                    stage['stop_reason'] = ('stationary_arrival_after_integral_zero'
                        if self.axis_arrival(tail)['reached'] else 'stationary_residual_after_integral_zero')
                    break
            if wall_deadline is not None and float(self.clock()) >= wall_deadline:
                stage['wall_deadline_reached_s'] = float(self.clock())
                break
        rows = stage['readbacks']
        if not rows:
            raise RuntimeError(self.motor+' observation window has no completed actual sample')
        arrival = self.axis_arrival(rows, count=count if quiet else 3)
        if integral and 'stop_reason' not in stage:
            stage['stop_reason'] = 'i1_observation_budget_exhausted'
        stage.update(completed=True, axis_arrival=arrival,
            actual_span_s=rows[-1]['readback_time_s']-rows[0]['readback_time_s'])
        if stationary and (any(any(row['moving'].values()) for row in rows[-3:])
                or np.abs(np.diff(np.array([row['q_position'] for row in rows[-3:]]),axis=0)).max() > .1):
            raise RuntimeError(self.motor+' baseline is not stationary')
        if quiet and not arrival['reached']:
            raise RuntimeError(self.motor+' lost the original target during baseline restoration')
        return stage

    def rebound_correction(self, candidate, settled):
        """Admit one encoder-cell I0 rebound without changing any arrival gate."""
        receipt = dict(eligible=False,attempted=False,maximum_corrections=1,
            position_tolerance=.2,step_tolerance=.1,original_goal_raw=self.target_raw[self.motor],
            fresh_i0_stationary_samples=3)
        self.axis_reports[self.motor]['i0_rebound_correction'] = receipt
        rows = settled['readbacks']
        entry = candidate['readbacks'][-1]
        tail = rows[-3:]
        entry_error = entry['q_position'][self.axis]-self.target[self.axis]
        errors = np.array([row['q_position'][self.axis]-self.target[self.axis] for row in rows])
        raw_entry = entry['present_raw'][self.motor]
        raw_settled = tail[-1]['present_raw'][self.motor]
        raw_goal = self.target_raw[self.motor]
        side = int(np.sign(raw_entry-raw_goal))
        receipt.update(entry_raw=raw_entry,settled_raw=raw_settled,raw_approach_side=side,
            entry_residual=float(entry_error),settled_residual=float(errors[-1]))
        def reject(reason):
            receipt['reason'] = reason
            raise RuntimeError(self.motor+' original target did not settle after integral zero; '+reason)
        if (len(tail) != 3 or any(any(row['moving'].values()) for row in tail)
                or any(abs(row['present_load_decoded']) > 32 for row in tail)
                or np.abs(np.diff(np.array([row['q_position'] for row in tail]),axis=0)).max() > .1):
            reject('rebound lacks fresh stationary low-output evidence')
        if (side == 0 or entry_error*(self.source_q[self.axis]-self.target[self.axis]) <= 0
                or np.any(errors*entry_error <= 0)
                or any(side*(row['present_raw'][self.motor]-raw_entry) not in (0,1) for row in rows)
                or any(row['present_raw'][self.motor] != raw_settled for row in tail)
                or side*(raw_settled-raw_entry) != 1):
            reject('rebound is not one stationary encoder cell on the original approach side')
        with self.owner.bus_lock:
            self.check()
            calibration = self.bus.calibration[self.motor]
            if not calibration.range_min <= raw_goal-1 < raw_goal+1 <= calibration.range_max:
                reject('original Goal lacks two calibrated encoder neighbours')
            decoded = []
            for offset in (-1,0,1):
                probe = dict(self.target_raw); probe[self.motor] += offset
                decoded.append(float(self.normalized(probe)[self.axis]))
            # Keep every calibration decode under the same bus lock. The
            # normalized-position adapter is bus-bound in production and
            # must not observe a mixed register state.
            raw_values = {}
            for raw in range(max(calibration.range_min,raw_goal-3),
                             min(calibration.range_max,raw_goal+3)+1):
                probe = dict(self.target_raw); probe[self.motor] = raw
                raw_values[raw] = float(self.normalized(probe)[self.axis])
            self.check()
        increments = np.diff(decoded)
        tick = float(abs(increments[0]))
        if (not np.isfinite(tick) or not 0 < tick < .2
                or not np.isclose(increments[0],increments[1],rtol=1e-7,atol=1e-9)
                or not np.isclose(abs(errors[-1]-entry_error),tick,rtol=1e-7,atol=1e-9)
                or abs(errors[-1]) <= abs(entry_error)):
            reject('encoder step cannot establish a smaller symmetric landing band')
        cutoff = .2-tick
        # The second pulse is justified only by a cell inside the smaller
        # landing band left after accounting for one encoder tick. A cell
        # that merely lies in the original +/-0.2 band is not enough.
        inner = {raw for raw,value in raw_values.items()
                 if abs(value-self.target[self.axis]) <= cutoff}
        if not inner:
            reject('encoder quantization leaves no attainable inner landing cell')
        inner_cells = [dict(raw=int(raw),decoded_position=float(raw_values[raw]),
                            residual=float(raw_values[raw]-self.target[self.axis])) for raw in sorted(inner)]
        receipt.update(eligible=True,encoder_step_position=tick,integral_cutoff_tolerance=cutoff,
            decoded_goal_position=decoded[1],cutoff_basis='position_tolerance_minus_one_calibrated_encoder_cell',
            attainable_inner_cells=inner_cells,inner_raws=sorted(inner),
            reason='single_cell_same_side_I0_rebound')
        receipt['attempted'] = True
        corrected = self.integral_pulse('rebound_candidate',landing_tolerance=cutoff,correction=receipt)
        receipt['integral_zero_verified_s'] = corrected['integral_zero_verified_s']
        result = self.observe_axis('rebound_integral_off_settling',settling=True,stationary=True)
        receipt['settled_after_correction'] = result['axis_arrival']['reached']
        if not result['axis_arrival']['reached']:
            raise RuntimeError(self.motor+' original target did not settle after the one bounded I0 rebound correction')
        return result

    def finish_axis(self, axis):
        self.bind_axis(axis)
        report = self.axis_reports[self.motor]
        self.verify_all_baseline()
        fresh = self.snapshot(self.motor+'/stage_admission', expected_goal=self.target_raw)
        q = np.asarray(fresh['q_position'])
        low, high = self.bounds[axis]
        if any(fresh['moving'].values()):
            raise RuntimeError('stage actual source is not stationary')
        self.source_q = self.previous_q = q.copy()
        self.current_axis_started = True
        if abs(q[axis]-self.target[axis]) <= .2:
            self.observe_axis('naturally_reached', count=3, quiet=True)
            self.pending_axes.discard(axis); self.completed_axes.add(axis)
            report.update(completed=True, reason='already_reached_before_axis_write')
            return
        if not low-1e-5 <= q[axis] <= high+1e-5:
            raise RuntimeError('stage actual source no longer fits its original verified command hull')
        report['applied'] = self.report['applied'] = True
        if self.control_started is None:
            self.control_started = float(self.clock())
            self.report['control_exposure'] = dict(started_s=self.control_started, maximum_s=MAX_CONTROL_S, within_limit=False)
        self.write_register('Torque_Limit', CLEANUP_CAP)
        self.observe_axis('cap_baseline', stationary=True)
        self.write_register('D_Coefficient', 32)
        self.observe_axis('damping_baseline', stationary=True)
        self.write_register('Torque_Limit', ACTIVE_CAP)
        if axis==2:
            report['integral_control']=dict(mode='stationary_release_pulses_v3',maximum_pulses=J3_MAX_PULSES,
                maximum_pulse_s=J3_MAX_INTEGRAL_S,maximum_cumulative_s=J3_MAX_INTEGRAL_S,
                physical_speed_bound_verified=False)
        for pulse_index in range(J3_MAX_PULSES if axis==2 else 1):
            start_position=float(self.last_actual_q[axis])
            candidate_name='candidate' if pulse_index==0 else f'release_candidate_{pulse_index+1}'
            settling_name='integral_off_settling' if pulse_index==0 else f'release_settling_{pulse_index+1}'
            candidate = self.integral_pulse(candidate_name)
            released=candidate.get('stop_reason')=='first_actual_motion_after_stationary_integral'
            exhausted=candidate.get('stop_reason')=='i1_observation_budget_exhausted'
            # Every quiet sample follows verified I0. Neither motion nor an
            # I1 target-band entry is a stationary arrival observation.
            settled = self.observe_axis(settling_name,settling=True,stationary=True,
                settle_progress=released or exhausted)
            if settled['axis_arrival']['reached']:
                break
            if exhausted:
                raise RuntimeError(self.motor+' original target not reached after verified I0 settling; '
                    'cumulative I1 observation budget exhausted within the I1 pulse budget')
            if not released:
                self.rebound_correction(candidate,settled)
                break
            final=float(settled['readbacks'][-1]['q_position'][axis])
            before=start_position-self.target[axis];after=final-self.target[axis]
            if before*after<=0 or abs(after)>=abs(before):
                raise RuntimeError(self.motor+' released integral lacks same-side settled progress')
            if axis == 2:
                self.j3_settled_release_progress = dict(
                    candidate_stage=candidate_name,settling_stage=settling_name,
                    integral_zero_verified_s=candidate['integral_zero_verified_s'],
                    settled_readback_time_s=settled['readbacks'][-1]['readback_time_s'],
                    start_position=start_position,settled_position=final,
                    original_target=float(self.target[axis]),stationary_arrival_claimed=False)
        else:
            raise RuntimeError(self.motor+' original target not reached within bounded release pulses')
        self.write_register('Torque_Limit', CLEANUP_CAP, recovery=True); self.check()
        self.write_register('D_Coefficient', 0, recovery=True); self.check()
        self.observe_axis('original_goal_retention', count=6, quiet=True)
        self.write_register('Torque_Limit', 1000)
        self.observe_axis('restored_original_goal', count=3, quiet=True)
        if report['integral_exposure']['within_limit'] is not True:
            raise RuntimeError(self.motor+' I1 exposure exceeded its bounded budget or I0 is unverified')
        report.update(completed=True, reason='axis_reached_after_original_baseline_restoration')
        self.pending_axes.discard(axis); self.completed_axes.add(axis)
        self.verify_all_baseline()

    def hold_original_goal(self):
        self.verify_all_baseline()
        self.current_axis_started = True
        self.pending_axes.clear()
        self.final_holding = True
        if self.last_actual_q is None:
            raise RuntimeError('final hold lacks a fresh preceding actual sample')
        self.source_q = self.previous_q = self.last_actual_q.copy()
        epoch = float(self.clock())
        stage = dict(epoch_s=epoch, readbacks=[], completed=False, actual_samples_only=True,
                     requested_minimum_span_s=1., model_feedback_label=False, period_s=PERIOD_S,
                     scheduler_lateness_notice_s=SCHEDULER_LATENESS_NOTICE_S,
                     scheduler_lateness_policy='record_and_continue_with_fresh_actual_checks',
                     scheduler_lateness_events=[],maximum_observed_start_lateness_s=0.,
                     schedule_basis='previous_sample_completion_plus_0.05s')
        self.report['original_goal_hold'] = stage
        previous_completed = None
        for index in range(1, 23):
            row = self.scheduled_snapshot(stage, 'all_axes_original_goal_hold', index, previous_completed,
                                          expected_goal=self.target_raw,all_present_load=True)
            loads = row['all_present_load_raw']
            if any(row['moving'].values()):
                raise RuntimeError('all-axis original Goal hold is moving')
            for name in self.touched_motors():
                raw = loads[name]
                if not 0 <= raw <= 2047 or abs(decode_present_load(raw).decoded) > 32:
                    raise RuntimeError(name+' final original Goal hold lacks low-output proof')
            previous_completed = row['returned_s']
            stage['readbacks'].append(row)
        rows = stage['readbacks']
        positions = [row['q_position'] for row in rows]
        times = [row['readback_time_s'] for row in rows]
        arrival = evaluate_position_arrival(self.target, positions, np.asarray(times)-self.source_epoch,
            position_tolerance=.2, step_tolerance=.1, quiet_samples=len(rows))
        selected = evaluate_selected_position_arrival(self.target,positions,np.asarray(times)-self.source_epoch,
            selected_axes=AXES,position_tolerance=.2,step_tolerance=.1,quiet_samples=len(rows))
        span = times[-1]-times[0]
        stage.update(sample_count=len(rows), actual_span_s=span, arrival=arrival,selected_axes_arrival=selected)
        if span < 1. or not selected['reached']:
            raise RuntimeError('selected original Goal did not remain reached for a full second after baseline restoration')
        self.verify_all_baseline()
        stage['completed'] = True
        self.report.update(original_goal_held=arrival['reached'],selected_axes_completed=True,selected_axes_held=True)
        self.publish_completion(positions, times, source='actual_full_second_after_selected_axes_baseline_restoration')

    def touched_motors(self):
        return list(dict.fromkeys(event['motor'] for event in self.report['parameter_writes'] if event['may_have_applied']))

    def recovery_parameters(self):
        """Read the complete profile while STOP remains latched."""
        with self.owner.bus_lock:
            self.check(recovery=True)
            columns = {key:self.read_all(key) for key in PARAMETERS}
            self.check(recovery=True)
        return {name:{key:columns[key][name] for key in PARAMETERS} for name in ORDER}

    def failed_hold_window(self, name, goal, expected, *, require_stationary=True):
        """Admit the verified low-output HOLD, then prove restored-cap stability."""
        epoch = float(self.clock())
        samples = FAILED_HOLD_SAMPLES if require_stationary else 1
        stage = dict(completed=False,sample_count=0,actual_samples_only=True,
            minimum_span_s=1. if require_stationary else 0.,period_s=PERIOD_S,readbacks=[],
            stationary_proof_required=require_stationary,stationary_verified=False)
        self.report['cleanup'][name] = stage
        original_axis = self.axis
        with self.owner.bus_lock:
            self.check(recovery=True)
            goal_q = self.normalized(goal)
            self.check(recovery=True)
        previous = None
        try:
            self.bind_axis(AXES[0])
            profile = expected[self.motor]
            self.p,self.i,self.d,self.cap = (profile['P_Coefficient'],profile['I_Coefficient'],
                profile['D_Coefficient'],profile['Torque_Limit'])
            for index in range(1,samples+1):
                deadline = epoch+index*PERIOD_S
                self.boundary(deadline,recovery=True)
                row = self.snapshot(name,expected_goal=goal,recovery=True,all_present_load=True)
                q = np.asarray(row['q_position'])
                raw_error = max(abs(row['present_raw'][motor]-goal[motor]) for motor in ORDER)
                step = None if previous is None else float(np.abs(q-previous).max())
                loads = {}
                for motor,value in row['all_present_load_raw'].items():
                    if not 0 <= value <= 2047:
                        raise RuntimeError('failed cleanup HOLD has an invalid raw load word')
                    loads[motor] = decode_present_load(value).decoded
                row.update(all_present_load_decoded=loads,max_abs_raw_goal_error=raw_error,
                    max_abs_goal_error=float(np.abs(q-goal_q).max()),max_abs_step=step)
                if (raw_error > FAILED_HOLD_MAX_RAW_ERROR
                        or row['max_abs_goal_error'] > FAILED_HOLD_MAX_RESIDUAL+1e-12
                        or require_stationary and (step is not None and step > FAILED_HOLD_MAX_STEP+1e-12
                                                  or any(row['moving'].values()))
                        or any(value != 1 for value in row['torque'].values())
                        or any(abs(value) > 32 for value in loads.values())):
                    raise RuntimeError('failed cleanup HOLD is not stationary, low-load, or within six raw cells')
                if index in (1,samples) and self.recovery_parameters() != expected:
                    raise RuntimeError('failed cleanup HOLD profile differs from the expected saved profile')
                previous = q.copy()
                row['returned_s'] = float(self.clock())
                stage['readbacks'].append(row)
                stage['sample_count'] = len(stage['readbacks'])
            span = stage['readbacks'][-1]['readback_time_s']-stage['readbacks'][0]['readback_time_s']
            if require_stationary and span < 1.:
                raise RuntimeError('failed cleanup HOLD proof is shorter than one second')
            stage.update(completed=True,actual_span_s=span,stationary_verified=require_stationary,
                final_parameters_by_motor=deepcopy(self.recovery_parameters()))
            return stage
        finally:
            self.bind_axis(original_axis)

    def restore_failed_hold_baseline(self, touched, ordered):
        cleanup = self.report['cleanup']
        hold = cleanup.get('hold') or {}
        attempts = hold.get('attempts') or []
        last = attempts[-1] if attempts else {}
        goal = _map(last.get('goal_readback_raw'),'failed cleanup HOLD Goal')
        if (last.get('goal_register_verified') is not True
                or goal != _map(last.get('requested_goal_raw'),'failed cleanup frozen Present')):
            raise RuntimeError('failed cleanup HOLD lacks a verified raw freeze')
        baseline = deepcopy(self.report['baseline_parameters_by_motor'])
        low_profile = deepcopy(baseline)
        for name in touched:
            low_profile[name]['Torque_Limit'] = CLEANUP_CAP
        # A heavy arm can drift under cap200 despite a verified HOLD and low
        # measured output. Do not demand stability from that reduced cap before
        # restoring its original holding capability. Goal, bounds, geometry,
        # identity/profile and low-load checks still precede the cap write.
        self.failed_hold_window('pre_baseline_restore',goal,low_profile,require_stationary=False)
        for name in ordered:
            self.bind_axis(ORDER.index(name))
            profile = low_profile[name]
            self.p,self.i,self.d,self.cap = (profile['P_Coefficient'],profile['I_Coefficient'],
                profile['D_Coefficient'],profile['Torque_Limit'])
            baseline_cap = baseline[name]['Torque_Limit']
            self.write_register('Torque_Limit',baseline_cap,recovery=True)
            cleanup['recovery_by_motor'][name]['baseline_cap_verified'] = True
        post = self.failed_hold_window('post_baseline_restore',goal,baseline)
        cleanup.update(restored=True,baseline_restored=True,torque_restored=True,
            pre_restore_quiet_verified=False,pre_restore_low_load_verified=True,
            post_restore_quiet_verified=True,
            profile_state='baseline1000')
        cleanup['parameters_restored_by_motor'] = {name:True for name in touched}
        self.report['final_parameters_by_motor'] = deepcopy(post['final_parameters_by_motor'])

    def retain_failed_hold_low_cap(self, touched, ordered):
        cleanup = self.report['cleanup']
        cleanup.update(restored=False,baseline_restored=False,torque_restored=False,
            post_restore_quiet_verified=False,profile_state='cleanup_cap200')
        for name in ordered:
            self.bind_axis(ORDER.index(name))
            try:
                self.write_register('Torque_Limit',CLEANUP_CAP,recovery=True)
                cleanup['recovery_by_motor'][name]['low_cap_verified'] = True
            except BaseException as error:
                cleanup['errors'].append(f'{name}/fallback cap200: {type(error).__name__}: {error}')
                cleanup['recovery_by_motor'][name]['low_cap_verified'] = False
            cleanup['parameters_restored_by_motor'][name] = False

    def fail_closed(self):
        self.owner.stop_event.set()
        prior_parameters = self.report.pop('final_parameters_by_motor',None)
        if prior_parameters is not None:
            self.report['parameters_before_failure'] = dict(parameters_by_motor=prior_parameters,
                historical_only=True,final_readback=False)
        previous_arrival = self.report.pop('original_target_arrival',None)
        if previous_arrival is not None:
            self.report['arrival_before_failure'] = previous_arrival
        self.report.update(original_target_reached=False, original_goal_held=False,
            selected_axes_completed=False,selected_axes_held=False,
            position_arrival_invalidated_reason='finishing failed; earlier original-target arrival is historical evidence only')
        cleanup = self.report['cleanup']
        cleanup.update(restored=False,baseline_restored=False,hold_performed=False,
            parameters_restored_by_motor={},baseline_restore_errors=[])
        touched = self.touched_motors()
        cleanup['recovery_by_motor'] = {}
        def attempt(label, action):
            try:
                action(); return True
            except BaseException as error:
                cleanup['errors'].append(f'{label}: {type(error).__name__}: {error}')
                return False
        # Clear the currently active integrator first. Failures on one motor
        # must not block independent I0 attempts on every other touched motor.
        ordered = ([self.motor] if self.motor in touched else [])+[name for name in touched if name != self.motor]
        for name in ordered:
            self.bind_axis(ORDER.index(name))
            status = cleanup['recovery_by_motor'][name] = {}
            status['integral_zero_verified'] = attempt(name+'/I0', lambda:self.write_register('I_Coefficient',0,recovery=True))
        for name in ordered:
            self.bind_axis(ORDER.index(name)); status = cleanup['recovery_by_motor'][name]
            status['low_cap_verified'] = attempt(name+'/cap200', lambda:self.write_register('Torque_Limit',CLEANUP_CAP,recovery=True))
            if status['integral_zero_verified'] and status['low_cap_verified']:
                status['damping_zero_verified'] = attempt(name+'/D0', lambda:self.write_register('D_Coefficient',0,recovery=True))
            cleanup['parameters_restored_by_motor'][name] = False
        if self.admitted:
            self.report['original_goal_preserved'] = False
            cleanup['hold_performed'] = attempt('existing HOLD', self.halt)
        ready = (cleanup.get('hold_performed') is True and cleanup.get('hold_verified') is True
            and touched and all(all(cleanup['recovery_by_motor'][name].get(key) is True for key in
                ('integral_zero_verified','low_cap_verified','damping_zero_verified')) for name in touched))
        if ready:
            try:
                self.restore_failed_hold_baseline(touched,ordered)
            except BaseException as error:
                cleanup['baseline_restore_errors'].append(
                    f'baseline restore: {type(error).__name__}: {error}')
                self.retain_failed_hold_low_cap(touched,ordered)
        else:
            cleanup['profile_state'] = 'cleanup_cap200' if touched else 'baseline1000'
        self.report.pop('completion_follow',None)
        self.report['final_equals_baseline'] = cleanup.get('restored') is True


def finish_original_arm_goal(owner=None, execution=None, *, enabled=False, cancelled=None,
                             clock=time.monotonic, wait=None):
    """Finish eligible J3/J4/J2 residuals once, preserving the original live Goal."""
    report = dict(schema=SCHEMA, enabled=enabled, mode='i1', completed=False, applied=False, skipped=False,
        stopped=False, original_goal_preserved=False, original_goal_held=False, stages={}, axes={}, readbacks=[],
        parameter_writes=[], touched_motors=[], cleanup=dict(restored=False, errors=[], parameters_restored_by_motor={}),
        model_feedback_label=False, model_history_committed=False, model_inference_calls=0, policy_commands_added=0,
        physical_task_success_verified=False, axis_order=[ORDER[axis] for axis in AXES])
    report.update(selected_axes=list(AXES),selected_axes_completed=False,selected_axes_held=False,
        deferred_axes=[],deferred_anchor_q=None,deferred_position_tolerance=.25)
    if enabled is not True or type(enabled) is not bool:
        raise ArmPositionServoError('arm position servo disabled; no IO', report)
    run = None
    try:
        run = _ArmRun(owner,execution,report,cancelled or owner.stop_event.is_set,clock,wait or time.sleep)
        report['started_s'] = float(clock())
        reason = run.preflight()
        if reason is None:
            for axis in AXES:
                if axis in run.pending_axes:
                    run.finish_axis(axis)
            run.hold_original_goal()
            reason = ('original_goal_held_after_all_baseline_restoration' if report['original_goal_held']
                      else 'selected_axes_completed_with_other_axis_residual')
        else:
            report['skipped'] = True
            if reason == 'already_reached':
                run.hold_original_goal()
            else:
                run.verify_all_baseline()
        run.check()
        report['touched_motors'] = run.touched_motors()
        report['cleanup'].update(restored=True, original_goal_preserved=True, hold_performed=False,
            parameters_restored_by_motor={name:True for name in report['touched_motors']},
            no_mutation_attempted=not report['touched_motors'])
        report.update(reason=reason,finished_s=float(clock()))
        if run.control_started is not None:
            report['control_exposure'].update(finished_s=report['finished_s'],
                duration_s=report['finished_s']-run.control_started,
                within_limit=report['finished_s'] < run.control_started+MAX_CONTROL_S)
            if report['control_exposure']['within_limit'] is not True:
                raise RuntimeError('arm position servo exceeded the 45-second control deadline')
        report['completed'] = True
        return _plain(report)
    except BaseException as error:
        report['completed'] = False
        report['error'] = f'{type(error).__name__}: {error}'
        if run is not None:
            try:
                run.fail_closed()
            except BaseException as cleanup_error:
                report['cleanup']['errors'].append(f'{type(cleanup_error).__name__}: {cleanup_error}')
            report.update(finished_s=float(clock()),touched_motors=run.touched_motors())
            if run.control_started is not None:
                report['control_exposure'].update(finished_s=report['finished_s'],
                    duration_s=report['finished_s']-run.control_started,
                    within_limit=report['finished_s'] < run.control_started+MAX_CONTROL_S)
        raise ArmPositionServoError(str(error), report) from error
