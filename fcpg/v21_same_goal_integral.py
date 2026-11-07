"""Explicit, one-shot P32 or damped I1 diagnostic on a frozen target.

The service must hold its operation lock, persist its once-only start receipt,
and clear STOP only for this admitted operation. This helper creates no owner,
opens no port, writes no file, and never reads or updates model history.
I1 requires an intact, restored P32 result and a separate D32 baseline.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Real
import time

import numpy as np

from .servo_register_decode import decode_present_load
from .v21_hardware_state import LOWER, ORDER, UPPER
from .v21_position_arrival import evaluate_position_arrival

MOTOR = 'elbow_flex'
AXIS = 2
MAX_STEP = np.array([1.5, 1.5, 1.5, 1., 1., 2.5])
IDENTITY = dict(ID=3, Model_Number=777, Firmware_Major_Version=3, Firmware_Minor_Version=10,
                Lock=1, Operating_Mode=0, Torque_Enable=1)
PERIOD_S = .05
LATENESS_S = .03
WINDOW_S = 2.


class SameGoalIntegralError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = _plain(report)


def _plain(value):
    """Keep failure evidence serializable, including existing HOLD ndarrays."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _array(value, shape, label):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf' or raw.shape != shape:
        raise ValueError(label+' has invalid numeric shape')
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(label+' is nonfinite')
    return result


def _integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(label+' is not a raw integer')
    return int(value)


def _map(value, label):
    if not isinstance(value, dict) or set(value) != set(ORDER):
        raise ValueError(label+' is not a complete six-motor map')
    return {name:_integer(value[name], label+'/'+name) for name in ORDER}


def _ramp(source, target):
    current = np.asarray(source, dtype=np.float64).copy()
    rows = []
    for _ in range(5):
        current = current+np.clip(target-current, -MAX_STEP, MAX_STEP)
        rows.append(current.copy())
    if not np.allclose(current, target, atol=1e-6, rtol=0):
        raise ValueError('original target cannot be reached by the original five bounded commands')
    return np.asarray(rows, dtype=np.float32)


def _source(execution):
    if not isinstance(execution, dict) or execution.get('completed') is not True:
        raise ValueError('source must be an actually completed saved execution')
    for field in ('policy_commands_attempted', 'policy_commands_returned', 'policy_goal_reads_attempted',
                  'policy_goal_reads_returned', 'policy_goal_registers_verified'):
        if type(execution.get(field)) is not int or execution[field] != 5:
            raise ValueError('source lacks five verified commands/Goal reads: '+field)
    commands = _array(execution.get('commands_position'), (5, 6), 'source commands')
    measured = _array(execution.get('measured_q_path'), (6, 6), 'source actual path')
    times = _array(execution.get('readback_times_s'), (6,), 'source actual clocks')
    follow = execution.get('normal_follow') or {}
    if (follow.get('budget_s') != 2. or follow.get('fixed_observation_window') is not True
            or follow.get('reason') != 'fixed_window_complete'):
        raise ValueError('source must retain the full two-second observation window')
    tail = _array(follow.get('positions'), (40, 6), 'source follow actual positions')
    tail_times = _array(follow.get('times_s'), (40,), 'source follow actual clocks')
    if (not (np.diff(times) > 0).all() or not (np.diff(tail_times) > 0).all()
            or tail_times[0] <= times[-1] or not 1.97 <= tail_times[-1]-times[-1] <= 2.03):
        raise ValueError('source readback clocks do not establish a full two-second follow')
    scheduled = times[-1]+PERIOD_S*np.arange(1, 41)
    if np.abs(tail_times-scheduled).max() > LATENESS_S:
        raise ValueError('source follow samples do not retain the original 20Hz window')
    if np.ptp(tail, axis=0).max() > .1:
        raise ValueError('source forty-sample tail is not stationary')
    previous_commands = np.vstack((measured[0], commands[:-1]))
    if (np.abs(commands-previous_commands) > MAX_STEP+1e-5).any():
        raise ValueError('source commands exceed the original per-axis MAX_STEP')
    target = commands[-1].copy()
    error = tail[-1]-target
    if not .2 < abs(error[AXIS]) <= 3. or np.abs(np.delete(error, AXIS)).max() > .25:
        raise ValueError('source lacks the bounded isolated J3 position residual')
    goals = execution.get('policy_goal_readbacks')
    if not isinstance(goals, list) or len(goals) != 5:
        raise ValueError('source lacks five original raw Goal receipts')
    for index, goal in enumerate(goals):
        if (goal.get('command_index') != index or goal.get('goal_register_verified') is not True
                or goal.get('model_feedback_label') is not False):
            raise ValueError('source Goal receipt is not independently verified')
        expected = _map(goal.get('expected_goal_raw'), 'expected source Goal')
        if expected != _map(goal.get('goal_raw'), 'actual source Goal'):
            raise ValueError('source raw Goal receipt differs from its requested cells')
        requested = goal.get('requested_goal_position')
        if not isinstance(requested, dict) or set(requested) != set(ORDER):
            raise ValueError('source Goal lacks its actual requested command')
        if not np.array_equal(_array([requested[k] for k in ORDER], (6,), 'source requested Goal'), commands[index]):
            raise ValueError('source Goal receipt does not bind the recorded command')
    arrival = execution.get('original_target_arrival') or {}
    if not np.array_equal(_array(arrival.get('original_target'), (6,), 'source original target'), target):
        raise ValueError('source original target was replaced')
    return commands, measured, tail, target, _map(goals[-1]['goal_raw'], 'frozen original Goal')


def _p32_source(result):
    """Read the actual P32 result directly, without inventing an execution."""
    if (not isinstance(result, dict) or result.get('schema') != 'v21-same-goal-service-result-v1'
            or result.get('mode') != 'p32' or result.get('state') != 'completed'
            or result.get('completed') is not True or result.get('stopped') is not True
            or result.get('cleanup', {}).get('restored') is not True):
        raise ValueError('i1 source must be a completed, restored P32 service result')
    diagnostic = result.get('diagnostic_result')
    if (not isinstance(diagnostic, dict) or diagnostic.get('schema') != 'v21-same-goal-gain-diagnostic-v1'
            or diagnostic.get('mode') != 'p32' or diagnostic.get('completed') is not True
            or diagnostic.get('cleanup', {}).get('restored') is not True):
        raise ValueError('i1 source lacks the completed, restored P32 diagnostic')
    target = _array(diagnostic.get('original_target'), (6,), 'P32 frozen original target')
    target_raw = _map(diagnostic.get('original_target_raw'), 'P32 frozen original Goal')
    for arrival in (diagnostic.get('original_target_arrival', {}), result.get('original_target_arrival', {})):
        if not np.array_equal(_array(arrival.get('original_target'), (6,), 'P32 arrival target'), target):
            raise ValueError('P32 arrival replaced the original target')
    previous_returned = None
    tail = None
    for name, p, cap in (('baseline', 16, 1000), ('cap_baseline', 16, 200), ('candidate', 32, 200)):
        stage = diagnostic.get('stages', {}).get(name, {})
        rows = stage.get('readbacks')
        if (stage.get('completed') is not True or stage.get('fixed_observation_window') is not True
                or stage.get('budget_s') != WINDOW_S or not isinstance(rows, list) or len(rows) != 40):
            raise ValueError('P32 source lacks forty samples over the full two-second '+name)
        epoch = _array([stage.get('epoch_s')], (1,), 'P32 stage epoch')[0]
        if previous_returned is not None and epoch < previous_returned:
            raise ValueError('P32 source stages overlap or are out of order')
        if (not np.array_equal(_array(stage.get('original_target'), (6,), 'P32 stage target'), target)
                or _map(stage.get('original_target_raw'), 'P32 stage Goal') != target_raw):
            raise ValueError('P32 source stage replaced the original target')
        times, positions = [], []
        for index, row in enumerate(rows):
            stamps = _array([row.get(key) for key in ('started_s', 'readback_time_s', 'returned_s')], (3,), 'P32 actual clocks')
            if ((np.diff(stamps) < 0).any() or abs(stamps[1]-epoch-(index+1)*PERIOD_S) > LATENESS_S
                    or stamps[-1]-epoch-(index+1)*PERIOD_S > LATENESS_S):
                raise ValueError('P32 source lacks the original 20Hz actual timing')
            if (_map(row.get('goal_raw'), 'P32 actual Goal') != target_raw
                    or row.get('model_feedback_label') is not False
                    or row.get('parameters') != dict(P_Coefficient=p, I_Coefficient=0, D_Coefficient=0, Torque_Limit=cap)
                    or any(value != 1 for value in _map(row.get('torque'), 'P32 torque').values())):
                raise ValueError('P32 source changed Goal or its recorded controller parameters')
            _map(row.get('present_raw'), 'P32 actual Present')
            _map(row.get('moving'), 'P32 actual Moving')
            positions.append(_array(row.get('q_position'), (6,), 'P32 actual position'))
            times.append(stamps[1])
        if not (np.diff(times) > 0).all() or not 1.97 <= times[-1]-epoch <= 2.03:
            raise ValueError('P32 source actual window is incomplete')
        previous_returned = float(rows[-1]['returned_s'])
        tail = np.asarray(positions)
        if (np.abs(np.diff(tail[-3:], axis=0)).max() > .1
                or any(any(_map(row['moving'], 'P32 tail Moving').values()) for row in rows[-3:])):
            raise ValueError('P32 source '+name+' tail is not stationary')
    error = tail[-1]-target
    if not .2 < abs(error[AXIS]) <= 3. or np.abs(np.delete(error, AXIS)).max() > .25:
        raise ValueError('P32 source lacks the bounded isolated J3 residual')
    return tail, target, target_raw


class _Run:
    def __init__(self, owner, report, cancelled, clock, wait):
        self.owner, self.report = owner, report
        self.cancelled, self.clock, self.wait = cancelled, clock, wait
        self.bus = owner.io.follower.bus
        # Defaults retain the standalone J3 diagnostic contract. A separate
        # active-transaction helper may pin its own axis on this instance.
        self.motor, self.axis = MOTOR, AXIS
        self.identity_expected = dict(IDENTITY)
        self.position_attempted = False
        self.cap_attempted = False
        self.gain_attempted = False
        self.integral_attempted = False
        self.baseline_verified = False
        self.previous_q = None
        self.source_q = None
        self.target = None
        self.target_raw = None
        self.p = 16
        self.i = 0
        self.d = 0
        self.cap = 1000

    def check(self, *, recovery=False):
        if not recovery and (self.owner.stop_event.is_set() or self.cancelled()):
            raise RuntimeError('fixed-target diagnostic cancelled by STOP')
        self.owner._lease_check()

    def boundary(self, deadline, *, recovery=False):
        self.check(recovery=recovery)
        now = float(self.clock())
        while now < deadline:
            self.wait(min(.01, deadline-now))
            self.check(recovery=recovery)
            now = float(self.clock())
        if not np.isfinite(now) or now-deadline > LATENESS_S:
            raise RuntimeError('fixed-target diagnostic exceeded the 0.03s deadline')

    def read(self, register):
        return _integer(self.bus.read(register, self.motor, normalize=False, num_retry=0), register)

    def read_all(self, register):
        return _map(self.bus.sync_read(register, normalize=False, num_retry=0), register)

    def identity(self):
        values = {key:self.read(key) for key in self.identity_expected}
        if values != self.identity_expected:
            raise RuntimeError(f'J{self.axis+1} pinned ID/model/firmware/Lock/mode/torque changed')
        return values

    def coefficient_layout(self, register=None):
        table = self.bus.model_ctrl_table[self.bus.motors[self.motor].model]
        expected = dict(I_Coefficient=(23, 1), D_Coefficient=(22, 1))
        names = expected if register is None else (register,)
        for name in names:
            if table.get(name) != expected[name]:
                raise RuntimeError('I/D control table must be one byte at addresses 23/22; 24/25 writes forbidden')
        return {name: dict(address=expected[name][0], bytes=1) for name in names}

    def normalized(self, raw):
        ids = {key:self.bus.motors[key].id for key in ORDER}
        if list(ids.values()) != [1, 2, 3, 4, 5, 6]:
            raise RuntimeError('six-motor identity mapping changed')
        for key in ORDER:
            calibration = self.bus.calibration[key]
            if not calibration.range_min <= raw[key] <= calibration.range_max:
                raise RuntimeError('actual raw position is outside calibration: '+key)
        values = self.bus._normalize({ids[key]:raw[key] for key in ORDER})
        if not isinstance(values, dict) or set(values) != set(ids.values()):
            raise RuntimeError('position normalization is incomplete')
        q = _array([values[ids[key]] for key in ORDER], (6,), 'normalized actual position')
        if (q < LOWER).any() or (q > UPPER).any():
            raise RuntimeError('actual position is outside the existing joint bounds')
        cells = self.bus._unnormalize({ids[key]:float(q[i]) for i,key in enumerate(ORDER)})
        if any(abs(_integer(cells[ids[key]], 'reconstructed Present')-raw[key]) > 1 for key in ORDER):
            raise RuntimeError('position normalization hides actual raw cells')
        return q

    def encoded(self, q):
        # This is precisely the float32 transport's normalized write conversion.
        q = np.asarray(q, dtype=np.float32)
        ids = {key:self.bus.motors[key].id for key in ORDER}
        cells = self.bus._unnormalize({ids[key]:float(q[i]) for i,key in enumerate(ORDER)})
        return {key:_integer(cells[ids[key]], 'expected Goal') for key in ORDER}

    def geometry(self, q, commands, *, stalled=False):
        self.owner.clearance.require_readback(q, label='fixed original target diagnostic actual/corridor')
        self.owner._physical_check(q, commands)
        if stalled:
            for axis in range(5):
                frozen = commands.copy()
                frozen[:, axis] = q[axis]
                self.owner._physical_check(q, frozen)

    def geometry_path(self, source, target):
        return _ramp(source, target)

    def snapshot(self, phase, *, expected_goal=None, recovery=False, deadline=None):
        self.check(recovery=recovery)
        row = dict(started_s=float(self.clock()), phase=phase, model_feedback_label=False)
        self.report['readbacks'].append(row)
        with self.owner.bus_lock:
            self.check(recovery=recovery)
            raw = self.read_all('Present_Position')
            row.update(present_raw=raw, readback_time_s=float(self.clock()))
            q = self.normalized(raw)
            goal = self.read_all('Goal_Position')
            row.update(q_position=q.tolist(), goal_raw=goal,
                moving=self.read_all('Moving'), torque=self.read_all('Torque_Enable'))
            load = self.read('Present_Load')
            if not 0 <= load <= 2047:
                raise RuntimeError(f'J{self.axis+1} load is not a valid raw direction/magnitude word')
            row.update(present_load_raw=load, present_load_decoded=decode_present_load(load).decoded,
                parameters={key:self.read(key) for key in ('P_Coefficient','I_Coefficient','D_Coefficient','Torque_Limit')})
        self.check(recovery=recovery)
        if expected_goal is not None and goal != expected_goal:
            raise RuntimeError('frozen Goal_Position changed during fixed-target observation')
        if any(value != 1 for value in row['torque'].values()):
            raise RuntimeError('actual torque enable changed')
        if row['parameters'] != dict(P_Coefficient=self.p, I_Coefficient=self.i, D_Coefficient=self.d, Torque_Limit=self.cap):
            raise RuntimeError('fixed-target control parameters changed')
        self.owner.clearance.require_readback(q, label='fixed-target diagnostic actual readback')
        if self.source_q is not None:
            if self.previous_q is not None and np.abs(q-self.previous_q).max() > .75:
                raise RuntimeError('actual per-sample displacement exceeded 0.75')
            if np.abs(np.delete(q-self.source_q, self.axis)).max() > .25:
                raise RuntimeError(
                    f'completed or untouched axis drift: non-J{self.axis+1} actual drift exceeded 0.25')
            low, high = sorted((self.source_q[self.axis], self.target[self.axis]))
            if not low-.25 <= q[self.axis] <= high+.25:
                raise RuntimeError(f'J{self.axis+1} left the original source-target corridor or overshot target')
            if not recovery or phase == 'post_integral_retention':
                self.geometry(q, self.geometry_path(q, self.target))
        self.previous_q = q.copy()
        row['returned_s'] = float(self.clock())
        if deadline is not None:
            self.boundary(deadline, recovery=recovery)
        return row

    def write_register(self, register, value, *, recovery=False):
        mode = self.report['mode']
        allowed = {'Torque_Limit': (200, 1000)}
        if mode == 'p32':
            allowed['P_Coefficient'] = (16, 32)
        else:
            allowed.update(I_Coefficient=(0, 1), D_Coefficient=(0, 32))
        if register not in allowed or value not in allowed[register]:
            raise RuntimeError('register is outside the fixed-target diagnostic write allowlist')
        if not self.baseline_verified:
            raise RuntimeError('no parameter write before complete baseline verification')
        self.check(recovery=recovery)
        event = dict(register=register, motor=self.motor, value=value, recovery=recovery,
                     started_s=float(self.clock()), may_have_applied=False, verified=False)
        self.report['parameter_writes'].append(event)
        with self.owner.bus_lock:
            self.check(recovery=recovery)
            self.identity()
            if mode == 'i1' and register in ('I_Coefficient', 'D_Coefficient'):
                event.update(self.coefficient_layout(register)[register])
            reducing_cap = recovery and register == 'Torque_Limit' and value == 200
            clearing_i = recovery and register == 'I_Coefficient' and value == 0
            if not reducing_cap and not clearing_i:
                parameters = {key: self.read(key) for key in
                              ('P_Coefficient', 'I_Coefficient', 'D_Coefficient', 'Torque_Limit')}
                expected_d = 32 if register == 'I_Coefficient' else 0
                if (parameters['I_Coefficient'] != 0
                        or register != 'D_Coefficient' and parameters['D_Coefficient'] != expected_d):
                    raise RuntimeError('I/D changed; do not tune or increase output with unknown controller state')
                if register == 'Torque_Limit' and parameters['P_Coefficient'] != 16:
                    raise RuntimeError('P16 not proven before torque increase or normal cap change')
                if register == 'P_Coefficient' and (parameters['Torque_Limit'] != 200
                        or not recovery and parameters['P_Coefficient'] != 16):
                    raise RuntimeError('P change requires the verified cap200 baseline')
                if register in ('I_Coefficient', 'D_Coefficient') and (
                        parameters['Torque_Limit'] != 200 or parameters['P_Coefficient'] != 16
                        or register == 'D_Coefficient' and not recovery and parameters['D_Coefficient'] != 0):
                    raise RuntimeError('I/D change requires P16/I0 and the verified cap200 baseline')
            self.check(recovery=recovery)
            event['may_have_applied'] = True
            if register == 'Torque_Limit':
                self.cap_attempted = True
            elif register == 'I_Coefficient':
                self.integral_attempted = True
            else:
                self.gain_attempted = True
            self.bus.write(register, self.motor, value, normalize=False, num_retry=0)
            event['write_returned_s'] = float(self.clock())
            readback = self.read(register)
            event.update(readback=readback, verified=readback == value, returned_s=float(self.clock()))
            if readback != value:
                raise RuntimeError(register+' write was not verified')
            if clearing_i and 'integral_exposure' in self.report:
                exposure = self.report['integral_exposure']
                if 'i0_verified_s' not in exposure:
                    exposure['i0_verified_s'] = float(self.clock())
                    exposure['duration_s'] = exposure['i0_verified_s']-exposure['started_s']
                    exposure['within_limit'] = exposure['duration_s'] <= exposure['maximum_s']
        self.check(recovery=recovery)
        if register == 'Torque_Limit':
            self.cap = value
        elif register == 'P_Coefficient':
            self.p = value
        elif register == 'I_Coefficient':
            self.i = value
        else:
            self.d = value
        return event

    def preflight(self, execution):
        if self.report['mode'] == 'i1':
            tail, target, target_raw = _p32_source(execution)
            commands = np.asarray([target])
            goals = [target_raw]
            self.report.update(source_kind='completed_restored_p32_service_result', source_tail_phase='candidate',
                               source_schema=execution['schema'], source_mode=execution['mode'])
        else:
            commands, measured, tail, target, target_raw = _source(execution)
            goals = [row['goal_raw'] for row in execution['policy_goal_readbacks']]
            self.report['source_kind'] = 'completed_fullwait_execution'
        self.target, self.target_raw = target, target_raw
        self.report.update(original_target=target.tolist(), original_target_raw=target_raw,
            source_tail_q=tail[-1].tolist(), source_execution_record=deepcopy(execution.get('execution_record')))
        self.check()
        runtime = self.owner.runtime
        if not runtime.stopped or runtime.pending is not None or runtime.executing:
            raise RuntimeError('fixed-target diagnostic requires stopped idle runtime')
        if self.owner.bridge.runtime is not runtime:
            raise RuntimeError('bridge does not belong to the admitted owner runtime')
        with self.owner.bus_lock:
            self.check()
            identity = self.identity()
            parameters = {key:self.read(key) for key in ('P_Coefficient','I_Coefficient','D_Coefficient','Torque_Limit','Max_Torque_Limit')}
            if parameters != dict(P_Coefficient=16,I_Coefficient=0,D_Coefficient=0,Torque_Limit=1000,Max_Torque_Limit=1000):
                raise RuntimeError('J3 baseline P16/I0/D0/Torque1000 not verified')
            if self.report['mode'] == 'i1':
                self.report['coefficient_layout'] = self.coefficient_layout()
            for index, command in enumerate(commands):
                if self.encoded(command) != goals[index]:
                    raise RuntimeError('current calibration does not reconstruct saved Goal cells')
        row = self.snapshot('admission')
        q = np.asarray(row['q_position'])
        if np.abs(q-tail[-1]).max() > .25 or any(row['moving'].values()):
            raise RuntimeError('current actual pose no longer matches the saved stationary tail')
        self.source_q = self.previous_q = q.copy()
        ramp = _ramp(q, target)
        self.geometry(q, ramp, stalled=True)
        # Audit both endpoint cross-sections and the entire possible J3 path.
        # The grid covers a wider +/-0.75 corridor than the live +/-0.25 limit.
        low, high = min(q[self.axis], target[self.axis])-.75, max(q[self.axis], target[self.axis])+.75
        grid = np.linspace(low, high, int(np.ceil((high-low)/.1))+1)
        for base in (q, target):
            for coordinate in grid:
                self.check()
                probe = base.copy(); probe[self.axis] = coordinate
                self.geometry(probe, _ramp(probe, target), stalled=True)
        # The corridor audit can be slow. Re-read the same actual cells and
        # controller state before allowing any write; never dispatch old q.
        fresh = self.snapshot('admission_after_geometry', expected_goal=row['goal_raw'])
        fresh_q = np.asarray(fresh['q_position'])
        if (np.abs(fresh_q-tail[-1]).max() > .25 or np.abs(fresh_q-q).max() > .1
                or any(fresh['moving'].values())):
            raise RuntimeError('current actual pose changed during the geometry precheck')
        with self.owner.bus_lock:
            self.check()
            self.identity()
            if self.read('Max_Torque_Limit') != 1000:
                raise RuntimeError('J3 maximum torque limit changed during the geometry precheck')
        self.baseline_verified = True
        self.report['admission'] = dict(identity=identity, parameters=parameters,
            current_q=q.tolist(), pre_dispatch_q=fresh_q.tolist(), corridor_j3=[low,high], corridor_grid_points=len(grid),
            single_axis_stall_checked=True, geometry_only_no_motor_writes=True)
        self.source_q = self.previous_q = fresh_q.copy()
        return _ramp(fresh_q, target)

    def transport(self, commands):
        stage = dict(commands=[], readbacks=[], model_history_committed=False)
        self.report['stages']['ramp'] = stage
        epoch = float(self.clock())
        stage['epoch_s'] = epoch
        previous = self.source_q.copy()
        for index, command in enumerate(commands):
            deadline = epoch+.1*index
            self.boundary(deadline)
            self.geometry(self.previous_q, commands[index:] if index == 0 else _ramp(self.previous_q,self.target))
            self.boundary(deadline)
            event = dict(index=index, command=command.tolist(), started_s=float(self.clock()), may_have_applied=True)
            stage['commands'].append(event)
            self.position_attempted = True
            self.check()
            actual = np.asarray(self.owner._write_q(command, previous), dtype=np.float32)
            event['returned_s'] = float(self.clock())
            if actual.shape != (6,) or not np.array_equal(actual, command):
                raise RuntimeError('existing transport changed the frozen diagnostic command')
            self.boundary(deadline)
            with self.owner.bus_lock:
                self.check()
                expected = self.encoded(command)
                goal = self.read_all('Goal_Position')
                event.update(expected_goal_raw=expected, goal_raw=goal, goal_register_verified=goal == expected)
                if goal != expected:
                    raise RuntimeError('diagnostic command Goal raw readback mismatch')
            self.boundary(deadline)
            for offset in (.05,.1):
                sample_deadline = deadline+offset
                self.boundary(sample_deadline)
                row = self.snapshot('ramp', expected_goal=expected, deadline=sample_deadline)
                stage['readbacks'].append(row)
            previous = actual.copy()
        if stage['commands'][-1]['goal_raw'] != self.target_raw:
            raise RuntimeError('ramp final Goal is not the saved original raw target')
        stage['completed'] = True

    def observe(self, name, *, epoch=None, require_stationary=False, integral=False, retention=False):
        epoch = float(self.clock()) if epoch is None else epoch
        budget = self.report['integral_max_seconds'] if integral else (.3 if retention else WINDOW_S)
        maximum_samples = int(round(budget/PERIOD_S))
        early_arrival_enabled = integral and budget == 10.
        stage = dict(epoch_s=epoch, parameters=dict(P=self.p,I=self.i,D=self.d,Torque_Limit=self.cap),
                     original_target=self.target.tolist(), original_target_raw=self.target_raw,
                     readbacks=[], completed=False, fixed_observation_window=not integral, budget_s=budget,
                     sample_count=0, maximum_samples=maximum_samples, early_arrival=False,
                     early_arrival_enabled=early_arrival_enabled, termination_reason='incomplete')
        if integral:
            stage.update(immediate_first_sample=True, max_integral_exposure_s=budget,
                         scheduled_last_sample_s=round((maximum_samples-1)*PERIOD_S, 10),
                         full_exposure_window_observed=False)
            if budget == WINDOW_S:
                stage['full_two_second_observation'] = False
        self.report['stages'][name] = stage
        for index in (range(maximum_samples) if integral else range(1,maximum_samples+1)):
            deadline = epoch+index*PERIOD_S
            # Retention belongs to cleanup: STOP remains latched while I0,
            # D0 and cap200 have already been proved. Its original-goal and
            # geometry/motion guards remain active on every actual sample.
            self.boundary(deadline, recovery=retention)
            stage['readbacks'].append(self.snapshot(name, expected_goal=self.target_raw,
                                                   deadline=deadline, recovery=retention))
            rows = stage['readbacks']
            stage.update(sample_count=len(rows),
                actual_observation_span_s=rows[-1]['readback_time_s']-rows[0]['readback_time_s'],
                last_sample_since_epoch_s=rows[-1]['readback_time_s']-epoch)
            if early_arrival_enabled and len(rows) >= 3:
                tail = rows[-3:]
                current_arrival = evaluate_position_arrival(self.target,
                    [row['q_position'] for row in tail], [row['readback_time_s']-epoch for row in tail],
                    position_tolerance=.2, step_tolerance=.1, quiet_samples=3)
                if current_arrival['reached']:
                    stage.update(early_arrival=len(rows) < maximum_samples,
                                 termination_reason='original_target_arrival')
                    break
        if stage['termination_reason'] == 'incomplete':
            stage['termination_reason'] = 'sample_window_complete'
        positions = [row['q_position'] for row in stage['readbacks']]
        times = [row['readback_time_s']-epoch for row in stage['readbacks']]
        arrival = evaluate_position_arrival(self.target, positions, times,
            position_tolerance=.2, step_tolerance=.1, quiet_samples=3)
        arrival.update(source='fixed_original_goal_pre_cleanup_actual_readbacks',
                       position_unit='lerobot_calibrated_position', excludes_cleanup_target=True)
        stage.update(completed=True, original_target_arrival=arrival, original_target_reached=arrival['reached'])
        stage.update(actual_observation_span_s=times[-1]-times[0], last_sample_since_epoch_s=times[-1])
        if require_stationary and (arrival['tail_stable'] is not True
                or any(any(row['moving'].values()) for row in stage['readbacks'][-3:])):
            raise RuntimeError(name+' did not reach a stationary baseline before changing a parameter')
        return stage

    def halt(self):
        self.owner.stop_event.set()
        self.owner._v21_hold_cache = None  # Force a fresh raw Present -> Goal freeze.
        self.owner._lease_check()
        value = self.owner._halt()
        self.report['cleanup']['hold'] = deepcopy(value)
        attempts = value.get('attempts',[]) if isinstance(value,dict) else []
        last = attempts[-1] if attempts else {}
        goal = _map(last.get('goal_readback_raw'), 'cleanup HOLD Goal')
        if (last.get('goal_register_verified') is not True or last.get('pose_validated') is not True
                or goal != _map(last.get('requested_goal_raw'), 'cleanup frozen Present')):
            raise RuntimeError('existing HOLD did not prove exact fresh Present-to-Goal freeze')
        self.report['cleanup']['hold_verified'] = True
        self.report['stopped'] = True
        return goal

    def quiet(self, name, goal):
        epoch = float(self.clock())
        rows = []
        self.report['cleanup'][name] = rows
        previous = None
        for index in range(1,4):
            deadline = epoch+index*PERIOD_S
            self.boundary(deadline, recovery=True)
            row = self.snapshot(name, expected_goal=goal, recovery=True, deadline=deadline)
            rows.append(row)
            q = np.asarray(row['q_position'])
            with self.owner.bus_lock:
                self.check(recovery=True)
                goal_q = self.normalized(goal)
            if (any(row['moving'].values()) or np.abs(q-goal_q).max() > .2
                    or previous is not None and np.abs(q-previous).max() > .1
                    or abs(row['present_load_decoded']) > 32):
                raise RuntimeError('cleanup lacks stationary low-load frozen-goal proof')
            previous = q
        return True

    def cleanup(self):
        result = self.report['cleanup']
        if not (self.position_attempted or self.cap_attempted or self.gain_attempted or self.integral_attempted):
            result.update(restored=False, no_mutation_attempted=True,
                          baseline_verified=self.baseline_verified)
            return
        self.owner.stop_event.set()
        if self.report['mode'] == 'i1':
            self.cleanup_integral()
            return
        try:
            # Position writes alone can fail too. In every mutated run first
            # establish the low cap, including after a lost acknowledgement.
            self.write_register('Torque_Limit',200,recovery=True)
            result['low_cap_verified'] = True
            self.write_register('P_Coefficient',16,recovery=True)
            with self.owner.bus_lock:
                self.check(recovery=True)
                self.identity()
                result['integral_zero_verified'] = self.read('I_Coefficient') == 0
                result['gain_restored'] = self.read('P_Coefficient') == 16 and self.read('D_Coefficient') == 0
                if not result['integral_zero_verified'] or not result['gain_restored']:
                    raise RuntimeError('gain/I0 restoration not verified; torque increase forbidden')
            goal = self.halt()
            result['pre_restore_quiet_verified'] = self.quiet('pre_restore_readbacks', goal)
            self.write_register('Torque_Limit',1000,recovery=True)
            result['low_cap_verified'] = False
            result['torque_restored'] = True
            result['post_restore_quiet_verified'] = self.quiet('post_restore_readbacks', goal)
            result['restored'] = True
        except BaseException as error:
            result['errors'].append(f'{type(error).__name__}: {error}')
            result['restored'] = False
            if self.baseline_verified:
                try:
                    self.write_register('Torque_Limit',200,recovery=True)
                    result['low_cap_verified'] = True
                    result['torque_restored'] = False
                    # A first low-cap acknowledgement can be lost. Once the
                    # fallback cap is proved, still try to remove P32; never
                    # increase torque or turn this failed cleanup into success.
                    try:
                        self.write_register('P_Coefficient',16,recovery=True)
                        result['gain_restored'] = True
                        result['integral_zero_verified'] = True
                    except BaseException as gain_error:
                        result['gain_restored'] = False
                        result['errors'].append(f'P16 restore unverified: {type(gain_error).__name__}: {gain_error}')
                except BaseException as cap_error:
                    result['low_cap_verified'] = False
                    result['errors'].append(f'low cap unverified: {type(cap_error).__name__}: {cap_error}')
            try:
                self.halt()
            except BaseException as hold_error:
                result['errors'].append(f'HOLD failed: {type(hold_error).__name__}: {hold_error}')

    def cleanup_integral(self):
        """I0 is independent of cap/P failures; keep damping if I is unknown."""
        result = self.report['cleanup']

        def attempt(label, action):
            try:
                action()
                return True
            except BaseException as error:
                result['errors'].append(f'{label}: {type(error).__name__}: {error}')
                return False

        def restore_low():
            # Do this FIRST, even if cap or another coefficient cannot be read
            # or restored. Setting I0 needs only the pinned owner/servo/address.
            result['integral_zero_verified'] = attempt('I0 restore',
                lambda: self.write_register('I_Coefficient', 0, recovery=True))
            result['low_cap_verified'] = attempt('low cap restore',
                lambda: self.write_register('Torque_Limit', 200, recovery=True))
            result['torque_restored'] = False
            result['gain_restored'] = False
            result['damping_restored'] = False
            if result['integral_zero_verified'] and result['low_cap_verified']:
                # This guard re-reads I0/P16/cap200 before removing D32.
                result['damping_restored'] = attempt('D0 restore',
                    lambda: self.write_register('D_Coefficient', 0, recovery=True))
                if result['damping_restored']:
                    def verify():
                        with self.owner.bus_lock:
                            self.check(recovery=True)
                            values = {key: self.read(key) for key in
                                      ('P_Coefficient', 'I_Coefficient', 'D_Coefficient', 'Torque_Limit')}
                            if values != dict(P_Coefficient=16, I_Coefficient=0, D_Coefficient=0, Torque_Limit=200):
                                raise RuntimeError('P16/I0/D0/cap200 recovery is unverified')
                    result['gain_restored'] = attempt('coefficient verification', verify)

        restore_low()
        try:
            if (result['errors'] or not result['integral_zero_verified']
                    or not result['gain_restored'] or not result['low_cap_verified']):
                raise RuntimeError('I0/D0/P16/cap200 recovery incomplete; high torque forbidden')
            candidate = self.report['stages'].get('candidate', {})
            if (self.report['integral_max_seconds'] == 10.
                    and candidate.get('completed') is True and candidate.get('original_target_reached') is True):
                retained = self.observe('post_integral_retention', retention=True)
                arrival = retained['original_target_arrival']
                rows = retained['readbacks']
                whole_window = evaluate_position_arrival(self.target,
                    [row['q_position'] for row in rows],
                    [row['readback_time_s']-retained['epoch_s'] for row in rows],
                    position_tolerance=.2, step_tolerance=.1, quiet_samples=6)
                arrival.update(source='actual_original_goal_after_I0_D0_cap200_before_HOLD',
                               retained=whole_window['reached'], no_goal_write=True,
                               retention_samples=6,
                               retention_max_abs_residual=whole_window['tail_max_abs_residual'],
                               retention_max_abs_step=whole_window['tail_max_abs_step'])
                self.report['original_goal_retained_arrival'] = arrival
                self.report['original_goal_retained'] = arrival['retained']
                result['original_goal_retained_arrival'] = arrival
            goal = self.halt()
            result['pre_restore_quiet_verified'] = self.quiet('pre_restore_readbacks', goal)
            self.write_register('Torque_Limit', 1000, recovery=True)
            result['torque_restored'] = True
            result['low_cap_verified'] = False
            result['post_restore_quiet_verified'] = self.quiet('post_restore_readbacks', goal)
            result['restored'] = True
        except BaseException as error:
            result['errors'].append(f'{type(error).__name__}: {error}')
            result['restored'] = False
            # Retry I0 independently even when cap restoration itself failed.
            # A fallback never turns this failed recovery into a success.
            restore_low()
            attempt('HOLD', self.halt)


def run_same_goal_integral(owner=None, source_execution=None, *, enabled=False, mode='p32',
                           integral_max_seconds=2, cancelled=None, clock=time.monotonic, wait=None):
    """Run an explicit bounded P32 or damped I1 trial; return actual evidence.

    ``completed`` means the observations finished, never that the original
    target was reached. Always inspect ``cleanup.restored`` before releasing
    the service's persistent motion seal. No automatic second attempt exists.
    """
    report = dict(schema='v21-same-goal-gain-diagnostic-v1', mode=mode, enabled=enabled,
        integral_max_seconds=integral_max_seconds,
        motor=MOTOR, completed=False, stopped=False, stages={}, readbacks=[], parameter_writes=[],
        cleanup=dict(restored=False,gain_restored=False,integral_zero_verified=False,hold_verified=False,
                     pre_restore_quiet_verified=False,torque_restored=False,post_restore_quiet_verified=False,errors=[]),
        model_history_committed=False, model_inference_calls=0, original_goal_changed_during_trial=False,
        only_p_gain_changes=mode == 'p32', integral_mode_enabled=enabled is True and mode == 'i1',
        physical_task_success_verified=False)
    if enabled is not True or type(enabled) is not bool:
        raise SameGoalIntegralError('fixed-target diagnostic disabled; no IO', report)
    if mode not in ('p32', 'i1'):
        raise SameGoalIntegralError('only explicit p32 or i1 mode is implemented; unsupported mode; no IO',report)
    if (isinstance(integral_max_seconds, (bool, np.bool_)) or not isinstance(integral_max_seconds, Real)
            or not np.isfinite(integral_max_seconds) or integral_max_seconds not in (2, 10)
            or mode == 'p32' and integral_max_seconds != 2):
        raise SameGoalIntegralError('integral_max_seconds must be numeric 2 or 10 for i1, and 2 for p32; no IO', report)
    report['integral_max_seconds'] = float(integral_max_seconds)
    run = None
    error = None
    try:
        run = _Run(owner,report,cancelled or owner.stop_event.is_set,clock,wait or time.sleep)
        report['started_s'] = float(clock())
        ramp = run.preflight(source_execution)
        run.transport(ramp)
        run.observe('baseline',require_stationary=True)
        run.write_register('Torque_Limit',200)
        run.observe('cap_baseline',require_stationary=True)
        if mode == 'i1':
            run.write_register('D_Coefficient', 32)
            run.observe('damping_baseline', require_stationary=True)
        # The configured exposure clock includes the write and its readback.
        epoch = float(clock())
        if mode == 'i1':
            report['integral_exposure'] = dict(started_s=epoch, maximum_s=float(integral_max_seconds),
                clock_origin='before_I1_write_and_verification', within_limit=False)
            run.write_register('I_Coefficient', 1)
        else:
            run.write_register('P_Coefficient',32)
        candidate = run.observe('candidate',epoch=epoch,integral=mode == 'i1')
        report.update(completed=True,original_target_arrival=candidate['original_target_arrival'],
                      original_target_reached=candidate['original_target_reached'])
    except BaseException as caught:
        error = caught
        report['error'] = f'{type(caught).__name__}: {caught}'
    finally:
        if run is not None:
            try:
                run.cleanup()
            except BaseException as cleanup_error:
                report['cleanup']['restored'] = False
                report['cleanup']['errors'].append(f'{type(cleanup_error).__name__}: {cleanup_error}')
            report['finished_s'] = float(clock())
    exposure = report.get('integral_exposure')
    if exposure is not None and not exposure['within_limit'] and error is None:
        error = RuntimeError(f'I1 exposure exceeded {integral_max_seconds:g} seconds or I0 clearing is unverified')
        report['error'] = str(error)
    if error is not None or report['cleanup']['restored'] is not True:
        raise SameGoalIntegralError(report.get('error') or 'fixed-target parameter restoration incomplete',report) from error
    return _plain(report)
