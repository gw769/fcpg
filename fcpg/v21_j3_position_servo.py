"""Finish an existing five-command J3 target without issuing another Goal.

The caller remains inside its active bridge transaction. This module creates
no owner, touches no model state, and adds no policy commands or y labels.
"""
import time

import numpy as np

from .v21_position_arrival import evaluate_position_arrival
from .v21_same_goal_integral import (
    AXIS, LATENESS_S, MAX_STEP, MOTOR, ORDER, PERIOD_S,
    _Run, _array, _map, _plain, _ramp,
)


SCHEMA = 'v21-j3-position-servo-v1'
FOLLOW_SCHEMA = 'v21-position-servo-completion-follow-v1'
SOURCE_FIELDS = ('completed', 'action_source', 'source_time_s', 'commands_position',
    'measured_q_path', 'readback_times_s', 'normal_follow', 'policy_goal_readbacks',
    'policy_commands_attempted', 'policy_commands_returned', 'policy_goal_reads_attempted',
    'policy_goal_reads_returned', 'policy_goal_registers_verified')


class J3PositionServoError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = _plain(report)


def _execution_source(execution):
    """Validate the unfinished live attempt as such; never make a saved trial."""
    if not isinstance(execution, dict) or execution.get('completed') is not False:
        raise ValueError('position servo requires the still-executing five-command attempt')
    if execution.get('action_source') not in ('v21_actor', 'v21_lewm_cem'):
        raise ValueError('position servo cannot finish a diagnostic or substituted action source')
    for field in SOURCE_FIELDS[-5:]:
        if type(execution.get(field)) is not int or execution[field] != 5:
            raise ValueError('position servo requires five real commands and verified raw Goal reads: '+field)
    commands = _array(execution.get('commands_position'), (5, 6), 'live commands')
    measured = _array(execution.get('measured_q_path'), (6, 6), 'live actual path')
    times = _array(execution.get('readback_times_s'), (6,), 'live actual clocks')
    epoch = _array([execution.get('source_time_s')], (1,), 'live source epoch')[0]
    if epoch != times[0] or not (np.diff(times) > 0).all():
        raise ValueError('live actual source clocks are inconsistent')
    previous = np.vstack((measured[0], commands[:-1]))
    if (np.abs(commands-previous) > MAX_STEP+1e-5).any():
        raise ValueError('live commands exceed the original MAX_STEP')
    goals = execution.get('policy_goal_readbacks')
    if not isinstance(goals, list) or len(goals) != 5:
        raise ValueError('live attempt lacks five raw Goal receipts')
    for index, row in enumerate(goals):
        if (row.get('command_index') != index or row.get('goal_register_verified') is not True
                or row.get('read_attempted') is not True or row.get('read_returned') is not True
                or row.get('model_feedback_label') is not False):
            raise ValueError('live Goal receipt was not actually read and verified')
        if _map(row.get('expected_goal_raw'), 'requested live Goal') != _map(row.get('goal_raw'), 'read live Goal'):
            raise ValueError('live raw Goal receipt differs from the requested cells')
        requested = row.get('requested_goal_position')
        if not isinstance(requested, dict) or set(requested) != set(ORDER):
            raise ValueError('live Goal receipt lacks the original command')
        if not np.array_equal(_array([requested[name] for name in ORDER], (6,), 'requested live command'), commands[index]):
            raise ValueError('live Goal receipt does not bind the original command')
    follow = execution.get('normal_follow') or {}
    if (follow.get('budget_s') != 2. or follow.get('fixed_observation_window') is not True
            or follow.get('reason') != 'fixed_window_complete'):
        raise ValueError('position servo requires the completed original two-second follow')
    positions = _array(follow.get('positions'), (40, 6), 'live follow actual positions')
    follow_times = _array(follow.get('times_s'), (40,), 'live follow actual clocks')
    if (not (np.diff(follow_times) > 0).all() or follow_times[0] <= times[-1]
            or np.abs(follow_times-(times[-1]+PERIOD_S*np.arange(1, 41))).max() > LATENESS_S
            or not 1.97 <= follow_times[-1]-times[-1] <= 2.03):
        raise ValueError('live follow did not retain all forty actual 20Hz samples')
    target = commands[-1].copy()
    if not np.array_equal(_array((execution.get('original_target_arrival') or {}).get('original_target'),
                                (6,), 'live frozen original target'), target):
        raise ValueError('live original target was replaced')
    return commands, positions, follow_times, target, _map(goals[-1]['goal_raw'], 'frozen live Goal'), float(epoch)


class _PositionRun(_Run):
    def __init__(self, owner, execution, report, cancelled, clock, wait):
        super().__init__(owner, report, cancelled, clock, wait)
        self.execution = execution
        self.bridge = owner.bridge
        self.runtime = owner.runtime
        self.pending = self.runtime.pending
        self.live_attempt = self.bridge.last_attempt
        self.admitted = False
        self.source_epoch = None

    def check(self, *, recovery=False):
        super().check(recovery=recovery)
        if not recovery and (self.owner.bridge is not self.bridge or self.owner.runtime is not self.runtime
                or self.bridge.runtime is not self.runtime or self.bridge.last_attempt is not self.live_attempt
                or self.runtime.stopped is not False or self.runtime.executing is not True
                or self.runtime.pending is not self.pending or self.pending is None
                or not self.bridge.lock.locked()):
            raise RuntimeError('position servo requires the same active locked bridge transaction')

    def preflight(self):
        self.check()
        commands, positions, times, target, target_raw, epoch = _execution_source(self.execution)
        if any(self.live_attempt.get(key) != self.execution.get(key) for key in SOURCE_FIELDS):
            raise RuntimeError('position servo source is not the bridge current execution evidence')
        self.admitted = True
        self.target, self.target_raw, self.source_epoch = target, target_raw, epoch
        self.report.update(original_target=target.tolist(), original_target_raw=target_raw,
                           source_time_s=epoch, source_follow_samples=40, source_follow_budget_s=2.)
        with self.owner.bus_lock:
            self.check()
            identity = self.identity()
            parameters = {key: self.read(key) for key in
                          ('P_Coefficient', 'I_Coefficient', 'D_Coefficient', 'Torque_Limit', 'Max_Torque_Limit')}
            if parameters != dict(P_Coefficient=16, I_Coefficient=0, D_Coefficient=0, Torque_Limit=1000, Max_Torque_Limit=1000):
                raise RuntimeError('position servo baseline P16/I0/D0/1000 is unverified')
            self.report['coefficient_layout'] = self.coefficient_layout()
            for command, receipt in zip(commands, self.execution['policy_goal_readbacks']):
                if self.encoded(command) != receipt['goal_raw']:
                    raise RuntimeError('live calibration does not reconstruct original Goal cells')
        row = self.snapshot('admission', expected_goal=target_raw)
        q = np.asarray(row['q_position'])
        if np.abs(q-positions[-1]).max() > .25:
            raise RuntimeError('fresh actual position changed from the live follow tail')
        self.geometry(q, np.repeat(q[None], 5, axis=0))
        self.baseline_verified = True
        self.report['original_goal_preserved'] = True
        self.report['admission'] = dict(identity=identity, parameters=parameters, current_q=q.tolist(),
            actual_goal_raw=row['goal_raw'], source='current_bridge_attempt_and_fresh_raw_present')
        actual = [*positions[-3:].tolist(), row['q_position']]
        actual_times = [*times[-3:].tolist(), row['readback_time_s']]
        self.publish_completion(actual, actual_times, source='live_follow_tail_and_fresh_admission')
        if self.report['original_target_reached']:
            return 'already_reached'
        error = q-target
        if np.abs(np.delete(error, AXIS)).max() > .2:
            return 'non_j3_residual_outside_position_tolerance'
        if not .2 < abs(error[AXIS]) <= 3.:
            return 'j3_residual_outside_finishing_range'
        if (np.abs(np.diff(np.asarray(actual), axis=0)).max() > .1 or any(row['moving'].values())
                or np.abs(np.delete(positions[-3:]-target, AXIS, axis=1)).max() > .2):
            return 'actual_follow_tail_not_stationary_at_other_joint_targets'
        self.source_q = self.previous_q = q.copy()
        self.geometry(q, _ramp(q, target), stalled=True)
        low, high = min(q[AXIS], target[AXIS])-.75, max(q[AXIS], target[AXIS])+.75
        grid = np.linspace(low, high, int(np.ceil((high-low)/.1))+1)
        for base in (q, target):
            for coordinate in grid:
                self.check()
                probe = base.copy(); probe[AXIS] = coordinate
                self.geometry(probe, _ramp(probe, target), stalled=True)
        fresh = self.snapshot('admission_after_geometry', expected_goal=target_raw)
        fresh_q = np.asarray(fresh['q_position'])
        if np.abs(fresh_q-q).max() > .1 or any(fresh['moving'].values()):
            raise RuntimeError('position servo actual baseline changed during the corridor precheck')
        with self.owner.bus_lock:
            self.check()
            self.identity()
            if self.read('Max_Torque_Limit') != 1000:
                raise RuntimeError('J3 maximum torque limit changed during the corridor precheck')
        self.source_q = self.previous_q = fresh_q.copy()
        self.report['admission'].update(corridor_j3=[low, high], corridor_grid_points=len(grid),
                                      single_axis_stall_checked=True, pre_dispatch_q=fresh_q.tolist())
        return None

    def publish_completion(self, positions, times, *, source):
        positions = _array(positions, (len(positions), 6), 'completion actual positions')
        times = _array(times, (len(positions),), 'completion actual clocks')
        arrival = evaluate_position_arrival(self.target, positions, times-self.source_epoch,
            position_tolerance=.2, step_tolerance=.1, quiet_samples=3)
        arrival.update(source=source, source_time_s=self.source_epoch, excludes_post_hold_readbacks=True,
                       position_unit='lerobot_calibrated_position', control_influence=False)
        count = int(getattr(self.owner.io, 'settle_samples', 3))
        position_tolerance = float(getattr(self.owner.io, 'settle_position_tolerance', 3.))
        step_tolerance = float(getattr(self.owner.io, 'settle_step_tolerance', .25))
        eligible = (len(positions) >= count+1 and np.abs(positions[-count:]-self.target).max() <= position_tolerance
                    and np.abs(np.diff(positions, axis=0)[-count:]).max() <= step_tolerance)
        self.report.update(original_target_arrival=arrival, original_target_reached=arrival['reached'],
            completion_follow=dict(schema=FOLLOW_SCHEMA, positions=positions.tolist(), times_s=times.tolist(),
                source_time_s=self.source_epoch, source=source, actual_span_s=float(times[-1]-times[0]),
                hold_eligible=bool(eligible), goal_reached=arrival['reached'], original_target=self.target.tolist(),
                baseline_restored=True, model_feedback_label=False))

    def restore_without_changing_goal(self):
        result = self.report['cleanup']
        # No STOP here: this success branch belongs to the original transaction.
        self.check()
        self.write_register('I_Coefficient', 0, recovery=True)
        self.check()
        result['integral_zero_verified'] = True
        self.write_register('Torque_Limit', 200, recovery=True)
        self.check()
        result['low_cap_verified'] = True
        self.write_register('D_Coefficient', 0, recovery=True)
        self.check()
        result['gain_restored'] = result['damping_restored'] = True
        retained = self.observe_original_goal_quiet('original_goal_retention', count=6)
        rows = retained['readbacks']
        retained_arrival = evaluate_position_arrival(self.target, [row['q_position'] for row in rows],
            [row['readback_time_s']-retained['epoch_s'] for row in rows],
            position_tolerance=.2, step_tolerance=.1, quiet_samples=6)
        self.report['original_goal_retained_arrival'] = retained_arrival
        self.report['original_goal_retained'] = retained_arrival['reached']
        if (not retained_arrival['reached'] or any(any(row['moving'].values()) for row in rows)
                or any(abs(row['present_load_decoded']) > 32 for row in rows)):
            raise RuntimeError('original Goal did not stay reached and quiet after I0/D0 restoration')
        self.check()
        self.write_register('Torque_Limit', 1000)
        result['low_cap_verified'] = False
        result['torque_restored'] = True
        self.observe_original_goal_quiet()
        final_rows = self.report['stages']['restored_original_goal']['readbacks']
        all_rows = rows+final_rows
        self.publish_completion([row['q_position'] for row in all_rows],
            [row['readback_time_s'] for row in all_rows], source='original_goal_after_P16_I0_D0_Torque1000_restore')
        if not self.report['original_target_reached']:
            raise RuntimeError('original Goal is not reached after restoring full baseline torque')
        self.check()
        result.update(restored=True, original_goal_preserved=True, hold_performed=False,
                      pre_restore_quiet_verified=True, post_restore_quiet_verified=True)

    def observe_original_goal_quiet(self, name='restored_original_goal', *, count=3):
        epoch = float(self.clock())
        stage = dict(epoch_s=epoch, readbacks=[], sample_count=0, completed=False, actual_only=True)
        self.report['stages'][name] = stage
        for index in range(1, count+1):
            deadline = epoch+index*PERIOD_S
            self.boundary(deadline)
            row = self.snapshot(name, expected_goal=self.target_raw, deadline=deadline)
            stage['readbacks'].append(row)
            stage['sample_count'] += 1
            if any(row['moving'].values()) or abs(row['present_load_decoded']) > 32:
                raise RuntimeError('restored original Goal lacks stationary low-output confirmation')
        stage.update(completed=True, actual_span_s=stage['readbacks'][-1]['readback_time_s']-stage['readbacks'][0]['readback_time_s'])
        return stage

    def fail_closed(self):
        result = self.report['cleanup']
        self.owner.stop_event.set()
        result.update(restored=False, hold_performed=False)
        def attempt(label, action):
            try:
                action()
                return True
            except BaseException as error:
                result['errors'].append(f'{label}: {type(error).__name__}: {error}')
                return False
        if self.baseline_verified and (self.cap_attempted or self.gain_attempted or self.integral_attempted):
            # Clear I independently even if P/D/cap recovery is unknown. Failed
            # execution stays at cap200; removing D32 needs fresh proven I0.
            result['integral_zero_verified'] = attempt('I0', lambda: self.write_register('I_Coefficient', 0, recovery=True))
            result['low_cap_verified'] = attempt('cap200', lambda: self.write_register('Torque_Limit', 200, recovery=True))
            result['torque_restored'] = False
            if result['integral_zero_verified'] and result['low_cap_verified']:
                result['damping_restored'] = attempt('D0', lambda: self.write_register('D_Coefficient', 0, recovery=True))
                result['gain_restored'] = result['damping_restored']
        if self.admitted:
            self.report['original_goal_preserved'] = False
            result['hold_performed'] = attempt('existing HOLD', self.halt)
        self.report.pop('completion_follow', None)


def finish_original_j3_goal(owner=None, execution=None, *, enabled=False, cancelled=None,
                            clock=time.monotonic, wait=None):
    """Finish only an isolated actual J3 residual on the existing fifth Goal."""
    report = dict(schema=SCHEMA, enabled=enabled, mode='i1', integral_max_seconds=10.,
        completed=False, applied=False, skipped=False, stopped=False, original_goal_preserved=False,
        motor=MOTOR, stages={}, readbacks=[], parameter_writes=[],
        cleanup=dict(restored=False, errors=[]), model_feedback_label=False, model_history_committed=False,
        model_inference_calls=0, policy_commands_added=0, physical_task_success_verified=False)
    if enabled is not True or type(enabled) is not bool:
        raise J3PositionServoError('position servo disabled; no IO', report)
    run = None
    try:
        run = _PositionRun(owner, execution, report, cancelled or owner.stop_event.is_set, clock, wait or time.sleep)
        report['started_s'] = float(clock())
        reason = run.preflight()
        if reason is not None:
            report.update(completed=True, skipped=True, reason=reason)
            report['cleanup'].update(restored=True, baseline_verified=True, no_mutation_attempted=True, hold_performed=False)
            report['finished_s'] = float(clock())
            return _plain(report)
        report['applied'] = True
        run.write_register('Torque_Limit', 200)
        run.observe('cap_baseline', require_stationary=True)
        run.write_register('D_Coefficient', 32)
        run.observe('damping_baseline', require_stationary=True)
        epoch = float(clock())
        report['integral_exposure'] = dict(started_s=epoch, maximum_s=10.,
            clock_origin='before_I1_write_and_verification', within_limit=False)
        run.write_register('I_Coefficient', 1)
        candidate = run.observe('candidate', epoch=epoch, integral=True)
        if not candidate['original_target_reached']:
            raise RuntimeError('original J3 target not reached within the ten-second finishing window')
        run.restore_without_changing_goal()
        if report['integral_exposure']['within_limit'] is not True:
            raise RuntimeError('position servo I1 exposure exceeded ten seconds or I0 is unverified')
        report.update(completed=True, reason='original_goal_reached_after_baseline_restoration')
        report['finished_s'] = float(clock())
        return _plain(report)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        if run is not None:
            try:
                run.fail_closed()
            except BaseException as cleanup_error:
                report['cleanup']['restored'] = False
                report['cleanup']['errors'].append(f'{type(cleanup_error).__name__}: {cleanup_error}')
            report['finished_s'] = float(clock())
        raise J3PositionServoError(str(error), report) from error
