"""One exact five-command V21 transaction using an existing leased IO owner.

No serial/camera construction here. Actual write times and readback times are
recorded independently; no hidden settle, resampling or substitute commands.
"""
from dataclasses import replace
from copy import deepcopy
import threading
import time

import numpy as np
import torch

from .v21_models import CommandTrace
from .v21_column_dino_transport import CEMActualResponseError, require_actual_cem_response


def completion_hold_record(value, started, returned):
    """JSON-safe evidence from the existing hold callback, never a new command."""
    result = dict(started_s=float(started), returned_s=float(returned),
                  held_position=None, goal_register_verified=None,
                  verification_scope='Goal_Position register, not settled position')
    if not isinstance(value, dict):
        result['record_available'] = False
        return result
    result['record_available'] = True
    result['reused_existing_hold']=bool(value.get('reused_existing_hold',False))
    result['goal_register_writes']=value.get('motor_register_writes')
    for key in ('reason','normal_validation_error','actuator_events','actual_readback'):
        if key in value:result[key]=deepcopy(value[key])
    try:
        last = value['attempts'][-1]
        held = np.asarray(last['held_state'], dtype=np.float64)
        if held.shape != (6,) or not np.isfinite(held).all():
            raise ValueError('invalid held-state shape or values')
        result['held_position'] = held.tolist()
        result['goal_register_verified'] = (value.get('event') == 'safe_hold_verified'
                                           and not last['mismatched_joints'])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        result['record_error'] = str(exc)
    return result


def validate_arm_finishing_restoration(report):
    """Check the arm helper's actual per-motor restoration evidence."""
    from .v21_hardware_state import ORDER

    fields=('P_Coefficient','I_Coefficient','D_Coefficient','Torque_Limit','Max_Torque_Limit')
    baseline=report.get('baseline_parameters_by_motor')
    final=report.get('final_parameters_by_motor')
    for values in (baseline,final):
        if not isinstance(values,dict) or set(values)!=set(ORDER):
            raise RuntimeError('arm finishing lacks complete per-motor parameter readbacks')
        for motor in ORDER:
            row=values[motor]
            if (not isinstance(row,dict) or set(row)!=set(fields)
                    or any(type(row[key]) is not int or row[key]<0 for key in fields)):
                raise RuntimeError('arm finishing has invalid per-motor parameter readbacks')
    if baseline!=final or report.get('final_equals_baseline') is not True:
        raise RuntimeError('arm finishing changed an original motor baseline')
    touched=report.get('touched_motors')
    allowed_motors=('elbow_flex','wrist_flex','shoulder_lift')
    if (not isinstance(touched,list) or any(type(motor) is not str for motor in touched)
            or len(touched)!=len(set(touched))
            or not set(touched)<=set(allowed_motors)
            or touched!=[motor for motor in allowed_motors if motor in touched]
            or bool(touched) is not report['applied']):
        raise RuntimeError('arm finishing has invalid actually-touched motor evidence')
    restored=report['cleanup'].get('parameters_restored_by_motor')
    if (not isinstance(restored,dict) or set(restored)!=set(touched)
            or any(value is not True for value in restored.values())):
        raise RuntimeError('arm finishing lacks restoration proof for every touched motor')
    writes=report.get('parameter_writes')
    allowed={'Torque_Limit':(200,300,1000),'D_Coefficient':(0,32),'I_Coefficient':(0,1)}
    if (not isinstance(writes,list)
            or any(not isinstance(event,dict)
                or event.get('motor') not in touched
                or event.get('register') not in allowed
                or type(event.get('value')) is not int
                or event['value'] not in allowed[event['register']]
                or event.get('may_have_applied') is not True
                or event.get('verified') is not True for event in writes)
            or list(dict.fromkeys(event['motor'] for event in writes))!=touched):
        raise RuntimeError('arm finishing restoration does not match actual parameter writes')
    for motor in touched:
        if baseline[motor]!=dict(zip(fields,(16,0,0,1000,1000))):
            raise RuntimeError('arm finishing tuned an unverified motor baseline')
        axis_report=(report.get('axes',{}).get(motor,{}) or {})
        control=axis_report.get('integral_control')
        pulse_count=1
        if control is not None:
            legacy_control=dict(mode='stationary_release_pulses_v1',
                    maximum_pulses=8,maximum_pulse_s=2.,maximum_cumulative_s=10.,
                    physical_speed_bound_verified=False)
            progress_control=dict(mode='stationary_release_pulses_v2',
                    maximum_pulses=8,initial_maximum_pulse_s=2.,maximum_pulse_s=10.,
                    maximum_cumulative_s=10.,physical_speed_bound_verified=False)
            slow_control=dict(mode='stationary_release_pulses_v3',
                    maximum_pulses=16,maximum_pulse_s=20.,maximum_cumulative_s=20.,
                    physical_speed_bound_verified=False)
            if motor!='elbow_flex' or control not in (legacy_control,progress_control,slow_control):
                raise RuntimeError('arm finishing has an invalid integral pulse control')
            exposure=axis_report.get('integral_exposure') or {}
            pulses=exposure.get('pulses')
            if not isinstance(pulses,list) or not 1<=len(pulses)<=control['maximum_pulses']:
                raise RuntimeError('arm finishing lacks bounded integral pulses')
            pulse_count=len(pulses)
            durations=[]
            previous_end=None
            for pulse_index,pulse in enumerate(pulses):
                if not isinstance(pulse,dict):
                    raise RuntimeError('arm finishing has an invalid integral pulse')
                remaining=control['maximum_cumulative_s']-sum(durations)
                expected_maximum=remaining if control==slow_control else min(2.,remaining)
                if control in (progress_control,slow_control):
                    progress=pulse.get('preceding_settled_progress')
                    if progress is not None:
                        previous=pulses[pulse_index-1] if pulse_index else {}
                        expected_candidate='candidate' if pulse_index==1 else f'release_candidate_{pulse_index}'
                        expected_settling='integral_off_settling' if pulse_index==1 else f'release_settling_{pulse_index}'
                        stages=axis_report.get('stages',{})
                        candidate=stages.get(expected_candidate,{})
                        settled=stages.get(expected_settling,{})
                        rows=settled.get('readbacks',[])
                        target=axis_report.get('original_target')
                        if (not isinstance(progress,dict) or pulse_index==0
                                or pulse.get('stage')!=f'release_candidate_{pulse_index+1}'
                                or pulse.get('budget_basis')!='verified_same_side_settled_release_progress'
                                or progress.get('candidate_stage')!=expected_candidate
                                or progress.get('settling_stage')!=expected_settling
                                or candidate.get('stop_reason')!='first_actual_motion_after_stationary_integral'
                                or progress.get('integral_zero_verified_s')!=previous.get('i0_verified_s')
                                or candidate.get('integral_zero_verified_s')!=previous.get('i0_verified_s')
                                or settled.get('completed') is not True or len(rows)<3
                                or progress.get('stationary_arrival_claimed') is not False):
                            raise RuntimeError('arm finishing lacks preceding settled release progress')
                        try:
                            tail=np.asarray([row['q_position'] for row in rows[-3:]],dtype=np.float64)
                            before=float(progress['start_position'])-target
                            after=float(progress['settled_position'])-target
                            valid=(tail.shape==(3,6) and np.isfinite(tail).all()
                                and np.isfinite(before) and np.isfinite(after) and before*after>0
                                and abs(after)<abs(before) and progress['original_target']==target
                                and progress['settled_position']==tail[-1,2]
                                and progress['settled_readback_time_s']==rows[-1]['readback_time_s']
                                and np.abs(np.diff(tail,axis=0)).max()<=.1
                                and all(not any(row['moving'].values())
                                    and row['parameters']['I_Coefficient']==0
                                    and row['started_s']>previous['i0_verified_s'] for row in rows[-3:])
                                and pulse['started_s']>=rows[-1]['returned_s'])
                        except (KeyError,TypeError,ValueError):
                            valid=False
                        if not valid:
                            raise RuntimeError('arm finishing has invalid settled release progress')
                        expected_maximum=remaining
                    else:
                        expected_basis=('remaining_cumulative_budget' if control==slow_control
                                        else 'initial_or_rebound_two_second_limit')
                        if (pulse.get('budget_basis')!=expected_basis
                                or control==slow_control and pulse_index>0
                                    and pulse.get('stage')!='rebound_candidate'):
                            raise RuntimeError('arm finishing lacks its integral pulse budget basis')
                start,end=pulse.get('started_s'),pulse.get('i0_verified_s')
                if (type(start) not in (int,float) or type(end) not in (int,float)
                        or not np.isfinite(start) or not np.isfinite(end) or not 0<=end-start<=expected_maximum
                        or previous_end is not None and start<=previous_end):
                    raise RuntimeError('arm finishing lacks timely verified I0 for every pulse')
                maximum=pulse.get('maximum_s');duration=pulse.get('duration_s')
                if (type(maximum) not in (int,float) or type(duration) not in (int,float)
                        or not np.isfinite(maximum) or not np.isfinite(duration)
                        or abs(maximum-expected_maximum)>1e-8
                        or abs(duration-(end-start))>1e-8 or duration>maximum
                        or pulse.get('within_limit') is not True):
                    raise RuntimeError('arm finishing has inconsistent integral pulse accounting')
                durations.append(end-start)
                previous_end=end
            total=exposure.get('duration_s')
            if (sum(durations)>control['maximum_cumulative_s'] or type(exposure.get('pulse_count')) is not int
                    or exposure['pulse_count']!=pulse_count
                    or type(total) not in (int,float) or not np.isfinite(total)
                    or abs(total-sum(durations))>1e-8
                    or exposure.get('i0_verified_s')!=previous_end
                    or exposure.get('within_limit') is not True or exposure.get('all_pulses_i0_verified') is not True):
                raise RuntimeError('arm finishing lacks cumulative integral pulse evidence')
        sequence=[(event['register'],event['value']) for event in writes if event['motor']==motor]
        expected=[('Torque_Limit',200),('D_Coefficient',32),('Torque_Limit',300)]
        expected += [('I_Coefficient',1),('I_Coefficient',0)]*pulse_count
        expected += [('Torque_Limit',200),('D_Coefficient',0),('Torque_Limit',1000)]
        if sequence != expected:
            raise RuntimeError('arm finishing parameter writes do not match the bounded active sequence')


def validate_arm_original_goal_hold(report,follow,goal):
    """An arm success needs the full final second, not only its last three q."""
    from .v21_position_arrival import evaluate_position_arrival
    from .v21_hardware_state import ORDER, position_servo_selected_contract

    if (not report['applied'] and not report['original_target_reached']
            and report.get('selected_axes_held') is not True):
        return
    hold=report.get('original_goal_hold')
    rows=hold.get('readbacks') if isinstance(hold,dict) else None
    attempts=hold.get('snapshot_attempts') if isinstance(hold,dict) else None
    lateness_events=hold.get('scheduler_lateness_events') if isinstance(hold,dict) else None
    raw=report.get('original_target_raw')
    if (report.get('selected_axes_completed') is not True or report.get('selected_axes_held') is not True
            or not isinstance(hold,dict)
            or hold.get('completed') is not True or hold.get('sample_count')!=22
            or hold.get('requested_minimum_span_s')!=1.
            or not isinstance(rows,list) or len(rows)!=22
            or not isinstance(attempts,list) or not 22<=len(attempts)<=44
            or hold.get('maximum_snapshot_attempts')!=2
            or hold.get('retry_policy')!='complete_snapshot_reacquisition_after_sync_read_connection_error'
            or hold.get('scheduler_lateness_notice_s')!=.03
            or hold.get('scheduler_lateness_policy')!='record_and_continue_with_fresh_actual_checks'
            or not isinstance(lateness_events,list)
            or not isinstance(raw,dict) or set(raw)!=set(ORDER)
            or any(type(value) is not int for value in raw.values())):
        raise RuntimeError('arm finishing lacks the final full-second original-goal hold')
    positions=np.asarray(follow['positions'],dtype=np.float64)
    times=np.asarray(follow['times_s'],dtype=np.float64)
    epoch=hold.get('epoch_s')
    if (not isinstance(epoch,(int,float)) or not np.isfinite(epoch)
            or hold.get('period_s')!=.05
            or hold.get('schedule_basis')!='previous_sample_completion_plus_0.05s'):
        raise RuntimeError('arm finishing hold lacks its actual completion-based sampling schedule')
    previous_completed=None
    schedule_valid=True
    observed_lateness=[]
    cursor=0
    dynamic_registers={'Present_Position','Goal_Position','Moving','Torque_Enable','Present_Load',
        'P_Coefficient','I_Coefficient','D_Coefficient','Torque_Limit'}
    for index,row in enumerate(rows,1):
        group=[]
        while cursor<len(attempts) and attempts[cursor].get('sample_index')==index:
            group.append(attempts[cursor]);cursor+=1
        expected=(epoch if previous_completed is None else previous_completed)+.05
        if len(group) not in (1,2):schedule_valid=False;break
        for number,attempt in enumerate(group,1):
            scheduled=attempt.get('scheduled_start_s');started=attempt.get('started_s')
            expected_basis=('previous_sample_completion_plus_0.05s' if number==1
                            else 'previous_attempt_completion_plus_0.05s')
            expected_scheduled=(expected if number==1 else group[number-2].get('finished_s',np.nan)+.05)
            clocks=np.asarray([scheduled,started],dtype=np.float64)
            reads=attempt.get('register_reads')
            recorded_lateness=attempt.get('start_lateness_s')
            actual_lateness=float(started-scheduled) if np.isfinite(clocks).all() else np.nan
            if (attempt.get('snapshot_attempt')!=number or attempt.get('schedule_basis')!=expected_basis
                    or not np.isfinite(clocks).all() or abs(float(scheduled)-expected_scheduled)>1e-8
                    or started<scheduled-1e-8
                    or not isinstance(recorded_lateness,(int,float)) or not np.isfinite(recorded_lateness)
                    or abs(float(recorded_lateness)-actual_lateness)>1e-8
                    or not isinstance(reads,dict) or not set(reads).issubset(dynamic_registers)):
                schedule_valid=False;break
            observed_lateness.append((index,number,attempt,actual_lateness))
            if number<len(group):
                finished=attempt.get('finished_s');failed_register=attempt.get('failed_register')
                event=reads.get(failed_register) if isinstance(failed_register,str) else None
                if (attempt.get('completed') is not False or attempt.get('discarded') is not True
                        or attempt.get('failed') is not True or attempt.get('retry_started') is not True
                        or not isinstance(finished,(int,float)) or not np.isfinite(finished) or finished<started
                        or not isinstance(event,dict) or event.get('operation')!='sync_read'
                        or event.get('num_retry')!=1 or not isinstance(event.get('failed_s'),(int,float))
                        or not np.isfinite(event['failed_s']) or event['failed_s']>finished+1e-8):
                    schedule_valid=False;break
            else:
                returned=attempt.get('returned_s');readback=attempt.get('readback_time_s')
                loads=attempt.get('all_present_load_raw')
                if (attempt.get('completed') is not True or attempt.get('discarded') is not False
                        or attempt.get('register_snapshot_complete') is not True
                        or not isinstance(returned,(int,float)) or not isinstance(readback,(int,float))
                        or not np.isfinite([readback,returned]).all() or readback<started-1e-8
                        or returned<readback-1e-8 or set(reads)!=dynamic_registers
                        or any(event.get('num_retry')!=1 or 'returned_s' not in event for event in reads.values())
                        or not isinstance(loads,dict) or set(loads)!=set(ORDER)
                        or any(type(value) is not int or not 0<=value<=2047 for value in loads.values())
                        or attempt!=row):
                    schedule_valid=False;break
                previous_completed=float(returned)
        if not schedule_valid:break
    if cursor!=len(attempts):schedule_valid=False
    expected_late=[row for row in observed_lateness if row[3]>.03]
    maximum_observed=max((row[3] for row in observed_lateness),default=0.)
    reported_maximum=hold.get('maximum_observed_start_lateness_s')
    if (not isinstance(reported_maximum,(int,float)) or not np.isfinite(reported_maximum)
            or abs(float(reported_maximum)-maximum_observed)>1e-8
            or len(lateness_events)!=len(expected_late)):
        schedule_valid=False
    if schedule_valid:
        for event,(index,number,attempt,lateness) in zip(lateness_events,expected_late):
            if not isinstance(event,dict):
                schedule_valid=False;break
            clocks=np.asarray([event.get('scheduled_start_s'),event.get('observed_start_s'),
                               event.get('observed_lateness_s')],dtype=np.float64)
            if (not np.isfinite(clocks).all() or event.get('phase')!=attempt.get('phase')
                    or event.get('sample_index')!=index or event.get('snapshot_attempt')!=number
                    or abs(clocks[0]-attempt['scheduled_start_s'])>1e-8
                    or abs(clocks[1]-attempt['started_s'])>1e-8 or abs(clocks[2]-lateness)>1e-8
                    or event.get('notice_threshold_s')!=.03 or event.get('snapshot_started') is not True
                    or event.get('scheduler_lateness_only') is not True):
                schedule_valid=False;break
    if (positions.shape!=(22,6) or times.shape!=(22,)
            or any(not isinstance(row,dict) or row.get('q_position')!=position.tolist()
                or row.get('readback_time_s')!=float(stamp)
                or row.get('goal_raw')!=raw
                or row.get('moving')!=dict.fromkeys(ORDER,0)
                for row,position,stamp in zip(rows,positions,times))
            or times[-1]-times[0]<1.
            or abs(float(hold.get('actual_span_s',-1))-(times[-1]-times[0]))>1e-8
            or not schedule_valid or (np.diff(times)<.05-1e-8).any()):
        raise RuntimeError('arm finishing hold differs from its actual completion samples')
    contract=position_servo_selected_contract(follow,goal)
    admission=report.get('admission')
    if (contract is None or any(report.get(key)!=value for key,value in contract.items())
            or not isinstance(admission,dict) or admission.get('current_q')!=contract['deferred_anchor_q']
            or hold.get('selected_axes_arrival')!=follow.get('selected_axes_arrival')):
        raise RuntimeError('arm finishing lacks selected-axis evidence bound to actual admission')
    arrival=evaluate_position_arrival(goal,positions,times-float(follow['source_time_s']),
        position_tolerance=.2,step_tolerance=.1,quiet_samples=22)
    reported=hold.get('arrival')
    if (report.get('original_goal_held') is not arrival['reached'] or not isinstance(reported,dict)
            or any(reported.get(key)!=value for key,value in arrival.items())):
        raise RuntimeError('arm finishing full-axis arrival differs from its actual final second')


def invalidate_finishing_arrival(result,reason):
    """Preserve earlier observations without calling them a final success."""
    if 'position_servo_timing' not in result:
        return
    previous=result.get('original_target_arrival')
    if previous is not None and 'position_servo_prior_arrival' not in result:
        result['position_servo_prior_arrival']=deepcopy(previous)
    result['original_target_reached']=False
    result['original_target_arrival']=dict(schema='v21-position-arrival-unverified-v1',
        original_target=deepcopy(result['commands_position'][-1]),reached=False,verified=False,
        reason='unverified_after_position_finishing_failure',error=reason,
        source='failed_finishing_transaction',position_unit='lerobot_calibrated_position',
        model_feedback_label=False)
    if result.get('position_servo',{}).get('schema')=='v21-arm-position-servo-v1':
        prior=result.get('selected_axes_arrival')
        if prior is not None and 'position_servo_prior_selected_arrival' not in result:
            result['position_servo_prior_selected_arrival']=deepcopy(prior)
        result.update(selected_axes_completed=False,selected_axes_held=False,
            selected_axes_arrival=dict(schema='v21-selected-position-arrival-unverified-v1',
                axes=deepcopy(result['position_servo'].get('selected_axes')),reached=False,verified=False,
                reason='unverified_after_position_finishing_failure',error=reason,model_feedback_label=False))


class V21TransactionBridge:
    def __init__(self, runtime, *, read_q, write_q, capture_endpoint, physical_check,
                 halt, clock=time.monotonic, wait=time.sleep, cancelled=lambda: False,
                 maximum_lateness_s=.030,write_metrics=None,read_metrics=None,normal_follow_budget_s=0.,
                 follow_position_tolerance=3.,follow_step_tolerance=.25,follow_quiet_samples=3,
                 completion_hold=None,request_stop=lambda:None,read_pre_dispatch_state=None,
                 verify_policy_goal=None,finish_position=None,cem_response_joint_count=5):
        self.runtime = runtime
        self.read_q, self.write_q = read_q, write_q
        self.capture_endpoint, self.physical_check, self.halt = capture_endpoint, physical_check, halt
        self.clock, self.wait, self.cancelled = clock, wait, cancelled
        self.completion_hold,self.request_stop=completion_hold,request_stop
        self.read_pre_dispatch_state=read_pre_dispatch_state
        self.verify_policy_goal=verify_policy_goal
        self.finish_position=finish_position
        self.maximum_lateness_s = float(maximum_lateness_s)
        self.write_metrics=write_metrics
        self.read_metrics=read_metrics
        self.normal_follow_budget_s=float(normal_follow_budget_s)
        self.follow_position_tolerance=float(follow_position_tolerance)
        self.follow_step_tolerance=float(follow_step_tolerance)
        self.follow_quiet_samples=int(follow_quiet_samples)
        if type(cem_response_joint_count) is not int or cem_response_joint_count not in (5, 6):
            raise ValueError('cem_response_joint_count must be exactly 5 or 6')
        self.cem_response_joint_count = cem_response_joint_count
        if (self.normal_follow_budget_s not in (0.,.4,2.)
                or not np.isfinite([self.follow_position_tolerance,self.follow_step_tolerance]).all()
                or self.follow_position_tolerance<=0 or self.follow_step_tolerance<=0
                or self.follow_quiet_samples<1):
            raise ValueError('invalid bounded-follow contract')
        if not 0 < self.maximum_lateness_s <= .25:
            raise ValueError("five-step bridge requires a sub-period deadline allowance")
        if finish_position is not None:
            if not callable(finish_position) or not callable(completion_hold):
                raise TypeError('position finishing requires finishing and goal-retention callbacks')
            if self.normal_follow_budget_s!=2.:
                raise ValueError('position finishing requires the original two-second follow')
        self.lock = threading.Lock()
        self.last_attempt = None
        self.last_source_state = None
        self.last_forecast = None
        self.last_actor_input = None
        self.last_plan_observation_source = None
        self.extra_readback_check = None  # Optional operation-scoped mechanical guard.
        self.extra_position_commands = None
        self.extra_position_note = None

    def _read(self):
        started = float(self.clock())
        q = np.asarray(self.read_q(), np.float32)
        returned = float(self.clock())
        if self.last_attempt is not None:
            timing = dict(started_s=started, returned_s=returned, duration_s=returned-started)
            if self.read_metrics is not None:
                timing['motor_io'] = self.read_metrics()
            self.last_attempt.setdefault('read_calls', []).append(timing)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("actual hardware readback must be finite [6]")
        self._check_extra_readback(q)
        return q.copy(), returned

    def _check_extra_readback(self,q):
        if self.extra_readback_check is None:return
        checked=np.array(q,copy=True)
        self.extra_readback_check(checked)
        if not np.array_equal(checked,q):
            raise RuntimeError('operation-scoped guard changed an actual readback')

    def _verify_written_policy_goal(self, command, index, deadline, result):
        """Read-only command-register evidence inside the original deadline."""
        if self.verify_policy_goal is None:return
        event=dict(command_index=int(index),callback_started_s=float(self.clock()),
                   read_attempted=False,read_returned=False,goal_register_verified=False,
                   motor_register_writes=0)
        result['policy_goal_readbacks'].append(event)
        try:
            if self.cancelled():raise RuntimeError('policy goal verification cancelled before callback')
            proof=self.verify_policy_goal(np.array(command,dtype=np.float32,copy=True))
            if not isinstance(proof,dict):raise RuntimeError('policy goal verification returned no evidence')
            event.update(deepcopy(proof))
            if (event.get('schema')!='v21-policy-goal-readback-v1'
                    or event.get('register')!='Goal_Position' or event.get('normalize') is not False
                    or event.get('read_attempted') is not True or event.get('read_returned') is not True
                    or event.get('goal_register_verified') is not True
                    or event.get('motor_register_writes')!=0
                    or not isinstance(event.get('expected_goal_raw'),dict)
                    or event.get('goal_raw')!=event['expected_goal_raw']):
                raise RuntimeError('policy goal verification did not establish the exact written raw goal')
            if self.cancelled():raise RuntimeError('policy goal verification cancelled after callback')
        except BaseException as error:
            diagnostic=getattr(error,'goal_readback_verification',None)
            if isinstance(diagnostic,dict):event.update(deepcopy(diagnostic))
            event.update(callback_failed_s=float(self.clock()),error=f'{type(error).__name__}: {error}')
            raise
        finally:
            returned=float(self.clock())
            event.update(callback_returned_s=returned,
                         callback_duration_s=returned-event['callback_started_s'],
                         policy_io_lateness_s=returned-deadline)
            result['policy_goal_reads_attempted']+=int(event.get('read_attempted') is True)
            result['policy_goal_reads_returned']+=int(event.get('read_returned') is True)
            result['policy_goal_registers_verified']+=int(event.get('goal_register_verified') is True)
            if returned-deadline>self.maximum_lateness_s:
                result['deadline_miss']=dict(kind='policy-goal-verification',command_index=int(index),
                    allowed_lateness_s=self.maximum_lateness_s,observed_lateness_s=returned-deadline,
                    excess_s=returned-deadline-self.maximum_lateness_s)
        if event['policy_io_lateness_s']>self.maximum_lateness_s:
            verified=event.get('goal_register_verified') is True
            # #region agent log
            try:
                import json as _json, time as _time
                with open('/home/james/feedback-corrected-grounding/.cursor/debug-55ff20.log','a') as _f:
                    _f.write(_json.dumps({'sessionId':'55ff20','runId':'post-fix','hypothesisId':'E','location':'v21_hardware_bridge.py:goal_deadline','message':'goal verify late','data':{'verified':verified,'lateness_s':round(float(event['policy_io_lateness_s']),4),'limit_s':self.maximum_lateness_s},'timestamp':int(_time.time()*1000)})+'\n')
            except Exception:
                pass
            # #endregion
            if not verified:
                raise RuntimeError('motor goal verification missed its deadline')

    def _boundary(self, deadline):
        delay = deadline-self.clock()
        if delay > 0:
            self.wait(delay)
        if self.cancelled():
            raise RuntimeError("five-command transaction cancelled")
        observed = float(self.clock())
        if observed-deadline > self.maximum_lateness_s:
            if self.last_attempt is not None:
                self.last_attempt['deadline_miss'] = dict(kind='read-or-boundary', deadline_s=deadline,
                    observed_s=observed, observed_lateness_s=observed-deadline,
                    allowed_lateness_s=self.maximum_lateness_s,
                    excess_s=observed-deadline-self.maximum_lateness_s)
            raise RuntimeError("five-command deadline missed")

    def plan(self, *, command_builder=None, action_source='v21_actor'):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("hardware transaction already active")
        try:
            if self.runtime.executing:
                raise RuntimeError("cannot plan during execution")
            self.runtime.stop()  # discard an unexecuted preview, retain actual memory
            endpoint = self.capture_endpoint()
            if self.extra_readback_check is not None:
                self._check_extra_readback(endpoint.q[0].detach().cpu().numpy())
            self.last_plan_observation_source = 'fresh_capture'
            return self.runtime.plan(endpoint=endpoint,command_builder=command_builder,action_source=action_source)
        finally:
            self.lock.release()

    def plan_from_completed_observation(self, *, command_builder=None, action_source='v21_actor'):
        """Continuous-loop only: use its newly completed actual observation once.

        A delayed loop captures fresh data instead. This is an observation
        scheduling choice, not permission to send an old command or bypass
        the actual-q/clock/physical checks in execute(). Reuse does not
        change the requested action_source.
        """
        extra = {}
        if command_builder is not None or action_source != 'v21_actor':
            extra.update(command_builder=command_builder, action_source=action_source)
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('hardware transaction already active')
        try:
            if self.runtime.executing:
                raise RuntimeError('cannot plan during execution')
            self.runtime.stop()
            state = self.runtime.state
            age = None if state is None else self.clock()-state.time.item()
            if age is not None and age < 0:
                raise RuntimeError('completed observation is from the future')
            if (self.last_attempt is not None and self.last_attempt.get('completed')
                    and age is not None and age <= .5):
                self.last_plan_observation_source = 'completed_transaction_endpoint'
                return self.runtime.plan(**extra)
            endpoint = self.capture_endpoint()
            self.last_plan_observation_source = 'fresh_capture'
            return self.runtime.plan(endpoint=endpoint, **extra)
        finally:
            self.lock.release()

    def _follow(self, result, epoch, q5, *, final_offset_s=.5):
        """Read the existing final goal's response; issue no extra goal writes."""
        diagnostic = result.get('action_source') in (
            'j3_pid_diagnostic', 'j3_range_diagnostic',
            'j3_compensation_diagnostic', 'j3_native_response_diagnostic')
        full_window = self.normal_follow_budget_s == 2. and not diagnostic
        follow=dict(budget_s=self.normal_follow_budget_s, positions=[], times_s=[],
                    goal_reached=None, reason='disabled', position_tolerance=self.follow_position_tolerance,
                    step_tolerance=self.follow_step_tolerance, quiet_samples=self.follow_quiet_samples)
        result['normal_follow']=follow
        if not self.normal_follow_budget_s:return
        if not diagnostic:
            follow.update(hold_eligible=False,
                          goal_reached_scope='original_command_position_arrival',
                          fixed_observation_window=full_window)
        cem_response_required = result.get('action_source') in ('v21_lewm_cem','j3_pid_diagnostic')
        if cem_response_required:
            follow.update(early_exit_requires_cem_response=True,
                          cem_response_satisfied=None, cem_response_diagnostic=None)
        goal=np.asarray(result.get('follow_goal_override') or result['commands_position'][-1],np.float32)
        previous=q5;quiet=0
        follow.update(goal_reached=False,reason='budget_exhausted')
        # Keep the same 20 Hz clocked observations for the explicit budget.
        # Following never adds policy commands or model readback labels.
        sample_count = round(self.normal_follow_budget_s / .05)
        # region agent log
        try:
            import json as _json, time as _time
            _cmd = np.asarray(result.get('commands_position'), dtype=np.float64)
            _src = np.asarray(result.get('source_q_position'), dtype=np.float64)
            _peak = None if _cmd.shape != (5, 6) or _src.shape != (6,) else (_cmd - _src).tolist()
            with open('/home/james/feedback-corrected-grounding/.cursor/debug-75d952.log','a') as _f:
                _f.write(_json.dumps({'sessionId':'75d952','hypothesisId':'A','location':'v21_hardware_bridge.py:_follow','message':'follow start','data':{'budget_s':self.normal_follow_budget_s,'final_offset_s':final_offset_s,'sample_count':sample_count,'goal_minus_q5':(goal.astype(float)-np.asarray(q5,dtype=float)).tolist(),'command_minus_source':_peak,'command_times_s':result.get('command_times_s'),'readback_times_s':result.get('readback_times_s'),'epoch':float(epoch)},'timestamp':int(_time.time()*1000)})+'\n')
        except Exception:
            pass
        # endregion
        for index in range(1, sample_count+1):
            deadline=epoch+final_offset_s+index*.05
            try:
                self._boundary(deadline)
                q,stamp=self._read()
                follow['positions'].append(q.tolist());follow['times_s'].append(stamp)
                # Recheck the current readback and remaining path to the same goal.
                self.physical_check(q,np.repeat(goal[None],5,axis=0))
                self._boundary(deadline)
                error=float(np.max(np.abs(q-goal)))
                step=float(np.max(np.abs(q-previous)))
                quiet=quiet+1 if error<=self.follow_position_tolerance and step<=self.follow_step_tolerance else 0
                follow.update(final_position_error=error,final_step_delta=step)
                previous=q
                if cem_response_required:
                    # The original six model samples remain untouched. This
                    # temporary endpoint is the just-observed following sample,
                    # with its own clock, used only for deciding whether to exit.
                    observed = {key: result[key] for key in (
                        'source_q_position', 'commands_position',
                        'measured_q_path', 'readback_times_s')}
                    observed.update(endpoint_q_position=q.tolist(), endpoint_time_s=float(stamp))
                    try:
                        require_actual_cem_response(
                            observed,
                            minimum_motion=self.follow_step_tolerance,
                            checked_joint_count=self.cem_response_joint_count,
                        )
                    except CEMActualResponseError as exc:
                        follow.update(cem_response_satisfied=False,
                                      cem_response_diagnostic=exc.diagnostic)
                    else:
                        follow.update(cem_response_satisfied=True, cem_response_diagnostic=None)
                    self._boundary(deadline)
                if (not diagnostic and not full_window
                        and quiet>=self.follow_quiet_samples) and (
                        not cem_response_required or follow['cem_response_satisfied']):
                    follow.update(goal_reached=True,reason='within_existing_hold_tolerances');break
            except RuntimeError as exc:
                if str(exc)!='five-command deadline missed':
                    raise
                follow['reason']='follow_read_lateness'
                break
        # region agent log
        try:
            import json as _json, time as _time
            _pos = np.asarray(follow['positions'], dtype=np.float64) if follow['positions'] else None
            _q5 = np.asarray(q5, dtype=np.float64)
            _delta = None if _pos is None or len(_pos)==0 else (_pos - _q5).tolist()
            with open('/home/james/feedback-corrected-grounding/.cursor/debug-75d952.log','a') as _f:
                _f.write(_json.dumps({'sessionId':'75d952','hypothesisId':'B','location':'v21_hardware_bridge.py:_follow','message':'follow end','data':{'reason':follow.get('reason'),'samples':len(follow['positions']),'elapsed_from_epoch_s':None if not follow['times_s'] else float(follow['times_s'][-1])-float(epoch),'cem_response_satisfied':follow.get('cem_response_satisfied'),'final_position_error':follow.get('final_position_error'),'final_step_delta':follow.get('final_step_delta'),'delta_from_q5':_delta,'blocked':(follow.get('cem_response_diagnostic') or {}).get('blocked_joint_indices')},'timestamp':int(_time.time()*1000)})+'\n')
        except Exception:
            pass
        # endregion
        if result.get('action_source') in ('j3_range_diagnostic', 'j3_compensation_diagnostic',
                                         'j3_native_response_diagnostic'):
            follow.update(goal_reached=quiet>=self.follow_quiet_samples,
                          reason='fixed_window_complete', fixed_observation_window=True)
        if not diagnostic:
            # The existing 3-unit tolerance only permits keeping a quiet goal
            # latched. It is not evidence that the command position was reached.
            follow['hold_eligible'] = bool(quiet >= self.follow_quiet_samples and (
                not cem_response_required or follow['cem_response_satisfied']))
            if full_window:
                follow['reason'] = 'fixed_window_complete'
            from .v21_position_arrival import evaluate_position_arrival
            arrival = evaluate_position_arrival(
                goal, [q5.tolist(), *follow['positions']],
                [result['readback_times_s'][-1]-epoch,
                 *(stamp-epoch for stamp in follow['times_s'])],
                position_tolerance=.2, step_tolerance=.1, quiet_samples=3)
            arrival.update(position_unit='lerobot_calibrated_position',
                source='actual_q5_and_follow_before_completion_hold',
                source_time_s=epoch, excludes_post_hold_readbacks=True,
                control_influence=False)
            follow.update(goal_reached=arrival['reached'], original_target_arrival=arrival)
            result.update(original_target_arrival=arrival,
                          original_target_reached=arrival['reached'])

    REBASE_ATTEMPTS = 4
    REBASE_SETTLE_S = .05

    def _rebase_before_epoch(self, q):
        """Heavy re-planning from the actual q happens here, before any deadline.

        Only in 'always' recompute mode for recomputable sources. A still-moving
        arm gets a short settle wait between attempts; any residual drift at the
        epoch readback is handled by ``begin_execution`` itself. No motor write
        occurs here.
        """
        runtime=self.runtime
        if getattr(runtime,'pre_execution_recompute','bounded')!='always':
            return []
        if getattr(runtime,'pending_action_source',None) not in getattr(runtime,'RECOMPUTABLE_SOURCES',()):
            return []
        device=runtime.pending.source_q.device
        rebases=[]
        for attempt in range(self.REBASE_ATTEMPTS):
            actual=torch.as_tensor(q,device=device)[None]
            age=float(self.clock())-runtime.pending.source_time.item()
            if runtime.pending_matches(actual) and age<=runtime.max_plan_age_s:
                break
            if self.cancelled():
                raise RuntimeError('five-command transaction cancelled before rebase')
            stamp=float(self.clock())
            runtime.rebase_pending(actual,torch.tensor([stamp],device=device,dtype=torch.float64))
            rebases.append(dict(runtime.last_rebase or {},attempt=attempt,no_policy_commands_sent=True))
            if attempt+1<self.REBASE_ATTEMPTS:
                self.wait(self.REBASE_SETTLE_S)
                if self.cancelled():
                    raise RuntimeError('five-command transaction cancelled before rebase')
                q,_=self._read()
        return rebases

    def _write_small_step_extension(self, result, epoch, first_index=5):
        """Send ten more inferred positions when the 40-step plan is small.

        The model trace stays five commands. These writes happen after that
        readback and before the follow, so the arm keeps the longer prefix.
        """
        note = self.extra_position_note
        rows = self.extra_position_commands
        self.extra_position_note = None
        self.extra_position_commands = None
        if note is not None:
            result['inferred_execution'] = dict(note)
        if rows is None:
            result.setdefault('executed_steps', 5)
            return
        commands = np.asarray(rows, np.float32)
        if commands.ndim != 2 or commands.shape[1] != 6 or not np.isfinite(commands).all():
            raise ValueError('extra position commands must be finite [N,6]')
        written = []
        previous = np.asarray(result['commands_position'][-1], np.float32)
        for index, command in enumerate(commands):
            deadline = epoch + (first_index + index) * 0.1
            self._boundary(deadline)
            sent_at = float(self.clock())
            actual = np.asarray(self.write_q(command, previous), np.float32)
            returned_at = float(self.clock())
            if actual.shape != (6,) or not np.allclose(actual, command, atol=1e-4, rtol=0):
                raise RuntimeError('transport changed an extension command')
            if returned_at - deadline > self.maximum_lateness_s:
                result['deadline_miss'] = dict(kind='extension', command_index=5 + index,
                    allowed_lateness_s=self.maximum_lateness_s,
                    observed_lateness_s=returned_at - deadline,
                    excess_s=returned_at - deadline - self.maximum_lateness_s)
                raise RuntimeError('motor write missed its deadline')
            self._verify_written_policy_goal(actual, 5 + index, deadline, result)
            written.append(actual.tolist())
            previous = actual.copy()
        result['extension_commands_position'] = written
        result['executed_steps'] = 5 + len(written)
        result['follow_goal_override'] = written[-1]

    def _transport_five(self, commands, q0, epoch, result):
        """Preserve the five-command timing and result contract."""
        self._transport_commands(commands, q0, epoch, result, final_offset_s=.5,
                                 hold_key='protective_hold_after_q5')

    def _finish_normal_position(self,result):
        """Keep optional same-goal settling separate from the five model labels."""
        from .v21_hardware_state import position_servo_completion_proof

        timing=dict(started_s=float(self.clock()),callback_invoked=False)
        result['position_servo_timing']=timing
        try:
            if self.cancelled():raise RuntimeError('position finishing cancelled before callback')
            if self.normal_follow_budget_s!=2. or self.completion_hold is None:
                raise RuntimeError('position finishing requires full follow and original-goal retention')
            timing['callback_invoked']=True
            report=self.finish_position(deepcopy(result))
            if not isinstance(report,dict):
                raise RuntimeError('position finishing returned no evidence')
            result['position_servo']=deepcopy(report)
            cleanup=report.get('cleanup')
            if (report.get('schema') not in ('v21-j3-position-servo-v1','v21-arm-position-servo-v1')
                    or report.get('completed') is not True
                    or type(report.get('applied')) is not bool
                    or not isinstance(cleanup,dict) or cleanup.get('restored') is not True
                    or report.get('original_goal_preserved') is not True
                    or report.get('model_feedback_label') is not False):
                raise RuntimeError('position finishing did not prove original-goal and baseline restoration')
            if report['schema']=='v21-arm-position-servo-v1':
                validate_arm_finishing_restoration(report)
            follow=report.get('completion_follow')
            _,times,arrival=position_servo_completion_proof(follow,result['commands_position'][-1])
            if report['schema']=='v21-arm-position-servo-v1':
                validate_arm_original_goal_hold(report,follow,result['commands_position'][-1])
                if 'selected_axes' in follow and follow.get('source_q0')!=result['measured_q_path'][0]:
                    raise RuntimeError('selected-axis finishing changed the original execution source')
            original=result['normal_follow']
            if (follow['source_time_s']!=result['source_time_s']
                    or len(original['times_s'])<3
                    or times[0]<original['times_s'][-3]
                    or times[-1]<=original['times_s'][-1]
                    or times[-1]>float(self.clock())):
                raise RuntimeError('position finishing evidence is not fresh for this execution')
            reported=report.get('original_target_arrival')
            if (not isinstance(reported,dict)
                    or any(reported.get(key)!=value for key,value in arrival.items())
                    or report.get('original_target_reached') is not arrival['reached']
                    or (report['applied'] and not arrival['reached']
                        and not (report['schema']=='v21-arm-position-servo-v1'
                                 and report.get('selected_axes_completed') is True
                                 and report.get('selected_axes_held') is True))):
                raise RuntimeError('position finishing arrival disagrees with actual final readbacks')
            if self.cancelled():raise RuntimeError('position finishing cancelled after callback')
            result.update(original_target_arrival=deepcopy(reported),
                          original_target_reached=arrival['reached'])
            if report['schema']=='v21-arm-position-servo-v1' and report.get('selected_axes_held') is True:
                result.update(selected_axes_completed=True,selected_axes_held=True,
                    selected_axes_arrival=deepcopy(follow['selected_axes_arrival']))
            return deepcopy(follow)
        except BaseException as error:
            partial=getattr(error,'report',None)
            if isinstance(partial,dict):result['position_servo']=deepcopy(partial)
            result['position_servo_error']=f'{type(error).__name__}: {error}'
            invalidate_finishing_arrival(result,result['position_servo_error'])
            raise
        finally:
            timing['returned_s']=float(self.clock())
            timing['duration_s']=timing['returned_s']-timing['started_s']

    def _sample_dynamic_actuators(self, telemetry, index, deadline, result):
        """Optional read-only evidence inside the remaining command gap."""
        started = float(self.clock())
        event = dict(command_index=int(index), deadline_s=deadline,
                     callback_started_s=started, callback_invoked=False,
                     callback_returned=False, snapshot=None)
        result.setdefault('dynamic_actuator_readbacks', []).append(event)
        if started >= deadline:
            event.update(skip_reason='no_remaining_command_gap', skipped_s=started)
            return
        try:
            if self.cancelled():
                raise RuntimeError('dynamic actuator telemetry cancelled before callback')
            event['callback_invoked'] = True
            snapshot = telemetry(command_index=int(index), deadline_s=deadline)
            event.update(callback_returned=True, snapshot=deepcopy(snapshot))
            if not isinstance(snapshot, dict):
                raise RuntimeError('dynamic actuator telemetry returned no readback record')
            if snapshot.get('deadline_missed') or snapshot.get('read_budget_exceeded'):
                raise RuntimeError('dynamic actuator telemetry exceeded its read budget')
            if self.cancelled():
                raise RuntimeError('dynamic actuator telemetry cancelled after callback')
        except BaseException as error:
            partial = getattr(error, 'dynamic_actuator_snapshot', None)
            if isinstance(partial, dict):
                event['snapshot'] = deepcopy(partial)
            event.update(error=f'{type(error).__name__}: {error}', callback_failed_s=float(self.clock()))
            raise
        finally:
            returned = float(self.clock())
            event.update(callback_returned_s=returned, callback_duration_s=returned-started,
                         gap_overrun_s=max(0., returned-deadline))
            if returned > deadline:
                result['dynamic_actuator_deadline_miss'] = dict(command_index=int(index),
                    deadline_s=deadline, observed_s=returned, overrun_s=returned-deadline,
                    original_motor_lateness_limit_s=self.maximum_lateness_s)
        if returned > deadline:
            event['error'] = 'RuntimeError: dynamic actuator telemetry overran the command gap'
            raise RuntimeError('dynamic actuator telemetry overran the command gap')

    def _transport_commands(self, commands, q0, epoch, result, *, final_offset_s, hold_key,
                            telemetry=None):
        """Shared actual write/read/Goal/follow/HOLD transport; no model work."""
        result["measured_q_path"].append(q0.tolist()); result["readback_times_s"].append(epoch)
        previous = q0.copy()
        for index, command in enumerate(commands):
            deadline = epoch+index*.1
            self._boundary(deadline)
            if index:
                measured, read_time = self._read()
                result["measured_q_path"].append(measured.tolist())
                result["readback_times_s"].append(read_time)
            # A slow read may not silently push the following motor write late.
            self._boundary(deadline)
            result["policy_commands_attempted"] += 1
            sent_at = float(self.clock())
            actual = np.asarray(self.write_q(command, previous), np.float32)
            returned_at = float(self.clock())
            result["policy_commands_returned"] += 1
            result["commands_position"].append(actual.tolist())
            result["command_times_s"].append(sent_at)
            result.setdefault("write_completed_times_s", []).append(returned_at)
            result.setdefault("write_durations_s", []).append(returned_at-sent_at)
            result.setdefault("write_lateness_s", []).append(returned_at-deadline)
            if self.write_metrics is not None:
                metrics=self.write_metrics()
                result.setdefault('motor_io_timing',[]).append(None if metrics is None else dict(metrics))
            if actual.shape != (6,) or not np.allclose(actual, command, atol=1e-4, rtol=0):
                raise RuntimeError("transport changed a model command")
            if returned_at-deadline > self.maximum_lateness_s:
                result['deadline_miss']=dict(command_index=index,allowed_lateness_s=self.maximum_lateness_s,
                    observed_lateness_s=returned_at-deadline,
                    excess_s=returned_at-deadline-self.maximum_lateness_s)
                raise RuntimeError("motor write missed its deadline")
            self._verify_written_policy_goal(actual,index,deadline,result)
            if telemetry is not None:
                self._sample_dynamic_actuators(telemetry, index, epoch+(index+1)*.1, result)
            previous = actual.copy()
        pending = self.extra_position_commands
        extra_n = 0 if pending is None else len(pending)
        if extra_n:
            self._write_small_step_extension(result, epoch, first_index=len(commands))
        end_offset = (len(commands) + extra_n) * .1
        self._boundary(epoch+end_offset)
        q5, q5_time = self._read()
        if q5_time-(epoch+end_offset) > self.maximum_lateness_s:
            raise RuntimeError("final readback missed its deadline")
        result["measured_q_path"].append(q5.tolist()); result["readback_times_s"].append(q5_time)
        self._follow(result,epoch,q5,final_offset_s=end_offset)
        compensation = result.get('action_source') == 'j3_compensation_diagnostic'
        native_response = result.get('action_source') == 'j3_native_response_diagnostic'
        if result.get('action_source') == 'j3_range_diagnostic' or compensation or native_response:
            from .v21_position_arrival import evaluate_position_arrival
            follow = result['normal_follow']
            original_target = result['original_target'] if compensation else commands[-1]
            try:
                arrival = evaluate_position_arrival(
                    original_target, [q5.tolist(), *follow['positions']],
                    [q5_time-epoch, *(stamp-epoch for stamp in follow['times_s'])],
                    position_tolerance=.2, step_tolerance=.1, quiet_samples=3)
            except (TypeError, ValueError) as error:
                # This additional assessment must not change motor cleanup.
                arrival = dict(reached=False, reason='invalid_arrival_readbacks',
                    original_target=np.asarray(original_target).tolist(),
                    error=f'{type(error).__name__}: {error}')
            arrival.update(position_unit='lerobot_calibrated_position',
                source=('actual_q5_and_follow_before_completion_hold' if compensation or native_response else
                        'actual_q10_and_follow_before_completion_hold'),
                source_time_s=epoch, excludes_post_hold_readbacks=True,
                control_influence=False)
            result['original_target_arrival'] = arrival
            result['original_target_reached'] = arrival['reached']
        completion_follow=result['normal_follow']
        if (self.finish_position is not None and len(commands)==5 and final_offset_s==.5
                and result.get('action_source') not in ('j3_pid_diagnostic','j3_range_diagnostic',
                    'j3_compensation_diagnostic','j3_native_response_diagnostic')):
            completion_follow=self._finish_normal_position(result)
        # A verified, quiet final goal may remain latched. Otherwise retain
        # the established freeze; neither path adds a model policy command.
        hold_started = self.clock()
        # Compensation must freeze at its measured endpoint. Retaining the
        # farther command goal would continue chasing beyond the original target.
        hold_result = (self.halt() if compensation or self.completion_hold is None else
                       self.completion_hold(result['commands_position'][-1],completion_follow,q5))
        result["completion_hold"] = completion_hold_record(hold_result, hold_started, self.clock())
        result['actuator_events']=list(result['completion_hold'].get('actuator_events',[]))
        if ((result.get('position_servo',{}).get('original_target_reached') is True
                or result.get('position_servo',{}).get('applied') is True
                or result.get('position_servo',{}).get('selected_axes_held') is True)
                and (result['completion_hold'].get('goal_register_verified') is not True
                    or result['completion_hold'].get('reused_existing_hold') is not True
                    or result['completion_hold'].get('goal_register_writes')!=0)):
            raise RuntimeError('position finishing did not retain the original goal during completion')
        result[hold_key] = True

    def execute_fixed_diagnostic(self, commands, source_q, *, enable=False):
        """One separately admitted J3 probe through the existing transport.

        No model plan, prediction, history update, camera capture or STOP clear
        occurs here. The existing owner must hold external operation admission
        and a durable motion seal; the PID adapter supplies that admission.
        The default does not inspect the owner or invoke any IO callback.
        """
        if type(enable) is not bool:
            raise TypeError('diagnostic enable must be an explicit bool')
        if not enable:
            return dict(dry_run=True, callbacks_invoked=0, hardware_used=False,
                        action_source='j3_pid_diagnostic', executable=False)
        source = np.array(source_q, dtype=np.float32, copy=True)
        sequence = np.array(commands, dtype=np.float32, copy=True)
        if (source.shape != (6,) or sequence.shape != (5, 6)
                or not np.isfinite(source).all() or not np.isfinite(sequence).all()):
            raise ValueError('diagnostic requires finite source [6] and commands [5,6]')
        expected = np.repeat(source[None], 5, axis=0)
        expected[:, 2] += np.array([-1.5, -3., -3., -3., -3.], np.float32)
        if not np.allclose(sequence, expected, atol=1e-4, rtol=0):
            raise ValueError('only the fixed J3 diagnostic is admitted')
        return self._execute_fixed_j3_diagnostic(sequence, source, action_source='j3_pid_diagnostic')

    def execute_j3_native_response_diagnostic(self, commands, source_q, *, enable=False):
        """One fixed P16 response sample with native five-step timing.

        The service owns admission and once-only consumption. This separate
        entry does not use the PID experiment's phase or alter its budget.
        """
        if type(enable) is not bool:
            raise TypeError('diagnostic enable must be an explicit bool')
        if not enable:
            return dict(dry_run=True, callbacks_invoked=0, hardware_used=False,
                        action_source='j3_native_response_diagnostic', executable=False)
        source = np.array(source_q, dtype=np.float32, copy=True)
        sequence = np.array(commands, dtype=np.float32, copy=True)
        if (source.shape != (6,) or sequence.shape != (5, 6)
                or not np.isfinite(source).all() or not np.isfinite(sequence).all()):
            raise ValueError('native response requires finite source [6] and commands [5,6]')
        expected = np.repeat(source[None], 5, axis=0)
        expected[:, 2] += np.array([-1.5, -3., -3., -3., -3.], np.float32)
        if not np.allclose(sequence, expected, atol=1e-4, rtol=0):
            raise ValueError('only the fixed five-command native J3 response profile is admitted')
        return self._execute_fixed_j3_diagnostic(
            sequence, source, action_source='j3_native_response_diagnostic')

    def execute_j3_range_diagnostic(self, commands, source_q, *, enable=False,
                                    amplitude=5, telemetry=None):
        """One explicitly admitted ten-command J3 range probe at unchanged P16.

        The caller owns service admission, verifies P16, and consumes its one
        execution before calling this transport. This method neither changes
        gains nor creates model history, RGB observations, or a new controller.
        """
        if type(enable) is not bool:
            raise TypeError('diagnostic enable must be an explicit bool')
        if not enable:
            return dict(dry_run=True, callbacks_invoked=0, hardware_used=False,
                        action_source='j3_range_diagnostic', executable=False)
        from .v21_j3_range_profile import range_offsets
        offsets = range_offsets(amplitude)
        if telemetry is not None and not callable(telemetry):
            raise TypeError('dynamic actuator telemetry must be an explicit callable')
        source = np.array(source_q, dtype=np.float32, copy=True)
        sequence = np.array(commands, dtype=np.float32, copy=True)
        if (source.shape != (6,) or sequence.shape != (10, 6)
                or not np.isfinite(source).all() or not np.isfinite(sequence).all()):
            raise ValueError('range diagnostic requires finite source [6] and commands [10,6]')
        expected = np.repeat(source[None], 10, axis=0)
        expected[:, 2] += np.asarray(offsets, np.float32)
        if not np.allclose(sequence, expected, atol=1e-4, rtol=0):
            raise ValueError('only the fixed ten-command J3 range diagnostic is admitted')
        return self._execute_fixed_j3_diagnostic(sequence, source, action_source='j3_range_diagnostic',
                                               amplitude=float(amplitude), telemetry=telemetry)

    def execute_j3_compensation_diagnostic(self, recipe, *, enable=False, verify_wire=None):
        """Execute only the explicitly admitted fixed range10 compensation recipe."""
        if type(enable) is not bool:
            raise TypeError('diagnostic enable must be an explicit bool')
        if not enable:
            return dict(dry_run=True, callbacks_invoked=0, hardware_used=False,
                        action_source='j3_compensation_diagnostic', executable=False)
        if not callable(verify_wire):
            raise RuntimeError('compensation requires existing-owner encoder binding verification')
        fixed = verify_wire(deepcopy(recipe))
        if not isinstance(fixed, dict) or not isinstance(fixed.get('wire'), dict):
            raise RuntimeError('compensation encoder binding verification returned no fixed recipe')
        from .v21_j3_compensation import validate_j3_compensation_recipe
        validate_j3_compensation_recipe(fixed)
        source = np.asarray(fixed['source_q'], dtype=np.float32)
        sequence = np.asarray(fixed['wire']['commands'], dtype=np.float32)
        if sequence.shape != (5, 6) or not np.isfinite(sequence).all():
            raise ValueError('compensation requires finite encoder-bound commands [5,6]')
        return self._execute_fixed_j3_diagnostic(sequence, source,
            action_source='j3_compensation_diagnostic', compensation_recipe=fixed)

    def _execute_fixed_j3_diagnostic(self, sequence, source, *, action_source,
                                     amplitude=None, telemetry=None, compensation_recipe=None):
        if (self.maximum_lateness_s != .03 or self.normal_follow_budget_s != .4
                or self.verify_policy_goal is None):
            raise RuntimeError('diagnostic requires original 30ms/.4s and per-write Goal verification')
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('hardware transaction already active')
        result = dict(completed=False, action_source=action_source,
            not_model_transition=True, model_feedback_label=False,
            physical_grasp_verified=False, requires_home_before_policy_resume=True,
            maximum_lateness_s=self.maximum_lateness_s,
            policy_commands_attempted=0, policy_commands_returned=0,
            commands_position=[], command_times_s=[], readback_times_s=[], measured_q_path=[],
            policy_goal_readbacks=[], policy_goal_reads_attempted=0,
            policy_goal_reads_returned=0, policy_goal_registers_verified=0)
        if action_source == 'j3_range_diagnostic':
            result.update(diagnostic_profile='j3-range-10x10hz-v1', diagnostic_command_count=10,
                          diagnostic_total_amplitude=amplitude,
                          command_interval_s=.1, final_readback_offset_s=1.,
                          position_unit='lerobot_calibrated_position', gain_writes=0)
            if telemetry is not None:
                result['dynamic_actuator_readbacks'] = []
        elif action_source == 'j3_native_response_diagnostic':
            result.update(diagnostic_profile='j3-native-response-5x10hz-v1',
                diagnostic_command_count=5, diagnostic_total_amplitude=3.,
                command_interval_s=.1, final_readback_offset_s=.5,
                position_unit='lerobot_calibrated_position', gain_writes=0,
                automatic_retry=False)
        elif action_source == 'j3_compensation_diagnostic':
            result.update(diagnostic_profile='j3-compensation-5x10hz-v1', diagnostic_command_count=5,
                diagnostic_total_amplitude=12.3, original_target=deepcopy(compensation_recipe['original_target']),
                compensated_goal=deepcopy(compensation_recipe['compensated_goal']),
                old_source=deepcopy(compensation_recipe['old_source']),
                compensation_recipe=deepcopy(compensation_recipe), command_interval_s=.1,
                final_readback_offset_s=.5, position_unit='lerobot_calibrated_position', gain_writes=0,
                automatic_retry=False)
        self.last_attempt = result
        self.last_source_state = self.last_forecast = self.last_actor_input = None
        admitted = False
        try:
            if (not self.runtime.stopped or self.runtime.pending is not None
                    or getattr(self.runtime, 'executing', False) or self.cancelled()):
                raise RuntimeError('diagnostic requires stopped model and separate uncancelled admission')
            admitted = True
            if self.read_pre_dispatch_state is not None:
                result['pre_dispatch_actuator_state'] = self.read_pre_dispatch_state()
            if self.cancelled():
                raise RuntimeError('diagnostic cancelled before geometry admission')
            from .v21_lewm_cem import passes_single_axis_stall
            q, _ = self._read()
            if not np.allclose(q, source, atol=1e-4, rtol=0):
                raise RuntimeError('diagnostic source changed before geometry admission')
            for start in range(0, len(sequence), 5):
                # Each existing gate accepts exactly five commands. Every block
                # is also checked from the original actual source, covering a
                # J3 that has failed to follow any of the earlier commands.
                if not passes_single_axis_stall(self.physical_check,
                        torch.from_numpy(q.copy())[None], torch.from_numpy(sequence[start:start+5].copy())[None]):
                    raise RuntimeError('diagnostic rejected by original geometry or single-axis-stall gate')
            result['original_geometry_admitted'] = True
            q0, epoch = self._read()
            if not np.allclose(q0, source, atol=1e-4, rtol=0):
                raise RuntimeError('diagnostic source changed before first command')
            result.update(source_time_s=epoch, source_q_position=q0.tolist(),
                          source_q_readback_time_s=epoch, source_q_age_at_dispatch_epoch_s=0.)
            if action_source == 'j3_range_diagnostic':
                self._transport_commands(sequence, q0, epoch, result, final_offset_s=1.,
                                         hold_key='protective_hold_after_q10', telemetry=telemetry)
            else:
                self._transport_five(sequence, q0, epoch, result)
            if self.cancelled():
                raise RuntimeError('diagnostic cancelled after completion hold')
            endpoint, endpoint_time = self._read()
            result.update(endpoint_q_position=endpoint.tolist(), endpoint_time_s=endpoint_time)
            if self.cancelled():
                raise RuntimeError('diagnostic cancelled during final endpoint readback')
            # Completion HOLD has already run. The admitted protocol asserts
            # STOP in finally; asserting its cancellation token here would make
            # every successful receipt look like an operator cancellation.
            # Never call runtime.complete or change the factual model history.
            self.runtime.stop()
            result['completed'] = True
            return result
        except BaseException as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
            if result['policy_commands_attempted']:
                self.runtime.requires_home_after_incomplete_execution = True
                result['requires_home_after_incomplete_execution'] = True
            if admitted:
                try:
                    result['emergency_hold'] = self._freeze_then_cleanup(result)
                except BaseException as hold_error:
                    result['diagnostic_cleanup_error'] = f'{type(hold_error).__name__}: {hold_error}'
            exc.last_attempt = deepcopy(result)
            raise
        finally:
            self.lock.release()

    def execute(self):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("hardware transaction already active")
        self.last_attempt = dict(completed=False, policy_commands_attempted=0, policy_commands_returned=0,
                                 commands_position=[], command_times_s=[], readback_times_s=[], measured_q_path=[],
                                 maximum_lateness_s=self.maximum_lateness_s)
        self.last_source_state = None
        self.last_forecast = None
        self.last_actor_input = None
        result = self.last_attempt
        if self.verify_policy_goal is not None:
            result.update(policy_goal_readbacks=[],policy_goal_reads_attempted=0,
                          policy_goal_reads_returned=0,policy_goal_registers_verified=0)
        forecast = None
        try:
            if self.runtime.pending is None or self.runtime.stopped or self.cancelled():
                raise RuntimeError("fresh planned transaction required")
            if self.read_pre_dispatch_state is not None:
                result['pre_dispatch_actuator_state']=self.read_pre_dispatch_state()
                if self.cancelled():
                    raise RuntimeError('five-command transaction cancelled during pre-dispatch observation')
            q, _ = self._read()
            rebases=self._rebase_before_epoch(q)
            if rebases:
                result['pre_dispatch_rebases']=rebases
                q, _ = self._read()
            expected = self.runtime.pending.trace.positions.detach().cpu().numpy()[0]
            self.physical_check(q, expected)
            q0, epoch = self._read()
            result['source_q_readback_time_s'] = epoch
            planned_source = getattr(self.runtime.pending, 'source_time', None)
            ready_time = getattr(self.runtime, 'pending_ready_time_s', None)
            if planned_source is not None:
                result['planning_timing'] = dict(
                    source_time_s=float(planned_source.item()),
                    ready_time_s=ready_time,
                    source_age_at_pre_dispatch_s=float(self.clock())-float(planned_source.item()),
                    post_planning_wait_s=None if ready_time is None else float(self.clock())-ready_time,
                    source_age_limit_s=getattr(self.runtime, 'max_plan_age_s', None),
                    post_planning_wait_limit_s=getattr(self.runtime, 'max_plan_dwell_s', None))
            device = self.runtime.pending.source_q.device
            commands = self.runtime.begin_execution(torch.as_tensor(q0, device=device)[None],
                torch.tensor([epoch], device=device, dtype=torch.float64))
            if getattr(self.runtime, 'pending_action_source', None) == 'v21_lewm_cem':
                epoch = float(self.clock())
                stamp = torch.tensor([epoch], device=device, dtype=torch.float64)
                self.runtime.pending = replace(self.runtime.pending, source_time=stamp)
                self.runtime.state = replace(self.runtime.state, time=stamp)
            # Keep the immutable actual source, not the later completion state.
            # Optional disk serialization happens only after all motor writes/HOLD.
            self.last_source_state = getattr(self.runtime, 'state', None)
            self.last_actor_input = getattr(self.runtime, 'last_actor_input', None)
            forecast = self.runtime.pending
            self.last_forecast = forecast
            update=getattr(self.runtime,'last_dispatch_update',None)
            if update is not None:
                result['pre_dispatch_state_update']=dict(update)
            # Legacy runtime adapters may construct a pending forecast without
            # the explicit source field. Their command path is the original
            # Actor path; current planners still provide and retain their
            # explicit source labels.
            action_source=getattr(self.runtime,'pending_action_source',None) or 'v21_actor'
            result.update(source_time_s=epoch, source_q_position=q0.tolist(),
                          source_q_age_at_dispatch_epoch_s=epoch-result['source_q_readback_time_s'],
                          action_source=action_source,
                          planning_observation_source=self.last_plan_observation_source)
            self._transport_five(commands.detach().cpu().numpy()[0],q0,epoch,result)
            # Log-only GPU transfers must not consume the first motor deadline.
            # Keep the exact original forecast; no recomputation or clock shift.
            result['predicted_q_path'] = forecast.response.predicted_q_path.detach().cpu().tolist()
            result['commanded_offsets_s'] = forecast.trace.offsets.detach().cpu().tolist()
            if self.cancelled():raise RuntimeError('five-command transaction cancelled after completion hold')
            endpoint = self.capture_endpoint()
            if self.cancelled():raise RuntimeError('five-command transaction cancelled during endpoint capture')
            result['endpoint_q_position'] = endpoint.q[0].detach().cpu().tolist()
            result['endpoint_time_s'] = endpoint.time.item()
            if self.extra_readback_check is not None:
                self._check_extra_readback(endpoint.q[0].detach().cpu().numpy())
            endpoint_goal=np.asarray(result['commands_position'][-1],dtype=np.float64)
            endpoint_actual=np.asarray(result['endpoint_q_position'],dtype=np.float64)
            if endpoint_actual.shape!=(6,) or not np.isfinite(endpoint_actual).all():
                raise ValueError('original-target endpoint assessment requires finite actual [6]')
            endpoint_residual=endpoint_actual-endpoint_goal
            endpoint_error=float(np.abs(endpoint_residual).max())
            held=result['completion_hold']
            result['endpoint_original_target_position']=dict(
                schema='v21-endpoint-original-target-position-v1',
                source='actual_endpoint_after_completion_hold',sample_count=1,
                original_target=endpoint_goal.tolist(),actual_position=endpoint_actual.tolist(),
                endpoint_time_s=result['endpoint_time_s'],residual_by_axis=endpoint_residual.tolist(),
                max_abs_residual=endpoint_error,residual_convention='actual_position minus frozen original_target',
                position_tolerance=.2,position_within_tolerance=bool(endpoint_error<=.2),
                strict_arrival_assessed=False,model_feedback_label=False,
                original_goal_retained_through_completion=bool(
                    held.get('reason')=='normal_completion_retained_fifth_goal'
                    and held.get('goal_register_verified') is True
                    and held.get('reused_existing_hold') is True
                    and held.get('goal_register_writes')==0))
            if (result.get('position_servo',{}).get('original_target_reached') is True
                    or result.get('position_servo',{}).get('applied') is True
                    or result.get('position_servo',{}).get('selected_axes_held') is True):
                from .v21_hardware_state import ORDER, position_servo_readback_proof
                goal=endpoint_goal
                actual=endpoint_actual
                finish_follow=result['position_servo']['completion_follow']
                hold_actual=result['completion_hold'].get('actual_readback',{}).get('present_position')
                if 'selected_axes' in finish_follow and (
                        not isinstance(hold_actual,dict) or set(hold_actual)!=set(ORDER)):
                    raise RuntimeError('selected-axis finishing lacks the actual completion HOLD readback')
                previous_actual=None if hold_actual is None else [hold_actual[motor] for motor in ORDER]
                proof=position_servo_readback_proof(finish_follow,goal,actual,previous_actual=previous_actual)
                fresh=bool(np.isfinite(result['endpoint_time_s'])
                           and finish_follow['times_s'][-1]<=result['endpoint_time_s']<=float(self.clock()))
                verified=bool(proof['verified'] and fresh)
                result['position_servo_endpoint_validation']=dict(proof,
                    endpoint_time_s=result['endpoint_time_s'],fresh_after_finishing=fresh,verified=verified)
                if not verified:
                    raise RuntimeError('position-servo arrival changed during endpoint capture')
            actual_trace = CommandTrace(torch.tensor(result["commands_position"], device=device,
                dtype=commands.dtype)[None], torch.tensor(np.asarray(result["command_times_s"])-epoch,
                device=device, dtype=torch.float64)[None], torch.ones(1, 5, device=device, dtype=torch.bool),
                "recorded", True)
            following=result['normal_follow']
            if following['positions']:
                # Extra following is real telemetry, not a replacement for the
                # original fifth model readback. Keep its separate clock.
                result['following_q_path']=result['measured_q_path'][:-1]+[following['positions'][-1]]
                result['following_readback_times_s']=result['readback_times_s'][:-1]+[following['times_s'][-1]]
            result['model_feedback_path']='original_five_step_readback'
            if self.cancelled():raise RuntimeError('five-command transaction cancelled before model completion')
            status = self.runtime.complete(actual_trace,
                torch.tensor(result["measured_q_path"], device=device, dtype=commands.dtype)[None],
                torch.tensor(result["readback_times_s"], device=device, dtype=torch.float64)[None], endpoint)
            if self.cancelled():raise RuntimeError('five-command transaction cancelled during model completion')
            result.update(completed=True, endpoint_time_s=endpoint.time.item(),
                          feedback_retimed=status["feedback_retimed"], history_chunks=status["history_chunks"])
            self.runtime.stop()
            return result
        except BaseException as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            invalidate_finishing_arrival(result,result['error'])
            if result['policy_commands_attempted'] > 0 and not result['completed']:
                # Actual partial motion cannot be represented as a completed
                # five-step history entry or ignored by the next plan.
                self.runtime.requires_home_after_incomplete_execution = True
                result['requires_home_after_incomplete_execution'] = True
            result['emergency_hold']=self._freeze_then_cleanup(result)
            result.setdefault('actuator_events',[]).extend(result['emergency_hold'].get('actuator_events',[]))
            # Preserve the dispatch prediction after the physical freeze, even
            # when only part of the block was sent. Never fill missing actual
            # samples with this forecast or synchronise CUDA inside deadlines.
            if forecast is not None:
                try:
                    result['predicted_q_path'] = forecast.response.predicted_q_path.detach().cpu().tolist()
                    result['commanded_offsets_s'] = forecast.trace.offsets.detach().cpu().tolist()
                    result['prediction_scope'] = 'original_complete_plan_not_complete_execution'
                    query = getattr(forecast, 'query_offsets', None)
                    if query is not None:
                        result['predicted_readback_offsets_s'] = query.detach().cpu().tolist()
                except Exception as logging_error:
                    result['prediction_logging_error'] = f'{type(logging_error).__name__}: {logging_error}'
            raise
        finally:
            self.lock.release()

    def _freeze_then_cleanup(self, errors):
        """Physical failure stays primary if model cleanup also fails."""
        self.request_stop()
        physical_failure=None
        try:
            started=self.clock()
            try:
                held=self.halt()
            except BaseException as exc:
                physical_failure=exc
                errors['physical_hold_error']=f'{type(exc).__name__}: {exc}'
                raise
            return completion_hold_record(held,started,self.clock())
        finally:
            try:
                self.runtime.stop()  # Never wait for model/GPU cleanup before the physical freeze.
            except BaseException as exc:
                errors['runtime_cleanup_error']=f'{type(exc).__name__}: {exc}'
                if physical_failure is None:raise

    def stop(self):
        errors={}
        self.last_stop_errors=errors
        self.last_stop_record=None
        self.last_stop_record=self._freeze_then_cleanup(errors)
        return self.last_stop_record
