"""Fake clocks/readbacks only: optional finishing never relabels model y."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.v21_position_arrival import evaluate_position_arrival
from test_v21_follow_arrival_independent import assert_original_y, full_window_bridge
from test_v21_hardware_bridge import setup


def report_from_reads(bridge, execution, *, applied=True, samples=9):
    positions=[];times=[]
    for _ in range(samples):
        bridge.wait(.05)
        positions.append(bridge.read_q().tolist())
        times.append(bridge.clock())
    goal=execution['commands_position'][-1]
    arrival=evaluate_position_arrival(goal,positions,np.asarray(times)-execution['source_time_s'],
        position_tolerance=.2,step_tolerance=.1,quiet_samples=3)
    follow=dict(schema='v21-position-servo-completion-follow-v1',positions=positions,times_s=times,
        source_time_s=execution['source_time_s'],original_target=deepcopy(goal),
        baseline_restored=True,hold_eligible=True,goal_reached=arrival['reached'])
    return dict(schema='v21-j3-position-servo-v1',completed=True,applied=applied,
        cleanup=dict(restored=True),original_goal_preserved=True,model_feedback_label=False,
        original_target_arrival=arrival,original_target_reached=arrival['reached'],
        completion_follow=follow)


def successful_bridge(*, after=-5.8, before=-3.2, applied=True, samples=9):
    bridge,runtime,writes,holds=full_window_bridge(lambda elapsed: before if elapsed<=2.+1e-8 else after)
    observed=[]
    def finish(execution):
        assert bridge.lock.locked() and not runtime.stopped
        assert execution['completed'] is False
        assert len(writes)==5 and len(execution['measured_q_path'])==6
        assert len(execution['normal_follow']['positions'])==40
        assert bridge.clock()==pytest.approx(12.5)
        observed.append(deepcopy(execution))
        return report_from_reads(bridge,execution,applied=applied,samples=samples)
    def retain(goal,follow,q5):
        from fcpg.v21_hardware_state import ORDER
        holds.append(bridge.clock())
        observed.append(dict(goal=deepcopy(goal),follow=deepcopy(follow),q5=q5.copy()))
        return dict(event='safe_hold_verified',reused_existing_hold=True,motor_register_writes=0,
            actual_readback=dict(present_position=dict(zip(ORDER,bridge.read_q().tolist()))),
            attempts=[dict(held_state=deepcopy(goal),mismatched_joints={})])
    bridge.finish_position=finish
    bridge.completion_hold=retain
    return bridge,runtime,writes,holds,observed


def arm_report(report):
    from fcpg.v21_hardware_state import ORDER
    from fcpg.v21_position_arrival import evaluate_selected_position_arrival
    baseline={motor:dict(P_Coefficient=16,I_Coefficient=0,D_Coefficient=0,
        Torque_Limit=500 if motor=='gripper' else 1000,
        Max_Torque_Limit=500 if motor=='gripper' else 1000) for motor in ORDER}
    touched=['elbow_flex','wrist_flex'] if report['applied'] else []
    parameters=[('Torque_Limit',200),('D_Coefficient',32),('Torque_Limit',300),('I_Coefficient',1),
        ('I_Coefficient',0),('Torque_Limit',200),('D_Coefficient',0),('Torque_Limit',1000)]
    report.update(schema='v21-arm-position-servo-v1',baseline_parameters_by_motor=baseline,
        final_parameters_by_motor=deepcopy(baseline),touched_motors=touched,final_equals_baseline=True,
        parameter_writes=[dict(motor=motor,register=key,value=value,may_have_applied=True,verified=True)
            for motor in touched for key,value in parameters])
    report['cleanup']['parameters_restored_by_motor']={motor:True for motor in touched}
    if report['original_target_reached']:
        follow=report['completion_follow']
        raw={motor:int(round(10000+100*q)) for motor,q in zip(ORDER,follow['original_target'])}
        epoch=follow['times_s'][0]-.05
        rows=[]
        for index,(q,t) in enumerate(zip(follow['positions'],follow['times_s']),1):
            columns=('Present_Position','Goal_Position','Moving','Torque_Enable','Present_Load',
                'P_Coefficient','I_Coefficient','D_Coefficient','Torque_Limit')
            register_reads={name:dict(started_s=t,returned_s=t,num_retry=1,
                operation='sync_read' if name in columns[:5] else 'read') for name in columns}
            rows.append(dict(phase='all_axes_original_goal_hold',q_position=q,
                scheduled_start_s=t,started_s=t,readback_time_s=t,returned_s=t,
                start_lateness_s=0.,
                sample_index=index,snapshot_attempt=1,schedule_basis='previous_sample_completion_plus_0.05s',
                completed=True,discarded=False,register_snapshot_complete=True,register_reads=register_reads,
                all_present_load_raw=dict.fromkeys(ORDER,0),goal_raw=deepcopy(raw),moving=dict.fromkeys(ORDER,0)))
        arrival=evaluate_position_arrival(follow['original_target'],follow['positions'],
            np.asarray(follow['times_s'])-follow['source_time_s'],
            position_tolerance=.2,step_tolerance=.1,quiet_samples=22)
        selected=evaluate_selected_position_arrival(follow['original_target'],follow['positions'],
            np.asarray(follow['times_s'])-follow['source_time_s'],selected_axes=[2,3,1],
            position_tolerance=.2,step_tolerance=.1,quiet_samples=22)
        contract=dict(selected_axes=[2,3,1],deferred_axes=[],
            deferred_anchor_q=deepcopy(follow['positions'][0]),deferred_position_tolerance=.25,
            source_q0=[0.]*6)
        report.update(contract)
        follow.update(contract,selected_axes_arrival=selected)
        report.update(original_target_raw=raw,original_goal_held=True,
            selected_axes_completed=True,selected_axes_held=True,
            admission=dict(current_q=deepcopy(contract['deferred_anchor_q'])),
            original_goal_hold=dict(readbacks=rows,sample_count=len(rows),completed=True,epoch_s=epoch,
                period_s=.05,scheduler_lateness_notice_s=.03,
                scheduler_lateness_policy='record_and_continue_with_fresh_actual_checks',
                scheduler_lateness_events=[],maximum_observed_start_lateness_s=0.,
                maximum_snapshot_attempts=2,
                retry_policy='complete_snapshot_reacquisition_after_sync_read_connection_error',
                snapshot_attempts=deepcopy(rows),
                schedule_basis='previous_sample_completion_plus_0.05s',
                requested_minimum_span_s=1.,actual_span_s=follow['times_s'][-1]-follow['times_s'][0],
                arrival=arrival,selected_axes_arrival=selected))
    return report


@pytest.fixture(scope='module')
def bounded_pulse_report():
    from test_v21_arm_position_servo import stationary_release_fixture, run
    owner,execution,_=stationary_release_fixture()
    return run(owner,execution)


@pytest.mark.parametrize('damage',[None,'missing_i0','late_i0','nonfinite_i0','overlap',
    'false_pulse_limit','false_total_limit','wrong_duration','wrong_total','wrong_maximum',
    'wrong_count','missing_last_i0','ninth_pulse','missing_i0_write','extra_i1_write'])
def test_bridge_validates_each_release_pulse_and_cumulative_receipt(bounded_pulse_report,damage):
    from fcpg.v21_hardware_bridge import validate_arm_finishing_restoration
    report=deepcopy(bounded_pulse_report)
    exposure=report['axes']['elbow_flex']['integral_exposure'];pulses=exposure['pulses']
    assert len(pulses)==3
    if damage=='missing_i0':pulses[1].pop('i0_verified_s')
    elif damage=='late_i0':pulses[1]['i0_verified_s']=pulses[1]['started_s']+2.01
    elif damage=='nonfinite_i0':pulses[1]['i0_verified_s']=float('nan')
    elif damage=='overlap':pulses[1]['started_s']=pulses[0]['i0_verified_s']-.01
    elif damage=='false_pulse_limit':pulses[1]['within_limit']=False
    elif damage=='false_total_limit':exposure['within_limit']=False
    elif damage=='wrong_duration':pulses[1]['duration_s']+=.01
    elif damage=='wrong_total':exposure['duration_s']+=.01
    elif damage=='wrong_maximum':pulses[1]['maximum_s']=10.
    elif damage=='wrong_count':exposure['pulse_count']=2
    elif damage=='missing_last_i0':exposure.pop('i0_verified_s')
    elif damage=='ninth_pulse':pulses.extend(deepcopy(pulses)*2)
    elif damage in ('missing_i0_write','extra_i1_write'):
        writes=report['parameter_writes']
        index=next(i for i,row in enumerate(writes) if row['register']=='I_Coefficient' and row['value']==0)
        if damage=='missing_i0_write':writes.pop(index)
        else:writes.insert(index,deepcopy(writes[index-1]))
    if damage is None:
        validate_arm_finishing_restoration(report)
    else:
        with pytest.raises(RuntimeError):validate_arm_finishing_restoration(report)


def add_arm_hold_retry(report):
    from fcpg.v21_position_arrival import evaluate_selected_position_arrival
    hold=report['original_goal_hold'];rows=hold['readbacks'];follow=report['completion_follow'];delay=.06
    for row in rows:
        for key in ('scheduled_start_s','started_s','readback_time_s','returned_s'):row[key]+=delay
        for event in row['register_reads'].values():
            event['started_s']+=delay;event['returned_s']+=delay
    follow['times_s']=[stamp+delay for stamp in follow['times_s']]
    relative=np.asarray(follow['times_s'])-follow['source_time_s']
    hold['arrival']=evaluate_position_arrival(follow['original_target'],follow['positions'],relative,
        position_tolerance=.2,step_tolerance=.1,quiet_samples=22)
    report['original_target_arrival']=evaluate_position_arrival(
        follow['original_target'],follow['positions'],relative,
        position_tolerance=.2,step_tolerance=.1,quiet_samples=3)
    report['original_target_reached']=report['original_target_arrival']['reached']
    selected=evaluate_selected_position_arrival(follow['original_target'],follow['positions'],relative,
        selected_axes=[2,3,1],position_tolerance=.2,step_tolerance=.1,quiet_samples=22)
    hold['selected_axes_arrival']=follow['selected_axes_arrival']=selected
    success=rows[0];success.update(snapshot_attempt=2,
        schedule_basis='previous_attempt_completion_plus_0.05s')
    failed=deepcopy(success);scheduled=success['scheduled_start_s']-delay
    failed.update(snapshot_attempt=1,schedule_basis='previous_sample_completion_plus_0.05s',
        scheduled_start_s=scheduled,started_s=scheduled,finished_s=scheduled+.01,
        completed=False,discarded=True,failed=True,retry_started=True,
        failed_register='Moving',register_snapshot_complete=False)
    failed.pop('returned_s');failed.pop('all_present_load_raw')
    failed['register_reads']={key:deepcopy(success['register_reads'][key])
        for key in ('Present_Position','Goal_Position','Moving')}
    for event in failed['register_reads'].values():
        event.update(started_s=scheduled,returned_s=scheduled+.001)
    failed['register_reads']['Moving'].pop('returned_s')
    failed['register_reads']['Moving'].update(failed_s=failed['finished_s'],
        error='_DynamicSyncReadConnectionError: injected timeout')
    hold['snapshot_attempts']=[failed,*deepcopy(rows)]
    return failed,hold['snapshot_attempts'][1]


def add_arm_hold_scheduler_pause(report, *, index=5, delay=.577):
    from fcpg.v21_position_arrival import evaluate_selected_position_arrival
    hold=report['original_goal_hold'];follow=report['completion_follow']
    for collection in (hold['readbacks'],hold['snapshot_attempts']):
        for offset,row in enumerate(collection):
            if offset<index:continue
            if offset>index:row['scheduled_start_s']+=delay
            for key in ('started_s','readback_time_s','returned_s'):row[key]+=delay
            for event in row['register_reads'].values():
                event['started_s']+=delay;event['returned_s']+=delay
            row['start_lateness_s']=row['started_s']-row['scheduled_start_s']
    follow['times_s'][index:]=[stamp+delay for stamp in follow['times_s'][index:]]
    relative=np.asarray(follow['times_s'])-follow['source_time_s']
    hold['actual_span_s']=follow['times_s'][-1]-follow['times_s'][0]
    hold['arrival']=evaluate_position_arrival(follow['original_target'],follow['positions'],relative,
        position_tolerance=.2,step_tolerance=.1,quiet_samples=22)
    report['original_target_arrival']=evaluate_position_arrival(
        follow['original_target'],follow['positions'],relative,
        position_tolerance=.2,step_tolerance=.1,quiet_samples=3)
    report['original_target_reached']=report['original_target_arrival']['reached']
    selected=evaluate_selected_position_arrival(follow['original_target'],follow['positions'],relative,
        selected_axes=[2,3,1],position_tolerance=.2,step_tolerance=.1,quiet_samples=22)
    hold['selected_axes_arrival']=follow['selected_axes_arrival']=selected
    row=hold['snapshot_attempts'][index]
    event=dict(phase=row['phase'],sample_index=index+1,snapshot_attempt=1,
        scheduled_start_s=row['scheduled_start_s'],observed_start_s=row['started_s'],
        observed_lateness_s=delay,notice_threshold_s=.03,snapshot_started=True,
        scheduler_lateness_only=True)
    hold['scheduler_lateness_events']=[event]
    hold['maximum_observed_start_lateness_s']=delay
    return event


@pytest.mark.parametrize('applied',[False,True])
def test_arm_finishing_preserves_each_motor_baseline_and_original_feedback(applied):
    bridge,runtime,writes,holds,_=successful_bridge(after=-5.8 if applied else -3.2,applied=applied,samples=22)
    callback=bridge.finish_position
    bridge.finish_position=lambda execution:arm_report(callback(execution))
    result=bridge.execute();servo=result['position_servo']
    assert result['completed'] and result['original_target_reached'] is applied
    assert servo['schema']=='v21-arm-position-servo-v1'
    assert servo['final_parameters_by_motor']['gripper']['Torque_Limit']==500
    assert 'gripper' not in servo['cleanup']['parameters_restored_by_motor']
    assert bool(servo['parameter_writes']) is applied
    assert len(writes)==5 and len(holds)==1
    assert_original_y(result,runtime)


@pytest.mark.parametrize('damage',['missing_baseline','missing_final','bool_parameter','false_equal',
    'gripper_1000','i_not_cleared','missing_restore','false_restore','invented_gripper_restore',
    'duplicate_touched','out_of_order_touched','gripper_touched','no_touched','missing_writes','unverified_write',
    'goal_write','unknown_axis_write','unknown_schema','lower_cap_wrong_motor',
    'lower_cap_without_profile','missing_pulse_evidence','claimed_speed_bound'])
def test_arm_finishing_rejects_missing_or_false_axis_restoration_proof(damage):
    bridge,runtime,writes,holds,_=successful_bridge(samples=22);callback=bridge.finish_position
    def damaged(execution):
        report=arm_report(callback(execution))
        if damage=='missing_baseline':report['baseline_parameters_by_motor'].pop('gripper')
        elif damage=='missing_final':report['final_parameters_by_motor'].pop('wrist_flex')
        elif damage=='bool_parameter':report['final_parameters_by_motor']['gripper']['I_Coefficient']=False
        elif damage=='false_equal':report['final_equals_baseline']=False
        elif damage=='gripper_1000':report['final_parameters_by_motor']['gripper']['Torque_Limit']=1000
        elif damage=='i_not_cleared':report['final_parameters_by_motor']['wrist_flex']['I_Coefficient']=1
        elif damage=='missing_restore':report['cleanup']['parameters_restored_by_motor'].pop('wrist_flex')
        elif damage=='false_restore':report['cleanup']['parameters_restored_by_motor']['elbow_flex']=False
        elif damage=='invented_gripper_restore':report['cleanup']['parameters_restored_by_motor']['gripper']=True
        elif damage=='duplicate_touched':report['touched_motors'].append('elbow_flex')
        elif damage=='out_of_order_touched':
            report['touched_motors'].reverse()
            report['parameter_writes'].sort(key=lambda row:report['touched_motors'].index(row['motor']))
        elif damage=='gripper_touched':report['touched_motors'].append('gripper')
        elif damage=='no_touched':report['touched_motors']=[]
        elif damage=='missing_writes':report['parameter_writes']=[]
        elif damage=='unverified_write':report['parameter_writes'][0]['verified']=False
        elif damage=='goal_write':report['parameter_writes'][0]['register']='Goal_Position'
        elif damage=='unknown_axis_write':report['parameter_writes'][0]['motor']='gripper'
        elif damage=='unknown_schema':report['schema']='v21-unknown-position-servo-v1'
        elif damage in ('lower_cap_wrong_motor','lower_cap_without_profile'):
            motor='wrist_flex' if damage=='lower_cap_wrong_motor' else 'elbow_flex'
            for event in report['parameter_writes']:
                if event['motor']==motor and (event['register'],event['value'])==('Torque_Limit',300):
                    event['value']=150
        elif damage in ('missing_pulse_evidence','claimed_speed_bound'):
            control=dict(mode='stationary_release_pulses_v1',maximum_pulses=8,
                maximum_pulse_s=2.,maximum_cumulative_s=10.,physical_speed_bound_verified=damage=='claimed_speed_bound')
            report.setdefault('axes',{}).setdefault('elbow_flex',{})['integral_control']=control
        return report
    bridge.finish_position=damaged
    with pytest.raises(RuntimeError):
        bridge.execute()
    assert not bridge.last_attempt['completed'] and not hasattr(runtime,'actual')
    assert bridge.last_attempt['position_servo']
    assert len(writes)==5 and len(holds)==1


def test_arm_skip_cannot_invent_restoration_or_conceal_parameter_writes():
    bridge,runtime,_,_,_=successful_bridge(after=-3.2,applied=False);callback=bridge.finish_position
    def concealed(execution):
        report=arm_report(callback(execution))
        report['parameter_writes']=[dict(motor='elbow_flex',register='I_Coefficient',value=0,
            may_have_applied=True,verified=True)]
        return report
    bridge.finish_position=concealed
    with pytest.raises(RuntimeError,match='actual parameter writes'):
        bridge.execute()
    assert not hasattr(runtime,'actual')


@pytest.mark.parametrize('damage',['short','missed_early_sample','different_goal','missing_hold','fake20hz'])
def test_arm_success_cannot_hide_an_invalid_full_second_behind_three_good_tail_samples(damage):
    bridge,runtime,_,_,_=successful_bridge(samples=22);callback=bridge.finish_position
    def bad_hold(execution):
        report=arm_report(callback(execution));hold=report['original_goal_hold']
        if damage=='short':hold['actual_span_s']=.95
        elif damage=='missed_early_sample':
            report['completion_follow']['positions'][0][2]+=.3
            hold['readbacks'][0]['q_position'][2]+=.3
        elif damage=='different_goal':hold['readbacks'][0]['goal_raw']['elbow_flex']+=1
        elif damage=='missing_hold':report.pop('original_goal_hold')
        elif damage=='fake20hz':hold['readbacks'][1]['scheduled_start_s']-=.01
        return report
    bridge.finish_position=bad_hold
    with pytest.raises((RuntimeError,ValueError)):bridge.execute()
    assert not hasattr(runtime,'actual') and bridge.last_attempt['original_target_reached'] is False


@pytest.mark.parametrize('damage',[None,'third_attempt','retry_clock','partial_evidence','failed_as_success'])
def test_arm_hold_retry_receipt_is_bounded_and_failed_attempt_is_not_a_sample(damage):
    bridge,runtime,_,_,_=successful_bridge(samples=22);callback=bridge.finish_position
    def retried(execution):
        report=arm_report(callback(execution));failed,success=add_arm_hold_retry(report)
        bridge.wait(.06)
        attempts=report['original_goal_hold']['snapshot_attempts']
        if damage=='third_attempt':
            third=deepcopy(success);third['snapshot_attempt']=3;attempts.insert(2,third)
        elif damage=='retry_clock':success['scheduled_start_s']+=.001
        elif damage=='partial_evidence':failed['register_reads']['Moving'].pop('failed_s')
        elif damage=='failed_as_success':report['original_goal_hold']['readbacks'][0]=deepcopy(failed)
        return report
    bridge.finish_position=retried
    if damage is None:
        result=bridge.execute()
        assert result['completed'] and len(result['position_servo']['original_goal_hold']['readbacks'])==22
    else:
        with pytest.raises(RuntimeError,match='hold|samples'):
            bridge.execute()
        assert not hasattr(runtime,'actual')


def test_arm_hold_accepts_one_receipted_non_realtime_scheduler_pause():
    bridge,runtime,_,_,_=successful_bridge(samples=22);callback=bridge.finish_position
    def delayed(execution):
        report=arm_report(callback(execution));add_arm_hold_scheduler_pause(report)
        bridge.wait(.577)
        return report
    bridge.finish_position=delayed
    result=bridge.execute();hold=result['position_servo']['original_goal_hold']
    assert result['completed'] and hold['completed']
    assert hold['scheduler_lateness_events'][0]['observed_lateness_s']==pytest.approx(.577)
    assert hold['maximum_observed_start_lateness_s']==pytest.approx(.577)
    assert hold['actual_span_s']>=1.577


def test_arm_hold_rejects_scheduler_pause_without_matching_audit_receipt():
    bridge,runtime,_,_,_=successful_bridge(samples=22);callback=bridge.finish_position
    def concealed(execution):
        report=arm_report(callback(execution));add_arm_hold_scheduler_pause(report)
        report['original_goal_hold']['scheduler_lateness_events']=[]
        bridge.wait(.577)
        return report
    bridge.finish_position=concealed
    with pytest.raises(RuntimeError,match='hold|samples'):bridge.execute()
    assert not hasattr(runtime,'actual')


def test_already_reached_then_finishing_failure_invalidates_only_the_final_claim():
    bridge,runtime,_,_,_=successful_bridge(before=-5.8,after=-5.8,applied=False)
    class HoldFailure(RuntimeError):pass
    def fail(execution):
        assert execution['original_target_reached'] is True
        bridge.wait(1.)
        error=HoldFailure('actual departure during full-second hold')
        error.report=dict(schema='v21-arm-position-servo-v1',completed=False,applied=False,
            original_target_reached=True,cleanup=dict(restored=False))
        raise error
    bridge.finish_position=fail
    with pytest.raises(HoldFailure):bridge.execute()
    attempt=bridge.last_attempt
    assert not attempt['completed'] and not attempt['original_target_reached']
    assert attempt['original_target_arrival']['verified'] is False
    assert attempt['position_servo_prior_arrival']['reached'] is True
    assert attempt['normal_follow']['original_target_arrival']['reached'] is True
    assert attempt['position_servo']['original_target_reached'] is True  # preserved partial report
    assert not hasattr(runtime,'actual')


def test_success_keeps_five_commands_original_y_and_full_follow_but_uses_fresh_endpoint():
    bridge,runtime,writes,holds,observed=successful_bridge()
    result=bridge.execute()
    assert result['completed'] and result['original_target_reached']
    assert result['position_servo']['applied'] is True
    assert len(writes)==5 and holds==[pytest.approx(12.95)]
    assert_original_y(result,runtime)
    np.testing.assert_allclose(result['command_times_s'],10.+np.arange(5)*.1,atol=1e-8,rtol=0)
    assert result['normal_follow']==observed[0]['normal_follow']
    assert result['normal_follow']['original_target_arrival']['reached'] is False
    assert result['normal_follow']['times_s'][-1]==pytest.approx(12.5)
    assert result['following_q_path'][-1][2]==pytest.approx(-3.2,abs=1e-6)
    assert result['endpoint_q_position'][2]==pytest.approx(-5.8,abs=1e-6)
    assert result['endpoint_time_s']==pytest.approx(12.95)
    assert observed[1]['follow']==result['position_servo']['completion_follow']
    assert observed[1]['q5'][2]==pytest.approx(-3.2,abs=1e-6)
    assert result['completion_hold']['goal_register_writes']==0
    assert runtime.feedback=={}


def test_endpoint_retains_actual_post_hold_error_without_relabeling_pre_hold_arrival():
    bridge,runtime,_,_,_=successful_bridge(before=-2.738113,after=-2.738113,applied=False)
    callback=bridge.finish_position
    bridge.finish_position=lambda execution:arm_report(callback(execution))
    capture=bridge.capture_endpoint
    def endpoint():
        observation=capture();observation.q[0,2]= -2.649381;return observation
    bridge.capture_endpoint=endpoint
    bridge.completion_hold=lambda goal,follow,q5:dict(event='safe_hold_verified',
        reason='v21_normal_completion_fallback',reused_existing_hold=False,motor_register_writes=1,
        attempts=[dict(held_state=follow['positions'][-1],mismatched_joints={})])
    result=bridge.execute();position=result['endpoint_original_target_position']
    assert result['completed'] and result['original_target_reached'] is False
    assert result['original_target_arrival']['max_abs_residual']==pytest.approx(3.061887,abs=1e-6)
    assert result['normal_follow']['original_target_arrival']['max_abs_residual']==pytest.approx(3.061887,abs=1e-6)
    assert position['max_abs_residual']==pytest.approx(3.150619,abs=1e-6)
    assert position['original_target']==result['commands_position'][-1]
    assert position['actual_position']==result['endpoint_q_position']
    assert position['position_within_tolerance'] is False
    assert position['sample_count']==1 and position['strict_arrival_assessed'] is False
    assert position['original_goal_retained_through_completion'] is False
    assert position['model_feedback_label'] is False and 'reached' not in position
    assert_original_y(result,runtime)


def test_callback_receives_copy_and_cannot_replace_original_feedback_or_forecast():
    bridge,runtime,writes,holds,observed=successful_bridge()
    callback=bridge.finish_position
    def mutate(execution):
        report=callback(execution)
        execution['commands_position'][-1][2]=88.
        execution['command_times_s'][-1]=99.
        execution['measured_q_path'][-1][2]=77.
        execution['readback_times_s'][-1]=98.
        execution['normal_follow']['positions'][-1][2]=66.
        return report
    bridge.finish_position=mutate
    result=bridge.execute()
    assert_original_y(result,runtime)
    assert result['commands_position'][-1][2]==pytest.approx(-5.8,abs=1e-6)
    assert result['command_times_s'][-1]==pytest.approx(10.4)
    assert result['normal_follow']==observed[0]['normal_follow']
    assert np.asarray(result['predicted_q_path']).sum()==0.


@pytest.mark.parametrize('before,after,reached',[(-5.8,-5.8,True),(-3.2,-3.2,False),(-5.8,-5.4,False)])
def test_no_tuning_outcomes_are_judged_from_fresh_actual_readbacks(before,after,reached):
    bridge,runtime,writes,holds,observed=successful_bridge(before=before,after=after,applied=False)
    result=bridge.execute()
    assert result['completed'] and result['original_target_reached'] is reached
    assert result['position_servo']['applied'] is False
    assert result['normal_follow']==observed[0]['normal_follow']
    assert result['normal_follow']['goal_reached'] is (before==-5.8)
    assert len(writes)==5 and len(holds)==1
    assert_original_y(result,runtime)


def test_callback_failure_keeps_partial_report_and_freezes_before_model_cleanup():
    bridge,runtime,writes,holds,_=successful_bridge();events=[]
    class ServoFailure(RuntimeError):
        report=dict(schema='v21-j3-position-servo-v1',completed=False,applied=True,
            cleanup=dict(restored=False),actual_positions=[[-3.2]*6])
    def fail(execution):raise ServoFailure('simulated restoration is uncertain')
    bridge.finish_position=fail
    bridge.request_stop=lambda:events.append('stop')
    bridge.halt=lambda:events.append('hold')
    original_stop=runtime.stop
    def stop():events.append('model_cleanup');original_stop()
    runtime.stop=stop
    with pytest.raises(ServoFailure,match='restoration'):
        bridge.execute()
    assert events==['stop','hold','model_cleanup']
    assert bridge.last_attempt['position_servo']==ServoFailure.report
    assert not bridge.last_attempt['completed'] and not hasattr(runtime,'actual')
    assert bridge.last_attempt['requires_home_after_incomplete_execution']
    assert len(writes)==5 and len(bridge.last_attempt['measured_q_path'])==6
    assert bridge.last_attempt['predicted_q_path']


@pytest.mark.parametrize('damage',['no_report','schema','completed','applied','restored','goal_preserved',
    'model_label','target','nan','stale','future','source_clock','arrival','false_applied_arrival'])
def test_malformed_or_unproven_finishing_never_commits_a_model_transition(damage):
    bridge,runtime,writes,holds,_=successful_bridge();callback=bridge.finish_position
    def damaged(execution):
        report=callback(execution)
        if damage=='no_report':return None
        if damage=='schema':report['schema']='unrecognized'
        elif damage=='completed':report['completed']=False
        elif damage=='applied':report['applied']=1
        elif damage=='restored':report['cleanup']['restored']=False
        elif damage=='goal_preserved':report['original_goal_preserved']=False
        elif damage=='model_label':report['model_feedback_label']=True
        elif damage=='target':report['completion_follow']['original_target'][2]-=1.
        elif damage=='nan':report['completion_follow']['positions'][-1][2]=float('nan')
        elif damage=='stale':report['completion_follow']['times_s']=[10.+i*.05 for i in range(9)]
        elif damage=='future':report['completion_follow']['times_s']=[20.+i*.05 for i in range(9)]
        elif damage=='source_clock':report['completion_follow']['source_time_s']=9.
        elif damage=='arrival':report['original_target_arrival']['max_abs_residual']=99.
        elif damage=='false_applied_arrival':
            follow=report['completion_follow'];follow['positions']=[[-3.2]*6 for _ in range(9)]
            report['original_target_arrival']=evaluate_position_arrival(follow['original_target'],
                follow['positions'],np.asarray(follow['times_s'])-follow['source_time_s'],
                position_tolerance=.2,step_tolerance=.1,quiet_samples=3)
            report['original_target_reached']=follow['goal_reached']=False
        return report
    bridge.finish_position=damaged
    with pytest.raises((RuntimeError,ValueError)):
        bridge.execute()
    assert not bridge.last_attempt['completed'] and not hasattr(runtime,'actual')
    assert runtime.stopped and len(writes)==5 and len(holds)==1
    assert bridge.last_attempt['position_servo_error']


def test_stop_arriving_during_finishing_preserves_report_without_model_completion():
    bridge,runtime,writes,holds,_=successful_bridge();callback=bridge.finish_position;cancel=[False]
    bridge.cancelled=lambda:cancel[0]
    def cancelled(execution):
        report=callback(execution);cancel[0]=True;return report
    bridge.finish_position=cancelled
    with pytest.raises(RuntimeError,match='cancelled after callback'):
        bridge.execute()
    assert bridge.last_attempt['position_servo']['completed'] is True
    assert not bridge.last_attempt['completed'] and not hasattr(runtime,'actual')
    assert len(writes)==5 and len(holds)==1


def test_successful_finishing_cannot_commit_if_completion_hold_changes_the_goal():
    bridge,runtime,writes,holds,_=successful_bridge()
    bridge.completion_hold=lambda goal,follow,q5:dict(event='safe_hold_verified',
        reused_existing_hold=False,motor_register_writes=1,
        attempts=[dict(held_state=q5,mismatched_joints={})])
    with pytest.raises(RuntimeError,match='retain the original goal'):
        bridge.execute()
    assert bridge.last_attempt['position_servo']['completed'] is True
    assert not bridge.last_attempt['completed'] and not hasattr(runtime,'actual')
    assert len(writes)==5 and len(holds)==1


@pytest.mark.parametrize('drift',[.15,.3])
def test_capture_endpoint_must_still_establish_precise_arrival(drift):
    bridge,runtime,writes,holds,_=successful_bridge();capture=bridge.capture_endpoint
    def departed():
        endpoint=capture();endpoint.q[0,2]+=drift;return endpoint
    bridge.capture_endpoint=departed
    with pytest.raises(RuntimeError,match='changed during endpoint capture'):
        bridge.execute()
    assert not bridge.last_attempt['completed'] and not hasattr(runtime,'actual')
    proof=bridge.last_attempt['position_servo_endpoint_validation']
    assert proof['verified'] is False
    assert bridge.last_attempt['original_target_reached'] is False
    assert bridge.last_attempt['original_target_arrival']['verified'] is False
    assert bridge.last_attempt['position_servo_prior_arrival']['reached'] is True
    assert proof['step_from_final_finishing_readback']==pytest.approx(drift,abs=1e-6)
    assert len(writes)==5 and len(holds)==2  # normal retention, then emergency freeze


@pytest.mark.parametrize('stamp',[12.5,99.,float('nan')])
def test_endpoint_must_be_a_real_later_readback_after_finishing(stamp):
    bridge,runtime,_,_,_=successful_bridge();capture=bridge.capture_endpoint
    def obsolete():
        endpoint=capture();endpoint.time.fill_(stamp);return endpoint
    bridge.capture_endpoint=obsolete
    with pytest.raises(RuntimeError,match='changed during endpoint capture'):
        bridge.execute()
    assert not bridge.last_attempt['position_servo_endpoint_validation']['fresh_after_finishing']
    assert not hasattr(runtime,'actual')


@pytest.mark.parametrize('source',['j3_pid_diagnostic','j3_range_diagnostic',
    'j3_compensation_diagnostic','j3_native_response_diagnostic'])
def test_diagnostic_transport_never_invokes_normal_position_finishing(source):
    bridge,_,writes,holds=setup(normal_follow_budget_s=.4)
    def unexpected(execution):pytest.fail('normal finishing invoked by diagnostic transport')
    bridge.finish_position=unexpected
    result=dict(completed=False,action_source=source,source_q_position=[0.]*6,source_time_s=10.,
        policy_commands_attempted=0,policy_commands_returned=0,commands_position=[],command_times_s=[],
        readback_times_s=[],measured_q_path=[],original_target=[1.]*6)
    bridge.last_attempt=result
    bridge._transport_five(np.ones((5,6),np.float32),np.zeros(6,np.float32),10.,result)
    assert 'position_servo' not in result and len(writes)==5 and len(holds)==1


def test_finishing_is_disabled_by_default():
    bridge,runtime,writes,holds=full_window_bridge(lambda elapsed:-3.2)
    assert bridge.finish_position is None
    result=bridge.execute()
    assert result['completed'] and 'position_servo' not in result
    assert result['original_target_reached'] is False
    assert_original_y(result,runtime)


def real_helper_bridge(*, no_tuning=False):
    """Real bridge/helper/Goal-retention code using the fake register owner."""
    from fcpg.v21_hardware_state import completed_command_hold, verify_policy_goal_position
    from fcpg.v21_j3_position_servo import finish_original_j3_goal
    from fcpg.v21_models import CommandTrace
    from test_v21_same_goal_integral import Owner, MOTOR, ORDER, cells, source_execution
    from test_v21_follow_arrival_independent import Pending, State

    owner=Owner();owner.clock.now=20.
    bridge,runtime,_,_=setup(normal_follow_budget_s=2.)
    source=source_execution()
    start=np.asarray(source['measured_q_path'][0],np.float32)
    commands=torch.tensor(source['commands_position'],dtype=torch.float32)[None]
    runtime.pending=Pending(torch.from_numpy(start)[None],torch.tensor([20.],dtype=torch.float64),
        CommandTrace.policy(commands,torch.full((1,5),.1)),runtime.pending.response)
    runtime.state=State(runtime.pending.source_q,runtime.pending.source_time)
    runtime.pending_action_source='v21_lewm_cem'
    runtime.executing=False
    def begin(q,stamp):
        runtime.executing=True
        return runtime.pending.trace.positions
    runtime.begin_execution=begin
    owner.runtime=runtime;owner.bridge=bridge
    owner.io.settle_samples=3;owner.io.settle_position_tolerance=3.;owner.io.settle_step_tolerance=.25
    owner.io.calibration_reconstructed_encoder_cells=lambda q:np.array(list(cells(q).values()))
    owner.io.clearance_gate=SimpleNamespace(require_readback=owner.require_readback,
        require_policy_path=owner._physical_check)
    owner.bus.set_q(start)
    def policy_response():
        count=sum(event['kind']=='position' for event in owner.bus.events)
        q=start.copy();q[2]=[4.5,3.8,2.6,2.,2.][count-1]
        if no_tuning:q[0]=.5  # separate non-J3 error makes finishing inapplicable
        owner.bus.set_q(q)
    owner.on_write_q=policy_response
    def restored(register,value):
        if (register,value)==('I_Coefficient',0):
            owner.bus.values[MOTOR]['Present_Load']=24
    owner.bus.on_write=restored
    original_read=owner.bus.sync_read
    def read(register,*,normalize=False,num_retry=0):
        assert normalize is False
        return original_read(register,normalize=False,num_retry=0)
    owner.bus.sync_read=read  # the retention reader's retry budget is harmless in a fake bus
    bridge.clock=owner.clock
    bridge.wait=lambda seconds:setattr(owner.clock,'now',owner.clock.now+seconds)
    bridge.read_q=lambda:owner.bus.q().astype(np.float32)
    bridge.write_q=owner._write_q
    bridge.physical_check=owner._physical_check
    bridge.cancelled=owner.stop_event.is_set
    bridge.request_stop=owner.stop_event.set
    bridge.halt=owner._halt
    bridge.capture_endpoint=lambda:State(torch.from_numpy(owner.bus.q().astype(np.float32))[None],
        torch.tensor([owner.clock()],dtype=torch.float64))
    def verify(command):
        with owner.bus_lock:
            return verify_policy_goal_position(owner.bus,command,clock=owner.clock)
    bridge.verify_policy_goal=verify
    def finish(execution):
        return finish_original_j3_goal(owner,execution,enabled=True,
            clock=owner.clock,wait=owner.clock.wait,cancelled=owner.stop_event.is_set)
    bridge.finish_position=finish
    def retain(goal,follow,q5):
        with owner.bus_lock:
            return completed_command_hold(owner,goal,follow,q5,actuator_events=[])
    bridge.completion_hold=retain
    return bridge,runtime,owner


@pytest.mark.parametrize('no_tuning',[False,True])
def test_real_helper_bridge_and_goal_retention_preserve_actual_model_feedback(no_tuning):
    bridge,runtime,owner=real_helper_bridge(no_tuning=no_tuning)
    result=bridge.execute()
    assert result['completed'] and result['position_servo']['completed']
    assert result['position_servo']['applied'] is (not no_tuning)
    assert result['original_target_reached'] is (not no_tuning)
    assert result['normal_follow']['original_target_arrival']['reached'] is False
    assert len(result['commands_position'])==5 and len(result['measured_q_path'])==6
    np.testing.assert_allclose(np.asarray(result['measured_q_path'])[:,2],[5.,4.5,3.8,2.6,2.,2.],atol=1e-6)
    np.testing.assert_array_equal(runtime.actual[1][0].numpy(),result['measured_q_path'])
    np.testing.assert_allclose(result['readback_times_s'],20.+np.arange(6)*.1,atol=1e-8)
    assert runtime.feedback=={} and result['model_feedback_path']=='original_five_step_readback'
    assert len(result['normal_follow']['positions'])==40
    assert not owner.stop_event.is_set() and owner.halts==0
    assert sum(event['kind']=='position' for event in owner.bus.events)==5
    assert not any(event['kind']=='halt' for event in owner.bus.events)
    assert result['completion_hold']['goal_register_writes']==0
    if not no_tuning:
        assert result['endpoint_q_position'][2]==0.
        assert result['position_servo_endpoint_validation']['verified']


def real_arm_bridge(*, include_j2=False, deferred_j1=False):
    """Real arm helper and real model runtime over the existing fake bus."""
    from fcpg.v21_arm_position_servo import finish_original_arm_goal
    from fcpg.v21_core import V21Config, V21ResearchCore
    from fcpg.v21_data import TaskVocabulary
    from fcpg.v21_memory import VisualObservation
    from fcpg.v21_runtime import EncodedEndpoint, V21ResearchRuntime
    from test_v21_same_goal_integral import ORDER

    bridge,old_runtime,owner=real_helper_bridge()
    commands=old_runtime.pending.trace.positions.clone()
    start=owner.bus.q().astype(np.float32);start[3]=1.
    if include_j2:
        start[1]=-.5
        commands[:,:,1]=.104655956
    if deferred_j1:
        start[0]=.5
        commands[:,:,0]=.099165
    owner.bus.set_q(start)
    owner.bus.values['gripper'].update(Torque_Limit=500,Max_Torque_Limit=500)
    original_write=owner.bus.write
    def write(register,motor,value,*,normalize,num_retry):
        if motor not in ('wrist_flex','shoulder_lift'):
            return original_write(register,motor,value,normalize=normalize,num_retry=num_retry)
        assert normalize is False and num_retry==0
        owner.bus.record('write',register,motor,value)
        owner.bus.values[motor][register]=value
        if register=='I_Coefficient' and value==0:
            owner.bus.values[motor]['Present_Load']=24
    owner.bus.write=write
    original_read=owner.bus.sync_read
    def read(register,*,normalize=False,num_retry=0):
        for name in ('wrist_flex','shoulder_lift'):
            motor=owner.bus.values[name]
            if register=='Present_Position' and motor['I_Coefficient'] and not owner.bus.after_halt:
                motor['Present_Position']+=int(np.clip(motor['Goal_Position']-motor['Present_Position'],-50,50))
        return original_read(register,normalize=normalize,num_retry=num_retry)
    owner.bus.sync_read=read
    def policy_response():
        count=sum(event['kind']=='position' for event in owner.bus.events)
        q=start.copy();q[2]=[4.5,3.8,2.6,2.,2.][count-1];q[3]=.7
        if include_j2:q[1]=-.1
        if deferred_j1:q[0]=.31
        owner.bus.set_q(q)
    owner.on_write_q=policy_response
    vocabulary=TaskVocabulary.build(['pick red block'])
    core=V21ResearchCore(V21Config(len(vocabulary.words))).eval()
    def runtime_physical(q,commands):
        owner._physical_check(q.detach().cpu().numpy()[0],commands.detach().cpu().numpy()[0])
    runtime=V21ResearchRuntime(core,vocabulary.encode(['pick red block']),
        physical_check=runtime_physical,lease_check=owner._lease_check,clock_fn=owner.clock)
    owner.runtime=bridge.runtime=runtime
    def capture():
        bridge.wait(.001)
        return EncodedEndpoint(torch.from_numpy(owner.bus.q().astype(np.float32))[None],
            torch.tensor([owner.clock()],dtype=torch.float64),None)
    bridge.capture_endpoint=capture
    bridge.finish_position=lambda execution:finish_original_arm_goal(owner,execution,enabled=True,
        clock=owner.clock,wait=owner.clock.wait,cancelled=owner.stop_event.is_set)
    q=torch.from_numpy(start)[None];stamp=torch.tensor([owner.clock()],dtype=torch.float64)
    observation=VisualObservation(torch.zeros(1,2,256,384),q[:,None].expand(-1,2,-1),
        stamp[:,None].expand(-1,2),torch.ones(1,2,dtype=torch.bool))
    with torch.inference_mode():
        runtime.observe(EncodedEndpoint(q,stamp,observation))
        runtime.plan(command_builder=lambda state:commands,action_source='v21_lewm_cem')
    return bridge,runtime,owner


@pytest.mark.parametrize('include_j2',[False,True])
@pytest.mark.parametrize('stationary_release',[False,True])
def test_real_arm_finishing_and_full_second_hold_advance_real_model_progress_by_only_five(include_j2,stationary_release):
    bridge,runtime,owner=real_arm_bridge(include_j2=include_j2)
    if stationary_release:
        owner.bus.motion_step=0
        state=dict(pulses=0,reads=0)
        original_write=owner.bus.on_write
        def write(register,value):
            original_write(register,value)
            if (register,value)==('I_Coefficient',1):
                state.update(pulses=state['pulses']+1,reads=0)
        def read(register,motor):
            values=owner.bus.values['elbow_flex']
            if register=='Present_Position' and values['I_Coefficient']==1:
                state['reads']+=1
                assert state['reads']<=4, 'I1 was not cut immediately after stationary release'
                if state['reads']==4:values['Present_Position']-=65
        owner.bus.on_write=write;owner.bus.on_read=read
    with torch.inference_mode():result=bridge.execute()
    report=result['position_servo'];follow=report['completion_follow']
    assert result['completed'] and result['original_target_reached']
    expected_motors=['elbow_flex','wrist_flex']+(['shoulder_lift'] if include_j2 else [])
    assert report['touched_motors']==expected_motors
    assert report['axes']['elbow_flex']['integral_control']['maximum_pulses']==16
    assert report['axes']['elbow_flex']['integral_exposure']['pulse_count']==(3 if stationary_release else 1)
    assert report['original_goal_hold']['sample_count']==22 and len(follow['positions'])==22
    assert report['original_goal_hold']['actual_span_s']>=1.
    assert result['endpoint_time_s']>follow['times_s'][-1]
    assert result['completion_hold']['goal_register_writes']==0
    assert result['endpoint_original_target_position']['original_goal_retained_through_completion']
    assert len(result['normal_follow']['positions'])==40
    assert result['normal_follow']['original_target_arrival']['reached'] is False
    assert len(result['commands_position'])==5 and len(result['measured_q_path'])==6
    np.testing.assert_allclose(result['readback_times_s'],20.+np.arange(6)*.1,atol=1e-8)
    np.testing.assert_allclose(np.asarray(result['measured_q_path'])[:,2],[5.,4.5,3.8,2.6,2.,2.],atol=1e-6)
    assert runtime.state.memory.current_readback_index.tolist()==[5]
    assert runtime.state.feedback_readback_index.tolist()==[5]
    assert runtime.state.history.valid.sum()==1 and runtime.state.q_error.shape==(1,5,6)
    torch.testing.assert_close(runtime.state.history.readbacks_position[:,-1],
        torch.tensor(result['measured_q_path'])[None],atol=0,rtol=0)
    np.testing.assert_allclose(runtime.state.feedback_offsets[0].numpy(),np.arange(1,6)*.1,atol=1e-8)
    assert runtime.state.history.readbacks_position[0,-1,-1,2]==2.
    assert runtime.state.q[0,2].item()==pytest.approx(.05 if stationary_release else 0.,abs=1e-6)
    assert not owner.stop_event.is_set() and owner.halts==0
    assert sum(event['kind']=='position' for event in owner.bus.events)==5
    assert owner.bus.values['gripper']['Torque_Limit']==500
    assert owner.bus.values['gripper']['Max_Torque_Limit']==500
    assert report['final_parameters_by_motor']==report['baseline_parameters_by_motor']
    assert report['cleanup']['parameters_restored_by_motor']==dict.fromkeys(expected_motors,True)
    if include_j2:
        actual_y=runtime.state.history.readbacks_position[0,-1]
        np.testing.assert_allclose(actual_y[:,1].numpy(),[-.5,-.1,-.1,-.1,-.1,-.1],atol=1e-6)
        original_j2_goal=result['commands_position'][-1][1]
        assert original_j2_goal==pytest.approx(.104655956,abs=1e-8)
        assert actual_y[-1,1].item()-original_j2_goal==pytest.approx(-.204655956,abs=1e-6)
        assert runtime.state.q[0,1].item()==pytest.approx(.1,abs=1e-6)
        assert owner.bus.values['shoulder_lift']['Goal_Position']==10010
        assert list(dict.fromkeys(event['motor'] for event in report['parameter_writes']))==expected_motors
        assert report['control_exposure']['within_limit'] is True


def test_real_arm_final_hold_drift_invalidates_previously_reached_without_committing_y():
    """A later physical loss cannot reuse true but historical follow arrival."""
    from test_v21_same_goal_integral import ORDER

    bridge,runtime,owner=real_arm_bridge()
    def follow_each_policy_goal():
        owner.bus.set_q([(owner.bus.values[name]['Goal_Position']-10000)/100. for name in ORDER])
    owner.on_write_q=follow_each_policy_goal
    finishing=[False];reads=[0];callback=bridge.finish_position
    def begin_finishing(execution):
        finishing[0]=True
        return callback(execution)
    bridge.finish_position=begin_finishing
    def drift_in_final_hold(register,motor):
        if finishing[0] and register=='Present_Position' and motor=='all':
            reads[0]+=1
            if reads[0]==6:
                owner.bus.values['wrist_flex']['Present_Position']+=30
    owner.bus.on_read=drift_in_final_hold
    with pytest.raises(RuntimeError):
        with torch.inference_mode():bridge.execute()
    result=bridge.last_attempt;report=result['position_servo']
    assert reads[0]>=6
    assert result['normal_follow']['original_target_arrival']['reached'] is True
    assert result['position_servo_prior_arrival']['reached'] is True
    assert report['arrival_before_failure']['reached'] is True
    assert result['original_target_reached'] is False
    assert result['original_target_arrival']['verified'] is False
    assert report['original_target_reached'] is False and report['original_goal_held'] is False
    assert report['original_goal_hold']['completed'] is False
    assert 0<len(report['original_goal_hold']['readbacks'])<22
    assert not report['parameter_writes'] and not report['touched_motors']
    assert not result['completed'] and 'endpoint_q_position' not in result
    assert owner.stop_event.is_set() and runtime.stopped and owner.halts>=1
    assert runtime.pending is None and not runtime.executing
    assert runtime.state.memory.current_readback_index.tolist()==[0]
    assert runtime.state.history.valid.sum()==0
    assert len(result['commands_position'])==5 and len(result['measured_q_path'])==6
    assert sum(event['kind']=='position' for event in owner.bus.events)==5
    assert owner.bus.values['gripper']['Torque_Limit']==500


@pytest.mark.parametrize('no_tuning',[False,True])
def test_real_partial_arm_finishing_preserves_full_axis_miss_and_original_model_y(no_tuning):
    from test_v21_same_goal_integral import ORDER

    bridge,runtime,owner=real_arm_bridge(deferred_j1=True)
    if no_tuning:
        def selected_already_reached():
            q=np.array([(owner.bus.values[name]['Goal_Position']-10000)/100. for name in ORDER])
            q[0]=.31;owner.bus.set_q(q)
        owner.on_write_q=selected_already_reached
    with torch.inference_mode():result=bridge.execute()
    report=result['position_servo'];hold=report['original_goal_hold'];follow=report['completion_follow']
    assert result['completed'] and report['applied'] is (not no_tuning)
    assert report['selected_axes_completed'] and report['selected_axes_held']
    assert result['selected_axes_completed'] and result['selected_axes_held']
    assert report['original_target_reached'] is False and result['original_target_reached'] is False
    assert report['original_goal_held'] is False and report['original_goal_preserved'] is True
    assert hold['sample_count']==22 and hold['actual_span_s']>=1.
    assert hold['arrival']['reached'] is False
    assert hold['selected_axes_arrival']['reached'] is True
    assert hold['selected_axes_arrival']['all_axes_reached'] is False
    assert hold['selected_axes_arrival']['quiet_samples']==22
    assert follow['goal_reached'] is False and follow['hold_eligible'] is True
    assert report['deferred_axes']==[0] and report['deferred_anchor_q'][0]==.31
    assert result['endpoint_original_target_position']['position_within_tolerance'] is False
    assert result['endpoint_original_target_position']['max_abs_residual']==pytest.approx(.210835,abs=1e-6)
    assert result['position_servo_endpoint_validation']['verified'] is True
    assert result['completion_hold']['goal_register_writes']==0
    assert result['endpoint_original_target_position']['original_goal_retained_through_completion']
    assert result['endpoint_time_s']>follow['times_s'][-1]
    assert len(result['commands_position'])==5 and len(result['measured_q_path'])==6
    assert len(result['normal_follow']['positions'])==40
    np.testing.assert_allclose(result['readback_times_s'],20.+np.arange(6)*.1,atol=1e-8)
    torch.testing.assert_close(runtime.state.history.readbacks_position[:,-1],
        torch.tensor(result['measured_q_path'])[None],atol=0,rtol=0)
    assert runtime.state.memory.current_readback_index.tolist()==[5]
    assert runtime.state.feedback_readback_index.tolist()==[5]
    assert runtime.state.history.valid.sum()==1
    assert sum(event['kind']=='position' for event in owner.bus.events)==5
    assert not owner.stop_event.is_set() and owner.halts==0
    assert owner.bus.values['gripper']['Torque_Limit']==owner.bus.values['gripper']['Max_Torque_Limit']==500


@pytest.mark.parametrize('no_tuning',[False,True])
def test_partial_finishing_cannot_hide_a_completion_goal_write(no_tuning):
    bridge,runtime,owner=real_arm_bridge(deferred_j1=True)
    if no_tuning:
        from test_v21_same_goal_integral import ORDER
        def selected_already_reached():
            q=np.array([(owner.bus.values[name]['Goal_Position']-10000)/100. for name in ORDER])
            q[0]=.31;owner.bus.set_q(q)
        owner.on_write_q=selected_already_reached
    bridge.completion_hold=lambda goal,follow,q5:dict(event='safe_hold_verified',
        reused_existing_hold=False,motor_register_writes=1,
        attempts=[dict(held_state=q5,mismatched_joints={})])
    with pytest.raises(RuntimeError,match='retain the original goal'):
        with torch.inference_mode():bridge.execute()
    result=bridge.last_attempt
    assert result['position_servo']['selected_axes_held'] is True  # historical completed helper
    assert result['selected_axes_completed'] is False and result['selected_axes_held'] is False
    assert result['selected_axes_arrival']['verified'] is False
    assert result['position_servo_prior_selected_arrival']['reached'] is True
    assert result['original_target_reached'] is False and not result['completed']
    assert runtime.state.history.valid.sum()==0
    assert owner.stop_event.is_set() and owner.halts>=1


@pytest.mark.parametrize('axis,delta',[(2,.15),(2,.21),(0,.15),(0,.26),(4,-.15)])
def test_partial_finishing_endpoint_cannot_lose_selected_or_remaining_axis_guards(axis,delta):
    bridge,runtime,owner=real_arm_bridge(deferred_j1=True);capture=bridge.capture_endpoint
    def changed_endpoint():
        endpoint=capture();endpoint.q[0,axis]+=delta;return endpoint
    bridge.capture_endpoint=changed_endpoint
    with pytest.raises(RuntimeError,match='changed during endpoint capture'):
        with torch.inference_mode():bridge.execute()
    result=bridge.last_attempt
    assert result['position_servo']['selected_axes_held'] is True
    assert result['position_servo_endpoint_validation']['verified'] is False
    assert result['selected_axes_completed'] is False and result['selected_axes_arrival']['verified'] is False
    assert result['original_target_reached'] is False and not result['completed']
    assert runtime.state.memory.current_readback_index.tolist()==[0]
    assert runtime.state.history.valid.sum()==0 and owner.stop_event.is_set()


def selected_follow(*, anchor=.21, source=.21, position=.21):
    """Independent numerical receipt for post-finishing validation counterexamples."""
    from fcpg.v21_position_arrival import evaluate_selected_position_arrival
    goal=np.zeros(6);q=np.zeros(6);q[0]=position
    positions=np.repeat(q[None],22,axis=0);times=np.arange(1,23)*.05
    original=evaluate_position_arrival(goal,positions,times,
        position_tolerance=.2,step_tolerance=.1,quiet_samples=3)
    selected=evaluate_selected_position_arrival(goal,positions,times,selected_axes=[2,3,1],
        position_tolerance=.2,step_tolerance=.1,quiet_samples=22)
    return dict(schema='v21-position-servo-completion-follow-v1',original_target=goal.tolist(),
        positions=positions.tolist(),times_s=times.tolist(),source_time_s=0.,
        baseline_restored=True,hold_eligible=True,goal_reached=original['reached'],
        selected_axes=[2,3,1],selected_axes_arrival=selected,deferred_axes=[0],
        deferred_anchor_q=[anchor,0.,0.,0.,0.,0.],deferred_position_tolerance=.25,
        source_q0=[source,0.,0.,0.,0.,0.])


@pytest.mark.parametrize('case',['interval_escape','full_arrival_loss','cumulative_anchor_drift'])
def test_later_readback_rejects_drift_even_when_each_step_is_less_than_point_one(case):
    from fcpg.v21_hardware_state import position_servo_completion_proof, position_servo_readback_proof
    if case=='interval_escape':
        follow=selected_follow();actual=[.29,0.,0.,0.,0.,0.]
    elif case=='full_arrival_loss':
        follow=selected_follow(anchor=.3,source=.3,position=.15);actual=[.23,0.,0.,0.,0.,0.]
    else:
        follow=selected_follow(anchor=.3,source=.7,position=.54);actual=[.59,0.,0.,0.,0.,0.]
    goal=follow['original_target']
    position_servo_completion_proof(follow,goal)
    proof=position_servo_readback_proof(follow,goal,actual)
    assert proof['step_from_final_finishing_readback']<.1
    assert proof['verified'] is False
    if case=='interval_escape':assert proof['deferred_original_corridor_valid'] is False
    elif case=='full_arrival_loss':assert proof['required_axes']==list(range(6))
    else:assert proof['deferred_max_abs_drift']>.25


def test_endpoint_cannot_hide_an_excessive_step_between_two_actual_post_hold_readbacks():
    from fcpg.v21_hardware_state import position_servo_readback_proof
    follow=selected_follow();actual=np.asarray(follow['positions'][-1]);previous=actual.copy()
    actual[2]=-.09;previous[2]=.09
    without_hold=position_servo_readback_proof(follow,follow['original_target'],actual)
    proof=position_servo_readback_proof(follow,follow['original_target'],actual,previous_actual=previous)
    assert without_hold['verified'] is True
    assert proof['step_from_final_finishing_readback']==pytest.approx(.09)
    assert proof['step_from_completion_hold_readback']==pytest.approx(.18)
    assert proof['verified'] is False


@pytest.mark.parametrize('case',['interval_escape','full_arrival_loss','selected_lost','other_lost'])
def test_actual_completion_hold_validates_partial_constraints_before_retaining_goal(case,monkeypatch):
    import fcpg.v21_hardware_state as state
    bridge,_,owner=real_helper_bridge()
    follow=selected_follow(anchor=.3,source=.3,position=.15) if case=='full_arrival_loss' else selected_follow()
    actual=np.asarray(follow['positions'][-1]).copy()
    if case=='interval_escape':actual[0]=.29
    elif case=='full_arrival_loss':actual[0]=.23
    elif case=='selected_lost':actual[2]=.15  # precise position but excessive new step
    else:actual[4]=.21
    owner.bus.set_goal(follow['original_target']);owner.bus.set_q(actual)
    events=[]
    def fallback(*args,**kwargs):return dict(event='test_emergency_fallback')
    monkeypatch.setattr(state,'verified_hold',fallback)
    with owner.bus_lock:
        result=state.completed_command_hold(owner,follow['original_target'],follow,actual,
            actuator_events=events)
    assert result['event']=='test_emergency_fallback'
    assert 'position-servo arrival changed' in result['normal_validation_error']
    assert events[0]['position_servo_readback_validation']['verified'] is False


@pytest.mark.parametrize('damage',['four_samples','selected_three','source_missing','anchor_bool',
    'missing_deferred','selected_deferred','relaxed_tolerance','false_selected_evidence'])
def test_selected_completion_contract_cannot_weaken_or_invent_actual_evidence(damage):
    from fcpg.v21_hardware_state import position_servo_completion_proof
    follow=selected_follow()
    if damage=='four_samples':
        follow['positions']=follow['positions'][:4];follow['times_s']=follow['times_s'][:4]
    elif damage=='selected_three':follow['selected_axes_arrival']['quiet_samples']=3
    elif damage=='source_missing':follow.pop('source_q0')
    elif damage=='anchor_bool':follow['deferred_anchor_q']=[True]*6
    elif damage=='missing_deferred':follow['deferred_axes']=[]
    elif damage=='selected_deferred':follow['deferred_axes']=[0,2]
    elif damage=='relaxed_tolerance':follow['deferred_position_tolerance']=.3
    else:follow['selected_axes_arrival']['all_axes_reached']=True
    with pytest.raises(ValueError):position_servo_completion_proof(follow,follow['original_target'])


def test_partial_completion_source_cannot_differ_from_the_original_execution_q0():
    bridge,runtime,owner=real_arm_bridge(deferred_j1=True);callback=bridge.finish_position
    def false_source(execution):
        report=callback(execution)
        report['source_q0'][0]+=.1
        report['completion_follow']['source_q0']=deepcopy(report['source_q0'])
        return report
    bridge.finish_position=false_source
    with pytest.raises(RuntimeError,match='changed the original execution source'):
        with torch.inference_mode():bridge.execute()
    assert not bridge.last_attempt['completed'] and runtime.state.history.valid.sum()==0
    assert owner.stop_event.is_set()


def test_real_partial_endpoint_is_checked_against_actual_completion_hold_readback():
    bridge,runtime,owner=real_arm_bridge(deferred_j1=True);retain=bridge.completion_hold;capture=bridge.capture_endpoint
    def hold_at_positive_offset(goal,follow,q5):
        owner.bus.values['elbow_flex']['Present_Position']+=9
        return retain(goal,follow,q5)
    def endpoint_at_negative_offset():
        endpoint=capture();endpoint.q[0,2]=-.09;return endpoint
    bridge.completion_hold=hold_at_positive_offset;bridge.capture_endpoint=endpoint_at_negative_offset
    with pytest.raises(RuntimeError,match='changed during endpoint capture'):
        with torch.inference_mode():bridge.execute()
    result=bridge.last_attempt;proof=result['position_servo_endpoint_validation']
    assert result['completion_hold']['actual_readback']['present_position']['elbow_flex']==.09
    assert proof['step_from_final_finishing_readback']==pytest.approx(.09)
    assert proof['step_from_completion_hold_readback']==pytest.approx(.18)
    assert not proof['verified'] and not result['selected_axes_completed']
    assert runtime.state.history.valid.sum()==0 and owner.stop_event.is_set()
