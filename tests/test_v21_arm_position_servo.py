"""Normal bounded arm finishing uses fake clocks/registers, never hardware."""
from copy import deepcopy
import json
import threading

import numpy as np
import pytest

from fcpg.v21_arm_position_servo import ArmPositionServoError, finish_original_arm_goal
from fcpg.v21_position_arrival import evaluate_position_arrival, evaluate_selected_position_arrival
from fcpg.v21_same_goal_integral import AXIS, MOTOR, ORDER, _ramp
from test_v21_same_goal_integral import Bus, Owner, cells, source_execution


J2, J3, J4 = ORDER[1:4]


class ArmBus(Bus):
    def __init__(self, owner):
        super().__init__(owner)
        self.steps = {J3:50, J4:20, J2:10}
        self.read_retries = []
        self.values[ORDER[5]]['Torque_Limit'] = self.values[ORDER[5]]['Max_Torque_Limit'] = 500
        for motor in ORDER:
            self.values[motor]['Present_Load'] = 0

    def read(self, register, motor, *, normalize, num_retry):
        assert normalize is False and num_retry in (0,1)
        self.read_retries.append(('read',register,motor,num_retry))
        self.record('read',register,motor)
        if self.on_read:
            self.on_read(register,motor)
        return self.values[motor][register]

    def sync_read(self, register, *, normalize, num_retry):
        assert normalize is False and num_retry in (0,1)
        self.read_retries.append(('sync_read',register,'all',num_retry))
        self.record('read',register,'all')
        if register == 'Present_Position' and not self.after_halt:
            for motor in (J3,J4,J2):
                values = self.values[motor]
                if values['I_Coefficient'] == 1:
                    values['Present_Position'] += int(np.clip(values['Goal_Position']-values['Present_Position'],
                        -self.steps[motor],self.steps[motor]))
        if self.on_read:
            self.on_read(register,'all')
        return {motor:self.values[motor][register] for motor in ORDER}

    def write(self, register, motor, value, *, normalize, num_retry):
        assert normalize is False and num_retry == 0
        assert register in ('I_Coefficient','D_Coefficient','Torque_Limit') and motor in (J3,J4,J2)
        self.record('write',register,motor,value)
        self.values[motor][register] = value
        if self.on_write:
            self.on_write(register,motor,value)


def fixture():
    owner = Owner(); owner.bus = ArmBus(owner); owner.io.follower.bus = owner.bus
    execution = source_execution()
    execution.update(completed=False,action_source='v21_lewm_cem',source_time_s=execution['readback_times_s'][0])
    execution['measured_q_path'][0][3] = 2.
    target = np.asarray(execution['commands_position'][-1])
    commands = _ramp(execution['measured_q_path'][0],target)
    execution['commands_position'] = commands.tolist()
    for index,(row,command) in enumerate(zip(execution['policy_goal_readbacks'],commands)):
        row.update(read_attempted=True,read_returned=True,goal_raw=cells(command),expected_goal_raw=cells(command),
            requested_goal_position=dict(zip(ORDER,command.tolist())))
    tail = np.array([0.,0.,3.06,.67,0.,30.])
    execution['measured_q_path'][1:] = np.repeat(tail[None],5,axis=0).tolist()
    execution['normal_follow']['positions'] = np.repeat(tail[None],40,axis=0).tolist()
    owner.bus.set_q(tail); owner.bus.set_goal(target)
    owner.runtime.stopped=False; owner.runtime.executing=True; owner.runtime.pending=object()
    owner.bridge.last_attempt=execution; owner.bridge.lock=threading.Lock(); owner.bridge.lock.acquire()
    owner.io.settle_samples=3; owner.io.settle_position_tolerance=3.; owner.io.settle_step_tolerance=.25
    owner._write_q=lambda *args,**kwargs:pytest.fail('arm servo must never write a policy Goal')
    return owner,execution


def run(owner,execution):
    return finish_original_arm_goal(owner,deepcopy(execution),enabled=True,clock=owner.clock,wait=owner.clock.wait)


def rejected(owner,execution,match=None):
    with pytest.raises(ArmPositionServoError,match=match) as caught:
        run(owner,execution)
    report=caught.value.report
    json.dumps(report,allow_nan=False)
    assert not report['completed']
    if report['cleanup']['restored']:
        assert report['cleanup']['baseline_restored']
        assert report['cleanup']['torque_restored']
        assert report['cleanup']['profile_state']=='baseline1000'
        assert report['cleanup']['pre_restore_low_load_verified']
        assert not report['cleanup']['pre_restore_quiet_verified']
        assert report['cleanup']['post_restore_quiet_verified']
        assert report['final_equals_baseline']
        assert report['final_parameters_by_motor']==report['baseline_parameters_by_motor']
    else:
        assert 'final_parameters_by_motor' not in report
    if 'parameters_before_failure' in report:
        assert report['parameters_before_failure']['historical_only']
        assert report['parameters_before_failure']['final_readback'] is False
    return report


def change_pose(owner,execution,q):
    execution['normal_follow']['positions'] = np.repeat(np.asarray(q)[None],40,axis=0).tolist()
    owner.bus.set_q(q)


def three_axis_fixture():
    owner,execution=fixture()
    execution['measured_q_path'][0][1]=-2.
    target=np.asarray(execution['commands_position'][-1]);target[1]=.104655956
    commands=_ramp(execution['measured_q_path'][0],target)
    execution['commands_position']=commands.tolist()
    execution['original_target_arrival']['original_target']=commands[-1].tolist()
    for receipt,command in zip(execution['policy_goal_readbacks'],commands):
        receipt.update(goal_raw=cells(command),expected_goal_raw=cells(command),
            requested_goal_position=dict(zip(ORDER,command.tolist())))
    q=owner.bus.q();q[1]=-.1
    change_pose(owner,execution,q)
    execution['measured_q_path'][1:]=np.repeat(q[None],5,axis=0).tolist()
    owner.bus.set_goal(commands[-1])
    return owner,execution


def shoulder_scheduler_fixture():
    owner,execution=three_axis_fixture()
    target=np.asarray(execution['commands_position'][-1])
    q=target.copy();q[1]=-.44
    change_pose(owner,execution,q)
    execution['measured_q_path'][1:]=np.repeat(q[None],5,axis=0).tolist()
    owner.bus.set_q(q);owner.bus.steps[J2]=17
    return owner,execution


def pause_during_shoulder_candidate(owner, *, delay=.617, action=None):
    state=dict(candidate_reads=0,injected=False)
    def read(register,motor):
        if (register=='Present_Position' and owner.bus.values[J2]['I_Coefficient']==1
                and not owner.stop_event.is_set()):
            state['candidate_reads']+=1
    def wait_pause():
        if state['candidate_reads']>=2 and not state['injected']:
            state['injected']=True;owner.clock.now+=delay
            if action is not None:action()
    owner.bus.on_read=read;owner.clock.on_wait=wait_pause
    return state


def deferred_axis_fixture(*, selected_reached=False, source_j1=2.):
    owner,execution=fixture()
    execution['measured_q_path'][0][0]=source_j1
    target=np.asarray(execution['commands_position'][-1]);target[0]=.089164929
    commands=_ramp(execution['measured_q_path'][0],target)
    execution['commands_position']=commands.tolist()
    execution['original_target_arrival']['original_target']=commands[-1].tolist()
    for receipt,command in zip(execution['policy_goal_readbacks'],commands):
        receipt.update(goal_raw=cells(command),expected_goal_raw=cells(command),
            requested_goal_position=dict(zip(ORDER,command.tolist())))
    q=owner.bus.q();q[0]=.3
    if selected_reached:q[[2,3,1]]=target[[2,3,1]]
    change_pose(owner,execution,q)
    execution['measured_q_path'][1:]=np.repeat(q[None],5,axis=0).tolist()
    owner.bus.set_goal(commands[-1])
    return owner,execution


def test_disabled_does_not_touch_owner_or_clock():
    class Forbidden:
        def __getattr__(self,name):pytest.fail('disabled owner access '+name)
    def forbidden():pytest.fail('disabled clock access')
    with pytest.raises(ArmPositionServoError,match='disabled; no IO'):
        finish_original_arm_goal(Forbidden(),clock=forbidden)


def test_both_axes_finish_original_goal_then_hold_all_axes_without_changing_y_or_gripper():
    owner,execution=fixture(); before=deepcopy(execution); pending=owner.runtime.pending
    report=run(owner,execution)
    json.dumps(report,allow_nan=False)
    assert report['completed'] and report['applied'] and report['original_target_reached']
    assert report['original_goal_held'] and report['original_goal_preserved']
    assert report['touched_motors']==[J3,J4]
    assert report['cleanup']['restored'] and not report['cleanup']['hold_performed']
    assert report['cleanup']['parameters_restored_by_motor']=={J3:True,J4:True}
    assert report['final_equals_baseline']
    assert report['baseline_parameters_by_motor']==report['final_parameters_by_motor']
    assert report['final_parameters_by_motor'][ORDER[5]]['Torque_Limit']==500
    assert report['final_parameters_by_motor'][ORDER[5]]['Max_Torque_Limit']==500
    assert execution==before and owner.runtime.pending is pending and owner.runtime.history==['untouched']
    assert owner.runtime.executing and not owner.stop_event.is_set() and not owner.halts
    assert AXIS==2 and MOTOR==J3
    writes=[(e['motor'],e['register'],e['value']) for e in owner.bus.events if e['kind']=='write']
    expected=[('Torque_Limit',200),('D_Coefficient',32),('Torque_Limit',None),
        ('I_Coefficient',1),('I_Coefficient',0),
        ('Torque_Limit',200),('D_Coefficient',0),('Torque_Limit',1000)]
    assert writes==[(motor,key,300 if value is None else value)
        for motor in (J3,J4) for key,value in expected]
    assert all(owner.bus.values[motor]['Goal_Position']==report['original_target_raw'][motor] for motor in ORDER)
    assert not [event for event in owner.bus.events if event['kind'] in ('position','halt')]
    for motor in (J3,J4):
        stages=report['axes'][motor]['stages']
        assert stages['candidate']['target_band_entry']['stationary_arrival_claimed'] is False
        assert stages['integral_off_settling']['axis_arrival']['reached']
        assert len(stages['original_goal_retention']['readbacks'])==6
        assert len(stages['restored_original_goal']['readbacks'])==3
        assert report['axes'][motor]['integral_exposure']['within_limit']
    assert report['axes'][J3]['stages']['candidate']['readbacks'][-1]['q_position'][3]>.2
    assert report['control_exposure']['within_limit'] and report['control_exposure']['duration_s']<30
    hold=report['original_goal_hold']
    assert hold['completed'] and hold['sample_count']==22 and hold['actual_span_s']>=1
    assert hold['arrival']['quiet_samples']==22 and hold['arrival']['reached']
    follow=report['completion_follow']
    assert len(follow['positions'])==22 and 'budget_s' not in follow
    expected_arrival=evaluate_position_arrival(report['original_target'],follow['positions'],
        np.asarray(follow['times_s'])-execution['source_time_s'],position_tolerance=.2,step_tolerance=.1,quiet_samples=3)
    for key,value in expected_arrival.items():assert report['original_target_arrival'][key]==value


@pytest.mark.parametrize('kind',['already','non_arm','past_source','past_target','unstable'])
def test_ineligible_poses_skip_with_fresh_actuals_and_no_writes(kind):
    owner,execution=fixture(); q=owner.bus.q()
    if kind=='already':q[2:4]=[.1,.1]
    elif kind=='non_arm':q[0]=.4
    elif kind=='past_source':q[2]=5.1
    elif kind=='past_target':q[2]=-.3
    change_pose(owner,execution,q)
    if kind=='unstable':execution['normal_follow']['positions'][-2][2]+=.12
    result=run(owner,execution)
    assert result['completed'] and result['skipped'] and not result['applied']
    assert result['touched_motors']==[] and result['parameter_writes']==[]
    assert result['final_equals_baseline'] and result['cleanup']['parameters_restored_by_motor']=={}
    assert not owner.halts and not owner.stop_event.is_set()
    assert result['original_target_reached'] is (kind=='already')
    assert result['original_goal_held'] is (kind=='already')
    assert result['selected_axes_completed'] is (kind=='already')
    assert result['selected_axes_held'] is (kind=='already')
    if kind!='already':
        assert len(result['completion_follow']['positions'])==4
        assert not {'selected_axes','selected_axes_arrival','deferred_axes','deferred_anchor_q',
            'deferred_position_tolerance'}&result['completion_follow'].keys()


def test_intermediate_command_overshoot_inside_verified_hull_is_recovered_to_final_target():
    owner,execution=fixture(); q=owner.bus.q()
    execution['measured_q_path'][0][3]=.3
    execution['commands_position'][0][3]=1.
    receipt=execution['policy_goal_readbacks'][0]; command=execution['commands_position'][0]
    receipt.update(goal_raw=cells(command),expected_goal_raw=cells(command),
        requested_goal_position=dict(zip(ORDER,command)))
    q[3]=.8
    change_pose(owner,execution,q)
    result=run(owner,execution)
    lower=result['command_envelope']['lower'][3]
    upper=result['command_envelope']['upper'][3]
    assert lower < result['original_target'][3] + 1e-5 < q[3] < upper
    assert result['finishing_intervals'][J4]==[lower,upper]
    assert result['finishing_interval_basis']=='q0_and_verified_policy_goal_hull'
    assert result['original_target_reached'] and result['original_goal_held']
    assert result['touched_motors']==[J3,J4]
    assert result['axes'][J4]['completed'] and result['axes'][J4]['applied']


@pytest.mark.parametrize('axis',[2,3])
def test_one_eligible_axis_does_not_write_the_other(axis):
    owner,execution=fixture(); q=owner.bus.q(); q[5-axis]=.1
    change_pose(owner,execution,q)
    result=run(owner,execution)
    assert result['original_target_reached'] and result['touched_motors']==[ORDER[axis]]


@pytest.mark.parametrize('damage',['wrong_j4_id','wrong_j4_cap','wrong_j4_layout','goal','active_runtime','source'])
def test_preflight_rejects_contradictory_live_identity_without_parameter_writes(damage):
    owner,execution=fixture()
    if damage=='wrong_j4_id':owner.bus.values[J4]['ID']=3
    elif damage=='wrong_j4_cap':owner.bus.values[J4]['Torque_Limit']=200
    elif damage=='wrong_j4_layout':owner.bus.model_ctrl_table['sts3215']['I_Coefficient']=(24,1)
    elif damage=='goal':owner.bus.values[J4]['Goal_Position']+=1
    elif damage=='active_runtime':owner.runtime.pending=None
    else:execution['completed']=True
    result=rejected(owner,execution)
    assert not [event for event in owner.bus.events if event['kind']=='write']


@pytest.mark.parametrize('fault',['i1_ack','i0_ack','cap300_ack','cap_ack','d0_ack','unreached','stop','timeout','pending_motion',
    'completed_axis_lost','final_position','final_step','first_final_step','final_load','baseline_changed','lease_changed'])
def test_faults_abort_without_faking_arrival_and_clear_each_touched_axis(fault):
    owner,execution=fixture(); fired=[]; final_started=[]; restored=[]
    if fault=='unreached':owner.bus.steps[J4]=0
    def written(register,motor,value):
        if motor==J4 and register=='I_Coefficient' and value==1:
            if fault=='i1_ack':raise ConnectionError('J4 I1 ACK lost')
            if fault=='timeout':owner.clock.now+=45
            if fault=='lease_changed':owner.runtime.pending=object()
        if motor==J4 and register=='Torque_Limit' and value==300 and fault=='cap300_ack':
            raise ConnectionError('J4 cap300 ACK lost')
        if motor==J4 and register=='I_Coefficient' and value==0 and fault=='i0_ack':
            raise ConnectionError('J4 I0 ACK lost')
        if motor==J4 and register=='Torque_Limit' and value==200 and owner.bus.values[J4]['I_Coefficient']==0:
            if fault=='cap_ack' and owner.bus.values[J4]['D_Coefficient']==32:raise ConnectionError('J4 cap ACK lost')
        if motor==J4 and register=='D_Coefficient' and value==0 and fault=='d0_ack':
            raise ConnectionError('J4 D0 ACK lost')
        if register=='Torque_Limit' and value==1000:
            restored.append(motor)
    def read(register,motor):
        if register!='Present_Position' or owner.stop_event.is_set():return
        j3=owner.bus.values[J3]; j4=owner.bus.values[J4]
        if fault=='pending_motion' and j3['I_Coefficient']==1 and not fired:
            fired.append(True); j4['Present_Position']+=30
        if j4['I_Coefficient']==1 and not fired:
            if fault=='stop':fired.append(True);owner.stop_event.set()
            elif fault=='completed_axis_lost':fired.append(True);j3['Present_Position']+=24
        if J4 in restored:
            final_started.append(True)
            # Three post-cap axis samples precede the all-axis hold.
            sample=len(final_started)
            if not fired and ((fault=='first_final_step' and sample==4) or sample==7):
                if fault in ('final_position','final_step','first_final_step'):
                    fired.append(True);j4['Present_Position']+=24 if fault=='final_position' else 15
                elif fault=='final_load':fired.append(True);j3['Present_Load']=33
                elif fault=='baseline_changed':fired.append(True);owner.bus.values[ORDER[5]]['Torque_Limit']=1000
    owner.bus.on_write=written; owner.bus.on_read=read
    result=rejected(owner,execution)
    assert result['cleanup']['hold_performed'] and owner.halts and owner.stop_event.is_set()
    assert not result['original_goal_preserved'] and 'completion_follow' not in result
    for motor in result['touched_motors']:
        expected=(result['baseline_parameters_by_motor'][motor]['Torque_Limit']
                  if result['cleanup']['restored'] else 200)
        assert owner.bus.values[motor]['Torque_Limit']==expected
    if fault=='i0_ack':
        assert not result['cleanup']['recovery_by_motor'][J4]['integral_zero_verified']
        assert result['cleanup']['recovery_by_motor'][J3]['integral_zero_verified']
        assert owner.bus.values[J4]['D_Coefficient']==32
    if fault in ('final_position','final_step','first_final_step','final_load','baseline_changed'):
        assert result['axes'][J4]['completed'] and fired
        assert not result['original_goal_hold']['completed']


def test_post_restore_actual_residual_is_not_replaced_with_integral_exact_goal():
    owner,execution=fixture()
    def written(register,motor,value):
        if register=='Torque_Limit' and value==1000:
            owner.bus.values[motor]['Present_Position']=owner.bus.values[motor]['Goal_Position']+13
    owner.bus.on_write=written
    result=run(owner,execution)
    assert result['original_target_reached'] and result['original_goal_held']
    for axis,motor in ((2,J3),(3,J4)):
        candidate=result['axes'][motor]['stages']['candidate']['axis_arrival']['residual']
        assert result['original_target_arrival']['residual_by_axis'][axis]==pytest.approx(.13,abs=1e-7)
        assert result['original_target_arrival']['residual_by_axis'][axis]!=candidate


def test_full_original_five_step_envelope_includes_q0_and_keeps_production_five_row_geometry():
    owner,execution=fixture()
    execution['measured_q_path'][0][2]=7.5
    commands=_ramp(execution['measured_q_path'][0],execution['commands_position'][-1])
    execution['commands_position']=commands.tolist()
    for receipt,command in zip(execution['policy_goal_readbacks'],commands):
        receipt.update(goal_raw=cells(command),expected_goal_raw=cells(command),
            requested_goal_position=dict(zip(ORDER,command.tolist())))
    q=owner.bus.q();q[2]=7.5;change_pose(owner,execution,q)
    physical_check=owner._physical_check
    def exact_shape(source,commands):
        assert np.asarray(commands).shape==(5,6)
        assert (np.abs(np.diff(np.vstack((source,commands)),axis=0))<=np.array([1.5,1.5,1.5,1.,1.,2.5])+1e-6).all()
        physical_check(source,commands)
    owner._physical_check=exact_shape
    result=run(owner,execution)
    assert result['original_target_reached'] and result['finishing_intervals'][J3]==[0.,7.5]
    assert result['command_envelope']['includes_original_q0']


def test_pending_axis_can_reach_with_a_small_quantization_overshoot_without_another_write():
    owner,execution=fixture();q=owner.bus.q();q[3]=.32;change_pose(owner,execution,q)
    moved=[]
    def naturally(register,motor):
        if register=='Present_Position' and owner.bus.values[J3]['I_Coefficient']==1 and len(moved)<3:
            owner.bus.values[J4]['Present_Position']-=8;moved.append(True)
    owner.bus.on_read=naturally
    result=run(owner,execution)
    assert result['touched_motors']==[J3] and result['original_goal_held']
    assert result['axes'][J4]['reason']=='already_reached_before_axis_write'
    assert result['axes'][J4]['completed'] and not result['axes'][J4]['applied']


def pending_shoulder_entry_fixture():
    owner,execution=three_axis_fixture()
    q=owner.bus.q();q[1]=-.23
    change_pose(owner,execution,q)
    execution['measured_q_path'][1:]=np.repeat(q[None],5,axis=0).tolist()
    return owner,execution


def test_pending_shoulder_recorded_shaped_two_cell_entry_requires_fresh_natural_arrival():
    """Replay the 953,954,954,953,955 shape using the fake bus's units."""
    owner,execution=pending_shoulder_entry_fixture();before=deepcopy(execution)
    path=iter((-.23,-.15,-.15,-.23,-.07))
    def naturally(register,motor):
        if register=='Present_Position' and owner.bus.values[J3]['I_Coefficient']==1:
            value=next(path,None)
            if value is not None:owner.bus.values[J2]['Present_Position']=round(10000+100*value)
    owner.bus.on_read=naturally
    result=run(owner,execution)
    entry,=result['pending_natural_target_band_entries']
    assert entry['motor']==J2 and entry['active_motor']==J3
    assert entry['phase']=='elbow_flex/candidate'
    assert entry['abs_step']==pytest.approx(.16) and entry['step_guard_exception']
    assert entry['previous_abs_error']>.2 and entry['abs_error']<.2
    assert not entry['stationary_arrival_claimed']
    natural=result['axes'][J2]['stages']['naturally_reached']
    assert natural['sample_count']==3 and natural['axis_arrival']['reached']
    assert all(row['readback_time_s']>entry['readback_time_s'] for row in natural['readbacks'])
    assert result['axes'][J2]['reason']=='already_reached_before_axis_write'
    assert not result['axes'][J2]['applied'] and result['touched_motors']==[J3,J4]
    assert execution==before and result['original_goal_held']
    assert all(row['goal_raw']==result['original_target_raw'] for row in natural['readbacks'])


@pytest.mark.parametrize('fault',['anchor','moving','goal','stop','geometry'])
def test_pending_target_entry_exception_preserves_motion_goal_and_anchor_guards(fault):
    owner,execution=pending_shoulder_entry_fixture();reads=[]
    def move(register,motor):
        if register!='Present_Position' or owner.bus.values[J3]['I_Coefficient']!=1:return
        reads.append(True)
        if fault=='second_entry':
            path=(-.07,-.15,-.23,-.07)
            if len(reads)<=len(path):owner.bus.values[J2]['Present_Position']=round(10000+100*path[len(reads)-1])
        elif len(reads)==1:
            value={'anchor':.04}.get(fault,-.07)
            owner.bus.values[J2]['Present_Position']=round(10000+100*value)
            if fault=='moving':owner.bus.values[J2]['Moving']=1
            if fault=='goal':owner.bus.values[J2]['Goal_Position']+=1
            if fault=='stop':owner.stop_event.set()
    owner.bus.on_read=move
    if fault=='geometry':
        def geometry():
            if reads:raise RuntimeError('pending target entry geometry rejected')
        owner.on_geometry=geometry
    match=('actual drift exceeded 0.25' if fault=='anchor' else
           'frozen Goal_Position changed' if fault=='goal' else
           'cancelled by STOP' if fault=='stop' else
           'pending target entry geometry rejected' if fault=='geometry' else
           'pending axis moved')
    result=rejected(owner,execution,match)
    entries=result.get('pending_natural_target_band_entries',[])
    assert len(entries)==(1 if fault=='second_entry' else 0)
    assert not result['axes'][J2]['completed'] and not result['original_target_reached']
    assert result['cleanup']['hold_performed'] and 'completion_follow' not in result


@pytest.mark.parametrize('path',[(-.39,),(-.10,),(-.07,-.15,-.23,-.07)])
def test_bounded_pending_axis_coupling_is_recorded_then_finished(path):
    owner,execution=pending_shoulder_entry_fixture();reads=[]
    def move(register,motor):
        if register!='Present_Position' or owner.bus.values[J3]['I_Coefficient']!=1:return
        if len(reads)<len(path):
            owner.bus.values[J2]['Present_Position']=round(10000+100*path[len(reads)])
        reads.append(True)
    owner.bus.on_read=move
    result=run(owner,execution)
    events=result['pending_axis_coupling_events']
    assert events and all(event['motor']==J2 and event['active_motor']==J3 for event in events)
    assert all(.1 < event['abs_step'] <= .25 for event in events)
    assert all(abs(event['fixed_anchor_offset']) <= .25 for event in events)
    assert result['axes'][J2]['completed'] and result['original_goal_held']
    assert result['completed'] and result['selected_axes_completed']


def test_j4_unknown_i0_cannot_prevent_j3_i0_or_allow_j4_damping_removal():
    owner,execution=fixture()
    def broken(register,motor,value):
        if motor==J4 and register=='I_Coefficient' and value==0:
            raise ConnectionError('J4 independent I0 readback unavailable')
    owner.bus.on_write=broken
    result=rejected(owner,execution)
    cleanup=[event for event in owner.bus.events if event['kind']=='write' and event['stopped']]
    assert [(event['motor'],event['register'],event['value']) for event in cleanup[:2]]==[
        (J4,'I_Coefficient',0),(J3,'I_Coefficient',0)]
    assert result['cleanup']['recovery_by_motor'][J3]['integral_zero_verified']
    assert not [event for event in cleanup if event['motor']==J4 and event['register']=='D_Coefficient']
    assert not [event for event in cleanup if event['register']=='Torque_Limit' and event['value']==1000]


@pytest.mark.parametrize('fault',['geometry','stop','changed_goal'])
def test_final_hold_keeps_geometry_stop_deadline_and_original_goal_guards_live(fault):
    owner,execution=fixture();restored=[];reads=[];fired=[]
    def written(register,motor,value):
        if (register,motor,value)==('Torque_Limit',J4,1000):restored.append(True)
    def sample(register,motor):
        if register=='Present_Position' and restored and not owner.stop_event.is_set():
            reads.append(True)
            if len(reads)==4 and fault!='geometry':
                fired.append(True)
                if fault=='stop':owner.stop_event.set()
                elif fault=='changed_goal':owner.bus.values[J3]['Goal_Position']+=1
    def geometry():
        if len(reads)==4 and fault=='geometry' and not fired:
            fired.append(True);raise RuntimeError('final actual geometry rejected')
    owner.bus.on_write=written;owner.bus.on_read=sample;owner.on_geometry=geometry
    result=rejected(owner,execution)
    assert fired and result['axes'][J4]['completed']
    assert not result['original_goal_hold']['completed'] and result['cleanup']['hold_performed']
    assert result['cleanup']['recovery_by_motor'][J3]['integral_zero_verified']
    assert result['cleanup']['recovery_by_motor'][J4]['integral_zero_verified']


def test_final_hold_records_scheduler_lateness_and_keeps_sampling_actual_state():
    owner,execution=fixture();restored=[];reads=[];injected=[]
    def written(register,motor,value):
        if (register,motor,value)==('Torque_Limit',J4,1000):restored.append(True)
    def sample(register,motor):
        if register=='Present_Position' and restored and not owner.stop_event.is_set():reads.append(True)
    def wait_pause():
        if restored and len(reads)>=3 and not injected:
            injected.append(True);owner.clock.now+=.071
    owner.bus.on_write=written;owner.bus.on_read=sample;owner.clock.on_wait=wait_pause
    result=run(owner,execution);hold=result['original_goal_hold']
    assert result['completed'] and hold['completed'] and injected
    assert hold['sample_count']==22 and hold['selected_axes_arrival']['reached']
    assert len(hold['scheduler_lateness_events'])==1
    event=hold['scheduler_lateness_events'][0]
    assert event['snapshot_started'] and event['scheduler_lateness_only']
    assert event['observed_lateness_s']==pytest.approx(.031)
    assert hold['maximum_observed_start_lateness_s']==pytest.approx(.031)
    assert 'deadline_miss' not in hold


def test_global_time_budget_includes_time_consumed_between_check_and_success_publication():
    import fcpg.v21_arm_position_servo as module
    owner,execution=fixture(); original=module._ArmRun.check
    injected=[]
    def check(run,*,recovery=False):
        original(run,recovery=recovery)
        if not recovery and run.report['original_goal_held'] and not injected:
            injected.append(True);owner.clock.now=run.control_started+45
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module._ArmRun,'check',check)
        result=rejected(owner,execution,'45-second')
    assert injected and result['control_exposure']['within_limit'] is False
    assert result['cleanup']['hold_performed']
    assert not result['selected_axes_completed'] and not result['selected_axes_held']
    assert result['original_goal_hold']['completed'] and result['original_goal_hold']['selected_axes_arrival']['reached']
    assert 'completion_follow' not in result and result['position_arrival_invalidated_reason']


def test_already_reached_then_sagging_cannot_reuse_admission_as_current_arrival():
    owner,execution=fixture();q=owner.bus.q();q[2:4]=[.1,.1];change_pose(owner,execution,q)
    samples=[]
    def sag(register,motor):
        if register=='Present_Position':
            samples.append(True)
            if len(samples)==2:owner.bus.values[J3]['Present_Position']+=24
    owner.bus.on_read=sag
    result=rejected(owner,execution)
    assert result['arrival_before_failure']['reached']
    assert not result['original_target_reached'] and not result['original_goal_held']
    assert 'original_target_arrival' not in result and result['position_arrival_invalidated_reason']
    assert result['readbacks'][-1]['q_position'][2]==.34
    assert result['touched_motors']==[] and result['cleanup']['hold_performed']


def test_shoulder_residual_above_point_two_finishes_last_without_widening_tolerance_or_changing_y():
    owner,execution=three_axis_fixture();before=deepcopy(execution)
    residual=owner.bus.q()[1]-execution['commands_position'][-1][1]
    assert residual==pytest.approx(-.204655956,abs=1e-8) and abs(residual)>.2
    result=run(owner,execution)
    assert result['axis_order']==[J3,J4,J2] and result['touched_motors']==[J3,J4,J2]
    assert result['original_goal_held'] and result['original_target_reached']
    writes=result['parameter_writes']
    assert [e['motor'] for e in writes if e['register']=='I_Coefficient' and e['value']==1]==[J3,J4,J2]
    for name in (J3,J4,J2):
        assert result['axes'][name]['completed'] and result['axes'][name]['integral_exposure']['within_limit']
        assert result['axes'][name]['stages']['candidate']['axis_arrival']['position_tolerance']==.2
        assert result['cleanup']['parameters_restored_by_motor'][name]
    # Shoulder y still records the five original reads, before any finishing.
    assert execution==before and all(row[1]==-.1 for row in execution['measured_q_path'][1:])
    assert result['original_target_arrival']['actual_position'][1]==0.
    assert result['baseline_parameters_by_motor']==result['final_parameters_by_motor']
    assert result['final_parameters_by_motor'][ORDER[5]]['Torque_Limit']==500
    assert not [e for e in writes if e['motor'] not in (J3,J4,J2)]
    hold=result['original_goal_hold']
    assert hold['completed'] and hold['sample_count']==22 and hold['actual_span_s']>=1
    assert hold['arrival']['quiet_samples']==22 and hold['arrival']['position_tolerance']==.2
    for previous,row in zip(hold['readbacks'],hold['readbacks'][1:]):
        assert row['scheduled_start_s']==pytest.approx(previous['returned_s']+.05)
        assert row['started_s']>=row['scheduled_start_s']
    assert result['control_exposure']['within_limit'] and result['control_exposure']['duration_s']<30


def test_all_stages_precheck_cartesian_offsets_with_execution_order_margins_and_target_centers():
    from itertools import product
    owner,execution=three_axis_fixture();result=run(owner,execution)
    order=(J3,J4,J2)
    initial=result['admission']['current_q'];target=result['original_target']
    for stage_index,name in enumerate(order):
        precheck=result['axes'][name]['corridor_precheck']
        others=[motor for motor in order if motor!=name]
        expected_offsets=[]
        for other in others:
            pending=order.index(other)>stage_index
            margin=.25 if pending else .2
            expected_offsets.append([-margin,0.,margin])
            assert precheck['other_axis_offsets'][other]==[-margin,0.,margin]
            assert precheck['other_axis_centers'][other]==(initial[ORDER.index(other)] if pending else target[ORDER.index(other)])
        assert precheck['other_axis_offset_combinations']==[list(values) for values in product(*expected_offsets)]
        assert len(precheck['other_axis_offset_combinations'])==9


@pytest.mark.parametrize('fault',['i1_ack','i0_ack','stop','timeout'])
def test_shoulder_failure_clears_its_i_and_independently_cleans_both_previous_axes(fault):
    owner,execution=three_axis_fixture();fired=[]
    def written(register,motor,value):
        if motor!=J2:return
        if register=='I_Coefficient' and value==1:
            if fault=='i1_ack':fired.append(True);raise ConnectionError('J2 I1 ACK lost')
            if fault=='stop':fired.append(True);owner.stop_event.set()
            if fault=='timeout':fired.append(True);owner.clock.now+=45
        if register=='I_Coefficient' and value==0 and fault=='i0_ack':
            fired.append(True);raise ConnectionError('J2 I0 ACK lost')
    owner.bus.on_write=written
    result=rejected(owner,execution)
    assert fired and result['touched_motors']==[J3,J4,J2]
    cleanup=[e for e in owner.bus.events if e['kind']=='write' and e['stopped']]
    assert [(e['motor'],e['register'],e['value']) for e in cleanup[:3]]==[
        (J2,'I_Coefficient',0),(J3,'I_Coefficient',0),(J4,'I_Coefficient',0)]
    for name in (J3,J4):assert result['cleanup']['recovery_by_motor'][name]['integral_zero_verified']
    assert result['cleanup']['recovery_by_motor'][J2]['integral_zero_verified'] is (fault!='i0_ack')
    if fault=='i0_ack':
        assert owner.bus.values[J2]['D_Coefficient']==32
        assert not [e for e in cleanup if e['motor']==J2 and e['register']=='D_Coefficient']
    expected_cap=1000 if result['cleanup']['restored'] else 200
    assert all(owner.bus.values[name]['Torque_Limit']==expected_cap for name in (J3,J4,J2))
    assert owner.bus.values[ORDER[5]]['Torque_Limit']==500
    assert result['cleanup']['hold_performed'] and not result['original_target_reached']


def test_pending_shoulder_cannot_accumulate_drift_outside_the_prechecked_corridor_across_stages():
    owner,execution=three_axis_fixture();moves={J3:0,J4:0}
    def read(register,motor):
        if register!='Present_Position' or owner.stop_event.is_set():return
        for active in (J3,J4):
            if owner.bus.values[active]['I_Coefficient']==1 and moves[active]<2:
                owner.bus.values[J2]['Present_Position']+=8;moves[active]+=1
    owner.bus.on_read=read
    result=rejected(owner,execution,'fixed admission corridor')
    assert moves=={J3:2,J4:2}
    assert result['touched_motors']==[J3,J4] and not result['axes'][J2]['applied']
    assert result['cleanup']['hold_performed']


@pytest.mark.parametrize('drift', [0., -.08, -.26])
def test_initially_in_band_later_joint_waits_for_its_own_sequential_arrival(drift):
    """Real failure: J2 residual -.136 -> -.218 at the first J3 I0 sample.

    Fake calibrated cells reproduce that one-cell boundary crossing; neither
    admission nor a pending sample is promoted to stationary arrival evidence.
    """
    owner,execution=three_axis_fixture();q=owner.bus.q();q[1]=-.03;q[3]=.12
    change_pose(owner,execution,q);before=deepcopy(execution);fired=[]
    def written(register,motor,value):
        if (register,motor,value)==('I_Coefficient',J3,0) and not fired and not owner.stop_event.is_set():
            fired.append(True)
            owner.bus.values[J2]['Present_Position']+=round(100*drift)
    owner.bus.on_write=written
    result=(rejected(owner,execution,'actual drift exceeded 0.25') if drift==-.26 else run(owner,execution))
    roles=result['preflight_classification']['final']
    assert roles['needs_finishing_axes']==[2] and roles['pending_axes']==[1,2,3]
    assert roles['completed_axes']==[]
    corridor=result['axes'][J3]['corridor_precheck']
    assert corridor['other_axis_offsets'][J2]==[-.25,0.,.25]
    assert corridor['other_axis_centers'][J2]==pytest.approx(-.03)
    assert fired and execution==before
    if drift==-.26:
        assert result['cleanup']['hold_performed'] and not result['selected_axes_completed']
        assert not result['axes'][J2]['applied']
    else:
        assert result['original_goal_held'] and result['axes'][J2]['completed']
        assert result['axes'][J2]['applied'] is (drift!=0.)
        if drift==0.:
            assert result['axes'][J2]['stages']['naturally_reached']['sample_count']==3
        else:
            assert result['axes'][J2]['stages']['restored_original_goal']['axis_arrival']['reached']
        assert result['axes'][J4]['stages']['naturally_reached']['sample_count']==3
        assert all(row['goal_raw']==result['original_target_raw'] for row in result['readbacks'])


def test_selected_axes_finish_with_real_j1_residual_and_truthful_full_six_axis_result():
    owner,execution=deferred_axis_fixture();before=deepcopy(execution)
    residual=owner.bus.q()[0]-execution['commands_position'][-1][0]
    assert residual==pytest.approx(.210835071,abs=1e-8) and residual>.2
    result=run(owner,execution)
    assert result['completed'] and result['applied'] and not result['skipped']
    assert result['selected_axes']==[2,3,1] and result['selected_axes_completed'] and result['selected_axes_held']
    assert not result['original_target_reached'] and not result['original_goal_held']
    assert result['reason']=='selected_axes_completed_with_other_axis_residual'
    assert result['deferred_axes']==[0] and result['deferred_anchor_q']==[.3,0.,3.06,.67,0.,30.]
    assert result['deferred_position_tolerance']==.25 and result['touched_motors']==[J3,J4]
    assert execution==before and owner.runtime.history==['untouched']
    assert result['original_goal_preserved'] and result['original_target']==before['commands_position'][-1]
    assert result['original_target_raw']==before['policy_goal_readbacks'][-1]['goal_raw']
    assert result['original_target_arrival']['residual_by_axis'][0]==pytest.approx(residual)
    assert all(owner.bus.values[name]['Goal_Position']==result['original_target_raw'][name] for name in ORDER)
    assert result['baseline_parameters_by_motor']==result['final_parameters_by_motor']
    assert result['final_parameters_by_motor'][ORDER[5]]['Torque_Limit']==500
    assert result['final_parameters_by_motor'][ORDER[5]]['Max_Torque_Limit']==500
    assert all(event['motor'] in (J3,J4) for event in result['parameter_writes'])
    hold=result['original_goal_hold'];follow=result['completion_follow']
    positions=[row['q_position'] for row in hold['readbacks']]
    times=[row['readback_time_s'] for row in hold['readbacks']]
    assert hold['completed'] and hold['sample_count']==22 and hold['actual_span_s']>=1.
    assert follow['positions']==positions and follow['times_s']==times
    assert (np.diff(times)>=.05-1e-9).all()
    expected_full=evaluate_position_arrival(result['original_target'],positions,
        np.asarray(times)-execution['source_time_s'],position_tolerance=.2,step_tolerance=.1,quiet_samples=22)
    expected_selected=evaluate_selected_position_arrival(result['original_target'],positions,
        np.asarray(times)-execution['source_time_s'],selected_axes=[2,3,1],
        position_tolerance=.2,step_tolerance=.1,quiet_samples=22)
    assert hold['arrival']==expected_full and not expected_full['reached']
    assert hold['selected_axes_arrival']==follow['selected_axes_arrival']==expected_selected
    assert expected_selected['reached'] and not expected_selected['all_axes_reached']
    assert follow['selected_axes']==[2,3,1] and follow['deferred_axes']==[0]
    assert follow['deferred_anchor_q']==result['deferred_anchor_q']
    assert follow['source_q0']==before['measured_q_path'][0]
    for motor in (J3,J4):
        precheck=result['axes'][motor]['corridor_precheck']
        assert precheck['other_axes'][-1]==0 and precheck['other_axis_centers'][ORDER[0]]==.3
        assert precheck['other_axis_offsets'][ORDER[0]]==[-.25,0.,.25]
        assert len(precheck['other_axis_offset_combinations'])==27
    expected_final=evaluate_position_arrival(result['original_target'],positions,
        np.asarray(times)-execution['source_time_s'],position_tolerance=.2,step_tolerance=.1,quiet_samples=3)
    for key,value in expected_final.items():assert result['original_target_arrival'][key]==value


def test_sampling_uses_previous_completion_and_slow_reads_do_not_fake_a_20hz_deadline_miss():
    owner,execution=fixture();q=owner.bus.q();q[3]=.1;change_pose(owner,execution,q)
    def slow_snapshot(register,motor):
        if register=='Present_Position' and not owner.stop_event.is_set():
            owner.clock.now+=.04
    owner.bus.on_read=slow_snapshot
    result=run(owner,execution)
    assert result['completed'] and result['touched_motors']==[J3]
    for name in ('cap_baseline','damping_baseline','candidate','integral_off_settling'):
        stage=result['axes'][J3]['stages'][name]
        assert stage['schedule_basis']=='previous_sample_completion_plus_0.05s'
        rows=stage['readbacks']
        assert [row['sample_index'] for row in rows]==list(range(1,len(rows)+1))
        for previous,row in zip(rows,rows[1:]):
            assert row['scheduled_start_s']==pytest.approx(previous['returned_s']+.05)
            assert row['started_s']>=row['scheduled_start_s']
        if len(rows)>1:
            assert np.diff([row['readback_time_s'] for row in rows]).min()>.08
    assert result['control_exposure']['duration_s']<30.


def test_candidate_scheduler_pause_is_telemetry_and_later_actual_samples_confirm_arrival():
    owner,execution=shoulder_scheduler_fixture();state=pause_during_shoulder_candidate(owner)
    result=run(owner,execution);candidate=result['axes'][J2]['stages']['candidate']
    assert state==dict(candidate_reads=3,injected=True)
    assert result['completed'] and result['touched_motors']==[J2]
    assert candidate['completed'] and candidate['sample_count']==3 and not candidate['axis_arrival']['reached']
    assert candidate['readbacks'][2]['q_position'][1]==pytest.approx(.07)
    assert abs(candidate['readbacks'][2]['q_position'][1]-candidate['readbacks'][1]['q_position'][1])>.1
    assert result['axes'][J2]['stages']['integral_off_settling']['axis_arrival']['reached']
    event=candidate['scheduler_lateness_events'][0]
    assert event['sample_index']==3 and event['snapshot_attempt']==1
    assert event['observed_lateness_s']==pytest.approx(.577)
    assert event['snapshot_started'] and event['scheduler_lateness_only']
    assert candidate['maximum_observed_start_lateness_s']==pytest.approx(.577)
    assert 'deadline_miss' not in candidate
    assert result['axes'][J2]['integral_exposure']['within_limit']
    assert result['control_exposure']['within_limit']
    assert all(row['goal_raw']==result['original_target_raw'] for row in candidate['readbacks'])


def test_integral_stops_at_first_real_band_entry_before_a_delayed_overshoot():
    """A recorded-shaped static residual needs effort, then immediate I0.

    The fake plant would cross the band on two I1 samples and overshoot on
    the next one, as in the saved shoulder trace. No physical claim is made.
    """
    owner,execution=shoulder_scheduler_fixture();owner.bus.steps[J2]=0
    q=owner.bus.q();q[1]=-.15;change_pose(owner,execution,q)
    execution['measured_q_path'][1:]=np.repeat(q[None],5,axis=0).tolist()
    sample_count=0
    def plant(register,motor):
        nonlocal sample_count
        if register=='Present_Position' and owner.bus.values[J2]['I_Coefficient']==1:
            sample_count+=1
            q1=(-.15 if sample_count<95 else -.07 if sample_count==95 else
                .18 if sample_count==96 else .34)
            owner.bus.values[J2]['Present_Position']=round(10000+100*q1)
    owner.bus.on_read=plant
    result=run(owner,execution);stages=result['axes'][J2]['stages']
    candidate=stages['candidate'];settled=stages['integral_off_settling']
    assert sample_count==95 and candidate['target_band_entry']['sample_index']==95
    assert not candidate['axis_arrival']['reached']
    assert candidate['stop_reason']=='first_actual_target_band_entry'
    assert settled['sample_count']==3 and settled['axis_arrival']['reached']
    assert all(row['parameters']==dict(P_Coefficient=16,I_Coefficient=0,
        D_Coefficient=32,Torque_Limit=300) for row in settled['readbacks'])
    assert all(row['started_s']>candidate['integral_zero_verified_s'] for row in settled['readbacks'])
    assert settled['readbacks'][0]['started_s']>candidate['readbacks'][-1]['returned_s']
    assert result['original_target_reached'] and result['original_goal_held']
    assert result['axes'][J2]['integral_exposure']['duration_s']<5.
    assert result['original_target_arrival']['actual_position'][1]==-.07
    assert result['original_target_arrival']['position_tolerance']==.2
    assert result['physical_task_success_verified'] is False


def test_first_band_entry_does_not_claim_arrival_when_i0_settles_outside_the_band():
    owner,execution=shoulder_scheduler_fixture();cleared=[]
    def drift(register,motor,value):
        if (register,motor,value)==('I_Coefficient',J2,0) and not owner.stop_event.is_set():
            cleared.append(True);owner.bus.values[J2]['Present_Position']=10034
    owner.bus.on_write=drift
    result=rejected(owner,execution,'did not settle after integral zero')
    stages=result['axes'][J2]['stages']
    assert cleared and stages['candidate']['target_band_entry']
    assert not stages['candidate']['axis_arrival']['reached']
    assert stages['integral_off_settling']['sample_count']==40
    assert not stages['integral_off_settling']['axis_arrival']['reached']
    assert result['original_target_reached'] is False and not result['original_goal_held']
    assert result['cleanup']['hold_performed'] and 'completion_follow' not in result


def encoder_rebound_fixture(*, landing_raw=3076, rebound_raw=3078, final_raw=3077,
                           synthetic_tick=None, initial_raw=3111, approach=None, source_position=None):
    """A fake plant with the recorded J3 calibration; it never opens hardware."""
    owner,execution=fixture();bus=owner.bus
    def decode(raw):return (200.*(raw-920)/2254-100. if synthetic_tick is None
                           else (raw-3075)*synthetic_tick)
    def encode(q):return (round(920+(float(q)+100.)*2254/200.) if synthetic_tick is None
                          else round(3075+float(q)/synthetic_tick))
    def normalize(values):
        assert owner.bus_lock._is_owned()
        return {key:decode(value) if key==3 else (value-10000)/100. for key,value in values.items()}
    def unnormalize(values):
        assert owner.bus_lock._is_owned()
        return {key:encode(value) if key==3 else round(10000+100*value) for key,value in values.items()}
    def raw_cells(q):
        raw=cells(q);raw[J3]=encode(np.float32(q[2]));return raw
    bus._normalize=normalize;bus._unnormalize=unnormalize
    bus.q=lambda:np.array([decode(bus.values[name]['Present_Position']) if name==J3 else
        (bus.values[name]['Present_Position']-10000)/100. for name in ORDER])
    bus.calibration[J3].range_min=920;bus.calibration[J3].range_max=3174
    target=np.asarray(execution['commands_position'][-1])
    target[2]=float(np.float32(91.25776672363281 if synthetic_tick is None else synthetic_tick*.5))
    source=target.copy();source[2]=decode(initial_raw)+1. if source_position is None else source_position
    commands=_ramp(source,target)
    execution['commands_position']=commands.tolist()
    execution['original_target_arrival']['original_target']=commands[-1].tolist()
    for receipt,command in zip(execution['policy_goal_readbacks'],commands):
        receipt.update(goal_raw=raw_cells(command),expected_goal_raw=raw_cells(command),
            requested_goal_position=dict(zip(ORDER,command.tolist())))
    present=raw_cells(target);present[J3]=initial_raw
    for name in ORDER:
        bus.values[name]['Present_Position']=present[name]
        bus.values[name]['Goal_Position']=raw_cells(target)[name]
        bus.values[name]['Present_Load']=24
    q=bus.q()
    execution['measured_q_path']=[source.tolist(),*np.repeat(q[None],5,axis=0).tolist()]
    execution['normal_follow']['positions']=np.repeat(q[None],40,axis=0).tolist()
    bus.steps[J3]=0
    state=dict(pulses=0,first_reads=0,correction_reads=0,i0_writes=0)
    if approach is None:approach=[3106,3102,3098,3097,3093,3090,3089,3085,3084,3083,3081,3077]
    def read(register,motor):
        if register!='Present_Position' or owner.stop_event.is_set() or bus.values[J3]['I_Coefficient']!=1:return
        if state['pulses']==1:
            state['first_reads']+=1
            bus.values[J3]['Present_Position']=approach[min(state['first_reads'],len(approach))-1]
        else:
            state['correction_reads']+=1
            bus.values[J3]['Present_Position']=landing_raw
    def write(register,motor,value):
        if motor!=J3 or register!='I_Coefficient' or owner.stop_event.is_set():return
        if value==1:state['pulses']+=1
        else:
            state['i0_writes']+=1
            bus.values[J3]['Present_Position']=rebound_raw if state['pulses']==1 else final_raw
    bus.on_read=read;bus.on_write=write
    return owner,execution,state


@pytest.mark.parametrize(('landing_raw','final_raw'),[(3076,3077),(3075,3076)])
def test_one_encoder_cell_i0_rebound_gets_one_inner_landing_then_original_baseline_hold(landing_raw,final_raw):
    owner,execution,state=encoder_rebound_fixture(landing_raw=landing_raw,final_raw=final_raw)
    before=deepcopy(execution);result=run(owner,execution)
    axis=result['axes'][J3];stages=axis['stages'];correction=axis['i0_rebound_correction']
    assert state==dict(pulses=2,first_reads=12,correction_reads=1,i0_writes=2)
    assert correction['eligible'] and correction['attempted'] and correction['settled_after_correction']
    assert correction['encoder_step_position']==pytest.approx(200/2254)
    assert correction['integral_cutoff_tolerance']==pytest.approx(.2-200/2254)
    assert correction['entry_residual']==pytest.approx(.1353122470859205)
    assert correction['settled_residual']==pytest.approx(.22404339171765741)
    assert not stages['integral_off_settling']['axis_arrival']['reached']
    assert stages['integral_off_settling']['sample_count']==40
    assert stages['rebound_candidate']['stop_reason']=='first_actual_inner_band_entry'
    assert stages['rebound_candidate']['target_band_entry']['actual_raw']==landing_raw
    assert not stages['rebound_candidate']['target_band_entry']['stationary_arrival_claimed']
    assert stages['rebound_integral_off_settling']['sample_count']==3
    exposure=axis['integral_exposure'];pulses=exposure['pulses']
    assert exposure['pulse_count']==2 and exposure['within_limit'] and exposure['all_pulses_i0_verified']
    assert exposure['duration_s']==pytest.approx(sum(pulse['duration_s'] for pulse in pulses))
    assert pulses[1]['maximum_s']==pytest.approx(20-pulses[0]['duration_s'])
    assert exposure['duration_s']<exposure['i0_verified_s']-exposure['started_s']-2.
    assert all(row['started_s']>pulses[1]['i0_verified_s']
        for row in stages['rebound_integral_off_settling']['readbacks'])
    assert result['original_target_reached'] and result['original_goal_held'] and result['final_equals_baseline']
    assert result['physical_task_success_verified'] is False and execution==before
    assert all(row['goal_raw']==result['original_target_raw'] for row in result['readbacks'])
    assert not [event for event in owner.bus.events if event['kind'] in ('position','halt')]
    assert all(owner.bus.values[J3][key]==value for key,value in
        dict(P_Coefficient=16,I_Coefficient=0,D_Coefficient=0,Torque_Limit=1000).items())


@pytest.mark.parametrize('rebound_raw',[3079,3073])
def test_larger_or_opposite_side_i0_rebound_never_reenters_i1(rebound_raw):
    owner,execution,state=encoder_rebound_fixture(rebound_raw=rebound_raw)
    result=rejected(owner,execution,'did not settle after integral zero')
    assert state['pulses']==1
    assert not result['axes'][J3]['i0_rebound_correction']['eligible']


@pytest.mark.parametrize(('landing_raw','match'),[
    (3073,'crossed_original_goal_cell'),(3079,'reversed_away_from_original_goal')])
def test_rebound_correction_clears_i_on_goal_cell_crossing_or_reversal(landing_raw,match):
    owner,execution,state=encoder_rebound_fixture(landing_raw=landing_raw)
    result=rejected(owner,execution,match)
    axis=result['axes'][J3]
    assert state['pulses']==2 and state['correction_reads']==1
    assert axis['stages']['rebound_candidate']['correction_rejected']
    assert axis['integral_exposure']['all_pulses_i0_verified']
    assert owner.bus.values[J3]['I_Coefficient']==0
    assert not result['original_target_reached'] and result['cleanup']['hold_performed']


def test_repeated_i0_rebound_fails_without_a_third_i1_pulse():
    owner,execution,state=encoder_rebound_fixture(final_raw=3078)
    result=rejected(owner,execution,'one bounded I0 rebound correction')
    assert state['pulses']==2 and state['i0_writes']==2
    axis=result['axes'][J3]
    assert axis['stages']['rebound_integral_off_settling']['sample_count']==40
    assert not axis['i0_rebound_correction']['settled_after_correction']
    assert not result['original_target_reached'] and not result['original_goal_held']


def test_inner_landing_precedes_goal_cell_crossing_for_the_opposite_approach_direction():
    owner,execution,state=encoder_rebound_fixture(landing_raw=3076,rebound_raw=3073,final_raw=3075,
        initial_raw=3039,source_position=87.,approach=[3044,3048,3052,3057,3061,3065,3069,3073,3074])
    result=run(owner,execution)
    correction=result['axes'][J3]['i0_rebound_correction']
    assert correction['raw_approach_side']==-1 and state['pulses']==2
    assert result['axes'][J3]['stages']['rebound_candidate']['stop_reason']=='first_actual_inner_band_entry'
    assert result['original_target_reached'] and result['original_goal_held']


def test_empty_quantized_inner_band_does_not_enable_a_second_pulse():
    owner,execution,state=encoder_rebound_fixture(synthetic_tick=.19,initial_raw=3091,
        approach=[3088,3085,3082,3079,3076],rebound_raw=3077)
    result=rejected(owner,execution,'no attainable inner landing cell')
    assert state['pulses']==1 and not result['axes'][J3]['i0_rebound_correction']['eligible']


def test_asymmetric_normalization_cannot_supply_a_rebound_landing_margin():
    owner,execution,state=encoder_rebound_fixture();normalize=owner.bus._normalize
    def asymmetric(values):
        result=normalize(values)
        if state['i0_writes']==1 and values[3]==3074:result[3]+=.004
        return result
    owner.bus._normalize=asymmetric
    result=rejected(owner,execution,'smaller symmetric landing band')
    assert state['pulses']==1 and not result['axes'][J3]['i0_rebound_correction']['eligible']


@pytest.mark.parametrize('fault',['moving','load','oscillating'])
def test_reentry_needs_fresh_stationary_low_output_i0_readbacks(fault):
    owner,execution,state=encoder_rebound_fixture();read=owner.bus.on_read;settling_reads=[]
    def corrupt(register,motor):
        read(register,motor)
        if state['i0_writes']==1 and not owner.stop_event.is_set() and register=='Present_Position':
            settling_reads.append(True)
            if fault=='moving':owner.bus.values[J3]['Moving']=1
            elif fault=='load':owner.bus.values[J3]['Present_Load']=33
            else:owner.bus.values[J3]['Present_Position']=3077 if len(settling_reads)%2 else 3078
    owner.bus.on_read=corrupt
    result=rejected(owner,execution,'baseline is not stationary' if fault=='moving' else
        'low-output evidence' if fault=='load' else 'stationary encoder cell')
    assert state['pulses']==1 and not result['original_target_reached']


@pytest.mark.parametrize('fault',['i1_ack','i0_ack','i0_readback'])
def test_second_pulse_write_uncertainty_is_counted_until_a_fresh_verified_i0(fault):
    owner,execution,state=encoder_rebound_fixture();write=owner.bus.on_write;fired=[]
    def corrupt(register,motor,value):
        write(register,motor,value)
        if (state['pulses']==2 and not fired and motor==J3 and register=='I_Coefficient'
                and value==(1 if fault=='i1_ack' else 0)):
            fired.append(True);owner.clock.now+=.1
            if fault=='i0_readback':owner.bus.values[J3]['I_Coefficient']=1
            else:raise ConnectionError('second pulse acknowledgement lost')
    owner.bus.on_write=corrupt
    result=rejected(owner,execution,'write was not verified' if fault=='i0_readback' else 'acknowledgement lost')
    exposure=result['axes'][J3]['integral_exposure'];pulses=exposure['pulses']
    assert fired and len(pulses)==2 and exposure['all_pulses_i0_verified']
    assert pulses[1]['duration_s']>=.1
    assert exposure['duration_s']==pytest.approx(sum(pulse['duration_s'] for pulse in pulses))
    assert pulses[1]['i0_verified_s']>pulses[0]['i0_verified_s']
    assert owner.bus.values[J3]['I_Coefficient']==0 and result['cleanup']['hold_performed']


def test_unverified_second_i0_never_reuses_first_pulse_clearance_or_restores_full_torque():
    owner,execution,state=encoder_rebound_fixture();write=owner.bus.on_write
    def refuse_i0(register,motor,value):
        write(register,motor,value)
        if (state['pulses']==2 and motor==J3 and (register,value)==('I_Coefficient',0)):
            owner.bus.values[J3]['I_Coefficient']=1
    owner.bus.on_write=refuse_i0
    result=rejected(owner,execution,'write was not verified')
    exposure=result['axes'][J3]['integral_exposure']
    assert len(exposure['pulses'])==2 and not exposure['all_pulses_i0_verified']
    assert exposure['within_limit'] is False and 'i0_verified_s' not in exposure
    assert 'i0_verified_s' in exposure['pulses'][0] and 'i0_verified_s' not in exposure['pulses'][1]
    assert not result['cleanup']['restored'] and owner.bus.values[J3]['Torque_Limit']==200


@pytest.mark.parametrize(('fault','match'),[
    ('stop','cancelled by STOP'),('goal','frozen Goal_Position changed'),
    ('geometry','reentry geometry rejected'),('other_axis','non-J3 actual drift'),
])
def test_second_pulse_keeps_stop_goal_geometry_and_other_axis_guards(fault,match):
    owner,execution,state=encoder_rebound_fixture();read=owner.bus.on_read
    def corrupt(register,motor):
        read(register,motor)
        if state['pulses']==2 and register=='Present_Position' and not owner.stop_event.is_set():
            if fault=='stop':owner.stop_event.set()
            elif fault=='goal':owner.bus.values[J3]['Goal_Position']+=1
            elif fault=='other_axis':owner.bus.values[ORDER[0]]['Present_Position']+=30
    owner.bus.on_read=corrupt
    if fault=='geometry':
        def geometry():
            if state['pulses']==2:raise RuntimeError('reentry geometry rejected')
        owner.on_geometry=geometry
    result=rejected(owner,execution,match)
    assert state['pulses']==2 and not result['original_target_reached']
    assert owner.bus.values[J3]['I_Coefficient']==0


@pytest.mark.parametrize(('delay','match'),[(20.1,'I1 wall deadline'),(45.1,'45-second control deadline')])
def test_second_pulse_cannot_reset_integral_or_total_control_deadlines(delay,match):
    owner,execution,state=encoder_rebound_fixture(landing_raw=3077);fired=[]
    def pause():
        if state['correction_reads']==1 and not fired:
            fired.append(True);owner.clock.now+=delay
    owner.clock.on_wait=pause
    result=rejected(owner,execution,match)
    exposure=result['axes'][J3]['integral_exposure'];pulses=exposure['pulses']
    assert fired and len(pulses)==2 and state['correction_reads']==1
    assert pulses[1]['maximum_s']==pytest.approx(20-pulses[0]['duration_s'])
    assert exposure['duration_s']>20 and not exposure['within_limit']
    assert exposure['all_pulses_i0_verified'] and owner.bus.values[J3]['I_Coefficient']==0
    assert result['control_exposure']['within_limit'] is (delay==20.1)


@pytest.mark.parametrize('register',['D_Coefficient','Torque_Limit'])
def test_successful_reentry_still_rejects_drift_during_original_baseline_restoration(register):
    owner,execution,state=encoder_rebound_fixture();write=owner.bus.on_write
    def drift(key,motor,value):
        write(key,motor,value)
        if (state['pulses']==2 and motor==J3 and key==register and
                value==(0 if register=='D_Coefficient' else 1000) and not owner.stop_event.is_set()):
            owner.bus.values[J3]['Present_Position']=3078
    owner.bus.on_write=drift
    result=rejected(owner,execution,'lost the original target during baseline restoration')
    assert state['pulses']==2 and not result['original_target_reached']


@pytest.mark.parametrize(('fault','match'),[
    ('stop','cancelled by STOP'),
    ('goal','frozen Goal_Position changed'),
    ('displacement','per-sample displacement exceeded'),
    ('geometry','post-pause geometry rejected'),
])
def test_candidate_scheduler_pause_does_not_bypass_actual_sample_guards(fault,match):
    owner,execution=shoulder_scheduler_fixture();state=None
    def action():
        if fault=='stop':owner.stop_event.set()
        elif fault=='goal':owner.bus.values[J2]['Goal_Position']+=1
        elif fault=='displacement':owner.bus.values[ORDER[0]]['Present_Position']+=76
    state=pause_during_shoulder_candidate(owner,action=action)
    if fault=='geometry':
        def geometry():
            if state['injected']:raise RuntimeError('post-pause geometry rejected')
        owner.on_geometry=geometry
    result=rejected(owner,execution,match)
    assert state['injected'] and state['candidate_reads']==(2 if fault=='stop' else 3)
    assert result['cleanup']['hold_performed'] and not result['original_target_reached']
    assert owner.bus.values[J2]['I_Coefficient']==0
    if fault!='stop':
        event=result['axes'][J2]['stages']['candidate']['scheduler_lateness_events'][0]
        assert event['observed_lateness_s']==pytest.approx(.577) and event['snapshot_started']


def test_scheduler_pause_crossing_i1_wall_deadline_starts_no_late_snapshot_and_clears_i():
    owner,execution=shoulder_scheduler_fixture()
    state=pause_during_shoulder_candidate(owner,delay=10.1)
    result=rejected(owner,execution,'I1 wall deadline')
    candidate=result['axes'][J2]['stages']['candidate']
    exposure=result['axes'][J2]['integral_exposure']
    assert state==dict(candidate_reads=2,injected=True)
    assert candidate['sample_count']==2
    assert all(row['started_s']<candidate['wall_deadline_s'] for row in candidate['readbacks'])
    event=candidate['scheduler_lateness_events'][0]
    assert not event['snapshot_started'] and event['blocked_by_i1_wall_deadline']
    assert event['observed_start_s']>=candidate['wall_deadline_s']
    assert exposure['duration_s']>10. and exposure['within_limit'] is False
    assert owner.bus.values[J2]['I_Coefficient']==0
    assert result['control_exposure']['within_limit']


def test_scheduler_pause_crossing_total_deadline_still_fails_at_forty_five_seconds():
    owner,execution=shoulder_scheduler_fixture()
    state=pause_during_shoulder_candidate(owner,delay=45.1)
    result=rejected(owner,execution,'45-second control deadline')
    candidate=result['axes'][J2]['stages']['candidate']
    assert state==dict(candidate_reads=2,injected=True) and candidate['sample_count']==2
    assert result['control_exposure']['duration_s']>=45.
    assert result['control_exposure']['within_limit'] is False
    assert owner.bus.values[J2]['I_Coefficient']==0


def test_one_sync_read_connection_failure_reacquires_one_complete_sample_with_partial_evidence():
    owner,execution=fixture();q=owner.bus.q();q[3]=.1;change_pose(owner,execution,q);failed=[]
    owner.bus.steps[J3]=20
    def drop(register,motor):
        if (register=='Moving' and motor=='all' and owner.bus.values[J3]['I_Coefficient']==1
                and not failed):
            failed.append(True);owner.clock.now+=.05045
            raise ConnectionError('injected candidate Moving packet timeout')
    owner.bus.on_read=drop
    result=run(owner,execution)
    candidate=result['axes'][J3]['stages']['candidate']
    discarded=[row for row in candidate['snapshot_attempts'] if row['discarded']]
    assert result['completed'] and failed and len(discarded)==1
    partial=discarded[0]
    attempts=[row for row in candidate['snapshot_attempts']
              if row['sample_index']==partial['sample_index']]
    assert [row['snapshot_attempt'] for row in attempts]==[1,2]
    assert attempts[1]['completed'] and not attempts[1]['discarded']
    assert attempts[1]['scheduled_start_s']==pytest.approx(partial['finished_s']+.05)
    assert partial['failed_register']=='Moving' and partial['retry_started']
    assert not partial['register_snapshot_complete'] and 'returned_s' not in partial
    assert partial['register_reads']['Present_Position']['values']
    assert partial['register_reads']['Goal_Position']['values']==result['original_target_raw']
    assert 'failed_s' in partial['register_reads']['Moving']
    assert candidate['sample_count']==len(candidate['readbacks'])
    assert all(event['num_retry']==1 for row in attempts
               for event in row['register_reads'].values())
    assert ('sync_read','Moving','all',1) in owner.bus.read_retries
    assert result['control_exposure']['duration_s']<30.


def test_final_hold_all_axis_load_timeout_reacquires_the_same_audited_sample():
    owner,execution=deferred_axis_fixture(selected_reached=True);failed=[]
    def drop(register,motor):
        if (register,motor)==('Present_Load','all') and not failed:
            failed.append(True);owner.clock.now+=.05045
            raise ConnectionError('injected final hold load packet timeout')
    owner.bus.on_read=drop
    result=run(owner,execution)
    hold=result['original_goal_hold']
    attempts=[row for row in hold['snapshot_attempts'] if row['sample_index']==1]
    assert result['completed'] and failed and len(attempts)==2
    assert attempts[0]['discarded'] and attempts[0]['failed_register']=='Present_Load'
    assert attempts[0]['register_reads']['Present_Load']['operation']=='sync_read'
    assert attempts[1]['completed'] and attempts[1]['all_present_load_raw']
    assert attempts[1]['scheduled_start_s']==pytest.approx(attempts[0]['finished_s']+.05)
    assert hold['sample_count']==len(hold['readbacks'])==22
    assert ('sync_read','Present_Load','all',1) in owner.bus.read_retries


def test_two_sync_read_connection_failures_stop_after_two_complete_snapshot_attempts():
    owner,execution=fixture();q=owner.bus.q();q[3]=.1;change_pose(owner,execution,q);failures=[]
    def drop(register,motor):
        if (register=='Moving' and motor=='all' and owner.bus.values[J3]['I_Coefficient']==1
                and len(failures)<2):
            failures.append(True);owner.clock.now+=.05045
            raise ConnectionError('repeated candidate Moving packet timeout')
    owner.bus.on_read=drop
    result=rejected(owner,execution,'repeated candidate')
    candidate=result['axes'][J3]['stages']['candidate']
    attempts=[row for row in candidate['snapshot_attempts'] if row['sample_index']==1]
    assert len(failures)==2 and [row['snapshot_attempt'] for row in attempts]==[1,2]
    assert all(row['discarded'] and not row['completed'] for row in attempts)
    assert all(row['failed_register']=='Moving' for row in attempts)
    assert attempts[1]['scheduled_start_s']==pytest.approx(attempts[0]['finished_s']+.05)
    assert candidate['maximum_snapshot_attempts']==2 and candidate['sample_count']==0


@pytest.mark.parametrize('fault',['sync_runtime','single_connection','constraint'])
def test_only_sync_read_connection_error_reacquires_a_complete_snapshot(fault):
    owner,execution=fixture();q=owner.bus.q();q[3]=.1;change_pose(owner,execution,q);fired=[]
    def fail(register,motor):
        active=owner.bus.values[J3]['I_Coefficient']==1
        if fired or not active:return
        if fault=='sync_runtime' and (register,motor)==('Moving','all'):
            fired.append(True);raise RuntimeError('injected nontransport sync failure')
        if fault=='single_connection' and (register,motor)==('Present_Load',J3):
            fired.append(True);raise ConnectionError('injected single-register read failure')
        if fault=='constraint' and (register,motor)==('Goal_Position','all'):
            fired.append(True);owner.bus.values[J3]['Goal_Position']+=1
    owner.bus.on_read=fail
    result=rejected(owner,execution)
    candidate=result['axes'][J3]['stages']['candidate']
    attempts=[row for row in candidate['snapshot_attempts'] if row['sample_index']==1]
    assert fired and len(attempts)==1
    assert attempts[0]['discarded'] and not attempts[0]['completed']
    assert attempts[0].get('retry_started') is not True
    assert attempts[0].get('failed_register') is None
    if fault=='single_connection':
        event=attempts[0]['register_reads']['Present_Load']
        assert event['operation']=='read' and event['num_retry']==1 and 'failed_s' in event
    if fault=='constraint':
        assert attempts[0]['register_snapshot_complete']


def test_unreached_candidate_stops_at_deadline_then_restores_cap_after_verified_hold():
    owner,execution=fixture();q=owner.bus.q();q[3]=.1;change_pose(owner,execution,q)
    owner.bus.steps[J3]=0
    result=rejected(owner,execution,'within the I1 pulse budget')
    candidate=result['axes'][J3]['stages']['candidate']
    exposure=result['axes'][J3]['integral_exposure']
    deadline=exposure['started_s']+20.
    assert deadline-.05<=candidate['wall_deadline_reached_s']<=deadline
    assert all(row['started_s']<deadline for row in candidate['readbacks'])
    assert 200<candidate['sample_count']<=400
    assert exposure['i0_verified_s']-exposure['started_s']<=20.
    assert owner.bus.values[J3]['I_Coefficient']==0
    assert owner.bus.values[J3]['Torque_Limit']==1000
    assert owner.bus.values[J3]['D_Coefficient']==0
    cleanup=result['cleanup']
    assert cleanup['restored'] and cleanup['profile_state']=='baseline1000'
    assert cleanup['pre_baseline_restore']['sample_count']==1
    assert cleanup['post_baseline_restore']['sample_count']==22
    assert cleanup['pre_baseline_restore']['actual_span_s']==0.
    assert cleanup['post_baseline_restore']['actual_span_s']>=1.
    assert result['axes'][J3]['integral_control']['maximum_cumulative_s']==20.
    assert exposure['pulses'][0]['maximum_s']==20.


def test_j3_integral_uses_verified_output_and_bounded_pulses_without_changing_goals():
    owner,execution=fixture();before=deepcopy(execution);enabled=[]
    def write(register,motor,value):
        if register=='I_Coefficient' and value==1:
            enabled.append((motor,owner.bus.values[motor]['Torque_Limit']))
    owner.bus.on_write=write
    result=run(owner,execution)
    assert enabled==[(J3,300),(J4,300)]
    assert result['axes'][J3]['integral_control']==dict(mode='stationary_release_pulses_v3',
        maximum_pulses=16,maximum_pulse_s=20.,maximum_cumulative_s=20.,
        physical_speed_bound_verified=False)
    candidate=result['axes'][J3]['stages']['candidate']
    assert all(row['parameters']['Torque_Limit']==300 for row in candidate['readbacks'])
    assert all(row['goal_raw']==result['original_target_raw'] for row in candidate['readbacks'])
    assert execution==before and result['original_goal_held'] and result['final_equals_baseline']


def test_j3_output_cap_must_read_back_before_integral_is_enabled():
    owner,execution=fixture()
    def lose_cap(register,motor,value):
        if motor==J3 and register=='Torque_Limit' and value==300:
            owner.bus.values[motor][register]=150
    owner.bus.on_write=lose_cap
    result=rejected(owner,execution,'Torque_Limit write was not verified')
    assert not [event for event in result['parameter_writes']
        if event['register']=='I_Coefficient' and event['value']==1]
    assert result['cleanup']['hold_performed'] and result['cleanup']['baseline_restored']


def test_j3_pulse_control_does_not_excuse_a_large_actual_step_even_on_goal_entry():
    """Recorded block3 shape, in fake encoder units; no speed-model claim."""
    owner,execution=fixture();q=owner.bus.q();q[2]=1.15;q[3]=.1
    change_pose(owner,execution,q);owner.bus.steps[J3]=0
    path=iter((1.15,.89,0.))
    def jump(register,motor):
        if register=='Present_Position' and owner.bus.values[J3]['I_Coefficient']==1:
            value=next(path,None)
            if value is not None:owner.bus.values[J3]['Present_Position']=round(10000+100*value)
    owner.bus.on_read=jump
    result=rejected(owner,execution,'actual per-sample displacement exceeded 0.75')
    failed=[row for row in result['readbacks'] if row.get('error')]
    assert failed[-1]['q_position'][2]==result['original_target'][2]==0.
    assert failed[-1]['parameters']['Torque_Limit']==300
    assert not result['axes'][J3]['completed'] and not result['original_target_reached']
    assert result['cleanup']['hold_performed'] and result['cleanup']['baseline_restored']


def stationary_release_fixture(*,step=20,reverse_after_i0=False):
    owner,execution=fixture();q=owner.bus.q();q[2]=.8;q[3]=.1
    change_pose(owner,execution,q);owner.bus.steps[J3]=0
    state=dict(pulse=0,reads=0,post_release_i1_reads=0)
    def write(register,motor,value):
        if motor!=J3 or register!='I_Coefficient':return
        if value==1:state.update(pulse=state['pulse']+1,reads=0)
        elif reverse_after_i0 and not owner.stop_event.is_set():
            owner.bus.values[J3]['Present_Position']+=step
    def read(register,motor):
        if register!='Present_Position' or owner.bus.values[J3]['I_Coefficient']!=1:return
        state['reads']+=1
        if state['reads']==4:
            owner.bus.values[J3]['Present_Position']-=step
        elif state['reads']>4:
            # Continuing I1 after breakaway would reproduce a dangerous jump.
            state['post_release_i1_reads']+=1
            owner.bus.values[J3]['Present_Position']-=89
    owner.bus.on_write=write;owner.bus.on_read=read
    return owner,execution,state


def test_stationary_integral_release_immediately_disables_i1_then_repeats_no_goals():
    """A fake stick/slip plant exercises the controller, not physical success."""
    owner,execution,state=stationary_release_fixture();before=deepcopy(execution)
    result=run(owner,execution);axis=result['axes'][J3]
    assert state['pulse']==3 and state['post_release_i1_reads']==0
    assert axis['integral_exposure']['pulse_count']==3
    assert axis['integral_exposure']['within_limit'] and result['original_goal_held']
    for index in (1,2):
        candidate=axis['stages']['candidate' if index==1 else 'release_candidate_2']
        settled=axis['stages']['integral_off_settling' if index==1 else 'release_settling_2']
        assert candidate['stop_reason']=='first_actual_motion_after_stationary_integral'
        assert not candidate['motion_release']['stationary_arrival_claimed']
        assert settled['sample_count']==3 and not settled['axis_arrival']['reached']
        assert all(row['started_s']>candidate['integral_zero_verified_s'] for row in settled['readbacks'])
        assert all(row['parameters']['I_Coefficient']==0 for row in settled['readbacks'])
    assert result['policy_commands_added']==0 and execution==before
    assert all(row['goal_raw']==result['original_target_raw'] for row in result['readbacks'])


def test_stationary_release_without_settled_progress_cannot_rearm_integral():
    owner,execution,state=stationary_release_fixture(reverse_after_i0=True)
    result=rejected(owner,execution,'lacks same-side settled progress')
    assert state['pulse']==1 and state['post_release_i1_reads']==0
    assert result['cleanup']['hold_performed'] and not result['original_target_reached']


def test_slow_first_release_and_more_than_ten_integral_seconds_require_actual_arrival():
    """Synthetic slow plant: time permits observation but never proves arrival."""
    from fcpg.v21_hardware_bridge import validate_arm_finishing_restoration
    owner,execution=fixture();q=owner.bus.q();q[2]=3.;q[3]=.1
    change_pose(owner,execution,q);owner.bus.steps[J3]=0;before=deepcopy(execution)
    waits=(64,29,5,30,32,51,46,48,32,36)
    state=dict(pulse=0,reads=0,post_release_i1_reads=0)
    def write(register,motor,value):
        if (register,motor,value)==('I_Coefficient',J3,1):
            state.update(pulse=state['pulse']+1,reads=0)
    def read(register,motor):
        if register!='Present_Position' or owner.bus.values[J3]['I_Coefficient']!=1:return
        state['reads']+=1;trigger=waits[state['pulse']-1]
        if state['reads']==trigger:owner.bus.values[J3]['Present_Position']-=30
        elif state['reads']>trigger:state['post_release_i1_reads']+=1
    owner.bus.on_write=write;owner.bus.on_read=read
    result=run(owner,execution);axis=result['axes'][J3];exposure=axis['integral_exposure']
    assert state['pulse']==10 and state['post_release_i1_reads']==0
    assert exposure['pulses'][0]['duration_s']>3.
    assert 10.<exposure['duration_s']<=20. and exposure['within_limit']
    assert exposure['all_pulses_i0_verified'] and result['control_exposure']['duration_s']<45.
    assert result['original_goal_held'] and result['original_target_arrival']['position_tolerance']==.2
    assert all(row['goal_raw']==result['original_target_raw'] for row in result['readbacks'])
    assert result['policy_commands_added']==0 and execution==before
    validate_arm_finishing_restoration(result)


@pytest.mark.parametrize('late_release', [True, False])
def test_progress_backed_j3_pulse_uses_remaining_budget_with_first_motion_cutoff(late_release):
    """Latest real trace's fourth pulse stalls at 2 s after three good releases.

    The delayed fourth release below is synthetic, not a physical prediction.
    The recorded 3122 -> 3118 -> 3117 -> 3115 progress establishes why another
    pulse may consume the remaining original budget, never a fresh ten seconds.
    """
    owner,execution=fixture();q=owner.bus.q();q[2]=.8;q[3]=.1
    change_pose(owner,execution,q);owner.bus.steps[J3]=0
    state=dict(pulse=0,reads=0,post_release_i1_reads=0)
    def write(register,motor,value):
        if (register,motor,value)==('I_Coefficient',J3,1):
            state.update(pulse=state['pulse']+1,reads=0)
    def read(register,motor):
        if register!='Present_Position' or owner.bus.values[J3]['I_Coefficient']!=1:return
        state['reads']+=1
        trigger=4 if state['pulse']<=3 else 45 if late_release else 10000
        if state['reads']==trigger:
            owner.bus.values[J3]['Present_Position']-=15 if state['pulse']<=3 else 20
        elif state['reads']>trigger:
            state['post_release_i1_reads']+=1
            owner.bus.values[J3]['Present_Position']-=89
    owner.bus.on_write=write;owner.bus.on_read=read
    result=run(owner,execution) if late_release else rejected(owner,execution,'within the I1 pulse budget')
    exposure=result['axes'][J3]['integral_exposure'];pulses=exposure['pulses']
    assert state['pulse']==4 and state['post_release_i1_reads']==0
    assert pulses[0]['maximum_s']==20.
    assert pulses[3]['duration_s']>2.
    assert pulses[3]['maximum_s']==pytest.approx(20.-sum(p['duration_s'] for p in pulses[:3]))
    assert pulses[3]['preceding_settled_progress']['settling_stage']=='release_settling_3'
    assert exposure['duration_s']<=20. and exposure['within_limit']
    assert result['control_exposure']['within_limit']
    assert all(row['parameters']['Torque_Limit']==300 for row in
        result['axes'][J3]['stages']['release_candidate_4']['readbacks'])
    assert result['original_goal_held'] is late_release
    if late_release:
        assert all(row['goal_raw']==result['original_target_raw'] for row in result['readbacks'])
        from fcpg.v21_hardware_bridge import validate_arm_finishing_restoration
        validate_arm_finishing_restoration(result)
        for damage in ('missing_progress','reversed_progress','unsettled','active_i1','too_long'):
            forged=deepcopy(result)
            axis=forged['axes'][J3];pulse=axis['integral_exposure']['pulses'][3]
            if damage=='missing_progress':pulse.pop('preceding_settled_progress')
            elif damage=='reversed_progress':pulse['preceding_settled_progress']['start_position']=0.
            elif damage=='unsettled':axis['stages']['release_settling_3']['readbacks'][-1]['moving'][J3]=1
            elif damage=='active_i1':axis['stages']['release_settling_3']['readbacks'][-1]['parameters']['I_Coefficient']=1
            elif damage=='too_long':pulse['i0_verified_s']=pulse['started_s']+20.1
            with pytest.raises(RuntimeError):validate_arm_finishing_restoration(forged)
    else:
        assert result['cleanup']['hold_performed'] and not result['original_target_reached']


def test_stationary_release_exhaustion_fails_without_an_unbounded_seventeenth_pulse():
    owner,execution,state=stationary_release_fixture(step=2)
    result=rejected(owner,execution,'within bounded release pulses')
    assert state['pulse']==16 and state['post_release_i1_reads']==0
    assert result['axes'][J3]['integral_exposure']['pulse_count']==16
    assert result['cleanup']['hold_performed'] and not result['original_target_reached']


def test_failed_hold_keeps_cap200_when_post_restore_quiet_proof_fails():
    owner,execution=fixture();q=owner.bus.q();q[3]=.1;change_pose(owner,execution,q)
    owner.bus.steps[J3]=0
    def corrupt_after_restore(register,motor,value):
        if (register,motor,value)==('Torque_Limit',J3,1000):
            owner.bus.values[J3]['Present_Load']=33
    owner.bus.on_write=corrupt_after_restore
    result=rejected(owner,execution,'within the I1 pulse budget')
    cleanup=result['cleanup']
    assert cleanup['pre_baseline_restore']['completed']
    assert not cleanup['post_baseline_restore']['completed']
    assert cleanup['baseline_restore_errors']
    assert not cleanup['restored'] and cleanup['profile_state']=='cleanup_cap200'
    assert owner.bus.values[J3]['Torque_Limit']==200


def test_already_reached_selected_axes_collect_full_hold_without_parameter_writes():
    owner,execution=deferred_axis_fixture(selected_reached=True);before=deepcopy(execution)
    result=run(owner,execution)
    assert result['completed'] and not result['applied'] and not result['skipped']
    assert result['selected_axes_completed'] and result['selected_axes_held']
    assert not result['original_target_reached'] and not result['original_goal_held']
    assert result['touched_motors']==[] and result['parameter_writes']==[]
    assert result['original_goal_hold']['sample_count']==22 and result['original_goal_hold']['actual_span_s']>=1.
    assert result['completion_follow']['selected_axes_arrival']['quiet_samples']==22
    assert result['hold_corridor_precheck']['axes']==[2,3,1,0]
    assert result['hold_corridor_precheck']['combinations']==81
    assert result['hold_corridor_precheck']['centers'][-1]==.3
    assert result['hold_corridor_precheck']['offsets'][-1]==[-.25,0.,.25]
    assert execution==before and not owner.halts and not owner.stop_event.is_set()


@pytest.mark.parametrize('timing',['before_hold','during_hold'])
def test_deferred_axis_naturally_reaching_does_not_force_full_six_axis_arrival_false(timing):
    owner,execution=deferred_axis_fixture();restored=[];samples=[];moved=[]
    def written(register,motor,value):
        if (register,motor,value)==('Torque_Limit',J4,1000):restored.append(True)
    def read(register,motor):
        if register!='Present_Position' or moved:return
        if restored:samples.append(True)
        if (timing=='before_hold' and owner.bus.values[J3]['I_Coefficient']==1
                or timing=='during_hold' and len(samples)==5):
            owner.bus.values[ORDER[0]]['Present_Position']-=8;moved.append(True)
    owner.bus.on_write=written;owner.bus.on_read=read
    result=run(owner,execution)
    assert moved and result['deferred_axes']==[0] and result['deferred_anchor_q'][0]==.3
    assert result['selected_axes_completed'] and result['selected_axes_held']
    assert result['original_target_reached'] and result['original_target_arrival']['quiet_samples']==3
    assert result['original_goal_held'] is (timing=='before_hold')
    hold=result['original_goal_hold']
    assert hold['arrival']['reached'] is (timing=='before_hold')
    assert hold['selected_axes_arrival']['all_axes_reached'] is (timing=='before_hold')


@pytest.mark.parametrize('axis',[0,4,5])
def test_initially_in_tolerance_untouched_axis_cannot_become_deferred_after_sag(axis):
    owner,execution=fixture();moves=[]
    def read(register,motor):
        if register=='Present_Position' and owner.bus.values[J3]['I_Coefficient']==1 and len(moves)<2:
            owner.bus.values[ORDER[axis]]['Present_Position']-=8;moves.append(True)
    owner.bus.on_read=read
    result=rejected(owner,execution,'completed or untouched axis')
    assert len(moves)==2 and result['deferred_axes']==[]
    assert not result['selected_axes_completed'] and not result['selected_axes_held']
    assert result['touched_motors']==[J3] and result['cleanup']['recovery_by_motor'][J3]['integral_zero_verified']
    assert result['cleanup']['hold_performed'] and 'completion_follow' not in result


def provisional_admission_fixture(*,source_j1=2.):
    owner,execution=deferred_axis_fixture(source_j1=source_j1)
    q=owner.bus.q();q[0]=.22
    change_pose(owner,execution,q)
    execution['measured_q_path'][1:]=np.repeat(q[None],5,axis=0).tolist()
    return owner,execution


def test_preflight_rebuilds_untouched_boundary_classification_with_complete_new_geometry():
    owner,execution=provisional_admission_fixture();before=deepcopy(execution);reads=[]
    def settle(register,motor):
        if register=='Present_Position' and not owner.stop_event.is_set():
            reads.append(True)
            if len(reads)==2:owner.bus.values[ORDER[0]]['Present_Position']=10030
    owner.bus.on_read=settle
    result=run(owner,execution)
    receipt=result['preflight_classification'];passes=result['admission_geometry_passes']
    assert receipt['provisional']['deferred_axes']==[]
    assert receipt['committed'] and receipt['final']['deferred_axes']==[0]
    assert len(receipt['rebuilds'])==1 and len(passes)==2
    assert receipt['rebuilds'][0]['before_any_parameter_write']
    assert receipt['rebuilds'][0]['stationary_arrival_claimed'] is False
    assert result['deferred_anchor_q'][0]==.3 and result['deferred_axes']==[0]
    for window in receipt['verification_windows']:
        assert window['completed'] and window['sample_count']==3
        assert all(row['phase']=='admission_after_geometry' for row in window['readbacks'])
        assert all(not any(row['moving'].values()) for row in window['readbacks'])
    assert all(e['started_s']>receipt['committed_s'] for e in result['parameter_writes'])
    assert 0 not in passes[0]['corridor_prechecks'][J3]['other_axes']
    corridor=passes[1]['corridor_prechecks'][J3]
    assert 0 in corridor['other_axes'] and corridor['other_axis_centers'][ORDER[0]]==.3
    assert corridor['other_axis_offsets'][ORDER[0]]==[-.25,0.,.25]
    assert len(corridor['other_axis_offset_combinations'])==27
    assert result['selected_axes_completed'] and result['selected_axes_held']
    assert not result['original_target_reached'] and not result['original_goal_held']
    assert result['original_target_arrival']['residual_by_axis'][0]>.2
    assert execution==before and result['touched_motors']==[J3,J4]


def test_preflight_rebuild_adds_a_selected_axis_that_settled_outside_the_target_band():
    owner,execution=fixture();q=owner.bus.q();q[3]=.27;change_pose(owner,execution,q)
    execution['measured_q_path'][1:]=np.repeat(q[None],5,axis=0).tolist();reads=[]
    def settle(register,motor):
        if register=='Present_Position' and not owner.stop_event.is_set():
            reads.append(True)
            if len(reads)==2:owner.bus.values[J4]['Present_Position']=10035
    owner.bus.on_read=settle
    result=run(owner,execution);receipt=result['preflight_classification']
    assert receipt['provisional']['pending_axes']==[2,3]
    assert receipt['provisional']['needs_finishing_axes']==[2]
    assert receipt['final']['needs_finishing_axes']==[2,3] and len(receipt['rebuilds'])==1
    assert J4 in result['admission_geometry_passes'][0]['corridor_prechecks']
    assert J4 in result['admission_geometry_passes'][1]['corridor_prechecks']
    assert result['axes'][J4]['applied'] and result['original_goal_held']


@pytest.mark.parametrize('fault',['moving','step','hull','goal','baseline','identity',
    'unstable_roles','second_change','new_geometry'])
def test_preflight_reclassification_cannot_bypass_fresh_state_and_geometry(fault):
    owner,execution=provisional_admission_fixture(source_j1=.25 if fault=='hull' else 2.);reads=[]
    def settle(register,motor):
        if register!='Present_Position' or owner.stop_event.is_set():return
        reads.append(True)
        if len(reads)==2:
            owner.bus.values[ORDER[0]]['Present_Position']=10030
            if fault=='moving':owner.bus.values[ORDER[0]]['Moving']=1
            if fault=='goal':owner.bus.values[ORDER[0]]['Goal_Position']+=1
            if fault=='baseline':owner.bus.values[J4]['D_Coefficient']=1
            if fault=='identity':owner.bus.values[J2]['ID']=7
        if fault=='step' and len(reads)==3:owner.bus.values[ORDER[0]]['Present_Position']=10041
        if fault=='unstable_roles' and len(reads)==3:owner.bus.values[ORDER[0]]['Present_Position']=10022
        if fault=='second_change' and len(reads)==5:owner.bus.values[ORDER[0]]['Present_Position']=10022
    owner.bus.on_read=settle
    if fault=='new_geometry':
        physical_check=owner._physical_check
        def check(q,commands):
            if len(reads)>=4 and np.isclose(q[0],.55) and np.isclose(q[3],.92):
                raise RuntimeError('new deferred joint corner rejected')
            physical_check(q,commands)
        owner._physical_check=check
    match={'moving':'actual baseline changed','step':'actual baseline changed',
        'hull':'outside the original verified command hull','goal':'frozen Goal_Position changed',
        'baseline':'actual parameters differ','identity':'pinned ID',
        'unstable_roles':'classification changed during fresh verification',
        'second_change':'classification changed a second time',
        'new_geometry':'new deferred joint corner rejected'}[fault]
    result=rejected(owner,execution,match)
    assert not result['preflight_classification']['committed']
    assert result['parameter_writes']==[] and result['touched_motors']==[]
    assert not result['selected_axes_completed'] and not result['selected_axes_held']
    assert result['cleanup']['hold_performed']
    if fault in ('second_change','new_geometry'):
        assert len(result['preflight_classification']['rebuilds'])==1


@pytest.mark.parametrize('mutation',['control_started','parameter_write'])
def test_preflight_reclassification_is_forbidden_after_any_control_mutation(mutation):
    from fcpg.v21_arm_position_servo import _ArmRun
    owner,execution=fixture();report={'axes':{},'parameter_writes':[]}
    servo=_ArmRun(owner,execution,report,owner.stop_event.is_set,owner.clock,owner.clock.wait)
    if mutation=='control_started':servo.control_started=owner.clock()
    else:report['parameter_writes'].append({'may_have_applied':False})
    with pytest.raises(RuntimeError,match='cannot change after a parameter write'):
        servo.require_unwritten_preflight()
    assert not owner.bus.events


@pytest.mark.parametrize('fault',['anchor','step','moving'])
def test_deferred_axis_actual_motion_aborts_and_preserves_historical_readbacks(fault):
    owner,execution=deferred_axis_fixture();before=deepcopy(execution);moves=[]
    def read(register,motor):
        if register!='Present_Position' or owner.stop_event.is_set() or owner.bus.values[J3]['I_Coefficient']!=1:return
        values=owner.bus.values[ORDER[0]]
        if fault=='anchor' and len(moves)<4:values['Present_Position']+=8;moves.append(True)
        elif fault=='step' and not moves:values['Present_Position']+=11;moves.append(True)
        elif fault=='moving' and not moves:values['Moving']=1;moves.append(True)
    owner.bus.on_read=read
    result=rejected(owner,execution)
    assert moves and result['deferred_axes']==[0] and result['deferred_anchor_q'][0]==.3
    assert not result['selected_axes_completed'] and not result['selected_axes_held']
    assert not result['original_target_reached'] and not result['original_goal_held']
    assert result['arrival_before_failure']['residual_by_axis'][0]==pytest.approx(.210835071,abs=1e-8)
    assert result['readbacks'][-1]['q_position'][0]==pytest.approx(.62 if fault=='anchor' else .41 if fault=='step' else .3)
    failed=[row for row in result['readbacks'] if row.get('moving',{}).get(ORDER[0])]
    assert bool(failed) is (fault=='moving')
    if failed:assert failed[-1]['phase']=='elbow_flex/candidate'
    assert result['cleanup']['recovery_by_motor'][J3]['integral_zero_verified']
    assert owner.bus.values[J3]['I_Coefficient']==0 and owner.bus.values[J3]['Torque_Limit']==1000
    assert result['cleanup']['hold_performed'] and 'completion_follow' not in result
    assert execution==before and not [e for e in result['parameter_writes'] if e['motor']==ORDER[0]]


def test_deferred_axis_must_stay_in_original_verified_command_hull_or_original_target_band():
    owner,execution=deferred_axis_fixture(source_j1=.31);moved=[]
    def read(register,motor):
        if register=='Present_Position' and owner.bus.values[J3]['I_Coefficient']==1 and not moved:
            owner.bus.values[ORDER[0]]['Present_Position']+=8;moved.append(True)
    owner.bus.on_read=read
    result=rejected(owner,execution,'original verified command hull and target band')
    assert moved and result['readbacks'][-1]['q_position'][0]==.38
    assert result['deferred_anchor_q'][0]==.3 and result['cleanup']['hold_performed']


def test_deferred_fixed_anchor_cannot_reset_between_selected_axis_stages():
    owner,execution=deferred_axis_fixture();moves={J3:0,J4:0}
    def read(register,motor):
        if register!='Present_Position' or owner.stop_event.is_set():return
        for active in (J3,J4):
            if owner.bus.values[active]['I_Coefficient']==1 and moves[active]<2:
                owner.bus.values[ORDER[0]]['Present_Position']+=8;moves[active]+=1
    owner.bus.on_read=read
    result=rejected(owner,execution,'fixed stationary admission corridor')
    assert moves=={J3:2,J4:2} and result['deferred_anchor_q'][0]==.3
    assert result['readbacks'][-1]['q_position'][0]==.62
    assert result['touched_motors']==[J3,J4] and result['axes'][J3]['completed']
    assert not result['selected_axes_completed'] and not result['selected_axes_held']
    assert all(result['cleanup']['recovery_by_motor'][name]['integral_zero_verified'] for name in (J3,J4))


def test_deferred_corner_is_prechecked_jointly_with_both_other_selected_axes_before_writes():
    owner,execution=deferred_axis_fixture();physical_check=owner._physical_check;rejected_corner=[]
    def check(source,commands):
        if np.allclose(np.asarray(source)[[0,1,3]],[.55,.3,.92],atol=1e-7):
            rejected_corner.append(np.asarray(source).tolist())
            raise RuntimeError('combined deferred and both selected-axis corner rejected')
        physical_check(source,commands)
    owner._physical_check=check
    result=rejected(owner,execution,'combined deferred and both selected-axis corner')
    assert rejected_corner and result['parameter_writes']==[] and result['touched_motors']==[]
    assert result['deferred_anchor_q'][0]==.3 and result['cleanup']['hold_performed']
    assert not result['selected_axes_completed'] and not result['selected_axes_held']


def test_geometry_cache_uses_exact_bytes_dtype_and_shape_without_rounding():
    from types import SimpleNamespace
    from fcpg.v21_arm_position_servo import _ExactGeometryCache
    calls=[]
    owner=SimpleNamespace(_physical_check=lambda q,c:calls.append((q.copy(),c.copy())))
    cache=_ExactGeometryCache(owner,lambda:('fixed_test_geometry',))
    q=np.zeros(6);commands=np.zeros((5,6))
    cache(q,commands);cache(q.copy(),commands.copy())
    assert len(calls)==1 and cache.receipt['success_hits']==1
    changed_q=q.copy();changed_q[0]=np.nextafter(0.,1.)
    cache(changed_q,commands)
    changed_commands=commands.copy();changed_commands[0,0]=np.nextafter(0.,1.)
    cache(q,changed_commands)
    cache(q.astype(np.float32),commands);cache(q,commands.astype(np.float32))
    cache(q[None],commands)
    assert len(calls)==6 and cache.receipt['unique_successes']==6
    another=_ExactGeometryCache(owner,lambda:('fixed_test_geometry',))
    another(q,commands)
    assert len(calls)==7 and another.receipt['success_hits']==0


@pytest.mark.parametrize('fault',['reject','mutate_input','mutate_configuration'])
def test_geometry_cache_never_remembers_rejection_changed_inputs_or_changed_configuration(fault):
    from types import SimpleNamespace
    from fcpg.v21_arm_position_servo import _ExactGeometryCache
    calls=[];configuration=[0]
    def checker(q,commands):
        calls.append(True)
        if fault=='reject':raise RuntimeError('rejected test geometry')
        if fault=='mutate_input':commands[0,0]+=1.
        if fault=='mutate_configuration':configuration[0]+=1
    cache=_ExactGeometryCache(SimpleNamespace(_physical_check=checker),lambda:tuple(configuration))
    for _ in range(2):
        configuration[0]=0
        with pytest.raises(RuntimeError):cache(np.zeros(6),np.zeros((5,6)))
    assert len(calls)==2 and cache.receipt['unique_successes']==0 and cache.receipt['success_hits']==0


def test_unrecognized_geometry_checker_is_always_uncached():
    from fcpg.v21_arm_position_servo import _ArmRun
    owner,execution=fixture()
    servo=_ArmRun(owner,execution,{'axes':{}},owner.stop_event.is_set,owner.clock,owner.clock.wait)
    assert servo.geometry_cache.receipt['enabled'] is False
    q=owner.bus.q();commands=np.repeat(q[None],5,axis=0)
    servo.geometry(q,commands);servo.geometry(q,commands)
    assert len(owner.paths)==2 and servo.geometry_cache.receipt['raw_checks']==2
    assert servo.geometry_cache.receipt['success_hits']==0


def test_cached_geometry_preserves_every_segment_stalled_variant_and_live_clearance_check():
    from fcpg.v21_arm_position_servo import _ArmRun,_ExactGeometryCache,_path
    traces=[];runs=[]
    for enabled in (False,True):
        owner,execution=fixture()
        servo=_ArmRun(owner,execution,{'axes':{}},owner.stop_event.is_set,owner.clock,owner.clock.wait)
        servo.target=np.asarray(execution['commands_position'][-1])
        cache=_ExactGeometryCache(owner,(lambda:('fixed_test_geometry',)) if enabled else None)
        probes=[]
        class RecordingCache:
            validate=cache.validate
            def __call__(self,q,commands):
                probes.append((q.copy(),commands.copy()))
                return cache(q,commands)
        servo.geometry_cache=RecordingCache()
        q=owner.bus.q()
        for value in (q[2],q[2],q[2]-.08,q[2]-.16):
            q[2]=value;servo.geometry(q,_path(q,servo.target),stalled=True)
        traces.append(probes);runs.append((owner,servo,cache))
    assert len(traces[0])==len(traces[1])
    for (source1,commands1),(source2,commands2) in zip(*traces):
        assert np.array_equal(source1,source2) and np.array_equal(commands1,commands2)
    assert len(runs[0][0].geometry)==len(runs[1][0].geometry)
    cached_owner,cached_servo,cache=runs[1]
    assert cache.receipt['success_hits']>0 and len(cached_owner.paths)<len(runs[0][0].paths)
    cached_owner.clearance.require_readback=lambda q,**kwargs:(_ for _ in ()).throw(RuntimeError('live clearance rejected'))
    count=cache.receipt['success_hits']
    with pytest.raises(RuntimeError,match='live clearance rejected'):
        cached_servo.geometry(q,_path(q,cached_servo.target),stalled=True)
    assert cache.receipt['success_hits']==count


@pytest.fixture(scope='module')
def real_geometry_kinematics():
    pytest.importorskip('mujoco')
    from fcpg.so101_kinematics import SO101MuJoCoKinematics
    from scripts.serve_so101_prod_v17_7 import DEFAULT_KINEMATICS_MODEL
    if not DEFAULT_KINEMATICS_MODEL.is_file():pytest.skip('offline official MuJoCo model is unavailable')
    return SO101MuJoCoKinematics(DEFAULT_KINEMATICS_MODEL)


@pytest.mark.parametrize('changed',['gate_identity','gate_threshold','kinematics_identity','data_identity',
    'lock_identity','body_pose','joint_axis','qpos0','ranges','tip','qpos_addresses','audit_alias',
    'normalizer','recovery_alias','hard_limit','forward_function','mujoco_function'])
def test_real_geometry_binding_rejects_changed_fk_gate_and_actual_global_aliases(real_geometry_kinematics,monkeypatch,changed):
    from dataclasses import replace
    from fcpg import prod_v20_3_hardware_service as hardware,so101_kinematics as fk
    from fcpg.v21_arm_position_servo import _ExactGeometryCache,_geometry_cache_binding
    import scripts.serve_so101_prod_v17_7 as gate_module
    kin=real_geometry_kinematics
    owner,_=fixture();owner.kinematics=kin
    owner.clearance=gate_module.V177ClearanceGate(kinematics=kin)
    owner._physical_check=hardware.V203HardwareService._physical_check.__get__(owner)
    binding=_geometry_cache_binding(owner)
    assert binding is not None
    cache=_ExactGeometryCache(owner,binding)
    q=np.array([-7.225433526011557,-45.574310415808974,71.51730257320318,
        56.87919463087246,-64.51948051948052,26.28120893561104])
    commands=np.repeat(q[None],5,axis=0)
    cache(q,commands);cache(q,commands)
    assert cache.receipt['raw_checks']==1 and cache.receipt['success_hits']==1
    restored=[]
    try:
        if changed=='gate_identity':owner.clearance=replace(owner.clearance)
        elif changed=='gate_threshold':owner.clearance=replace(owner.clearance,table_z_mm=owner.clearance.table_z_mm+.01)
        elif changed=='kinematics_identity':owner.kinematics=object()
        elif changed=='data_identity':monkeypatch.setattr(kin,'data',kin._mujoco.MjData(kin.model))
        elif changed=='lock_identity':monkeypatch.setattr(kin,'_forward_lock',threading.RLock())
        elif changed in ('body_pose','joint_axis','qpos0','ranges','tip','qpos_addresses'):
            array={'body_pose':kin.model.body_pos,'joint_axis':kin.model.jnt_axis,'qpos0':kin.model.qpos0,
                'ranges':kin.all_joint_ranges,'tip':kin.fixed_tip_local,'qpos_addresses':kin.all_qpos_addresses}[changed]
            old=array.copy();restored.append((array,old));array.flat[0]+=1 if changed=='qpos_addresses' else .000001
        elif changed=='audit_alias':monkeypatch.setattr(hardware,'_audit_v177_chunk',lambda *a,**k:{'safe':True})
        elif changed=='normalizer':monkeypatch.setattr(fk,'normalized_arm_positions_to_radians',lambda *a:None)
        elif changed=='recovery_alias':monkeypatch.setattr(gate_module,'_nonmonotonic_recovery_joints',lambda *a:())
        elif changed=='hard_limit':
            bounds=gate_module.SO101_POSITION_LOWER.copy();bounds[0]+=.01
            monkeypatch.setattr(gate_module,'SO101_POSITION_LOWER',bounds)
        elif changed=='forward_function':monkeypatch.setattr(type(kin),'forward_anchors',lambda *a:None)
        elif changed=='mujoco_function':monkeypatch.setattr(kin._mujoco,'mj_kinematics',lambda *a:None)
        with pytest.raises(RuntimeError,match='configuration changed'):cache(q,commands)
        assert cache.receipt['raw_checks']==1 and cache.receipt['success_hits']==1
    finally:
        for array,old in restored:array[:]=old
