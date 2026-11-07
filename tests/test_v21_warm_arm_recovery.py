"""Read-only maintenance reconnect is bound to the same incomplete ARM output."""
from copy import deepcopy
import hashlib
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from fcpg.v21_hardware_state import ORDER,connect_preserving_warm_hold,verified_hold,existing_owner_connection_initializer
from test_v21_hardware_state import io_fixture


def failed_arm_source_baseline1000():
    from test_v21_arm_position_servo import J3,change_pose,fixture,rejected
    owner,source=fixture()
    q=owner.bus.q();q[3]=.1;change_pose(owner,source,q)
    owner.bus.steps[J3]=0
    source['position_servo']=rejected(owner,source,'within the I1 pulse budget')
    exposure=source['position_servo']['axes'][J3]['integral_exposure']
    assert exposure['pulse_count']==1 and exposure['all_pulses_i0_verified']
    assert exposure['within_limit'] and exposure['duration_s']<=2.
    source['requires_home_after_incomplete_execution']=True
    assert source['position_servo']['cleanup']['profile_state']=='baseline1000'
    return source


def write_incomplete_source(tmp_path,source):
    (tmp_path/'incomplete-execution.json').write_text(json.dumps({'requires_home':True}))
    (tmp_path/'last-attempt.json').write_text(json.dumps(source))


def io_for_failed_source(source,*,profile_state,goal_raw=None,held_position=None):
    io,bus=io_fixture()
    profile=source['position_servo']['baseline_parameters_by_motor']
    profile={name:dict(values) for name,values in profile.items()}
    if profile_state=='cleanup_cap200':
        for name in source['position_servo']['touched_motors']:
            profile[name]['Torque_Limit']=200
    elif profile_state!='baseline1000':
        raise ValueError('test fixture needs an explicit evidenced profile')
    bus.profile=profile
    bus.bad['J3_Torque_Limit']=profile['elbow_flex']['Torque_Limit']
    original=source['position_servo']['cleanup']['hold']['attempts'][-1]
    goal_raw=original['requested_goal_raw'] if goal_raw is None else goal_raw
    held_position=original['held_state'] if held_position is None else held_position
    bus.goal=np.array([(goal_raw[name]-10000)/100 for name in ORDER],dtype=float)
    np.testing.assert_allclose(bus.goal,np.asarray(held_position),atol=1e-8,rtol=0)
    bus.bias=np.zeros(6);bus.bias[2]=.01
    bus.calibration={name:SimpleNamespace(range_min=0,range_max=20000) for name in ORDER}
    bus._normalize=lambda values:{key:(raw-10000)/100. for key,raw in values.items()}
    bus._unnormalize=lambda values:{key:round(10000+100*q) for key,q in values.items()}
    io.calibration_reconstructed_encoder_cells=lambda q:np.rint(10000+100*np.asarray(q))
    previous=bus.sync_read
    def read(register,normalize=True,num_retry=0):
        if register in ('Present_Position','Goal_Position'):
            q=bus.goal+(bus.bias if register=='Present_Position' else 0)
            return dict(zip(ORDER,(q if normalize else np.rint(10000+100*q)).tolist()))
        if register in next(iter(profile.values())):
            return {name:values[register] for name,values in profile.items()}
        if register=='ID':return {name:index+1 for index,name in enumerate(ORDER)}
        defaults={'Model_Number':777,'Firmware_Major_Version':3,'Firmware_Minor_Version':10,
            'Lock':1,'Present_Load':0,'Status':0,'Present_Velocity':0}
        if register in defaults:return {name:bus.bad.get(register,defaults[register]) for name in ORDER}
        return previous(register,normalize=normalize,num_retry=num_retry)
    bus.sync_read=read
    return io,bus,source


def incomplete_arm_io(tmp_path):
    source=failed_arm_source_baseline1000()
    write_incomplete_source(tmp_path,source)
    return io_for_failed_source(source,profile_state='baseline1000')


def cap200_source(tmp_path,*,legacy=False):
    from test_v21_position_servo_recovery import setup
    source=deepcopy(setup(tmp_path).last_attempt)
    cleanup=source['position_servo']['cleanup']
    assert cleanup['profile_state']=='cleanup_cap200'
    if legacy:
        cleanup.pop('profile_state')
    write_incomplete_source(tmp_path,source)
    return source


def recovery_lineage(tmp_path,*,completed):
    from fcpg.v21_position_servo_recovery import PositionServoRecoveryError
    from test_v21_position_servo_recovery import J3,run,setup
    service=setup(tmp_path)
    source=deepcopy(service.last_attempt)
    bus=service.owner.bus
    bus.values[J3]['Present_Position']=bus.values[J3]['Goal_Position']+7
    with pytest.raises(PositionServoRecoveryError) as caught:
        run(service)
    first=caught.value.report
    first_path=next(tmp_path.glob('position-servo-recovery-*.json'))
    if not completed:
        write_incomplete_source(tmp_path,source)
        return source,first_path,first,None
    second=run(service)
    paths=sorted(tmp_path.glob('position-servo-recovery-*.json'),key=lambda p:int(p.stem.rsplit('-',1)[-1]))
    assert len(paths)==2 and second['recovery_hold_source']==str(first_path)
    write_incomplete_source(tmp_path,source)
    return source,paths[-1],first,second


def write_pending_home_receipt(tmp_path, source, goal, position, *, journal_position=None):
    if journal_position is None:
        journal_position=position
    commands=[list(position) for _ in range(5)]
    journal=tmp_path/'command_journal/chunk_123/events.jsonl'
    journal.parent.mkdir(parents=True,exist_ok=True)
    events=[dict(event='chunk_intent',requested=commands,requested_commands=5)]
    for index,command in enumerate(commands):
        events.extend((
            dict(event='home_command_intent',command_index=index,monotonic_s=float(index*2+1),
                 requested=command,previous_command=commands[max(0,index-1)]),
            dict(event='home_command_result',command_index=index,monotonic_s=float(index*2+2),
                 sent=command,observed_state=command),
        ))
    events.append(dict(event='home_chunk_completed',monotonic_s=11.,
                       settled_state=journal_position,endpoint_outcome='settled'))
    journal.write_text(''.join(json.dumps(row)+'\n' for row in events))
    baseline=source['position_servo']['baseline_parameters_by_motor']
    registers={
        'ID':{name:index+1 for index,name in enumerate(ORDER)},
        'Model_Number':{name:777 for name in ORDER},
        'Firmware_Major_Version':{name:3 for name in ORDER},
        'Firmware_Minor_Version':{name:10 for name in ORDER},
        'Lock':{name:1 for name in ORDER},
        'Operating_Mode':{name:0 for name in ORDER},
        'Acceleration':{name:254 for name in ORDER},
        'Status':{name:0 for name in ORDER},
    }
    for register in ('P_Coefficient','I_Coefficient','D_Coefficient',
                     'Torque_Limit','Max_Torque_Limit'):
        registers[register]={name:baseline[name][register] for name in ORDER}
    snapshot=dict(goal_raw=goal,present_raw=goal,
                  goal_position=dict(zip(ORDER,position)),
                  present_position=dict(zip(ORDER,position)),
                  moving={name:0 for name in ORDER},torque={name:1 for name in ORDER},
                  motor_register_writes=0)
    receipt=dict(schema='v21-home-physical-settled-receipt-v1',
        kind='pending-home-physical-settled',requires_home=True,completed=True,
        physical_settled=True,raw_freeze_verified=False,model_transition=False,
        raw_readback_source='same_raw_register_read',source_time_s=source['source_time_s'],
        source_sha256=hashlib.sha256((tmp_path/'last-attempt.json').read_bytes()).hexdigest(),
        settled_time_s=9.,persisted_time_s=10.,settled_q_position=position,
        journal_settled_q_position=journal_position,
        goal_raw=goal,present_raw=goal,goal_position=position,present_position=position,
        present_minus_goal={name:0. for name in ORDER},endpoint_outcome='settled',
        policy_commands_returned=5,commands_position=commands,
        command_journal=str(journal.resolve()),
        command_journal_sha256=hashlib.sha256(journal.read_bytes()).hexdigest(),
        command_journal_event_count=len(events),actuator_snapshot=snapshot,
        actuator_diagnostics=dict(registers=registers,errors={},motor_register_writes=0))
    path=tmp_path/'pending-home-receipt.json'
    path.write_text(json.dumps(receipt))
    return path,receipt,journal


def test_pending_home_accepts_post_hold_present_different_from_journal_settled(tmp_path):
    _,_,source=incomplete_arm_io(tmp_path)
    original=source['position_servo']['cleanup']['hold']['attempts'][-1]
    goal=dict(original['requested_goal_raw'])
    position=[(goal[name]-10000)/100. for name in ORDER]
    journal_position=list(position)
    journal_position[2]-=.03
    write_pending_home_receipt(
        tmp_path,source,goal,position,journal_position=journal_position)

    from fcpg.v21_hardware_state import load_incomplete_arm_reconnection
    recovery=load_incomplete_arm_reconnection(tmp_path)

    receipt=recovery['pending_home_receipt_snapshot']
    assert receipt['settled_q_position']==position
    assert receipt['journal_settled_q_position']==journal_position


def test_recorded_incomplete_arm_warm_reconnect_is_read_only_and_home_required(tmp_path):
    io,bus,source=incomplete_arm_io(tmp_path)
    result=connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None)
    assert result['arm_recovery'] is True and result['requires_home'] is True
    assert result['motor_register_writes']==0 and 'goal_write' not in bus.events
    assert result['profile_state']=='baseline1000' and result['baseline_restored'] is True
    assert result['compensation_restored'] is False
    assert result['configuration']['parameters_by_motor']['elbow_flex']['Torque_Limit']==1000
    assert result['configuration']['parameters_by_motor']['gripper']['Torque_Limit']==500
    assert result['goal_raw']==source['position_servo']['cleanup']['hold']['attempts'][-1]['requested_goal_raw']


def test_pending_physical_home_receipt_overrides_old_failed_hold_without_writes(tmp_path):
    io,bus,source=incomplete_arm_io(tmp_path)
    original=source['position_servo']['cleanup']['hold']['attempts'][-1]
    goal=dict(original['requested_goal_raw'])
    goal['elbow_flex'] += 4
    position=[(goal[name]-10000)/100. for name in ORDER]
    write_pending_home_receipt(tmp_path,source,goal,position)
    bus.goal=np.asarray(position)
    bus.bias=np.zeros(6)

    from fcpg.v21_hardware_state import load_incomplete_arm_reconnection
    recovery=load_incomplete_arm_reconnection(tmp_path)
    result=connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None)

    assert result['goal_raw']==goal
    assert result['holding_position']==dict(zip(ORDER,position))
    assert recovery['hold_source']==str((tmp_path/'pending-home-receipt.json').resolve())
    assert recovery['original_hold_cache']['raw_freeze_verified'] is False
    assert recovery['original_hold_cache']['actual_present_raw']==goal
    assert result['requires_home'] is True and result['motor_register_writes']==0
    assert 'goal_write' not in bus.events


def test_malformed_pending_physical_home_receipt_fails_before_bus_connect(tmp_path):
    io,bus,_=incomplete_arm_io(tmp_path)
    (tmp_path/'pending-home-receipt.json').write_text('{}')
    with pytest.raises(ValueError,match='unrecognized pending HOME receipt'):
        connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None)
    assert bus.events==[]


@pytest.mark.parametrize('damage',[
    'receipt_symlink','source_change','journal_symlink','journal_sequence',
    'moving','torque','parameters',
])
def test_damaged_pending_home_evidence_fails_before_bus_connect(tmp_path,damage):
    io,bus,source=incomplete_arm_io(tmp_path)
    original=source['position_servo']['cleanup']['hold']['attempts'][-1]
    goal=dict(original['requested_goal_raw'])
    position=[(goal[name]-10000)/100. for name in ORDER]
    path,receipt,journal=write_pending_home_receipt(tmp_path,source,goal,position)
    if damage=='receipt_symlink':
        saved=tmp_path/'saved-receipt.json';path.replace(saved);path.symlink_to(saved)
    elif damage=='source_change':
        changed=deepcopy(source);changed['source_time_s']+=1
        (tmp_path/'last-attempt.json').write_text(json.dumps(changed))
    elif damage=='journal_symlink':
        saved=tmp_path/'saved-journal.jsonl';journal.replace(saved);journal.symlink_to(saved)
    elif damage=='journal_sequence':
        rows=[json.loads(line) for line in journal.read_text().splitlines()]
        rows[3]['command_index']=4
        journal.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        receipt['command_journal_sha256']=hashlib.sha256(journal.read_bytes()).hexdigest()
        path.write_text(json.dumps(receipt))
    elif damage=='moving':
        receipt['actuator_snapshot']['moving']['elbow_flex']=1
        path.write_text(json.dumps(receipt))
    elif damage=='torque':
        receipt['actuator_snapshot']['torque']['elbow_flex']=0
        path.write_text(json.dumps(receipt))
    else:
        receipt['actuator_diagnostics']['registers']['Torque_Limit']['elbow_flex']=999
        path.write_text(json.dumps(receipt))

    with pytest.raises((ValueError,RuntimeError)):
        connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None)

    assert bus.events==[]


def test_stopped_short_j3_residual_on_restored_baseline_reconnects_without_compensation(tmp_path):
    io,bus,source=incomplete_arm_io(tmp_path)
    bus.bias[2]=.06
    result=connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None)
    assert result['needs_fixed_goal_compensation'] is False and result['requires_home'] is True
    assert result['compensation_restored'] is False and result['motor_register_writes']==0
    assert 'goal_write' not in bus.events
    assert result['goal_raw']==source['position_servo']['cleanup']['hold']['attempts'][-1]['requested_goal_raw']


def static_encoder_residual(io,bus,raw_offsets,position_per_tick):
    """Fake fixed Goal with independently calibrated raw and position residuals."""
    goal_q=bus.goal.copy()
    goal_raw=np.rint(10000+100*goal_q).astype(int)
    raw_offsets=np.asarray(raw_offsets,dtype=int)
    position_per_tick=np.asarray(position_per_tick,dtype=float)
    previous=bus.sync_read
    def read(register,normalize=True,num_retry=0):
        if register in ('Present_Position','Goal_Position'):
            raw=goal_raw+(raw_offsets if register=='Present_Position' else 0)
            values=goal_q+(raw-goal_raw)*position_per_tick if normalize else raw
            return dict(zip(ORDER,values.tolist()))
        return previous(register,normalize=normalize,num_retry=num_retry)
    bus.sync_read=read
    bus._normalize=lambda values:{key:float(goal_q[key-1]+(raw-goal_raw[key-1])*position_per_tick[key-1])
                                  for key,raw in values.items()}
    io.calibration_reconstructed_encoder_cells=lambda q:np.rint(
        goal_raw+(np.asarray(q)-goal_q)/position_per_tick)
    return raw_offsets


@pytest.mark.parametrize('profile_state',['baseline1000','cleanup_cap200'])
def test_static_two_tick_j2_four_tick_j3_residual_admits_home_without_writes(tmp_path,profile_state):
    if profile_state=='baseline1000':
        io,bus,source=incomplete_arm_io(tmp_path)
    else:
        source=cap200_source(tmp_path)
        io,bus,_=io_for_failed_source(source,profile_state=profile_state)
    static_encoder_residual(io,bus,[1,2,4,0,0,0],[.08,.08234,.088725,.08,.08,.08])

    result=connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None)

    assert result['requires_home'] is True and result['motor_register_writes']==0
    assert result['profile_state']==profile_state
    assert result['needs_fixed_goal_compensation'] is (profile_state=='cleanup_cap200')
    assert result['position_delta']==[0.]*6
    assert result['after'][-1]['present_minus_goal']['shoulder_lift']==pytest.approx(.16468)
    assert result['after'][-1]['present_minus_goal']['elbow_flex']==pytest.approx(.3549)
    assert result['goal_raw']==source['position_servo']['cleanup']['hold']['attempts'][-1]['requested_goal_raw']
    assert 'goal_write' not in bus.events


@pytest.mark.parametrize('offset,resolution,violation',[
    (7,.01,'other_raw_error'),  # Raw bound still applies below the position bound.
    (2,.31,'other_q_error'),   # Position bound still applies below the raw bound.
])
def test_other_joint_static_residual_keeps_both_home_bounds(tmp_path,offset,resolution,violation):
    io,bus,_=incomplete_arm_io(tmp_path)
    static_encoder_residual(io,bus,[0,offset,0,0,0,0],[.01,resolution,.01,.01,.01,.01])

    with pytest.raises(RuntimeError,match=violation):
        connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None)

    assert 'goal_write' not in bus.events


@pytest.mark.parametrize('phase',['samples','cameras'])
def test_bounded_residual_does_not_admit_pose_drift(tmp_path,phase):
    io,bus,_=incomplete_arm_io(tmp_path)
    offsets=static_encoder_residual(io,bus,[0,2,4,0,0,0],[.08,.08234,.088725,.08,.08,.08])
    wait=lambda _:None
    if phase=='samples':
        def wait(_):offsets[1]+=2
        expected='actual position is not quiet'
    else:
        camera=io.follower.cameras['column']
        connect=camera.connect
        def move():
            connect()
            offsets[1]+=2
        camera.connect=move
        expected='physical pose changed while connecting cameras'

    with pytest.raises(RuntimeError,match=expected):
        connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=wait)

    assert 'goal_write' not in bus.events


def test_warm_reconnect_uses_latest_verified_failed_home_cleanup_hold(tmp_path):
    source=cap200_source(tmp_path)
    io,bus,_=io_for_failed_source(source,profile_state='cleanup_cap200')
    original=source['position_servo']['cleanup']['hold']['attempts'][-1]
    goal=dict(original['requested_goal_raw'])
    goal['elbow_flex']+=3
    held=list(original['held_state']);held[2]+=.03
    report=dict(schema='v21-position-servo-baseline-recovery-v1',completed=False,
        configuration_restored=False,source_time_s=source['source_time_s'],
        touched_motors=['elbow_flex'],frozen_hold_goal_raw=original['requested_goal_raw'],
        original_baseline_by_motor=source['position_servo']['baseline_parameters_by_motor'],
        parameter_writes=[dict(motor='elbow_flex',register='Torque_Limit',value=200,
            fallback=True,may_have_applied=True,verified=True,readback=200)],
        cleanup=dict(errors=[],hold_verified=True,cap200_verified={'elbow_flex':True},
            hold=dict(event='safe_hold_verified',reason='v21_emergency_stop',
                reused_existing_hold=False,motor_register_writes=1,
                attempts=[dict(requested_goal_raw=goal,goal_readback_raw=goal,
                    goal_register_verified=True,pose_validated=True,held_state=held)])))
    path=tmp_path/'position-servo-recovery-999.json'
    path.write_text(json.dumps(report))
    bus.goal=np.asarray(held);bus.bias[2]=.03

    result=connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None)

    assert result['goal_raw']==goal and result['holding_position']==dict(zip(ORDER,held))
    assert result['configuration']['requires_home'] is True
    assert result['profile_state']=='cleanup_cap200' and result['baseline_restored'] is False
    assert result['needs_fixed_goal_compensation'] is True
    assert result['motor_register_writes']==0 and 'goal_write' not in bus.events


def test_legacy_failed_cleanup_cap200_reconnect_is_read_only(tmp_path):
    source=cap200_source(tmp_path,legacy=True)
    io,bus,_=io_for_failed_source(source,profile_state='cleanup_cap200')

    result=connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None)

    assert result['profile_state']=='cleanup_cap200' and result['baseline_restored'] is False
    assert result['needs_fixed_goal_compensation'] is True
    assert result['configuration']['parameters_by_motor']['elbow_flex']['Torque_Limit']==200
    assert result['motor_register_writes']==0 and 'goal_write' not in bus.events


def test_failed_position_servo_recovery_cap200_reconnect_uses_latest_hold_without_writes(tmp_path):
    source,_,first,_=recovery_lineage(tmp_path,completed=False)
    attempt=first['cleanup']['hold']['attempts'][-1]
    io,bus,_=io_for_failed_source(source,profile_state='cleanup_cap200',
        goal_raw=attempt['goal_readback_raw'],held_position=attempt['held_state'])

    result=connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None)

    assert result['goal_raw']==attempt['goal_readback_raw']
    assert result['holding_position']==dict(zip(ORDER,attempt['held_state']))
    assert result['profile_state']=='cleanup_cap200' and result['baseline_restored'] is False
    assert result['motor_register_writes']==0 and 'goal_write' not in bus.events


def test_newer_failed_position_recovery_overrides_failed_fixed_goal_hold_and_profile(tmp_path):
    from fcpg.v21_fixed_goal_compensation import FixedGoalCompensationError
    from test_v21_fixed_goal_compensation import run as fixed_run,setup as fixed_setup
    fixed_service=fixed_setup(tmp_path,error_cells=33)
    source=deepcopy(fixed_service.last_attempt)
    def fail_active_damping(register,motor,value):
        if register=='D_Coefficient' and value==32:
            raise RuntimeError('injected fixed-Goal operation failure')
    fixed_service.owner.bus.on_write=fail_active_damping
    with pytest.raises(FixedGoalCompensationError) as caught:
        fixed_run(fixed_service)
    fixed_report=caught.value.report
    assert fixed_report['configuration_restored'] is True
    assert fixed_report['cleanup']['profile_state']=='baseline1000'
    fixed_receipt=dict(schema='v21-fixed-goal-service-result-v1',state='failed',completed=False,
        configuration_restored=True,source_execution_path=str((tmp_path/'last-attempt.json').resolve()),
        source_execution=source,report=fixed_report)
    (tmp_path/'fixed-goal-compensation.json').write_text(json.dumps(fixed_receipt))

    recovery_source,_,recovery_report,_=recovery_lineage(tmp_path,completed=False)
    assert recovery_source==source
    recovery_hold=recovery_report['cleanup']['hold']['attempts'][-1]
    assert recovery_hold['goal_readback_raw']!=fixed_report['cleanup']['effective_hold_goal_raw']
    io,bus,_=io_for_failed_source(source,profile_state='cleanup_cap200',
        goal_raw=recovery_hold['goal_readback_raw'],held_position=recovery_hold['held_state'])

    result=connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None)

    assert result['goal_raw']==recovery_hold['goal_readback_raw']
    assert result['profile_state']=='cleanup_cap200'
    assert result['baseline_restored'] is False and result['compensation_restored'] is False
    assert result['configuration']['parameters_by_motor']['elbow_flex']['Torque_Limit']==200
    assert result['motor_register_writes']==0 and 'goal_write' not in bus.events


def test_completed_multihop_position_servo_recovery_reconnects_baseline1000_without_writes(tmp_path):
    source,latest,first,second=recovery_lineage(tmp_path,completed=True)
    first_hold=first['cleanup']['hold']['attempts'][-1]
    io,bus,_=io_for_failed_source(source,profile_state='baseline1000',
        goal_raw=second['frozen_hold_goal_raw'],held_position=first_hold['held_state'])

    result=connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None)

    assert result['goal_raw']==second['frozen_hold_goal_raw']
    assert result['profile_state']=='baseline1000' and result['baseline_restored'] is True
    assert result['needs_fixed_goal_compensation'] is False
    assert result['configuration']['parameters_by_motor']['elbow_flex']['Torque_Limit']==1000
    assert result['motor_register_writes']==0 and 'goal_write' not in bus.events
    assert second['recovery_hold_source']!=str(latest)


@pytest.mark.parametrize('damage',['goal','profile'])
def test_bad_completed_recovery_goal_or_profile_fails_closed_before_bus_connect(tmp_path,damage):
    source,latest,first,second=recovery_lineage(tmp_path,completed=True)
    first_hold=first['cleanup']['hold']['attempts'][-1]
    io,bus,_=io_for_failed_source(source,profile_state='baseline1000',
        goal_raw=second['frozen_hold_goal_raw'],held_position=first_hold['held_state'])
    report=json.loads(latest.read_text())
    if damage=='goal':
        report['frozen_hold_goal_raw']['elbow_flex']+=1
    else:
        report['final_parameters_by_motor']['elbow_flex']['Torque_Limit']=999
    latest.write_text(json.dumps(report))

    with pytest.raises(ValueError):
        connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None)

    assert bus.events==[]


def test_recovery_receipt_snapshot_change_is_rejected_before_warm_connect(tmp_path):
    source,latest,_,_=recovery_lineage(tmp_path,completed=True)
    io,bus,_=io_for_failed_source(source,profile_state='baseline1000')
    runtime=SimpleNamespace(stopped=False,requires_home_after_incomplete_execution=False)
    runtime.stop=lambda:setattr(runtime,'stopped',True)
    owner=SimpleNamespace(output=tmp_path,runtime=runtime,stop_event=threading.Event(),
        _lease_check=lambda:None)
    initialize=existing_owner_connection_initializer(owner)
    report=json.loads(latest.read_text())
    report['snapshot_revision_for_test']=1
    latest.write_text(json.dumps(report))

    with pytest.raises(RuntimeError,match='admission changed'):
        initialize(io)

    assert bus.events==[]


@pytest.mark.parametrize('state',['completed','other_source'])
def test_only_same_source_completed_compensation_allows_fresh_baseline1000_reconnect(tmp_path,state):
    from test_v21_fixed_goal_compensation import setup,run
    service=setup(tmp_path,error_cells=1)
    source=deepcopy(service.last_attempt)
    write_incomplete_source(tmp_path,source)
    io,bus,_=io_for_failed_source(source,profile_state='cleanup_cap200')
    report=run(service)
    envelope=dict(schema='v21-fixed-goal-service-result-v1',state='completed',completed=True,
        configuration_restored=True,source_execution_path=str((tmp_path/'last-attempt.json').resolve()),
        source_execution=source,report=report)
    if state=='other_source':
        envelope['source_execution']=dict(source,source_time_s=source['source_time_s']+1)
    (tmp_path/'fixed-goal-compensation.json').write_text(json.dumps(envelope))
    bus.profile['elbow_flex']['Torque_Limit']=1000
    bus.bad['J3_Torque_Limit']=1000
    if state=='completed':
        result=connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None)
        assert result['compensation_restored'] is True and result['requires_home'] is True
        assert result['needs_fixed_goal_compensation'] is False
    else:
        with pytest.raises(RuntimeError,match='torque limit'):
            connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None)
    assert 'goal_write' not in bus.events


@pytest.mark.parametrize('damage',['cold','mixed','unknown_cap','other_cap','gain','goal','load','status','moving','velocity','camera'])
def test_incomplete_arm_reconnect_rejects_without_any_startup_goal_or_parameter_write(tmp_path,damage):
    io,bus,_=incomplete_arm_io(tmp_path)
    if damage=='cold':bus.torque[:]=0
    elif damage=='mixed':bus.torque[0]=0
    elif damage=='unknown_cap':bus.bad['J3_Torque_Limit']=201
    elif damage=='other_cap':bus.profile['wrist_flex']['Torque_Limit']=200
    elif damage=='gain':bus.profile['elbow_flex']['I_Coefficient']=1
    elif damage=='goal':bus.goal[2]+=.01
    elif damage=='load':bus.bad['Present_Load']=33
    elif damage=='status':bus.bad['Status']=1
    elif damage=='moving':bus.bad['Moving']=1
    elif damage=='velocity':bus.bad['Present_Velocity']=1
    else:
        def fail_camera():raise RuntimeError('camera connect failed')
        io.follower.cameras['column'].connect=fail_camera
    with pytest.raises(RuntimeError):
        connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None,
            legacy_preseed=lambda *_:pytest.fail('cold preseed forbidden during incomplete recovery'),
            legacy_connect=lambda *_:pytest.fail('configure forbidden during incomplete recovery'))
    assert 'goal_write' not in bus.events


def test_incomplete_arm_warm_start_recovers_only_failed_camera_without_motor_writes(tmp_path):
    io,bus,_=incomplete_arm_io(tmp_path)
    column=io.follower.cameras['column']
    calls=[]
    column.connect=lambda:(_ for _ in ()).throw(RuntimeError('first Column frame timed out'))
    def recover(view,error):
        calls.append((view,str(error)))
        io.follower.cameras[view].is_connected=True
        return np.zeros((480,640,3),dtype=np.uint8)
    io._recover_camera=recover

    result=connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None)

    assert calls==[('column','first Column frame timed out')]
    assert result['camera_recovery_views']==['column']
    assert result['motor_register_writes']==0 and result['requires_home'] is True
    assert 'goal_write' not in bus.events
    assert io.follower.cameras['wrist'].is_connected and column.is_connected


def test_incomplete_arm_failed_camera_recovery_remains_zero_motor_writes(tmp_path):
    io,bus,_=incomplete_arm_io(tmp_path)
    column=io.follower.cameras['column']
    column.connect=lambda:(_ for _ in ()).throw(RuntimeError('first Column frame timed out'))
    calls=[]
    def recover(view,error):
        calls.append((view,str(error)))
        raise TimeoutError('bounded camera-only recovery exhausted')
    io._recover_camera=recover

    with pytest.raises(TimeoutError,match='camera-only recovery exhausted'):
        connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None)

    assert calls==[('column','first Column frame timed out')]
    assert 'goal_write' not in bus.events
    assert bus.is_connected


def test_existing_owner_latches_before_connect_and_reuses_true_original_raw_hold(tmp_path):
    io,bus,source=incomplete_arm_io(tmp_path)
    runtime=SimpleNamespace(stopped=False,requires_home_after_incomplete_execution=False)
    runtime.stop=lambda:setattr(runtime,'stopped',True)
    owner=SimpleNamespace(output=tmp_path,runtime=runtime,stop_event=threading.Event(),io=io,
        _lease_check=lambda:None)
    initialize=existing_owner_connection_initializer(owner)
    assert owner.stop_event.is_set() and runtime.stopped and runtime.requires_home_after_incomplete_execution
    assert not bus.is_connected
    initialize(io)
    original=source['position_servo']['cleanup']['hold']['attempts'][-1]
    assert owner._v21_hold_cache['reference_present_raw']==original['requested_goal_raw']
    assert owner._v21_hold_cache['reference_present']==original['held_state']
    assert owner._v21_hold_cache['raw_freeze_verified'] is True
    io._hold_at_measured_pose=lambda *_args,**_kwargs:pytest.fail('genuine reverified raw freeze must not rewrite Goal')
    hold=verified_hold(owner)
    assert hold['reused_existing_hold'] and hold['motor_register_writes']==0
    assert 'goal_write' not in bus.events


def test_incomplete_arm_owner_rejects_gripper_tuning_before_connect(tmp_path):
    io,bus,_=incomplete_arm_io(tmp_path)
    owner=SimpleNamespace(output=tmp_path)
    with pytest.raises(RuntimeError,match='gripper500'):
        existing_owner_connection_initializer(owner,gripper_force='firm')
    assert bus.events==[]


def test_output_without_incomplete_marker_cannot_admit_reduced_cap(tmp_path):
    io,bus,_=incomplete_arm_io(tmp_path)
    (tmp_path/'incomplete-execution.json').unlink()
    with pytest.raises(RuntimeError,match='evidence disappeared'):
        connect_preserving_warm_hold(io,incomplete_arm_output=tmp_path,wait=lambda _:None)
    assert bus.events==[]


def test_source_change_before_cache_publication_cannot_seed_unrelated_raw_freeze(tmp_path,monkeypatch):
    import fcpg.v21_hardware_state as module
    io,bus,source=incomplete_arm_io(tmp_path)
    runtime=SimpleNamespace(stopped=True,requires_home_after_incomplete_execution=True,stop=lambda:None)
    owner=SimpleNamespace(output=tmp_path,runtime=runtime,stop_event=threading.Event(),_lease_check=lambda:None)
    initialize=existing_owner_connection_initializer(owner)
    def altered_source(_io,**kwargs):
        source['source_time_s']+=1
        (tmp_path/'last-attempt.json').write_text(json.dumps(source))
        return {'arm_recovery':True}
    monkeypatch.setattr(module,'connect_preserving_warm_hold',altered_source)
    with pytest.raises(RuntimeError,match='before restoring HOLD'):
        initialize(io)
    assert not hasattr(owner,'_v21_hold_cache')
    assert bus.events==[]
