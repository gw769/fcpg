"""Only fake registers/clocks: never construct hardware, cameras or a model."""
import json
import threading
from types import MethodType

import numpy as np
import pytest

from test_v21_hardware_state import hold_owner, ORDER
from test_v21_hardware_bridge import setup
from scripts.canary_so101_v21_research import TrialOwner
from scripts.serve_so101_realtime_fcpg_lewm import HardwareSO101IO


def fixture(tmp_path=None):
    owner,bus=hold_owner()
    owner.stop_event=threading.Event()
    owner.io.settle_samples=3
    owner.output=tmp_path
    owner.io._append_command_journal=HardwareSO101IO._append_command_journal
    for name in ('_record_hold','_halt','_completion_hold','_write_q','_read_q'):
        setattr(owner,name,MethodType(getattr(TrialOwner,name),owner))
    def write(command,previous):
        bus.sync_write('Goal_Position',dict(zip(ORDER,command.tolist())))
        return command.copy()
    owner.io._write_position_command=write
    return owner,bus


def follow(bus):
    return dict(budget_s=.4,goal_reached=True,positions=[(bus.goal+bus.bias).tolist()]*3)


def servo_follow(bus):
    from fcpg.v21_position_arrival import evaluate_position_arrival
    positions=np.repeat((bus.goal+bus.bias)[None],9,axis=0)
    times=10.+np.arange(1,10)*.05
    arrival=evaluate_position_arrival(bus.goal,positions,times-10.,
        position_tolerance=.2,step_tolerance=.1,quiet_samples=3)
    return dict(schema='v21-position-servo-completion-follow-v1',
        original_target=bus.goal.tolist(),positions=positions.tolist(),times_s=times.tolist(),
        source_time_s=10.,baseline_restored=True,hold_eligible=True,goal_reached=arrival['reached'])


def test_finishing_uses_real_later_quiet_samples_without_relabeling_old_q5():
    owner,bus=fixture();bus.bias[:]=0.
    q5=bus.goal+2.6
    proof=servo_follow(bus)
    result=owner._completion_hold(bus.goal.copy(),proof,q5)
    assert result['reused_existing_hold'] is True and result['motor_register_writes']==0
    assert 'goal_write' not in bus.events and 'budget_s' not in proof
    verification=result['actuator_events'][0]
    assert verification['completion_evidence_schema']==proof['schema']
    assert verification['original_target_arrival']['reached'] is True
    np.testing.assert_array_equal(q5,bus.goal+2.6)


def test_nonapplicable_finishing_can_retain_a_quiet_goal_without_claiming_arrival():
    owner,bus=fixture();bus.bias[:]=2.6
    proof=servo_follow(bus)
    result=owner._completion_hold(bus.goal.copy(),proof,bus.goal+bus.bias)
    assert result['reused_existing_hold'] is True and result['motor_register_writes']==0
    assert proof['goal_reached'] is False
    assert result['actuator_events'][0]['original_target_arrival']['reached'] is False


@pytest.mark.parametrize('damage',['schema','restored','quorum','duplicated_time','reversed_time',
    'nonnumeric','nan','target','clock','reached','ineligible'])
def test_invalid_finishing_proof_freezes_without_pretending_to_be_normal_follow(damage):
    owner,bus=fixture();bus.bias[:]=0.;proof=servo_follow(bus)
    if damage=='schema':proof.update(schema='unrecognized',budget_s=2.)
    elif damage=='restored':proof['baseline_restored']=False
    elif damage=='quorum':
        proof['positions']=proof['positions'][:3];proof['times_s']=proof['times_s'][:3]
    elif damage=='duplicated_time':proof['times_s'][-1]=proof['times_s'][-2]
    elif damage=='reversed_time':proof['times_s']=proof['times_s'][::-1]
    elif damage=='nonnumeric':proof['positions']=[['9']*6]*9
    elif damage=='nan':proof['positions'][-1][2]=float('nan')
    elif damage=='target':proof['original_target'][2]+=.1
    elif damage=='clock':proof['source_time_s']=11.
    elif damage=='reached':proof['goal_reached']=False
    elif damage=='ineligible':proof['hold_eligible']=False
    result=owner._completion_hold(bus.goal.copy(),proof,bus.goal+2.6)
    assert result['reason']=='v21_normal_completion_fallback'
    assert result['motor_register_writes']==1 and result['normal_validation_error']


@pytest.mark.parametrize('bias',[.2,.3])
def test_fresh_registers_cannot_hide_post_finishing_drift_under_coarse_hold_tolerance(bias):
    owner,bus=fixture();bus.bias[:]=0.;proof=servo_follow(bus)
    bus.bias[2]=bias
    result=owner._completion_hold(bus.goal.copy(),proof,bus.goal+2.6)
    assert result['motor_register_writes']==1
    assert result['normal_validation_error']


def bridge_fixture(owner):
    bridge,runtime,_,_=setup()
    runtime.pending.trace.positions.fill_(9.)
    bridge.read_q=owner._read_q
    bridge.write_q=owner._write_q
    bridge.halt=owner._halt
    bridge.completion_hold=owner._completion_hold
    bridge.request_stop=owner.stop_event.set
    bridge.cancelled=owner.stop_event.is_set
    bridge.normal_follow_budget_s=.4
    return bridge,runtime


@pytest.mark.parametrize('budget', [.4,2.])
def test_completed_chunks_keep_policy_goal_with_constant_servo_bias(tmp_path,budget):
    owner,bus=fixture(tmp_path)
    for _ in range(3):
        bridge,runtime=bridge_fixture(owner)
        bridge.normal_follow_budget_s=budget
        before=bus.events.count('goal_write')
        result=bridge.execute()
        assert result['completed'] and len(result['commands_position'])==5
        assert len(result['measured_q_path'])==6
        assert result['commands_position'][-1]==[9.]*6
        assert result['measured_q_path'][-1]==[10.]*6
        assert result['completion_hold']['held_position']==[9.]*6
        assert result['completion_hold']['goal_register_writes']==0
        assert result['completion_hold']['reason']=='normal_completion_retained_fifth_goal'
        assert bus.events.count('goal_write')-before==5
        assert runtime.actual[0].positions.shape==(1,5,6)
        np.testing.assert_array_equal(bus.goal,np.full(6,9.))
    saved=[json.loads(line) for line in (tmp_path/'actuator-events.jsonl').read_text().splitlines()]
    assert len(saved)==3 and all(x['retained_existing_goal'] for x in saved)
    assert all(len(x['readbacks'])==5 for x in saved)


@pytest.mark.parametrize('damage',['disabled','unreached','quorum','quiet','moving','torque','wrong_goal','late_motion','raw_position','read_failure'])
@pytest.mark.parametrize('budget', [.4,2.])
def test_missing_or_invalid_normal_proof_freezes_and_records_reason(damage,tmp_path,budget):
    owner,bus=fixture(tmp_path)
    goal=bus.goal.copy();tail=follow(bus);q5=bus.goal+bus.bias
    tail['budget_s']=budget
    if damage=='disabled':tail['budget_s']=0.
    elif damage=='unreached':tail['goal_reached']=False
    elif damage=='quorum':tail['positions']=tail['positions'][:2]
    elif damage=='quiet':tail['positions'][-1]=[10.5]*6
    elif damage=='moving':bus.bad['Moving']=1
    elif damage=='torque':bus.torque[0]=0
    elif damage=='wrong_goal':goal[0]+=.1
    elif damage=='late_motion':bus.bias[0]+=.5
    else:
        read=bus.sync_read
        def corrupt(register,normalize=True,**kwargs):
            if damage=='read_failure' and register=='Moving':raise ConnectionError('fake read lost')
            value=read(register,normalize=normalize,**kwargs)
            if damage=='raw_position' and register=='Present_Position' and not normalize:
                value[ORDER[0]]+=100
            return value
        bus.sync_read=corrupt
    record=owner._completion_hold(goal,tail,q5)
    assert record['reason']=='v21_normal_completion_fallback'
    assert record['normal_validation_error']
    assert record['motor_register_writes']==1 and not record['reused_existing_hold']
    assert not record['actuator_events'][0]['retained_existing_goal']
    written=record['actuator_events'][-1]
    assert written['event']=='goal_write' and written['goal_register_verified']
    assert written['write_started_s']<=written['write_returned_s']<=written['verification_returned_s']
    saved=[json.loads(line) for line in (tmp_path/'actuator-events.jsonl').read_text().splitlines()]
    assert saved[0]['error']==record['normal_validation_error']
    assert saved[-1]['requested_position']==written['goal_readback']


def test_raw_goal_identity_accepts_only_the_actual_quantized_cell():
    owner,bus=fixture()
    owner.io.calibration_reconstructed_encoder_cells=lambda q:np.asarray(1000+10*np.asarray(q),dtype=np.int64)
    bus.goal[:]=9.1
    goal=np.full(6,9.19,np.float32)  # same truncation cell as the normalized write
    record=owner._completion_hold(goal,follow(bus),bus.goal+bus.bias)
    assert record['motor_register_writes']==0
    assert record['attempts'][-1]['held_state'][0]==9.1
    goal[0]=9.21  # one cell away must not pass a floating-point tolerance
    rejected=owner._completion_hold(goal,follow(bus),bus.goal+bus.bias)
    assert rejected['motor_register_writes']==1
    assert 'raw readback mismatch' in rejected['normal_validation_error']


def test_goal_change_during_verification_is_not_retained():
    owner,bus=fixture();read=bus.sync_read
    def change(register,**kwargs):
        value=read(register,**kwargs)
        if register=='Moving':bus.goal[0]+=.1
        return value
    bus.sync_read=change
    result=owner._completion_hold(np.full(6,9.),follow(bus),bus.goal+bus.bias)
    assert result['motor_register_writes']==1
    assert 'changed during completion verification' in result['normal_validation_error']


@pytest.mark.parametrize('cancel_on',[1,2,3,4,5])
def test_cancel_after_any_normal_register_read_freezes_before_logging(cancel_on,tmp_path):
    owner,bus=fixture(tmp_path);read=bus.sync_read;calls=[0]
    def cancel(register,**kwargs):
        value=read(register,**kwargs);calls[0]+=1
        if calls[0]==cancel_on:owner.stop_event.set()
        return value
    bus.sync_read=cancel
    journal=owner.io._append_command_journal
    def persist(path,event):
        if path is None:return  # Existing IO calls the no-op journal hook.
        assert not owner.bus_lock._is_owned()
        assert bus.events.count('goal_write')==1  # freeze precedes disk activity
        journal(path,event)
    owner.io._append_command_journal=persist
    with pytest.raises(RuntimeError,match='cancelled'):
        owner._completion_hold(np.full(6,9.),follow(bus),bus.goal+bus.bias)
    assert owner.stop_event.is_set() and bus.events.count('goal_write')==1
    saved=[json.loads(line) for line in (tmp_path/'actuator-events.jsonl').read_text().splitlines()]
    assert not saved[0]['retained_existing_goal']
    assert saved[-1]['role']=='emergency_freeze' and saved[-1]['goal_register_verified']


def test_stop_latches_while_normal_read_is_in_flight_and_prevents_queued_write(tmp_path):
    owner,bus=fixture(tmp_path);bridge,_=bridge_fixture(owner)
    entered=threading.Event();release=threading.Event();latched=threading.Event()
    read=bus.sync_read;calls=[0];errors=[]
    def blocked(register,**kwargs):
        calls[0]+=1
        if calls[0]==1:
            entered.set()
            assert release.wait(2.)
        return read(register,**kwargs)
    bus.sync_read=blocked
    def latch():owner.stop_event.set();latched.set()
    bridge.request_stop=latch
    def normal():
        try:owner._completion_hold(np.full(6,9.),follow(bus),bus.goal+bus.bias)
        except BaseException as exc:errors.append(exc)
    worker=threading.Thread(target=normal);worker.start()
    assert entered.wait(2.)
    stopper=threading.Thread(target=bridge.stop);stopper.start()
    assert latched.wait(2.) and owner.stop_event.is_set()
    release.set();worker.join(2.);stopper.join(2.)
    assert not worker.is_alive() and not stopper.is_alive()
    assert len(errors)==1 and 'cancelled' in str(errors[0])
    assert bus.events.count('goal_write')==1
    with pytest.raises(RuntimeError,match='STOP is latched'):
        owner._write_q(np.full(6,8.),np.full(6,9.))
    assert bus.events.count('goal_write')==1
    assert bridge.last_stop_record['goal_register_writes']==0


def test_cancel_after_normal_return_cannot_complete_model_transaction():
    owner,bus=fixture();bridge,runtime=bridge_fixture(owner)
    normal=bridge.completion_hold
    def cancelled(*args):
        result=normal(*args);owner.stop_event.set();return result
    bridge.completion_hold=cancelled
    with pytest.raises(RuntimeError,match='cancelled after completion hold'):bridge.execute()
    assert not bridge.last_attempt['completed'] and not hasattr(runtime,'actual')
    assert bridge.last_attempt['emergency_hold']['goal_register_writes']==1
    assert len(bridge.last_attempt['commands_position'])==5


def test_partial_freeze_write_failure_is_durable_even_without_a_return(tmp_path):
    owner,bus=fixture(tmp_path);write=bus.sync_write
    def fail(*args,**kwargs):write(*args,**kwargs);raise ConnectionError('fake write acknowledgement lost')
    bus.sync_write=fail
    with pytest.raises(ConnectionError,match='acknowledgement lost'):owner._halt()
    event=json.loads((tmp_path/'actuator-events.jsonl').read_text().splitlines()[-1])
    assert event['requested_position'] and event['write_started_s']
    assert 'write_returned_s' not in event
    assert 'acknowledgement lost' in event['error']


@pytest.mark.parametrize('budget',[0.,.4,2.])
def test_bridge_without_a_quiet_reached_goal_preserves_original_freeze(budget):
    owner,bus=fixture();bridge,runtime=bridge_fixture(owner)
    bridge.normal_follow_budget_s=budget
    if budget:bus.bias[:]=4.
    result=bridge.execute()
    assert result['completed'] and len(result['commands_position'])==5
    assert len(result['measured_q_path'])==6 and runtime.actual[0].positions.shape==(1,5,6)
    assert result['completion_hold']['reason']=='v21_normal_completion_fallback'
    assert result['completion_hold']['normal_validation_error']
    assert bus.events.count('goal_write')==6
    assert result['actuator_events'][-1]['event']=='goal_write'


def test_exception_latches_before_runtime_stop_and_freezes_without_remaining_writes():
    owner,bus=fixture();bridge,runtime=bridge_fixture(owner);read=owner._read_q;calls=[0]
    def faulty():
        calls[0]+=1
        if calls[0]==3:raise RuntimeError('fake transaction read failure')
        return read()
    bridge.read_q=faulty
    stop=runtime.stop
    def stopped():
        assert owner.stop_event.is_set()
        stop()
    runtime.stop=stopped
    with pytest.raises(RuntimeError,match='transaction read failure'):bridge.execute()
    assert bridge.last_attempt['policy_commands_returned']==1
    assert bus.events.count('goal_write')==2
    assert bridge.last_attempt['emergency_hold']['goal_register_writes']==1
    assert not bridge.last_attempt['completed']


def test_each_freeze_retry_records_its_actual_write_and_confirmation(tmp_path):
    owner,bus=fixture(tmp_path);read=bus.sync_read;confirmations=[0]
    def mismatch(register,**kwargs):
        result=read(register,**kwargs)
        if register=='Goal_Position' and not kwargs.get('normalize',True):
            confirmations[0]+=1
            if confirmations[0]==1:result[ORDER[0]]-=1.
        return result
    bus.sync_read=mismatch
    result=owner._halt()
    assert result['motor_register_writes']==2
    saved=[json.loads(line) for line in (tmp_path/'actuator-events.jsonl').read_text().splitlines()]
    assert len(saved)==2 and all(x['event']=='goal_write' for x in saved)
    assert not saved[0]['goal_register_verified'] and saved[0]['mismatched_joints']
    assert saved[1]['goal_register_verified'] and not saved[1]['mismatched_joints']
    assert saved[0]['verification_returned_s']<=saved[1]['write_started_s']


def test_freeze_confirmation_read_failure_is_not_reported_as_verified(tmp_path):
    owner,bus=fixture(tmp_path);read=bus.sync_read
    def fail(register,**kwargs):
        if register=='Goal_Position':raise ConnectionError('fake verification read failed')
        return read(register,**kwargs)
    bus.sync_read=fail
    with pytest.raises(ConnectionError,match='verification read failed'):owner._halt()
    saved=json.loads((tmp_path/'actuator-events.jsonl').read_text())
    assert saved['write_returned_s'] and saved['verification_started_s']
    assert 'verification_returned_s' not in saved and not saved.get('goal_register_verified')
    assert 'verification read failed' in saved['error']


def test_final_normal_goal_read_failure_keeps_the_failed_read_event(tmp_path):
    owner,bus=fixture(tmp_path);read=bus.sync_read;calls=[0]
    def fail(register,**kwargs):
        calls[0]+=1
        if calls[0]==5:raise ConnectionError('fake final goal read failed')
        return read(register,**kwargs)
    bus.sync_read=fail
    result=owner._completion_hold(np.full(6,9.),follow(bus),bus.goal+bus.bias)
    assert result['motor_register_writes']==1
    failed=result['actuator_events'][0]['readbacks'][-1]
    assert failed['register']=='Goal_Position' and failed['failed_s']>=failed['started_s']
    assert 'final goal read failed' in failed['error'] and 'returned_s' not in failed
    assert result['actuator_events'][-1]['goal_register_verified']


def test_cancel_during_capture_never_completes_the_model_transition():
    owner,bus=fixture();bridge,runtime=bridge_fixture(owner)
    capture=bridge.capture_endpoint
    def cancelled():
        endpoint=capture();owner.stop_event.set();return endpoint
    bridge.capture_endpoint=cancelled
    with pytest.raises(RuntimeError,match='cancelled during endpoint capture'):bridge.execute()
    assert not bridge.last_attempt['completed'] and not hasattr(runtime,'actual')
    assert bridge.last_attempt['emergency_hold']['goal_register_writes']==1
    assert len(bridge.last_attempt['commands_position'])==5


def test_cancel_during_final_guard_never_calls_model_completion():
    from types import SimpleNamespace
    import torch
    owner,bus=fixture();bridge,runtime=bridge_fixture(owner)
    capture=bridge.capture_endpoint
    bridge.capture_endpoint=lambda:SimpleNamespace(time=capture().time,q=torch.full((1,6),10.))
    seen=[0]
    def guard(q):
        seen[0]+=1
        if seen[0]==11:owner.stop_event.set()  # two initial, four mid, q5, three follow, endpoint
    bridge.extra_readback_check=guard
    with pytest.raises(RuntimeError,match='cancelled before model completion'):bridge.execute()
    assert seen[0]==11 and not hasattr(runtime,'actual')
    assert not bridge.last_attempt['completed'] and bus.events.count('goal_write')==6


@pytest.mark.parametrize('operation',['stop','execute_fault'])
def test_physical_freeze_precedes_cleanup_waiting_on_another_threads_runtime_lock(operation):
    owner,bus=fixture();bridge,runtime=bridge_fixture(owner)
    runtime_lock=threading.Lock()
    model_entered=threading.Event();release_model=threading.Event();frozen=threading.Event()
    cleanup_entered=threading.Event();errors=[]
    original_stop=runtime.stop;original_halt=bridge.halt
    def model_work():
        with runtime_lock:
            model_entered.set()
            release_model.wait()
    def cleanup():
        cleanup_entered.set()
        with runtime_lock:original_stop()
    def halt():
        assert owner.stop_event.is_set()
        value=original_halt();frozen.set();return value
    def fault():raise RuntimeError('fake read fault before a policy write')
    runtime.stop=cleanup;bridge.halt=halt
    if operation=='execute_fault':bridge.read_q=fault
    def invoke():
        try:
            bridge.stop() if operation=='stop' else bridge.execute()
        except BaseException as exc:errors.append(exc)
    model=threading.Thread(target=model_work);caller=threading.Thread(target=invoke)
    model.start()
    try:
        assert model_entered.wait(2.)
        caller.start()
        assert frozen.wait(2.), 'physical freeze waited for the model/runtime lock'
        assert cleanup_entered.wait(2.) and caller.is_alive()
        assert not release_model.is_set() and bus.events.count('goal_write')==1
        with pytest.raises(RuntimeError,match='STOP is latched'):
            owner._write_q(np.full(6,8.),np.full(6,9.))
        assert bus.events.count('goal_write')==1
    finally:
        release_model.set();model.join(2.)
        if caller.ident is not None:caller.join(2.)
    assert not model.is_alive() and not caller.is_alive()
    if operation=='stop':assert not errors
    else:
        assert len(errors)==1 and 'fake read fault' in str(errors[0])
        assert not bridge.last_attempt['completed']


@pytest.mark.parametrize('operation',['stop','execute_fault'])
def test_runtime_cleanup_exception_cannot_skip_the_physical_freeze(operation):
    owner,bus=fixture();bridge,runtime=bridge_fixture(owner)
    def cleanup():
        assert owner.stop_event.is_set() and bus.events.count('goal_write')==1
        raise RuntimeError('fake runtime cleanup failed')
    def fault():raise RuntimeError('fake initial read failed')
    runtime.stop=cleanup
    if operation=='execute_fault':bridge.read_q=fault
    with pytest.raises(RuntimeError,match='runtime cleanup failed'):
        bridge.stop() if operation=='stop' else bridge.execute()
    assert bus.events.count('goal_write')==1
    errors=bridge.last_stop_errors if operation=='stop' else bridge.last_attempt
    assert 'runtime cleanup failed' in errors['runtime_cleanup_error']
    assert 'physical_hold_error' not in errors


def test_hold_and_journal_failure_preserve_original_error_without_add_note(tmp_path):
    class LegacyHoldError(ConnectionError):
        def __getattribute__(self,name):
            if name=='add_note':raise AttributeError(name)
            return super().__getattribute__(name)
    owner,_=fixture(tmp_path)
    original=LegacyHoldError('original physical hold failed')
    def operation(events):
        events.append(dict(event='fake_hold_write'))
        raise original
    def journal(*args):raise OSError('fake event disk failed')
    owner.io._append_command_journal=journal
    assert not hasattr(original,'add_note')
    with pytest.raises(LegacyHoldError) as caught:owner._record_hold(operation)
    assert caught.value is original
    assert 'fake event disk failed' in owner.last_actuator_persistence_error


@pytest.mark.parametrize('operation',['stop','execute_fault'])
def test_physical_failure_stays_primary_when_runtime_cleanup_also_fails(operation):
    owner,bus=fixture();bridge,runtime=bridge_fixture(owner);order=[]
    original=ConnectionError('original physical HOLD write failure')
    write=bus.sync_write
    def failed_write(*args,**kwargs):
        assert owner.stop_event.is_set()
        order.append('physical_hold');write(*args,**kwargs);raise original
    def cleanup():
        order.append('runtime_cleanup')
        raise RuntimeError('secondary model cleanup failure')
    def fault():raise ValueError('initial execution read failure')
    bus.sync_write=failed_write;runtime.stop=cleanup
    if operation=='execute_fault':bridge.read_q=fault
    with pytest.raises(ConnectionError) as caught:
        bridge.stop() if operation=='stop' else bridge.execute()
    assert caught.value is original
    assert order==['physical_hold','runtime_cleanup'] and bus.events.count('goal_write')==1
    errors=bridge.last_stop_errors if operation=='stop' else bridge.last_attempt
    assert 'original physical HOLD write failure' in errors['physical_hold_error']
    assert 'secondary model cleanup failure' in errors['runtime_cleanup_error']
    if operation=='execute_fault':
        assert 'initial execution read failure' in errors['error']
        assert not errors['completed']
