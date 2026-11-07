"""Manual HOME adapter with no camera/serial access and no model-history writes."""
from types import SimpleNamespace
import json
import threading

import numpy as np
import pytest

from fcpg.v21_home import execute_home_readback_chunk, home_clearance_gate, plan_home_chunk
from scripts.serve_so101_prod_v17_7 import V177ClearanceGate


class FlatKinematics:
    def forward_anchors(self, q):
        return np.tile([0., 0., .1], (3, 1))


def fake_owner(*,cancel_after=None,mutate=False,reject=False,settle_error=False):
    now=[0.];q=np.zeros(6,np.float32);writes=[];holds=[];audits=[];journal=[];settles=[]
    class Event:
        stopped=False
        def is_set(self):return self.stopped
        def set(self):self.stopped=True
        def wait(self,delay):
            now[0]+=delay
            if cancel_after is not None and len(writes)>=cancel_after:self.set()
            return self.stopped
    event=Event()
    def audit(current,commands):
        audits.append((np.asarray(current).copy(),commands.copy()))
        if reject:raise RuntimeError('mechanical limit')
    def write(command,previous):
        writes.append(command.copy());q[:]=command
        return command+1 if mutate else command.copy()
    def settle(target,**kwargs):
        settles.append((target.copy(),kwargs))
        if settle_error:raise RuntimeError('existing settle contract failed')
        return SimpleNamespace(outcome='settled')
    owner=SimpleNamespace(stop_event=event,bus_lock=threading.RLock(),
        clearance=V177ClearanceGate(kinematics=FlatKinematics(),table_z_mm=0.),
        _lease_check=lambda:None,read_checked=lambda:q.copy(),_physical_check=audit,
        _write_q=write,_halt=lambda:holds.append(now[0]),
        io=SimpleNamespace(_start_command_journal=lambda requested:None,
            _append_command_journal=lambda path,row:journal.append(row),
            _settle_commanded_endpoint=settle,
            observe=lambda:pytest.fail('HOME must not require missing Column frames')))
    return owner,now,writes,holds,audits,settles,journal


def test_home_retains_audit_readbacks_pacing_and_existing_settlement():
    owner,now,writes,holds,audits,settles,journal=fake_owner()
    commands=np.arange(30,dtype=np.float32).reshape(5,6)/10
    result=execute_home_readback_chunk(owner,commands,clock=lambda:now[0])
    assert result['completed'] and not result['model_transition']
    assert result['policy_commands_attempted']==result['policy_commands_returned']==5
    assert owner.last_home_attempt is result
    np.testing.assert_array_equal(writes,commands)
    np.testing.assert_allclose(result['command_times_s'],[0,.2,.4,.6,.8])
    assert len(result['measured_q_path'])==6 and len(audits)==6
    assert len(settles)==1 and len(holds)==1
    assert settles[0][1]['period_s']==.2 and settles[0][1]['stop_event'] is owner.stop_event
    assert journal[-1]['event']=='home_chunk_completed'


def test_completed_home_chunk_persists_physical_pending_receipt(tmp_path, monkeypatch):
    import fcpg.v21_home as module
    owner,now,_,_,_,_,_=fake_owner()
    owner.output=tmp_path
    (tmp_path/'last-attempt.json').write_text(json.dumps({'source_time_s':17.0}))
    journal_path=tmp_path/'command_journal/chunk_1/events.jsonl'
    journal_path.parent.mkdir(parents=True)
    def append(_path,row):
        with journal_path.open('a') as stream:
            stream.write(json.dumps(row,default=lambda value:value.tolist())+'\n')
    def start(requested):
        append(journal_path,dict(event='chunk_intent',requested=requested,requested_commands=5))
        return journal_path
    owner.io._start_command_journal=start
    owner.io._append_command_journal=append
    raw=dict(zip(('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'),
                 [10000,10000,10000,10000,10000,10000]))
    position={name:(value-10000)/100. for name,value in raw.items()}
    diagnostics=dict(motor_register_writes=0,errors={},registers={})
    snapshot=dict(goal_raw=raw,present_raw=raw,goal_position=position,present_position=position,
                  moving={name:0 for name in raw},torque={name:1 for name in raw},
                  motor_register_writes=0)
    monkeypatch.setattr(module, '_settled_register_state', lambda owner, final: dict(
        goal_raw=raw, present_raw=raw, goal_position=position, present_position=position,
        raw_readback_source='same_raw_register_read',
        present_minus_goal={name:0. for name in raw},actuator_snapshot=snapshot,
        actuator_diagnostics=diagnostics))
    result=execute_home_readback_chunk(owner,np.zeros((5,6),np.float32),clock=lambda:now[0])
    path=tmp_path/'pending-home-receipt.json'
    receipt=json.loads(path.read_text())
    assert result['completed'] and result['pending_home_receipt_path']==str(path.resolve())
    assert receipt['schema']=='v21-home-physical-settled-receipt-v1'
    assert receipt['requires_home'] is True and receipt['physical_settled'] is True
    assert receipt['raw_freeze_verified'] is False
    assert receipt['goal_raw']==raw and receipt['present_raw']==raw
    assert receipt['goal_position']==receipt['settled_q_position']
    assert receipt['journal_settled_q_position']==result['settled_q_position']
    assert receipt['command_journal']==str(journal_path.resolve())


def test_home_completes_only_after_final_hold_and_completion_journal_return():
    owner,now,_,_,_,_,journal=fake_owner()
    stages=[]
    def halt():
        assert owner.last_home_attempt['completed'] is False
        assert not any(row['event']=='home_chunk_completed' for row in journal)
        now[0]+=.03
        stages.append('hold')
    def append(path,row):
        if row['event']=='home_chunk_completed':
            assert stages==['hold']
            assert owner.last_home_attempt['completed'] is False
            assert owner.last_home_attempt['completion_hold']['returned'] is True
            stages.append('journal')
        journal.append(row)
    owner._halt=halt
    owner.io._append_command_journal=append
    result=execute_home_readback_chunk(owner,np.zeros((5,6),np.float32),clock=lambda:now[0])
    assert stages==['hold','journal'] and result['completed'] is True
    assert result['completion_journal_returned'] is True
    assert journal[-1]['monotonic_s']>result['settled_time_s']


@pytest.mark.parametrize('failure_phase',['completion_hold','completion_journal'])
def test_finalization_failure_never_leaves_a_completed_home(failure_phase):
    owner,now,writes,holds,_,_,journal=fake_owner()
    original=RuntimeError('final HOME operation failed')
    def halt():
        holds.append(now[0])
        if failure_phase=='completion_hold' and len(holds)==1:
            raise original
    def append(path,row):
        if row['event']=='home_chunk_completed':
            assert len(holds)==1
            if failure_phase=='completion_journal':
                raise original
        journal.append(row)
    owner._halt=halt
    owner.io._append_command_journal=append
    with pytest.raises(RuntimeError) as raised:
        execute_home_readback_chunk(owner,np.zeros((5,6),np.float32),clock=lambda:now[0])
    assert raised.value is original and owner.stop_event.is_set()
    record=owner.last_home_attempt
    assert record['completed'] is False and record['failure_phase']==failure_phase
    assert record['error']==str(original) and record['error_type']=='RuntimeError'
    assert record['completion_hold']['returned'] is (failure_phase=='completion_journal')
    assert record['completion_journal_returned'] is False
    assert record['fault_cleanup_hold']['returned'] is True and record['fault_journal']['returned'] is True
    assert len(writes)==5 and len(holds)==2
    assert record['policy_commands_attempted']==record['policy_commands_returned']==5
    assert not any(row['event']=='home_chunk_completed' for row in journal)
    assert journal[-1]['event']=='home_chunk_fault' and journal[-1]['completed'] is False


@pytest.mark.parametrize('hold_fails,journal_fails',[(True,False),(False,True),(True,True)])
@pytest.mark.parametrize('error_type',[RuntimeError,KeyboardInterrupt])
def test_fault_cleanup_failures_preserve_original_error_and_failed_evidence(hold_fails,journal_fails,error_type):
    owner,now,writes,holds,_,_,journal=fake_owner()
    original=error_type('original settlement failure')
    def settle(*args,**kwargs):raise original
    def halt():
        holds.append(now[0])
        if hold_fails:raise ConnectionError('fault HOLD unavailable')
    def append(path,row):
        if row['event']=='home_chunk_fault' and journal_fails:
            raise OSError('fault journal unavailable')
        journal.append(row)
    owner.io._settle_commanded_endpoint=settle
    owner._halt=halt
    owner.io._append_command_journal=append
    with pytest.raises(error_type) as raised:
        execute_home_readback_chunk(owner,np.zeros((5,6),np.float32),clock=lambda:now[0])
    assert raised.value is original and owner.stop_event.is_set()
    record=owner.last_home_attempt
    assert record['completed'] is False and record['error']==str(original)
    assert record['failure_phase']=='settlement' and record['completion_hold']['attempted'] is False
    assert record['fault_cleanup_hold']['returned'] is (not hold_fails)
    assert record['fault_journal']['returned'] is (not journal_fails)
    if hold_fails:
        assert record['fault_cleanup_hold']['error_type']=='ConnectionError'
    if journal_fails:
        assert record['fault_journal']['error_type']=='OSError'
    else:
        assert journal[-1]['error']==str(original)
        assert journal[-1]['fault_cleanup_hold']==record['fault_cleanup_hold']
    assert len(writes)==5 and len(holds)==1
    assert not any(row['event']=='home_chunk_completed' for row in journal)


@pytest.mark.parametrize('chunk,period',[
    (np.zeros((4,6)),.2),(np.full((5,6),np.nan),.2),(np.zeros((5,6)),0.)])
def test_invalid_request_cannot_reuse_a_previous_success_and_performs_no_io(chunk,period):
    owner,now,writes,holds,audits,settles,journal=fake_owner()
    owner.last_home_attempt={'completed':True}
    with pytest.raises(ValueError):
        execute_home_readback_chunk(owner,chunk,period_s=period,clock=lambda:now[0])
    record=owner.last_home_attempt
    assert record['completed'] is False and record['failure_phase']=='validation'
    assert record['policy_commands_attempted']==record['policy_commands_returned']==0
    assert not writes and not holds and not audits and not settles and not journal


@pytest.mark.parametrize('options,sent',[
    ({'reject':True},0),({'cancel_after':2},2),({'mutate':True},1),({'settle_error':True},5)])
def test_faults_preserve_stop_without_remaining_commands(options,sent):
    owner,now,writes,holds,_,_,_=fake_owner(**options)
    with pytest.raises(RuntimeError):
        execute_home_readback_chunk(owner,np.zeros((5,6),np.float32),clock=lambda:now[0])
    assert len(writes)==sent and holds and owner.stop_event.is_set()
    assert owner.last_home_attempt['policy_commands_attempted']==sent
    assert owner.last_home_attempt['policy_commands_returned']==sent
    assert owner.last_home_attempt['completed'] is False


@pytest.mark.parametrize('settle_error',[False,True])
def test_direct_home_settlement_invalidates_unobserved_write_history(settle_error):
    from fcpg.v21_actuator_history import history_for
    owner,now,writes,holds,_,_,_=fake_owner(settle_error=settle_error)
    history=history_for(owner)
    history.observe_read(np.zeros(6),-.2,-.1)
    history.observe_operation('position_goal_write',-.09,-.08,command=np.ones(6))
    if settle_error:
        with pytest.raises(RuntimeError,match='settle contract failed'):
            execute_home_readback_chunk(owner,np.zeros((5,6),np.float32),clock=lambda:now[0])
    else:
        execute_home_readback_chunk(owner,np.zeros((5,6),np.float32),clock=lambda:now[0])
    context=history.snapshot(now[0])
    assert context['history_boundary']['reason']=='home_direct_io_settlement'
    assert context['last_observed_operation'] is None
    assert context['recent_actual_readbacks']==[] and len(writes)==5 and len(holds)==1


def test_lost_lease_never_dispatches_home():
    owner,now,writes,holds,_,_,_=fake_owner()
    owner._lease_check=lambda:(_ for _ in ()).throw(RuntimeError('lease lost'))
    with pytest.raises(RuntimeError,match='lease lost'):
        execute_home_readback_chunk(owner,np.zeros((5,6),np.float32),clock=lambda:now[0])
    assert not writes and holds and owner.stop_event.is_set()


def test_already_stopped_never_dispatches_home():
    owner,now,writes,holds,_,_,_=fake_owner();owner.stop_event.set()
    with pytest.raises(RuntimeError,match='cancelled'):
        execute_home_readback_chunk(owner,np.zeros((5,6),np.float32),clock=lambda:now[0])
    assert not writes and holds


def test_stop_arriving_while_waiting_for_bus_prevents_the_write():
    from scripts.canary_so101_v21_research import TrialOwner
    owner=TrialOwner.__new__(TrialOwner);owner.stop_event=threading.Event();writes=[]
    class BusLock:
        def __enter__(self):owner.stop_event.set()
        def __exit__(self,*args):pass
    owner.bus_lock=BusLock()
    owner.io=SimpleNamespace(_write_position_command=lambda *args:writes.append(args))
    with pytest.raises(RuntimeError,match='STOP is latched'):
        owner._write_q(np.zeros(6),np.zeros(6))
    assert not writes


def test_bus_writer_preserves_exact_command_when_stop_is_clear():
    from scripts.canary_so101_v21_research import TrialOwner
    owner=TrialOwner.__new__(TrialOwner);owner.stop_event=threading.Event()
    owner.bus_lock=threading.RLock();writes=[]
    def write(command,previous):writes.append((command,previous));return command
    owner.io=SimpleNamespace(_write_position_command=write)
    command=np.arange(6,dtype=np.float32);previous=np.zeros(6,np.float32)
    assert owner._write_q(command,previous) is command
    assert writes[0][0] is command and writes[0][1] is previous


def real_home_owner():
    pytest.importorskip('mujoco')
    from scripts.serve_so101_prod_v17_7 import SO101MuJoCoKinematics, DEFAULT_KINEMATICS_MODEL
    return SimpleNamespace(
        clearance=V177ClearanceGate(kinematics=SO101MuJoCoKinematics(DEFAULT_KINEMATICS_MODEL)),
        io=SimpleNamespace(settle_position_tolerance=3.))


def failed_home_source():
    # Actual run's final chunk source; calibrated position units, not degrees.
    return np.array([-1.8063583374,-61.9596557617,81.5439224243,
                     62.3322143555,-51.5844154358,31.0118274689],np.float32)


def test_recorded_near_floor_chunk_is_rejected_before_any_write():
    owner=real_home_owner();start=failed_home_source()
    end=start.copy();end[1]-=12.;end[2]+=12.
    commands=np.stack([start+(end-start)*a for a in np.linspace(0,1,6)[1:]])
    owner.clearance.require_policy_path(start,commands)  # old nominal check passed
    assert owner.clearance.clearance_mm(commands[-1]) == pytest.approx(-2.7057,abs=.01)
    with pytest.raises(RuntimeError,match='clearance'):
        home_clearance_gate(owner).require_policy_path(start,commands)
    fake,now,writes,holds,_,_,_=fake_owner()
    fake.clearance=owner.clearance;fake.read_checked=lambda:start.copy()
    with pytest.raises(RuntimeError,match='clearance'):
        execute_home_readback_chunk(fake,commands,clock=lambda:now[0])
    assert not writes and holds and fake.stop_event.is_set()


def test_buffer_is_local_and_never_reduces_stricter_existing_gate():
    from dataclasses import replace
    owner=real_home_owner();original=owner.clearance
    gate=home_clearance_gate(owner)
    assert gate.planned_clearance_mm==2. and gate.readback_clearance_mm==-3.
    assert owner.clearance is original and original.planned_clearance_mm==-3.
    owner.clearance=replace(original,planned_clearance_mm=25.)
    assert home_clearance_gate(owner).planned_clearance_mm==25.


def test_new_route_reaches_home_with_recorded_tracking_bias_offline():
    owner=real_home_owner();current=failed_home_source()
    target=np.array([.1445086747,-97.9415359497,98.2253799438,
                     62.0805358887,-51.3766250610,31.3403415680],np.float32)
    bias=np.array([0.,.2256088257,.4223632813,.0838928223,0.,0.],np.float32)
    for _ in range(80):
        if np.max(np.abs(current-target))<=3.:break
        commands=plan_home_chunk(owner,current,target)
        for command in commands:
            owner.clearance.require_readback(command+bias,label='recorded-bias fixture')
        current=commands[-1]+bias
    else:
        pytest.fail('buffered HOME failed to reach target in the unchanged budget')
    assert owner.clearance.readback_clearance_mm==-3.


def test_below_hard_floor_never_gets_automatic_recovery():
    owner=real_home_owner()
    current=np.array([-1.8063583374,-73.7340469360,93.9662857056,
                      62.4161071777,-51.5844154358,31.0118274689],np.float32)
    assert owner.clearance.clearance_mm(current)<-3.
    with pytest.raises(RuntimeError,match='HOME source'):
        plan_home_chunk(owner,current,np.zeros(6,np.float32))


def test_inside_home_buffer_allows_only_existing_upward_recovery():
    owner=real_home_owner();current=failed_home_source()
    current[1]-=9.6;current[2]+=9.6
    gate=home_clearance_gate(owner)
    assert gate.readback_clearance_mm < gate.clearance_mm(current) < gate.planned_clearance_mm
    target=current.copy();target[1]=-97.;target[2]=98.
    commands=plan_home_chunk(owner,current,target)
    audit=gate.require_policy_path(current,commands)
    assert audit['mode']=='monotonic_recovery'
    heights=[gate.clearance_mm(current),*[gate.clearance_mm(q) for q in commands]]
    assert np.all(np.diff(heights)>=-1e-9) and heights[-1]>=gate.planned_clearance_mm
