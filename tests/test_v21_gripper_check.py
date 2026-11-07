from types import SimpleNamespace
import threading
import numpy as np
import pytest

from fcpg.v21_gripper_check import empty_gripper_follow_check
from fcpg.v21_hardware_bridge import V21TransactionBridge


def owner_fixture(cancel_time=None):
    now=[10.];last=[10.]
    q=np.array([0.,0.,0.,0.,0.,31.],np.float32);goal=q.copy()
    writes=[];holds=[];checks=[];event=threading.Event()
    runtime=SimpleNamespace(pending=None,executing=False,state=object(),stopped=True)
    def stop():runtime.stopped=True;runtime.pending=None
    runtime.stop=stop
    def read():
        q[:]+=(goal-q)*(1-np.exp(-(now[0]-last[0])/.25));last[0]=now[0]
        return q.copy()
    def write(target,previous):
        read();goal[:]=target;writes.append(target.copy());now[0]+=.002
        return target.copy()
    def halt():
        goal[:]=read();holds.append(now[0])
    def physical(source,commands):
        assert commands.shape==(5,6)
        assert np.all(commands[:,:5]==source[:5])
        assert 0<=commands[:,5].min()<=commands[:,5].max()<=100
        checks.append(commands.copy())
    bridge=V21TransactionBridge(runtime,read_q=read,write_q=write,capture_endpoint=lambda:None,
        physical_check=physical,halt=halt,clock=lambda:now[0],
        wait=lambda dt:now.__setitem__(0,now[0]+dt),
        cancelled=lambda:event.is_set() or (cancel_time is not None and now[0]>=10+cancel_time))
    owner=SimpleNamespace(bridge=bridge,runtime=runtime,stop_event=event,
        io=SimpleNamespace(settle_position_tolerance=3.,settle_step_tolerance=.25),_physical_check=physical)
    return owner,q.copy(),writes,holds,checks


def test_nominal_q5_and_extra_readbacks_remain_separate_and_opening_restores():
    owner,home,writes,holds,checks=owner_fixture()
    result=empty_gripper_follow_check(owner,home)
    assert result['completed'] and not result['model_transition']
    assert len(writes)==10 and len(holds)==2 and len(checks)==2
    first,second=result['phases']
    for p in result['phases']:
        assert len(p['measured_q_path'])==6 and len(p['tail_positions'])==8
        assert p['readback_times_s'][-1]-p['source_time_s']==pytest.approx(.5)
        assert p['tail_times_s'][-1]-p['source_time_s']==pytest.approx(.9)
    assert first['commands_position'][-1][5]==25.
    assert abs(first['tail_positions'][-1][5]-25)<abs(first['measured_q_path'][-1][5]-25)
    assert second['commands_position'][-1][5]==31.
    assert owner.runtime.state is None and owner.runtime.stopped and owner.stop_event.is_set()


def test_stop_during_tail_prevents_restore_commands():
    owner,home,writes,holds,checks=owner_fixture(cancel_time=.65)
    with pytest.raises(RuntimeError,match='cancelled'):
        empty_gripper_follow_check(owner,home)
    assert len(writes)==5 and holds and len(checks)==1
    assert not owner.gripper_check_record['completed']
    assert len(owner.gripper_check_record['phases'][0]['measured_q_path'])==6
    assert owner.runtime.stopped and owner.stop_event.is_set()


def test_non_home_pose_rejects_before_any_command():
    owner,home,writes,holds,checks=owner_fixture();home[0]=10
    with pytest.raises(RuntimeError,match='HOME'):
        empty_gripper_follow_check(owner,home)
    assert not writes and not checks


def test_physical_rejection_cannot_be_bypassed():
    owner,home,writes,holds,checks=owner_fixture()
    def reject(*args):raise RuntimeError('physical guard')
    owner._physical_check=reject
    with pytest.raises(RuntimeError,match='physical guard'):
        empty_gripper_follow_check(owner,home)
    assert not writes and holds
