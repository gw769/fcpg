"""Per-write raw Goal verification; no physical transport is constructed."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from fcpg.v21_hardware_state import ORDER, verify_policy_goal_position
from scripts.canary_so101_v21_research import TrialOwner
from test_v21_hardware_bridge import setup


class Bus:
    def __init__(self):
        self.motors={name:SimpleNamespace(id=i+1) for i,name in enumerate(ORDER)}
        self.goal={name:2000 for name in ORDER}
        self.reads=[]
        self.read_hook=lambda:None
        self.mismatch=False
        self.fail=None
        self.conversions=[]

    def _unnormalize(self, values):
        self.conversions.append(dict(values))
        # Include an inverted encoder and truncation to exercise exact cells.
        return {i:int(2000+(1 if i!=3 else -1)*float(v)*10) for i,v in values.items()}

    def _normalize(self, values):
        return {i:(1 if i!=3 else -1)*(float(v)-2000)/10 for i,v in values.items()}

    def latch_policy_write(self, command):
        values=self._unnormalize({self.motors[name].id:float(command[j]) for j,name in enumerate(ORDER)})
        self.goal={name:values[self.motors[name].id] for name in ORDER}

    def sync_read(self, register, **kwargs):
        assert register=='Goal_Position' and kwargs==dict(normalize=False,num_retry=0)
        self.reads.append((register,kwargs))
        self.read_hook()
        if self.fail is not None:raise self.fail
        value=dict(self.goal)
        if self.mismatch:value['elbow_flex']+=1
        return value

    def sync_write(self,*args,**kwargs):
        pytest.fail('verification must not issue motor register writes')


def test_exact_raw_cells_use_same_conversion_one_read_and_local_normalization():
    bus=Bus();command=np.array([.1999,-.1555,94.892311,70.605873,-53.5,31.648455],np.float32)
    bus.latch_policy_write(command)
    record=verify_policy_goal_position(bus,command)
    assert len(bus.reads)==1 and len(bus.conversions)==2
    assert bus.conversions[0]==bus.conversions[1]
    assert record['goal_raw']==record['expected_goal_raw'] and record['goal_register_verified']
    assert record['motor_register_writes']==0 and not record['model_feedback_label']
    assert record['read_started_s']<=record['read_returned_s']<=record['returned_s']
    assert record['goal_position_from_raw']['shoulder_pan']!=float(command[0])


def test_one_encoder_cell_mismatch_rejected_without_retries():
    bus=Bus();goal=np.ones(6,np.float32);bus.latch_policy_write(goal);bus.mismatch=True
    with pytest.raises(RuntimeError,match='raw readback mismatch') as caught:
        verify_policy_goal_position(bus,goal)
    evidence=caught.value.goal_readback_verification
    assert len(bus.reads)==1 and evidence['read_returned']
    assert not evidence['goal_register_verified']
    assert list(evidence['mismatched_joints'])==['elbow_flex']


def test_stop_before_read_skips_bus_and_stop_during_read_retains_read_count():
    bus=Bus();goal=np.ones(6,np.float32);bus.latch_policy_write(goal)
    with pytest.raises(RuntimeError,match='cancelled before') as caught:
        verify_policy_goal_position(bus,goal,cancelled=lambda:True)
    assert not bus.reads and not caught.value.goal_readback_verification['read_attempted']
    stopped=[False];bus.read_hook=lambda:stopped.__setitem__(0,True)
    with pytest.raises(RuntimeError,match='cancelled after') as caught:
        verify_policy_goal_position(bus,goal,cancelled=lambda:stopped[0])
    assert len(bus.reads)==1
    assert caught.value.goal_readback_verification['read_returned']
    assert caught.value.goal_readback_verification['goal_register_verified']


def attach(bridge,bus):
    original_write=bridge.write_q
    def write(command,previous):
        value=original_write(command,previous)
        bus.latch_policy_write(value)
        return value
    bridge.write_q=write
    bridge.verify_policy_goal=lambda goal:verify_policy_goal_position(
        bus,goal,cancelled=bridge.cancelled,clock=bridge.clock)


def test_five_goal_reads_do_not_change_write_times_or_q0_through_q5_labels():
    original,old_runtime,_,_=setup();baseline=original.execute()
    bridge,runtime,writes,holds=setup();bus=Bus();attach(bridge,bus)
    bus.read_hook=lambda:bridge.wait(.001)
    result=bridge.execute()
    assert len(writes)==5 and len(holds)==1 and len(bus.reads)==5
    assert result['policy_goal_reads_attempted']==result['policy_goal_reads_returned']==result['policy_goal_registers_verified']==5
    assert result['commands_position']==baseline['commands_position']
    assert result['measured_q_path']==baseline['measured_q_path']
    assert result['command_times_s']==pytest.approx(baseline['command_times_s'],abs=1e-12)
    assert result['write_completed_times_s']==pytest.approx(baseline['write_completed_times_s'],abs=1e-12)
    assert result['readback_times_s']==pytest.approx(baseline['readback_times_s'],abs=1e-12)
    assert len(result['read_calls'])==len(baseline['read_calls'])
    for index,event in enumerate(result['policy_goal_readbacks']):
        assert event['command_index']==index
        assert event['read_started_s']>=result['write_completed_times_s'][index]
        assert event['read_returned_s']<result['readback_times_s'][index+1]
        assert event['callback_duration_s']==pytest.approx(.001)
        assert event['motor_register_writes']==0


def test_third_goal_mismatch_keeps_true_partial_write_read_counts_and_original_error():
    bridge,runtime,writes,holds=setup();bus=Bus();attach(bridge,bus)
    bus.read_hook=lambda:setattr(bus,'mismatch',len(bus.reads)==3)
    with pytest.raises(RuntimeError,match='raw readback mismatch'):
        bridge.execute()
    result=bridge.last_attempt
    assert len(writes)==3 and len(holds)==1 and runtime.stopped
    assert result['policy_commands_attempted']==result['policy_commands_returned']==3
    assert result['policy_goal_reads_attempted']==result['policy_goal_reads_returned']==3
    assert result['policy_goal_registers_verified']==2
    assert len(result['measured_q_path'])==3 and len(result['readback_times_s'])==3
    assert 'raw readback mismatch' in result['error']
    assert result['requires_home_after_incomplete_execution'] and not result['completed']


def test_failed_read_is_not_retried_or_counted_as_returned():
    bridge,runtime,writes,holds=setup();bus=Bus();attach(bridge,bus)
    bus.read_hook=lambda:bridge.wait(.051)
    failure=ConnectionError('original goal read transport failure');bus.fail=failure
    with pytest.raises(ConnectionError,match='original goal read transport failure') as caught:
        bridge.execute()
    assert caught.value is failure
    result=bridge.last_attempt
    assert len(writes)==1 and len(bus.reads)==1 and len(holds)==1
    assert result['policy_goal_reads_attempted']==1 and result['policy_goal_reads_returned']==0
    assert result['policy_goal_registers_verified']==0
    assert len(result['measured_q_path'])==1
    assert result['deadline_miss']['observed_lateness_s']==pytest.approx(.053)


def test_goal_read_time_consumes_existing_thirty_ms_deadline_without_retiming():
    bridge,runtime,writes,holds=setup();bus=Bus();attach(bridge,bus)
    bus.read_hook=lambda:bridge.wait(.029)
    with pytest.raises(RuntimeError,match='goal verification missed its deadline'):
        bridge.execute()
    result=bridge.last_attempt;miss=result['deadline_miss']
    assert len(writes)==1 and len(holds)==1
    assert result['write_durations_s'][0]==pytest.approx(.002)
    assert result['write_lateness_s'][0]==pytest.approx(.002)
    assert result['policy_goal_readbacks'][0]['policy_io_lateness_s']==pytest.approx(.031)
    assert miss['allowed_lateness_s']==.03 and miss['observed_lateness_s']==pytest.approx(.031)
    assert miss['kind']=='policy-goal-verification'


def test_stop_after_write_never_starts_verification_read():
    bridge,runtime,writes,holds=setup(cancel_at=1);bus=Bus();attach(bridge,bus)
    with pytest.raises(RuntimeError,match='cancelled before callback'):
        bridge.execute()
    assert len(writes)==1 and not bus.reads and len(holds)==1
    assert bridge.last_attempt['policy_goal_reads_attempted']==0


def test_stop_arriving_during_goal_read_keeps_returned_evidence_and_stops_next_write():
    bridge,runtime,writes,holds=setup();bus=Bus();attach(bridge,bus)
    stopped=[False];bridge.cancelled=lambda:stopped[0]
    bus.read_hook=lambda:stopped.__setitem__(0,True)
    with pytest.raises(RuntimeError,match='cancelled after read'):
        bridge.execute()
    result=bridge.last_attempt
    assert len(writes)==1 and len(bus.reads)==1 and len(holds)==1
    assert result['policy_goal_reads_returned']==result['policy_goal_registers_verified']==1
    assert len(result['measured_q_path'])==1 and not result['completed']


def test_live_owner_uses_existing_lease_and_bus_lock_and_skips_lock_after_stop():
    owner=TrialOwner.__new__(TrialOwner)
    owner.stop_event=threading.Event();owner.bus_lock=threading.Lock();calls=[]
    owner._lease_check=lambda:calls.append('lease')
    bus=Bus();goal=np.ones(6,np.float32);bus.latch_policy_write(goal)
    owner.io=SimpleNamespace(follower=SimpleNamespace(bus=bus))
    def locked():assert owner.bus_lock.locked()
    bus.read_hook=locked
    owner._verify_policy_goal(goal)
    assert calls==['lease'] and len(bus.reads)==1
    owner.stop_event.set()
    with pytest.raises(RuntimeError,match='cancelled before bus lock'):
        owner._verify_policy_goal(goal)
    assert calls==['lease'] and len(bus.reads)==1
