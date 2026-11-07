"""Offline owner context: no device, model inference, waits, or extra reads."""
from copy import deepcopy
import threading
from types import MethodType

import numpy as np
import pytest

from fcpg.v21_actuator_history import ActuatorContextHistory, history_for
from scripts.canary_so101_v21_research import TrialOwner
from test_v21_hardware_state import hold_owner


def test_missing_history_is_not_a_zero_velocity_or_a_synthetic_goal():
    row=ActuatorContextHistory().snapshot(10.)
    assert row['recent_actual_readbacks']==[]
    assert row['latest_observed_displacement'] is None
    assert row['last_nonzero_observed_displacement'] is None
    assert row['last_observed_operation'] is None
    assert not row['complete_bus_write_history'] and not row['lewm_input_used']


def test_direction_uses_measured_displacement_and_retains_its_true_age():
    h=ActuatorContextHistory()
    h.observe_read(np.zeros(6),1.,1.01)
    q=np.array([0.,0.,-.2,0.,0.,0.])
    h.observe_read(q,1.1,1.11)
    h.observe_read(q,3.,3.01)
    q[2]=999  # stored values must not alias caller buffers
    result=h.snapshot(4.)
    assert result['latest_observed_displacement']['delta_position']==[0.]*6
    previous=result['last_nonzero_observed_displacement']
    assert previous['direction']==[0,0,-1,0,0,0]
    assert previous['average_velocity_position_per_host_s'][2]==pytest.approx(-2.)
    assert previous['age_s']==pytest.approx(2.89)
    assert not previous['instantaneous_servo_velocity']
    result['recent_actual_readbacks'][0]['q_position'][0]=999
    assert h.snapshot(4.)['recent_actual_readbacks'][0]['q_position'][0]==0.


def test_failed_read_cannot_bridge_an_unknown_interval():
    h=ActuatorContextHistory();h.observe_read(np.zeros(6),1.,1.01)
    h.observe_read_failure('timeout',2.)
    assert h.snapshot(2.1)['last_read_error']['error']=='timeout'
    h.observe_read(np.ones(6),3.,3.01)
    assert h.snapshot(4.)['latest_observed_displacement'] is None
    assert h.snapshot(4.)['last_nonzero_observed_displacement'] is None


def test_boundary_discards_pre_home_or_direct_io_history():
    h=ActuatorContextHistory();h.observe_read(np.zeros(6),1.,1.01)
    h.observe_operation('position_goal_write',1.2,1.21,command=np.ones(6))
    h.invalidate('direct HOME path',2.)
    row=h.snapshot(3.)
    assert row['last_observed_operation'] is None and row['recent_actual_readbacks']==[]
    assert row['history_boundary']['reason']=='direct HOME path'


def test_snapshot_does_not_export_future_reads_or_operations():
    h=ActuatorContextHistory();h.observe_read(np.zeros(6),1.,1.01)
    h.observe_read(np.ones(6),3.,3.01)
    h.observe_operation('position_goal_write',3.1,3.11,command=np.ones(6))
    row=h.snapshot(2.)
    assert len(row['recent_actual_readbacks'])==1
    assert row['latest_observed_displacement'] is None and row['last_observed_operation'] is None


def owner_fixture():
    owner,bus=hold_owner();owner.stop_event=threading.Event()
    owner._lease_check=lambda:None
    for name in ('_read_q','_write_q','_read_pre_dispatch_state','_record_hold','_halt'):
        setattr(owner,name,MethodType(getattr(TrialOwner,name),owner))
    owner.io._write_position_command=lambda command,previous:(
        bus.sync_write('Goal_Position',dict(zip(bus.motors,np.asarray(command).tolist()))) or command)
    return owner,bus


def test_owner_records_real_reads_and_write_without_extra_bus_io():
    owner,bus=owner_fixture();calls=[]
    read=owner.io._read_state_only
    owner.io._read_state_only=lambda:(calls.append('read') or read())
    source=owner._read_q();command=source+.1
    owner._write_q(command,source);owner._read_q()
    row=history_for(owner).snapshot(1e9)
    assert calls==['read','read'] and bus.events==['goal_write']
    last=row['last_observed_operation']
    assert last['kind']=='position_goal_write' and last['returned']
    assert last['requested_position']==command.tolist()
    assert not last['goal_register_verified'] and not last['actual_motion_verified']
    pre=owner._read_pre_dispatch_state()
    assert pre['owner_observation_history']['last_observed_operation']['requested_position']==command.tolist()
    assert calls==['read','read'] and bus.events==['goal_write']


def test_partial_write_failure_is_not_recorded_as_a_returned_command():
    owner,bus=owner_fixture()
    def fail(command,previous):
        raise ConnectionError('partial packet delivery unknown')
    owner.io._write_position_command=fail
    with pytest.raises(ConnectionError):owner._write_q(np.ones(6),np.zeros(6))
    row=history_for(owner).snapshot(1e9)['last_observed_operation']
    assert not row['returned'] and row['returned_position'] is None
    assert 'partial packet' in row['error'] and not row['actual_motion_verified']


def test_latched_stop_prevents_write_and_preserves_last_observed_operation():
    owner,bus=owner_fixture();history_for(owner)
    owner.stop_event.set()
    before=deepcopy(owner._actuator_context_history.last_operation)
    with pytest.raises(RuntimeError,match='STOP'):owner._write_q(np.ones(6),np.zeros(6))
    assert bus.events==[] and owner._actuator_context_history.last_operation==before


def test_hold_context_differentiates_write_from_reverification():
    owner,bus=owner_fixture()
    owner._halt()
    first=history_for(owner).snapshot(1e9)['last_observed_operation']
    owner._halt()
    second=history_for(owner).snapshot(1e9)['last_observed_operation']
    assert first['kind']==second['kind']=='hold'
    assert first['motor_register_writes']==1 and second['motor_register_writes']==0
    assert not first['reused_existing_hold'] and second['reused_existing_hold']
    assert first['goal_readback_raw']==second['goal_readback_raw']
    assert bus.events==['goal_write']


def test_real_normal_completion_keeps_verified_goal_without_inventing_a_write():
    from test_v21_normal_completion_hold import fixture, follow
    owner,bus=fixture()
    record=owner._completion_hold(bus.goal.copy(),follow(bus),bus.goal+bus.bias)
    row=history_for(owner).snapshot(1e9)['last_observed_operation']
    assert record['motor_register_writes']==0 and not bus.events
    assert row['kind']=='hold' and row['retained_existing_goal'] is True
    assert row['goal_register_verified'] is True
    assert row['motor_register_writes']==0 and row['goal_readback_raw']


def test_failed_hold_does_not_retain_previous_success_as_latest_operation():
    owner,bus=owner_fixture();owner._halt()
    def uncertain(events):
        bus.sync_write('Goal_Position',dict(zip(bus.motors,[9.]*6)))
        raise ConnectionError('HOLD verification failed')
    with pytest.raises(ConnectionError):owner._record_hold(uncertain)
    row=history_for(owner).snapshot(1e9)['last_observed_operation']
    assert row['kind']=='hold' and not row['returned']
    assert row.get('goal_register_verified') is not True
    assert 'verification failed' in row['error']
