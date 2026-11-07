"""Fake registers only: no hardware connection or physical task success claims."""
from types import SimpleNamespace,MethodType
import threading

import numpy as np
import pytest

from fcpg.v21_hardware_state import ORDER,connect_preserving_warm_hold,verified_hold,actuator_snapshot


class Bus:
    def __init__(self,torque=1):
        self.is_connected=False;self.protocol_version=0;self.goal=np.full(6,9.);self.bias=np.ones(6)
        self.torque=np.full(6,torque);self.events=[];self.bad={}
        self.motors={name:SimpleNamespace(id=i+1) for i,name in enumerate(ORDER)}
        self.calibration={name:SimpleNamespace(range_min=0,range_max=2000) for name in ORDER}
    def _normalize(self,values):return {motor_id:(raw-1000)/10 for motor_id,raw in values.items()}
    def _unnormalize(self,values):return {motor_id:round(1000+10*q) for motor_id,q in values.items()}
    def connect(self):self.events.append('connect');self.is_connected=True
    def disconnect(self,disable_torque=True):
        assert disable_torque is False
        self.events.append('disconnect_preserving_torque');self.is_connected=False
    def sync_read(self,register,normalize=True,num_retry=0):
        if register=='Goal_Position':values=self.goal
        elif register=='Present_Position':values=self.goal+self.bias
        elif register=='Torque_Enable':values=self.torque
        else:
            default={'Operating_Mode':0,'P_Coefficient':16,'I_Coefficient':0,'D_Coefficient':0,
                     'Return_Delay_Time':0,'Acceleration':254,'Maximum_Acceleration':254,'Moving':0}[register]
            values=np.full(6,self.bad.get(register,default))
        if register in ('Goal_Position','Present_Position') and not normalize:values=1000+10*values
        return dict(zip(ORDER,values.tolist()))
    def read(self,register,motor,**kwargs):
        if motor=='elbow_flex' and register=='Torque_Limit':
            return self.bad.get('J3_Torque_Limit',1000)
        assert motor=='gripper'
        return self.bad.get(register,{'Max_Torque_Limit':500,'Protection_Current':250,'Overload_Torque':25}[register])
    def sync_write(self,register,values,normalize=True,num_retry=0):
        assert register=='Goal_Position'  # No hidden Torque/PID changes permitted.
        self.events.append('goal_write')
        self.goal=np.array([values[k] for k in ORDER],float)
        if not normalize:self.goal=(self.goal-1000)/10


def io_fixture(torque=1):
    bus=Bus(torque)
    class Camera:
        is_connected=False
        def connect(self):self.is_connected=True;bus.events.append('camera_connect')
    class Follower:
        def __init__(self):self.bus=bus;self.cameras={'wrist':Camera(),'column':Camera()};self.calibrated=True
        @property
        def is_calibrated(self):bus.events.append('calibration_read');return self.calibrated
        @property
        def is_connected(self):return bus.is_connected and all(c.is_connected for c in self.cameras.values())
    io=SimpleNamespace(follower=Follower(),settle_position_tolerance=3.,settle_step_tolerance=.25,
        calibration_reconstructed_encoder_cells=lambda q:np.rint(1000+10*np.asarray(q)),
        clearance_gate=SimpleNamespace(require_readback=lambda *a,**k:None,require_policy_path=lambda *a,**k:None))
    return io,bus


def test_warm_connection_preserves_goals_torque_and_parameters_without_writes():
    io,bus=io_fixture();before=bus.goal.copy()
    result=connect_preserving_warm_hold(io,wait=lambda _:None)
    assert result['connection_mode']=='warm_preserved_goal' and result['motor_register_writes']==0
    assert 'goal_write' not in bus.events and io.follower.is_connected
    np.testing.assert_array_equal(before,bus.goal)
    assert bus.events.count('calibration_read')==2 and result['position_delta']==[0.]*6


def test_cold_connection_keeps_the_established_safe_initializer():
    io,bus=io_fixture(torque=0);calls=[]
    result=connect_preserving_warm_hold(io,
        legacy_preseed=lambda b:(calls.append('preseed') or {'seed_verified':True}),
        legacy_connect=lambda f:calls.append('configure'),wait=lambda _:None)
    assert calls==['preseed','configure'] and 'disconnect_preserving_torque' in bus.events
    assert result['connection_mode']=='cold_existing_safety_path' and result['seed_verified']


@pytest.mark.parametrize('damage',['configuration','calibration','mixed_torque','moving_pose','changed_goal','raw_zero','j3_output_cap'])
def test_invalid_warm_connection_holds_without_unloading_or_reconfiguring(damage):
    io,bus=io_fixture()
    if damage=='configuration':bus.bad['D_Coefficient']=32
    elif damage=='j3_output_cap':bus.bad['J3_Torque_Limit']=200
    elif damage=='calibration':io.follower.calibrated=False
    elif damage=='mixed_torque':bus.torque[2]=0
    elif damage=='raw_zero':bus.goal[0]=-100
    else:
        camera=io.follower.cameras['column'];original=camera.connect
        def change():
            original()
            if damage=='changed_goal':bus.goal[1]+=.5
            else:bus.bias[1]+=.5
        camera.connect=change
    with pytest.raises(RuntimeError):connect_preserving_warm_hold(io,wait=lambda _:None)
    assert bus.events.count('goal_write')==1  # Only the verified raw emergency hold.


def hold_owner():
    from scripts.serve_so101_realtime_fcpg_lewm import HardwareSO101IO
    io,bus=io_fixture();bus.is_connected=True
    io._read_state_only=lambda:bus.goal+bus.bias
    io._append_command_journal=lambda *a:None
    io._hold_at_measured_pose=MethodType(HardwareSO101IO._hold_at_measured_pose,io)
    return SimpleNamespace(io=io,bus_lock=threading.RLock()),bus


def test_repeated_stop_does_not_ratchet_a_fixed_goal_toward_sagged_present():
    owner,bus=hold_owner()
    first=verified_hold(owner);goal=bus.goal.copy()
    assert first['motor_register_writes']==1
    for _ in range(5):
        record=verified_hold(owner)
        assert record['reused_existing_hold'] and record['motor_register_writes']==0
    np.testing.assert_array_equal(bus.goal,goal)
    assert bus.events.count('goal_write')==1


def test_changed_or_invalidated_goal_gets_a_fresh_verified_stop():
    owner,bus=hold_owner();verified_hold(owner)
    bus.goal[0]+=.5
    assert not verified_hold(owner)['reused_existing_hold']
    owner._v21_hold_cache=None
    assert not verified_hold(owner)['reused_existing_hold']
    assert bus.events.count('goal_write')==3


def test_read_only_actuator_snapshot_has_both_units_and_no_motor_writes():
    _,bus=io_fixture();before=bus.goal.copy()
    result=actuator_snapshot(bus)
    assert result['present_minus_goal']['shoulder_lift']==1.
    assert result['present_raw']['shoulder_lift']==1100
    np.testing.assert_array_equal(before,bus.goal)
    assert not bus.events and result['motor_register_writes']==0


def test_snapshot_finish_time_includes_all_register_reads(monkeypatch):
    import fcpg.v21_hardware_state as state
    now=[0.];_,bus=io_fixture();read=bus.sync_read
    monkeypatch.setattr(state.time,'monotonic',lambda:now[0])
    def measured(*args,**kwargs):
        result=read(*args,**kwargs);now[0]+=.1;return result
    bus.sync_read=measured
    result=state.actuator_snapshot(bus)
    assert result['started_s']==0.
    assert result['finished_s']==pytest.approx(.4)
    assert result['position_read_intervals_s']=={
        'Present_Position':{'started_s':0.,'returned_s':.1},
        'Goal_Position':{'started_s':.1,'returned_s':.2},
    }


def test_snapshot_decodes_the_same_raw_samples_even_while_positions_drift():
    _,bus=io_fixture();read=bus.sync_read;calls=[];events=[]
    def drifting(register,**kwargs):
        calls.append((register,kwargs['normalize']))
        value=read(register,**kwargs)
        bus.goal+=.1
        return value
    bus.sync_read=drifting
    result=actuator_snapshot(bus,read_events=events)
    assert calls==[(name,False) for name in
                   ('Present_Position','Goal_Position','Torque_Enable','Moving')]
    for raw_key,normalized_key in [('present_raw','present_position'),('goal_raw','goal_position')]:
        for name in ORDER:
            assert result[normalized_key][name]==pytest.approx((result[raw_key][name]-1000)/10)
    assert result['present_minus_goal'][ORDER[0]]==pytest.approx(.9)
    assert result['normalised_position_source']=='same_raw_register_read'
    assert [event['register'] for event in events]==[name for name,_ in calls]
    assert not bus.events and result['motor_register_writes']==0


@pytest.mark.parametrize('fault',['missing','extra','nonfinite','boolean'])
def test_snapshot_rejects_invalid_local_normalization_without_writes(fault):
    _,bus=io_fixture();normalize=bus._normalize
    def corrupt(raw):
        values=normalize(raw)
        if fault=='missing':values.pop(1)
        elif fault=='extra':values[99]=0.
        elif fault=='nonfinite':values[1]=float('nan')
        else:values[1]=True
        return values
    bus._normalize=corrupt
    with pytest.raises(ValueError,match='normalization'):
        actuator_snapshot(bus)
    assert not bus.events


@pytest.mark.parametrize('cancel_on',[1,2,3,4])
def test_snapshot_cancellation_stops_at_each_actual_register_read(cancel_on):
    _,bus=io_fixture();read=bus.sync_read;calls=[];events=[]
    def cancel(register,**kwargs):
        calls.append(register)
        return read(register,**kwargs)
    bus.sync_read=cancel
    with pytest.raises(RuntimeError,match='cancelled'):
        actuator_snapshot(bus,cancelled=lambda:len(calls)>=cancel_on,read_events=events)
    assert len(calls)==cancel_on and len(events)==cancel_on
    assert all('returned_s' in event for event in events)
    assert not bus.events


def test_clamped_normalization_cannot_make_invalid_raw_positions_look_safe():
    io,bus=io_fixture();original=bus.sync_read
    def corrupt(register,normalize=True,**kwargs):
        result=original(register,normalize=normalize,**kwargs)
        if register=='Present_Position' and not normalize:result['shoulder_lift']=2100
        return result
    bus.sync_read=corrupt
    # The actual LeRobot codec clamps raw values to calibrated limits. The
    # warm-connect reconstruction must still reject a raw cell beyond them.
    bus._normalize=lambda values:{motor_id:(min(2000,max(0,raw))-1000)/10
                                  for motor_id,raw in values.items()}
    with pytest.raises(RuntimeError,match='out-of-range'):
        connect_preserving_warm_hold(io,wait=lambda _:None)
    assert bus.events.count('goal_write')==1
