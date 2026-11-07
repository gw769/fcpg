"""Fake fixed readbacks only: no object, contact model, force or hardware IO."""
import numpy as np

from test_v21_normal_completion_hold import fixture, bridge_fixture
from test_v21_hardware_state import ORDER


def closing_case(*,gripper_error,arm_error=0.):
    owner,bus=fixture()
    bridge,runtime=bridge_fixture(owner)
    runtime.pending.trace.positions[:,:,5]=5.
    present=np.full(6,9.);present[5]=5.+gripper_error;present[1]+=arm_error
    owner.io._read_state_only=lambda:present.copy()
    read=bus.sync_read
    def fixed_present(register,normalize=True,**kwargs):
        if register=='Present_Position':
            values=present if normalize else 1000+10*present
            return dict(zip(ORDER,values.tolist()))
        return read(register,normalize=normalize,**kwargs)
    bus.sync_read=fixed_present
    result=bridge.execute()
    return result,bus,present


def test_large_j6_position_residual_freezes_and_removes_closing_goal_error():
    result,bus,present=closing_case(gripper_error=5.)
    assert result['normal_follow']['goal_reached'] is False
    assert result['normal_follow']['reason']=='budget_exhausted'
    assert len(result['normal_follow']['positions'])==8
    assert result['completion_hold']['reason']=='v21_normal_completion_fallback'
    assert result['commands_position'][-1][5]==5.
    assert result['measured_q_path'][-1][5]==10.
    assert result['completion_hold']['held_position'][5]==10.
    assert bus.goal[5]==present[5]  # old closing target 5 is now the fixed present 10
    assert bus.events.count('goal_write')==6 and len(result['commands_position'])==5


def test_an_arm_follow_failure_also_freezes_j6_inside_its_position_tolerance():
    result,bus,present=closing_case(gripper_error=2.,arm_error=4.)
    assert abs(present[5]-5.)<=3.  # The gripper alone is within the unchanged tolerance.
    assert result['normal_follow']['goal_reached'] is False
    assert result['normal_follow']['final_position_error']==4.
    assert result['completion_hold']['held_position'][5]==7.
    assert bus.goal[5]==7. and bus.events.count('goal_write')==6


def test_quiet_in_tolerance_closing_residual_can_keep_the_goal_but_proves_no_contact():
    result,bus,present=closing_case(gripper_error=2.)
    assert result['normal_follow']['goal_reached'] is False
    assert result['normal_follow']['hold_eligible'] is True
    assert result['completion_hold']['reason']=='normal_completion_retained_fifth_goal'
    assert result['completion_hold']['goal_register_writes']==0
    assert bus.goal[5]==5. and present[5]==7.
    assert bus.events.count('goal_write')==5
    assert 'contact_verified' not in result and 'grasp_success' not in result
