from types import SimpleNamespace
import torch

from fcpg.prod_v20_policy import empty_history
from fcpg.v21_alignment_session import retain_unchanged_arm_goals


def case():
    q=torch.tensor([[0.,10.,20.,30.,-50.,31.]])
    h=empty_history(1);h.valid[0,-1]=True
    h.commands_position[0,-1]=q[:,None].repeat(1,5,1)[0]
    h.commands_position[0,-1,:,1]=9.
    u=q[:,None].repeat(1,5,1);u[0,:,0]=torch.linspace(-.4,-2.,5)
    return SimpleNamespace(q=q,history=h),u


def test_unchanged_arm_axes_keep_the_last_real_goal_not_biased_present():
    state,u=case();old=u.clone()
    result,kept=retain_unchanged_arm_goals(state,u)
    assert result[0,:,1].tolist()==[9.]*5 and 2 in kept
    torch.testing.assert_close(result[0,:,[0,4,5]],old[0,:,[0,4,5]])
    torch.testing.assert_close(u,old)


def test_explicit_axis_motion_and_out_of_range_old_goal_are_not_overridden():
    state,u=case();u[0,:,1]+=1.
    state.history.commands_position[0,-1,:,2]=10.
    result,kept=retain_unchanged_arm_goals(state,u)
    torch.testing.assert_close(result,u)
    assert 2 not in kept and 3 not in kept


def test_h0_never_invents_a_previous_setpoint():
    state,u=case();state.history.valid[:]=False
    result,kept=retain_unchanged_arm_goals(state,u)
    assert result is u and kept==[]
