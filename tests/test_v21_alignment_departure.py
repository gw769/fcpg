"""One setup block uses existing Actor arm commands and the shared auxiliary budget."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_v21_live_service import service  # noqa: F401


@pytest.fixture
def departure(service,monkeypatch):
    import fcpg.v21_live_service as live
    service.alignment_motion_enabled=True
    service.owner.q=np.array([0.,-95.,96.,62.,-51.,31.],np.float32)
    service.owner.read_checked=service.owner._read_q
    service.owner.io.settle_position_tolerance=3.
    monkeypatch.setattr(live.established,'_load_v177_home_state',lambda *a,**k:service.owner.q.copy())
    seen=[]
    def actor(state,text,offsets):
        expected=torch.arange(5,dtype=torch.float64)[None]*float(torch.tensor(.1))
        torch.testing.assert_close(offsets,expected,rtol=0,atol=0)
        commands=state.q[:,None].repeat(1,5,1).clone()
        commands[0,:,1]+=torch.linspace(.2,1.,5)
        commands[0,:,5]=5.
        seen.append(commands)
        return SimpleNamespace(commands_position=commands)
    monkeypatch.setattr(service.owner.runtime.core,'act',actor)
    return service,seen


def test_departure_is_one_counted_open_jaw_block_and_retains_actor_arm_values(departure):
    service,seen=departure
    result=service.handle('alignment-departure')
    actual=result['last_execution']
    commands=np.asarray(actual['commands_position'])
    original=np.asarray(actual['view_acquisition']['original_actor_commands'])[0]
    np.testing.assert_allclose(commands[:,:5],original[:,:5])
    np.testing.assert_array_equal(commands[:,5],np.full(5,31.))
    assert torch.all(seen[-1][:,:,5]==5.)
    assert actual['action_source']=='visual_alignment_open_departure'
    assert actual['view_acquisition']['intervention']=='hold_existing_gripper_open_only'
    assert result['auxiliary_trial']['block_attempts']==1
    assert result['auxiliary_trial']['policy_commands_returned']==5
    assert result['runtime']['stopped'] and not result['runtime']['pending']
    assert service.owner.writes==5 and service._carry is None


def test_departure_cannot_be_repeated_to_bypass_auxiliary_budget(departure):
    service,_=departure
    service.handle('alignment-departure');service.handle('alignment-departure')
    with pytest.raises(RuntimeError,match='first two blocks after HOME'):service.handle('alignment-departure')
    assert service.owner.writes==10 and service.auxiliary_budget.block_attempts==2


@pytest.mark.parametrize('failure',['closed','unfolded','disabled','legacy'])
def test_invalid_setup_never_writes_motors(departure,failure):
    service,_=departure
    if failure=='closed':service.owner.q[5]=5.
    elif failure=='unfolded':service.owner.q[1]=-40.
    elif failure=='disabled':service.alignment_motion_enabled=False
    else:service.grounding_correction=True
    with pytest.raises(RuntimeError):service.handle('alignment-departure')
    assert service.owner.writes==0 and service.owner.runtime.stopped


def test_nonprogressing_actor_is_not_replaced_by_a_handwritten_arm_pose(departure,monkeypatch):
    service,_=departure
    monkeypatch.setattr(service.owner.runtime.core,'act',lambda s,*a:SimpleNamespace(
        commands_position=s.q[:,None].repeat(1,5,1)))
    with pytest.raises(RuntimeError,match='does not unfold'):service.handle('alignment-departure')
    assert service.owner.writes==0 and service.auxiliary_budget.returned==0
