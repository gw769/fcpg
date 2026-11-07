"""Small real readback changes must recondition the model, not pass old commands."""
import pytest
import torch

from fcpg.v21_runtime import PreExecutionStateChanged
from test_v21_planning_memory import runtime, endpoint  # noqa: F401


def test_quiet_drift_rereads_memory_recomputes_actor_and_rechecks_physics(runtime,monkeypatch):
    value,now=runtime
    value.configure_source_update_tolerance(.25)
    now[0]=10.1;image=endpoint(now[0],4);value.plan(endpoint=image)
    candidate=value._pending_state;old_memory=candidate.memory
    image_times=old_memory.image_times.clone();image_q=old_memory.q_at_image.clone()
    calls=[];checked=[];original=value.core.act
    def act(state,*a,**k):
        result=original(state,*a,**k)
        calls.append((state.q.clone(),result.commands_position.clone()))
        return result
    monkeypatch.setattr(value.core,'act',act)
    value.physical_check=lambda q,u:checked.append((q.detach().clone(),u.detach().clone()))
    actual=image.q+.247
    now[0]=10.12
    commands=value.begin_execution(actual,torch.tensor([10.12],dtype=torch.float64))
    assert len(calls)==1 and len(checked)==1
    torch.testing.assert_close(calls[0][0],actual)
    torch.testing.assert_close(commands,calls[0][1])
    torch.testing.assert_close(checked[0][1],commands)
    assert value.state.memory is old_memory
    torch.testing.assert_close(old_memory.image_times,image_times)
    torch.testing.assert_close(old_memory.q_at_image,image_q)
    torch.testing.assert_close(value.pending.source_q,actual)
    assert not value.state.history.valid.any()
    assert value.last_dispatch_update['commands_recomputed']


@pytest.mark.parametrize('mode',['large','assisted','expired'])
def test_large_drift_assisted_plan_and_expired_plan_still_reject(runtime,mode):
    value,now=runtime;value.configure_source_update_tolerance(.25)
    now[0]=10.1;image=endpoint(now[0],4);value.plan(endpoint=image)
    q=image.q+(.3 if mode=='large' else .07)
    if mode=='assisted':value.pending_action_source='v21_actor_grounding_corrected'
    stamp=311. if mode=='expired' else 10.12
    now[0]=stamp
    with pytest.raises(PreExecutionStateChanged):value.begin_execution(q,torch.tensor([stamp],dtype=torch.float64))
    assert value.stopped and not value.executing


def test_new_commands_cannot_bypass_physical_rejection(runtime):
    value,now=runtime;value.configure_source_update_tolerance(.25)
    now[0]=10.1;image=endpoint(now[0],4);value.plan(endpoint=image)
    def reject(q,u):raise RuntimeError('unchanged mechanical protection')
    value.physical_check=reject
    now[0]=10.12
    with pytest.raises(RuntimeError,match='mechanical protection'):
        value.begin_execution(image.q+.07,torch.tensor([10.12],dtype=torch.float64))
    assert value.stopped and not value.executing


def test_recorded_actor_input_keeps_planning_clock_when_q_did_not_change(runtime):
    value,now=runtime
    now[0]=10.1;image=endpoint(now[0],4);value.plan(endpoint=image)
    actor_input=value._pending_state
    now[0]=10.12
    value.begin_execution(image.q,torch.tensor([10.12],dtype=torch.float64))
    assert value.last_actor_input is actor_input
    assert value.last_actor_input.time.item()==pytest.approx(10.1)
    assert value.state.time.item()==pytest.approx(10.12)


@pytest.mark.parametrize('bad',[3.,float('nan'),-.1])
def test_position_tolerance_cannot_be_miswired_as_quiet_tolerance(runtime,bad):
    value,_=runtime
    with pytest.raises(ValueError):value.configure_source_update_tolerance(bad)


def _geometry_builder(calls):
    def builder(state,offsets):
        calls.append(state.q.clone())
        return state.q[:,None,:].expand(1,5,6).clone()+torch.arange(1,6.)[None,:,None]*.01
    builder.wants_command_offsets=True
    return builder


@pytest.mark.parametrize('mode',['large','expired','geometry_tick'])
def test_always_mode_recomputes_instead_of_aborting(runtime,mode):
    """'always': drift beyond the quiet allowance or an aged plan is re-derived, never abandoned."""
    value,now=runtime;value.configure_source_update_tolerance(.25)
    value.configure_pre_execution_recompute('always')
    now[0]=10.1;image=endpoint(now[0],4);calls=[];checked=[]
    if mode=='geometry_tick':
        value.plan(endpoint=image,command_builder=_geometry_builder(calls),action_source='v21_actor_execution_geometry')
        q=image.q+.09;stamp=10.12
    else:
        value.plan(endpoint=image)
        q=image.q+(.9 if mode=='large' else .07);stamp=311. if mode=='expired' else 10.12
    value.physical_check=lambda a,u:checked.append((a.detach().clone(),u.detach().clone()))
    now[0]=stamp
    commands=value.begin_execution(q,torch.tensor([stamp],dtype=torch.float64))
    assert value.executing and not value.stopped
    torch.testing.assert_close(value.pending.source_q,q)
    torch.testing.assert_close(checked[-1][0],q)
    torch.testing.assert_close(checked[-1][1],commands)
    update=value.last_dispatch_update
    assert update['commands_recomputed'] and update['images_reused_without_retimestamp']
    assert update['expired_plan_recomputed']==(mode=='expired')
    if mode=='geometry_tick':
        # The stored builder, not the raw Actor, produced the dispatched commands.
        assert len(calls)==2 and torch.allclose(calls[-1],q)
        torch.testing.assert_close(commands,_geometry_builder([])(value.state,None))
        assert not value.recomputing_at_dispatch


def test_always_mode_keeps_strict_matching_for_assisted_sources(runtime):
    value,now=runtime;value.configure_source_update_tolerance(.25)
    value.configure_pre_execution_recompute('always')
    now[0]=10.1;image=endpoint(now[0],4);value.plan(endpoint=image)
    value.pending_action_source='v21_actor_grounding_corrected'
    now[0]=10.12
    with pytest.raises(PreExecutionStateChanged):
        value.begin_execution(image.q+.07,torch.tensor([10.12],dtype=torch.float64))
    assert value.stopped


def test_rebase_before_epoch_reuses_images_and_recomputes_with_builder(runtime):
    value,now=runtime;value.configure_pre_execution_recompute('always')
    now[0]=10.1;image=endpoint(now[0],4);calls=[]
    value.plan(endpoint=image,command_builder=_geometry_builder(calls),action_source='v21_actor_execution_geometry')
    memory=value._pending_state.memory
    moved=image.q+1.5;now[0]=10.4
    commands=value.rebase_pending(moved,torch.tensor([10.4],dtype=torch.float64))
    assert value.pending is not None and not value.stopped and not value.executing
    torch.testing.assert_close(value.pending.source_q,moved)
    assert value._pending_state.memory is memory
    assert len(calls)==2 and torch.allclose(calls[-1],moved)
    assert value.last_rebase['maximum_pose_change']==pytest.approx(1.5)
    assert value.pending_matches(moved) and not value.pending_matches(image.q)
    # A matching epoch readback then dispatches without a second recompute.
    now[0]=10.41
    dispatched=value.begin_execution(moved,torch.tensor([10.41],dtype=torch.float64))
    torch.testing.assert_close(dispatched,commands)
    assert value.last_dispatch_update is None and len(calls)==2


def test_rebase_is_refused_in_bounded_mode(runtime):
    value,now=runtime
    now[0]=10.1;image=endpoint(now[0],4);value.plan(endpoint=image)
    with pytest.raises(RuntimeError,match='not recomputable'):
        value.rebase_pending(image.q+.5,torch.tensor([10.2],dtype=torch.float64))


def test_bridge_rebases_before_epoch_and_never_writes_during_rebase(runtime):
    from fcpg.v21_hardware_bridge import V21TransactionBridge
    value,now=runtime;value.configure_pre_execution_recompute('always')
    now[0]=10.1;image=endpoint(now[0],4);calls=[]
    value.plan(endpoint=image,command_builder=_geometry_builder(calls),action_source='v21_actor_execution_geometry')
    actual=[image.q[0].numpy()+.6];writes=[]
    def read_q():return actual[0].copy()
    def write_q(command,previous):writes.append(command);return command
    bridge=V21TransactionBridge(value,read_q=read_q,write_q=write_q,capture_endpoint=lambda:endpoint(now[0],5),
        physical_check=lambda q,u:None,halt=lambda:{},clock=lambda:now[0],wait=lambda s:now.__setitem__(0,now[0]+s))
    rebases=bridge._rebase_before_epoch(actual[0].copy())
    assert len(rebases)==1 and rebases[0]['no_policy_commands_sent'] and writes==[]
    assert value.pending_matches(torch.from_numpy(actual[0])[None]) and not value.executing
