"""Debug execution retains the explicit CEM plan; fake IO only."""
from copy import deepcopy

import pytest
import torch

from fcpg.v21_hardware_bridge import V21TransactionBridge
from fcpg.v21_runtime import PreExecutionStateChanged
from test_v21_live_service import service  # noqa: F401


@pytest.mark.parametrize('wait_s,allowed',[(2.,True),(11.,False)])
def test_wait_does_not_trigger_hidden_capture_or_cem(service,monkeypatch,wait_s,allowed):
    service.replan_on_execute=False
    service.lewm_cem=True
    service.owner.runtime.max_plan_age_s=10.
    planned,captured,writes=[],[],[]
    def builder(state,offsets):
        planned.append(float(state.time.item()))
        commands=state.q[:,None].expand(-1,5,-1).clone()
        commands[:,:,5] += .5
        return commands
    builder.wants_command_offsets=True
    monkeypatch.setattr(service,'_lewm_cem_builder',builder)
    owner=service.owner
    capture=owner.capture
    def observe():
        captured.append(service.clock());return capture()
    def write(command,previous):
        owner.q=command.copy();writes.append(command.copy());return command.copy()
    bridge=V21TransactionBridge(owner.runtime,read_q=owner.read_checked,write_q=write,
        capture_endpoint=observe,physical_check=lambda q,u:None,halt=owner._halt,
        clock=service.clock,wait=owner.advance_clock,
        cancelled=owner.stop_event.is_set,request_stop=owner.stop_event.set)
    monkeypatch.setattr(owner,'bridge',bridge)
    service.handle('plan')
    commands=owner.runtime.pending.trace.positions.clone()
    original_plan=deepcopy(service.plan_record)
    owner.advance_clock(wait_s)
    if allowed:
        status=service.handle('execute')
        assert status['model']['execution_planning']=='explicit_plan_only'
        assert len(writes)==5
        torch.testing.assert_close(torch.tensor(writes)[None],commands,rtol=0,atol=0)
        # One planning capture and one post-execution feedback capture only.
        assert len(captured)==2
        assert bridge.last_actor_input.time.item()==original_plan['source_time_s']
    else:
        with pytest.raises(PreExecutionStateChanged):service.handle('execute')
        assert not writes and len(captured)==1
    assert len(planned)==1 and service.plan_record==original_plan
    assert owner.runtime.stopped and owner.runtime.pending is None


def test_pose_change_is_not_silently_replanned(service,monkeypatch):
    service.replan_on_execute=False
    service.lewm_cem=True
    def builder(state,offsets):return state.q[:,None].expand(-1,5,-1).clone()
    builder.wants_command_offsets=True
    monkeypatch.setattr(service,'_lewm_cem_builder',builder)
    service.handle('plan')
    def forbidden(*a,**kw):pytest.fail('execute must not silently replan')
    monkeypatch.setattr(service,'_plan_with_policy',forbidden)
    def changed():
        rt=service.owner.runtime
        rt.begin_execution(rt.pending.source_q+2.,rt.pending.source_time)
    monkeypatch.setattr(service.owner.bridge,'execute',changed)
    with pytest.raises(PreExecutionStateChanged):service.handle('execute')
    assert service.owner.writes==0 and service.owner.runtime.stopped
