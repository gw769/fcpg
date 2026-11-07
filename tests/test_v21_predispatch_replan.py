"""Fresh replanning may retry only a proved pre-dispatch state change."""
import pytest
import torch

from fcpg.v21_runtime import PreExecutionStateChanged
from test_v21_live_service import service  # noqa: F401 -- fake IO only


def changed():
    return PreExecutionStateChanged(torch.zeros(1,6),torch.full((1,6),.07),.02)


def test_new_state_gets_one_fresh_plan_and_only_five_writes(service):
    service.handle('plan');original=service.owner.bridge.execute;calls=[]
    def execute():
        calls.append(True)
        if len(calls)==1:
            service.owner.bridge.last_attempt=dict(policy_commands_attempted=0,policy_commands_returned=0)
            service.owner.runtime.stop()
            raise changed()
        return original()
    service.owner.bridge.execute=execute
    result=service.handle('execute')
    assert len(calls)==2 and service.owner.writes==5
    assert len(result['last_execution']['pre_dispatch_replans'])==1
    assert result['last_execution']['pre_dispatch_replans'][0]['no_policy_commands_sent']


def test_replanning_is_bounded_without_any_write(service):
    service.handle('plan');calls=[]
    def execute():
        calls.append(True)
        service.owner.bridge.last_attempt=dict(policy_commands_attempted=0,policy_commands_returned=0)
        service.owner.runtime.stop();raise changed()
    service.owner.bridge.execute=execute
    with pytest.raises(PreExecutionStateChanged):service.handle('execute')
    assert len(calls)==3 and service.owner.writes==0 and service.owner.runtime.stopped


@pytest.mark.parametrize('reason',['stop','write_attempted','unknown_write_count'])
def test_stop_or_possible_dispatch_is_never_retried(service,reason):
    service.handle('plan');calls=[]
    def execute():
        calls.append(True)
        if reason=='stop':service.owner.stop_event.set()
        service.owner.bridge.last_attempt=({} if reason=='unknown_write_count' else
            dict(policy_commands_attempted=1 if reason=='write_attempted' else 0,policy_commands_returned=0))
        service.owner.runtime.stop();raise changed()
    service.owner.bridge.execute=execute
    with pytest.raises(PreExecutionStateChanged):service.handle('execute')
    assert len(calls)==1 and service.owner.writes==0
