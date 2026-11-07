from types import SimpleNamespace

import pytest
import torch

from fcpg.v21_hardware_bridge import V21TransactionBridge
from fcpg.v21_runtime import PreExecutionStateChanged
from test_v21_live_service import service  # noqa: F401


def wire_trial(service, monkeypatch, *, fail_first=False):
    import fcpg.v21_live_service as live
    monkeypatch.setattr(live.established, '_load_v177_home_state', lambda *a, **k:service.owner.q.copy())
    service.owner.read_checked=service.owner._read_q
    service.owner.io.settle_position_tolerance=3.
    captures=[]; reused=[]; calls=[]
    capture=service.owner.capture
    execute=service.owner.bridge.execute
    def fresh():
        captures.append(True)
        service.owner.bridge.last_plan_observation_source='fresh_capture'
        return capture()
    service.owner.capture=fresh
    def completed():
        reused.append(service.owner.runtime.state)
        service.owner.bridge.last_plan_observation_source='completed_transaction_endpoint'
        service.owner.runtime.stop()
        return service.owner.runtime.plan()
    service.owner.bridge.plan_from_completed_observation=completed
    def run():
        calls.append(True)
        if fail_first and len(calls)==1:
            service.owner.bridge.last_attempt=dict(policy_commands_attempted=0, policy_commands_returned=0)
            service.owner.runtime.stop()
            raise PreExecutionStateChanged(torch.zeros(1,6),torch.ones(1,6),.1)
        result=execute()
        result['policy_commands_returned']=5
        result['endpoint_time_s']=result['readback_times_s'][-1]
        return result
    service.owner.bridge.execute=run
    return captures,reused,calls


def test_continuous_loop_has_one_h0_capture_and_one_per_completion(service,monkeypatch):
    captures,reused,calls=wire_trial(service,monkeypatch)
    service.handle('trial');service.trial.join(timeout=30.)
    assert service.trial.status()['outcome']=='budget_exhausted'
    assert len(captures)==17  # H0 + sixteen real completion frames, no plan/execute duplicates
    assert len(reused)==15 and len(calls)==16 and service.owner.writes==80
    assert [int(s.history.valid.sum()) for s in reused]==[1,2,3]+[3]*12


def test_interactive_execute_still_captures_fresh_after_plan(service):
    captures=[];original=service.owner.capture
    def capture():captures.append(True);return original()
    service.owner.capture=capture
    service.handle('plan'); assert len(captures)==1
    service.handle('execute')
    assert len(captures)==3  # explicit plan + fresh pre-dispatch + actual completion
    assert service.owner.writes==5


def test_delayed_continuous_prepared_plan_gets_fresh_images(service,monkeypatch):
    captures=[];original=service.owner.capture
    def capture():captures.append(True);return original()
    service.owner.capture=capture
    service.handle('plan')
    monkeypatch.setattr(service.trial,'owns_current_thread',lambda:True)
    old_clock=service.clock
    service.clock=lambda:old_clock()+1.
    service.handle('execute')
    assert len(captures)==3 and service.owner.writes==5


def test_pre_dispatch_retry_in_continuous_loop_uses_a_fresh_observation(service,monkeypatch):
    captures,reused,calls=wire_trial(service,monkeypatch,fail_first=True)
    service.handle('trial');service.trial.join(timeout=30.)
    assert service.trial.status()['outcome']=='budget_exhausted'
    assert len(captures)==18 and len(calls)==17 and service.owner.writes==80


def bridge_case(age, *, completed=True):
    calls=[]
    runtime=SimpleNamespace(state=SimpleNamespace(time=torch.tensor([10.],dtype=torch.float64)),executing=False)
    runtime.stop=lambda:calls.append('stop')
    runtime.plan=lambda **kw:calls.append(('plan',kw)) or 'commands'
    bridge=V21TransactionBridge(runtime,read_q=lambda:None,write_q=lambda *a:pytest.fail('no writes'),
        capture_endpoint=lambda:calls.append('capture') or 'fresh endpoint', physical_check=lambda *a:None,
        halt=lambda:pytest.fail('no hardware hold from planning'),clock=lambda:10.+age)
    bridge.last_attempt=dict(completed=completed)
    return bridge,calls


@pytest.mark.parametrize('age,completed,expect_fresh',[(.1,True,False),(.51,True,True),(.1,False,True)])
def test_only_recent_completed_observation_is_reused(age,completed,expect_fresh):
    bridge,calls=bridge_case(age,completed=completed)
    assert bridge.plan_from_completed_observation()=='commands'
    assert ('capture' in calls)==expect_fresh
    assert calls[-1][1] == ({'endpoint':'fresh endpoint'} if expect_fresh else {})


def test_future_completion_clock_is_rejected_not_retimestamped():
    bridge,calls=bridge_case(-.1)
    with pytest.raises(RuntimeError,match='future'):
        bridge.plan_from_completed_observation()
    assert calls==['stop']


def test_stop_after_cached_plan_prevents_the_next_execute(service,monkeypatch):
    captures,reused,calls=wire_trial(service,monkeypatch)
    original=service.owner.bridge.plan_from_completed_observation
    def plan():
        result=original()
        service.trial.cancel()
        return result
    service.owner.bridge.plan_from_completed_observation=plan
    service.handle('trial');service.trial.join(timeout=15.)
    assert len(calls)==1 and service.owner.writes==5
    assert service.trial.status()['outcome']=='operator_stopped'
