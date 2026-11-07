"""Shadow logging uses frozen sources and never owns a motor command."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_v21_live_service import service  # noqa: F401
from scripts import serve_so101_v21_research as serve


def test_plan_prediction_is_persisted_before_any_write(service):
    commands=service.owner.bridge.plan()
    source=service.owner.runtime._pending_state
    forecast=service.owner.runtime.pending
    def plan(actual_source,actual_forecast):
        assert actual_source is source and actual_forecast is forecast
        assert service.owner.writes==0
        return {'input_sha256':'test-input','control_influence':False}
    service.lewm_shadow=SimpleNamespace(plan=plan)
    service._record_plan(commands)
    receipt=service.lewm_shadow_plan
    path=Path(receipt['path'])
    assert path.is_absolute() and path.exists()
    assert json.loads(path.read_text())==receipt['record']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256']
    assert receipt['persisted_at_s']<=service.clock()
    assert service.owner.writes==0 and service.owner.runtime.pending is forecast


def test_complete_uses_saved_dispatch_source_and_keeps_failed_blocks(service):
    dispatch=object();forecast=object()
    service.owner.bridge.last_source_state=dispatch
    service.owner.bridge.last_forecast=forecast
    attempt={'completed':False,'policy_commands_returned':2,'error':'physical guard'}
    service.owner.bridge.last_attempt=attempt
    calls=[]
    def complete(source,plan,result,receipt):
        assert source is dispatch and plan is forecast and result is attempt
        calls.append(True)
        return {'comparison_valid':False,'reason':'partial execution'}
    service.lewm_shadow=SimpleNamespace(complete=complete)
    service._save_lewm_shadow_attempt()
    service._save_lewm_shadow_attempt()
    assert calls==[True] and service.owner.writes==0
    saved=json.loads((service.output/attempt['lewm_shadow_evaluation']['path']).read_text())
    assert saved['comparison_valid'] is False
    assert attempt['error']=='physical guard' and not attempt['completed']


def test_shadow_failure_preserves_primary_physical_error(service):
    service.owner.bridge.last_source_state=object()
    service.owner.bridge.last_forecast=object()
    attempt={'error':'physical stop','completed':False}
    service.owner.bridge.last_attempt=attempt
    def failed(*_args):raise ValueError('invalid prediction')
    service.lewm_shadow=SimpleNamespace(complete=failed)
    service._save_lewm_shadow_attempt()
    assert attempt['error']=='physical stop'
    assert attempt['lewm_shadow_evaluation']['error']=='ValueError: invalid prediction'
    assert service.owner.writes==0


def test_failed_response_holds_before_shadow_evaluation(service,monkeypatch):
    service.lewm_cem=True
    service.lewm_cem_objective='actor_command_tracking'
    monkeypatch.setattr(service,'_lewm_cem_builder',
        lambda state,offsets=None:state.q[:,None].expand(-1,5,-1).clone()+.1)
    original=service.owner.bridge.execute
    def execute():
        forecast=service.owner.runtime.pending
        source=service.owner.runtime._pending_state
        result=original()
        service.owner.bridge.last_forecast=forecast
        service.owner.bridge.last_source_state=source
        return result
    service.owner.bridge.execute=execute
    def reject(*_args,**_kwargs):raise ValueError('response did not follow')
    monkeypatch.setattr('fcpg.v21_live_service.require_actual_cem_response',reject)
    holds_before=service.owner.holds
    checked=[]
    def complete(source,forecast,attempt,receipt):
        assert service.owner.holds>holds_before
        assert service.owner.stop_event.is_set()
        checked.append(True)
        return {'scored':False,'reason':'test response failure preserved'}
    service.lewm_shadow=SimpleNamespace(plan=lambda *_args:{},complete=complete)
    service.handle('plan')
    with pytest.raises(ValueError,match='response did not follow'):service.handle('execute')
    assert checked==[True]
    assert service.owner.writes==5


@pytest.mark.parametrize('argv',[
    ['--lewm-shadow-checkpoint','candidate.pt'],
    ['--lewm-shadow-parent-sha256','a'*64],
])
def test_incomplete_shadow_configuration_fails_before_server(argv,monkeypatch):
    def forbidden(*_args,**_kwargs):raise AssertionError('server must not open')
    monkeypatch.setattr(serve,'make_server',forbidden)
    with pytest.raises(SystemExit) as exc:serve.main(argv)
    assert exc.value.code==2
