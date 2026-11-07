"""Independent binding/failure review with the established fake hardware fixture."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from test_v21_live_service import service  # noqa: F401
from test_v21_pipeline_repair_integration import enable_completed_input_recording


def fixture_planner(model, state, proposal, dt, gate, *, trace, **kwargs):
    trace.begin(proposal, q=state.q, actor_commands=proposal)
    selected = state.q[:, None].expand(1, 5, 6).clone() + .5
    trace.record('selected', commands=selected)
    trace.finish(proposal)
    return selected, dict(action_source='v21_lewm_cem')


def test_preview_plan_and_next_plan_cannot_cross_bind_execution_trace(service, monkeypatch):
    from fcpg import v21_lewm_cem
    enable_completed_input_recording(service)
    service.replan_on_execute = False
    service.lewm_cem = True
    service.lewm_cem_objective = 'actor_response_tracking'
    monkeypatch.setattr(v21_lewm_cem, 'plan_commands', fixture_planner)
    paths = []
    for _ in range(2):
        service.handle('preview')
        preview_trace = deepcopy(service.lewm_cem_trace_record)
        plan = service.handle('plan')['plan']
        plan_trace = deepcopy(plan['cem_trace'])
        assert plan_trace['path'] != preview_trace['path']
        with pytest.raises(RuntimeError, match='pending plan'):
            service.handle('preview')
        assert service.lewm_cem_trace_record == plan_trace
        result = service.handle('execute')['last_execution']
        assert result['cem_trace'] == plan_trace
        payload = torch.load(service.output/plan_trace['path'], weights_only=False)
        selected = next(event['values']['commands'] for event in payload['events'] if event['event'] == 'selected')
        assert np.array_equal(selected[0].numpy(), result['commands_position'])
        assert payload['source_time_s'] == plan['source_time_s']
        saved_input = torch.load(service.output/result['policy_input_snapshot'], weights_only=False)
        assert saved_input['cem_trace'] == plan_trace
        paths.append(plan_trace['path'])
    assert paths[0] != paths[1]
    assert service.owner.writes == 10


def test_trace_disk_error_does_not_mask_failed_planner_error(service, monkeypatch):
    from fcpg import v21_lewm_cem
    service.record_policy_inputs = True
    service.lewm_cem_objective = 'actor_response_tracking'
    def rejected(model, state, proposal, dt, gate, *, trace, **kwargs):
        trace.begin(proposal, q=state.q)
        trace.finish(proposal, error='original candidate rejection')
        raise ValueError('original candidate rejection')
    def disk_failure(*args, **kwargs):
        raise OSError('injected audit disk failure')
    monkeypatch.setattr(v21_lewm_cem, 'plan_commands', rejected)
    monkeypatch.setattr(torch, 'save', disk_failure)
    with pytest.raises(ValueError, match='original candidate rejection'):
        service._lewm_cem_builder(service.owner.runtime.state, torch.arange(5)[None].double()*.1)
    assert not service.lewm_cem_trace_record['saved']
    assert 'injected audit disk failure' in service.lewm_cem_trace_record['error']
    assert service.owner.writes == 0


def test_successful_planner_trace_disk_failure_disarms_without_hardware_hold(service, monkeypatch):
    from fcpg import v21_lewm_cem
    service.record_policy_inputs = True
    service.replan_on_execute = False
    service.lewm_cem = True
    service.lewm_cem_objective = 'actor_response_tracking'
    monkeypatch.setattr(v21_lewm_cem, 'plan_commands', fixture_planner)
    monkeypatch.setattr(torch, 'save', lambda *a, **kw: (_ for _ in ()).throw(OSError('trace disk fault')))
    with monkeypatch.context() as guard:
        guard.setattr(service.owner, '_halt', lambda: pytest.fail('audit disk failure must not dispatch a HOLD'))
        with pytest.raises(RuntimeError, match='CEM source trace could not be persisted'):
            service.handle('plan')
    assert service.owner.writes == 0
    assert service.owner.runtime.pending is None and service.owner.runtime.stopped
    assert service.owner.stop_event.is_set()
