from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from test_v21_live_service import service  # noqa: F401
from test_v21_hardware_bridge import setup


def test_bridge_keeps_actual_source_separate_from_completed_state():
    bridge, runtime, writes, holds = setup()
    source = SimpleNamespace(q=torch.ones(1, 6))
    runtime.state = source
    complete = runtime.complete
    def finish(*a, **kw):
        result = complete(*a, **kw)
        runtime.state = SimpleNamespace(q=torch.full((1, 6), 2.))
        return result
    runtime.complete = finish
    bridge.execute()
    assert bridge.last_source_state is source
    assert bridge.last_source_state is not runtime.state
    assert len(writes) == 5 and len(holds) == 1


def test_optional_snapshot_uses_actual_source_and_adds_no_commands(service):
    service.record_policy_inputs = True
    original = service.owner.execute
    def execute():
        source = service.owner.runtime._pending_state
        offsets = service.owner.runtime.pending.trace.offsets.cpu().tolist()
        result = original()
        service.owner.bridge.last_source_state = source
        service.owner.bridge.last_actor_input = service.owner.runtime.last_actor_input
        result['protective_hold_after_q5'] = True
        result['commanded_offsets_s'] = offsets
        return result
    service.owner.bridge.execute = execute
    service.handle('plan')
    result = service.handle('execute')['last_execution']
    payload = torch.load(service.output/result['policy_input_snapshot'], weights_only=False)
    assert not payload['executable']
    assert payload['kind'] == 'actual-actor-input' and payload['schema'] == 2
    assert payload['state'].q.device.type == 'cpu'
    assert payload['state'].time[0] == result['source_time_s']
    assert not payload['state'].history.valid.any()
    assert service.owner.runtime.state.history.valid.any()
    assert service.owner.writes == 5
    assert payload['commands_position'] == result['commands_position']
    copy = deepcopy(payload['state'].q)
    with torch.inference_mode():
        service.owner.bridge.last_source_state.q.add_(1)
    torch.testing.assert_close(payload['state'].q, copy)


def test_snapshot_rejects_unfinished_transaction_without_hardware_write(service):
    service.record_policy_inputs = True
    service.owner.bridge.last_actor_input = service.owner.runtime.state
    with pytest.raises(RuntimeError, match='completed and held'):
        service._save_execution(dict(completed=False,action_source='v21_actor'))
    assert service.owner.writes == 0
    assert not list(service.output.glob('policy-input-*.pt'))


def test_snapshot_records_execution_geometry_action_source(service):
    """Capped nocolor trials plan as v21_actor_execution_geometry; logging must still fire."""
    service.record_policy_inputs = True
    source = service.owner.runtime.state
    service.owner.bridge.last_actor_input = source
    result = dict(
        completed=True, protective_hold_after_q5=True, action_source='v21_actor_execution_geometry',
        source_time_s=float(source.time[0]), source_q_position=source.q[0].detach().cpu().tolist(),
        commands_position=[[0.]*6]*5, commanded_offsets_s=[0.]*5, measured_q_path=[[0.]*6]*6,
    )
    service._save_execution(result)
    assert result['policy_input_snapshot'].startswith('policy-input-')
    payload = torch.load(service.output / result['policy_input_snapshot'], weights_only=False)
    assert payload['action_source'] == 'v21_actor_execution_geometry'
    assert payload['executable'] is False
    assert service.owner.writes == 0
