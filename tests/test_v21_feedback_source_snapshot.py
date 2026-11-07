from dataclasses import fields
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.prod_v20_policy import ActualHistory
from fcpg.v21_lewm_shadow import _packet, _json_sha
from fcpg.v21_models import CommandTrace
from fcpg.v21_candidate_preview import snapshot_feedback_source, preview_actor_candidates


@pytest.fixture
def service(tmp_path):
    history = ActualHistory(
        torch.arange(90, dtype=torch.float32).reshape(1, 3, 5, 6)/10,
        torch.arange(108, dtype=torch.float32).reshape(1, 3, 6, 6)/20,
        torch.arange(2304, dtype=torch.float32).reshape(1, 3, 2, 384)/7,
        torch.full((1, 3, 5), .1), torch.tensor([[False, False, True]]))
    state = SimpleNamespace(
        q=torch.ones(1, 6), time=torch.tensor([10.], dtype=torch.float64), history=history,
        history_ends=torch.tensor([[0., 0., 9.5]], dtype=torch.float64),
        q_error=torch.arange(30, dtype=torch.float32).reshape(1, 5, 6)/13,
        feedback_available=torch.tensor([True]), feedback_time=torch.tensor([9.5], dtype=torch.float64),
        feedback_offsets=torch.arange(1, 6, dtype=torch.float64)[None]*.1,
        feedback_retimed=torch.tensor([True]))
    endpoint = SimpleNamespace(q=torch.full((1, 6), 7.), time=torch.tensor([11.], dtype=torch.float64))
    calls = []
    def forbidden(*_args, **_kwargs): raise AssertionError('runtime/model/action call forbidden')
    runtime = SimpleNamespace(state=state, stopped=True, pending=None, executing=False,
                              observe=forbidden, stop=forbidden, plan=forbidden, preview=forbidden)
    def capture():
        calls.append('capture')
        return endpoint
    owner = SimpleNamespace(runtime=runtime, capture=capture, write_q=forbidden,
                            bridge=SimpleNamespace(execute=forbidden, stop=forbidden, physical_check=forbidden))
    return SimpleNamespace(owner=owner, output=tmp_path/'out', endpoint=endpoint, calls=calls,
        _refresh_status=lambda: calls.append('refresh'), _publish=lambda: calls.append('publish'),
        fingertips={}, targets={}, grasp_target='body', sequence=3)


def read_snapshot(reference):
    path = Path(reference['path'])
    assert path.is_absolute()
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == reference['sha256']
    return json.loads(raw)


def test_snapshot_uses_fresh_q_and_preserves_every_committed_history_tensor(service):
    state = service.owner.runtime.state
    before = {field.name: getattr(state.history, field.name).clone() for field in fields(ActualHistory)}
    reference = snapshot_feedback_source(service, service.endpoint)
    saved = read_snapshot(reference)
    assert service.calls == []
    assert service.owner.runtime.state is state and service.owner.runtime.stopped
    assert service.owner.runtime.pending is None and not service.owner.runtime.executing
    assert saved['fresh_endpoint']['q']['values'] == [[7.]*6]
    assert saved['committed_state']['q']['values'] == [[1.]*6]
    assert saved['fresh_endpoint']['time']['values'] == [11.]
    assert saved['committed_state']['time']['values'] == [10.]
    assert saved['executable'] is False and saved['model_state_updated'] is False
    assert saved['source_semantics'] == 'committed actual history + fresh measured q, not full current ActualState'
    for field in fields(ActualHistory):
        original = getattr(state.history, field.name)
        assert torch.equal(original, before[field.name])
        assert saved['history'][field.name] == dict(dtype=str(original.dtype), shape=list(original.shape),
                                                   values=original.tolist())
    assert torch.count_nonzero(torch.tensor(saved['history']['scene_pooled']['values'])) > 2000
    assert saved['history_ends']['values'] == state.history_ends.tolist()
    assert saved['actor_feedback']['q_error']['values'] == state.q_error.tolist()
    assert saved['actor_feedback']['feedback_available']['values'] == [True]


def test_history_hash_exactly_matches_shadow_packet_format(service):
    state, endpoint = service.owner.runtime.state, service.endpoint
    trace = CommandTrace.policy(endpoint.q[:, None].expand(-1, 5, -1), torch.full((1, 5), .1))
    packet = _packet(endpoint.q, state.history, trace, torch.arange(1, 6, dtype=torch.float64)[None]*.1)
    reference = snapshot_feedback_source(service, endpoint)
    saved = read_snapshot(reference)
    assert saved['history'] == packet['history']
    assert saved['history_sha256'] == reference['history_sha256'] == _json_sha(packet['history'])


@pytest.mark.parametrize('field,value', [('stopped', False), ('executing', True), ('pending', object())])
def test_stop_and_no_pending_are_required_before_capture_or_snapshot(service, field, value):
    setattr(service.owner.runtime, field, value)
    with pytest.raises(RuntimeError, match='STOP'):
        preview_actor_candidates(service)
    with pytest.raises(RuntimeError, match='STOP'):
        snapshot_feedback_source(service, service.endpoint)
    assert service.calls == [] and not service.output.exists()


def test_invisible_goal_still_exports_without_model_or_actuator_calls(service):
    state = service.owner.runtime.state
    result = preview_actor_candidates(service)
    assert not result['available'] and result['selected'] is None
    assert not result['action_dispatched'] and not result['executable']
    assert 'unavailable' in result['reason']
    assert read_snapshot(result['feedback_source_snapshot'])['fresh_endpoint']['q']['values'] == [[7.]*6]
    assert service.calls == ['capture', 'refresh', 'publish']
    assert service.owner.runtime.state is state


def make_visible(service):
    service.fingertips = {'column': dict(available=True, q_at_image=[7.]*6, fm_px=[[0., 0.], [1., 0.]],
                                       m_px=[.5, 0.], image_time_s=11., source='observed')}
    service.targets = {'targets': {'pickup': {'column': dict(visible=True, cog_px=[2., 2.],
        image_time_s=11., source='operator_colour_body', axis_px=[1., 0.])}}}


def test_head_target_unavailable_branch_also_contains_snapshot(service):
    make_visible(service)
    service.grasp_target = 'head'
    result = preview_actor_candidates(service)
    assert not result['available'] and 'head target' in result['reason']
    read_snapshot(result['feedback_source_snapshot'])


def test_existing_preview_error_preserves_the_feedback_export(service):
    make_visible(service)
    def failed(*_args, **_kwargs): raise ValueError('existing projection unavailable')
    service.owner.runtime.preview = failed
    result = preview_actor_candidates(service)
    assert not result['available'] and 'existing projection unavailable' in result['reason']
    read_snapshot(result['feedback_source_snapshot'])


def test_successful_preview_keeps_existing_candidate_values_and_attaches_snapshot(service, monkeypatch):
    from fcpg import v21_candidate_preview as module
    make_visible(service)
    state = service.owner.runtime.state
    commands = torch.arange(30, dtype=torch.float32).reshape(1, 5, 6)
    service.owner.runtime.text_ids = torch.ones(1, 1, dtype=torch.long)
    service.owner.runtime.core = SimpleNamespace(act=lambda *_args: SimpleNamespace(commands_position=commands))
    service.owner.kinematics = SimpleNamespace(forward_anchor_frames=lambda _q: (None, None),
                                             forward_anchors=lambda _q: np.zeros((1, 3)))
    service._column_fit = None
    service._column_tip_px = lambda _q: np.array([.5, 0.])
    service._column_grasp_target_xyz = lambda: np.zeros(3)
    monkeypatch.setattr(module, 'predict_projection', lambda *_args: np.array([[[0., 0.], [1., 0.], [2., 0.]]]))
    def evaluate(q, actor_commands, goal, **_kwargs):
        assert np.array_equal(actor_commands, commands[0].numpy())
        return dict(selected=None, candidates=[dict(name='unchanged', commands_position=actor_commands.tolist())],
                    executable=False, action_dispatched=False)
    monkeypatch.setattr(module, 'evaluate_candidates', evaluate)
    def preview(_endpoint, *, command_builder):
        command_builder(state, torch.arange(5, dtype=torch.float64)[None]*.1)
    service.owner.runtime.preview = preview
    result = preview_actor_candidates(service)
    assert result['available']
    assert result['candidates'][0]['commands_position'] == commands[0].tolist()
    read_snapshot(result['feedback_source_snapshot'])
    assert service.owner.runtime.state is state


def test_empty_existing_preview_result_returns_a_snapshot_reference(service):
    make_visible(service)
    service.owner.runtime.preview = lambda *_args, **_kwargs: None
    result = preview_actor_candidates(service)
    assert not result['available'] and result['reason'] == 'candidate preview returned no evaluation'
    read_snapshot(result['feedback_source_snapshot'])


def test_older_endpoint_is_not_misrepresented_as_a_fresh_source(service):
    service.endpoint.time = torch.tensor([9.], dtype=torch.float64)
    with pytest.raises(ValueError, match='must not precede'):
        snapshot_feedback_source(service, service.endpoint)
    assert not service.output.exists()
