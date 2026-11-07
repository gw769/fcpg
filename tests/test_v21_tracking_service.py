"""Explicit response-tracking integration; all IO is the existing CPU fake."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from scripts import serve_so101_v21_research as serve
from test_v21_live_service import service  # noqa: F401


def _forbidden(*_args, **_kwargs):
    raise AssertionError('response tracking must not enter a visual or hardware path')


class _UnreadableVisualState:
    def __bool__(self):
        _forbidden()

    def get(self, *_args, **_kwargs):
        _forbidden()


def test_tracking_builder_needs_no_column_transport_or_visual_target(service, monkeypatch):
    service.lewm_cem = True
    service.lewm_cem_objective = 'actor_response_tracking'
    state = service.owner.runtime.state
    offsets = torch.arange(5, dtype=torch.float64)[None] * .1
    proposal = state.q[:, None].expand(-1, 5, -1).clone() + .75
    selected = proposal - .25
    actor = Mock(return_value=SimpleNamespace(commands_position=proposal))
    planner = Mock(return_value=(selected, dict(objective='actor_response_tracking')))
    monkeypatch.setattr(service.owner.runtime.core, 'act', actor)
    monkeypatch.setattr('fcpg.v21_lewm_cem.plan_commands', planner)
    monkeypatch.setattr('fcpg.v21_live_service.bind_column_dino_fg', _forbidden)
    monkeypatch.setattr('fcpg.v21_visual_task_cost.bind_visual_task_cost', _forbidden)
    monkeypatch.setattr(service, 'fingertips', _UnreadableVisualState())
    monkeypatch.setattr(service, 'targets', _UnreadableVisualState())
    monkeypatch.setattr(service, 'column_dino_fg_response', _UnreadableVisualState())

    actual = service._lewm_cem_builder(state, offsets)

    actor.assert_called_once_with(state, service.owner.runtime.text_ids, offsets)
    planner.assert_called_once()
    args, kwargs = planner.call_args
    assert args[0] is service.owner.runtime.core.lewm
    assert args[1] is state and args[2] is proposal
    assert torch.equal(args[3], torch.full((1, 5), .1, dtype=state.q.dtype))
    assert args[4] is service.owner.runtime.physical_check
    assert kwargs['objective'] == 'actor_response_tracking'
    assert not kwargs.get('require_candidate_sensitive_visual', False)
    assert all(kwargs.get(key) is None for key in (
        'project', 'target_px', 'observed_midpoint_px', 'capture_q'))
    assert actual is selected
    assert service.lewm_cem_record['visual_goal'] is False
    assert service.lewm_cem_record['column_dino_fg_transport_installed'] is False
    assert service.lewm_cem_record['visual_task_cost'] == dict(
        available=False, reason='measured_six_joint_response_unavailable')
    assert service.owner.runtime.pending is None and service.owner.writes == 0


@pytest.mark.parametrize('condition', ['none', 'five_columns', 'nonfinite_fit', 'no_token', 'no_memory',
                                      'unreadable_visuals', 'nested_unreadable_visuals', 'empty_visuals'])
def test_optional_visual_preconditions_keep_tracking_lazy(service, monkeypatch, condition):
    from dataclasses import replace
    import numpy as np
    service.lewm_cem = True
    service.lewm_cem_objective = 'actor_response_tracking'
    state = service.owner.runtime.state
    state = replace(state, object_relative=torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0.]]),
                    memory=replace(state.memory, valid=torch.ones_like(state.memory.valid)))
    response = dict(jacobian_px_per_joint=np.ones((2, 6)), lower_q=np.full(5, -200.),
        upper_q=np.full(5, 200.), residual_rms_px=.1, design_condition=2.,
        lower_q6=0., upper_q6=100., sample_count=10)
    if condition == 'none': response = None
    if condition == 'five_columns': response['jacobian_px_per_joint'] = np.ones((2, 5))
    if condition == 'nonfinite_fit': response['jacobian_px_per_joint'][0, 0] = np.nan
    if condition == 'no_token': state = replace(state, object_relative=None)
    if condition == 'no_memory': state = replace(state, memory=None)
    proposal = state.q[:, None].expand(-1, 5, -1).clone()
    planner = Mock(return_value=(proposal, dict(objective='actor_response_tracking')))
    monkeypatch.setattr(service.owner.runtime.core, 'act', Mock(return_value=SimpleNamespace(commands_position=proposal)))
    monkeypatch.setattr('fcpg.v21_lewm_cem.plan_commands', planner)
    monkeypatch.setattr('fcpg.v21_visual_task_cost.bind_visual_task_cost', _forbidden)
    monkeypatch.setattr(service, 'column_dino_fg_response', response)
    monkeypatch.setattr(service, 'fingertips', _UnreadableVisualState())
    monkeypatch.setattr(service, 'targets', _UnreadableVisualState())
    if condition == 'nested_unreadable_visuals':
        monkeypatch.setattr(service, 'fingertips', {'column': _UnreadableVisualState()})
        monkeypatch.setattr(service, 'targets', {'targets': _UnreadableVisualState()})
    if condition == 'empty_visuals':
        monkeypatch.setattr(service, 'fingertips', {})
        monkeypatch.setattr(service, 'targets', {})
    actual = service._lewm_cem_builder(state, torch.arange(5, dtype=torch.float64)[None]*.1)
    assert actual is proposal
    assert planner.call_args.kwargs['visual_task_cost'] is None
    assert service.lewm_cem_record['visual_task_cost']['available'] is False
    assert service.lewm_cem_record['visual_task_cost']['reason']
    assert service.owner.writes == 0


def test_valid_six_joint_preconditions_can_reach_optional_binding(service, monkeypatch):
    from dataclasses import replace
    import numpy as np
    service.lewm_cem = True
    service.lewm_cem_objective = 'actor_response_tracking'
    state = service.owner.runtime.state
    state = replace(state, object_relative=torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0.]]),
                    memory=replace(state.memory, valid=torch.ones_like(state.memory.valid)))
    response = dict(jacobian_px_per_joint=np.ones((2, 6)), lower_q=np.full(5, -200.),
        upper_q=np.full(5, 200.), residual_rms_px=.1, design_condition=2.,
        lower_q6=0., upper_q6=100., sample_count=10)
    proposal = state.q[:, None].expand(-1, 5, -1).clone()
    bound = object()
    binder = Mock(return_value=bound)
    planner = Mock(return_value=(proposal, dict(objective='actor_response_tracking')))
    monkeypatch.setattr(service.owner.runtime.core, 'act', Mock(return_value=SimpleNamespace(commands_position=proposal)))
    monkeypatch.setattr('fcpg.v21_lewm_cem.plan_commands', planner)
    monkeypatch.setattr('fcpg.v21_visual_task_cost.bind_visual_task_cost', binder)
    monkeypatch.setattr(service, 'column_dino_fg_response', response)
    tip, target = {}, {}
    monkeypatch.setattr(service, 'fingertips', {'column': tip})
    monkeypatch.setattr(service, 'targets', {'targets': {'pickup': {'column': target}}})
    assert service._lewm_cem_builder(state, torch.arange(5, dtype=torch.float64)[None]*.1) is proposal
    binder.assert_called_once_with(response, tip, target, state, actor_commands=proposal)
    assert planner.call_args.kwargs['visual_task_cost'] is bound
    assert service.lewm_cem_record['visual_goal'] is True
    assert service.owner.writes == 0


def test_tracking_plan_and_fake_execution_preserve_source_receipt_and_gates(service, monkeypatch):
    service.lewm_cem = True
    service.lewm_cem_objective = 'actor_response_tracking'
    runtime = service.owner.runtime
    previous_state = runtime.state
    receipt = dict(objective='actor_response_tracking', ranking_state_source='lewm_predicted_readback')

    def plan(_lewm, state, _proposal, _dt, physical_check, **kwargs):
        assert kwargs['objective'] == 'actor_response_tracking'
        commands = state.q[:, None].expand(-1, 5, -1).clone() + .25
        physical_check(state.q, commands)
        return commands, receipt

    monkeypatch.setattr('fcpg.v21_lewm_cem.plan_commands', plan)
    monkeypatch.setattr(service, '_vertical_pinch_builder', _forbidden)
    monkeypatch.setattr(service, '_column_dino_fg_probe', _forbidden)
    physical = Mock(wraps=runtime.physical_check)
    prediction = Mock(wraps=runtime.core.predict)
    monkeypatch.setattr(runtime, 'physical_check', physical)
    monkeypatch.setattr(runtime.core, 'predict', prediction)

    planned = service.handle('plan')
    assert planned['plan']['action_source'] == 'v21_lewm_cem'
    assert planned['plan']['lewm_cem']['objective'] == 'actor_response_tracking'
    assert planned['model']['lewm_cem_objective'] == 'actor_response_tracking'
    assert runtime.state is previous_state
    assert not service._uses_vertical_pinch_geometry()
    assert service.owner.writes == 0 and physical.call_count >= 2
    prediction.assert_called_once()

    result = service.handle('execute')['last_execution']
    assert result['completed'] and service.owner.writes == 5
    assert result['action_source'] == 'v21_lewm_cem'
    assert result['lewm_cem']['objective'] == 'actor_response_tracking'
    assert result['lewm_cem']['visual_goal'] is False
    assert result['lewm_cem']['ranking_state_source'] == 'lewm_predicted_readback'
    assert runtime.state is not previous_state
    assert physical.call_count >= 3 and prediction.call_count >= 2
    assert service.execution_history[-1]['lewm_cem'] == result['lewm_cem']


def test_visual_failure_cannot_switch_to_response_tracking(service, monkeypatch):
    service.lewm_cem = True
    service.lewm_cem_objective = 'column_visual'
    service.fingertips = {}
    service.targets = {}
    attempts = []

    def reject(*_args, **kwargs):
        attempts.append(kwargs)
        assert kwargs.get('objective', 'column_visual') == 'column_visual'
        assert kwargs['require_candidate_sensitive_visual'] is True
        raise RuntimeError('missing_exposure_paired_column_dino_fg_transport')

    monkeypatch.setattr('fcpg.v21_lewm_cem.plan_commands', reject)
    monkeypatch.setattr(service, '_vertical_pinch_builder', _forbidden)
    with pytest.raises(RuntimeError, match='missing_exposure_paired'):
        service.handle('plan')
    assert len(attempts) == 1
    assert service.lewm_cem_objective == 'column_visual'
    assert service.owner.runtime.pending is None
    assert service.owner.runtime.stopped and service.owner.writes == 0


@pytest.mark.parametrize('incompatible', ['--raw-actor', '--no-lewm-cem'])
def test_cli_tracking_conflicts_fail_before_opening_server(incompatible, monkeypatch, capsys):
    monkeypatch.setattr(serve, 'make_server', _forbidden)
    monkeypatch.setattr(serve, 'V21LiveService', _forbidden)
    with pytest.raises(SystemExit) as exc:
        serve.main(['--lewm-cem-objective', 'actor_response_tracking', incompatible])
    assert exc.value.code == 2
    assert 'requires LeWM CEM and cannot use raw Actor mode' in capsys.readouterr().err


def test_cli_keeps_visual_default_and_requires_explicit_tracking_choice():
    default = serve.build_parser().parse_args([])
    assert default.lewm_cem and default.lewm_cem_objective == 'column_visual'
    explicit = serve.build_parser().parse_args([
        '--lewm-cem', '--lewm-cem-objective', 'actor_response_tracking'])
    assert explicit.lewm_cem and explicit.lewm_cem_objective == 'actor_response_tracking'
    assert not explicit.raw_actor and not explicit.enable_hardware
