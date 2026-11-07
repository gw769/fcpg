from types import SimpleNamespace

import pytest
import torch

from fcpg.v21_approach_horizon import select_approach_commands
from test_v21_live_service import service  # noqa: F401


def cue(distance):
    return torch.tensor([[1., .9, distance / 224., 0., .4, .15, 0., 1.]])


def test_reproduced_near_object_small_step_does_not_extend_to_fifteen():
    q = torch.tensor([[-13.15, -9.84, 29.19, 58.64, -64.99, 31.73]])
    # Small average over six axes hid a substantial accumulated J2 advance.
    full = q[:, None].repeat(1, 40, 1)
    full[0, :, 1] += torch.arange(1, 41).clamp(max=15) * 1.96
    full[0, :, 2] -= torch.arange(1, 41).clamp(max=15) * 1.19
    result, note = select_approach_commands(q, full, cue(75.))
    assert note['mean_abs_step'] < .45
    assert note['execute_steps'] == 5 and note['actor_target_index'] == 4
    assert result.shape == (5, 6)
    alphas = torch.linspace(.2, 1., 5)
    expected = q[0] + alphas[:, None] * (full[0, 4] - q[0])
    torch.testing.assert_close(result, expected)
    assert not note['physical_alignment_verified']


def test_near_object_large_prediction_is_scaled_and_does_not_close_off_center():
    q = torch.zeros(1, 6); q[0, 5] = 30.
    full = q[:, None].repeat(1, 40, 1)
    full[..., 1] = 20.; full[..., 2] = -10.; full[..., 5] = 25.
    result, note = select_approach_commands(q, full, cue(35.))
    assert note['execute_steps'] == 5 and note['conservative_approach']
    torch.testing.assert_close(result[-1, :5], full[0, 4, :5])
    torch.testing.assert_close(result[:, 5], torch.linspace(29., 25., 5))
    assert not note['close_deferred_until_centered']
    assert result[-1, 1] == -2 * result[-1, 2]


def test_centered_does_not_certify_grasp_but_allows_bounded_predicted_close():
    q = torch.zeros(1, 6); q[0, 5] = 30.
    full = q[:, None].repeat(1, 40, 1); full[..., 5] = 20.
    result, note = select_approach_commands(q, full, cue(10.))
    torch.testing.assert_close(result[:, 5], torch.linspace(28., 20., 5))
    assert not note['close_deferred_until_centered']
    assert not note['physical_alignment_verified']


@pytest.mark.parametrize('relative', [None, torch.zeros(1, 8), torch.full((1, 8), float('nan'))])
def test_missing_or_occluded_target_never_enables_long_extension(relative):
    q = torch.zeros(1, 6)
    result, note = select_approach_commands(q, torch.ones(1, 40, 6), relative)
    assert result.shape == (5, 6) and note['conservative_approach']
    assert note['wrist_target_distance_px'] is None


@pytest.mark.parametrize('step', [.001, 0.1, 1.])
def test_distant_target_also_executes_only_five_steps_even_when_small(step):
    q = torch.zeros(1, 6)
    full = (torch.arange(1, 41) * step)[None, :, None].expand(1, 40, 6).clone()
    result, note = select_approach_commands(q, full, cue(130.))
    assert note['execute_steps'] == 5 and not note['conservative_approach']
    assert note['small_step_extension_enabled'] is False
    torch.testing.assert_close(result, full[0, :5])


def test_live_builder_sends_near_object_reference_through_cem(monkeypatch):
    from fcpg.v21_live_service import V21LiveService
    from fcpg import v21_lewm_cem
    q = torch.zeros(1, 6); q[0, 5] = 30.
    full = q[:, None].repeat(1, 40, 1); full[..., 1] = 20.
    calls = []
    def act(state, text_ids, offsets):
        calls.append(offsets.shape)
        return SimpleNamespace(commands_position=full)
    service = V21LiveService.__new__(V21LiveService)
    service.record_policy_inputs = False
    service.lewm_cem_objective = 'actor_response_tracking'
    service.lewm_cem_candidate_parameterization = 'scaled_actor_sequence'
    service.lewm_cem_action_scale = 1.
    service.owner = SimpleNamespace(runtime=SimpleNamespace(
        core=SimpleNamespace(actor=SimpleNamespace(horizon=40), act=act, lewm=None),
        text_ids=None, physical_check=lambda *_: None))
    cem_calls = []
    def plan(lewm, state, proposal, *args, **kwargs):
        cem_calls.append(proposal.clone())
        return proposal, dict(objective='actor_response_tracking')
    monkeypatch.setattr(v21_lewm_cem, 'plan_commands', plan)
    state = SimpleNamespace(q=q, object_relative=cue(40.))
    commands = service._lewm_cem_builder(state, torch.arange(5)[None].double() * .1)
    assert calls == [torch.Size([1, 40])]
    assert commands.shape == (1, 5, 6)
    torch.testing.assert_close(commands[0, :, 1], torch.linspace(4., 20., 5))
    assert service._horizon40_extension is None
    assert service.lewm_cem_record['execute_steps'] == 5
    assert len(cem_calls) == 1
    assert service.lewm_cem_record['cem_bypassed'] is False
    assert service.lewm_cem_record['cem_forecast_steps'] == 5


def test_small_plan_dispatch_sends_only_five_commands(service, monkeypatch):
    import builtins
    import io
    import threading
    from fcpg.v21_hardware_bridge import V21TransactionBridge
    from fcpg import v21_lewm_cem

    original_open = builtins.open
    def isolated_open(path, *args, **kwargs):
        if str(path).endswith('/.cursor/debug-75d952.log'):
            return io.StringIO()
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', isolated_open)

    service.lewm_cem = True
    service.replan_on_execute = False
    service.record_policy_inputs = False
    service.lewm_cem_objective = 'actor_response_tracking'
    def plan(lewm, state, proposal, *args, **kwargs):
        return proposal, dict(objective='actor_response_tracking')
    monkeypatch.setattr(v21_lewm_cem, 'plan_commands', plan)
    owner = service.owner
    owner.runtime.core.actor.horizon = 40
    def act(state, text_ids, offsets):
        commands = state.q[:, None].expand(-1, 40, -1).clone()
        commands[:, :, 5] += .01 * torch.arange(1, 41)
        return SimpleNamespace(commands_position=commands)
    monkeypatch.setattr(owner.runtime.core, 'act', act)
    # Model the existing gain IO explicitly; never use a physical servo bus.
    gains = {}
    bus = SimpleNamespace(
        read=lambda register, name, **kw: gains.get((register, name), 16),
        write=lambda register, name, value, **kw: gains.__setitem__((register, name), value))
    owner.io.follower = SimpleNamespace(bus=bus)
    owner.bus_lock = threading.RLock()
    writes = []
    def write(command, previous):
        owner.q = command.copy()
        writes.append(command.copy())
        return command.copy()
    owner.bridge = V21TransactionBridge(owner.runtime, read_q=owner.read_checked,
        write_q=write, capture_endpoint=owner.capture, physical_check=lambda q, u: None,
        halt=owner._halt, clock=service.clock, wait=owner.advance_clock,
        cancelled=owner.stop_event.is_set, request_stop=owner.stop_event.set)
    plan = service.handle('plan')['plan']
    assert plan['lewm_cem']['mean_abs_step'] < .45
    result = service.handle('execute')['last_execution']
    assert len(writes) == 5 and result['executed_steps'] == 5
    assert result.get('extension_commands_position') is None
    assert result['policy_commands_attempted'] == result['policy_commands_returned'] == 5
