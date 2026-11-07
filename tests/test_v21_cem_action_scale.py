from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg import v21_lewm_cem as cem
from fcpg.prod_v20_policy import DEFAULT_LOWER, DEFAULT_UPPER, empty_history
from fcpg.v21_cem_trace import CEMTrace


MODE = 'amplified_actor_sequence'
LEGACY_MODE = 'scaled_actor_endpoint_hold'
OBJECTIVE = 'actor_response_tracking'
EXPANDED_DELTA = (4.5, 4.5, 4.5, 3.0, 3.0, 7.5)


@pytest.fixture(autouse=True)
def small_search(monkeypatch):
    monkeypatch.setattr(cem, 'CEM_POPULATION', 6)
    monkeypatch.setattr(cem, 'CEM_ELITES', 2)
    monkeypatch.setattr(cem, 'CEM_ITERS', 2)


def inputs():
    q = torch.tensor([[0., -90., 85., 60., -50., 30.]])
    endpoint_delta = q.new_tensor([1., 1., -1., -.5, .5, 1.])
    actor = q[:, None] + torch.arange(1, 6).reshape(1, 5, 1) * endpoint_delta / 5
    return SimpleNamespace(q=q, history=empty_history(1)), actor, torch.full((1, 5), .1)


def identity_lewm(source, command, history, query):
    return SimpleNamespace(predicted_q_path=torch.cat((source[:, None], command.positions), 1),
        q_uncertainty=torch.ones_like(command.positions))


def assert_bounded_sequence(q, commands, action_scale=1.5):
    commands = commands.reshape(-1, 5, 6)
    source = q[:, None].expand(len(commands), 1, 6)
    increments = torch.diff(torch.cat((source, commands), 1), dim=1)
    limits = torch.minimum(commands.new_tensor(cem.MAX_DELTA)*action_scale,
                           commands.new_tensor(cem.MAX_STEP)*5)
    assert ((commands-source).abs() <= limits+1e-4).all()
    assert (increments.abs() <= commands.new_tensor(cem.MAX_STEP)+1e-4).all()
    assert (commands >= commands.new_tensor(DEFAULT_LOWER)-1e-4).all()
    assert (commands <= commands.new_tensor(DEFAULT_UPPER)+1e-4).all()


@pytest.mark.parametrize('scale', [True, False, None, '1.5', 0, -1, 1.25, 1.5001,
    2., 3.0001, float('nan'), float('inf'), float('-inf'), torch.tensor(1.5)])
def test_invalid_scale_rejected_before_model_or_physical_access(scale):
    def forbidden(*args):
        raise AssertionError('invalid scale reached model or geometry')
    with pytest.raises(ValueError, match='action_scale must be'):
        cem.plan_commands(forbidden, None, None, None, forbidden, action_scale=scale)


@pytest.mark.parametrize('objective,mode', [
    ('column_visual', MODE), ('actor_command_tracking', MODE),
    (OBJECTIVE, None), (OBJECTIVE, 'five_command_sequence'),
    (OBJECTIVE, 'scaled_actor_sequence'),
    (OBJECTIVE, 'scaled_actor_endpoint_hold'), (OBJECTIVE, 'bounded_endpoint_then_hold')])
@pytest.mark.parametrize('action_scale', [1.5, 3.0])
def test_expanded_scale_requires_explicit_response_sequence_mode(objective, mode, action_scale):
    def forbidden(*args):
        raise AssertionError('unsupported expanded mode reached model or geometry')
    with pytest.raises(ValueError, match='expanded action_scale requires'):
        cem.plan_commands(forbidden, None, None, None, forbidden, objective=objective,
            candidate_parameterization=mode, action_scale=action_scale)


def test_physical_box_expands_all_six_axes_and_preserves_absolute_bounds():
    state, actor, _ = inputs()
    for scalar in (1.5, np.float32(1.5), np.float64(1.5)):
        low, high = cem._physical_box(actor, state.q, action_scale=scalar)
        torch.testing.assert_close(state.q[0]-low, actor.new_tensor(EXPANDED_DELTA), atol=0, rtol=0)
        torch.testing.assert_close(high-state.q[0], actor.new_tensor(EXPANDED_DELTA), atol=0, rtol=0)
    lower, upper = actor.new_tensor(DEFAULT_LOWER), actor.new_tensor(DEFAULT_UPPER)
    for boundary in (lower+.01, upper-.01):
        low, high = cem._physical_box(actor, boundary[None], action_scale=1.5)
        torch.testing.assert_close(low, torch.maximum(lower, boundary-actor.new_tensor(EXPANDED_DELTA)))
        torch.testing.assert_close(high, torch.minimum(upper, boundary+actor.new_tensor(EXPANDED_DELTA)))
    _, no_descend_high = cem._physical_box(actor, state.q, allow_descend_j2=False, action_scale=1.5)
    assert no_descend_high[1] == state.q[0, 1]
    assert cem.MAX_DELTA == (8., 8., 8., 5., 5., 10.)
    assert cem.MAX_STEP == (4., 4., 4., 2.5, 2.5, 5.)


@pytest.mark.parametrize('action_scale', [1.5, 3.0])
def test_actor_reference_amplifies_each_axis_and_preserves_sequence(action_scale):
    state, actor, _ = inputs()
    original_q, original_actor = state.q.clone(), actor.clone()
    expanded, meta = cem.prepare_actor_reference(
        actor, state.q, MODE, action_scale=action_scale)
    expected = state.q[:, None] + action_scale * (actor-state.q[:, None])
    torch.testing.assert_close(expanded, expected, atol=1e-5, rtol=0)
    assert meta['action_scale'] == action_scale
    assert meta['shared_alpha'] is False
    assert meta['actor_time_shape_preserved'] and not meta['endpoint_hold']
    assert torch.unique(expanded[0, :, 5]).numel() == 5
    assert_bounded_sequence(state.q, expanded, action_scale)
    assert torch.equal(state.q, original_q) and torch.equal(actor, original_actor)


def test_saturated_j2_does_not_shrink_other_arm_axes_or_gripper():
    state, actor, _ = inputs()
    actor = actor.clone()
    actor[0, :, 1] = state.q[0, 1] + torch.tensor([3.8, 7.6, 11.4, 15.2, 19.0])
    reference, meta = cem.prepare_actor_reference(actor, state.q, MODE, action_scale=3.0)
    requested = state.q[:, None] + 3.0 * (actor-state.q[:, None])
    assert not torch.equal(reference[..., 1], requested[..., 1])
    torch.testing.assert_close(reference[..., [0, 2, 3, 4, 5]],
        requested[..., [0, 2, 3, 4, 5]], atol=1e-5, rtol=0)
    torch.testing.assert_close(reference[0, :, 5]-state.q[0, 5],
        torch.tensor([.6, 1.2, 1.8, 2.4, 3.0]), atol=1e-5, rtol=0)
    assert meta['shared_alpha'] is False


def test_default_and_explicit_one_preserve_winner_cost_candidates_and_rng():
    state, actor, dt = inputs()
    results = []
    for kwargs in ({}, {'action_scale': 1.0}):
        rng, trace = torch.Generator().manual_seed(72), CEMTrace()
        selected, receipt = cem.plan_commands(identity_lewm, state, actor, dt, lambda q, u: None,
            objective=OBJECTIVE, candidate_parameterization=LEGACY_MODE,
            generator=rng, trace=trace, **kwargs)
        results.append((selected, receipt, rng.get_state(), trace.payload()))
    first, explicit = results
    assert torch.equal(first[0], explicit[0]) and first[1] == explicit[1]
    assert torch.equal(first[2], explicit[2])
    for left, right in zip(first[3]['events'], explicit[3]['events'], strict=True):
        assert left['event'] == right['event']
        if left['event'] == 'cem_iteration':
            assert torch.equal(left['values']['candidates'], right['values']['candidates'])
    assert first[1]['action_scale'] == 1.
    assert first[1]['max_delta'] == first[1]['cost_scale'] == list(cem.MAX_DELTA)


@pytest.mark.parametrize('action_scale', [1.5, 3.0])
def test_every_candidate_stage_and_winner_use_expanded_box_and_original_slew_limits(action_scale):
    state, actor, dt = inputs()
    rejections = []
    def geometry(q, commands):
        assert_bounded_sequence(q, commands, action_scale)
        if (commands[..., 0] < q[0, 0]-.05).any():
            rejections.append(True)
            raise RuntimeError('exercise replacement sampling')
    def model(q, command, history, query):
        assert_bounded_sequence(q[:1], command.positions, action_scale)
        return identity_lewm(q, command, history, query)
    trace = CEMTrace()
    selected, receipt = cem.plan_commands(model, state, actor, dt, geometry,
        objective=OBJECTIVE, candidate_parameterization=MODE, action_scale=action_scale,
        generator=torch.Generator().manual_seed(20), trace=trace)
    assert rejections
    assert_bounded_sequence(state.q, selected, action_scale)
    # Identity dynamics must retain the amplified Actor sequence.
    reference, _ = cem.prepare_actor_reference(actor, state.q, MODE, action_scale=action_scale)
    torch.testing.assert_close(selected, reference, atol=0, rtol=0)
    assert receipt['action_scale'] == action_scale
    effective_delta = [min(base*action_scale, step*5)
        for base, step in zip(cem.MAX_DELTA, cem.MAX_STEP)]
    assert receipt['max_delta'] == effective_delta
    if action_scale == 3.0:
        assert receipt['requested_max_delta'] == [9., 9., 9., 6., 6., 15.]
        assert receipt['max_delta'] == [7.5, 7.5, 7.5, 5., 5., 12.5]
        assert receipt['horizon_limited'] is True and receipt['command_horizon'] == 5
    else:
        assert not {'requested_max_delta', 'horizon_limited', 'command_horizon'} & receipt.keys()
    assert receipt['max_step'] == list(cem.MAX_STEP)
    assert receipt['cost_scale'] == list(cem.MAX_DELTA)
    assert receipt['retained_candidates'] > 0
    assert receipt['response_forecast_admission']['minimum_motion'] == .25
    seen = set()
    for event in trace.payload()['events']:
        name, values = event['event'], event['values']
        if name == 'planning_input':
            assert values['action_scale'] == action_scale
        elif name == 'cem_iteration':
            assert_bounded_sequence(state.q, values['candidates'], action_scale)
        elif name == 'cem_final_mean':
            assert_bounded_sequence(state.q, values['action_sequence'], action_scale)
        elif name == 'cem_retained_candidates':
            for candidates in values['candidates']:
                assert_bounded_sequence(state.q, candidates, action_scale)
        elif name in ('retained_candidates_appended', 'ranked_pool_evaluation', 'selected'):
            assert_bounded_sequence(state.q, values['commands'], action_scale)
        seen.add(name)
    assert {'cem_iteration', 'cem_final_mean', 'cem_retained_candidates',
        'retained_candidates_appended', 'ranked_pool_evaluation', 'selected'} <= seen


def test_expansion_does_not_renormalize_tracking_cost():
    state, actor, dt = inputs()
    path = torch.cat((state.q[:, None], actor+.4), 1)
    models = [cem.V21JointResponseCEM(identity_lewm, state.q, state.history, dt,
        lambda q, u: None, objective=OBJECTIVE, reference_commands=actor, action_scale=scale)
        for scale in (1., 1.5, 3.)]
    costs = [model._trajectory_cost(path[0]) for model in models]
    torch.testing.assert_close(costs[0], costs[1], atol=0, rtol=0)
    torch.testing.assert_close(costs[0], costs[2], atol=0, rtol=0)
    expected = ((path[:, 1:]-actor)/path.new_tensor(cem.MAX_DELTA)).square().mean()
    torch.testing.assert_close(costs[1], expected, atol=0, rtol=0)
    for model in models:
        assert model._cost_components(path)['scale'] == list(cem.MAX_DELTA)


@pytest.mark.parametrize('action_scale', [1.5, 3.0])
def test_expansion_does_not_bypass_geometry_or_forecast_response_admission(action_scale):
    state, actor, dt = inputs()
    def forbidden_model(*args):
        raise AssertionError('unsafe commands reached LeWM')
    def unsafe(*args):
        raise RuntimeError('unsafe geometry fixture')
    with pytest.raises(RuntimeError, match='no physically admissible seed'):
        cem.plan_commands(forbidden_model, state, actor, dt, unsafe, objective=OBJECTIVE,
            candidate_parameterization=MODE, action_scale=action_scale,
            generator=torch.Generator().manual_seed(91))
    def static_lewm(q, command, history, query):
        return SimpleNamespace(predicted_q_path=q[:, None].expand(-1, 6, -1).clone(),
            q_uncertainty=torch.ones_like(command.positions))
    with pytest.raises(RuntimeError, match='no CEM candidate passed physical and predicted directional response'):
        cem.plan_commands(static_lewm, state, actor, dt, lambda q, u: None, objective=OBJECTIVE,
            candidate_parameterization=MODE, action_scale=action_scale,
            generator=torch.Generator().manual_seed(91))
    assert cem.MIN_DIRECTIONAL_RESPONSE == .25
