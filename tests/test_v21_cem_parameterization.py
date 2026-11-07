"""Candidate-shape comparisons and config admission, with CPU models/fake IO only."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from fcpg import v21_lewm_cem as cem
from fcpg.prod_v20_policy import empty_history
from fcpg.v21_live_service import V21LiveService
from scripts import serve_so101_v21_research as cli
from test_v21_live_service import service  # noqa: F401 -- fake IO fixture


HOLD = 'bounded_endpoint_then_hold'
SEQUENCE = 'five_command_sequence'
OBJECTIVES = ('actor_command_tracking', 'actor_response_tracking')


@pytest.fixture
def small_cem(monkeypatch):
    monkeypatch.setattr(cem, 'CEM_POPULATION', 6)
    monkeypatch.setattr(cem, 'CEM_ELITES', 2)
    monkeypatch.setattr(cem, 'CEM_ITERS', 2)


def identity_lewm(q, trace, history, query):
    return SimpleNamespace(predicted_q_path=torch.cat((q[:, None], trace.positions), 1),
                           q_uncertainty=torch.ones_like(trace.positions))


def run_plan(objective, **kwargs):
    q = torch.zeros(1, 6)
    actor = torch.zeros(1, 5, 6)
    actor[0, :, 0] = torch.tensor([1., .25, 1.25, .5, .75])
    return cem.plan_commands(identity_lewm, SimpleNamespace(q=q, history=empty_history(1)),
                            actor, torch.full((1, 5), .1), lambda _q, _u: None,
                            objective=objective, generator=torch.Generator().manual_seed(19), **kwargs)


@pytest.mark.parametrize('parameterization', [HOLD, SEQUENCE])
def test_both_objectives_share_explicit_parameterization(small_cem, parameterization):
    results = [run_plan(objective, candidate_parameterization=parameterization)
               for objective in OBJECTIVES]
    expected = torch.zeros(5, 6)
    expected[:, 0] = .75 if parameterization == HOLD else torch.tensor([1., .25, 1.25, .5, .75])
    for selected, receipt in results:
        assert receipt['candidate_parameterization'] == parameterization
        torch.testing.assert_close(torch.tensor(receipt['reference_commands_position']), expected)
        torch.testing.assert_close(selected[0], expected)
        assert receipt['max_delta'] == list(cem.MAX_DELTA)
        assert receipt['max_step'] == list(cem.MAX_STEP)
        assert receipt['single_arm_axis_stall_checked'] is True


@pytest.mark.parametrize(('objective', 'parameterization'), [
    ('actor_command_tracking', HOLD), ('actor_response_tracking', SEQUENCE), ('column_visual', SEQUENCE)])
def test_omitted_parameterization_preserves_objective_default(small_cem, objective, parameterization):
    default = run_plan(objective)
    explicit = run_plan(objective, candidate_parameterization=parameterization)
    assert torch.equal(default[0], explicit[0])
    assert default[1] == explicit[1]


@pytest.mark.parametrize('objective', OBJECTIVES)
@pytest.mark.parametrize('parameterization', [HOLD, SEQUENCE])
def test_every_ranked_candidate_keeps_bounds_and_slew(monkeypatch, small_cem, objective, parameterization):
    source = torch.zeros(1, 6)
    actor = torch.full((1, 5, 6), 100.)
    pools = []
    original = cem.V21JointResponseCEM.rollout_embeddings

    def inspect(model, embedding, action, future):
        pools.append(future.detach().clone())
        return original(model, embedding, action, future)

    monkeypatch.setattr(cem.V21JointResponseCEM, 'rollout_embeddings', inspect)
    selected, receipt = cem.plan_commands(
        identity_lewm, SimpleNamespace(q=source, history=empty_history(1)), actor,
        torch.full((1, 5), .1), lambda _q, _u: None, objective=objective,
        candidate_parameterization=parameterization, generator=torch.Generator().manual_seed(21))
    low, high = cem._physical_box(actor, source)
    assert pools
    pools.append(selected[:, None])
    for pool in pools:
        assert torch.isfinite(pool).all()
        assert (pool >= low - 1e-6).all() and (pool <= high + 1e-6).all()
        assert ((pool - source[:, None, None]).abs() <= pool.new_tensor(cem.MAX_DELTA) + 1e-6).all()
        path = torch.cat((source[:, None, None].expand(1, pool.shape[1], 1, 6), pool), dim=2)
        assert (path.diff(dim=2).abs() <= pool.new_tensor(cem.MAX_STEP) + 1e-6).all()
        if parameterization == HOLD:
            # The existing MAX_DELTA is twice MAX_STEP: the endpoint is reached by q2.
            assert torch.equal(pool[:, :, 1:], pool[:, :, -1:].expand_as(pool[:, :, 1:]))
    assert receipt['candidate_parameterization'] == parameterization
    assert torch.all(actor == 100.)


@pytest.mark.parametrize('parameterization', ['', 'hold', 3, True])
def test_planner_rejects_unknown_parameterization_before_forecast(parameterization):
    def forbidden(*a, **kw):
        pytest.fail('invalid parameterization reached the model or physical gate')

    with pytest.raises(ValueError, match='candidate parameterization'):
        cem.plan_commands(forbidden, None, None, None, forbidden,
                          candidate_parameterization=parameterization)


@pytest.mark.parametrize('kwargs', [
    {'lewm_cem': True, 'lewm_cem_candidate_parameterization': 'hold'},
    {'lewm_cem': False, 'lewm_cem_candidate_parameterization': HOLD},
    {'lewm_cem': True, 'raw_actor': True, 'lewm_cem_candidate_parameterization': SEQUENCE}])
def test_invalid_service_config_never_constructs_owner(tmp_path, kwargs):
    calls = []
    with pytest.raises(ValueError, match='parameterization'):
        V21LiveService(tmp_path / 'model', tmp_path / 'output',
                       owner_factory=lambda *a, **kw: calls.append(kw), **kwargs)
    assert calls == [] and not (tmp_path / 'output').exists()


@pytest.mark.parametrize('parameterization', [HOLD, SEQUENCE])
def test_explicit_service_constructor_exposes_config_without_motion(service, tmp_path, parameterization):
    assert service.lewm_cem_candidate_parameterization is None
    configured = V21LiveService(
        tmp_path / 'model2', tmp_path / 'output2', device='cpu',
        owner_factory=type(service.owner), target_diagnostics=False, clock=service.clock,
        lewm_cem=True, lewm_cem_objective='actor_response_tracking',
        lewm_cem_candidate_parameterization=parameterization)
    try:
        assert configured.lewm_cem_candidate_parameterization == parameterization
        assert configured.status()['model']['lewm_cem_candidate_parameterization'] == parameterization
        assert configured.owner.runtime.stopped and configured.owner.writes == 0
    finally:
        configured.close()


@pytest.mark.parametrize('objective', (*OBJECTIVES, 'column_visual'))
@pytest.mark.parametrize('parameterization', [None, HOLD, SEQUENCE])
def test_service_builder_forwards_only_explicit_config(service, monkeypatch, objective, parameterization):
    service.lewm_cem = True
    service.lewm_cem_objective = objective
    service.lewm_cem_candidate_parameterization = parameterization
    state = service.owner.runtime.state
    commands = state.q[:, None].expand(1, 5, 6).clone()
    monkeypatch.setattr(service.owner.runtime.core, 'act', Mock(return_value=SimpleNamespace(commands_position=commands)))
    planner = Mock(return_value=(commands, {}))
    monkeypatch.setattr(cem, 'plan_commands', planner)
    assert service._lewm_cem_builder(state, torch.arange(5)[None].double() * .1) is commands
    kwargs = planner.call_args.kwargs
    if parameterization is None:
        assert 'candidate_parameterization' not in kwargs
    else:
        assert kwargs['candidate_parameterization'] == parameterization
    assert service.status()['model']['lewm_cem_candidate_parameterization'] == parameterization
    assert service.owner.writes == 0


@pytest.mark.parametrize('args', [
    ['--lewm-cem-candidate-parameterization', 'invalid'],
    ['--lewm-cem-candidate-parameterization', HOLD, '--no-lewm-cem'],
    ['--lewm-cem-candidate-parameterization', SEQUENCE, '--raw-actor']])
def test_invalid_cli_config_precedes_server_and_owner(monkeypatch, args):
    def forbidden(*a, **kw):
        pytest.fail('invalid configuration reached server or owner creation')

    monkeypatch.setattr(cli, 'make_server', forbidden)
    monkeypatch.setattr(cli, 'V21LiveService', forbidden)
    with pytest.raises(SystemExit) as caught:
        cli.main(['--enable-hardware', *args])
    assert caught.value.code == 2


@pytest.mark.parametrize('parameterization', [None, HOLD, SEQUENCE])
def test_cli_forwards_only_explicit_config(monkeypatch, parameterization):
    calls = []
    server = SimpleNamespace(serve_forever=lambda **kw: None, shutdown=lambda: None,
                             server_close=lambda: None)
    monkeypatch.setattr(cli, 'make_server', lambda *a: server)
    monkeypatch.setattr(cli.signal, 'signal', lambda *a: None)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')

    def construct(*a, **kw):
        calls.append(kw)
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(cli, 'V21LiveService', construct)
    args = [] if parameterization is None else ['--lewm-cem-candidate-parameterization', parameterization]
    cli.main(['--enable-hardware', *args])
    assert len(calls) == 1
    if parameterization is None:
        assert 'lewm_cem_candidate_parameterization' not in calls[0]
    else:
        assert calls[0]['lewm_cem_candidate_parameterization'] == parameterization
