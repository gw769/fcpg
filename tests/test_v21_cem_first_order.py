"""CEM-first ordering and forecast admission; CPU contracts, no hardware IO."""
from types import SimpleNamespace

import pytest
import torch

from fcpg.prod_v20_policy import empty_history
from fcpg.v21_lewm_cem import (
    MAX_DELTA,
    MAX_STEP,
    V21JointResponseCEM,
    constrain_to_physical,
    plan_commands,
)


def candidates(*positions):
    commands = torch.zeros(1, len(positions), 5, 6)
    commands[0, :, :, 0] = torch.tensor(positions)[:, None]
    return commands


class GoalFreeLeWM:
    """A strict four-argument model cannot accidentally receive a task goal."""

    def __init__(self, events=None, corrupt=None):
        self.calls = []
        self.events = events
        self.corrupt = corrupt

    def __call__(self, q, trace, history, query):
        if self.events is not None:
            assert self.events and self.events[-1] in ('ranked', 'forecast')
            self.events.append('forecast')
        trace.validate(len(q))
        assert q.shape == (1, 6)
        assert trace.positions.shape == (1, 5, 6)
        assert history.commands_position.shape[0] == 1
        assert query.shape == (1, 5) and query.dtype == torch.float64
        assert torch.all(query > trace.offsets)
        assert torch.all(query.diff(dim=-1) > 0)
        self.calls.append((q.clone(), trace.positions.clone(), query.clone()))
        future = q[:, None] + .5 * (trace.positions - q[:, None])
        result = SimpleNamespace(
            predicted_q_path=torch.cat((q[:, None], future), dim=1),
            q_uncertainty=torch.ones_like(future),
        )
        if self.corrupt is not None:
            self.corrupt(result, trace.positions)
        return result


@pytest.fixture
def small_cem(monkeypatch):
    # Exercise the real CEM iterations; only reduce the CPU test population.
    monkeypatch.setattr('fcpg.v21_lewm_cem.CEM_POPULATION', 6)
    monkeypatch.setattr('fcpg.v21_lewm_cem.CEM_ELITES', 2)
    monkeypatch.setattr('fcpg.v21_lewm_cem.CEM_ITERS', 2)


def run_plan(lewm, actor, physical_check=lambda _q, _u: None):
    state = SimpleNamespace(q=torch.zeros(1, 6), history=empty_history(1))
    return plan_commands(
        lewm, state, actor, torch.full((1, 5), .1), physical_check,
        objective='actor_command_tracking',
        generator=torch.Generator().manual_seed(19),
    )


def fixed_pool(monkeypatch, pool):
    monkeypatch.setattr(
        'fcpg.v21_lewm_cem._candidate_pool',
        lambda _plan, _actor, _low, _high, _generator: pool.clone(),
    )


def test_command_ranking_never_calls_lewm_and_keeps_fixed_actor_reference():
    def forbidden_lewm(*_args):
        pytest.fail('CEM candidate ranking called LeWM before selection')

    actor = candidates(1.)[:, 0]
    model = V21JointResponseCEM(
        forbidden_lewm, torch.zeros(1, 6), empty_history(1),
        torch.full((1, 5), .1), lambda _q, _u: None,
        objective='actor_command_tracking', reference_commands=actor,
    )
    actor.fill_(100.)  # Caller mutation cannot change the fixed reference.
    pool = candidates(2., 1.)
    path = model.rollout_embeddings(torch.zeros(1, 1, 1), torch.zeros(1, 1, 6), pool)
    costs = model.goal_cost(path, torch.zeros(1, 1, 1))
    assert costs[0, 1] == 0
    assert costs[0, 0] > costs[0, 1]
    selected, cost, admitted = model.rank_admitted(pool)
    assert admitted == 2 and cost == pytest.approx(0.)
    torch.testing.assert_close(selected, pool[:, 1])


def test_complete_cem_ranking_precedes_goal_free_forecast(monkeypatch, small_cem):
    events = []
    lewm = GoalFreeLeWM(events)
    original_rank = V21JointResponseCEM.rank_admitted

    def observed_rank(model, commands):
        assert not lewm.calls  # Includes every real CEM optimization iteration.
        result = original_rank(model, commands)
        assert not lewm.calls
        events.append('ranked')
        return result

    monkeypatch.setattr(V21JointResponseCEM, 'rank_admitted', observed_rank)
    selected, receipt = run_plan(lewm, candidates(1.)[:, 0])
    assert events == ['ranked', 'forecast']
    assert len(lewm.calls) == 1
    torch.testing.assert_close(lewm.calls[0][1], selected)
    torch.testing.assert_close(lewm.calls[0][2], torch.arange(1, 6)[None].double() * .1)
    assert receipt['objective'] == 'actor_command_tracking'
    assert receipt['ranking_state_source'] == 'candidate_command'
    assert receipt['lewm_after_cem'] is True
    assert receipt['task_visual_optimization'] is False
    assert receipt['used_actor_fallback'] is False
    assert receipt['used_hold_fallback'] is False
    prediction = torch.tensor(receipt['selected_predicted_q_path'])
    torch.testing.assert_close(prediction[1:], selected[0] * .5)
    assert not torch.equal(prediction[1:], selected[0])


def test_reference_is_bounded_before_cem_scoring(monkeypatch, small_cem):
    actor = candidates(100.)[:, 0]
    selected, receipt = run_plan(GoalFreeLeWM(), actor)
    reference = torch.tensor(receipt['reference_commands_position'])
    torch.testing.assert_close(reference[:, 0], torch.tensor([1.5, 3., 3., 3., 3.]))
    torch.testing.assert_close(selected[0], reference)
    assert receipt['cost'] == pytest.approx(0.)
    assert receipt['candidate_parameterization'] == 'bounded_endpoint_then_hold'
    assert torch.all(actor[..., 0] == 100.)


def test_unsafe_first_prediction_uses_second_backup_and_its_own_receipt(monkeypatch, small_cem):
    pool = candidates(1., .5)
    fixed_pool(monkeypatch, pool)
    checked = []

    def physical_check(_q, commands):
        checked.append(commands.clone())
        if torch.any(commands[..., 0] > 3.):
            raise RuntimeError('predicted path crosses physical boundary')

    def corrupt_first(result, commands):
        if torch.all(commands[..., 0] == 1.):
            result.predicted_q_path[:, 1:, 0] = 4.

    lewm = GoalFreeLeWM(corrupt=corrupt_first)
    selected, receipt = run_plan(lewm, pool[:, 0], physical_check)
    assert len(lewm.calls) == 2
    torch.testing.assert_close(lewm.calls[0][1], pool[:, 0])
    torch.testing.assert_close(lewm.calls[1][1], pool[:, 1])
    torch.testing.assert_close(selected, pool[:, 1])
    assert receipt['backup_index'] == 1
    assert receipt['backups'] == 2 + receipt['retained_candidates']
    expected_cost = ((selected - pool[:, 0]) / selected.new_tensor(MAX_DELTA)).square().mean()
    assert receipt['cost'] == pytest.approx(float(expected_cost))
    assert receipt['cost'] > 0.  # The rejected first backup had zero cost.
    prediction = torch.tensor(receipt['selected_predicted_q_path'])
    torch.testing.assert_close(prediction[1:], selected[0] * .5)
    assert any(torch.all(value[..., 0] == 4.) for value in checked)


@pytest.mark.parametrize('kind', ['nan_future', 'wrong_source', 'nan_sigma', 'zero_sigma', 'unsafe_future'])
def test_invalid_post_selection_prediction_never_returns_an_execution_command(monkeypatch, small_cem, kind):
    fixed_pool(monkeypatch, candidates(1., .5))

    def corrupt(result, _commands):
        if kind == 'nan_future':
            result.predicted_q_path[:, 1, 0] = float('nan')
        elif kind == 'wrong_source':
            result.predicted_q_path[:, 0, 0] = .1
        elif kind == 'nan_sigma':
            result.q_uncertainty[:, 0, 0] = float('nan')
        elif kind == 'zero_sigma':
            result.q_uncertainty[:, 0, 0] = 0.
        elif kind == 'unsafe_future':
            result.predicted_q_path[:, 1:, 0] = 4.

    def physical_check(_q, commands):
        if torch.any(commands[..., 0] > 3.):
            raise RuntimeError('physical boundary')

    lewm = GoalFreeLeWM(corrupt=corrupt)
    with pytest.raises(RuntimeError, match='no physically admissible backup after LeWM'):
        run_plan(lewm, candidates(1.)[:, 0], physical_check)
    # The fixed fresh pool has two entries; both iterations also retain their
    # two legal elites. Every backup must fail the same forecast admission.
    assert len(lewm.calls) == 2 + 2 * 2


def test_sustained_candidates_do_not_follow_out_and_back_noise():
    q = torch.tensor([[10., -10., 20., -20., 30., 40.]])
    span = torch.tensor(MAX_DELTA)
    low, high = q[0] - span, q[0] + span
    endpoint = q[0] + torch.tensor([3., -3., 3., -2., 2., 5.])
    pool = q[:, None, None].expand(1, 1, 5, 6).clone()
    # Each axis reverses repeatedly before the requested final destination.
    pool[0, 0, 0] = q[0] - span
    pool[0, 0, 1] = q[0] + span
    pool[0, 0, 2] = q[0] - span
    pool[0, 0, 3] = q[0]
    pool[0, 0, 4] = endpoint
    original = pool.clone()
    constrained = constrain_to_physical(
        pool, physical_check=lambda _q, _u: None, q=q,
        seed=q.expand(5, -1), low=low, high=high, sustained=True,
    )[0, 0]
    steps = torch.cat((q, constrained), dim=0).diff(dim=0)
    direction = torch.sign(endpoint - q[0])
    assert torch.all(steps * direction >= 0)
    assert torch.all(steps.abs() <= torch.tensor(MAX_STEP) + 1e-6)
    torch.testing.assert_close(constrained[-1], endpoint)
    torch.testing.assert_close(constrained[2:], endpoint.expand(3, -1))
    torch.testing.assert_close(pool, original)
