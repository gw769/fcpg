"""CEM must optimize LeWM readbacks, not reinterpret commands as readbacks.

The half-gain dynamics below deliberately separate commanded positions from
predicted response. These are CPU contract tests, not physical success evidence.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg.prod_v20_policy import empty_history
from fcpg.v21_lewm_cem import INADMISSIBLE_COST, V21JointResponseCEM, plan_commands


class HalfGainLeWM:
    """Strict goal-free signature, with optional per-candidate corruption."""

    def __init__(self, corrupt=None):
        self.calls = []
        self.corrupt = corrupt

    def __call__(self, q, trace, history, query, *, patches=None, patch_valid=None):   # V21LeWM.forward's keywords
        trace.validate(len(q))
        assert history.commands_position.shape[0] == len(q)
        assert query.shape == (len(q), 5)
        assert query.dtype == torch.float64
        assert torch.all(query > trace.offsets)
        assert torch.all(query.diff(dim=-1) > 0)
        self.calls.append((q.clone(), trace, history, query.clone()))
        predicted = q[:, None] + .5 * (trace.positions - q[:, None])
        result = SimpleNamespace(
            predicted_q_path=torch.cat((q[:, None], predicted), dim=1),
            q_uncertainty=torch.ones_like(predicted),
        )
        if self.corrupt is not None:
            self.corrupt(result, trace.positions)
        return result


def candidates(*positions):
    result = torch.zeros(1, len(positions), 5, 6)
    result[0, :, :, 0] = torch.tensor(positions)[:, None]
    return result


def visual_model(lewm, physical_check=lambda _q, _u: None, **kwargs):
    return V21JointResponseCEM(
        lewm=lewm, q=torch.zeros(1, 6), history=empty_history(1),
        dt=torch.full((1, 5), .1), physical_check=physical_check,
        project=lambda q: np.asarray(q[:2]), target_px=(1., 0.),
        observed_midpoint_px=(0., 0.), capture_q=np.zeros(6), **kwargs,
    )


def rollout(model, commands):
    return model.rollout_embeddings(
        commands.new_zeros(1, 1, 1), commands.new_zeros(1, 1, 6), commands)


def test_all_candidates_are_forecast_with_goal_free_batched_lewm():
    lewm = HalfGainLeWM()
    model = visual_model(lewm)
    pool = candidates(1., 2., 3.)
    path = rollout(model, pool)
    assert path.shape == (1, 3, 6, 6)
    torch.testing.assert_close(path[:, :, 0], torch.zeros(1, 3, 6))
    torch.testing.assert_close(path[:, :, 1:], pool * .5)
    assert len(lewm.calls) == 1
    q, trace, history, query = lewm.calls[0]
    assert q.shape == (3, 6)
    torch.testing.assert_close(trace.positions, pool[0])
    torch.testing.assert_close(query, torch.arange(1, 6).double()[None].expand(3, -1) * .1)
    assert history.valid.shape == (3, 3)


def test_goal_cost_and_final_rank_both_use_predicted_readbacks():
    model = visual_model(HalfGainLeWM())
    pool = candidates(1., 2.)
    path = rollout(model, pool)
    costs = model.goal_cost(path, torch.zeros(1, 1, 1))
    # Command=1 looks perfect only when commands are incorrectly treated as q.
    assert costs[0, 1] < costs[0, 0]
    selected, cost, admitted = model.rank_admitted(pool)
    assert admitted == 2
    torch.testing.assert_close(selected, pool[:, 1])
    assert cost == pytest.approx(0.)


def test_visual_candidate_spread_is_measured_on_predicted_response():
    model = visual_model(HalfGainLeWM(), require_candidate_sensitive_visual=True)
    selected, _, admitted = model.rank_admitted(candidates(0., 1.))
    assert selected is not None and admitted == 2
    assert model.candidate_spread_px == pytest.approx(.5)


@pytest.mark.parametrize('kind', [
    'nan_future', 'inf_future', 'wrong_source',
    'nan_sigma', 'inf_sigma', 'zero_sigma', 'negative_sigma',
])
def test_invalid_predicted_candidate_cannot_be_scored_or_ranked(kind):
    def corrupt(result, commands):
        bad = commands[:, 0, 0] == 2.
        if kind == 'nan_future':
            result.predicted_q_path[bad, 5, 4] = float('nan')
        elif kind == 'inf_future':
            result.predicted_q_path[bad, 5, 4] = float('inf')
        elif kind == 'wrong_source':
            result.predicted_q_path[bad, 0, 4] = 1.
        else:
            values = dict(nan_sigma=float('nan'), inf_sigma=float('inf'),
                          zero_sigma=0., negative_sigma=-1.)
            result.q_uncertainty[bad, 4, 4] = values[kind]

    model = visual_model(HalfGainLeWM(corrupt))
    pool = candidates(1., 2.)
    costs = model.goal_cost(rollout(model, pool), torch.zeros(1, 1, 1))
    assert costs[0, 0] < INADMISSIBLE_COST
    assert costs[0, 1] >= INADMISSIBLE_COST
    selected, _, admitted = model.rank_admitted(pool)
    assert admitted == 1
    torch.testing.assert_close(selected, pool[:, 0])
    torch.testing.assert_close(model.backups, pool[0, :1])


@pytest.mark.parametrize('kind', ['missing_path', 'short_path', 'integer_path', 'missing_sigma'])
def test_malformed_batched_forecast_never_returns_a_candidate(kind):
    def corrupt(result, _commands):
        if kind == 'missing_path':
            del result.predicted_q_path
        elif kind == 'short_path':
            result.predicted_q_path = result.predicted_q_path[:, 1:]
        elif kind == 'integer_path':
            result.predicted_q_path = result.predicted_q_path.long()
        elif kind == 'missing_sigma':
            del result.q_uncertainty

    model = visual_model(HalfGainLeWM(corrupt))
    try:
        selected, cost, admitted = model.rank_admitted(candidates(1., 2.))
    except (ValueError, RuntimeError):
        pass  # A whole-batch contract failure may be rejected at the call boundary.
    else:
        assert selected is None and cost is None and admitted == 0
    assert model.backups is None


@pytest.mark.parametrize('rejected_x, expected_command', [(.5, 2.), (2., 1.)])
def test_physical_gate_checks_both_commands_and_predicted_paths(rejected_x, expected_command):
    checks = []

    def physical_check(q, commands):
        checks.append(commands.clone())
        assert torch.equal(q, torch.zeros_like(q))
        if torch.all(commands[..., 0] == rejected_x):
            raise RuntimeError('disallowed physical pose')

    model = visual_model(HalfGainLeWM(), physical_check)
    selected, _, admitted = model.rank_admitted(candidates(1., 2.))
    assert admitted == 1
    assert torch.all(selected[..., 0] == expected_command)
    # Prediction .5 is checked independently from commands 1 and 2.
    if rejected_x == .5:
        assert any(torch.all(checked[..., 0] == .5) for checked in checks)
    assert any(torch.all(checked[..., 0] == 2.) for checked in checks)


def test_actor_response_tracking_compensates_lag_toward_fixed_actor_reference_without_projector():
    lewm = HalfGainLeWM()
    pool = candidates(1., 2.)
    model = V21JointResponseCEM(
        lewm=lewm, q=torch.zeros(1, 6), history=empty_history(1),
        dt=torch.full((1, 5), .1), physical_check=lambda _q, _u: None,
        objective='actor_response_tracking', reference_commands=pool[:, 0],
    )
    selected, _, admitted = model.rank_admitted(pool)
    assert admitted == 2
    # The Actor fixes the desired trajectory at 1. Half-gain response requires
    # command=2; simply returning the Actor proposal would leave response=.5.
    torch.testing.assert_close(selected, pool[:, 1])
    reordered, _, _ = model.rank_admitted(pool.flip(1))
    torch.testing.assert_close(reordered, selected)
    assert lewm.calls


def test_explicit_actor_response_plan_runs_cem_without_visual_transport(monkeypatch):
    def forbidden_fallback(*_args, **_kwargs):
        raise AssertionError('explicit LeWM objective must not use Actor fallback ordering')

    monkeypatch.setattr('fcpg.v21_lewm_cem.prefer_actor_if_visual_flat', forbidden_fallback)
    monkeypatch.setattr('fcpg.v21_lewm_cem.CEM_POPULATION', 8)
    monkeypatch.setattr('fcpg.v21_lewm_cem.CEM_ELITES', 2)
    monkeypatch.setattr('fcpg.v21_lewm_cem.CEM_ITERS', 2)
    lewm = HalfGainLeWM()
    state = SimpleNamespace(q=torch.zeros(1, 6), history=empty_history(1))
    selected, record = plan_commands(
        lewm, state, candidates(1.)[:, 0], torch.full((1, 5), .1),
        lambda _q, _u: None, objective='actor_response_tracking',
        generator=torch.Generator().manual_seed(7),
    )
    assert selected.shape == (1, 5, 6) and torch.isfinite(selected).all()
    assert record['action_source'] == 'v21_lewm_cem'
    assert record['used_actor_fallback'] is False
    assert record['used_hold_fallback'] is False
    assert any(call[0].shape[0] > 1 for call in lewm.calls)
