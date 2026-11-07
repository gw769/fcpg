from types import SimpleNamespace

import pytest
import torch

from fcpg import v21_lewm_cem as cem
from fcpg.prod_v20_policy import empty_history
from fcpg.v21_cem_trace import CEMTrace


def _identity_lewm(q, trace, history, query):
    del history, query
    return SimpleNamespace(
        predicted_q_path=torch.cat((q[:, None], trace.positions), 1),
        q_uncertainty=torch.ones_like(trace.positions),
    )


def test_rejected_candidate_is_replaced_by_the_verified_seed():
    q = torch.zeros(1, 6)
    seed = torch.zeros(5, 6)
    candidates = torch.ones(1, 1, 5, 6)
    low, high = torch.full((6,), -2.0), torch.full((6,), 2.0)
    stats = {}

    def only_seed_is_safe(_q, commands):
        if bool(commands.any()):
            raise RuntimeError('unsafe candidate')

    result = cem.constrain_to_physical(
        candidates, physical_check=only_seed_is_safe, q=q, seed=seed,
        low=low, high=high, attempts=0, stats=stats,
    )

    torch.testing.assert_close(result[0, 0], seed)
    assert stats == {'seed_replacements': 1}


def test_replacement_seed_is_revalidated_after_normalization():
    q = torch.zeros(1, 6)
    seed = torch.ones(5, 6)
    candidates = torch.zeros(1, 1, 5, 6)
    low, high = torch.full((6,), -2.0), torch.full((6,), 2.0)

    def only_zero_is_safe(_q, commands):
        if bool(commands.any()):
            raise RuntimeError('unsafe candidate')

    with pytest.raises(RuntimeError, match='replacement seed'):
        cem.constrain_to_physical(
            candidates, physical_check=only_zero_is_safe, q=q, seed=seed,
            low=low, high=high, attempts=0,
        )


def test_planning_budget_expires_at_the_deadline():
    now = [0.0]
    budget = cem._PlanningBudget(1.0, clock=lambda: now[0])
    now[0] = 1.0

    with pytest.raises(cem.CEMPlanningTimeout, match='at complete'):
        budget.checkpoint('complete')


def test_physical_gate_time_is_charged_before_next_candidate(monkeypatch):
    monkeypatch.setattr(cem, 'CEM_POPULATION', 4)
    monkeypatch.setattr(cem, 'CEM_ELITES', 2)
    monkeypatch.setattr(cem, 'CEM_ITERS', 1)
    q = torch.zeros(1, 6)
    actor = torch.zeros(1, 5, 6)
    actor[0, :, 0] = torch.arange(1, 6) * 0.1
    state = SimpleNamespace(q=q, history=empty_history(1))
    now = [0.0]
    calls = [0]

    def physical_check(_q, _commands):
        calls[0] += 1
        now[0] = 1.0

    trace = CEMTrace()
    with pytest.raises(cem.CEMPlanningTimeout, match='physical_gate_complete'):
        cem.plan_commands(
            _identity_lewm, state, actor, torch.full((1, 5), 0.1), physical_check,
            objective='actor_response_tracking',
            candidate_parameterization='bounded_endpoint_then_hold',
            planning_budget_s=1.0, planning_clock=lambda: now[0], trace=trace,
        )

    assert calls == [1]
    assert any(event['event'] == 'planning_budget_exhausted'
               and event['values']['stage'] == 'physical_gate_complete'
               for event in trace.payload()['events'])


def test_planning_budget_aborts_without_returning_a_candidate(monkeypatch):
    monkeypatch.setattr(cem, 'CEM_POPULATION', 4)
    monkeypatch.setattr(cem, 'CEM_ELITES', 2)
    monkeypatch.setattr(cem, 'CEM_ITERS', 2)
    q = torch.zeros(1, 6)
    actor = torch.zeros(1, 5, 6)
    actor[0, :, 0] = torch.arange(1, 6) * 0.1
    state = SimpleNamespace(q=q, history=empty_history(1))
    now = [0.0]

    def clock():
        now[0] += 0.01
        return now[0]

    trace = CEMTrace()
    with pytest.raises(cem.CEMPlanningTimeout, match='CEM planning exceeded'):
        cem.plan_commands(
            _identity_lewm, state, actor, torch.full((1, 5), 0.1),
            lambda _q, _commands: None,
            objective='actor_response_tracking',
            candidate_parameterization='bounded_endpoint_then_hold',
            planning_budget_s=0.025, planning_clock=clock, trace=trace,
        )

    payload = trace.payload()
    assert payload['complete'] is False
    assert 'CEMPlanningTimeout' in payload['error']
    assert any(event['event'] == 'planning_budget_exhausted'
               for event in payload['events'])
