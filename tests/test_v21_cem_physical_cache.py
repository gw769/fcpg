"""Exact plan-local CEM geometry cache contracts; no hardware IO."""
from types import SimpleNamespace

import pytest
import torch

from fcpg import v21_lewm_cem as cem
from fcpg.prod_v20_policy import empty_history


def identity_lewm(q, trace, history, query):
    del history, query
    return SimpleNamespace(predicted_q_path=torch.cat((q[:, None], trace.positions), 1),
                           q_uncertainty=torch.ones_like(trace.positions))


def test_success_cache_is_exact_and_never_caches_a_rejected_gate():
    calls = []
    def physical_check(q, commands):
        calls.append((q.clone(), commands.clone()))
        if bool((commands[..., 0] > 2.).any()):
            raise RuntimeError('unsafe')

    cache = cem._SuccessfulPhysicalCheckCache(physical_check)
    q = torch.zeros(1, 6); allowed = torch.ones(1, 5, 6); rejected = allowed.clone()
    rejected[..., 0] = 3.
    cache(q, allowed)
    cache(q.clone(), allowed.clone())
    assert len(calls) == 1 and cache.receipt()['success_hits'] == 1
    for _ in range(2):
        try:
            cache(q, rejected)
        except RuntimeError:
            pass
    assert len(calls) == 3
    assert cache.receipt()['unique_successes'] == 1


def test_mutating_gate_cannot_become_a_cached_success_on_second_attempt():
    calls = []
    def mutate(_q, commands):
        calls.append(1)
        commands.data.add_(1.)
    cache = cem._SuccessfulPhysicalCheckCache(mutate)
    q, commands = torch.zeros(1, 6), torch.ones(1, 5, 6)
    assert not cem.passes_physical(cache, q, commands)
    assert not cem.passes_physical(cache, q, commands)
    assert len(calls) == 2 and cache.receipt()['success_hits'] == 0
    assert cache.receipt()['unique_successes'] == 0


def test_complete_stall_cache_is_exact_and_never_caches_a_rejection():
    calls = []

    def physical_check(q, commands):
        calls.append((q.clone(), commands.clone()))
        if bool((commands[..., 0] > 2.).any()):
            raise RuntimeError('unsafe')

    cache = cem._SuccessfulSingleAxisStallCache(physical_check)
    q = torch.zeros(1, 6)
    allowed = torch.ones(1, 5, 6)
    rejected = allowed.clone()
    rejected[..., 0] = 3.

    assert cache(q, allowed)
    assert cache(q.clone(), allowed.clone())
    assert len(calls) == 6
    assert cache.receipt()['success_hits'] == 1
    assert not cache(q, rejected)
    assert not cache(q, rejected)
    assert len(calls) == 8
    assert cache.receipt()['unique_successes'] == 1


def _plan(*, cache_enabled, counter):
    q = torch.zeros(1, 6)
    actor = torch.zeros(1, 5, 6)
    actor[0, :, 0] = torch.tensor([1., .25, 1.25, .5, .75])
    def physical_check(_q, _commands):
        counter.append(1)
    return cem.plan_commands(
        identity_lewm, SimpleNamespace(q=q, history=empty_history(1)), actor,
        torch.full((1, 5), .1), physical_check,
        objective='actor_response_tracking',
        candidate_parameterization='bounded_endpoint_then_hold',
        use_physical_check_cache=cache_enabled,
        generator=torch.Generator().manual_seed(19))


def test_cached_and_uncached_cem_have_identical_output_and_gate_semantics(monkeypatch):
    monkeypatch.setattr(cem, 'CEM_POPULATION', 6)
    monkeypatch.setattr(cem, 'CEM_ELITES', 2)
    monkeypatch.setattr(cem, 'CEM_ITERS', 2)
    cached_calls, uncached_calls = [], []
    cached, cached_receipt = _plan(cache_enabled=True, counter=cached_calls)
    uncached, uncached_receipt = _plan(cache_enabled=False, counter=uncached_calls)

    torch.testing.assert_close(cached, uncached)
    cached_stats = cached_receipt.pop('physical_gate_cache')
    uncached_stats = uncached_receipt.pop('physical_gate_cache')
    cached_budget = cached_receipt.pop('planning_budget')
    uncached_budget = uncached_receipt.pop('planning_budget')
    assert cached_receipt == uncached_receipt
    assert cached_stats['enabled'] is True and uncached_stats['enabled'] is False
    assert cached_stats['success_hits'] > 0
    assert cached_stats['full_stall_success_cache']['success_hits'] > 0
    assert len(cached_calls) == (cached_stats['raw_checks']
                                 + cached_receipt['final_physical_audit']['raw_checks'])
    assert len(uncached_calls) == (uncached_stats['raw_checks']
                                   + uncached_receipt['final_physical_audit']['raw_checks'])
    assert len(cached_calls) < len(uncached_calls)
    assert cached_budget['budget_checks'] < uncached_budget['budget_checks']
    assert cached_budget['budget_s'] == uncached_budget['budget_s']
    assert not cached_budget['exhausted'] and not uncached_budget['exhausted']


def test_final_uncached_audit_can_veto_a_search_cache_success(monkeypatch):
    monkeypatch.setattr(cem, 'CEM_POPULATION', 6)
    monkeypatch.setattr(cem, 'CEM_ELITES', 2)
    monkeypatch.setattr(cem, 'CEM_ITERS', 2)
    seen = set()

    def pass_each_exact_path_once(q, commands):
        key = (cem._tensor_key(q), cem._tensor_key(commands))
        if key in seen:
            raise RuntimeError('second raw check rejected')
        seen.add(key)

    q = torch.zeros(1, 6)
    actor = torch.zeros(1, 5, 6)
    actor[0, :, 0] = torch.tensor([1., .25, 1.25, .5, .75])
    with pytest.raises(RuntimeError, match='final uncached physical audit'):
        cem.plan_commands(
            identity_lewm, SimpleNamespace(q=q, history=empty_history(1)), actor,
            torch.full((1, 5), .1), pass_each_exact_path_once,
            objective='actor_response_tracking',
            candidate_parameterization='bounded_endpoint_then_hold',
            use_physical_check_cache=True,
            generator=torch.Generator().manual_seed(19))
