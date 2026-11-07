"""Admission must cover the measured J2-moves/J3-stalls failure pattern."""
import torch
import pytest
from types import SimpleNamespace

from fcpg.v21_lewm_cem import passes_physical, passes_single_axis_stall
from fcpg.prod_v20_policy import empty_history


def table_check(q, commands):
    # Toy clearance: J2 descends, J3 extension compensates. Both moving is
    # safe, but descent alone crosses the table. No robot fixtures are used.
    if (commands[..., 1] + commands[..., 2] > .1).any():
        raise RuntimeError('table clearance')


def test_synchronous_safe_descent_is_rejected_if_j3_stalls():
    q = torch.zeros(1, 6)
    commands = torch.zeros(1, 5, 6)
    commands[..., 1] = 1.
    commands[..., 2] = -1.
    assert passes_physical(table_check, q, commands)
    assert not passes_single_axis_stall(table_check, q, commands)


def test_independently_safe_motion_survives_any_single_axis_stall():
    q = torch.zeros(1, 6)
    commands = torch.zeros(1, 5, 6)
    commands[..., 1] = -1.
    commands[..., 2] = -1.
    before = commands.clone()
    assert passes_single_axis_stall(table_check, q, commands)
    assert torch.equal(commands, before) and torch.equal(q, torch.zeros_like(q))


@pytest.mark.parametrize('objective', ['actor_command_tracking', 'actor_response_tracking'])
def test_both_tracking_objectives_apply_stall_guard_before_cem_search(monkeypatch, objective):
    from fcpg import v21_lewm_cem as cem
    q = torch.zeros(1, 6)
    proposal = q[:, None].expand(1, 5, 6).clone()
    proposal[..., 1] = 1.
    proposal[..., 2] = -1.
    assert passes_physical(table_check, q, proposal)

    def inspect_seed(candidate, source, gate, *_args):
        gate(source, candidate)
        pytest.fail('response ranking bypassed the single-axis stall guard')

    monkeypatch.setattr(cem, '_legal_seed', inspect_seed)
    with pytest.raises(RuntimeError, match='single arm-axis tracking stall'):
        cem.plan_commands(None, SimpleNamespace(q=q, history=empty_history(1)),
                          proposal, torch.full((1, 5), .1), table_check,
                          objective=objective)
