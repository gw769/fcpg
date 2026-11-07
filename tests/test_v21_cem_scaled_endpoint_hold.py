from types import SimpleNamespace

import pytest
import torch

from fcpg import v21_lewm_cem as cem
from fcpg.prod_v20_policy import DEFAULT_LOWER, DEFAULT_UPPER, empty_history
from fcpg.v21_cem_trace import CEMTrace


MODE = 'scaled_actor_endpoint_hold'


def assert_endpoint_hold(q, commands):
    """Commands [B,N,5,6] must approach their fixed endpoint without reversal."""
    source = q[:, None, None].expand(commands.shape[0], commands.shape[1], 1, 6)
    steps = torch.diff(torch.cat((source, commands), 2), dim=2)
    direction = (commands[:, :, -1:] - source).sign()
    assert (steps * direction >= -1e-5).all()
    assert (steps.abs() <= commands.new_tensor(cem.MAX_STEP)+1e-4).all()
    assert ((commands-source).abs() <= commands.new_tensor(cem.MAX_DELTA)+1e-4).all()
    # The permitted endpoint box is only two existing steps from q0.
    torch.testing.assert_close(commands[:, :, 1:], commands[:, :, -1:].expand_as(commands[:, :, 1:]), atol=1e-5, rtol=0)


def test_scaled_endpoint_matches_scaled_sequence_terminal_and_preserves_nonradial_feedback():
    q = torch.zeros(1, 6)
    actor = torch.zeros(1, 5, 6)
    actor[0, :, 1] = torch.tensor([8., 16., 24., 32., 40.])
    actor[0, :, 2] = torch.tensor([-8., -16., -24., -32., -38.])
    alternative = actor.clone()
    alternative[..., 2] += torch.tensor([.1, .2, .3, .4, .5])
    sequence, sequence_meta = cem.prepare_actor_reference(actor, q, 'scaled_actor_sequence')
    held, meta = cem.prepare_actor_reference(actor, q, MODE)
    altered, _ = cem.prepare_actor_reference(alternative, q, MODE)
    assert meta['alpha'] == sequence_meta['alpha']
    assert torch.equal(held[:, -1], sequence[:, -1])
    assert meta['endpoint_hold'] and not meta['actor_time_shape_preserved']
    assert (held-altered).abs().max() > .03
    assert_endpoint_hold(q, held[:, None])


def test_random_actor_reversals_never_survive_endpoint_reference():
    generator = torch.Generator().manual_seed(44)
    lo, hi = torch.tensor(DEFAULT_LOWER), torch.tensor(DEFAULT_UPPER)
    for _ in range(100):
        q = lo + (hi-lo) * torch.rand(1, 6, generator=generator)
        actor = q[:, None] + 50 * torch.randn(1, 5, 6, generator=generator)
        reference, _ = cem.prepare_actor_reference(actor, q, MODE)
        assert_endpoint_hold(q, reference[:, None])
        assert (reference >= lo-1e-4).all() and (reference <= hi+1e-4).all()


@pytest.mark.parametrize('objective', ['actor_response_tracking', 'actor_command_tracking'])
def test_every_evaluated_candidate_is_monotone_then_held(monkeypatch, objective):
    monkeypatch.setattr(cem, 'CEM_POPULATION', 6)
    monkeypatch.setattr(cem, 'CEM_ELITES', 2)
    monkeypatch.setattr(cem, 'CEM_ITERS', 2)
    q = torch.zeros(1, 6)
    actor = torch.tensor([[[1., 40., -38., 6., .1, 1.]]] * 5).reshape(1, 5, 6)
    def model(source, command, history, query):
        assert_endpoint_hold(source[:1], command.positions[None])
        return SimpleNamespace(predicted_q_path=torch.cat((source[:, None], command.positions), 1),
            q_uncertainty=torch.ones_like(command.positions))
    def geometry(source, commands):
        if (commands[..., 0] < -.1).any():
            raise RuntimeError('geometry rejection fixture to exercise replacement sampling')
    trace = CEMTrace()
    selected, receipt = cem.plan_commands(model, SimpleNamespace(q=q, history=empty_history(1)), actor,
        torch.full((1, 5), .1), geometry, objective=objective,
        candidate_parameterization=MODE, generator=torch.Generator().manual_seed(20), trace=trace)
    assert_endpoint_hold(q, selected[:, None])
    for event in trace.payload()['events']:
        values = event['values']
        if event['event'] == 'cem_iteration':
            assert_endpoint_hold(q, values['candidates'])
        if event['event'] == 'ranked_pool_evaluation':
            assert_endpoint_hold(q, values['commands'])
    assert receipt['candidate_parameterization'] == MODE
    assert receipt['reference_transform']['endpoint_hold']
    assert receipt['max_delta'] == list(cem.MAX_DELTA)
    assert receipt['max_step'] == list(cem.MAX_STEP)
