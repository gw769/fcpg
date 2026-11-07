from types import SimpleNamespace

import pytest
import torch

from fcpg import v21_lewm_cem as cem
from fcpg.prod_v20_policy import DEFAULT_LOWER, DEFAULT_UPPER, empty_history


def test_common_scale_preserves_shape_and_feedback_that_clipping_erases():
    q = torch.zeros(1, 6)
    actor = torch.zeros(1, 5, 6)
    actor[0, :, 1] = torch.tensor([8., 16., 24., 32., 40.])
    actor[0, :, 2] = torch.tensor([-8., -16., -24., -32., -38.])
    changed = actor.clone()
    changed[..., 2] += torch.tensor([.1, .2, .3, .4, .5])
    old, _ = cem.prepare_actor_reference(actor, q, 'bounded_endpoint_then_hold')
    old_changed, _ = cem.prepare_actor_reference(changed, q, 'bounded_endpoint_then_hold')
    assert torch.equal(old, old_changed)
    value, meta = cem.prepare_actor_reference(actor, q, 'scaled_actor_sequence')
    other, _ = cem.prepare_actor_reference(changed, q, 'scaled_actor_sequence')
    assert meta['alpha'] == 1.0 and meta['shared_alpha'] is False
    torch.testing.assert_close(value, actor)
    assert (other-value).abs().max() > .03
    assert not torch.equal(value[:, 1], value[:, -1])
    assert meta['time_interpolated'] is False and meta['dynamics_reachability_verified'] is False


def test_scaled_sequence_keeps_existing_limits_for_random_poses_and_reversals():
    generator = torch.Generator().manual_seed(91)
    lo, hi = torch.tensor(DEFAULT_LOWER), torch.tensor(DEFAULT_UPPER)
    for _ in range(100):
        q = (lo + (hi-lo) * torch.rand(1, 6, generator=generator))
        actor = q[:, None] + 50 * torch.randn(1, 5, 6, generator=generator)
        value, meta = cem.prepare_actor_reference(actor, q, 'scaled_actor_sequence')
        assert meta['alpha'] == 1.0 and meta['shared_alpha'] is False
        assert (value >= lo-1e-4).all() and (value <= hi+1e-4).all()
        torch.testing.assert_close(value, actor.clamp(lo, hi))


def test_new_reference_is_not_a_claim_to_preserve_radial_magnitude():
    q = torch.zeros(1, 6)
    actor = torch.ones(1, 5, 6)*40
    first, _ = cem.prepare_actor_reference(actor, q, 'scaled_actor_sequence')
    second, _ = cem.prepare_actor_reference(actor*2, q, 'scaled_actor_sequence')
    torch.testing.assert_close(first, actor)
    torch.testing.assert_close(second, actor*2)


def test_old_reference_modes_are_bitwise_unchanged():
    q = torch.tensor([[0., -80., 90., 50., 0., 30.]])
    actor = q[:, None] + torch.arange(30).reshape(1, 5, 6).float()
    lo, hi = cem._physical_box(actor, q)
    limited = cem._limit_horizon(actor[0], q[0], actor.new_tensor(cem.MAX_STEP), lo, hi)[None]
    sequential, _ = cem.prepare_actor_reference(actor, q, 'five_command_sequence')
    assert torch.equal(limited, sequential)
    expected = cem._limit_horizon(limited[0, -1:].expand(5, -1), q[0], actor.new_tensor(cem.MAX_STEP), lo, hi)[None]
    held, _ = cem.prepare_actor_reference(actor, q, 'bounded_endpoint_then_hold')
    assert torch.equal(held, expected)


def test_new_mode_still_rejects_unsafe_geometry(monkeypatch):
    monkeypatch.setattr(cem, 'CEM_POPULATION', 6)
    monkeypatch.setattr(cem, 'CEM_ELITES', 2)
    monkeypatch.setattr(cem, 'CEM_ITERS', 2)
    state = SimpleNamespace(q=torch.zeros(1, 6), history=empty_history(1))
    def model(*args):
        raise AssertionError('should reject before forecasting')
    def unsafe(q, u):
        raise RuntimeError('fixture unsafe geometry')
    with pytest.raises(RuntimeError, match='no physically admissible seed'):
        cem.plan_commands(model, state, torch.ones(1, 5, 6)*40, torch.full((1, 5), .1), unsafe,
            objective='actor_response_tracking', candidate_parameterization='scaled_actor_sequence',
            generator=torch.Generator().manual_seed(19))
