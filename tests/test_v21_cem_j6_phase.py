"""J6 phase/direction checks for response CEM; CPU only, no device IO."""
from types import SimpleNamespace

import pytest
import torch

from fcpg import v21_lewm_cem as cem
from fcpg.prod_v20_policy import empty_history


def identity_lewm(q, trace, history, offsets):
    del history, offsets
    return SimpleNamespace(
        predicted_q_path=torch.cat((q[:, None], trace.positions), 1),
        q_uncertainty=torch.ones_like(trace.positions),
    )


def actor_reference(source, delta):
    q = torch.tensor([[0., -90., 85., 60., -50., source]])
    commands = q[:, None].expand(-1, 5, -1).clone()
    commands[0, :, 5] += torch.linspace(delta / 5., delta, 5)
    return q, commands


@pytest.mark.parametrize('source,delta,phase,direction', [
    (30., -2., 'closing', -1),
    (5., 2., 'opening', 1),
    (30., .25, 'hold', 0),
])
def test_phase_name_matches_so101_j6_calibrated_direction(source, delta, phase, direction):
    q, actor = actor_reference(source, delta)
    contract = cem.j6_phase_contract(q, actor)
    assert contract['phase'] == phase and contract['direction'] == direction
    assert contract['calibrated_direction'] == {'positive': 'opening', 'negative': 'closing'}


def test_actor_reference_that_crosses_gripper_phases_is_rejected():
    q, actor = actor_reference(30., -2.)
    actor[0, :, 5] = q[0, 5] + torch.tensor([1., .5, -.5, -1., -2.])
    with pytest.raises(ValueError, match='opposite gripper phase|terminal progress'):
        cem.j6_phase_contract(q, actor)


def test_small_initial_gripper_twitch_is_recorded_but_does_not_change_phase():
    q, actor = actor_reference(31.143232, -6.431339)
    actor[0, :, 5] = q[0, 5] + torch.tensor([
        0.353918, 0.527351, -0.495932, -3.416918, -6.431339])
    contract = cem.j6_phase_contract(q, actor)
    assert contract['phase'] == 'closing'
    assert contract['direction'] == -1
    assert contract['mixed_phase_prefix'] is True
    assert contract['first_directional_index'] == 2
    assert contract['initial_opposite_jitter_max'] == pytest.approx(0.527351)


def test_scaled_real_trace_uses_terminal_phase_and_keeps_candidate_gate():
    q = torch.tensor([[0.07225434, -97.282829, 98.757767, 61.91275, -51.584415,
                       31.143232]])
    actor = q[:, None].expand(-1, 5, -1).clone()
    actor[0, :, 5] += torch.tensor([0.329710, 0.531166, 0.583483, 0.267176, -0.201286])

    contract = cem.j6_phase_contract(q, actor, action_scale=3.0)

    assert contract['phase'] == 'closing'
    assert contract['direction'] == -1
    assert contract['first_directional_index'] == 4
    assert contract['initial_opposite_jitter_max'] == pytest.approx(0.583483, abs=1e-5)
    assert contract['phase_inference_motion'] == pytest.approx(1 / 12)
    assert contract['minimum_phase_motion'] == pytest.approx(cem.MIN_DIRECTIONAL_RESPONSE)


def test_unscaled_subthreshold_terminal_trace_remains_hold():
    q = torch.tensor([[0., -90., 85., 60., -50., 31.143232]])
    actor = q[:, None].expand(-1, 5, -1).clone()
    actor[0, :, 5] += torch.tensor([0.329710, 0.531166, 0.583483, 0.267176, -0.201286])
    contract = cem.j6_phase_contract(q, actor, action_scale=1.0)
    assert contract['phase'] == 'hold' and contract['direction'] == 0


def test_phase_reversal_after_directional_progress_is_still_rejected():
    q, actor = actor_reference(30., -2.)
    actor[0, :, 5] = q[0, 5] + torch.tensor([-.2, -.8, -1.2, .4, -2.])
    with pytest.raises(ValueError, match='opposite gripper phase'):
        cem.j6_phase_contract(q, actor)


@pytest.mark.parametrize('source,delta,phase', [
    (30., -1., 'closing'),
    (5., 1., 'opening'),
    (30., 0., 'hold'),
])
def test_response_cem_never_inverts_or_invents_actor_gripper_phase(
        monkeypatch, source, delta, phase):
    monkeypatch.setattr(cem, 'CEM_POPULATION', 6)
    monkeypatch.setattr(cem, 'CEM_ELITES', 2)
    monkeypatch.setattr(cem, 'CEM_ITERS', 2)
    q, actor = actor_reference(source, delta)
    state = SimpleNamespace(q=q, history=empty_history(1))
    checked = []

    def physical_check(_q, commands):
        checked.append(commands.detach().clone())

    selected, receipt = cem.plan_commands(
        identity_lewm, state, actor, torch.full((1, 5), .1), physical_check,
        objective='actor_response_tracking',
        candidate_parameterization='amplified_actor_sequence', action_scale=3.,
        generator=torch.Generator().manual_seed(19),
    )
    contract = receipt['j6_phase_contract']
    assert contract['phase'] == phase
    if phase == 'hold':
        assert torch.equal(selected[0, :, 5], q[0, 5].expand(5))
    else:
        direction = contract['direction']
        signed = (selected[0, :, 5] - q[0, 5]) * direction
        assert (signed >= 0).all() and signed[-1] > cem.MIN_DIRECTIONAL_RESPONSE
    assert checked
    for commands in checked:
        assert cem.j6_phase_admitted(q, commands, contract)


def test_candidate_cannot_claim_closing_response_from_an_opening_command():
    q, actor = actor_reference(30., -2.)
    contract = cem.j6_phase_contract(q, actor)
    wrong = q[:, None].expand(-1, 5, -1).clone()
    wrong[0, :, 5] += torch.linspace(.2, 1., 5)
    assert not cem.j6_phase_admitted(q, wrong, contract)
    closing_then_return = q[:, None].expand(-1, 5, -1).clone()
    closing_then_return[0, :, 5] -= torch.tensor([1., 2., 2., 1., 0.])
    assert not cem.j6_phase_admitted(q, closing_then_return, contract)


def test_amplified_reference_uses_raw_phase_and_clamps_scaled_twitch(monkeypatch):
    monkeypatch.setattr(cem, 'CEM_POPULATION', 6)
    monkeypatch.setattr(cem, 'CEM_ELITES', 2)
    monkeypatch.setattr(cem, 'CEM_ITERS', 2)
    q, actor = actor_reference(31.143232, -6.431339)
    actor[0, :, 5] = q[0, 5] + torch.tensor([
        0.353918, 0.527351, -0.495932, -3.416918, -6.431339])
    state = SimpleNamespace(q=q, history=empty_history(1))

    def physical_check(_q, _commands):
        return None

    selected, receipt = cem.plan_commands(
        identity_lewm, state, actor, torch.full((1, 5), .1), physical_check,
        objective='actor_response_tracking',
        candidate_parameterization='amplified_actor_sequence', action_scale=3.,
        generator=torch.Generator().manual_seed(23),
    )
    correction = receipt['j6_phase_contract']['reference_correction']
    assert correction['applied'] is True
    assert correction['reason'] == 'clamped_transformed_initial_opposite_jitter'
    assert correction['raw_reference'][0][5] > q[0, 5].item()
    assert correction['corrected_reference'][0][5] == pytest.approx(q[0, 5].item())
    signed = (selected[0, :, 5] - q[0, 5]) * -1
    assert (signed >= 0).all() and signed[-1] > cem.MIN_DIRECTIONAL_RESPONSE
