import numpy as np
import pytest
import torch
from types import SimpleNamespace

from fcpg.v21_column_dino_transport import require_actual_cem_response, require_actual_probe_response
from fcpg.prod_v20_policy import empty_history
from fcpg.v21_lewm_cem import INADMISSIBLE_COST, V21JointResponseCEM, plan_commands


def identity_lewm(q, trace, history, offsets, *, patches=None, patch_valid=None):    # V21LeWM.forward's keywords
    return SimpleNamespace(predicted_q_path=torch.cat((q[:, None], trace.positions), 1),
                           q_uncertainty=torch.ones_like(trace.positions))


def test_out_of_envelope_rollout_does_not_discard_supported_candidates():
    """The local DINO response may reject one rollout without being extrapolated."""
    model = V21JointResponseCEM(
        lewm=identity_lewm, q=torch.zeros(1, 6), history=empty_history(1),
        dt=torch.full((1, 5), .1), physical_check=lambda _q, _u: None,
        project=lambda q: (_ for _ in ()).throw(ValueError('outside measured envelope'))
        if float(q[0]) > 1.0 else np.asarray((q[0], 0.0)),
        target_px=(1.0, 0.0), observed_midpoint_px=(0.0, 0.0),
        capture_q=np.zeros(6), require_candidate_sensitive_visual=True,
    )
    commands = torch.zeros(1, 3, 5, 6)
    commands[0, 1, :, 0] = .5
    commands[0, 2, :, 0] = 2.0
    selected, cost, admitted = model.rank_admitted(commands)
    assert selected is not None and cost is not None
    assert admitted == 2
    assert model.backups.shape == (2, 5, 6)
    assert float(model.backups[..., 0].max()) <= 1.0
    assert model.visual_supported_candidates == 2
    assert model.candidate_spread_px == .5
    assert model.visual_transport_reason is None


def _strict_model(project, physical_check=lambda _q, _u: None):
    return V21JointResponseCEM(
        lewm=identity_lewm, q=torch.zeros(1, 6), history=empty_history(1),
        dt=torch.full((1, 5), .1), physical_check=physical_check,
        project=project, target_px=(1.0, 0.0), observed_midpoint_px=(0.0, 0.0),
        capture_q=np.zeros(6), require_candidate_sensitive_visual=True,
    )


@pytest.mark.parametrize('project, reason', [
    (None, 'missing_exposure_paired_column_dino_fg_transport'),
    (lambda _q: np.zeros(2), 'candidate_insensitive_column_dino_fg_transport'),
    (lambda _q: np.full(2, np.nan), 'insufficient_in_envelope_column_dino_fg_candidates'),
])
def test_required_visual_rejects_unrankable_pool_and_clears_backups(project, reason):
    model = _strict_model(project)
    model.backups = torch.ones(2, 5, 6)
    commands = torch.zeros(1, 2, 5, 6)
    commands[0, 1, :, 0] = .5
    assert model.rank_admitted(commands) == (None, None, 0)
    assert model.backups is None
    assert model.visual_transport_reason == reason


def test_visual_spread_excludes_physically_rejected_candidates():
    def physical_check(_q, commands):
        if float(commands[..., 0].max()) > .1:
            raise RuntimeError('physical gate')

    model = _strict_model(lambda q: np.asarray(q[:2]), physical_check)
    commands = torch.zeros(1, 2, 5, 6)
    commands[0, 1, :, 0] = .5
    assert model.rank_admitted(commands) == (None, None, 0)
    assert model.visual_supported_candidates == 1
    assert model.visual_transport_reason == 'insufficient_in_envelope_column_dino_fg_candidates'
    assert model.backups is None


@pytest.mark.parametrize('bad_cost', [INADMISSIBLE_COST, float('nan'), float('inf')])
def test_invalid_cost_never_enters_backup_or_visual_spread(bad_cost):
    model = _strict_model(lambda q: np.asarray(q[:2]))
    prefix_cost = model._prefix_cost
    model._prefix_cost = lambda q: q.new_tensor(bad_cost) if float(q[0]) > .75 else prefix_cost(q)
    commands = torch.zeros(1, 3, 5, 6)
    commands[0, 1, :, 0] = .5
    commands[0, 2, :, 0] = 1.
    selected, cost, admitted = model.rank_admitted(commands)
    assert selected is not None and np.isfinite(cost)
    assert admitted == model.visual_supported_candidates == 2
    assert model.candidate_spread_px == .5
    assert float(model.backups[..., 0].max()) == .5


def test_nonfinite_command_cannot_enter_backups_even_if_projection_ignores_axis():
    model = _strict_model(lambda q: np.asarray(q[:2]))
    commands = torch.zeros(1, 3, 5, 6)
    commands[0, 1, :, 0] = .5
    commands[0, 2, :, 0] = 1.
    commands[0, 2, :, 4] = float('nan')
    _, _, admitted = model.rank_admitted(commands)
    assert admitted == model.visual_supported_candidates == 2
    assert torch.isfinite(model.backups).all()


@pytest.mark.parametrize('project, reason', [
    (None, 'missing_exposure_paired_column_dino_fg_transport'),
    (lambda _q: np.zeros(2), 'candidate_insensitive_column_dino_fg_transport'),
    (lambda _q: np.full(2, np.nan), 'insufficient_in_envelope_column_dino_fg_candidates'),
])
def test_strict_plan_never_returns_actor_fallback_or_calls_lewm_without_visual_evidence(
        monkeypatch, project, reason):
    def forbidden(*_args):
        raise AssertionError('unrankable visual pool must not reach Actor fallback or LeWM')

    monkeypatch.setattr('fcpg.v21_lewm_cem.prefer_actor_if_visual_flat', forbidden)
    q = torch.zeros(1, 6)
    state = SimpleNamespace(q=q, history=empty_history(1))
    with pytest.raises(RuntimeError, match=reason):
        plan_commands(
            identity_lewm, state, torch.zeros(1, 5, 6), torch.full((1, 5), .1),
            lambda _q, _u: None, project=project, target_px=(1., 0.),
            observed_midpoint_px=(0., 0.), capture_q=np.zeros(6),
            require_candidate_sensitive_visual=True,
            generator=torch.Generator().manual_seed(0),
        )


def test_commanded_but_unmoved_probe_is_not_an_actual_dino_response():
    result = dict(source_q_position=[0.] * 6, measured_q_path=[[0.] * 6, [0., 0., 0., 0., 0., 0.]])
    try:
        require_actual_probe_response(result, 2, -1, minimum_motion=.25)
    except ValueError as exc:
        assert 'did not achieve' in str(exc)
    else:
        raise AssertionError('unmoved probe must not be accepted')


def test_commanded_but_unmoved_cem_axis_is_not_a_completed_feedback_transition():
    result = dict(
        source_q_position=[0.] * 6,
        commands_position=[[0., 0., -2., 0., 0., 0.]] * 5,
        measured_q_path=[[0.] * 6, [0.] * 6],
    )
    try:
        require_actual_cem_response(result, minimum_motion=.25)
    except ValueError as exc:
        assert 'did not achieve' in str(exc)
    else:
        raise AssertionError('unmoved CEM elbow must not be accepted')


def test_cem_actual_response_accepts_a_directional_not_exact_endpoint():
    result = dict(
        source_q_position=[0.] * 6,
        commands_position=[[1., 0., -2., 0., 0., 0.]] * 5,
        measured_q_path=[[0.] * 6, [.7, 0., -1.2, 0., 0., 0.]],
    )
    np.testing.assert_allclose(
        require_actual_cem_response(result, minimum_motion=.25),
        [.7, 0., -1.2, 0., 0.],
    )
