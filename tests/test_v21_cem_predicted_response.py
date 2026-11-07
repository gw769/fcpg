from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fcpg import v21_lewm_cem as cem
from fcpg.prod_v20_policy import empty_history
from fcpg.v21_column_dino_transport import require_actual_cem_response, CEMActualResponseError


def test_forecast_definition_matches_actual_pure_criterion_for_identical_numbers():
    rng = np.random.default_rng(31)
    q = rng.normal(size=(100, 6)).astype(np.float32)
    commands = q[:, None] + rng.normal(size=(100, 5, 6)).astype(np.float32)
    predicted = np.concatenate([q[:, None], q[:, None]+rng.normal(scale=.3,size=(100,5,6)).astype(np.float32)],axis=1)
    diagnostic = cem.predicted_response_admission(torch.tensor(q),torch.tensor(commands),torch.tensor(predicted))
    for i in range(len(q)):
        try:
            require_actual_cem_response(dict(source_q_position=q[i], commands_position=commands[i],
                measured_q_path=predicted[i]), minimum_motion=.25)
            expected = True
        except CEMActualResponseError as error:
            expected = False
            np.testing.assert_array_equal(diagnostic['driven_arm_axes'][i], error.diagnostic['driven_axes'])
            np.testing.assert_array_equal(diagnostic['commanded_peak_delta'][i], error.diagnostic['commanded_peak_delta'])
            np.testing.assert_array_equal(diagnostic['predicted_directional_peak'][i], error.diagnostic['achieved_directional_motion'])
        assert bool(diagnostic['admitted'][i]) == expected
    assert diagnostic['predictions_used_as_actual_evidence'] is False


def test_j4_like_failure_rejected_at_the_strict_existing_threshold():
    q = torch.zeros(1,6)
    commands = torch.zeros(1,5,6);commands[:,:,3] = -.7104
    predicted = torch.zeros(1,6,6);predicted[:,1:,3] = -.171539
    assert not cem.predicted_response_admission(q,commands,predicted)['admitted'][0]
    predicted[:,1:,3] = -.25
    assert not cem.predicted_response_admission(q,commands,predicted)['admitted'][0]
    predicted[:,1:,3] = -.251
    assert cem.predicted_response_admission(q,commands,predicted)['admitted'][0]
    predicted[:,1:,3] = .7  # opposite response never satisfies requested direction
    assert not cem.predicted_response_admission(q,commands,predicted)['admitted'][0]


def test_command_peak_and_first_tie_define_direction_not_final_goal():
    q = torch.zeros(1,6)
    commands = torch.zeros(1,5,6);commands[0,:,2] = torch.tensor([-1.,1.,.2,.2,.2])
    predicted = torch.zeros(1,6,6);predicted[:,1:,2] = .8
    result = cem.predicted_response_admission(q,commands,predicted)
    assert result['commanded_peak_delta'][0,2] == -1 and not result['admitted'][0]


def test_only_arm_axes_and_material_commands_require_response():
    q = torch.zeros(1,6)
    commands = torch.zeros(1,5,6);commands[:,:,5] = 4.;commands[:,:,3] = .25
    assert cem.predicted_response_admission(q,commands,torch.zeros(1,6,6))['admitted'][0]
    commands[:,:,3] = .251
    assert not cem.predicted_response_admission(q,commands,torch.zeros(1,6,6))['admitted'][0]


def test_six_axis_forecast_rejects_static_j6_that_legacy_five_axis_accepts():
    q = torch.zeros(1,6)
    commands = torch.zeros(1,5,6);commands[:,:,5] = 4.
    predicted = torch.zeros(1,6,6)
    legacy = cem.predicted_response_admission(
        q,commands,predicted,checked_joint_count=5)
    amplified = cem.predicted_response_admission(
        q,commands,predicted,checked_joint_count=6)
    assert legacy['admitted'][0]
    assert not amplified['admitted'][0]
    assert amplified['checked_joint_count'] == 6
    assert amplified['driven_joint_axes'][0].tolist() == [False,False,False,False,False,True]
    assert amplified['blocked_joint_axes'][0].tolist() == [False,False,False,False,False,True]


def test_default_and_explicit_five_joint_checks_remain_compatible_when_only_j6_is_commanded():
    result=dict(source_q_position=[0.]*6,commands_position=[[0.,0.,0.,0.,0.,4.]]*5,
                measured_q_path=[[0.]*6]*6)
    implicit=require_actual_cem_response(result,minimum_motion=.25)
    explicit=require_actual_cem_response(result,minimum_motion=.25,checked_joint_count=5)
    np.testing.assert_array_equal(implicit,np.zeros(5))
    np.testing.assert_array_equal(explicit,implicit)


def test_six_joint_check_rejects_materially_commanded_unmoved_j6_with_joint_diagnostic():
    result=dict(source_q_position=[0.]*6,commands_position=[[0.,0.,0.,0.,0.,4.]]*5,
                measured_q_path=[[0.]*6]*6)
    with pytest.raises(CEMActualResponseError,match='actual joint response; J6:') as caught:
        require_actual_cem_response(result,minimum_motion=.25,checked_joint_count=6)
    diagnostic=caught.value.diagnostic
    assert diagnostic['checked_joint_count']==6
    assert diagnostic['driven_axes']==[False,False,False,False,False,True]
    assert diagnostic['blocked_joint_indices']==[5]
    assert diagnostic['commanded_peak_delta']==[0.,0.,0.,0.,0.,4.]
    assert diagnostic['achieved_directional_motion']==[0.]*6
    assert diagnostic['actual_endpoint_delta']==[0.]*6
    assert not any('arm' in key for key in diagnostic)


def test_six_joint_check_returns_all_six_actual_deltas_after_directional_j6_response():
    measured=np.zeros((6,6));measured[2:,5]=.3
    result=dict(source_q_position=[0.]*6,commands_position=[[0.,0.,0.,0.,0.,4.]]*5,
                measured_q_path=measured.tolist())
    np.testing.assert_array_equal(
        require_actual_cem_response(result,minimum_motion=.25,checked_joint_count=6),
        [0.,0.,0.,0.,0.,.3])


@pytest.mark.parametrize('count',[True,0,4,7,5.0,None])
def test_checked_joint_count_must_be_explicitly_five_or_six(count):
    result=dict(source_q_position=[0.]*6,commands_position=[[0.]*6]*5,
                measured_q_path=[[0.]*6]*6)
    with pytest.raises(ValueError,match='checked_joint_count'):
        require_actual_cem_response(result,minimum_motion=.25,checked_joint_count=count)


def test_invalid_forecast_never_passes_as_zero_response():
    q = torch.zeros(1,6);commands = torch.zeros(1,5,6);predicted=torch.zeros(1,6,6)
    predicted[:,1:,2] = float('nan')
    assert not cem.predicted_response_admission(q,commands,predicted)['admitted'][0]


def test_new_mode_stops_before_returning_any_unresponsive_candidate(monkeypatch):
    monkeypatch.setattr(cem,'CEM_POPULATION',6);monkeypatch.setattr(cem,'CEM_ELITES',2);monkeypatch.setattr(cem,'CEM_ITERS',2)
    q = torch.zeros(1,6);actor=torch.ones(1,5,6)
    pool = actor[:,None].expand(1,14,5,6).clone()
    monkeypatch.setattr(cem,'_candidate_pool',lambda *args:pool)
    def model(source, command, history, query):
        return SimpleNamespace(predicted_q_path=source[:,None].expand(-1,6,6).clone(),
                               q_uncertainty=torch.ones_like(command.positions))
    with pytest.raises(RuntimeError,match='predicted directional response admission'):
        cem.plan_commands(model,SimpleNamespace(q=q,history=empty_history(1)),actor,
            torch.full((1,5),.1),lambda q,u:None,objective='actor_response_tracking',
            candidate_parameterization='scaled_actor_endpoint_hold',generator=torch.Generator().manual_seed(20))
    # Historical baseline remains an explicit comparison, not silently modified.
    _, receipt=cem.plan_commands(model,SimpleNamespace(q=q,history=empty_history(1)),actor,
        torch.full((1,5),.1),lambda q,u:None,objective='actor_response_tracking',
        candidate_parameterization='bounded_endpoint_then_hold',generator=torch.Generator().manual_seed(20))
    assert 'response_forecast_admission' not in receipt
