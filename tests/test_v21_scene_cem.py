from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fcpg.prod_v20_policy import empty_history
from fcpg.v21_memory import VisualMemory
from fcpg.v21_lewm_cem import plan_commands
from fcpg.v21_scene_cem import SceneGoal, V21SceneGoalCEM, endpoint_chunks, plan_scene_commands
from fcpg.v21_scene_lewm import V21SceneLeWM


class ToySceneEncoder(nn.Module):
    def forward(self, patches, valid):
        return torch.where(valid[..., None], patches[..., 0, :], torch.zeros_like(patches[..., 0, :]))


class ToySceneLeWM(nn.Module):
    """Analytic test only: scene x is J1; NOT a trained model or robot result."""
    horizon = 40
    def __init__(self):
        super().__init__()
        self.scene_encoder = ToySceneEncoder()
        self.register_buffer('position_lower', torch.full((6,), -4.))
        self.register_buffer('position_upper', torch.full((6,), 4.))
        self.register_buffer('increment_scale', torch.full((6,), .2))
        self.calls = 0

    def forward(self, q, trace, history, offsets, **kwargs):
        self.calls += 1
        command = trace.active_at(offsets, q)
        scene = command[..., :1, None].expand(-1, -1, 2, 384)
        return SimpleNamespace(predicted_q_path=torch.cat((q[:, None], command), 1),
            q_uncertainty=torch.ones_like(command), predicted_scene=scene)


def case():
    q = torch.zeros(1, 6)
    memory = VisualMemory.empty(1)
    memory.valid[:, :, -1] = True
    state = SimpleNamespace(q=q, history=empty_history(1), memory=memory,
                            time=torch.zeros(1, dtype=torch.float64))
    dt = torch.full((1, 40), .1, dtype=torch.float64)
    return ToySceneLeWM().eval(), state, dt


def goal(value, valid=None):
    return SceneGoal(torch.full((1, 2, 256, 384), float(value)),
        torch.ones(1, 2, dtype=torch.bool) if valid is None else valid, reference='synthetic-test-only')


def test_goal_change_reverses_ranking_without_changing_world_model_prediction():
    lewm, state, dt = case()
    adapter = V21SceneGoalCEM(lewm, state, dt, lambda q, u: None, goal(1))
    commands = torch.zeros(1, 2, 40, 6)
    commands[0, 0, :, 0], commands[0, 1, :, 0] = 1., -1.
    predicted = adapter.rollout_embeddings(torch.zeros(1, 1, 1), torch.zeros(1, 1, 6), commands)
    before = predicted.clone()
    positive = adapter.goal_cost(predicted, adapter.encode_goal(goal(1)))
    negative = adapter.goal_cost(predicted, adapter.encode_goal(goal(-1)))
    assert positive.argmin() == 0 and negative.argmin() == 1
    assert torch.equal(predicted, before)
    assert lewm.calls == 1  # goals change scores, never dynamics inputs


def test_missing_goal_view_is_masked_and_no_goal_is_rejected():
    lewm, state, dt = case()
    only_wrist = goal(1, torch.tensor([[True, False]]))
    only_wrist.features[:, 1] = float('nan')
    adapter = V21SceneGoalCEM(lewm, state, dt, lambda q, u: None, only_wrist)
    candidate = torch.zeros(1, 1, 40, 6)
    predicted = adapter.rollout_embeddings(torch.zeros(1, 1, 1), torch.zeros(1, 1, 6), candidate)
    assert torch.isfinite(adapter.goal_cost(predicted, adapter.goal_embedding)).all()
    with pytest.raises(ValueError, match='view'):
        V21SceneGoalCEM(lewm, state, dt, lambda q, u: None,
                       goal(1, torch.zeros(1, 2, dtype=torch.bool)))


def test_planner_returns_exact_selected_prefix_and_never_mutates_history():
    lewm, state, dt = case()
    history = state.memory.features.clone()
    actor = torch.zeros(1, 40, 6)
    selected, receipt = plan_scene_commands(lewm, state, actor, dt, lambda q, u: None,
        scene_goal=goal(1), population=32, elites=8, iterations=3,
        generator=torch.Generator().manual_seed(5), planning_budget_s=30)
    assert selected.shape == (1, 5, 6)
    assert receipt['prediction_horizon'] == 40 and receipt['execution_horizon'] == 5
    assert receipt['scene_goal_used'] and not receipt['physical_success_verified']
    torch.testing.assert_close(selected[0], torch.tensor(receipt['selected_commands40'])[:5])
    assert torch.equal(history, state.memory.features)
    assert receipt['cost'] < receipt['actor_scene_cost']


def test_unsafe_best_scene_cannot_be_selected_and_all_unsafe_is_not_actor_fallback():
    lewm, state, dt = case()
    def checker(q, commands):
        if (commands[..., 0] > .5).any():
            raise ValueError('physical limit')
    adapter = V21SceneGoalCEM(lewm, state, dt, checker, goal(1))
    commands = torch.zeros(1, 2, 40, 6)
    commands[0, 0, :, 0], commands[0, 1, :, 0] = 1., .3
    predicted = adapter.rollout_embeddings(torch.zeros(1, 1, 1), torch.zeros(1, 1, 6), commands)
    cost = adapter.goal_cost(predicted, adapter.goal_embedding)
    assert not torch.isfinite(cost[0, 0]) and cost.argmin() == 1
    def reject(q, commands):
        raise ValueError('physical limit')
    with pytest.raises(RuntimeError, match='admissible'):
        plan_scene_commands(lewm, state, torch.zeros(1, 40, 6), dt, reject,
            scene_goal=goal(1), population=4, elites=2, iterations=1, planning_budget_s=30)


def test_native40_is_required_not_a_repeated_five_step_forecast():
    lewm, state, dt = case()
    with pytest.raises(ValueError, match='40'):
        plan_scene_commands(lewm, state, torch.zeros(1, 5, 6), dt[:, :5], lambda q, u: None,
                            scene_goal=goal(1))


def test_public_planner_routes_scene_goal_explicitly_and_does_not_ignore_it():
    lewm, state, dt = case()
    actor = torch.zeros(1, 40, 6)
    selected, report = plan_commands(lewm, state, actor, dt, lambda q, u: None,
        objective='scene_goal', scene_goal=goal(1), planning_budget_s=30,
        generator=torch.Generator().manual_seed(5))
    assert selected.shape == (1, 5, 6) and report['scene_goal_used']
    with pytest.raises(ValueError, match='objective=scene_goal'):
        plan_commands(lewm, state, actor[:, :5], dt[:, :5], lambda q, u: None,
                      objective='actor_response_tracking', scene_goal=goal(1))
    with pytest.raises(ValueError, match='legacy'):
        plan_commands(lewm, state, actor, dt, lambda q, u: None,
            objective='scene_goal', scene_goal=goal(1), target_px=[10, 20])


@pytest.mark.parametrize('normalization', ['batchnorm', 'layernorm'])
def test_real_scene_model_keeps_capture_state_clock_and_full_patch_bank(normalization):
    torch.set_num_threads(2)
    _, state, dt = case()
    model = V21SceneLeWM(normalization=normalization).eval()
    observed = []
    handle = model.register_forward_pre_hook(lambda module, args, kwargs: observed.append(kwargs),
                                              with_kwargs=True)
    state.time.fill_(1.)
    state.memory.image_times.fill_(.5)
    state.memory.q_at_image.fill_(.3)
    adapter = V21SceneGoalCEM(model, state, dt, lambda q, u: None, goal(1))
    # Concurrent updates to the caller must not change an in-flight plan context.
    state.memory.q_at_image.fill_(9.)
    state.memory.features.fill_(7.)
    response = adapter.rollout_embeddings(state.q[:, None], state.q[:, None], torch.zeros(1, 2, 40, 6))
    handle.remove()
    assert response.shape == (1, 2, 40, 2, 384) and torch.isfinite(response).all()
    assert len(observed) == 1 and observed[0]['patches'].shape == (2, 2, 3, 256, 384)
    assert observed[0]['now_time'].tolist() == [1., 1.]
    assert (observed[0]['memory'].q_at_image == .3).all()
    assert not observed[0]['patches'].any()


def test_training_mode_and_future_current_observation_are_rejected():
    lewm, state, dt = case()
    lewm.train()
    with pytest.raises(ValueError, match='eval'):
        V21SceneGoalCEM(lewm, state, dt, lambda q, u: None, goal(1))
    lewm.eval()
    state.memory.image_times.fill_(10.)
    with pytest.raises(ValueError, match='causal'):
        V21SceneGoalCEM(lewm, state, dt, lambda q, u: None, goal(1))


def test_chunk_parameterization_is_linear_continuous_and_preserves_endpoints():
    q = torch.ones(1, 6) * 10
    candidate = torch.randn(1, 3, 40, 6)
    chunks = endpoint_chunks(candidate, q)
    torch.testing.assert_close(chunks[:, :, 4::5], candidate[:, :, 4::5])
    previous = torch.cat((q[:, None, None].expand(1, 3, 1, 6), chunks[:, :, :-1]), 2)
    increments = (chunks - previous).reshape(1, 3, 8, 5, 6)
    torch.testing.assert_close(increments, increments[:, :, :, :1].expand_as(increments), atol=2e-6, rtol=1e-5)
