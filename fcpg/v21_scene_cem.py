"""Opt-in, offline-first native40 CEM using learned visual goal distance.

The goal is encoded by the checkpoint's observation-only scene encoder. It
affects scoring, never the dynamics inputs or actual observation history.
This module does not import a robot driver, issue commands, or select phases.
"""
from dataclasses import dataclass, fields, replace

import torch

from .lewm_planning import GroundedCEMPlanner
from .v21_cem_trace import record_cem_event
from .v21_horizon_precision import compute_context
from .v21_lewm_cem import _PlanningBudget, passes_physical, repeat_history
from .v21_models import CommandTrace


@dataclass(frozen=True)
class SceneGoal:
    features: torch.Tensor  # [1,2,256,384], frozen DINO features of a goal image pair
    valid: torch.Tensor  # [1,2], wrist then column
    reference: str
    view_order: tuple[str, str] = ('wrist', 'column')


def _snapshot(value):
    return replace(value, **{f.name: getattr(value, f.name).detach().clone()
        for f in fields(value) if isinstance(getattr(value, f.name), torch.Tensor)})


def _repeat_memory(memory, count):
    return replace(memory, **{f.name: getattr(memory, f.name).expand(count, *getattr(memory, f.name).shape[1:])
        for f in fields(memory) if isinstance(getattr(memory, f.name), torch.Tensor)})


def endpoint_chunks(candidates, q):
    """Eight linear five-step chunks, constructed BEFORE any LeWM scoring.

    CEM varies eight endpoints rather than forty unrelated writes. The selected
    chunk is never smoothed, scaled or rewritten after it was scored.
    """
    endpoints = candidates[:, :, 4::5]
    starts = torch.cat((q[:, None, None].expand(-1, candidates.shape[1], 1, -1),
                        endpoints[:, :, :-1]), dim=2)
    fraction = torch.arange(1, 6, device=candidates.device, dtype=candidates.dtype) / 5
    commands = starts[..., None, :] + fraction[None, None, None, :, None] * (
        endpoints - starts)[..., None, :]
    return commands.flatten(2, 3)


class V21SceneGoalCEM:
    """One frozen factual context; batched, action-conditioned native40 forecasts."""

    def __init__(self, lewm, state, dt, physical_check, scene_goal, *,
                 precision='fp32', batch_size=8, budget=None, trace=None, scene_value=None):
        if getattr(lewm, 'horizon', None) != 40 or not hasattr(lewm, 'scene_encoder'):
            raise ValueError('scene_goal requires a learned-scene native40 LeWM checkpoint')
        if any(m.training for m in lewm.modules()):
            raise ValueError('scene_goal planning requires eval mode, including its scene encoder')
        if state.q.shape != (1, 6) or not torch.isfinite(state.q).all():
            raise ValueError('scene_goal requires a single finite actual q [1,6]')
        if (dt.shape != (1, 40) or dt.dtype != torch.float64
                or not torch.isfinite(dt).all() or not (dt > 0).all()):
            raise ValueError('scene_goal requires 40 positive float64 time intervals')
        if not callable(physical_check) or batch_size < 1:
            raise ValueError('a physical checker and positive inference batch size are required')
        self.lewm, self.physical_check = lewm, physical_check
        if scene_value is not None and any(m.training for m in scene_value.modules()):
            raise ValueError('scene_value planning requires an eval-mode scorer')
        self.scene_value = scene_value
        self.q, self.time = state.q.detach().clone(), state.time.detach().clone()
        self.memory, self.history = _snapshot(state.memory), _snapshot(state.history)
        if (self.memory.features.shape != (1, 2, 3, 256, 384)
                or self.memory.valid.shape != (1, 2, 3) or self.memory.valid.dtype != torch.bool):
            raise ValueError('scene_goal requires the complete dual-view three-frame visual bank')
        if (self.time.shape != (1,) or self.time.dtype != torch.float64
                or not torch.isfinite(self.time).all()
                or not torch.isfinite(self.memory.image_times[self.memory.valid]).all()
                or (self.memory.image_times[self.memory.valid] > self.time[0] + 1e-9).any()):
            raise ValueError('scene_goal context must contain only causal actual exposures')
        # Mask absent exposures, not real evidence; leave the caller's memory untouched.
        self.memory = replace(self.memory, features=torch.where(
            self.memory.valid[..., None, None], self.memory.features, 0.))
        if not torch.isfinite(self.memory.features).all():
            raise ValueError('nonfinite actual visual evidence')
        self.dt = dt.detach().clone().to(self.q.device)
        self.query_offsets = self.dt.cumsum(1)
        self.command_offsets = torch.cat((self.dt.new_zeros(1, 1), self.query_offsets[:, :-1]), 1)
        self.precision, self.batch_size = precision, int(batch_size)
        compute_context(self.q.device, precision)  # reject unsupported precision before planning
        self.budget, self.trace = budget, trace
        self.goal_reference = scene_goal.reference
        self.goal_embedding = self.encode_goal(scene_goal)
        record_cem_event(trace, 'scene_goal_context', q=self.q, now_time=self.time,
            memory=self.memory, history=self.history, goal=scene_goal,
            encoded_goal=self.goal_embedding, goal_in_dynamics=False)
        self.forward_calls = 0
        self.last_q_paths = self.last_uncertainty = self.last_admitted = None

    def _check_budget(self, stage):
        if self.budget is not None:
            self.budget.checkpoint(stage)

    @torch.no_grad()
    def encode_goal(self, goal):
        if not isinstance(goal, SceneGoal) or goal.view_order != ('wrist', 'column'):
            raise ValueError('scene goal view order must be wrist,column')
        if (goal.features.shape != (1, 2, 256, 384) or goal.valid.shape != (1, 2)
                or goal.valid.dtype != torch.bool or not goal.reference):
            raise ValueError('scene goal requires per-view patches, bool validity and a reference')
        valid = goal.valid.to(self.q.device) & self.memory.valid.any(-1)
        if not valid.any():
            raise ValueError('scene goal and current memory have no common observed view')
        self._check_budget('encode_scene_goal')
        with compute_context(self.q.device, self.precision):
            encoded = self.lewm.scene_encoder(goal.features.to(self.q), valid)
        if encoded.shape != (1, 2, 384) or not torch.isfinite(encoded[valid]).all():
            raise ValueError('scene encoder returned an invalid goal')
        # NaNs carry missing-camera identity with the goal, so changing a goal
        # cannot accidentally reuse the previous goal's camera mask.
        return encoded.float().masked_fill(~valid[..., None], float('nan'))

    def path_admitted(self, commands):
        """Reuse the caller's physical check on each consecutive five-step chunk.

        After chunk one the anchor is hypothetical, not an actual readback.
        These checks do NOT constitute collision calibration or motor evidence.
        """
        if commands.shape != (1, 40, 6) or not torch.isfinite(commands).all():
            return False
        previous = self.q
        for begin in range(0, 40, 5):
            self._check_budget('scene_physical_chunk')
            chunk = commands[:, begin:begin + 5]
            if not passes_physical(self.physical_check, previous, chunk):
                return False
            previous = chunk[:, -1]
        return True

    @torch.no_grad()
    def rollout_embeddings(self, history_embedding, history_action, future_action):
        del history_embedding, history_action  # actual history is the frozen snapshot above
        if future_action.ndim != 4 or future_action.shape[0] != 1 or future_action.shape[2:] != (40, 6):
            raise ValueError('scene_goal candidates must be [1,N,40,6]')
        self._check_budget('scene_rollout_start')
        actions = future_action[0].to(self.q)
        count = len(actions)
        scenes = self.q.new_full((count, 40, 2, 384), float('nan'))
        paths = self.q.new_full((count, 41, 6), float('nan'))
        uncertainty = self.q.new_full((count, 40, 6), float('nan'))
        admitted = torch.zeros(count, device=self.q.device, dtype=torch.bool)
        valid_indices = [i for i in range(count) if self.path_admitted(actions[i:i + 1])]
        for start in range(0, len(valid_indices), self.batch_size):
            indices = valid_indices[start:start + self.batch_size]
            n = len(indices)
            memory = _repeat_memory(self.memory, n)
            command_trace = CommandTrace(actions[indices], self.command_offsets.expand(n, -1),
                torch.ones(n, 40, device=self.q.device, dtype=torch.bool), 'planned', True)
            self._check_budget('native40_scene_forward')
            with compute_context(self.q.device, self.precision):
                response = self.lewm(self.q.expand(n, -1), command_trace,
                    repeat_history(self.history, n), self.query_offsets.expand(n, -1),
                    patches=memory.features, patch_valid=memory.valid,
                    memory=memory, now_time=self.time.expand(n))
            self.forward_calls += 1
            if (response.predicted_q_path.shape != (n, 41, 6)
                    or response.q_uncertainty.shape != (n, 40, 6)
                    or response.predicted_scene.shape != (n, 40, 2, 384)):
                raise ValueError('scene LeWM must produce native40 scenes and anchored q paths')
            paths[indices] = response.predicted_q_path.to(paths)
            uncertainty[indices] = response.q_uncertainty.to(uncertainty)
            for local, index in enumerate(indices):
                valid = (torch.isfinite(response.predicted_scene[local]).all()
                    and torch.equal(paths[index, 0], self.q[0])
                    and torch.isfinite(uncertainty[index]).all()
                    and (uncertainty[index] > 0).all()
                    and self.path_admitted(paths[index:index + 1, 1:]))
                if valid:
                    scenes[index] = response.predicted_scene[local].to(scenes)
                    admitted[index] = True
            self._check_budget('native40_scene_complete')
        self.last_q_paths, self.last_uncertainty, self.last_admitted = paths, uncertainty, admitted
        record_cem_event(self.trace, 'scene_rollout', commands=actions,
            predicted_q_path=paths, q_uncertainty=uncertainty, admitted=admitted,
            goal_in_dynamics=False, actual_memory_updated=False)
        return scenes[None]

    def latent_goal_cost(self, predicted_embedding, goal_embedding):
        if predicted_embedding.ndim != 5 or predicted_embedding.shape[2:] != (40, 2, 384):
            raise ValueError('scene forecasts must be [1,N,40,2,384]')
        if goal_embedding.shape != (1, 2, 384):
            raise ValueError('scene goal embedding must be [1,2,384]')
        valid = torch.isfinite(goal_embedding).all(-1) & self.memory.valid.any(-1)
        if not valid.any():
            raise ValueError('no common finite scene goal view')
        goal = torch.where(valid[..., None], goal_embedding, 0.).float()
        error = (predicted_embedding[:, :, -1].float() - goal[:, None]).square().mean(-1)
        cost = torch.where(valid[:, None], error, 0.).sum(-1) / valid.sum(-1)[:, None]
        finite = torch.isfinite(predicted_embedding).flatten(2).all(-1) & torch.isfinite(cost)
        return cost.masked_fill(~finite, float('inf'))

    def goal_cost(self, predicted_embedding, goal_embedding):
        latent = self.latent_goal_cost(predicted_embedding, goal_embedding)
        if self.scene_value is None:
            return latent
        admitted = torch.isfinite(latent)
        cost = torch.full_like(latent, float('inf'))
        if admitted.any():
            terminal = predicted_embedding[:, :, -1][admitted].float()
            valid = torch.isfinite(goal_embedding).all(-1) & self.memory.valid.any(-1)
            goal = torch.nan_to_num(goal_embedding).expand(len(terminal), -1, -1)
            mask = valid.expand(len(terminal), -1)
            # No tracking, phase, joint-lock or hand-written pixel cost mixed in.
            output = self.scene_value(terminal, goal, mask, mask)
            value = output.distance_to_goal.float()
            cost[admitted] = value.masked_fill(~torch.isfinite(value), float('inf'))
        return cost


@torch.no_grad()
def plan_scene_commands(lewm, state, actor_commands, dt, physical_check, *, scene_goal,
                        population=32, elites=8, iterations=3, generator=None,
                        planning_budget_s=8., planning_clock=None, precision='fp32', trace=None,
                        scene_value=None):
    """Rank native40 plans by learned scene goal; return the EXACT first five.

    Experimental opt-in only. A lower latent cost is not grasp verification or
    permission to replace the deployed phase/assist controller.
    """
    if actor_commands.shape != (1, 40, 6) or not torch.isfinite(actor_commands).all():
        raise ValueError('scene_goal requires a finite native40 Actor proposal [1,40,6]')
    budget = _PlanningBudget(planning_budget_s, clock=planning_clock, trace=trace)
    model = V21SceneGoalCEM(lewm, state, dt, physical_check, scene_goal,
                          precision=precision, budget=budget, trace=trace, scene_value=scene_value)
    actor = actor_commands.detach().clone().to(model.q)
    if scene_value is not None:
        actor = endpoint_chunks(actor[:, None], model.q)[:, 0]
    low, high = lewm.position_lower.to(model.q), lewm.position_upper.to(model.q)
    # Learned data scale controls proposal noise, not a post-selection action
    # multiplier. Normalization bounds are NOT a replacement for physical_check.
    std = lewm.increment_scale.to(model.q)[None, None].expand_as(actor).clone()
    std *= torch.arange(1, 41, device=model.q.device, dtype=model.q.dtype).sqrt()[None, :, None]
    planner = GroundedCEMPlanner(40, 6, num_samples=population, num_elites=elites,
        iterations=iterations, action_low=tuple(low.tolist()), action_high=tuple(high.tolist()))
    history_embedding, history_action = model.q[:, None], model.q[:, None]
    plan = planner.plan(model, history_embedding, history_action, model.goal_embedding,
        initial_mean=actor, initial_std=std, generator=generator, trace=trace,
        constrain=lambda candidates: endpoint_chunks(candidates, model.q))
    # The average of safe candidates need not be safe, or even a good visual
    # plan. Re-evaluate real retained candidates, final mean AND the raw Actor.
    pool = torch.cat((actor, plan.retained_candidates[0], plan.action_sequence), 0)
    if scene_value is not None:
        pool = endpoint_chunks(pool[None], model.q)[0]
    predicted = model.rollout_embeddings(history_embedding, history_action, pool[None])
    costs = model.goal_cost(predicted, model.goal_embedding)[0]
    if not torch.isfinite(costs).any():
        raise RuntimeError('scene_goal found no admissible candidate; no Actor fallback')
    index = int(costs.argmin())
    selected = pool[index:index + 1].clone()
    q_path = model.last_q_paths[index:index + 1]
    if not model.path_admitted(selected) or not model.path_admitted(q_path[:, 1:]):
        raise RuntimeError('scene_goal selected candidate no longer admissible')
    budget.checkpoint('scene_selection_complete')
    latent_cost = model.latent_goal_cost(predicted, model.goal_embedding)[0]
    receipt = dict(objective='scene_value' if scene_value is not None else 'scene_goal', prediction_horizon=40, execution_horizon=5,
        scene_goal_used=True, goal_reference=model.goal_reference,
        cem_candidate_parameterization='native40_endpoint_chunks',
        selected_prefix_modified_after_scoring=False,
        goal_views=torch.isfinite(model.goal_embedding).all(-1)[0].tolist(),
        cost=float(costs[index]), actor_scene_cost=float(costs[0]) if torch.isfinite(costs[0]) else None,
        distance_to_goal=float(costs[index]) if scene_value is not None else None,
        latent_distance_diagnostic=float(latent_cost[index]), latent_distance_weight=0. if scene_value is not None else 1.,
        selected_pool_index=index, selected_source=('actor' if index == 0 else
            'cem_mean' if index == len(pool) - 1 else 'cem_retained'),
        selected_commands40=selected[0].tolist(), selected_predicted_q_path40=q_path[0].tolist(),
        predicted_scene_terminal=predicted[0, index, -1].tolist(),
        candidate_count=len(pool), admissible_count=int(torch.isfinite(costs).sum()),
        native40_forward_calls=model.forward_calls, command_offsets_s=model.command_offsets[0].tolist(),
        query_offsets_s=model.query_offsets[0].tolist(), prediction_is_recursive=False,
        actor_tracking_weight=0., goal_in_dynamics=False, actual_memory_updated=False,
        hardware_used=False, physical_success_verified=False, deployment_approved=False,
        handwritten_phase_logic_applied=False, single_axis_stall_checked=False,
        task_assistance_calls=0, searched_joint_indices=[0, 1, 2, 3, 4, 5],
        physical_check_scope='caller-supplied checker; later chunk anchors are hypothetical',
        planning_budget=budget.receipt())
    record_cem_event(trace, 'scene_selected', receipt=receipt)
    return selected[:, :5].clone(), receipt
