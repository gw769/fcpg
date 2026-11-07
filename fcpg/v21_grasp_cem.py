"""CEM selects open-jaw alignment/approach using two measured visual costs.

Search parameters are five arm endpoints, independent of Actor direction.
Each endpoint expands to five smooth commands followed by a hold through 40.
LeWM predicts all 40 actual responses; only the first five commands can leave
this planner. This is local alignment/approach, never a grasp/contact detector.
"""
import numpy as np
import torch

from .lewm_planning import GroundedCEMPlanner
from .v21_cem_trace import record_cem_event
from .v21_encoder import DIM, PATCHES
from .v21_execution_geometry import J5_WRAP_MIN, J5_WRAP_MAX
from .v21_grasp_cost import WristGraspCost
from .v21_lewm_cem import (INADMISSIBLE_COST, MAX_DELTA, MAX_STEP,
    _PlanningBudget, _SuccessfulSingleAxisStallCache, passes_single_axis_stall,
    predicted_response_admission, repeat_history)
from .v21_models import CommandTrace
from .v21_wrist_guard import require_wrist_path


class GraspCEMModel:
    def __init__(self, lewm, state, task, physical_check, *, lower, upper,
                 clearance, trace=None, planning_budget_s=8.):
        if not isinstance(task, WristGraspCost):
            raise TypeError('CEM grasp planning requires the measured two-term cost')
        self.lewm, self.state, self.task = lewm, state, task
        self.q = state.q
        if self.q.shape != (1, 6) or not torch.isfinite(self.q).all():
            raise ValueError('one finite factual joint source required')
        self.trace, self.clearance = trace, clearance
        self.budget = _PlanningBudget(planning_budget_s, trace=trace)
        self.physical_check = physical_check
        self.gate = _SuccessfulSingleAxisStallCache(physical_check)
        q = self.q[0].detach().cpu().numpy()
        source = task.evaluate(np.repeat(q[None], 41, axis=0))
        self.alignment_first = source['angle_error_deg'][0] > task.alignment_tolerance_deg
        r, o = task.response, task.observation
        lower, upper = np.asarray(lower, np.float64), np.asarray(upper, np.float64)
        if lower.shape != (6,) or upper.shape != (6,) or not np.isfinite([lower, upper]).all() or (lower >= upper).any():
            raise ValueError('finite physical joint bounds required')
        bound = np.asarray(MAX_DELTA[:5])
        self.low = np.maximum.reduce((lower[:5], q[:5]-bound,
            r.reference_q[:5]-r.observed_radius, o.q_position[:5]-r.measured_support))
        self.high = np.minimum.reduce((upper[:5], q[:5]+bound,
            r.reference_q[:5]+r.observed_radius, o.q_position[:5]+r.measured_support))
        self.low[4] = max(self.low[4], J5_WRAP_MIN)
        self.high[4] = min(self.high[4], J5_WRAP_MAX)
        if (self.low >= self.high).any() or (q[:5] < self.low).any() or (q[:5] > self.high).any():
            raise ValueError('current pose is outside measured/local/wrist-wrap support')
        self.source_clearance = float(clearance(q))
        if not np.isfinite(self.source_clearance):
            raise ValueError('current clearance is unavailable')
        if self.alignment_first and self.source_clearance < 20.:
            raise ValueError('alignment rotation needs clearance before CEM; retreat from contact first')
        self._commands = None
        self.evaluated = 0

    def expand(self, endpoints):
        # Endpoint-only parameterization keeps actual motor timing explicit.
        q = self.q.to(endpoints)[0]
        fraction = (torch.arange(1, 41, device=endpoints.device, dtype=endpoints.dtype) / 5).clamp(max=1)
        commands = q[None, None, :].expand(len(endpoints), 40, 6).clone()
        commands[:, :, :5] += fraction[None, :, None] * (endpoints[:, None, :] - q[:5])
        return commands

    def constrain(self, endpoints):
        result = endpoints.clone()
        if self.alignment_first:
            result[..., :4] = self.q[0, :4].to(result)
        return result

    def path_admitted(self, path, *, command=False):
        q = self.q[0].detach().cpu().numpy()
        rows = path.detach().cpu().numpy()
        if not np.isfinite(rows).all():
            return False
        try:
            require_wrist_path(q, rows, max_delta=MAX_DELTA[4] if command else None,
                               max_step=MAX_STEP[4] if command else None)
        except ValueError:
            return False
        if self.alignment_first:
            tolerance = 1e-5 if command else .25
            if (np.abs(rows[:, :4] - q[:4]) > tolerance).any():
                return False
            heights = np.asarray([self.clearance(row) for row in rows])
            if not np.isfinite(heights).all() or (heights < self.source_clearance - .5).any():
                return False
        return True

    def rollout_embeddings(self, history_embedding, history_action, future_action):
        del history_embedding, history_action
        if future_action.ndim != 4 or future_action.shape[0] != 1 or future_action.shape[2:] != (1, 5):
            raise ValueError('grasp CEM expects one endpoint per candidate')
        self.budget.checkpoint('candidate_expansion')
        commands = self.expand(future_action[0, :, 0])
        self._commands = commands
        path = commands.new_full((1, len(commands), 41, 6), float('nan'))
        valid = []
        for i, command in enumerate(commands):
            self.budget.checkpoint('command_admission')
            if self.path_admitted(command[:5], command=True) and self.gate(self.q, command[None, :5]):
                valid.append(i)
        for start in range(0, len(valid), 8):
            self.budget.checkpoint('lewm40')
            ids = valid[start:start+8]
            u = commands[ids]
            q = self.q.to(u).expand(len(ids), -1)
            dt = u.new_full((len(ids), 40), .1)
            query = dt.double().cumsum(-1)
            offsets = torch.cat((query.new_zeros(len(ids), 1), query[:, :-1]), -1)
            # policy() intentionally describes the five writes on hardware.
            # Build a separate, explicitly planned native40 search trace.
            planned = CommandTrace(u, offsets, torch.ones_like(offsets, dtype=torch.bool), 'planned', True)
            features = getattr(getattr(self.state, 'memory', None), 'features', None)
            scene = {}
            if isinstance(features, torch.Tensor) and tuple(features.shape[1:]) == (2, 3, PATCHES, DIM):   # V21_ENCODER
                scene['patches'] = features.to(u).expand(len(ids), -1, -1, -1, -1)
                scene['patch_valid'] = self.state.memory.valid.to(u.device).expand(len(ids), -1, -1)
            response = self.lewm(q, planned,
                repeat_history(self.state.history, len(ids)).to(u.device), query, **scene)
            pred, sigma = response.predicted_q_path, response.q_uncertainty
            if pred.shape != (len(ids), 41, 6) or sigma.shape != (len(ids), 40, 6):
                raise ValueError('native40 LeWM response required; no repeated five-step previews')
            admitted = predicted_response_admission(q, u[:, :5], pred[:, :6], checked_joint_count=6)['admitted']
            finite = torch.isfinite(pred).all((1, 2)) & torch.isfinite(sigma).all((1, 2)) & (sigma > 0).all((1, 2))
            for row, index in enumerate(ids):
                if (bool(admitted[row] & finite[row])
                        and self.path_admitted(pred[row, 1:6])
                        and self.gate(self.q, pred[row:row+1, 1:6])):
                    path[0, index] = pred[row].to(path)
            self.evaluated += len(ids)
            record_cem_event(self.trace, 'grasp_lewm40_batch', indices=ids, commands=u,
                             predicted=pred, uncertainty=sigma, prefix_response_admitted=admitted)
        return path

    def evaluate(self, path):
        result = self.task.evaluate(path.detach().cpu().numpy())
        # Receding-horizon progress must already occur in the executed prefix.
        if self.alignment_first:
            if max(result['angle_error_deg'][1:6]) > result['angle_error_deg'][0] + .25:
                raise ValueError('first five responses worsen alignment')
            if result['distance_px'][5] > result['distance_px'][0] + 2.:
                raise ValueError('first five responses retreat from target')
        elif max(result['angle_error_deg'][1:6]) > self.task.alignment_tolerance_deg:
            raise ValueError('approach loses line alignment')
        return result

    def goal_cost(self, predicted_embedding, goal_embedding):
        del goal_embedding
        cost = predicted_embedding.new_full(predicted_embedding.shape[:2], INADMISSIBLE_COST)
        components = []
        for i, path in enumerate(predicted_embedding[0]):
            try:
                row = self.evaluate(path)
                cost[0, i] = row['cost']
                components.append(row)
            except ValueError as error:
                components.append(dict(rejected=str(error)))
        record_cem_event(self.trace, 'grasp_candidate_scores', commands=self._commands,
                         predictions=predicted_embedding, components=components, costs=cost)
        return cost


@torch.no_grad()
def plan_grasp_commands(lewm, state, actor_commands, dt, physical_check, *,
                        task, lower, upper, clearance, generator=None, trace=None,
                        population=24, elites=6, iterations=3, planning_budget_s=8.):
    """Native40 forecast and two costs, with exactly five returned commands."""
    if actor_commands.shape != (1, 40, 6) or not torch.isfinite(actor_commands).all():
        raise ValueError('grasp CEM requires the native40 Actor proposal')
    if dt.shape != (1, 5) or not torch.isfinite(dt).all() or not torch.allclose(dt, torch.full_like(dt, .1)):
        raise ValueError('execution remains five commands on the existing 0.1s grid')
    if trace is not None:
        trace.begin(actor_commands, generator, actor_commands=actor_commands, q=state.q,
                    objective='wrist_distance_alignment', visual_task=task.receipt())
    failure = None
    try:
        model = GraspCEMModel(lewm, state, task, physical_check, lower=lower, upper=upper,
                              clearance=clearance, trace=trace, planning_budget_s=planning_budget_s)
        seed = actor_commands[:, 4:5, :5].clone()
        seed = seed.clamp(seed.new_tensor(model.low), seed.new_tensor(model.high))
        seed = model.constrain(seed)
        planner = GroundedCEMPlanner(horizon=1, action_dim=5, num_samples=population,
            num_elites=elites, iterations=iterations, action_low=tuple(model.low), action_high=tuple(model.high))
        dummy = seed.new_zeros(1, 1, 1)
        plan = planner.plan(model, dummy, seed.new_zeros(1, 1, 5), dummy,
            initial_mean=seed, generator=generator, constrain=model.constrain, trace=trace,
            admissible=lambda u, p, c: c < INADMISSIBLE_COST)
        hold = state.q[:, None, :5].to(seed)
        pool = torch.cat((hold[:, None], plan.action_sequence[:, None],
                          plan.retained_candidates[0][None]), dim=1)
        path = model.rollout_embeddings(dummy, seed.new_zeros(1, 1, 5), pool)
        cost = model.goal_cost(path, dummy)[0]
        valid = torch.nonzero(cost < INADMISSIBLE_COST).flatten()
        if not len(valid) or not bool(cost[0] < INADMISSIBLE_COST):
            raise RuntimeError('no supported CEM trajectory and hold reference')
        evaluated = [model.evaluate(path[0, i]) for i in valid]
        angular_spread = np.ptp([r['angle_error_deg'][5] for r in evaluated])
        distance_spread = np.ptp([r['distance_px'][5] for r in evaluated])
        if (model.alignment_first and angular_spread < .25) or max(angular_spread, distance_spread) < .25:
            raise RuntimeError('LeWM visual candidate predictions are insensitive')
        winner = int(cost.argmin())
        hold_score = model.evaluate(path[0, 0])
        selected_score = model.evaluate(path[0, winner])
        if winner == 0 or float(cost[winner]) >= float(cost[0]) - 1e-6:
            raise RuntimeError('CEM found no improvement over predicted hold')
        if model.alignment_first and selected_score['angle_error_deg'][5] >= hold_score['angle_error_deg'][5] - .25:
            raise RuntimeError('CEM alignment improvement is only beyond the executed five steps')
        selected = model.expand(pool[0, winner, 0][None])[:, :5].contiguous()
        # Final raw checks do not rely on search-time caches.
        if not (passes_single_axis_stall(physical_check, state.q, selected)
                and passes_single_axis_stall(physical_check, state.q, path[:, winner, 1:6])):
            raise RuntimeError('selected grasp CEM prefix failed final physical audit')
        if (torch.diff(torch.cat((state.q[:, None], selected), 1), dim=1).abs() > selected.new_tensor(MAX_STEP) + 1e-6).any():
            raise RuntimeError('selected grasp prefix exceeds the existing per-step limit')
        model.budget.checkpoint('complete')
        receipt = dict(objective='wrist_distance_alignment', cem_bypassed=False,
            ranking_state_source='lewm_predicted_readback', task_visual_optimization=True,
            inferred_steps=40, cem_forecast_steps=40, execute_steps=5,
            small_step_extension_enabled=False, candidate_parameterization='five_step_endpoint_then_hold_to_40',
            phase='align_open_jaw' if model.alignment_first else 'approach_open_jaw',
            visual_task_cost=task.receipt(), selected_scores=selected_score, hold_scores=hold_score,
            predicted_q_path40=path[0, winner].detach().cpu().tolist(),
            evaluated_candidates=model.evaluated, candidate_angle_spread_deg=float(angular_spread),
            candidate_distance_spread_px=float(distance_spread),
            physical_audit_scope='executed_prefix_and_single_arm_axis_stalls',
            physical_alignment_verified=False, physical_grasp_verified=False)
        record_cem_event(trace, 'selected', commands=selected, receipt=receipt)
        return selected, receipt
    except BaseException as error:
        failure = error
        raise
    finally:
        if trace is not None:
            trace.finish(actor_commands, generator, error=failure)
