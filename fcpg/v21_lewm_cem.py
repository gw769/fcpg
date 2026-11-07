"""Host-side V21 CEM: reuse GroundedCEMPlanner. LeWM stays goal-free.

actor_command_tracking: FCPG/Actor → CEM ranking → LeWM backup admission.
Other objectives rank LeWM candidate responses. This module does not write motors.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from numbers import Real
import time
import numpy as np
import torch

from .lewm_planning import GroundedCEMPlanner
from .prod_v20_policy import ActualHistory, DEFAULT_LOWER, DEFAULT_UPPER
from .v21_models import CommandTrace
from .v21_cem_trace import CEMTrace, record_cem_event
from .v21_encoder import DIM, PATCHES

ACTION_SOURCE = 'v21_lewm_cem'


def _repeat_visual_memory(memory, count):
    if memory is None or memory.features.shape[0] == count:
        return memory
    return replace(memory,
        features=memory.features.expand(count, -1, -1, -1, -1),
        q_at_image=memory.q_at_image.expand(count, -1, -1, -1),
        image_times=memory.image_times.expand(count, -1, -1),
        valid=memory.valid.expand(count, -1, -1))
CEM_POPULATION = 32
CEM_ELITES = 8
CEM_ITERS = 3
CEM_PLANNING_BUDGET_S = 8.0
CEM_RESAMPLE_ATTEMPTS = 2
LEGAL_SEED_ATTEMPTS = 24
TRACKING_OBJECTIVES = ('actor_command_tracking', 'actor_response_tracking')
PREFIX_PATH_INDEX = 3  # source + three commands
# A sub-quarter-pixel span across the complete ranked pool is indistinguishable
# from a frozen F/G midpoint at the 224px planning resolution.  Do not send a
# command merely because the physical gate accepted it.
MIN_CANDIDATE_MOTION_PX = 0.25
INADMISSIBLE_COST = 1e6
# One five-command chunk. J1–J3 are the arm, J4/J5 the wrists, J6 the gripper.
# Loosened from 3°/2° so a short Actor prefix is not uniformly shrunk to a twitch.
MAX_DELTA = (8.0, 8.0, 8.0, 5.0, 5.0, 10.0)
MAX_STEP = (4.0, 4.0, 4.0, 2.5, 2.5, 5.0)
CLOSED_Q6 = 5.0  # same shut pose as the live close window
MIN_DIRECTIONAL_RESPONSE = 0.25  # unchanged deployed require_actual_cem_response threshold
J6_INDEX = 5
# A real Actor close can begin with a sub-unit opening twitch before the
# closing phase becomes dominant.  This is a bounded prefix allowance, not a
# license for phase reversals after the gripper has started moving.
MAX_INITIAL_OPPOSITE_JITTER = 0.75


class CEMPlanningTimeout(TimeoutError):
    """The bounded planner exhausted its wall-clock budget without a result."""


class _PlanningBudget:
    def __init__(self, seconds, *, clock=None, trace=None):
        if (isinstance(seconds, bool) or not isinstance(seconds, Real)
                or not np.isfinite(seconds) or seconds <= 0):
            raise ValueError('planning_budget_s must be finite and positive')
        self.seconds = float(seconds)
        self.clock = time.monotonic if clock is None else clock
        self.trace = trace
        self.started_s = float(self.clock())
        self.deadline_s = self.started_s + self.seconds
        self.checks = 0

    def checkpoint(self, stage):
        self.checks += 1
        now = float(self.clock())
        if not np.isfinite(now):
            raise CEMPlanningTimeout('CEM planning clock is not finite')
        if now >= self.deadline_s:
            elapsed = now - self.started_s
            record_cem_event(self.trace, 'planning_budget_exhausted', stage=stage,
                elapsed_s=elapsed, budget_s=self.seconds, checks=self.checks)
            raise CEMPlanningTimeout(
                f'CEM planning exceeded {self.seconds:.3f}s at {stage} '
                f'after {self.checks} budget checks')

    def receipt(self):
        return dict(budget_s=self.seconds, budget_checks=self.checks,
                    exhausted=False)


def j6_phase_contract(q, reference, *, minimum_motion=MIN_DIRECTIONAL_RESPONSE,
                      action_scale=1.0):
    """Bind CEM's gripper phase to the Actor reference.

    SO-101 calibrated J6 increases when opening and decreases when closing. A
    response-tracking CEM may vary magnitude, but it may not invent the opposite
    phase or turn an Actor hold into gripper motion.
    """
    if (not isinstance(q, torch.Tensor) or q.shape != (1, 6)
            or not isinstance(reference, torch.Tensor) or reference.shape != (1, 5, 6)
            or not torch.isfinite(q).all() or not torch.isfinite(reference).all()):
        raise ValueError('J6 phase contract requires finite q[1,6] and Actor reference[1,5,6]')
    if not np.isfinite(minimum_motion) or minimum_motion <= 0:
        raise ValueError('J6 phase minimum must be finite and positive')
    action_scale = _validate_action_scale(action_scale)
    if not np.isfinite(MAX_INITIAL_OPPOSITE_JITTER) or MAX_INITIAL_OPPOSITE_JITTER <= 0:
        raise ValueError('J6 initial opposite-jitter limit must be finite and positive')
    delta = (reference[0, :, J6_INDEX] - q[0, J6_INDEX]).double()
    terminal = float(delta[-1])
    peak_index = int(delta.abs().argmax())
    peak = float(delta[peak_index])
    # Infer the phase from the endpoint that Actor actually intends to hold.
    # Amplification can turn a sub-threshold raw endpoint into a meaningful
    # command, so only the inference threshold scales. Candidate admission
    # still requires the unscaled minimum response below.
    inference_motion = float(minimum_motion) / action_scale
    common = dict(joint_index=J6_INDEX, source_position=float(q[0, J6_INDEX]),
        actor_peak_index=peak_index, actor_peak_delta=peak,
        actor_terminal_delta=terminal, minimum_phase_motion=float(minimum_motion),
        phase_inference_motion=inference_motion, action_scale=action_scale,
        initial_opposite_jitter_tolerance=float(MAX_INITIAL_OPPOSITE_JITTER),
        calibrated_direction=dict(positive='opening', negative='closing'),
        candidate_contract='same_source_side_and_terminal_progress')
    if abs(terminal) <= inference_motion:
        return dict(common, phase='hold', direction=0, first_directional_index=None,
                    initial_opposite_jitter_max=0.0, mixed_phase_prefix=False)
    direction = 1 if terminal > 0 else -1
    signed = delta * direction
    directional = signed >= inference_motion
    # Terminal progress above guarantees at least one directional sample.
    first_directional_index = int(torch.nonzero(directional, as_tuple=False)[0].item())
    prefix = signed[:first_directional_index]
    opposite = prefix[prefix < 0]
    opposite_max = float((-opposite).max()) if opposite.numel() else 0.0
    if opposite_max > MAX_INITIAL_OPPOSITE_JITTER:
        raise ValueError('Actor J6 reference crosses into the opposite gripper phase')
    # Once the primary phase has started, any return to the opposite side is a
    # genuine phase reversal and remains a hard rejection.
    if bool((signed[first_directional_index:] < -1e-6).any()):
        raise ValueError('Actor J6 reference crosses into the opposite gripper phase')
    return dict(common, phase='opening' if direction > 0 else 'closing', direction=direction,
                first_directional_index=first_directional_index,
                initial_opposite_jitter_max=opposite_max,
                mixed_phase_prefix=bool(opposite.numel()))


def j6_phase_admitted(q, commands, contract):
    """Whether a candidate preserves the Actor-derived J6 phase."""
    if (not isinstance(q, torch.Tensor) or q.shape != (1, 6)
            or not isinstance(commands, torch.Tensor) or commands.shape != (1, 5, 6)
            or not torch.isfinite(q).all() or not torch.isfinite(commands).all()
            or not isinstance(contract, dict)):
        return False
    delta = commands[0, :, J6_INDEX] - q[0, J6_INDEX].to(commands)
    phase, direction = contract.get('phase'), contract.get('direction')
    minimum = float(contract.get('minimum_phase_motion', MIN_DIRECTIONAL_RESPONSE))
    if phase == 'hold' and direction == 0:
        return bool((delta.abs() <= 1e-6).all())
    if phase not in ('opening', 'closing') or direction not in (-1, 1):
        return False
    signed = delta * int(direction)
    return bool((signed >= -1e-6).all() and signed[-1] > minimum)


def _apply_j6_phase_contract(commands, q, contract):
    if contract is None:
        return commands
    source = q.reshape(6)[J6_INDEX].to(commands)
    if contract['phase'] == 'hold':
        commands[..., J6_INDEX] = source
    elif contract['direction'] > 0:
        commands[..., J6_INDEX] = torch.maximum(commands[..., J6_INDEX], source)
    else:
        commands[..., J6_INDEX] = torch.minimum(commands[..., J6_INDEX], source)
    return commands


def _validate_action_scale(action_scale):
    if (isinstance(action_scale, bool) or not isinstance(action_scale, Real)
            or action_scale not in (1.0, 1.5, 3.0)):
        raise ValueError('action_scale must be 1.0, 1.5, or 3.0')
    return float(action_scale)


def _scaled_max_delta(action_scale):
    action_scale = _validate_action_scale(action_scale)
    return tuple(value * action_scale for value in MAX_DELTA)


def _reachable_max_delta(action_scale):
    # The V21 command horizon stays at five. A larger requested box cannot
    # create a sixth step or imply the commanded displacement was measured.
    return tuple(min(delta, 5 * step) for delta, step in
                 zip(_scaled_max_delta(action_scale), MAX_STEP))


def predicted_response_admission(q, commands, predicted, *,
                                 minimum_motion=MIN_DIRECTIONAL_RESPONSE,
                                 checked_joint_count=5):
    """Apply the existing directional arm-response definition to predictions.

    This is a forecast admission, never actual-readback evidence. The reference
    The default matches require_actual_cem_response: first five axes, command
    peak chosen by maximum absolute displacement (first tie), same-sign response
    peak strictly greater than the unchanged threshold for every driven axis.
    The amplified six-axis mode also checks J6. Only the predicted q0..q5
    horizon exists here; no follow samples are invented.
    """
    if (not isinstance(q, torch.Tensor) or q.ndim != 2 or q.shape[-1] != 6
            or not isinstance(commands, torch.Tensor) or commands.shape != (len(q), 5, 6)
            or not isinstance(predicted, torch.Tensor) or predicted.shape != (len(q), 6, 6)):
        raise ValueError('predicted response admission requires q[B,6], u[B,5,6], prediction[B,6,6]')
    if not np.isfinite(minimum_motion) or minimum_motion <= 0:
        raise ValueError('minimum forecast response must be finite and positive')
    if type(checked_joint_count) is not int or checked_joint_count not in (5, 6):
        raise ValueError('checked_joint_count must be 5 or 6')
    # Float64 matches the original actual-response NumPy arithmetic exactly.
    source, targets, forecast = q.double(), commands.double(), predicted.double()
    valid = (torch.isfinite(source).all(-1) & torch.isfinite(targets).all((1, 2))
        & torch.isfinite(forecast).all((1, 2)) & (forecast[:, 0] == source).all(-1))
    commanded = targets[..., :checked_joint_count]-source[:, None, :checked_joint_count]
    peak_index = commanded.abs().argmax(1)
    peak = commanded.gather(1, peak_index[:, None]).squeeze(1)
    achieved = ((forecast[..., :checked_joint_count]
                 - source[:, None, :checked_joint_count]) * peak.sign()[:, None]).amax(1)
    driven = commanded.abs().amax(1) > float(minimum_motion)
    blocked = driven & (achieved <= float(minimum_motion))
    return dict(admitted=valid & ~blocked.any(-1), valid_input=valid,
        commanded_peak_delta=peak,
        driven_joint_axes=driven, blocked_joint_axes=blocked,
        # Compatibility aliases. In six-axis mode these include J6 as well.
        driven_arm_axes=driven, blocked_arm_axes=blocked,
        predicted_directional_peak=achieved,
        predicted_endpoint_delta=(forecast[:, -1, :checked_joint_count]
                                  - source[:, :checked_joint_count]),
        checked_joint_count=checked_joint_count,
        minimum_motion=float(minimum_motion), response_source='lewm_predicted_readback',
        response_samples='predicted q0 plus five planned readbacks; no invented follow endpoint',
        predictions_used_as_actual_evidence=False)


def passes_physical(physical_check, q, commands):
    if not torch.isfinite(q).all() or not torch.isfinite(commands).all():
        return False
    q_in, u_in = q.detach().clone(), commands.detach().clone()
    try:
        physical_check(q_in, u_in)
    except (RuntimeError, ValueError, TypeError):
        return False
    return bool(torch.equal(q_in, q) and torch.equal(u_in, commands))


def passes_single_axis_stall(physical_check, q, commands):
    """Check nominal motion and each arm joint remaining at its measured source.

    A safe synchronous trajectory is insufficient when another joint moves but
    one servo does not follow. This is an admission stress test, not a forecast
    or a claim to cover every multi-joint tracking failure.
    """
    if not passes_physical(physical_check, q, commands):
        return False
    for joint in range(5):
        stalled = commands.detach().clone()
        stalled[..., joint] = q[..., joint, None]
        if not passes_physical(physical_check, q, stalled):
            return False
    return True


def _tensor_key(value):
    if not isinstance(value, torch.Tensor):
        raise TypeError('physical gate cache requires tensors')
    tensor = value.detach().contiguous()
    # The gate may distinguish dtypes or device placement.  Retain both, and
    # use exact storage bytes rather than rounded numerical values.
    raw = tensor.cpu().view(torch.uint8).numpy().tobytes()
    return (str(tensor.device), str(tensor.dtype), tuple(tensor.shape), raw)


class _SuccessfulPhysicalCheckCache:
    """Plan-local exact cache for successful static geometry admissions only.

    It never spans planning calls, never caches a rejection/exception, and is
    not used by the real dispatch/readback checks.  Every new q/trajectory
    still reaches the original gate, including every stalled-axis trajectory.
    """
    def __init__(self, physical_check, *, enabled=True):
        self.physical_check = physical_check
        self.enabled = bool(enabled)
        self._successful = set()
        self.hits = 0
        self.misses = 0

    def __call__(self, q, commands):
        key = (_tensor_key(q), _tensor_key(commands))
        if self.enabled and key in self._successful:
            self.hits += 1
            return None
        self.misses += 1
        result = self.physical_check(q, commands)
        # passes_physical rejects callbacks that rewrite their inputs. A
        # rewritten input must never turn into a cached success on retry.
        if self.enabled and key == (_tensor_key(q), _tensor_key(commands)):
            self._successful.add(key)
        return result

    def receipt(self):
        return dict(enabled=self.enabled, success_hits=int(self.hits),
                    raw_checks=int(self.misses),
                    unique_successes=int(len(self._successful)),
                    scope='single_cem_plan_exact_q_and_full_command_bytes_only')


class _SuccessfulSingleAxisStallCache:
    """Cache only a complete nominal-plus-five-stalled-axis admission.

    Search evaluates the same command in constraint, rollout, cost and final
    ranking stages. Once all six physical paths passed for exact tensor bytes,
    repeating that complete audit cannot add information. Rejections and
    exceptions are never cached, and the selected plan is checked again with
    the original callback before it can be returned.
    """

    def __init__(self, physical_check, *, enabled=True):
        self.physical_check = physical_check
        self.enabled = bool(enabled)
        self._successful = set()
        self.hits = 0
        self.misses = 0

    def __call__(self, q, commands):
        key = (_tensor_key(q), _tensor_key(commands))
        if self.enabled and key in self._successful:
            self.hits += 1
            return True
        self.misses += 1
        if not passes_single_axis_stall(self.physical_check, q, commands):
            return False
        if self.enabled and key == (_tensor_key(q), _tensor_key(commands)):
            self._successful.add(key)
        return True

    def receipt(self):
        return dict(enabled=self.enabled, success_hits=int(self.hits),
                    complete_audits=int(self.misses),
                    unique_successes=int(len(self._successful)),
                    scope='single_cem_plan_exact_q_and_full_command_bytes_only',
                    audit='nominal_plus_each_of_five_arm_axes_stalled')


def _forecast_clears_gate(model, selected, physical_check):
    forecast = model.forecast(selected)
    pred = getattr(forecast, 'predicted_q_path', None)
    sigma = getattr(forecast, 'q_uncertainty', None)
    # V21LeWM returns the factual source plus five future readbacks. Missing,
    # truncated or unanchored predictions are not a successful physical check.
    if (not isinstance(pred, torch.Tensor) or pred.shape != (1, 6, 6)
            or not pred.is_floating_point() or not torch.isfinite(pred).all()
            or not torch.equal(pred[:, 0].to(model.q), model.q)):
        return False, forecast
    if (not isinstance(sigma, torch.Tensor) or sigma.shape != (1, 5, 6)
            or not sigma.is_floating_point() or not torch.isfinite(sigma).all()
            or (sigma <= 0).any()):
        return False, forecast
    if (getattr(model, 'response_forecast_admission', False)
            and not bool(model.response_admission_mask(selected, pred)[0])):
        return False, forecast
    # This validates the model contract, not uncertainty calibration. No new
    # uncertainty threshold or relaxation of the existing physical gate.
    return passes_physical(physical_check, model.q, pred[:, 1:]), forecast


def pick_after_lewm(model, backups, physical_check):
    """Try every physical backup. First LeWM forecast that still clears the gate wins."""
    if backups is None or backups.numel() == 0:
        raise RuntimeError('no physically admissible backup')
    for index in range(backups.shape[0]):
        selected = backups[index].reshape(1, 5, 6)
        if not passes_physical(physical_check, model.q, selected):
            continue
        ok, forecast = _forecast_clears_gate(model, selected, physical_check)
        if ok:
            return selected, index, forecast
    raise RuntimeError('no physically admissible backup after LeWM')


def prefer_actor_if_visual_flat(backups, actor_commands):
    """When overlay cannot rank, land the Actor-shaped backup, not a near-hold twitch."""
    if backups is None or backups.numel() == 0:
        return backups
    actor = actor_commands.reshape(5, 6).to(device=backups.device, dtype=backups.dtype)
    dist = (backups - actor).abs().reshape(backups.shape[0], -1).amax(dim=1)
    return backups[torch.argsort(dist)].contiguous()


def repeat_history(history, batch):
    if history.commands_position.shape[0] == batch:
        return history
    if history.commands_position.shape[0] != 1:
        raise ValueError('repeat_history expects a single planning history')
    return ActualHistory(
        history.commands_position.repeat(batch, 1, 1, 1),
        history.readbacks_position.repeat(batch, 1, 1, 1),
        history.scene_pooled.repeat(batch, 1, 1, 1),
        history.time_deltas.repeat(batch, 1, 1),
        history.valid.repeat(batch, 1),
    )


def _as_vec2(value):
    if value is None:
        return None
    out = np.asarray(value, dtype=np.float64).reshape(-1)
    if out.shape != (2,) or not np.isfinite(out).all():
        return None
    return out


def _unit(value):
    vec = _as_vec2(value)
    if vec is None:
        return None
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    return vec / norm


def _positive_scalar(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out) or out <= 1e-3:
        return None
    return out


class V21JointResponseCEM:
    """LatentDynamicsModel adapter. LeWM sees q, the command trace, history, and the current DINO pool."""

    def __init__(self, lewm, q, history, dt, physical_check, *,
                 project=None, target_px=None, observed_midpoint_px=None, capture_q=None,
                 long_axis_px=None, grasp_axis_px=None, length_px=None, width_px=None,
                 require_candidate_sensitive_visual=False,
                 objective='column_visual', reference_commands=None, trace=None,
                 response_forecast_admission=False, response_joint_count=5,
                 action_scale=1.0, visual_task_cost=None):
        from .v21_visual_task_cost import (
            BothViewsVisualCost, ImageAnchoredVisualCost, MeasuredVisualTaskCost)
        from .v21_fish_task_cost import FishTaskCost
        if visual_task_cost is not None and (
                not isinstance(visual_task_cost, (
                    MeasuredVisualTaskCost, ImageAnchoredVisualCost, BothViewsVisualCost, FishTaskCost))
                or objective != 'actor_response_tracking'):
            raise ValueError('measured visual task cost requires actor_response_tracking')
        self.visual_task_cost = visual_task_cost
        self.action_scale = _validate_action_scale(action_scale)
        self.requested_max_delta = _scaled_max_delta(self.action_scale)
        self.max_delta = _reachable_max_delta(self.action_scale)
        if objective not in ('column_visual', 'actor_response_tracking', 'actor_command_tracking'):
            raise ValueError('unknown CEM objective')
        if objective != 'column_visual' and require_candidate_sensitive_visual:
            raise ValueError('response tracking cannot impersonate visual ranking')
        if objective != 'column_visual' and (
                not isinstance(reference_commands, torch.Tensor)
                or reference_commands.shape != (1, 5, 6)
                or not torch.isfinite(reference_commands).all()):
            raise ValueError('response tracking requires a fixed finite Actor reference')
        self.objective = objective
        self.reference_commands = None if reference_commands is None else reference_commands.detach().clone()
        self.lewm = lewm
        self.q = q
        self.history = history
        self.patches = None
        self.patch_valid = None
        self.visual_memory = None
        self.visual_now = None
        self.dt = dt
        self.physical_check = physical_check
        self.project = project
        self.target_px = _as_vec2(target_px)
        self.observed_midpoint_px = _as_vec2(observed_midpoint_px)
        self.capture_q = None if capture_q is None else np.asarray(capture_q, dtype=np.float64)
        self.long_axis_px = _unit(long_axis_px)
        # Same gold line as the JPEG overlay: always ⊥ long axis, ignore stored grasp_axis.
        if self.long_axis_px is not None:
            self.grasp_axis_px = np.array((-self.long_axis_px[1], self.long_axis_px[0]), dtype=np.float64)
        else:
            self.grasp_axis_px = _unit(grasp_axis_px)
        self.length_px = _positive_scalar(length_px)
        self.width_px = _positive_scalar(width_px)
        self.require_candidate_sensitive_visual = bool(require_candidate_sensitive_visual)
        self.visual_transport_reason = None
        self.visual_supported_candidates = None
        self.candidate_spread_px = None
        self.backups = None
        self.backup_costs = None
        self.backup_predictions = None
        self._last_commands = None
        self.trace = trace
        self.response_forecast_admission = bool(response_forecast_admission)
        if type(response_joint_count) is not int or response_joint_count not in (5, 6):
            raise ValueError('response_joint_count must be 5 or 6')
        self.response_joint_count = response_joint_count
        self.response_admission_evaluated = 0
        self.response_admission_rejected = 0
        self.j6_phase_contract = None

    def response_admission_mask(self, commands, prediction):
        q = self.q.to(commands).expand(len(commands), -1)
        diagnostic = predicted_response_admission(
            q, commands, prediction, checked_joint_count=self.response_joint_count)
        self.response_admission_evaluated += len(commands)
        self.response_admission_rejected += int((~diagnostic['admitted']).sum())
        record_cem_event(self.trace, 'predicted_response_admission', commands=commands,
            diagnostic=diagnostic)
        return diagnostic['admitted']

    def command_phase_admitted(self, commands):
        check = getattr(self.visual_task_cost, 'require_commands', None)
        if check is not None:
            try:
                check(self.q, commands)
            except (ValueError, RuntimeError):
                return False
        if self.j6_phase_contract is None:
            return True
        return j6_phase_admitted(self.q.to(commands), commands, self.j6_phase_contract)

    def has_visual_goal(self):
        """Whether one exposure-time Column C/M pair can be transported safely."""
        if not callable(self.project) or self.target_px is None or self.observed_midpoint_px is None:
            return False
        if self.capture_q is None or self.capture_q.shape != (6,) or not np.isfinite(self.capture_q).all():
            return False
        return True

    def rollout_embeddings(self, history_embedding, history_action, future_action):
        """Predict each admitted candidate from the same factual q/history."""
        del history_embedding, history_action
        batch, samples, horizon, dim = future_action.shape
        if batch != 1 or (horizon, dim) != (5, 6):
            raise ValueError('V21 CEM candidates must be [B, N, 5, 6]')
        self._last_commands = future_action.detach().clone()
        if self.objective == 'actor_command_tracking':
            # Explicit CEM-first mode: these are command candidates, never
            # reported as LeWM predictions. Forecast only the ranked backups.
            source = self.q.to(future_action)[:, None, None].expand(1, samples, 1, 6)
            return torch.cat((source, future_action), dim=2)
        path = future_action.new_full((1, samples, 6, 6), float('nan'))
        valid = [i for i in range(samples) if self.command_phase_admitted(future_action[:, i])
                 and passes_physical(self.physical_check, self.q, future_action[:, i])]
        # Keep native inference batches bounded while preserving candidate order.
        for start in range(0, len(valid), 8):
            indices = valid[start:start+8]
            commands = future_action[0, indices]
            q = self.q.to(commands).expand(len(indices), -1).clone()
            dt = self.dt.to(commands).expand(len(indices), -1)
            patches = None if self.patches is None else self.patches.to(commands).expand(len(indices), -1, -1, -1, -1)
            mask = None if self.patch_valid is None else self.patch_valid.to(commands.device).expand(len(indices), -1, -1)
            visual = {}
            if getattr(self.lewm, 'reads_visual_memory', False):
                visual['memory'] = _repeat_visual_memory(self.visual_memory, len(indices))
                visual['now_time'] = None if self.visual_now is None else self.visual_now.to(commands.device).expand(len(indices))
            response = self.lewm(q, CommandTrace.policy(commands, dt),
                repeat_history(self.history, len(indices)).to(commands.device),
                dt.double().cumsum(-1), patches=patches, patch_valid=mask, **visual)
            pred = getattr(response, 'predicted_q_path', None)
            sigma = getattr(response, 'q_uncertainty', None)
            if self.trace is not None:
                record_cem_event(self.trace, 'lewm_candidate_batch', candidate_indices=indices,
                    q=q, commands=commands, dt=dt, query_offsets=dt.double().cumsum(-1),
                    predicted=pred, sigma=sigma)
            if (not isinstance(pred, torch.Tensor) or pred.shape != (len(indices), 6, 6)
                    or not pred.is_floating_point()
                    or not isinstance(sigma, torch.Tensor) or sigma.shape != (len(indices), 5, 6)
                    or not sigma.is_floating_point()):
                continue
            finite = (torch.isfinite(pred).all(dim=(1, 2))
                & torch.isfinite(sigma).all(dim=(1, 2)) & (sigma > 0).all(dim=(1, 2))
                & (pred[:, 0].to(q) == q).all(dim=1))
            if self.response_forecast_admission:
                finite &= self.response_admission_mask(commands, pred)
            for row, index in enumerate(indices):
                if bool(finite[row]) and passes_physical(
                        self.physical_check, self.q, pred[row:row+1, 1:]):
                    path[0, index] = pred[row].to(path)
        return path

    def forecast(self, commands):
        """One goal-free LeWM call on the CEM winner [1, 5, 6]."""
        if commands.shape != (1, 5, 6):
            raise ValueError('LeWM forecast expects a selected [1, 5, 6] command')
        q = self.q.to(commands)
        dt = self.dt.to(device=commands.device, dtype=commands.dtype).expand(1, -1)
        history = repeat_history(self.history, 1).to(commands.device)
        query = dt.double().cumsum(-1)
        patches = None if self.patches is None else self.patches.to(commands)
        valid = None if self.patch_valid is None else self.patch_valid.to(commands.device)
        visual = {}
        if getattr(self.lewm, 'reads_visual_memory', False):
            visual['memory'] = self.visual_memory
            visual['now_time'] = self.visual_now
        response = self.lewm(q, CommandTrace.policy(commands, dt), history, query,
                             patches=patches, patch_valid=valid, **visual)
        if self.trace is not None:
            record_cem_event(self.trace, 'selected_forecast', q=q, commands=commands,
                dt=dt, query_offsets=query, predicted=getattr(response, 'predicted_q_path', None),
                sigma=getattr(response, 'q_uncertainty', None))
        return response

    def goal_cost(self, predicted_embedding, goal_embedding):
        del goal_embedding
        if self._last_commands is None or self._last_commands.shape[:2] != predicted_embedding.shape[:2]:
            raise RuntimeError('goal_cost requires the matching CEM rollout')
        batch, samples = predicted_embedding.shape[:2]
        costs = predicted_embedding.new_zeros(batch, samples)
        source = self.q.detach().to(predicted_embedding).reshape(6)
        for b in range(batch):
            for n in range(samples):
                commands = self._last_commands[b, n]
                if (not self.command_phase_admitted(commands[None])
                        or not passes_physical(self.physical_check, source[None], commands[None])):
                    costs[b, n] = INADMISSIBLE_COST
                    continue
                costs[b, n] = self._trajectory_cost(predicted_embedding[b, n])
        if self.trace is not None:
            record_cem_event(self.trace, 'candidate_scores', commands=self._last_commands,
                predicted=predicted_embedding, reference=self.reference_commands,
                costs=costs, components=self._cost_components(predicted_embedding))
        return costs

    def _cost_components(self, path):
        if self.objective == 'column_visual':
            return dict(objective=self.objective, decomposition_available=False)
        error = (path[..., 1:, :] - self.reference_commands.to(path)) / path.new_tensor(MAX_DELTA)
        squared = error.square()
        visual_cost = None
        if self.visual_task_cost is not None:
            visual_cost = path.new_full(path.shape[:-2], INADMISSIBLE_COST)
            for index in np.ndindex(tuple(visual_cost.shape)):
                try:
                    visual_cost[index] = self.visual_task_cost.evaluate(path[index].detach().cpu().numpy())['cost']
                except (ValueError, TypeError):
                    pass
        return dict(objective=self.objective, decomposition_available=True,
            **({} if visual_cost is None else dict(visual_task_additive_cost=visual_cost)),
            valid=torch.isfinite(path).all(dim=(-2, -1)),
            normalized_error=error, by_axis=squared.mean(dim=-2) / 6,
            by_time=squared.mean(dim=-1) / 5, scale=list(MAX_DELTA))

    def _trajectory_cost(self, path):
        if path.shape != (6, 6) or not torch.isfinite(path).all():
            return path.new_tensor(INADMISSIBLE_COST)
        if self.objective != 'column_visual':
            # Fixed FCPG/Actor reference, never recomputed from a candidate.
            scale = path.new_tensor(MAX_DELTA)
            response_cost = ((path[1:] - self.reference_commands[0].to(path)) / scale).square().mean()
            if self.visual_task_cost is None:
                return response_cost
            try:
                visual = self.visual_task_cost.evaluate(path.detach().cpu().numpy())
            except (ValueError, TypeError):
                return path.new_tensor(INADMISSIBLE_COST)
            visual_term = path.new_tensor(float(visual['cost']))
            self.visual_distance_rank = 'distance_plus_angle'
            return visual_term
        return self._prefix_cost(path[PREFIX_PATH_INDEX])

    def rank_admitted(self, commands):
        """Rank only physical, visually supported candidates when vision is required."""
        if commands.ndim != 4 or commands.shape[0] != 1 or commands.shape[-2:] != (5, 6):
            raise ValueError('candidate pool must be [1, N, 5, 6]')
        self.backups = None
        self.backup_costs = None
        self.backup_predictions = None
        self.visual_transport_reason = None
        self.visual_supported_candidates = None
        self.candidate_spread_px = None
        self.visual_task_candidate_spread_px = None
        if self.require_candidate_sensitive_visual and not self.has_visual_goal():
            self.visual_transport_reason = 'missing_exposure_paired_column_dino_fg_transport'
            self.visual_supported_candidates = 0
            return None, None, 0
        dummy = commands.new_zeros(1, 1)
        path = self.rollout_embeddings(dummy, commands.new_zeros(1, 1, 6), commands)
        source = self.q.detach().to(commands).reshape(6)
        keep, scores, predicted = [], [], []
        for index in range(commands.shape[1]):
            candidate = commands[:, index]
            if (not self.command_phase_admitted(candidate)
                    or not passes_physical(self.physical_check, source[None], candidate)):
                continue
            prefix = path[0, index, PREFIX_PATH_INDEX]
            if not torch.isfinite(path[0, index]).all():
                continue
            if self.require_candidate_sensitive_visual:
                try:
                    midpoint = self._predicted_m(prefix)
                except (ValueError, TypeError, RuntimeError, AttributeError):
                    continue
            score = float(self._trajectory_cost(path[0, index]))
            if not np.isfinite(score) or score >= INADMISSIBLE_COST:
                continue
            keep.append(index)
            scores.append(score)
            if self.require_candidate_sensitive_visual:
                predicted.append(midpoint)
        if self.trace is not None:
            record_cem_event(self.trace, 'ranked_pool_evaluation', commands=commands,
                predicted=path, reference=self.reference_commands,
                admitted_indices=keep, admitted_costs=scores,
                components=self._cost_components(path))
        if self.require_candidate_sensitive_visual:
            self.visual_supported_candidates = len(predicted)
            if len(predicted) < 2:
                self.visual_transport_reason = 'insufficient_in_envelope_column_dino_fg_candidates'
            else:
                self.candidate_spread_px = float(np.ptp(np.stack(predicted), axis=0).max())
                if self.candidate_spread_px < MIN_CANDIDATE_MOTION_PX:
                    self.visual_transport_reason = 'candidate_insensitive_column_dino_fg_transport'
            if self.visual_transport_reason:
                return None, None, 0
        if not keep:
            return None, None, 0
        if getattr(self.visual_task_cost, 'sensitivity_space', None) == 'cost':
            # Stage 3/4 score predicted joint/metric states, not pixels. Never
            # label metre/gripper variation as an image-pixel response.
            if (getattr(self.visual_task_cost, 'phase', None) != 'done'
                    and (len(keep) < 2 or float(np.ptp(scores)) < 1e-6)):
                self.visual_transport_reason = 'lewm_task_cost_is_candidate_insensitive'
                return None, None, 0
        elif self.visual_task_cost is not None:
            pixels = np.asarray([self.visual_task_cost.evaluate(
                path[0, index].detach().cpu().numpy())['predicted_midpoints_px'][1:]
                for index in keep])
            # A fitted map alone is insufficient: this particular admitted
            # pool must have distinguishable predicted image outcomes.
            spread = float(np.max(np.ptp(pixels, axis=0)))
            self.visual_task_candidate_spread_px = spread
            if len(keep) < 2 or spread < MIN_CANDIDATE_MOTION_PX:
                # Column pixels often do not move once the jaw is close.
                # Leave the pool empty so the legal actor seed can still run.
                self.visual_transport_reason = 'measured visual task candidate predictions are insensitive'
                return None, None, 0
        score_order = np.argsort(np.asarray(scores, dtype=np.float64))
        order = [keep[i] for i in score_order]
        self.backups = commands[0, order].contiguous()
        self.backup_costs = [scores[i] for i in score_order]
        self.backup_predictions = path[0, order].contiguous()
        best = order[0]
        record_cem_event(self.trace, 'final_ranked_pool', admitted_indices=keep,
            admitted_costs=scores, order=order)
        return commands[:, best], scores[int(score_order[0])], len(order)

    def _predicted_m(self, predicted_q):
        prefix = np.asarray(predicted_q.detach().cpu().numpy() if hasattr(predicted_q, 'detach') else predicted_q,
                            dtype=np.float64).reshape(6)
        projected = np.asarray(self.project(prefix), dtype=np.float64).reshape(2)
        if self.observed_midpoint_px is not None and self.capture_q is not None:
            origin = np.asarray(self.project(self.capture_q), dtype=np.float64).reshape(2)
            projected = self.observed_midpoint_px + projected - origin
        if not np.isfinite(projected).all():
            raise ValueError('Column DINO predicted midpoint must be finite')
        return projected

    def _inside_body_ellipse(self, pred_m):
        return point_in_body_ellipse(
            pred_m, self.target_px, self.long_axis_px, self.length_px, self.width_px,
            self.grasp_axis_px)

    def _prefix_cost(self, predicted_q):
        if self.require_candidate_sensitive_visual and not self.has_visual_goal():
            return predicted_q.new_tensor(INADMISSIBLE_COST)
        if self.project is None or self.target_px is None:
            return predicted_q.new_zeros(())
        try:
            pred_m = self._predicted_m(predicted_q)
            target = self.target_px
            r_pred = float(np.linalg.norm(pred_m - target))
            mid = self.observed_midpoint_px
            # Column FG: if live M is already on the body, stop driving into the table.
            inside = self._inside_body_ellipse(pred_m) or (
                mid is not None and self._inside_body_ellipse(mid))
            cost = 0.0 if inside else r_pred / 224.0
            if mid is not None:
                cm = target - mid
                r_now = float(np.linalg.norm(cm))
                if r_now > 1e-3:
                    unit = cm / r_now
                    delta = pred_m - mid
                    along = float(np.dot(delta, unit))
                    lateral = float(np.linalg.norm(delta - along * unit))
                    if inside:
                        cost += (lateral + max(0.0, -along)) / 224.0
                    else:
                        cost += (lateral + max(0.0, -along) + max(0.0, r_pred - r_now)) / 224.0
                if self.long_axis_px is not None:
                    cost += abs(float(np.dot(pred_m - mid, self.long_axis_px))) / 224.0
                if self.grasp_axis_px is not None:
                    delta = pred_m - mid
                    across = abs(float(delta[0] * self.grasp_axis_px[1] - delta[1] * self.grasp_axis_px[0]))
                    cost += across / 224.0
            joints = np.asarray(
                predicted_q.detach().cpu().numpy() if hasattr(predicted_q, 'detach') else predicted_q,
                dtype=np.float64).reshape(6)
            if inside:
                cost += max(0.0, float(joints[5]) - CLOSED_Q6) / 40.0
            if not np.isfinite(cost):
                return predicted_q.new_tensor(INADMISSIBLE_COST)
            return predicted_q.new_tensor(float(cost))
        except (ValueError, TypeError, RuntimeError, AttributeError):
            return predicted_q.new_tensor(INADMISSIBLE_COST)


def point_in_body_ellipse(point, target, long_axis, length, width, grasp_axis=None):
    point, target = _as_vec2(point), _as_vec2(target)
    length, width = _positive_scalar(length), _positive_scalar(width)
    if point is None or target is None or length is None or width is None:
        return False
    half_l, half_w = 0.5 * length, 0.5 * width
    delta = point - target
    axis = _unit(long_axis)
    pinch = _unit(grasp_axis)
    if axis is None or pinch is None:
        return float(np.linalg.norm(delta)) <= min(half_l, half_w)
    along = float(np.dot(delta, axis)) / half_l
    across = float(np.dot(delta, pinch)) / half_w
    return (along * along + across * across) <= 1.0


def plan_commands(lewm, state, actor_commands, dt, physical_check, *,
                  project=None, target_px=None, observed_midpoint_px=None, capture_q=None,
                  long_axis_px=None, grasp_axis_px=None, length_px=None, width_px=None,
                  clearance_mm=None, generator=None,
                  require_candidate_sensitive_visual=False, objective='column_visual',
                  candidate_parameterization=None, use_physical_check_cache=True, trace=None,
                  action_scale=1.0, planning_budget_s=CEM_PLANNING_BUDGET_S,
                  planning_clock=None, visual_task_cost=None,
                  single_axis_stall=True, limit_step=True,
                  scene_goal=None, scene_precision='fp32', scene_value=None):
    """Legacy five-step planning, or explicit scene_goal/scene_value native40.

    scene_goal requires the learned-scene checkpoint and actual goal patches;
    it does not silently inherit pixel heuristics or Actor-tracking objectives.
    scene_value additionally requires its frozen learned distance scorer.
    Existing runtime callers retain their current objective and behavior.
    """
    action_scale = _validate_action_scale(action_scale)
    if trace is not None:
        if not isinstance(trace, CEMTrace):
            raise TypeError('trace must be CEMTrace or None')
        trace.begin(actor_commands, generator, q=state.q, history=state.history,
            actor_commands=actor_commands, dt=dt, objective=objective,
            candidate_parameterization=candidate_parameterization,
            use_physical_check_cache=use_physical_check_cache, action_scale=action_scale,
            planning_budget_s=planning_budget_s)
    error = None
    try:
        if objective in ('scene_goal', 'scene_value'):
            if (objective == 'scene_value') != (scene_value is not None):
                raise ValueError('scene_value requires its explicit learned scorer; no latent fallback')
            if (scene_goal is None or action_scale != 1.0
                    or require_candidate_sensitive_visual
                    or candidate_parameterization not in (None, 'native40_endpoint_chunks')
                    or any(value is not None for value in (project, target_px,
                        observed_midpoint_px, capture_q, long_axis_px, grasp_axis_px,
                        length_px, width_px, clearance_mm, visual_task_cost))):
                raise ValueError('scene_goal requires goal patches, native40 commands and no legacy pixel/scaling options')
            from .v21_scene_cem import plan_scene_commands
            return plan_scene_commands(lewm, state, actor_commands, dt, physical_check,
                scene_goal=scene_goal, generator=generator, trace=trace,
                planning_budget_s=planning_budget_s, planning_clock=planning_clock,
                precision=scene_precision, scene_value=scene_value)
        if scene_goal is not None or scene_value is not None:
            raise ValueError('scene goal patches/scorer require objective=scene_goal or scene_value')
        return _plan_commands(lewm, state, actor_commands, dt, physical_check,
            project=project, target_px=target_px, observed_midpoint_px=observed_midpoint_px,
            capture_q=capture_q, long_axis_px=long_axis_px, grasp_axis_px=grasp_axis_px,
            length_px=length_px, width_px=width_px, clearance_mm=clearance_mm,
            generator=generator, require_candidate_sensitive_visual=require_candidate_sensitive_visual,
            objective=objective, candidate_parameterization=candidate_parameterization,
            use_physical_check_cache=use_physical_check_cache, trace=trace,
            action_scale=action_scale, planning_budget_s=planning_budget_s,
            planning_clock=planning_clock, visual_task_cost=visual_task_cost,
            single_axis_stall=single_axis_stall, limit_step=limit_step)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if trace is not None:
            trace.finish(actor_commands, generator, error=error)


def _plan_commands(lewm, state, actor_commands, dt, physical_check, *,
                  project=None, target_px=None, observed_midpoint_px=None, capture_q=None,
                  long_axis_px=None, grasp_axis_px=None, length_px=None, width_px=None,
                  clearance_mm=None, generator=None,
                  require_candidate_sensitive_visual=False, objective='column_visual',
                  candidate_parameterization=None, use_physical_check_cache=True, trace=None,
                  action_scale=1.0, planning_budget_s=CEM_PLANNING_BUDGET_S,
                  planning_clock=None, visual_task_cost=None,
                  single_axis_stall=True, limit_step=True):
    """Plan with an explicit command-first or response-first CEM objective."""
    del clearance_mm
    if candidate_parameterization not in (None, 'bounded_endpoint_then_hold', 'five_command_sequence',
                                          'scaled_actor_sequence', 'scaled_actor_endpoint_hold',
                                          'amplified_actor_sequence', 'smooth_actor_endpoint'):
        raise ValueError('unknown CEM candidate parameterization')
    if type(use_physical_check_cache) is not bool:
        raise ValueError('use_physical_check_cache must be bool')
    if type(single_axis_stall) is not bool:
        raise ValueError('single_axis_stall must be bool')
    if type(limit_step) is not bool:
        raise ValueError('limit_step must be bool')
    if candidate_parameterization is None:
        candidate_parameterization = ('bounded_endpoint_then_hold'
            if objective == 'actor_command_tracking' else 'five_command_sequence')
    action_scale = _validate_action_scale(action_scale)
    planning_budget = _PlanningBudget(
        planning_budget_s, clock=planning_clock, trace=trace)
    if action_scale != 1.0 and (objective != 'actor_response_tracking'
            or candidate_parameterization not in ('amplified_actor_sequence', 'smooth_actor_endpoint')):
        raise ValueError('expanded action_scale requires actor_response_tracking with amplified_actor_sequence')
    if candidate_parameterization == 'amplified_actor_sequence' and action_scale == 1.0:
        raise ValueError('amplified_actor_sequence requires action_scale greater than one')
    if actor_commands.shape != (1, 5, 6):
        raise ValueError('Actor proposal must be [1, 5, 6]')
    model = V21JointResponseCEM(
        lewm, state.q, state.history, dt, physical_check,
        project=project, target_px=target_px,
        observed_midpoint_px=observed_midpoint_px, capture_q=capture_q,
        long_axis_px=long_axis_px, grasp_axis_px=grasp_axis_px,
        length_px=length_px, width_px=width_px,
        require_candidate_sensitive_visual=require_candidate_sensitive_visual,
        objective=objective, reference_commands=actor_commands, trace=trace,
        response_forecast_admission=candidate_parameterization in (
            'scaled_actor_endpoint_hold', 'amplified_actor_sequence', 'smooth_actor_endpoint'),
        response_joint_count=(6 if candidate_parameterization in ('amplified_actor_sequence', 'smooth_actor_endpoint') else 5),
        action_scale=action_scale, visual_task_cost=visual_task_cost,
    )
    features = getattr(getattr(state, 'memory', None), 'features', None)
    if isinstance(features, torch.Tensor) and tuple(features.shape[1:]) == (2, 3, PATCHES, DIM):   # V21_ENCODER
        model.patches = features.detach()
        model.patch_valid = state.memory.valid.detach()
        if getattr(lewm, 'reads_visual_memory', False):
            model.visual_memory = state.memory
            model.visual_now = state.time.detach()
    if visual_task_cost is not None:
        record_cem_event(trace, 'visual_task_input', receipt=visual_task_cost.receipt())
    if require_candidate_sensitive_visual and not model.has_visual_goal():
        raise RuntimeError('missing_exposure_paired_column_dino_fg_transport')
    on_body = bool(model.observed_midpoint_px is not None
                   and model._inside_body_ellipse(model.observed_midpoint_px))

    low, high = _physical_box(actor_commands, state.q, action_scale=action_scale)
    step = actor_commands.new_tensor(MAX_STEP)
    proposal, reference_transform = prepare_actor_reference(
        actor_commands, state.q, candidate_parameterization, action_scale=action_scale)
    # The search box must contain the unscaled Actor commands. Clamping them
    # back into the old 3°/8° box would reintroduce a shrink.
    low = torch.minimum(low, proposal.amin(dim=1)[0])
    high = torch.maximum(high, proposal.amax(dim=1)[0])
    command_projector = getattr(visual_task_cost, 'project_commands', None)
    if command_projector is not None:
        proposal = command_projector(proposal)
        low = torch.minimum(low, proposal.amin(dim=1)[0])
        high = torch.maximum(high, proposal.amax(dim=1)[0])
    command_bounds = getattr(visual_task_cost, 'command_bounds', None)
    if command_bounds is not None:
        low, high = command_bounds(low.clone(), high.clone())
    # Infer gripper phase from the raw Actor trajectory.  Scaling the arm
    # reference can amplify a harmless initial twitch enough to look like a
    # phase crossing; the phase contract must be tied to what Actor actually
    # emitted, then the transformed proposal is projected into that contract.
    raw_reference = proposal.detach().clone()
    j6_contract = (j6_phase_contract(state.q, actor_commands, action_scale=action_scale)
                   if objective in TRACKING_OBJECTIVES and command_projector is None else None)
    if j6_contract is not None:
        proposal = _apply_j6_phase_contract(proposal.clone(), state.q, j6_contract)
        changed = not torch.equal(raw_reference, proposal)
        correction_reason = ('clamped_transformed_initial_opposite_jitter'
                             if changed else None)
        j6_contract = deepcopy(j6_contract)
        j6_contract['reference_correction'] = dict(
            applied=changed,
            reason=correction_reason,
            raw_reference=raw_reference[0].detach().cpu().tolist(),
            corrected_reference=proposal[0].detach().cpu().tolist(),
        )
        reference_transform = dict(reference_transform,
            j6_phase_correction_applied=changed,
            j6_phase_correction_reason=correction_reason)
        low, high = low.clone(), high.clone()
        if j6_contract['direction'] > 0:
            low[J6_INDEX] = torch.maximum(low[J6_INDEX], state.q[0, J6_INDEX].to(low))
        elif j6_contract['direction'] < 0:
            high[J6_INDEX] = torch.minimum(high[J6_INDEX], state.q[0, J6_INDEX].to(high))
        model.j6_phase_contract = deepcopy(j6_contract)

    physical_gate_cache = _SuccessfulPhysicalCheckCache(
        physical_check, enabled=use_physical_check_cache)

    def budgeted_physical_check(q, commands):
        planning_budget.checkpoint('physical_gate')
        result = physical_gate_cache(q, commands)
        planning_budget.checkpoint('physical_gate_complete')
        return result

    full_stall_cache = _SuccessfulSingleAxisStallCache(
        budgeted_physical_check, enabled=use_physical_check_cache)

    def physical_gated(q, commands):
        if objective in TRACKING_OBJECTIVES and single_axis_stall:
            if not full_stall_cache(q, commands):
                raise RuntimeError('CEM candidate unsafe under a single arm-axis tracking stall')
        else:
            budgeted_physical_check(q, commands)

    def candidate_gated(q, commands):
        check = getattr(visual_task_cost, 'require_commands', None)
        if check is not None:
            check(q, commands)
        if j6_contract is not None and not j6_phase_admitted(q, commands, j6_contract):
            raise RuntimeError('CEM candidate violates the Actor-derived J6 phase/direction contract')
        physical_gated(q, commands)

    model.physical_check = physical_gated
    sustained = candidate_parameterization in ('bounded_endpoint_then_hold', 'scaled_actor_endpoint_hold')
    smooth = candidate_parameterization == 'smooth_actor_endpoint'
    if objective != 'column_visual':
        model.reference_commands = proposal.detach().clone()
    seed = _legal_seed(proposal, state.q, candidate_gated, low, high, step, generator,
                       LEGAL_SEED_ATTEMPTS, j6_contract, smooth=smooth,
                       command_projector=command_projector)
    record_cem_event(trace, 'bounded_actor_reference', raw_actor=actor_commands,
        source=state.q, raw_reference=raw_reference, reference=proposal, low=low, high=high, step=step,
        transform=reference_transform, j6_phase_contract=j6_contract, legal_seed=seed)
    constraint_stats = dict(resampled=0, seed_replacements=0)
    constrain = lambda candidates: constrain_to_physical(
        candidates, physical_check=candidate_gated, q=state.q, seed=seed,
        low=low, high=high, generator=generator, sustained=sustained,
        attempts=CEM_RESAMPLE_ATTEMPTS, j6_contract=j6_contract,
        stats=constraint_stats, smooth=smooth, limit_step=limit_step,
        command_projector=command_projector)
    planner = GroundedCEMPlanner(
        horizon=5,
        action_dim=6,
        num_samples=CEM_POPULATION,
        num_elites=CEM_ELITES,
        iterations=CEM_ITERS,
        action_low=tuple(float(x) for x in low),
        action_high=tuple(float(x) for x in high),
    )
    dummy = actor_commands.new_zeros(1, 1, 1)
    plan = planner.plan(
        model,
        dummy,
        actor_commands.new_zeros(1, 1, 6),
        dummy,
        initial_mean=proposal,
        generator=generator,
        constrain=constrain,
        trace=trace,
        admissible=lambda _commands, _prediction, cost: cost < INADMISSIBLE_COST,
    )
    pool = constrain(_candidate_pool(plan, proposal, low, high, generator))
    model.retained_candidate_count = 0
    if plan.retained_candidates:
        retained = plan.retained_candidates[0]
        # These absolute commands already passed the iteration constraint. Do
        # not rewrite them with the stochastic constraint a second time. The
        # combined pool still repeats all original ranking/forecast checks.
        model.retained_candidate_count = len(retained)
        record_cem_event(trace, 'retained_candidates_appended', commands=retained,
            prior_costs=plan.retained_costs[0], sources=plan.retained_sources[0],
            final_pool_indices=torch.arange(pool.shape[1], pool.shape[1]+len(retained), device=pool.device),
            costs_require_final_revalidation=True)
        pool = torch.cat((pool, retained[None].to(pool)), dim=1)
    selected, cost, admitted = model.rank_admitted(pool)
    if require_candidate_sensitive_visual and model.visual_transport_reason:
        raise RuntimeError(model.visual_transport_reason)
    if selected is None or model.backups is None:
        if getattr(visual_task_cost, 'strict_model_rank', False):
            raise RuntimeError(model.visual_transport_reason or 'no finite LeWM-scored stage candidate; no Actor/script fallback')
        if model.response_forecast_admission:
            raise RuntimeError('no CEM candidate passed physical and predicted directional response admission')
        if seed is None:
            raise RuntimeError('no admissible CEM backup with a finite supported cost')
        # The final pool had no finite visual cost. Keep the already-legal
        # actor seed so one empty ranking does not abort the whole trial.
        model.backups = seed.detach()[None]
        model.backup_costs = [1e5]
        rolled = model.rollout_embeddings(
            dummy, actor_commands.new_zeros(1, 1, 6), seed.detach()[None, None])
        model.backup_predictions = rolled[0]
        selected, cost, admitted = seed.detach()[None], 1e5, 1
    # Response-first modes already checked LeWM; command-first mode forecasts
    # ranked backups below, preserving their cost/prediction association.
    backup_index = 0
    selected = model.backups[0:1]
    if objective == 'actor_command_tracking':
        selected, backup_index, forecast = pick_after_lewm(model, model.backups, physical_gated)
        model.backup_predictions = None
        model.selected_forecast = forecast.predicted_q_path[0].detach().clone()
        cost = model.backup_costs[backup_index]
    selected_prediction = (model.selected_forecast if objective == 'actor_command_tracking'
                           else model.backup_predictions[backup_index])[None]
    final_raw_checks = 0

    def final_uncached_physical_check(q, commands):
        nonlocal final_raw_checks
        planning_budget.checkpoint('final_physical_gate')
        final_raw_checks += 1
        result = physical_check(q, commands)
        planning_budget.checkpoint('final_physical_gate_complete')
        return result

    final_admission = (passes_single_axis_stall
                       if objective in TRACKING_OBJECTIVES and single_axis_stall
                       else passes_physical)
    if not final_admission(final_uncached_physical_check, state.q, selected):
        raise RuntimeError('selected CEM command failed final uncached physical audit')
    if not final_admission(
            final_uncached_physical_check, state.q, selected_prediction[:, 1:]):
        raise RuntimeError('selected LeWM prediction failed final uncached physical audit')
    model.final_physical_audit = dict(
        cache_bypassed=True,
        selected_command_checked=True,
        selected_prediction_checked=True,
        single_arm_axis_stall_checked=objective in TRACKING_OBJECTIVES and single_axis_stall,
        raw_checks=int(final_raw_checks),
    )
    model.physical_gate_cache = dict(
        physical_gate_cache.receipt(),
        full_stall_success_cache=full_stall_cache.receipt(),
    )
    model.constraint_stats = deepcopy(constraint_stats)
    planning_budget.checkpoint('complete')
    model.planning_budget = planning_budget.receipt()
    receipt = _record(
        model, on_body, cost=cost, admitted=admitted,
        backup_index=backup_index, candidate_parameterization=candidate_parameterization)
    if candidate_parameterization in (
            'scaled_actor_sequence', 'scaled_actor_endpoint_hold', 'amplified_actor_sequence', 'smooth_actor_endpoint'):
        receipt['reference_transform'] = reference_transform
    if model.response_forecast_admission:
        selected_prediction = (model.selected_forecast if objective == 'actor_command_tracking'
                               else model.backup_predictions[backup_index])[None]
        admission = predicted_response_admission(
            state.q, selected, selected_prediction,
            checked_joint_count=model.response_joint_count)
        assert bool(admission['admitted'][0])
        receipt['response_forecast_admission'] = dict(enabled=True,
            criterion=('existing_driven_joint_command_peak_direction_strict_minimum'
                       if model.response_joint_count == 6
                       else 'existing_driven_arm_command_peak_direction_strict_minimum'),
            minimum_motion=MIN_DIRECTIONAL_RESPONSE,
            checked_joint_count=model.response_joint_count,
            checked_arm_joint_count=min(model.response_joint_count, 5),
            response_source='lewm_predicted_readback', predictions_used_as_actual_evidence=False,
            actual_response_still_required_after_execution=True,
            uncertainty_threshold_added=False,
            evaluated_candidates=model.response_admission_evaluated,
            rejected_candidates=model.response_admission_rejected,
            selected_driven_joint_axes=admission['driven_joint_axes'][0].cpu().tolist(),
            selected_driven_arm_axes=admission['driven_arm_axes'][0].cpu().tolist(),
            selected_commanded_peak_delta=admission['commanded_peak_delta'][0].cpu().tolist(),
            selected_predicted_directional_peak=admission['predicted_directional_peak'][0].cpu().tolist(),
            forecast_response_samples=admission['response_samples'])
    record_cem_event(trace, 'selected', commands=selected, receipt=receipt)
    return selected, receipt


def _record(model, on_body, *, cost, admitted, candidate_parameterization, backup_index=0,
            visual_transport_reason=None):
    """Stable CEM receipt. Hold is never a landing option."""
    reason = visual_transport_reason or model.visual_transport_reason
    visual_ready = model.has_visual_goal() and reason is None
    backups = 0 if model.backups is None else int(model.backups.shape[0])
    receipt = dict(
        action_source=ACTION_SOURCE,
        population=CEM_POPULATION,
        elites=CEM_ELITES,
        iterations=CEM_ITERS,
        retained_candidates=int(getattr(model, 'retained_candidate_count', 0)),
        actor_prior_weight=0.0,
        objective=model.objective,
        ranking_state_source='candidate_command' if model.objective == 'actor_command_tracking' else 'lewm_predicted_readback',
        selected_predicted_q_path=(model.selected_forecast if model.objective == 'actor_command_tracking'
            else model.backup_predictions[backup_index]).detach().cpu().tolist(),
        reference_commands_position=None if model.reference_commands is None else model.reference_commands[0].detach().cpu().tolist(),
        candidate_parameterization=candidate_parameterization,
        single_arm_axis_stall_checked=model.objective in TRACKING_OBJECTIVES,
        task_visual_optimization=model.objective == 'column_visual' or model.visual_task_cost is not None,
        **({} if model.visual_task_cost is None else dict(visual_task_cost=dict(
            model.visual_task_cost.receipt(),
            candidate_spread_px=getattr(model, 'visual_task_candidate_spread_px', None),
            visual_distance_rank=getattr(model, 'visual_distance_rank', None)))),
        prefix_path_index=PREFIX_PATH_INDEX,
        cost=cost,
        admitted=admitted,
        used_actor_fallback=False,
        used_hold_fallback=False,
        backup_index=int(backup_index),
        backups=backups,
        max_delta=list(model.max_delta),
        max_step=list(MAX_STEP),
        action_scale=model.action_scale,
        cost_scale=list(MAX_DELTA),
        overlay_geometry=bool(model.target_px is not None and model.observed_midpoint_px is not None),
        used_long_axis=model.long_axis_px is not None,
        used_grasp_axis=model.grasp_axis_px is not None,
        used_body_ellipse=model.length_px is not None and model.width_px is not None,
        live_m_on_body=on_body,
        allow_descend=True,
        lewm_after_cem=model.objective == 'actor_command_tracking',
        project_mode='column_dino_fg_candidate_sensitive' if visual_ready else None,
        visual_transport_reason=reason,
        visual_supported_candidates=model.visual_supported_candidates,
        candidate_spread_px=model.candidate_spread_px,
        physical_gate_cache=deepcopy(getattr(model, 'physical_gate_cache', None)),
        final_physical_audit=deepcopy(getattr(model, 'final_physical_audit', None)),
        planning_budget=deepcopy(getattr(model, 'planning_budget', None)),
        constraint_stats=deepcopy(getattr(model, 'constraint_stats', None)),
        j6_phase_contract=deepcopy(getattr(model, 'j6_phase_contract', None)),
    )
    if model.max_delta != model.requested_max_delta:
        receipt.update(requested_max_delta=list(model.requested_max_delta),
            command_horizon=5, horizon_limited=True)
    return receipt


def _limit_horizon(commands, q, step, low, high):
    out = commands.reshape(5, 6).clone()
    prev = q.reshape(6).to(device=out.device, dtype=out.dtype)
    for t in range(5):
        out[t] = (prev + (out[t] - prev).clamp(-step, step)).clamp(low, high)
        prev = out[t]
    return out


def smooth_endpoint_chunk(q, endpoint, low, high, step):
    """One monotone synchronized path; ease in/out only within slew headroom.

    Blending minimum-jerk timing with constant speed preserves a reachable
    terminal target even when an axis needs all five existing step allowances.
    This transforms candidates BEFORE LeWM evaluation, never after selection.
    """
    source = q.reshape(6).to(endpoint)
    target = endpoint.clamp(low, high)
    delta = (target - source).clamp(-5 * step, 5 * step)
    t = torch.arange(1, 6, device=endpoint.device, dtype=endpoint.dtype) / 5
    eased = 10 * t**3 - 15 * t**4 + 6 * t**5
    peak = torch.diff(torch.cat((t.new_zeros(1), eased))).max()
    permitted = (step / delta.abs().clamp_min(1e-12)).amin(-1)
    blend = ((permitted - .2) / (peak - .2)).clamp(0, 1)
    phase = t + blend[..., None] * (eased - t)
    return source + phase[..., :, None] * delta[..., None, :]


def prepare_actor_reference(actor_commands, q, parameterization, *, action_scale=1.0):
    """Bound an Actor sequence without relabeling its time or physical outcome.

    ``amplified_actor_sequence`` first multiplies every Actor displacement from
    the factual source by ``action_scale``. Each joint is then independently
    limited by its unchanged per-command slew and absolute position bounds, so
    one saturated arm joint cannot shrink the gripper or the other arm joints.
    The five Actor samples remain five distinct temporal targets.

    ``scaled_actor_sequence`` and ``scaled_actor_endpoint_hold`` retain their
    legacy uniform shrink-to-fit transform for scale-one replay; the endpoint
    mode then approaches its terminal point and holds it. None of these modes
    is a dynamics-reachability claim. Geometry remains the separate existing
    gate.
    """
    if actor_commands.shape != (1, 5, 6) or q.shape != (1, 6):
        raise ValueError('reference requires Actor [1,5,6] and q [1,6]')
    if not torch.isfinite(actor_commands).all() or not torch.isfinite(q).all():
        raise ValueError('reference inputs must be finite')
    low, high = _physical_box(actor_commands, q, action_scale=action_scale)
    step = actor_commands.new_tensor(MAX_STEP)
    if parameterization == 'smooth_actor_endpoint':
        requested = q + action_scale * (actor_commands[:, -1] - q)
        proposal = smooth_endpoint_chunk(q, requested, low, high, step)
        return proposal, dict(kind='actual_source_to_actor_q5_smooth_chunk', action_scale=action_scale,
            source_position=q[0].detach().cpu().tolist(),
            raw_actor_q5=actor_commands[0, -1].detach().cpu().tolist(),
            requested_endpoint=requested[0].detach().cpu().tolist(),
            endpoint_position=proposal[0, -1].detach().cpu().tolist(),
            interpolation='minimum_jerk_blended_with_constant_speed_within_existing_slew',
            time_interpolated=True, actor_time_shape_preserved=False, endpoint_hold=False,
            dynamics_reachability_verified=False, geometry_admission_still_required=True)
    if parameterization not in (
            'scaled_actor_sequence', 'scaled_actor_endpoint_hold', 'amplified_actor_sequence'):
        if parameterization not in ('bounded_endpoint_then_hold', 'five_command_sequence'):
            raise ValueError('unknown Actor reference parameterization')
        proposal = _limit_horizon(actor_commands[0], q.reshape(6), step, low, high)[None]
        if parameterization == 'bounded_endpoint_then_hold':
            proposal = _limit_horizon(proposal[0, -1:].expand(5, -1),
                q.reshape(6), step, low, high)[None]
        return proposal, dict(kind=parameterization)
    source = q.to(actor_commands)
    if (source < low).any() or (source > high).any():
        raise ValueError('source q is outside absolute joint bounds')
    delta = actor_commands - source[:, None]
    if parameterization == 'amplified_actor_sequence':
        requested = source[:, None] + action_scale * delta
        proposal = _limit_horizon(
            requested[0], source.reshape(6), step, low, high)[None]
        return proposal, dict(
            kind='per_axis_limited_amplified_actor_sequence',
            action_scale=action_scale,
            shared_alpha=False,
            time_interpolated=False,
            endpoint_hold=False,
            actor_time_shape_preserved=True,
            dynamics_reachability_verified=False,
            geometry_admission_still_required=True,
            raw_max_delta=delta.abs().amax(dim=1)[0].detach().cpu().tolist(),
            requested_max_delta=(requested-source[:, None]).abs().amax(dim=1)[0].detach().cpu().tolist(),
            scaled_max_delta=(proposal-source[:, None]).abs().amax(dim=1)[0].detach().cpu().tolist(),
        )
    # Keep the Actor commands. Do not multiply by a shared ratio or per-step scale.
    proposal = actor_commands.clamp(source.new_tensor(DEFAULT_LOWER), source.new_tensor(DEFAULT_UPPER))
    increments = torch.diff(torch.cat((source[:, None], proposal), 1), dim=1)
    # #region agent log
    try:
        import json as _json, time as _time
        with open('/home/james/feedback-corrected-grounding/.cursor/debug-55ff20.log', 'a') as _f:
            _f.write(_json.dumps({'sessionId': '55ff20', 'hypothesisId': 'cap', 'location': 'v21_lewm_cem.py:prepare_actor_reference', 'message': 'actor commands kept', 'data': {'alpha': 1.0, 'shared_alpha': False, 'raw_max_delta': [round(float(x), 3) for x in delta.abs().amax(dim=1)[0]], 'kept_max_delta': [round(float(x), 3) for x in (proposal - source[:, None]).abs().amax(dim=1)[0]]}, 'timestamp': int(_time.time() * 1000)}) + '\n')
    except Exception:
        pass
    # #endregion
    metadata = dict(kind='unscaled_actor_sequence',
        alpha=1.0, shared_alpha=False, time_interpolated=False, endpoint_hold=False,
        dynamics_reachability_verified=False, geometry_admission_still_required=True,
        raw_max_delta=delta.abs().amax(dim=1)[0].detach().cpu().tolist(),
        raw_max_step=increments.abs().amax(dim=1)[0].detach().cpu().tolist(),
        scaled_max_delta=(proposal-source[:, None]).abs().amax(dim=1)[0].detach().cpu().tolist())
    if parameterization == 'scaled_actor_endpoint_hold':
        endpoint = proposal[:, -1].clone()
        metadata.update(kind='uniform_spatial_scaling_actor_sequence_then_endpoint_hold',
            endpoint_hold=True, actor_time_shape_preserved=False,
            scaled_actor_endpoint_position=endpoint[0].detach().cpu().tolist(),
            scaled_actor_sequence_max_delta=metadata['scaled_max_delta'])
        proposal = _limit_horizon(endpoint[0:1].expand(5, -1),
            source.reshape(6), step, low, high)[None]
        metadata['scaled_max_delta'] = (proposal-source[:, None]).abs().amax(dim=1)[0].detach().cpu().tolist()
    return proposal, metadata


def _legal_seed(proposal, q, physical_check, low, high, step, generator, attempts=LEGAL_SEED_ATTEMPTS,
                j6_contract=None, *, smooth=False, command_projector=None):
    proposal = _apply_j6_phase_contract(proposal.clone(), q, j6_contract)
    if command_projector is not None:
        proposal = command_projector(proposal)
    if passes_physical(physical_check, q, proposal):
        return proposal[0]
    # #region agent log
    try:
        import json as _json, time as _time
        _why = None
        try:
            physical_check(q.detach().clone(), proposal.detach().clone())
        except (RuntimeError, ValueError, TypeError) as _err:
            _why = f'{type(_err).__name__}: {_err}'
        with open('/home/james/feedback-corrected-grounding/.cursor/debug-55ff20.log', 'a') as _f:
            _f.write(_json.dumps({'sessionId': '55ff20', 'hypothesisId': 'gate', 'location': 'v21_lewm_cem.py:_legal_seed', 'message': 'proposal rejected', 'data': {'why': None if _why is None else _why[:300]}, 'timestamp': int(_time.time() * 1000)}) + '\n')
    except Exception:
        pass
    # #endregion
    source = q.reshape(6).to(device=proposal.device, dtype=proposal.dtype)
    span = (high - low).clamp_min(1e-3)
    base = source.view(1, 6).expand(5, 6)
    for attempt in range(attempts):
        noise = torch.randn(
            5, 6, device=proposal.device, dtype=proposal.dtype, generator=generator)
        trial = _limit_horizon(
            base + (0.5 ** (attempt // 6)) * span * noise, source, step, low, high)[None]
        if smooth:
            trial = smooth_endpoint_chunk(q, trial[:, -1], low, high, step)
        trial = _apply_j6_phase_contract(trial, q, j6_contract)
        if command_projector is not None:
            trial = command_projector(trial)
        if passes_physical(physical_check, q, trial):
            return trial[0]
    raise RuntimeError('no physically admissible seed; CEM will not search outside the gate')


def _physical_box(reference, q, *, allow_descend_j2=True, action_scale=1.0):
    span = reference.new_tensor(_reachable_max_delta(action_scale))
    center = q.reshape(6).to(device=reference.device, dtype=reference.dtype)
    low = (center - span).clamp(min=reference.new_tensor(DEFAULT_LOWER))
    high = (center + span).clamp(max=reference.new_tensor(DEFAULT_UPPER))
    if not allow_descend_j2:
        # On this arm, more-positive J2 lowers the anchors. Stay level until
        # column visual M is on the fish.
        high = high.clone()
        high[1] = torch.minimum(high[1], center[1])
    return low, high


def constrain_to_physical(candidates, *, physical_check, q, seed, low, high, generator=None,
                          attempts=6, sustained=False, j6_contract=None, stats=None, smooth=False,
                          limit_step=True, command_projector=None):
    """Replace any sample that fails the physical gate. CEM never sees those."""
    if candidates.ndim != 4 or candidates.shape[-2:] != (5, 6):
        raise ValueError('constrain_to_physical expects [B, N, 5, 6]')
    low_v, high_v = low.view(1, 1, 1, -1), high.view(1, 1, 1, -1)
    out = candidates.clamp(low_v, high_v)
    if sustained:
        out = out[:, :, -1:, :].expand_as(out).clone()
    step = out.new_tensor(MAX_STEP)
    source = q.reshape(6).to(device=out.device, dtype=out.dtype)
    if smooth:
        out = smooth_endpoint_chunk(q, out[:, :, -1], low, high, step)
    elif limit_step:
        for t in range(5):
            prev = source if t == 0 else out[:, :, t - 1]
            out[:, :, t] = prev + (out[:, :, t] - prev).clamp(-step, step)
            out[:, :, t] = out[:, :, t].clamp(low, high)
    else:
        out = out.clamp(low, high)
    out = _apply_j6_phase_contract(out, q, j6_contract)
    if command_projector is not None:
        out = command_projector(out)
    span = (high - low).clamp_min(1e-3).view(1, 1, 1, -1)
    seed = seed.to(out).reshape(5, 6)
    if limit_step:
        seed = _limit_horizon(seed, source, step, low, high)
    if smooth:
        seed = smooth_endpoint_chunk(q, seed[-1], low, high, step)
    seed = _apply_j6_phase_contract(seed, q, j6_contract)
    if command_projector is not None:
        seed = command_projector(seed)
    if not passes_physical(physical_check, q, seed[None]):
        raise RuntimeError('CEM replacement seed does not pass the complete physical gate')
    batch, samples = out.shape[:2]
    for b in range(batch):
        for n in range(samples):
            if passes_physical(physical_check, q, out[b, n][None]):
                continue
            replaced = False
            for attempt in range(attempts):
                scale = 0.5 ** (attempt + 1)
                noise = torch.randn(
                    5, 6, device=out.device, dtype=out.dtype, generator=generator)
                trial = _limit_horizon(seed + scale * span.reshape(6) * noise, source, step, low, high)
                if sustained:
                    trial = _limit_horizon(trial[-1:].expand(5, -1), source, step, low, high)
                if smooth:
                    trial = smooth_endpoint_chunk(q, trial[-1], low, high, step)
                trial = _apply_j6_phase_contract(trial, q, j6_contract)
                if command_projector is not None:
                    trial = command_projector(trial)
                if passes_physical(physical_check, q, trial[None]):
                    out[b, n] = trial
                    if stats is not None:
                        stats['resampled'] = int(stats.get('resampled', 0)) + 1
                    replaced = True
                    break
            if not replaced:
                # The seed has already passed the complete physical gate. Do
                # not leave the rejected sample in the pool for every later
                # LeWM/cost/ranking pass to reject again.
                out[b, n] = seed
                if stats is not None:
                    stats['seed_replacements'] = int(stats.get('seed_replacements', 0)) + 1
    return out


def _candidate_pool(plan, actor_commands, low, high, generator):
    mean, std = plan.distribution_mean, plan.distribution_std
    noise = torch.randn(
        1, CEM_POPULATION * 2, 5, 6,
        device=mean.device, dtype=mean.dtype, generator=generator,
    )
    sampled = (mean[:, None] + std[:, None] * noise).clamp(
        low.view(1, 1, 1, -1), high.view(1, 1, 1, -1))
    extras = torch.stack((actor_commands[0], plan.action_sequence[0]), 0)[None]
    return torch.cat((sampled, extras), 1)
