"""V20.3 acceptance probes on explicitly supplied training/development samples.

No dataset is opened here and no hardware API is reachable.  These helpers do
not select checkpoints or claim held evidence.  Inference interventions are
matched on the same current endpoint; their outputs never enter actual memory.
"""
from __future__ import annotations

from dataclasses import fields, replace
from functools import wraps
import math
from typing import Mapping

import torch
from torch import Tensor

from .prod_v20_3 import V203ActualState, V203Core, detach_actual
from .prod_v20_1_observer import ObserverOutput
from .prod_v20_1_belief import JointPairBelief, pair_nll
from .prod_v20_policy import ActualHistory, empty_history
from .prod_v20_3_training import hold_commands, legal_action_shuffle


class _Sums:
    def __init__(self):
        self.values: dict[str, list[float | int]] = {}

    def add(self, name: str, value: Tensor, valid: Tensor | None = None):
        sample = value.detach().double()
        if valid is not None:
            sample = sample[valid]
        sample = sample.flatten()
        if not torch.isfinite(sample).all():
            raise ValueError(f"non-finite development metric {name}")
        total, count = self.values.get(name, [0., 0])
        self.values[name] = [float(total) + float(sample.sum().cpu()), int(count) + sample.numel()]

    def result(self):
        return {name: dict(sum=total, count=count, mean=total / count if count else None)
                for name, (total, count) in self.values.items()}


def _ordered_mask(ordered, batch: int, device) -> Tensor:
    value = torch.as_tensor(ordered, device=device, dtype=torch.bool)
    if value.ndim == 1:
        value = value[:, None]
    return torch.broadcast_to(value, (batch, 2))


def _map_pair(belief: JointPairBelief) -> Tensor:
    """Select one maximum-weight whole pair and its most likely physical gauge."""
    index = belief.log_weights.argmax(-1)[..., None, None, None].expand(-1, -1, 1, 2, 2)
    pair = belief.means_px.gather(2, index).squeeze(2)
    probability = belief.identity_probability.gather(2, belief.log_weights.argmax(-1)[..., None]).squeeze(-1)
    return torch.where((probability < .5)[..., None, None], pair.flip(-2), pair)


def _conditional_log_density(belief: JointPairBelief, points: Tensor, ordered: Tensor) -> Tensor:
    """Spatial 4-D density conditional on both tips being visible.

    ``points`` [B,2,S,2,2]; unordered views use the normalized symmetrized
    density, not a hard maximum masquerading as a probability density.
    """
    covariance = belief.endpoint_joint_covariance.float()
    inverse = torch.linalg.inv(covariance)
    logdet = torch.linalg.slogdet(covariance).logabsdet
    weights = (belief.log_weights.float() + belief.visible_probability.float().clamp_min(1e-8).log().sum(-1)).log_softmax(-1)

    def assigned(x):
        delta = x[:, :, :, None].float().flatten(-2) - belief.means_px[:, :, None].float().flatten(-2)
        squared = torch.einsum("bvski,bvkij,bvskj->bvsk", delta, inverse, delta)
        component = -0.5 * (squared + logdet[:, :, None] + 4 * math.log(2 * math.pi))
        return component

    direct, swapped = assigned(points), assigned(points.flip(-2))
    identity = belief.identity_probability.float()[:, :, None]
    physical = torch.logaddexp(direct + identity.log(), swapped + torch.log1p(-identity))
    symmetric = torch.logaddexp(direct, swapped) - math.log(2)
    component = torch.where(ordered[..., None, None], physical, symmetric)
    return (component + weights[:, :, None]).logsumexp(-1)


def approximate_pair_hpd_coverage(belief: JointPairBelief, target_fm_px: Tensor, ordered=True,
                                  *, samples: int = 256, seed: int = 20100) -> dict[str, Tensor]:
    """Deterministic Monte Carlo approximation of conditional whole-pair HPD.

    Samples are drawn from the predicted mixture, not from the labels.  The
    density threshold is its empirical lower 50%/10% quantile.  Report this as
    approximate *conditional joint-pair* coverage, not exact marginal coverage
    or visibility calibration.  Finite sampling error is not a model score.
    """
    if samples < 32:
        raise ValueError("HPD Monte Carlo requires at least 32 predicted samples")
    belief.validate()
    batch = len(belief.log_weights)
    order = _ordered_mask(ordered, batch, belief.log_weights.device)
    if target_fm_px.shape != (batch, 2, 2, 2):
        raise ValueError("target pair must be [B,2,2,2]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weights = (belief.log_weights.detach().float().cpu() + belief.visible_probability.detach().float().cpu().clamp_min(1e-8).log().sum(-1)).softmax(-1)
    component = torch.multinomial(weights.reshape(-1, 4), samples, replacement=True, generator=generator).reshape(batch, 2, samples)
    means = belief.state_mean.detach().float().cpu().gather(2, component[..., None].expand(-1, -1, -1, 4))
    factor = torch.linalg.cholesky(belief.state_covariance.detach().float().cpu()).gather(2, component[..., None, None].expand(-1, -1, -1, 4, 4))
    noise = torch.randn(batch, 2, samples, 4, generator=generator)
    state = means + torch.einsum("bvsij,bvsj->bvsi", factor, noise)
    drawn = torch.stack((state[..., :2] - state[..., 2:], state[..., :2] + state[..., 2:]), -2)
    # Draw physical identity INSIDE the sampled component. Unordered labels
    # instead use its normalized symmetric density, independent of pi.
    identity = belief.identity_probability.detach().float().cpu().gather(2, component)
    swap_probability = torch.where(order.detach().cpu()[..., None], 1 - identity, torch.full_like(identity, .5))
    swap = torch.rand(batch, 2, samples, generator=generator) < swap_probability
    drawn = torch.where(swap[..., None, None], drawn.flip(-2), drawn).to(belief.center_px.device)
    density = _conditional_log_density(belief, drawn, order)
    actual_density = _conditional_log_density(belief, target_fm_px[:, :, None], order).squeeze(-1)
    return {
        "conditional_pair_hpd50_approx": actual_density >= torch.quantile(density, .5, dim=-1),
        "conditional_pair_hpd90_approx": actual_density >= torch.quantile(density, .1, dim=-1),
    }


@torch.no_grad()
def observer_metrics(observation: ObserverOutput, target_fm_px: Tensor, point_valid: Tensor,
                     ordered=True, *, seed: int = 20100, coverage_samples: int = 256) -> dict:
    belief = observation.belief
    belief.validate()
    batch = len(belief.log_weights)
    target = target_fm_px.to(belief.center_px).float()
    if target.shape != (batch, 2, 2, 2):
        raise ValueError("target pair must be [B,2,2,2]")
    valid = torch.as_tensor(point_valid, device=target.device, dtype=torch.bool)
    if valid.shape == (batch, 2):
        valid = valid[..., None].expand(-1, -1, 2)
    if valid.shape != (batch, 2, 2) or torch.any(valid & ~torch.isfinite(target).all(-1)):
        raise ValueError("valid spatial labels must be finite [B,2,2]")
    target = torch.where(valid[..., None], target, torch.zeros_like(target))
    order = _ordered_mask(ordered, batch, target.device)
    predicted = _map_pair(belief).float()
    same = (predicted - target).norm(dim=-1)
    swapped = (predicted.flip(-2) - target).norm(dim=-1)
    unordered_swap = (swapped * valid).sum(-1) < (same * valid).sum(-1)
    unordered_error = torch.where(unordered_swap[..., None], swapped, same)
    swap = ~order & ((swapped * valid).sum(-1) < (same * valid).sum(-1))
    predicted = torch.where(swap[..., None, None], predicted.flip(-2), predicted)
    error = (predicted - target).norm(dim=-1)
    nll = pair_nll(belief, target, valid, ordered=order, reduction="none", identity_aware=True)
    pair_valid = valid.all(-1)
    labeled_view = valid.any(-1)
    target_span = (target[..., 1, :] - target[..., 0, :]).norm(dim=-1)
    predicted_span = (predicted[..., 1, :] - predicted[..., 0, :]).norm(dim=-1)
    midpoint = (predicted.mean(-2) - target.mean(-2)).norm(dim=-1)
    coverage = approximate_pair_hpd_coverage(belief, target, order, samples=coverage_samples, seed=seed)
    pi = belief.identity_probability[..., None]
    physical_visibility = pi * belief.visible_probability + (1 - pi) * belief.visible_probability.flip(-1)
    visibility = (belief.weights[..., None] * physical_visibility).sum(-2)
    map_identity = belief.identity_probability.gather(2, belief.log_weights.argmax(-1)[..., None]).squeeze(-1)
    identity_ambiguous = torch.isclose(map_identity, torch.full_like(map_identity, .5), atol=1e-6, rtol=0)
    identity_uncertain = (map_identity >= .25) & (map_identity <= .75)
    output = {}
    for view, name in enumerate(("wrist", "column")):
        meter = _Sums()
        # Exact identity ties have no physical MAP decision. Report unordered
        # localization for them, and exclude them from physical ordered EPE.
        decidable = ~order[:, view] | ~identity_ambiguous[:, view]
        meter.add("tip_epe_px", error[:, view], valid[:, view] & decidable[:, None])
        meter.add("unordered_tip_epe_px", unordered_error[:, view], valid[:, view])
        for tip, label in enumerate(("F", "M")):
            meter.add(f"ordered_{label}_epe_px", error[:, view, tip], valid[:, view, tip] & order[:, view] & ~identity_ambiguous[:, view])
        meter.add("identity_exact_ambiguity_rate", identity_ambiguous[:, view].float(), labeled_view[:, view])
        meter.add("identity_probability_025_075_rate", identity_uncertain[:, view].float(), labeled_view[:, view])
        meter.add("pair_midpoint_epe_px", midpoint[:, view], pair_valid[:, view])
        meter.add("pair_span_absolute_error_px", (predicted_span - target_span).abs()[:, view], pair_valid[:, view])
        meter.add("map_pair_span_px", predicted_span[:, view], labeled_view[:, view])
        meter.add("joint_pair_nll", nll[:, view], labeled_view[:, view])
        meter.add("predicted_visible_mass_on_labeled_tips", visibility[:, view], valid[:, view])
        # A zero projected span can be physically valid.  This is descriptive,
        # never a hardware gate; conditional collapse has real open-pair labels.
        meter.add("projected_overlap_lt1px_rate", (predicted_span[:, view] < 1).float(), pair_valid[:, view])
        meter.add("collapse_given_labeled_span_gt8px_rate", (predicted_span[:, view] < 1).float(), pair_valid[:, view] & (target_span[:, view] > 8))
        for metric, values in coverage.items():
            meter.add(metric, values[:, view].float(), pair_valid[:, view])
        output[name] = meter.result()
    output["metadata"] = dict(evidence="training-development", point_readout="MAP whole pair, never independent marginal means",
                               coverage="Monte Carlo approximate conditional joint-pair HPD", coverage_samples=coverage_samples,
                               coverage_seed=seed, unseen_claim=False, visibility_calibration="not established by positive point labels",
                               ordered_density="per-component physical identity mixture before component marginalization",
                               ordered_epe="most likely physical gauge; exact pi=0.5 ties excluded and separately counted")
    return output


def _shuffle_history(history: ActualHistory, permutation: Tensor) -> ActualHistory:
    return ActualHistory(**{field.name: getattr(history, field.name)[permutation] for field in fields(history)})


def _reverse_history(history: ActualHistory) -> ActualHistory:
    """Reverse only present actual records; invalid padding remains on the left."""
    values = {field.name: getattr(history, field.name).clone() for field in fields(history)}
    for row in range(len(history.valid)):
        selected = history.valid[row].nonzero().flatten()
        for field in fields(history):
            values[field.name][row, selected] = getattr(history, field.name)[row, selected.flip(0)]
    return ActualHistory(**values)


def actor_interventions(core: V203Core, state: V203ActualState, task_embedding: Tensor,
                        *, permutation: Tensor | None = None) -> dict[str, Tensor]:
    """Inference-only matched endpoint probes; no state is written back."""
    batch = len(state.q_position)
    permutation = torch.arange(batch, device=task_embedding.device).roll(1) if permutation is None else permutation
    zero_history = empty_history(batch, device=state.q_position.device, dtype=state.q_position.dtype)
    scene_zero = replace(state.observation, scene_tokens=torch.zeros_like(state.observation.scene_tokens))
    variants = {
        "factual": (state, task_embedding),
        "scene_zero": (replace(state, observation=scene_zero), task_embedding),
        "task_blank": (state, torch.zeros_like(task_embedding)),
        "task_shuffle": (state, task_embedding[permutation]),
        "actor_history_zero": (replace(state, history=zero_history), task_embedding),
        "actor_history_shuffle": (replace(state, history=_shuffle_history(state.history, permutation)), task_embedding),
        "actor_history_reverse": (replace(state, history=_reverse_history(state.history)), task_embedding),
        "q_only_policy": (replace(state, observation=scene_zero, grounding_tracks=torch.zeros_like(state.grounding_tracks),
                           innovation_tracks=torch.zeros_like(state.innovation_tracks), history=zero_history), torch.zeros_like(task_embedding)),
    }
    result = {name: core.act(current, task).commands_position for name, (current, task) in variants.items()}
    # Mask RAW residual families before the shared encoder. Presence,
    # visibility/availability metadata and encoder bias remain matched.
    for mode in ("full", "belief_only", "q_only", "neither"):
        result[f"innovation_{mode}"] = core.act(state, task_embedding, innovation_mode=mode).commands_position
    return result


@torch.no_grad()
def replay_filter_history(core: V203Core, chain: Mapping, mode: str, permutation: Tensor):
    """Change only the process encoder history during a causal replay.

    The current endpoint RGB/q, actual command/readback targets and their clock
    are unchanged. The Actor later receives its factual raw history, isolating
    changes carried through forecasts and innovation. Actual-q localization
    must remain unchanged by a forecast-only history intervention in V20.3.
    """
    if mode not in ("zero", "reverse", "shuffle"):
        raise ValueError("filter process-history intervention must be zero/reverse/shuffle")
    q, observations, valid = chain["q_position"], chain["measurements"], chain["view_valid"]
    geometry = chain.get("geometry")
    first = observations[0]
    if geometry is not None and not getattr(core, "INITIAL_CONSUMES_RAW_MEASUREMENT", False):
        first = core.observer.assimilate(first, geometry.make_prior(q[:, 0]), valid[:, 0])
    state = core.initial_from_measurement(first, q[:, 0], valid[:, 0], chain.get("initial_time_s", q.new_zeros(len(q))), geometry=geometry)
    states = [state]
    for horizon in range(3):
        actual_history = state.history
        changed = {"zero": lambda: empty_history(len(q), device=q.device, dtype=q.dtype),
                   "reverse": lambda: _reverse_history(actual_history),
                   "shuffle": lambda: _shuffle_history(actual_history, permutation)}[mode]()
        prior_state = replace(state, history=changed)
        intervals = chain["time_deltas"][:, horizon]
        prediction = core.predict(prior_state, chain["commands_position"][:, horizon], intervals, geometry)
        state = detach_actual(core.complete_from_measurement(state, prediction, observations[horizon + 1],
            chain["measured_q_paths_position"][:, horizon], valid[:, horizon + 1],
            state.endpoint_time_s + intervals.sum(-1), geometry=geometry))
        states.append(state)
    return states


def actor_input_jacobians(core: V203Core, state: V203ActualState, task_embedding: Tensor,
                          *, seed: int = 20100, probes: int = 2) -> dict[str, dict]:
    """Independent RMS Jacobian estimates via output Rademacher projections.

    These are dependency probes, not proof of beneficial use.  The operation
    differentiates normalized commanded outputs w.r.t. independent input
    leaves and does not accumulate or mutate any model parameter gradient.
    """
    if probes < 1:
        raise ValueError("at least one Jacobian projection is required")
    with torch.enable_grad():
        scene = state.observation.scene_tokens.detach().clone().requires_grad_()
        ground = state.grounding_tracks.detach().clone().requires_grad_()
        innovation = state.innovation_tracks.detach().clone().requires_grad_()
        q = state.q_position.detach().clone().requires_grad_()
        task = task_embedding.detach().clone().requires_grad_()
        history = ActualHistory(**{field.name: getattr(state.history, field.name).detach().clone() for field in fields(state.history)})
        history = replace(history, readbacks_position=history.readbacks_position.requires_grad_(),
                          scene_pooled=history.scene_pooled.requires_grad_(), time_deltas=history.time_deltas.requires_grad_())
        isolated = replace(state, observation=replace(state.observation, scene_tokens=scene), grounding_tracks=ground,
                           innovation_tracks=innovation, q_position=q, history=history)
        prediction = core.act(isolated, task).commands_position / core.q_loss_scale
        leaves = {"scene": scene, "grounding": ground, "innovation": innovation, "q": q, "task": task,
                  "history_readback": history.readbacks_position, "history_scene": history.scene_pooled,
                  "history_time": history.time_deltas}
        squares = {name: torch.zeros_like(value) for name, value in leaves.items()}
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for index in range(probes):
            direction = (torch.randint(0, 2, prediction.shape, generator=generator) * 2 - 1).to(prediction)
            gradients = torch.autograd.grad((prediction * direction).sum(), list(leaves.values()), retain_graph=index + 1 < probes, allow_unused=True)
            for (name, _), gradient in zip(leaves.items(), gradients):
                if gradient is not None:
                    squares[name] += gradient.detach().square() / probes
        result = {name: dict(rms=float(value.mean().sqrt().cpu()), input_elements=value.numel(),
                             method="Rademacher estimated normalized-output Jacobian RMS", probes=probes)
                  for name, value in squares.items()}
        result["scene_wrist"] = dict(rms=float(squares["scene"][:, 0].mean().sqrt().cpu()), probes=probes)
        result["scene_column"] = dict(rms=float(squares["scene"][:, 1].mean().sqrt().cpu()), probes=probes)
        return result


def _servo_lag(q: Tensor, commands: Tensor, dt: Tensor, tau_s) -> Tensor:
    tau = torch.as_tensor(tau_s, device=q.device, dtype=q.dtype)
    if tau.shape not in (torch.Size([]), torch.Size([6])) or not torch.isfinite(tau).all() or not (tau > 0).all():
        raise ValueError("servo lag time constants must be positive scalar or per-joint six-vector")
    previous, predicted = q, []
    for step in range(5):
        alpha = -torch.expm1(-dt[:, step, None] / tau)
        previous = previous + alpha * (commands[:, step] - previous)
        predicted.append(previous)
    return torch.stack(predicted, 1)


@torch.no_grad()
def fit_servo_lag(training_chain: Mapping, *, evaluation_endpoint_ids=(),
                  candidate_tau_s: Tensor | None = None) -> dict:
    """Fit six time constants only on a declared disjoint training prefix.

    No held data is opened. The baseline is unavailable when excitation cannot
    identify every joint; an arbitrary default is never called fitted/strong.
    """
    if training_chain.get("evidence_role") != "servo-fit-training-prefix":
        raise ValueError("servo-lag fitting requires an explicit training-prefix role")
    endpoints = torch.as_tensor(training_chain["endpoint_index"]).flatten().cpu()
    excluded = torch.as_tensor(evaluation_endpoint_ids, dtype=torch.long).flatten().cpu()
    if len(excluded) and torch.isin(endpoints, excluded).any():
        raise ValueError("servo-lag training prefix overlaps evaluation endpoints")
    q = training_chain["q_position"][:, :-1].reshape(-1, 6)
    commands = training_chain["commands_position"].reshape(-1, 5, 6)
    actual = training_chain["measured_q_paths_position"][:, :, 1:].reshape(-1, 5, 6)
    dt = training_chain["time_deltas"].reshape(-1, 5)
    if not all(torch.isfinite(x).all() for x in (q, commands, actual, dt)) or not (dt > 0).all():
        raise ValueError("servo-lag fit requires finite positive-timed transitions")
    grid = torch.logspace(math.log10(.005), math.log10(2.), 128, device=q.device) if candidate_tau_s is None else torch.as_tensor(candidate_tau_s, device=q.device)
    if grid.ndim != 1 or len(grid) < 2 or not torch.isfinite(grid).all() or not (grid > 0).all():
        raise ValueError("servo-lag grid must contain positive finite candidates")
    scores = torch.stack([(_servo_lag(q, commands, dt, candidate) - actual).square().mean((0, 1)) for candidate in grid])
    best = scores.argmin(0)
    excitation = (commands - q[:, None]).square().mean((0, 1)).sqrt()
    identified = (excitation > 1.e-3) & ((scores.max(0).values - scores.min(0).values) > 1.e-8)
    return dict(kind="training-prefix-fitted-per-joint-first-order-servo", available=bool(identified.all()),
                tau_s=grid[best].cpu().tolist(), identified_joints=identified.cpu().tolist(),
                training_rows=len(training_chain["q_position"]), training_transitions=len(q),
                training_endpoint_count=len(endpoints.unique()), evaluation_endpoint_overlap=False,
                excitation_rms_position=excitation.cpu().tolist(),
                fit_mse_position2=scores.min(0).values.cpu().tolist(),
                candidate_range_s=[float(grid.min()), float(grid.max())], candidates=len(grid),
                evidence="training-development", held_episode_opened=False)


def _restore_model_mode(function):
    """An evaluation call must not permanently change a stage's train flags."""
    @wraps(function)
    def wrapped(core, *args, **kwargs):
        modes = [(module, module.training) for module in core.modules()]
        core.eval()
        try:
            return function(core, *args, **kwargs)
        finally:
            # Restore per-module flags, including independently frozen stages.
            for module, training in modes:
                module.training = training
    return wrapped


@torch.no_grad()
@_restore_model_mode
def evaluate_chain(core: V203Core, chain: Mapping, *, jacobians: bool = False,
                   servo_lag_fit: Mapping | None = None) -> dict:
    """One fixed supplied archive-development batch, H0..H3 causal replay.

    This is a diagnostic report, NOT a model-selection or robot-execution gate.
    The optional ``target_fm_px``/``point_valid`` arrays are spatial labels for
    H0..H4.  Archive trajectories without these cannot claim fingertip EPE.
    """
    q, commands = chain["q_position"], chain["commands_position"]
    paths, intervals = chain["measured_q_paths_position"], chain["time_deltas"]
    observations, valid, task = chain["measurements"], chain["view_valid"], chain["task_embedding"]
    batch = len(q)
    if q.shape != (batch, 5, 6) or paths.shape != (batch, 4, 6, 6) or commands.shape != (batch, 4, 5, 6) or len(observations) != 5:
        raise ValueError("evaluation requires exactly one H0..H4 chain per row")
    if intervals.shape != (batch, 4, 5) or not (intervals > 0).all() or not torch.isfinite(intervals).all():
        raise ValueError("evaluation intervals must be finite positive seconds")
    if not torch.allclose(paths[:, :, 0], q[:, :-1], atol=1e-4, rtol=0) or not torch.allclose(paths[:, :, -1], q[:, 1:], atol=1e-4, rtol=0):
        raise ValueError("evaluation q/image/readback chain is not aligned")
    geometry = chain.get("geometry")
    first = observations[0]
    if geometry is not None and not getattr(core, "INITIAL_CONSUMES_RAW_MEASUREMENT", False):
        first = core.observer.assimilate(first, geometry.make_prior(q[:, 0]), valid[:, 0])
    state = core.initial_from_measurement(first, q[:, 0], valid[:, 0], chain.get("initial_time_s", q.new_zeros(batch)), geometry=geometry)
    permutation = torch.arange(batch, device=q.device).roll(1)
    filter_states = {mode: replay_filter_history(core, chain, mode, permutation)
                     for mode in ("zero", "reverse", "shuffle")}
    sums = _Sums()
    spatial, jacobian_report = {}, {}
    for horizon in range(4):
        commanded = commands[:, horizon]
        actual = paths[:, horizon, 1:]
        prediction = core.predict(state, commanded, intervals[:, horizon], geometry)
        holding = hold_commands(state.q_position)
        shuffled = legal_action_shuffle(commanded, state.q_position, core.actor.position_lower,
                                          core.actor.position_upper, permutation)
        dynamics = {
            "factual": prediction.predicted_q_path[:, 1:],
            "hold": holding,
            "direct_command": commanded,
            "hold_action": core.lewm(state.q_position, holding, state.history, intervals[:, horizon]).predicted_q_path[:, 1:],
            "legal_action_shuffle": core.lewm(state.q_position, shuffled, state.history, intervals[:, horizon]).predicted_q_path[:, 1:],
        }
        if servo_lag_fit is not None and servo_lag_fit.get("available"):
            if servo_lag_fit.get("kind") != "training-prefix-fitted-per-joint-first-order-servo" or servo_lag_fit.get("evaluation_endpoint_overlap") is not False:
                raise ValueError("unverified servo-lag baseline provenance")
            dynamics["servo_lag_fitted"] = _servo_lag(state.q_position, commanded, intervals[:, horizon], servo_lag_fit["tau_s"])
        for name, output in dynamics.items():
            normalized = (output - actual).abs() / core.q_loss_scale
            sums.add(f"lewm/{name}/future_normalized_mae", normalized.mean((1, 2)))
            for joint in range(6):
                sums.add(f"lewm/{name}/joint{joint}_mae_position", (output - actual).abs()[:, :, joint].mean(-1))
        actions = actor_interventions(core, state, task, permutation=permutation)
        for mode, replay in filter_states.items():
            changed = replace(replay[horizon], history=state.history)
            actions[f"filter_history_{mode}"] = core.act(changed, task).commands_position
        for name, action in actions.items():
            error = ((action - commanded).abs() / core.q_loss_scale).mean((1, 2))
            sums.add(f"actor/{name}/normalized_mae", error)
            sums.add(f"actor/h{horizon}/{name}/normalized_mae", error)
            sums.add(f"actor/{name}/delta_from_factual", ((action - actions["factual"]).abs() / core.q_loss_scale).mean((1, 2)))
        sums.add("actor/repeat/normalized_mae", ((holding - commanded).abs() / core.q_loss_scale).mean((1, 2)))
        if "target_fm_px" in chain and "point_valid" in chain:
            spatial[f"h{horizon}/raw_measurement"] = observer_metrics(observations[horizon], chain["target_fm_px"][:, horizon], chain["point_valid"][:, horizon], chain.get("ordered", True))
            spatial[f"h{horizon}/filtered"] = observer_metrics(state.observation, chain["target_fm_px"][:, horizon], chain["point_valid"][:, horizon], chain.get("ordered", True))
        if jacobians and horizon == 3:
            jacobian_report = actor_input_jacobians(core, state, task)
        if horizon < 3:
            next_time = state.endpoint_time_s + intervals[:, horizon].sum(-1)
            state = detach_actual(core.complete_from_measurement(state, prediction, observations[horizon + 1],
                                   paths[:, horizon], valid[:, horizon + 1], next_time, geometry=geometry))
    return dict(evidence="training-development", unseen_claim=False, chain_rows=batch, endpoints_evaluated=4,
                actual_localization_geometry_available=geometry is not None,
                metric_geometry=bool(getattr(geometry, "metric_3d", False)),
                geometry_kind=getattr(geometry, "geometry_kind", "unavailable"),
                metrics=sums.result(), spatial=spatial, jacobians=jacobian_report,
                protocol=dict(readback_target="q1..q5 only", history="same current endpoint, actual-only factual replay",
                              actor_history_interventions="explicit raw history only; filtered grounding/innovation held fixed",
                              filter_history_interventions="change forecast history only; actual-q localization and factual Actor history remain fixed",
                              innovation_interventions="pre-encoder belief/q residual masks; shared presence metadata retained",
                              shuffle="donor increments reanchored at recipient q and bounded", shuffle_nontrivial=batch > 1,
                              task_shuffle_changed_rows=int((task != task[permutation]).any(-1).sum().cpu()),
                              geometry_adapter_available=geometry is not None,
                              servo_lag_fit=dict(servo_lag_fit) if servo_lag_fit is not None else {"available": False, "reason": "training prefix has not been fitted"},
                              held_episode_access=False, robot_commands_sent=False))
