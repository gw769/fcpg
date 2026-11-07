"""Planning utilities for the geometry-grounded LeWM latent dynamics model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .v21_cem_trace import record_cem_event


class LatentDynamicsModel(Protocol):
    def rollout_embeddings(
        self,
        history_embedding: torch.Tensor,
        history_action: torch.Tensor,
        future_action: torch.Tensor,
    ) -> torch.Tensor: ...

    def goal_cost(
        self,
        predicted_embedding: torch.Tensor,
        goal_embedding: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class CEMPlan:
    action_sequence: torch.Tensor
    cost: torch.Tensor
    distribution_mean: torch.Tensor
    distribution_std: torch.Tensor
    # Per-batch, already constrained search candidates. Historical costs are
    # provenance only: consumers must re-evaluate before selecting a command.
    retained_candidates: tuple[torch.Tensor, ...] = ()
    retained_costs: tuple[torch.Tensor, ...] = ()
    retained_sources: tuple[torch.Tensor, ...] = ()  # [K,2]: iteration, sample


class GroundedCEMPlanner:
    """Cross-entropy planner over recursive LeWM latent rollouts.

    The planner is environment-independent.  A MetaWorld rollout wrapper only
    needs to maintain the last ``history_size`` grounded observations, encode a
    visual goal, and execute the first ``receding_horizon`` actions from the
    returned sequence.
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        num_samples: int = 512,
        num_elites: int = 64,
        iterations: int = 5,
        momentum: float = 0.1,
        min_std: float = 0.05,
        action_low: float | tuple[float, ...] = -1.0,
        action_high: float | tuple[float, ...] = 1.0,
    ) -> None:
        if horizon <= 0 or action_dim <= 0:
            raise ValueError("horizon and action_dim must be positive")
        if not 0 < num_elites <= num_samples:
            raise ValueError("num_elites must be in [1, num_samples]")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.num_samples = int(num_samples)
        self.num_elites = int(num_elites)
        self.iterations = int(iterations)
        self.momentum = float(momentum)
        self.min_std = float(min_std)
        self.action_low = action_low
        self.action_high = action_high

    def _bound(
        self,
        value: float | tuple[float, ...],
        reference: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if result.ndim == 0:
            result = result.expand(self.action_dim)
        if result.shape != (self.action_dim,):
            raise ValueError(f"{name} must be scalar or have shape {(self.action_dim,)}")
        return result

    @torch.no_grad()
    def plan(
        self,
        model: LatentDynamicsModel,
        history_embedding: torch.Tensor,
        history_action: torch.Tensor,
        goal_embedding: torch.Tensor,
        initial_mean: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        constrain: object | None = None,
        trace: object | None = None,
        admissible: object | None = None,
        initial_std: torch.Tensor | None = None,
    ) -> CEMPlan:
        if history_embedding.ndim != 3:
            raise ValueError("history_embedding must have shape [B, H, D]")
        batch_size = history_embedding.shape[0]
        low = self._bound(self.action_low, history_embedding, "action_low")
        high = self._bound(self.action_high, history_embedding, "action_high")
        if not bool(torch.all(low < high)):
            raise ValueError("every action_low value must be below action_high")
        shape = (batch_size, self.horizon, self.action_dim)
        if initial_mean is None:
            mean = ((low + high) / 2.0).view(1, 1, -1).expand(shape).clone()
        else:
            if initial_mean.shape != shape:
                raise ValueError(f"initial_mean must have shape {shape}")
            mean = initial_mean.clone().clamp(low, high)
        if initial_std is None:
            std = ((high - low) / 2.0).view(1, 1, -1).expand(shape).clone()
        else:
            if (initial_std.shape != shape or not torch.isfinite(initial_std).all()
                    or not (initial_std > 0).all()):
                raise ValueError(f"initial_std must be finite, positive and have shape {shape}")
            std = initial_std.to(mean).clone().clamp_min(self.min_std)
        low_view = low.view(1, 1, 1, -1)
        high_view = high.view(1, 1, 1, -1)
        retained, retained_costs, retained_sources = ([[] for _ in range(batch_size)] for _ in range(3))

        for iteration in range(self.iterations):
            noise = torch.randn(
                batch_size,
                self.num_samples,
                self.horizon,
                self.action_dim,
                device=history_embedding.device,
                dtype=history_embedding.dtype,
                generator=generator,
            )
            candidates = mean[:, None] + std[:, None] * noise
            candidates = candidates.clamp(low_view, high_view)
            before_constraint = candidates.detach().clone() if trace is not None else None
            if constrain is not None:
                candidates = constrain(candidates)
                if candidates.shape != (batch_size, self.num_samples, self.horizon, self.action_dim):
                    raise ValueError("constrain must keep candidate shape [B, N, H, A]")
                candidates = candidates.clamp(low_view, high_view)
            predicted = model.rollout_embeddings(
                history_embedding,
                history_action,
                candidates,
            )
            cost = model.goal_cost(predicted, goal_embedding)
            if cost.shape != (batch_size, self.num_samples):
                raise ValueError("candidate costs must have shape [B,N]")
            eligible = (torch.isfinite(candidates).all(dim=(-2, -1)) & torch.isfinite(cost)
                        & torch.isfinite(predicted).reshape(batch_size, self.num_samples, -1).all(-1))
            if admissible is not None:
                admitted = admissible(candidates, predicted, cost)
                if (not isinstance(admitted, torch.Tensor) or admitted.shape != eligible.shape
                        or admitted.dtype != torch.bool):
                    raise ValueError("admissible must return a bool mask [B,N]")
                eligible &= admitted.to(eligible.device)
            for batch in range(batch_size):
                indices = torch.nonzero(eligible[batch], as_tuple=False).flatten()
                indices = indices[torch.argsort(cost[batch, indices])[:self.num_elites]]
                retained[batch].append(candidates[batch, indices].detach().clone())
                retained_costs[batch].append(cost[batch, indices].detach().clone())
                retained_sources[batch].append(torch.stack((torch.full_like(indices, iteration), indices), -1))
            elite_index = torch.topk(
                cost.masked_fill(~eligible, float('inf')),
                k=self.num_elites,
                dim=1,
                largest=False,
            ).indices
            gather_index = elite_index[..., None, None].expand(
                -1,
                -1,
                self.horizon,
                self.action_dim,
            )
            elite = torch.gather(candidates, dim=1, index=gather_index)
            elite_valid = torch.gather(eligible, dim=1, index=elite_index)
            if bool(elite_valid.all()):
                # Preserve the all-valid numerical path for existing callers.
                elite_mean = elite.mean(dim=1)
                elite_std = elite.std(dim=1, unbiased=False).clamp_min(self.min_std)
            else:
                mask = elite_valid[..., None, None]
                count = mask.sum(dim=1).clamp_min(1)
                safe_elite = torch.where(mask, elite, torch.zeros_like(elite))
                elite_mean = safe_elite.sum(dim=1) / count
                centered = torch.where(mask, elite - elite_mean[:, None], torch.zeros_like(elite))
                elite_std = (centered.square().sum(dim=1) / count).sqrt().clamp_min(self.min_std)
            mean_before, std_before = mean, std
            has_elite = elite_valid.any(dim=1)[:, None, None]
            mean = torch.where(has_elite,
                self.momentum * mean + (1.0 - self.momentum) * elite_mean, mean)
            std = torch.where(has_elite,
                self.momentum * std + (1.0 - self.momentum) * elite_std, std)
            record_cem_event(trace, 'cem_iteration', iteration=iteration,
                mean_before=mean_before, std_before=std_before, noise=noise,
                before_constraint=before_constraint, candidates=candidates,
                predicted=predicted, cost=cost, elite_index=elite_index,
                elite=elite, eligible=eligible, elite_valid=elite_valid,
                mean_after=mean, std_after=std,
                momentum=self.momentum, min_std=self.min_std)

        action_sequence = mean.clamp(low, high)
        if constrain is not None:
            action_sequence = constrain(action_sequence[:, None]).clamp(low_view, high_view)[:, 0]
        predicted = model.rollout_embeddings(
            history_embedding,
            history_action,
            action_sequence[:, None],
        )
        final_cost = model.goal_cost(predicted, goal_embedding)
        final_valid = (torch.isfinite(action_sequence).all(dim=(-2, -1))[:, None]
                       & torch.isfinite(final_cost)
                       & torch.isfinite(predicted).reshape(batch_size, 1, -1).all(-1))
        if admissible is not None:
            admitted = admissible(action_sequence[:, None], predicted, final_cost)
            if (not isinstance(admitted, torch.Tensor) or admitted.shape != final_valid.shape
                    or admitted.dtype != torch.bool):
                raise ValueError("admissible must return a bool mask [B,N]")
            final_valid &= admitted.to(final_valid.device)
        final_cost = final_cost.masked_fill(~final_valid, float('inf'))[:, 0]
        record_cem_event(trace, 'cem_final_mean', action_sequence=action_sequence,
            predicted=predicted, cost=final_cost, distribution_mean=mean,
            distribution_std=std)
        retained = tuple(torch.cat(values) for values in retained)
        retained_costs = tuple(torch.cat(values) for values in retained_costs)
        retained_sources = tuple(torch.cat(values) for values in retained_sources)
        record_cem_event(trace, 'cem_retained_candidates', candidates=retained,
            costs=retained_costs, sources=retained_sources,
            costs_require_final_revalidation=True)
        return CEMPlan(
            action_sequence=action_sequence,
            cost=final_cost,
            distribution_mean=mean,
            distribution_std=std,
            retained_candidates=retained,
            retained_costs=retained_costs,
            retained_sources=retained_sources,
        )
