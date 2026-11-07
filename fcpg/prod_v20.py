"""V20 mechanical-prior visual feedback stack.

All public joint values are LeRobot calibrated positions, NOT degrees.  The
observer's geometry adapter owns conversion into physical joint coordinates.
This module never opens a camera/serial port and never sends robot commands.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import nn

from .prod_v20_observer import PairBelief, ObserverOutput, V20PairObserver
from .prod_v20_policy import ActualHistory, V20Actor, V20LeWM, empty_history, append_completed

VERSION = "20.0.0"
CONTRACT = "mechanical-prior-paired-visual-feedback-v1"
POSITION_UNIT = "lerobot_calibrated_position"
VIEW_ORDER = ("wrist", "column")


@dataclass(frozen=True)
class V20Config:
    seed: int = 20000
    components: int = 4
    dimension: int = 192
    history_steps: int = 3
    command_steps: int = 5
    task_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    position_unit: str = POSITION_UNIT
    actor_mode: str = "full"

    def __post_init__(self):
        if (self.components, self.dimension, self.history_steps, self.command_steps) != (4, 192, 3, 5):
            raise ValueError("V20 architecture is fixed; do not silently create another experiment")
        if self.position_unit != POSITION_UNIT or self.actor_mode not in ("full", "no_innovation"):
            raise ValueError("invalid V20 contract")


def detach_belief(belief: PairBelief) -> PairBelief:
    return PairBelief(**{field.name: getattr(belief, field.name).detach() for field in fields(belief)})


def belief_statistics(belief: PairBelief) -> torch.Tensor:
    """Per-pair statistics, never independent marginal means disguised as jaws."""
    covariance = belief.covariance_px2.float() / (224.0 ** 2)
    return torch.cat((
        belief.center_px.float() / 224.0,
        belief.direction.float(),
        belief.opening_px.float()[..., None] / 224.0,
        covariance.flatten(-2),
        belief.visible_probability.float(),
        belief.log_weights.float().exp()[..., None],
    ), dim=-1)  # 12


def matched_prior_statistics(actual: PairBelief, prior: PairBelief) -> torch.Tensor:
    """Match each observed PAIR to the entire predicted mixture.

    Mixture responsibilities retain uncertainty and remove arbitrary component
    numbering.  Joint permutation of a component's G/E remains well defined.
    No independent F/M marginal matching is performed.
    """
    current = actual.means_px.float().flatten(-2)
    predicted = prior.means_px.float().flatten(-2)
    distance = (current[..., :, None, :] - predicted[..., None, :, :]).square().sum(-1)
    variance = prior.covariance_px2.float().diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(4.)
    responsibility = (prior.log_weights.float()[..., None, :] - distance / (2. * variance[..., None, :])).softmax(-1)
    return responsibility @ belief_statistics(prior)


class V20ControlEncoding(nn.Module):
    """Actor-owned encoders; observation/dynamics losses cannot update them."""
    def __init__(self):
        super().__init__()
        self.grounding = nn.Sequential(nn.LayerNorm(396), nn.Linear(396, 192), nn.GELU(), nn.Linear(192, 192))
        self.innovation = nn.Sequential(nn.Linear(48, 192), nn.GELU(), nn.Linear(192, 192))

    def forward(self, observation: ObserverOutput, prior: PairBelief | None,
                measured_q_path: torch.Tensor | None, predicted_q_path: torch.Tensor | None,
                q_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = belief_statistics(observation.belief)
        grounding = self.grounding(torch.cat((observation.component_features.float(), stats), -1))
        batch = len(stats)
        if prior is None:
            residual = torch.zeros_like(stats)
            present = stats.new_zeros(batch, 2, 4, 1)
        else:
            residual = stats - matched_prior_statistics(observation.belief, prior)
            present = stats.new_ones(batch, 2, 4, 1)
        path_residual = stats.new_zeros(batch, 30)
        if measured_q_path is not None:
            if predicted_q_path is None or measured_q_path.shape != (batch, 6, 6) or predicted_q_path.shape != (batch, 6, 6):
                raise ValueError("innovation requires paired actual/predicted six-state paths")
            path_residual = ((measured_q_path[:, 1:].float() - predicted_q_path[:, 1:].float()) / q_scale).flatten(1)
        # Moment residual, entropy change and observation availability belong
        # to the same current component, not to an unordered second track set.
        entropy = -(observation.belief.log_weights.float().exp() * observation.belief.log_weights.float()).sum(-1)
        if prior is not None:
            entropy = entropy + (prior.log_weights.float().exp() * prior.log_weights.float()).sum(-1)
        else:
            entropy = entropy * 0
        visibility_delta = residual[..., 9:11]
        covariance_delta = residual[..., 5:7]
        encoded = torch.cat((residual, path_residual[:, None, None].expand(-1, 2, 4, -1),
                             entropy[..., None, None].expand(-1, -1, 4, 1), present,
                             visibility_delta, covariance_delta), -1)
        return grounding, self.innovation(encoded)


@dataclass(frozen=True)
class V20ActualState:
    observation: ObserverOutput
    q_position: torch.Tensor
    history: ActualHistory
    grounding_tracks: torch.Tensor
    innovation_tracks: torch.Tensor
    endpoint_time_s: torch.Tensor
    view_valid: torch.Tensor


@dataclass(frozen=True)
class V20Prediction:
    commands_position: torch.Tensor
    predicted_q_path: torch.Tensor
    q_uncertainty: torch.Tensor
    prior: PairBelief
    source_q_position: torch.Tensor
    source_time_s: torch.Tensor
    time_deltas: torch.Tensor


class V20Core(nn.Module):
    def __init__(self, config: V20Config = V20Config()):
        super().__init__()
        self.config = config
        self.observer = V20PairObserver()
        self.lewm = V20LeWM()
        self.actor = V20Actor()
        self.control_encoding = V20ControlEncoding()
        self.register_buffer("q_loss_scale", torch.ones(6))

    def freeze_for_stage(self, stage: str):
        self.requires_grad_(False)
        self.eval()
        if stage in ("appearance", "fcpg", "refit"):
            self.observer.requires_grad_(True).train()
        elif stage == "lewm":
            self.lewm.requires_grad_(True).train()
        elif stage in ("actor", "dense", "fish"):
            self.actor.requires_grad_(True).train()
            self.control_encoding.requires_grad_(True).train()
        else:
            raise ValueError(f"unknown training stage {stage}")

    def initial(self, rgb: torch.Tensor, dino_tokens: torch.Tensor,
                q_position: torch.Tensor, view_valid: torch.Tensor,
                endpoint_time_s: torch.Tensor, geometry: Any = None) -> V20ActualState:
        prior = geometry.make_prior(q_position) if geometry is not None else None
        observation = self.observer(rgb, dino_tokens, view_valid, prior=prior)
        return self.initial_from_measurement(observation, q_position, view_valid, endpoint_time_s)

    def initial_from_measurement(self, observation: ObserverOutput, q_position: torch.Tensor,
                                 view_valid: torch.Tensor, endpoint_time_s: torch.Tensor) -> V20ActualState:
        grounding, innovation = self.control_encoding(observation, None, None, None, self.q_loss_scale)
        return V20ActualState(observation, q_position, empty_history(len(q_position), device=q_position.device, dtype=q_position.dtype),
                              grounding, innovation, endpoint_time_s, view_valid)

    def act(self, state: V20ActualState, task_embedding: torch.Tensor):
        innovation = state.innovation_tracks
        if self.config.actor_mode == "no_innovation":
            innovation = torch.zeros_like(innovation)
        return self.actor(state.observation.scene_tokens, state.grounding_tracks, innovation,
                          state.q_position, state.history, task_embedding, view_valid=state.view_valid)

    def predict(self, state: V20ActualState, commands_position: torch.Tensor,
                time_deltas: torch.Tensor, geometry: Any = None) -> V20Prediction:
        output = self.lewm(state.q_position, commands_position, state.history, time_deltas)
        if geometry is None:
            # Without actual extrinsics there is no justified pixel transport.
            # Preserve the last actual spatial distribution, still predicting q.
            prior = detach_belief(state.observation.belief)
        else:
            with torch.no_grad():
                prior = geometry.propagate(detach_belief(state.observation.belief), state.q_position,
                                           output.predicted_q_path[:, -1].detach(), output.q_uncertainty[:, -1].detach())
        return V20Prediction(commands_position, output.predicted_q_path, output.q_uncertainty,
                             prior, state.q_position, state.endpoint_time_s, time_deltas)

    def complete_from_measurement(self, previous: V20ActualState, prediction: V20Prediction,
                                  measurement: ObserverOutput, measured_q_path: torch.Tensor,
                                  view_valid: torch.Tensor, endpoint_time_s: torch.Tensor,
                                  actual_time_deltas: torch.Tensor | None = None) -> V20ActualState:
        if not torch.allclose(prediction.source_q_position, previous.q_position, atol=1.e-4, rtol=0) or not torch.allclose(prediction.source_time_s, previous.endpoint_time_s, atol=1.e-6, rtol=0):
            raise ValueError("prediction belongs to a different actual endpoint")
        if measured_q_path.shape != (len(previous.q_position), 6, 6):
            raise ValueError("actual readback path must contain q0 and five readbacks")
        if not torch.allclose(measured_q_path[:, 0], previous.q_position, atol=1.e-4, rtol=0):
            raise ValueError("completed path starts at a different endpoint")
        # Raw current image likelihood is incorporated exactly once.
        observation = self.observer.assimilate(measurement, prediction.prior, view_valid)
        grounding, innovation = self.control_encoding(observation, prediction.prior,
                                                       measured_q_path, prediction.predicted_q_path,
                                                       self.q_loss_scale)
        elapsed = prediction.time_deltas if actual_time_deltas is None else actual_time_deltas
        if elapsed.shape != prediction.time_deltas.shape or not bool((elapsed > 0).all()):
            raise ValueError("actual transition time intervals are invalid")
        history = append_completed(previous.history, prediction.commands_position, measured_q_path,
                                   previous.observation.scene_tokens.mean(-2), elapsed)
        return V20ActualState(observation, measured_q_path[:, -1], history, grounding, innovation,
                              endpoint_time_s, view_valid)


def detach_actual(state: V20ActualState) -> V20ActualState:
    observation = state.observation
    detached_observation = ObserverOutput(**{
        field.name: (detach_belief(getattr(observation, field.name)) if field.name == "belief"
                     else getattr(observation, field.name).detach() if isinstance(getattr(observation, field.name), torch.Tensor)
                     else getattr(observation, field.name)) for field in fields(observation)
    })
    return V20ActualState(detached_observation, state.q_position.detach(), state.history,
                          state.grounding_tracks.detach(), state.innovation_tracks.detach(),
                          state.endpoint_time_s.detach(), state.view_valid.detach())
