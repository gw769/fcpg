"""Multi-scale causal history memory for a frozen SO-101 LeWM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fcpg.so101_state_residual_lewm import StateResidualLeWM


@dataclass
class MultiScaleMemoryOutput:
    embedding: torch.Tensor
    memory_embedding: torch.Tensor
    adapter_embedding: torch.Tensor
    target_embedding: torch.Tensor
    backbone_prediction: torch.Tensor
    prediction: torch.Tensor
    target_state: torch.Tensor
    command_state: torch.Tensor
    backbone_state_prediction: torch.Tensor
    state_prediction: torch.Tensor


class MultiScaleHistoryLeWM(nn.Module):
    """Add causal long-history tokens without changing the three-frame backbone.

    The memory entries must precede the backbone's oldest recent observation.
    They contain cached visual embeddings, measured joint state, the following
    executed action chunk, and their age in action chunks.  A zero-initialized
    output projection makes a fresh adapter exactly match its frozen backbone.
    """

    def __init__(
        self,
        backbone: StateResidualLeWM,
        *,
        latent_dim: int = 192,
        model_dim: int = 384,
        heads: int = 6,
        latent_residual_cap: float = 0.1,
        state_residual_cap: float = 0.25,
        recent_history: int = 3,
        use_recent_reliability: bool = False,
        use_fourier_age_encoding: bool = False,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or model_dim <= 0 or heads <= 0:
            raise ValueError("memory model dimensions must be positive")
        if model_dim % heads:
            raise ValueError("memory model_dim must be divisible by heads")
        if latent_dim != backbone.latent_dim:
            raise ValueError("memory latent_dim must match the LeWM backbone")
        if latent_residual_cap <= 0 or state_residual_cap <= 0:
            raise ValueError("memory residual caps must be positive")
        if recent_history <= 0:
            raise ValueError("recent_history must be positive")

        self.backbone = backbone
        self.latent_dim = int(latent_dim)
        self.model_dim = int(model_dim)
        self.heads = int(heads)
        self.latent_residual_cap = float(latent_residual_cap)
        self.state_residual_cap = float(state_residual_cap)
        self.recent_history = int(recent_history)
        self.use_recent_reliability = bool(use_recent_reliability)
        self.use_fourier_age_encoding = bool(use_fourier_age_encoding)

        fused_dim = self.latent_dim * 3
        self.query_projection = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, self.model_dim),
        )
        self.memory_projection = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, self.model_dim),
        )
        self.age_projection = nn.Sequential(
            nn.Linear(10 if self.use_fourier_age_encoding else 2, 64),
            nn.GELU(),
            nn.Linear(64, self.model_dim),
        )
        self.reliability_projection = (
            nn.Sequential(
                nn.Linear(self.recent_history, 64),
                nn.GELU(),
                nn.Linear(64, self.model_dim),
            )
            if self.use_recent_reliability
            else None
        )
        self.query_norm = nn.LayerNorm(self.model_dim)
        self.memory_norm = nn.LayerNorm(self.model_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.model_dim,
            self.heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.model_dim)
        self.latent_head = nn.Linear(self.model_dim, self.latent_dim)
        self.state_head = nn.Linear(self.model_dim, backbone.state_dim)
        nn.init.zeros_(self.latent_head.weight)
        nn.init.zeros_(self.latent_head.bias)
        nn.init.zeros_(self.state_head.weight)
        nn.init.zeros_(self.state_head.bias)

        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> "MultiScaleHistoryLeWM":
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Expose the shared visual encoder to evaluation and retrieval tools."""
        return self.backbone.encode(batch)

    def encode_memory_age(self, memory_age: torch.Tensor) -> torch.Tensor:
        """Encode causal age while preserving exact v6 checkpoint shapes."""
        base = torch.stack(
            (torch.log1p(memory_age), torch.reciprocal(1.0 + memory_age)), dim=-1
        )
        if not self.use_fourier_age_encoding:
            return base
        frequencies = memory_age.new_tensor((1.0, 2.0, 4.0, 8.0))
        phase = memory_age[..., None] / frequencies
        return torch.cat((base, torch.sin(phase), torch.cos(phase)), dim=-1)

    def _validate_memory(
        self,
        memory_embedding: torch.Tensor,
        memory_actions: torch.Tensor,
        memory_state: torch.Tensor,
        memory_age: torch.Tensor,
        memory_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        if memory_embedding.ndim != 3:
            raise ValueError("memory_embedding must have shape [B,M,D]")
        if memory_embedding.shape[-1] != self.latent_dim:
            raise ValueError("memory_embedding has the wrong latent dimension")
        if memory_actions.ndim != 3 or memory_state.ndim != 3:
            raise ValueError("memory actions and state must be rank-three tensors")
        if memory_embedding.shape[:2] != memory_actions.shape[:2]:
            raise ValueError("memory embeddings and actions must align")
        if memory_embedding.shape[:2] != memory_state.shape[:2]:
            raise ValueError("memory embeddings and state must align")
        if memory_state.shape[-1] != self.backbone.state_dim:
            raise ValueError("memory_state has the wrong state dimension")
        if memory_age.shape != memory_embedding.shape[:2]:
            raise ValueError("memory_age must have shape [B,M]")
        # Value checks on CUDA would synchronize every training forward.  The
        # dataset validates these values before transfer; retain the fail-fast
        # check for CPU callers and unit tests without stalling the hot path.
        if memory_age.device.type == "cpu" and (
            not bool(torch.isfinite(memory_age).all())
            or bool((memory_age <= 0).any())
        ):
            raise ValueError("memory_age must contain finite positive chunk ages")
        if memory_valid is None:
            valid = torch.ones(
                memory_embedding.shape[:2],
                dtype=torch.bool,
                device=memory_embedding.device,
            )
        else:
            if memory_valid.shape != memory_embedding.shape[:2]:
                raise ValueError("memory_valid must have shape [B,M]")
            valid = memory_valid.to(device=memory_embedding.device, dtype=torch.bool)
        if valid.device.type == "cpu" and bool((~valid.any(dim=1)).any()):
            raise ValueError("every sample must contain at least one valid memory token")
        return valid

    def _augment(
        self,
        *,
        backbone_prediction: torch.Tensor,
        backbone_state_prediction: torch.Tensor,
        actions: torch.Tensor,
        memory_embedding: torch.Tensor,
        memory_actions: torch.Tensor,
        memory_state: torch.Tensor,
        memory_age: torch.Tensor,
        memory_valid: torch.Tensor | None,
        recent_visual_valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = self._validate_memory(
            memory_embedding,
            memory_actions,
            memory_state,
            memory_age,
            memory_valid,
        )
        action_embedding = self.backbone.base.action_encoder(actions)
        memory_action_embedding = self.backbone.base.action_encoder(memory_actions)
        query_state_embedding = self.backbone.state_encoder(backbone_state_prediction)
        memory_state_embedding = self.backbone.state_encoder(memory_state)

        query = self.query_projection(
            torch.cat(
                (backbone_prediction, query_state_embedding, action_embedding), dim=-1
            )
        )
        if self.reliability_projection is not None:
            if recent_visual_valid is None:
                recent_visual_valid = torch.ones(
                    backbone_prediction.shape[0],
                    self.recent_history,
                    device=backbone_prediction.device,
                    dtype=backbone_prediction.dtype,
                )
            if recent_visual_valid.shape != (
                backbone_prediction.shape[0],
                self.recent_history,
            ):
                raise ValueError(
                    "recent_visual_valid must have shape [B,recent_history]"
                )
            reliability = recent_visual_valid.to(
                device=query.device, dtype=query.dtype
            )
            if reliability.device.type == "cpu" and (
                not bool(torch.isfinite(reliability).all())
                or bool((reliability < 0).any())
                or bool((reliability > 1).any())
            ):
                raise ValueError("recent_visual_valid must contain values in [0,1]")
            query = query + self.reliability_projection(reliability)[:, None]
        memory = self.memory_projection(
            torch.cat(
                (memory_embedding, memory_state_embedding, memory_action_embedding),
                dim=-1,
            )
        )
        age_features = self.encode_memory_age(memory_age)
        memory = memory + self.age_projection(age_features.to(dtype=memory.dtype))
        attended, _ = self.cross_attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=~valid,
            need_weights=False,
        )
        hidden = self.output_norm(attended)
        latent_delta = self.latent_residual_cap * torch.tanh(self.latent_head(hidden))
        state_delta = self.state_residual_cap * torch.tanh(self.state_head(hidden))
        return (
            backbone_prediction + latent_delta,
            backbone_state_prediction + state_delta,
            hidden,
        )

    def predict_context(
        self,
        context_embedding: torch.Tensor,
        actions: torch.Tensor,
        current_state: torch.Tensor,
        *,
        memory_embedding: torch.Tensor,
        memory_actions: torch.Tensor,
        memory_state: torch.Tensor,
        memory_age: torch.Tensor,
        memory_valid: torch.Tensor | None = None,
        recent_visual_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            _,
            backbone_prediction,
            command_state,
            backbone_state_prediction,
        ) = self.backbone.predict_context(context_embedding, actions, current_state)
        prediction, state_prediction, _ = self._augment(
            backbone_prediction=backbone_prediction,
            backbone_state_prediction=backbone_state_prediction,
            actions=actions,
            memory_embedding=memory_embedding,
            memory_actions=memory_actions,
            memory_state=memory_state,
            memory_age=memory_age,
            memory_valid=memory_valid,
            recent_visual_valid=recent_visual_valid,
        )
        return (
            backbone_prediction,
            prediction,
            command_state,
            backbone_state_prediction,
            state_prediction,
        )

    def forward(
        self,
        pixels: torch.Tensor,
        actions: torch.Tensor,
        normalized_state: torch.Tensor,
        *,
        memory_pixels: torch.Tensor,
        memory_actions: torch.Tensor,
        memory_state: torch.Tensor,
        memory_age: torch.Tensor,
        memory_valid: torch.Tensor | None = None,
        recent_visual_valid: torch.Tensor | None = None,
    ) -> MultiScaleMemoryOutput:
        backbone_output = self.backbone(pixels, actions, normalized_state)
        memory_encoded = self.backbone.encode(
            {"pixels": memory_pixels, "action": memory_actions}
        )
        prediction, state_prediction, adapter_embedding = self._augment(
            backbone_prediction=backbone_output.prediction,
            backbone_state_prediction=backbone_output.state_prediction,
            actions=actions,
            memory_embedding=memory_encoded["emb"],
            memory_actions=memory_actions,
            memory_state=memory_state,
            memory_age=memory_age,
            memory_valid=memory_valid,
            recent_visual_valid=recent_visual_valid,
        )
        return MultiScaleMemoryOutput(
            embedding=backbone_output.embedding,
            memory_embedding=memory_encoded["emb"],
            adapter_embedding=adapter_embedding,
            target_embedding=backbone_output.target_embedding,
            backbone_prediction=backbone_output.prediction,
            prediction=prediction,
            target_state=backbone_output.target_state,
            command_state=backbone_output.command_state,
            backbone_state_prediction=backbone_output.state_prediction,
            state_prediction=state_prediction,
        )


@torch.inference_mode()
def rollout_multiscale_candidates(
    model: MultiScaleHistoryLeWM,
    context_embedding: torch.Tensor,
    context_state: torch.Tensor,
    prior_actions: torch.Tensor,
    candidate_actions: torch.Tensor,
    *,
    memory_embedding: torch.Tensor,
    memory_actions: torch.Tensor,
    memory_state: torch.Tensor,
    memory_age: torch.Tensor,
    memory_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recursively predict candidate prefixes while retaining causal memory."""
    if context_embedding.ndim != 2:
        raise ValueError("context_embedding must have shape [T,D]")
    if context_state.ndim != 2:
        raise ValueError("context_state must have shape [T,S]")
    if context_embedding.shape[0] != model.recent_history:
        raise ValueError("context length must match model.recent_history")
    if context_state.shape != (
        model.recent_history,
        model.backbone.state_dim,
    ):
        raise ValueError("context_state has an incompatible shape")
    if prior_actions.ndim != 2 or prior_actions.shape[0] != model.recent_history - 1:
        raise ValueError("prior_actions must contain one transition per context gap")
    if candidate_actions.ndim != 3 or candidate_actions.shape[0] == 0:
        raise ValueError("candidate_actions must have shape [C,H,A]")
    if candidate_actions.shape[1] == 0:
        raise ValueError("candidate horizon must be positive")
    if candidate_actions.shape[2] != prior_actions.shape[1]:
        raise ValueError("candidate and prior action dimensions must match")
    if memory_embedding.ndim != 2 or memory_embedding.shape[1] != model.latent_dim:
        raise ValueError("memory_embedding must have shape [M,D]")
    memory_count = memory_embedding.shape[0]
    if memory_actions.shape != (memory_count, candidate_actions.shape[2]):
        raise ValueError("memory_actions has an incompatible shape")
    if memory_state.shape != (memory_count, model.backbone.state_dim):
        raise ValueError("memory_state has an incompatible shape")
    if memory_age.shape != (memory_count,):
        raise ValueError("memory_age must have shape [M]")
    if memory_valid is not None and memory_valid.shape != (memory_count,):
        raise ValueError("memory_valid must have shape [M]")

    candidates = candidate_actions.shape[0]
    latent_history = context_embedding.unsqueeze(0).expand(candidates, -1, -1)
    state_history = context_state.unsqueeze(0).expand(candidates, -1, -1)
    action_history = prior_actions.unsqueeze(0).expand(candidates, -1, -1)
    expanded_memory_embedding = memory_embedding.unsqueeze(0).expand(
        candidates, -1, -1
    )
    expanded_memory_actions = memory_actions.unsqueeze(0).expand(candidates, -1, -1)
    expanded_memory_state = memory_state.unsqueeze(0).expand(candidates, -1, -1)
    expanded_memory_age = memory_age.unsqueeze(0).expand(candidates, -1)
    expanded_memory_valid = (
        None
        if memory_valid is None
        else memory_valid.unsqueeze(0).expand(candidates, -1)
    )

    predicted_latents: list[torch.Tensor] = []
    predicted_states: list[torch.Tensor] = []
    for horizon in range(candidate_actions.shape[1]):
        action_history = torch.cat(
            (action_history, candidate_actions[:, horizon : horizon + 1]), dim=1
        )
        output = model.predict_context(
            latent_history[:, -model.recent_history :],
            action_history[:, -model.recent_history :],
            state_history[:, -model.recent_history :],
            memory_embedding=expanded_memory_embedding,
            memory_actions=expanded_memory_actions,
            memory_state=expanded_memory_state,
            memory_age=expanded_memory_age + float(horizon),
            memory_valid=expanded_memory_valid,
        )
        next_latent = output[1][:, -1]
        next_state = output[4][:, -1]
        latent_history = torch.cat((latent_history, next_latent[:, None]), dim=1)
        state_history = torch.cat((state_history, next_state[:, None]), dim=1)
        predicted_latents.append(next_latent)
        predicted_states.append(next_state)
    return torch.stack(predicted_latents, dim=1), torch.stack(predicted_states, dim=1)


def freeze_recent_context(
    pixels: torch.Tensor,
    memory_pixels: torch.Tensor,
    suffix_frames: int,
) -> torch.Tensor:
    """Replace a recent context suffix with the last causally available frame."""
    if pixels.ndim != 5 or memory_pixels.ndim != 5:
        raise ValueError("pixels and memory_pixels must have shape [B,T,C,H,W]")
    if (
        pixels.shape[0] != memory_pixels.shape[0]
        or pixels.shape[2:] != memory_pixels.shape[2:]
    ):
        raise ValueError("pixels and memory_pixels must share batch/image dimensions")
    history = pixels.shape[1] - 1
    if history <= 0:
        raise ValueError("pixels must include context and one target frame")
    if not 0 <= suffix_frames <= history:
        raise ValueError("suffix_frames must be between zero and the context length")
    if suffix_frames == 0:
        return pixels
    if memory_pixels.shape[1] == 0:
        raise ValueError("at least one causal memory frame is required")

    frozen = pixels.clone()
    first_missing = history - suffix_frames
    source = (
        pixels[:, first_missing - 1]
        if first_missing > 0
        else memory_pixels[:, 0]
    )
    frozen[:, first_missing:history] = source[:, None]
    return frozen


def random_freeze_recent_context(
    pixels: torch.Tensor,
    memory_pixels: torch.Tensor,
    probability: float,
) -> torch.Tensor:
    """Apply a random causal suffix freeze independently to each sample."""
    return random_freeze_recent_context_with_validity(
        pixels, memory_pixels, probability
    )[0]


def random_freeze_recent_context_with_validity(
    pixels: torch.Tensor,
    memory_pixels: torch.Tensor,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return randomly frozen pixels and the corresponding recent-valid mask."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0,1]")
    freeze_recent_context(pixels, memory_pixels, 0)
    history = pixels.shape[1] - 1
    recent_valid = torch.ones(
        pixels.shape[0], history, dtype=torch.bool, device=pixels.device
    )
    if probability == 0.0:
        return pixels, recent_valid
    selected = torch.rand(pixels.shape[0], device=pixels.device) < probability
    suffixes = torch.randint(
        1, history + 1, (pixels.shape[0],), device=pixels.device
    )
    result = pixels.clone()
    for suffix in range(1, history + 1):
        rows = selected & (suffixes == suffix)
        result[rows] = freeze_recent_context(
            pixels[rows], memory_pixels[rows], suffix
        )
        recent_valid[rows, history - suffix :] = False
    return result, recent_valid


__all__ = [
    "MultiScaleHistoryLeWM",
    "MultiScaleMemoryOutput",
    "freeze_recent_context",
    "random_freeze_recent_context",
    "random_freeze_recent_context_with_validity",
]
