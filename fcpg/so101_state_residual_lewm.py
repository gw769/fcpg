"""State-conditioned residual dynamics for the SO-101 LeWM backbone."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class StateResidualOutput:
    embedding: torch.Tensor
    target_embedding: torch.Tensor
    base_prediction: torch.Tensor
    prediction: torch.Tensor
    target_state: torch.Tensor
    command_state: torch.Tensor
    state_prediction: torch.Tensor


class StateResidualLeWM(nn.Module):
    """Add measured joint state to a pretrained image/action LeWM.

    The pretrained JEPA remains the primary visual dynamics model. A compact
    residual branch observes the current six-joint state, the encoded action
    chunk, and the base latent prediction. Its zero-initialized heads make the
    initial latent prediction exactly equal to the pretrained checkpoint and
    the initial state prediction equal to the final command in each chunk.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        state_dim: int = 6,
        latent_dim: int = 192,
        hidden_dim: int = 384,
        latent_residual_cap: float = 0.25,
        state_residual_cap: float = 1.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if latent_residual_cap <= 0 or state_residual_cap <= 0:
            raise ValueError("residual caps must be positive")

        self.base = base
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.latent_residual_cap = float(latent_residual_cap)
        self.state_residual_cap = float(state_residual_cap)

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.latent_dim * 3),
            nn.Linear(self.latent_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.latent_head = nn.Linear(hidden_dim, self.latent_dim)
        self.state_head = nn.Linear(hidden_dim, self.state_dim)
        nn.init.zeros_(self.latent_head.weight)
        nn.init.zeros_(self.latent_head.bias)
        nn.init.zeros_(self.state_head.weight)
        nn.init.zeros_(self.state_head.bias)

    def _command_state(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError("actions must have shape [B, H, chunk_dim]")
        if actions.shape[-1] % self.state_dim:
            raise ValueError("action chunk dimension must be divisible by state_dim")
        return actions.reshape(*actions.shape[:2], -1, self.state_dim)[..., -1, :]

    @staticmethod
    def _validated_residual_scale(residual_scale: float) -> float:
        scale = float(residual_scale)
        if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
            raise ValueError("residual_scale must be finite and lie in [0, 1]")
        return scale

    def predict_from_embedding(
        self,
        embedding: torch.Tensor,
        actions: torch.Tensor,
        current_state: torch.Tensor,
        *,
        action_embedding: torch.Tensor | None = None,
        residual_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if embedding.ndim != 3 or current_state.ndim != 3:
            raise ValueError("embedding and current_state must be rank-three tensors")
        if embedding.shape[:2] != (current_state.shape[0], current_state.shape[1] + 1):
            raise ValueError("current_state must align with all but the final embedding")
        if current_state.shape[-1] != self.state_dim:
            raise ValueError(f"expected {self.state_dim} state dimensions")

        return self.predict_context(
            embedding[:, :-1],
            actions,
            current_state,
            action_embedding=action_embedding,
            residual_scale=residual_scale,
        )

    def predict_context(
        self,
        context_embedding: torch.Tensor,
        actions: torch.Tensor,
        current_state: torch.Tensor,
        *,
        action_embedding: torch.Tensor | None = None,
        residual_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict from deployable history without requiring a future target token."""
        if context_embedding.ndim != 3 or current_state.ndim != 3:
            raise ValueError("context_embedding and current_state must be rank-three tensors")
        if context_embedding.shape[:2] != current_state.shape[:2]:
            raise ValueError("context_embedding and current_state histories must align")
        if actions.shape[:2] != current_state.shape[:2]:
            raise ValueError("actions and current_state histories must align")
        if current_state.shape[-1] != self.state_dim:
            raise ValueError(f"expected {self.state_dim} state dimensions")

        scale = self._validated_residual_scale(residual_scale)
        if action_embedding is None:
            action_embedding = self.base.action_encoder(actions)
        base_prediction = self.base.predict(context_embedding, action_embedding)
        if base_prediction.shape != current_state.new_empty(
            (*current_state.shape[:2], self.latent_dim)
        ).shape:
            raise ValueError("base prediction has an incompatible shape")

        state_embedding = self.state_encoder(current_state)
        hidden = self.fusion(
            torch.cat((base_prediction, state_embedding, action_embedding), dim=-1)
        )
        latent_delta = (
            scale * self.latent_residual_cap * torch.tanh(self.latent_head(hidden))
        )
        state_delta = scale * self.state_residual_cap * torch.tanh(self.state_head(hidden))
        command_state = self._command_state(actions)
        return (
            base_prediction,
            base_prediction + latent_delta,
            command_state,
            command_state + state_delta,
        )

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Expose the frozen/base visual encoder to inference utilities."""
        return self.base.encode(batch)

    def forward(
        self,
        pixels: torch.Tensor,
        actions: torch.Tensor,
        normalized_state: torch.Tensor,
    ) -> StateResidualOutput:
        if normalized_state.shape[1] != pixels.shape[1]:
            raise ValueError("state and image history lengths must match")
        encoded = self.base.encode({"pixels": pixels, "action": actions})
        embedding = encoded["emb"]
        base_prediction, prediction, command_state, state_prediction = (
            self.predict_from_embedding(
                embedding,
                actions,
                normalized_state[:, :-1],
                action_embedding=encoded["act_emb"],
            )
        )
        return StateResidualOutput(
            embedding=embedding,
            target_embedding=embedding[:, 1:],
            base_prediction=base_prediction,
            prediction=prediction,
            target_state=normalized_state[:, 1:],
            command_state=command_state,
            state_prediction=state_prediction,
        )


class TemporalStateResidualLeWM(StateResidualLeWM):
    """Causal temporal residual dynamics on top of a frozen visual LeWM.

    Unlike the compact v4 MLP, this branch lets each prediction step attend to
    earlier state/action/prediction tuples. The output contract stays identical
    to :class:`StateResidualLeWM`, so deployment and visualization code can use
    either checkpoint without a separate inference path.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        state_dim: int = 6,
        latent_dim: int = 192,
        model_dim: int = 512,
        layers: int = 2,
        heads: int = 8,
        ffn_dim: int = 1024,
        max_steps: int = 32,
        latent_residual_cap: float = 0.25,
        state_residual_cap: float = 1.0,
    ) -> None:
        if model_dim <= 0 or layers <= 0 or heads <= 0 or ffn_dim <= 0:
            raise ValueError("temporal model dimensions must be positive")
        if model_dim % heads:
            raise ValueError("temporal model_dim must be divisible by heads")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        super().__init__(
            base,
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dim=model_dim,
            latent_residual_cap=latent_residual_cap,
            state_residual_cap=state_residual_cap,
        )
        self.model_dim = int(model_dim)
        self.layers = int(layers)
        self.heads = int(heads)
        self.ffn_dim = int(ffn_dim)
        self.max_steps = int(max_steps)

        # Replace the v4 pointwise MLP with a causal temporal branch.
        del self.fusion
        self.input_norm = nn.LayerNorm(self.latent_dim * 3)
        self.input_projection = nn.Linear(self.latent_dim * 3, self.model_dim)
        self.horizon_embedding = nn.Parameter(
            torch.zeros(1, self.max_steps, self.model_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.heads,
            dim_feedforward=self.ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer,
            num_layers=self.layers,
            norm=nn.LayerNorm(self.model_dim),
            enable_nested_tensor=False,
        )
        self.latent_head = nn.Linear(self.model_dim, self.latent_dim)
        self.state_head = nn.Linear(self.model_dim, self.state_dim)
        nn.init.zeros_(self.latent_head.weight)
        nn.init.zeros_(self.latent_head.bias)
        nn.init.zeros_(self.state_head.weight)
        nn.init.zeros_(self.state_head.bias)

    def predict_context(
        self,
        context_embedding: torch.Tensor,
        actions: torch.Tensor,
        current_state: torch.Tensor,
        *,
        action_embedding: torch.Tensor | None = None,
        residual_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if context_embedding.ndim != 3 or current_state.ndim != 3:
            raise ValueError("context_embedding and current_state must be rank-three tensors")
        if context_embedding.shape[:2] != current_state.shape[:2]:
            raise ValueError("context_embedding and current_state histories must align")
        if actions.shape[:2] != current_state.shape[:2]:
            raise ValueError("actions and current_state histories must align")
        if current_state.shape[-1] != self.state_dim:
            raise ValueError(f"expected {self.state_dim} state dimensions")
        steps = current_state.shape[1]
        if steps > self.max_steps:
            raise ValueError(
                f"temporal sequence has {steps} steps, exceeding max_steps={self.max_steps}"
            )

        scale = self._validated_residual_scale(residual_scale)
        if action_embedding is None:
            action_embedding = self.base.action_encoder(actions)
        base_prediction = self.base.predict(context_embedding, action_embedding)
        expected_shape = (*current_state.shape[:2], self.latent_dim)
        if base_prediction.shape != expected_shape:
            raise ValueError("base prediction has an incompatible shape")

        state_embedding = self.state_encoder(current_state)
        tokens = self.input_projection(
            self.input_norm(
                torch.cat((base_prediction, state_embedding, action_embedding), dim=-1)
            )
        )
        tokens = tokens + self.horizon_embedding[:, :steps]
        causal_mask = torch.triu(
            torch.ones(steps, steps, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        hidden = self.temporal(tokens, mask=causal_mask)
        latent_delta = (
            scale * self.latent_residual_cap * torch.tanh(self.latent_head(hidden))
        )
        state_delta = scale * self.state_residual_cap * torch.tanh(self.state_head(hidden))
        command_state = self._command_state(actions)
        return (
            base_prediction,
            base_prediction + latent_delta,
            command_state,
            command_state + state_delta,
        )
