"""LeWM-derived transformer blocks adapted for FCPG.

The upstream LeWM implementation is vendored under ``third_party/lewm``.
These project-local blocks preserve its AdaLN-zero conditioning and simple
autoregressive predictor while making causal attention an explicit choice for
spatial grounding and action-token sequences.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


LEWM_UPSTREAM_COMMIT = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"


def modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation."""

    return value * (1.0 + scale) + shift


class LeWMFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class LeWMAttention(nn.Module):
    """Scaled dot-product attention with an explicit causal policy."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int | None = None,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if heads <= 0 or dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = int(heads)
        self.dim_head = int(dim_head or (dim // heads))
        self.inner_dim = self.heads * self.dim_head
        self.dropout = float(dropout)
        self.causal = bool(causal)
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = value.shape
        normalized = self.norm(value)
        qkv = self.to_qkv(normalized).chunk(3, dim=-1)
        query, key, content = (
            tensor.view(batch, tokens, self.heads, self.dim_head).transpose(1, 2)
            for tensor in qkv
        )
        dropout = self.dropout if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            query,
            key,
            content,
            dropout_p=dropout,
            is_causal=self.causal,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, self.inner_dim)
        return self.to_out(attended)


class LeWMBlock(nn.Module):
    """Unconditional LeWM transformer block."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.attention = LeWMAttention(
            dim,
            heads=heads,
            dropout=dropout,
            causal=causal,
        )
        self.feed_forward = LeWMFeedForward(dim, mlp_dim, dropout=dropout)
        self.norm_attention = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.attention(self.norm_attention(value))
        value = value + self.feed_forward(self.norm_mlp(value))
        return value


class LeWMConditionalBlock(nn.Module):
    """AdaLN-zero block adapted from LeWM's action-conditioned predictor."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.attention = LeWMAttention(
            dim,
            heads=heads,
            dropout=dropout,
            causal=causal,
        )
        self.feed_forward = LeWMFeedForward(dim, mlp_dim, dropout=dropout)
        self.norm_attention = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if value.shape != condition.shape:
            raise ValueError(
                f"value and condition must have matching shapes, got "
                f"{tuple(value.shape)} and {tuple(condition.shape)}"
            )
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(condition).chunk(6, dim=-1)
        )
        attention_input = modulate(self.norm_attention(value), shift_attn, scale_attn)
        value = value + gate_attn * self.attention(attention_input)
        mlp_input = modulate(self.norm_mlp(value), shift_mlp, scale_mlp)
        value = value + gate_mlp * self.feed_forward(mlp_input)
        return value


class LeWMTransformer(nn.Module):
    """Small transformer supporting LeWM-style AdaLN conditioning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        causal: bool = False,
        conditional: bool = False,
        condition_dim: int | None = None,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.conditional = bool(conditional)
        self.input_projection = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        resolved_condition_dim = int(condition_dim or input_dim)
        self.condition_projection = (
            nn.Linear(resolved_condition_dim, hidden_dim)
            if self.conditional and resolved_condition_dim != hidden_dim
            else nn.Identity()
        )
        block_type = LeWMConditionalBlock if self.conditional else LeWMBlock
        self.layers = nn.ModuleList(
            [
                block_type(
                    hidden_dim,
                    heads=heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    causal=causal,
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        resolved_output_dim = int(output_dim or hidden_dim)
        self.output_projection = (
            nn.Linear(hidden_dim, resolved_output_dim)
            if hidden_dim != resolved_output_dim
            else nn.Identity()
        )

    def forward(
        self,
        value: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = self.input_projection(value)
        if self.conditional:
            if condition is None:
                raise ValueError("condition is required for a conditional LeWM transformer")
            condition = self.condition_projection(condition)
            if condition.shape != value.shape:
                raise ValueError(
                    f"projected condition shape {tuple(condition.shape)} does not match "
                    f"value shape {tuple(value.shape)}"
                )
            for layer in self.layers:
                value = layer(value, condition)
        else:
            if condition is not None:
                raise ValueError("condition is only valid for a conditional LeWM transformer")
            for layer in self.layers:
                value = layer(value)
        return self.output_projection(self.norm(value))


class LeWMEmbedder(nn.Module):
    """Pointwise MLP used for continuous action conditioning."""

    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim or (4 * emb_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value.float())


class LeWMARPredictor(nn.Module):
    """Action-conditioned autoregressive predictor for one-step latent targets."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        max_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self.max_frames = int(max_frames)
        self.position = nn.Parameter(torch.zeros(1, self.max_frames, latent_dim))
        self.action_encoder = LeWMEmbedder(action_dim, latent_dim)
        self.predictor = LeWMTransformer(
            input_dim=latent_dim,
            hidden_dim=latent_dim,
            output_dim=latent_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            causal=True,
            conditional=True,
            condition_dim=latent_dim,
        )
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3:
            raise ValueError(f"latent must have shape [B, T, D], got {tuple(latent.shape)}")
        if action.ndim != 3 or action.shape[:2] != latent.shape[:2]:
            raise ValueError(
                f"action must have shape [B, T, A] aligned with latent, got {tuple(action.shape)}"
            )
        frames = latent.shape[1]
        if frames > self.max_frames:
            raise ValueError(f"received {frames} frames, maximum is {self.max_frames}")
        value = latent + self.position[:, :frames]
        condition = self.action_encoder(action)
        return self.predictor(value, condition)
