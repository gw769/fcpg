"""Representation regularizers used by FCPG adapters.

SIGReg is adapted from the MIT-licensed LeWM implementation; see
THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import torch
from torch import nn


class SIGReg(nn.Module):
    """Sketch isotropic Gaussian regularizer used by LeWM-style predictors."""

    def __init__(self, knots: int = 17, num_proj: int = 256) -> None:
        super().__init__()
        self.num_proj = int(num_proj)
        t = torch.linspace(0.0, 3.0, int(knots), dtype=torch.float32)
        dt = 3.0 / max(int(knots) - 1, 1)
        weights = torch.full((int(knots),), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, projection: torch.Tensor) -> torch.Tensor:
        if projection.ndim == 2:
            projection = projection.unsqueeze(0)
        if projection.ndim != 3:
            raise ValueError(
                f"SIGReg expects [T, B, D] or [B, D], got {tuple(projection.shape)}"
            )
        projection = projection.float()
        random_projection = torch.randn(
            projection.shape[-1],
            self.num_proj,
            device=projection.device,
            dtype=projection.dtype,
        )
        random_projection = random_projection / random_projection.norm(
            p=2, dim=0, keepdim=True
        ).clamp_min(1e-6)
        x_t = (projection @ random_projection).unsqueeze(-1) * self.t.to(dtype=projection.dtype)
        # x_t is [T, B, P, K]; SIGReg estimates each temporal slice's
        # characteristic function over the batch dimension.
        error = (
            (x_t.cos().mean(dim=-3) - self.phi.to(dtype=projection.dtype)).square()
            + x_t.sin().mean(dim=-3).square()
        )
        statistic = (error @ self.weights.to(dtype=projection.dtype)) * projection.shape[1]
        return statistic.mean()
