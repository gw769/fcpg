"""Reusable PyTorch modules for FCPG experiments.

The first implementation path is intentionally small: local image evidence and
proprioceptive/keypoint vectors are used to predict a 2D projection residual.
The class names here are policy-backbone agnostic so the same module can be
plugged into deterministic action-chunk or later ACT experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FCPGOutput:
    """Outputs from a residual-only FCPG module."""

    delta_norm: torch.Tensor
    delta_px: torch.Tensor
    q_corrected_px: torch.Tensor | None = None


def coord_channels(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create normalized XY coordinate channels for CoordConv-style patches."""

    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


def add_coord_channels(patch: torch.Tensor) -> torch.Tensor:
    """Append normalized XY coordinate channels to a BCHW patch tensor."""

    if patch.ndim != 4:
        raise ValueError(f"patch must have shape [B, C, H, W], got {tuple(patch.shape)}")
    batch, _, height, width = patch.shape
    coords = coord_channels(batch, height, width, patch.device, patch.dtype)
    return torch.cat([patch, coords], dim=1)


class ResidualCorrectionHead(nn.Module):
    """Predict normalized 2D residuals from local visual and proprio features.

    The output is normalized by `patch_radius`. Use `predict_delta_px` or
    `FCPGResidualOnly` when pixel-space residuals are needed.
    """

    def __init__(self, vector_dim: int, hidden_dim: int = 128, patch_radius: int = 16) -> None:
        super().__init__()
        self.vector_dim = int(vector_dim)
        self.hidden_dim = int(hidden_dim)
        self.patch_radius = int(patch_radius)
        patch_size = 2 * self.patch_radius + 1
        pooled_size = patch_size // 4
        image_dim = 64 * pooled_size * pooled_size
        self.cnn = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(image_dim + self.vector_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),
        )

    def forward(self, vector: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        image_feat = self.cnn(add_coord_channels(patch))
        return self.head(torch.cat([image_feat, vector], dim=1))

    def predict_delta_px(self, vector: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        return self.forward(vector, patch) * float(self.patch_radius)


class ReliabilityConfidenceHead(nn.Module):
    """Predict a scalar reliability score for a corrected grounded token."""

    def __init__(
        self,
        vector_dim: int,
        pred_feature_dim: int = 0,
        hidden_dim: int = 128,
        patch_radius: int = 16,
    ) -> None:
        super().__init__()
        self.vector_dim = int(vector_dim)
        self.pred_feature_dim = int(pred_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.patch_radius = int(patch_radius)
        patch_size = 2 * self.patch_radius + 1
        pooled_size = patch_size // 4
        image_dim = 64 * pooled_size * pooled_size
        self.cnn = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(image_dim + self.vector_dim + self.pred_feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, vector: torch.Tensor, patch: torch.Tensor, pred_features: torch.Tensor | None = None) -> torch.Tensor:
        image_feat = self.cnn(add_coord_channels(patch))
        parts = [image_feat, vector]
        if self.pred_feature_dim:
            if pred_features is None:
                raise ValueError("pred_features are required when pred_feature_dim > 0")
            parts.append(pred_features)
        return self.head(torch.cat(parts, dim=1))


class FCPGResidualOnly(nn.Module):
    """Residual-only FCPG adapter for policy integration.

    It predicts `delta_q` around `q_nominal`. If nominal image coordinates are
    provided, it also returns `q_corrected = q_nominal + delta_q`.
    """

    def __init__(self, vector_dim: int, hidden_dim: int = 128, patch_radius: int = 16) -> None:
        super().__init__()
        self.patch_radius = int(patch_radius)
        self.residual_head = ResidualCorrectionHead(vector_dim, hidden_dim=hidden_dim, patch_radius=patch_radius)

    def forward(self, vector: torch.Tensor, patch: torch.Tensor, q_nominal_px: torch.Tensor | None = None) -> FCPGOutput:
        delta_norm = self.residual_head(vector, patch)
        delta_px = delta_norm * float(self.patch_radius)
        q_corrected = None if q_nominal_px is None else q_nominal_px + delta_px
        return FCPGOutput(delta_norm=delta_norm, delta_px=delta_px, q_corrected_px=q_corrected)



def pixels_to_grid_torch(
    pixels_xy: torch.Tensor,
    image_width: int,
    image_height: int,
    align_corners: bool = False,
) -> torch.Tensor:
    """Convert image pixel coordinates `(x, y)` to `grid_sample` coordinates.

    `pixels_xy` can have shape `[B, N, 2]` or any leading dimensions ending in
    two coordinates. The returned tensor has the same leading dimensions.
    """

    if pixels_xy.shape[-1] != 2:
        raise ValueError(f"pixels_xy must end with 2 coordinates, got {tuple(pixels_xy.shape)}")
    width = float(image_width)
    height = float(image_height)
    if align_corners:
        x = 2.0 * pixels_xy[..., 0] / max(width - 1.0, 1.0) - 1.0
        y = 2.0 * pixels_xy[..., 1] / max(height - 1.0, 1.0) - 1.0
    else:
        x = (2.0 * pixels_xy[..., 0] + 1.0) / width - 1.0
        y = (2.0 * pixels_xy[..., 1] + 1.0) / height - 1.0
    return torch.stack([x, y], dim=-1)


def sample_feature_map_at_pixels(
    feature_map: torch.Tensor,
    pixels_xy: torch.Tensor,
    image_size: tuple[int, int] | None = None,
    align_corners: bool = False,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Sample feature vectors at image-space pixel coordinates.

    Args:
        feature_map: Tensor with shape `[B, C, Hf, Wf]`.
        pixels_xy: Tensor with shape `[B, N, 2]` in source image pixels. Shape
            `[N, 2]` is accepted only when `B == 1`.
        image_size: `(width, height)` of the coordinate system used by
            `pixels_xy`. If omitted, feature-map width/height are used.

    Returns:
        Tensor with shape `[B, N, C]`.
    """

    if feature_map.ndim != 4:
        raise ValueError(f"feature_map must have shape [B, C, H, W], got {tuple(feature_map.shape)}")
    batch, channels, feat_h, feat_w = feature_map.shape
    if pixels_xy.ndim == 2:
        if batch != 1:
            raise ValueError("pixels_xy with shape [N, 2] is only valid for batch size 1")
        pixels_xy = pixels_xy.unsqueeze(0)
    if pixels_xy.ndim != 3 or pixels_xy.shape[-1] != 2:
        raise ValueError(f"pixels_xy must have shape [B, N, 2], got {tuple(pixels_xy.shape)}")
    if pixels_xy.shape[0] != batch:
        raise ValueError(f"batch mismatch: feature_map batch={batch}, pixels batch={pixels_xy.shape[0]}")
    image_width, image_height = image_size or (feat_w, feat_h)
    grid = pixels_to_grid_torch(
        pixels_xy.to(device=feature_map.device, dtype=feature_map.dtype),
        image_width=image_width,
        image_height=image_height,
        align_corners=align_corners,
    )
    grid = grid.view(batch, pixels_xy.shape[1], 1, 2)
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return sampled.squeeze(-1).transpose(1, 2).contiguous().view(batch, pixels_xy.shape[1], channels)


def sample_image_patches_at_pixels(
    image: torch.Tensor,
    points_px: torch.Tensor,
    radius: int,
    round_centers: bool = True,
) -> torch.Tensor:
    """Extract zero-padded square image patches with one batched grid-sample call."""

    if image.ndim != 4:
        raise ValueError(f"image must have shape [B, C, H, W], got {tuple(image.shape)}")
    if points_px.ndim != 3 or points_px.shape[-1] != 2:
        raise ValueError(f"points_px must have shape [B, N, 2], got {tuple(points_px.shape)}")
    if image.shape[0] != points_px.shape[0]:
        raise ValueError(f"batch mismatch: image={image.shape[0]} points={points_px.shape[0]}")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    batch, channels, height, width = image.shape
    num_points = points_px.shape[1]
    side = 2 * int(radius) + 1
    centers = points_px.to(device=image.device, dtype=image.dtype)
    if round_centers:
        centers = centers.round()
    offsets = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    offset_grid = torch.stack([offset_x, offset_y], dim=-1)
    sample_pixels = centers[:, :, None, None, :] + offset_grid[None, None, :, :, :]
    grid = pixels_to_grid_torch(
        sample_pixels,
        image_width=width,
        image_height=height,
        align_corners=False,
    )
    grid = grid.view(batch, num_points * side, side, 2)
    sampled = F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled.view(batch, channels, num_points, side, side).permute(0, 2, 1, 3, 4).contiguous()


class GroundedTokenAdapter(nn.Module):
    """Convert sampled visual features and proprio embeddings into tokens.

    This module is the minimal policy-facing bridge for GeoProp-style grounding
    and FCPG residual-only grounding:

    - GeoProp-lite: pass `q_nominal` as `points_px`.
    - FCPG residual-only: pass `q_corrected` as `points_px`.
    - Reliability variant: pass `reliability` with shape `[B, N]` or `[B, N, 1]`.
    """

    def __init__(
        self,
        visual_dim: int,
        proprio_dim: int,
        token_dim: int,
        hidden_dim: int = 128,
        gate_visual_only: bool = True,
        include_reliability: bool = True,
    ) -> None:
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.proprio_dim = int(proprio_dim)
        self.token_dim = int(token_dim)
        self.gate_visual_only = bool(gate_visual_only)
        self.include_reliability = bool(include_reliability)
        reliability_dim = 1 if self.include_reliability else 0
        input_dim = self.visual_dim + self.proprio_dim + reliability_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.token_dim),
        )

    def forward(
        self,
        feature_map: torch.Tensor,
        points_px: torch.Tensor,
        proprio: torch.Tensor,
        image_size: tuple[int, int] | None = None,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual = sample_feature_map_at_pixels(feature_map, points_px, image_size=image_size)
        return self.forward_sampled(visual, proprio, reliability=reliability)

    def forward_sampled(
        self,
        visual: torch.Tensor,
        proprio: torch.Tensor,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build grounded tokens from already sampled visual features."""

        if visual.ndim != 3:
            raise ValueError(f"visual must have shape [B, N, C], got {tuple(visual.shape)}")
        if visual.shape[-1] != self.visual_dim:
            raise ValueError(f"expected visual dim {self.visual_dim}, got {visual.shape[-1]}")
        if proprio.ndim != 3:
            raise ValueError(f"proprio must have shape [B, N, D], got {tuple(proprio.shape)}")
        if proprio.shape[:2] != visual.shape[:2]:
            raise ValueError(f"proprio shape {tuple(proprio.shape)} is incompatible with visual shape {tuple(visual.shape)}")
        if proprio.shape[-1] != self.proprio_dim:
            raise ValueError(f"expected proprio dim {self.proprio_dim}, got {proprio.shape[-1]}")
        rel = None
        if reliability is not None:
            if reliability.ndim == 2:
                reliability = reliability.unsqueeze(-1)
            if reliability.shape[:2] != visual.shape[:2] or reliability.shape[-1] != 1:
                raise ValueError(
                    f"reliability must have shape [B, N] or [B, N, 1], got {tuple(reliability.shape)}"
                )
            rel = reliability.to(device=visual.device, dtype=visual.dtype)
            if self.gate_visual_only:
                visual = visual * rel
        parts = [visual, proprio.to(device=visual.device, dtype=visual.dtype)]
        if self.include_reliability:
            if rel is None:
                rel = torch.ones((*visual.shape[:2], 1), device=visual.device, dtype=visual.dtype)
            parts.append(rel)
        flat = torch.cat(parts, dim=-1).view(-1, self.visual_dim + self.proprio_dim + (1 if self.include_reliability else 0))
        tokens = self.net(flat).view(visual.shape[0], visual.shape[1], self.token_dim)
        if reliability is not None and not self.gate_visual_only:
            tokens = tokens * rel
        return tokens
