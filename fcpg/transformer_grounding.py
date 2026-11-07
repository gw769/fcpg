"""Transformer adapter for feedback-corrected proprioceptive grounding.

The adapter keeps the nominal geometric projection as an explicit anchor. A
pretrained dense ViT supplies local visual tokens around each anchor, and a
small transformer predicts the projection residual and grounding reliability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import timm
import torch
from torch import nn
from torch.nn import functional as F

from fcpg.lewm_core import LEWM_UPSTREAM_COMMIT, LeWMTransformer
from fcpg.modules import sample_feature_map_at_pixels


DEFAULT_VIT_MODEL = "vit_base_patch16_224.augreg_in1k"


@dataclass(frozen=True)
class DenseVisionOutput:
    feature_map: torch.Tensor
    global_feature: torch.Tensor


@dataclass(frozen=True)
class TransformerGroundingOutput:
    delta_px: torch.Tensor
    q_corrected_px: torch.Tensor
    reliability: torch.Tensor
    reliability_logit: torch.Tensor
    keypoint_latent: torch.Tensor
    feature_map: torch.Tensor | None = None
    global_feature: torch.Tensor | None = None


@dataclass(frozen=True)
class LeWMGroundingOutput:
    """Residual-only grounding output produced by the LeWM corrector."""

    delta_px: torch.Tensor
    q_corrected_px: torch.Tensor
    keypoint_latent: torch.Tensor
    feature_map: torch.Tensor | None = None
    global_feature: torch.Tensor | None = None


def sample_local_feature_tokens(
    feature_map: torch.Tensor,
    points_px: torch.Tensor,
    image_size: tuple[int, int],
    radius_px: float,
    grid_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a square local token grid around image-space anchor points.

    Returns tokens with shape ``[B, N, grid_size**2, C]`` and normalized
    relative offsets with shape ``[grid_size**2, 2]``.
    """

    if grid_size <= 0 or grid_size % 2 == 0:
        raise ValueError("grid_size must be a positive odd integer")
    if radius_px <= 0:
        raise ValueError("radius_px must be positive")
    if points_px.ndim != 3 or points_px.shape[-1] != 2:
        raise ValueError(f"points_px must have shape [B, N, 2], got {tuple(points_px.shape)}")

    offsets_1d = torch.linspace(
        -float(radius_px),
        float(radius_px),
        grid_size,
        device=points_px.device,
        dtype=points_px.dtype,
    )
    offset_y, offset_x = torch.meshgrid(offsets_1d, offsets_1d, indexing="ij")
    offsets_px = torch.stack([offset_x, offset_y], dim=-1).reshape(-1, 2)
    sample_points = points_px[:, :, None, :] + offsets_px[None, None, :, :]
    batch, num_keypoints, num_local_tokens, _ = sample_points.shape
    sampled = sample_feature_map_at_pixels(
        feature_map,
        sample_points.reshape(batch, num_keypoints * num_local_tokens, 2),
        image_size=image_size,
        padding_mode="border",
    )
    sampled = sampled.reshape(batch, num_keypoints, num_local_tokens, feature_map.shape[1])
    return sampled, offsets_px / float(radius_px)


class ViTDenseEncoder(nn.Module):
    """ImageNet-pretrained ViT-B/16 exposed as a dense feature pyramid.

    Four intermediate 14x14 token maps are projected and fused. The backbone
    can remain frozen while the lightweight fusion projection is trained.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_VIT_MODEL,
        pretrained: bool = True,
        selected_layers: Sequence[int] = (2, 5, 8, 11),
        output_dim: int = 256,
        input_size: int = 224,
        freeze_backbone: bool = True,
        trainable_backbone_blocks: int = 0,
    ) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.selected_layers = tuple(int(layer) for layer in selected_layers)
        self.output_dim = int(output_dim)
        self.input_size = int(input_size)
        self.freeze_backbone = bool(freeze_backbone)
        self.trainable_backbone_blocks = int(trainable_backbone_blocks)
        if not self.selected_layers:
            raise ValueError("selected_layers must not be empty")

        self.backbone = timm.create_model(self.model_name, pretrained=pretrained, num_classes=0)
        backbone_dim = int(self.backbone.num_features)
        self.feature_projections = nn.ModuleList(
            [nn.Conv2d(backbone_dim, self.output_dim, kernel_size=1) for _ in self.selected_layers]
        )
        groups = min(32, self.output_dim)
        while self.output_dim % groups != 0:
            groups -= 1
        self.fusion = nn.Sequential(
            nn.Conv2d(self.output_dim, self.output_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, self.output_dim),
            nn.GELU(),
        )
        self.global_projection = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, self.output_dim),
        )

        cfg = self.backbone.pretrained_cfg
        mean = torch.tensor(cfg.get("mean", (0.5, 0.5, 0.5)), dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(cfg.get("std", (0.5, 0.5, 0.5)), dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)
        if self.trainable_backbone_blocks < 0:
            raise ValueError("trainable_backbone_blocks must be non-negative")
        self.backbone_fully_frozen = self.freeze_backbone and self.trainable_backbone_blocks == 0
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            if self.trainable_backbone_blocks > len(self.backbone.blocks):
                raise ValueError(
                    f"requested {self.trainable_backbone_blocks} trainable blocks, "
                    f"backbone has {len(self.backbone.blocks)}"
                )
            if self.trainable_backbone_blocks > 0:
                for block in self.backbone.blocks[-self.trainable_backbone_blocks :]:
                    block.requires_grad_(True)
                self.backbone.norm.requires_grad_(True)
            if self.backbone_fully_frozen:
                self.backbone.eval()

    def train(self, mode: bool = True) -> "ViTDenseEncoder":
        super().train(mode)
        if self.backbone_fully_frozen:
            self.backbone.eval()
        return self

    def _intermediate_features(self, image: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        def extract() -> list[tuple[torch.Tensor, torch.Tensor]]:
            return self.backbone.get_intermediate_layers(
                image,
                n=list(self.selected_layers),
                reshape=True,
                return_prefix_tokens=True,
                norm=True,
            )

        if self.backbone_fully_frozen:
            with torch.no_grad():
                return extract()
        return extract()

    def extract_backbone_features(
        self,
        rgb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cacheable intermediate maps and the final prefix token.

        The feature tensor has shape ``[B, L, C, H, W]``, where ``L`` is the
        number of selected ViT blocks. These tensors are upstream of the
        trainable dense projections, so caching them does not freeze the
        policy-specific projection and fusion layers.
        """

        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must have shape [B, 3, H, W], got {tuple(rgb.shape)}")
        image = F.interpolate(
            rgb,
            size=(self.input_size, self.input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        image = (image - self.image_mean.to(dtype=image.dtype)) / self.image_std.to(dtype=image.dtype)
        intermediates = self._intermediate_features(image)
        feature_maps = torch.stack([feature for feature, _ in intermediates], dim=1)
        prefix = intermediates[-1][1][:, 0]
        return feature_maps, prefix

    def project_backbone_features(
        self,
        feature_maps: torch.Tensor,
        prefix: torch.Tensor,
    ) -> DenseVisionOutput:
        """Apply trainable dense projections to online or cached ViT maps."""

        if feature_maps.ndim != 5:
            raise ValueError(
                "feature_maps must have shape [B, L, C, H, W], "
                f"got {tuple(feature_maps.shape)}"
            )
        if feature_maps.shape[1] != len(self.feature_projections):
            raise ValueError(
                f"expected {len(self.feature_projections)} feature levels, "
                f"got {feature_maps.shape[1]}"
            )
        if prefix.ndim != 2 or prefix.shape[0] != feature_maps.shape[0]:
            raise ValueError(f"prefix must have shape [B, C], got {tuple(prefix.shape)}")
        projected = [
            projection(feature_maps[:, index])
            for index, projection in enumerate(self.feature_projections)
        ]
        fused = torch.stack(projected, dim=0).mean(dim=0)
        feature_map = self.fusion(fused)
        global_feature = self.global_projection(prefix)
        return DenseVisionOutput(feature_map=feature_map, global_feature=global_feature)

    def forward(self, rgb: torch.Tensor) -> DenseVisionOutput:
        feature_maps, prefix = self.extract_backbone_features(rgb)
        return self.project_backbone_features(feature_maps, prefix)


class AnchorConditionedResidualTransformer(nn.Module):
    """Predict per-keypoint projection residuals from anchor-local ViT tokens."""

    def __init__(
        self,
        visual_dim: int,
        obs_dim: int,
        num_keypoints: int,
        model_dim: int = 256,
        local_grid_size: int = 5,
        local_radius_px: float = 16.0,
        local_layers: int = 4,
        global_layers: int = 2,
        heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_residual_px: float = 20.0,
    ) -> None:
        super().__init__()
        if model_dim % heads != 0:
            raise ValueError(f"model_dim={model_dim} must be divisible by heads={heads}")
        self.visual_dim = int(visual_dim)
        self.obs_dim = int(obs_dim)
        self.num_keypoints = int(num_keypoints)
        self.model_dim = int(model_dim)
        self.local_grid_size = int(local_grid_size)
        self.local_radius_px = float(local_radius_px)
        self.max_residual_px = float(max_residual_px)

        query_input_dim = 2 + 3 + self.num_keypoints + self.obs_dim
        self.query_projection = nn.Sequential(
            nn.LayerNorm(query_input_dim),
            nn.Linear(query_input_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.visual_projection = nn.Linear(self.visual_dim, self.model_dim)
        self.relative_position = nn.Sequential(
            nn.Linear(2, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        local_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=heads,
            dim_feedforward=self.model_dim * int(mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        global_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=heads,
            dim_feedforward=self.model_dim * int(mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.local_transformer = nn.TransformerEncoder(local_layer, num_layers=int(local_layers), norm=nn.LayerNorm(self.model_dim))
        self.keypoint_transformer = nn.TransformerEncoder(
            global_layer,
            num_layers=int(global_layers),
            norm=nn.LayerNorm(self.model_dim),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, 2),
        )
        self.reliability_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim // 2),
            nn.GELU(),
            nn.Linear(self.model_dim // 2, 1),
        )

    def _query_vectors(
        self,
        q_nominal_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        batch, num_keypoints, _ = q_nominal_px.shape
        if num_keypoints != self.num_keypoints:
            raise ValueError(f"expected {self.num_keypoints} keypoints, got {num_keypoints}")
        if keypoints_xyz.shape != (batch, num_keypoints, 3):
            raise ValueError(f"keypoints_xyz has incompatible shape {tuple(keypoints_xyz.shape)}")
        if robot_obs.shape != (batch, self.obs_dim):
            raise ValueError(f"robot_obs must have shape {(batch, self.obs_dim)}, got {tuple(robot_obs.shape)}")
        width, height = image_size
        q_norm = torch.stack(
            [q_nominal_px[..., 0] / float(width), q_nominal_px[..., 1] / float(height)],
            dim=-1,
        )
        onehot = torch.eye(
            self.num_keypoints,
            device=q_nominal_px.device,
            dtype=q_nominal_px.dtype,
        ).unsqueeze(0).expand(batch, -1, -1)
        obs = robot_obs[:, None, :].expand(-1, self.num_keypoints, -1)
        return torch.cat([q_norm, keypoints_xyz, onehot, obs], dim=-1)

    def forward(
        self,
        feature_map: torch.Tensor,
        q_nominal_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> TransformerGroundingOutput:
        local_visual, relative_offsets = sample_local_feature_tokens(
            feature_map,
            q_nominal_px,
            image_size=image_size,
            radius_px=self.local_radius_px,
            grid_size=self.local_grid_size,
        )
        return self.forward_from_local_tokens(
            local_visual,
            relative_offsets,
            q_nominal_px,
            keypoints_xyz,
            robot_obs,
            image_size,
        )

    def forward_from_local_tokens(
        self,
        local_visual: torch.Tensor,
        relative_offsets: torch.Tensor,
        q_nominal_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> TransformerGroundingOutput:
        """Run the identical corrector from cacheable local ViT tokens."""

        batch = q_nominal_px.shape[0]
        expected_tokens = self.local_grid_size**2
        if local_visual.shape != (
            batch,
            self.num_keypoints,
            expected_tokens,
            self.visual_dim,
        ):
            raise ValueError(
                "local_visual must have shape "
                f"[B,{self.num_keypoints},{expected_tokens},{self.visual_dim}]"
            )
        if relative_offsets.shape != (expected_tokens, 2):
            raise ValueError(
                f"relative_offsets must have shape [{expected_tokens},2]"
            )
        query = self.query_projection(
            self._query_vectors(q_nominal_px, keypoints_xyz, robot_obs, image_size)
        )
        local_tokens = self.visual_projection(local_visual)
        local_tokens = local_tokens + self.relative_position(relative_offsets).view(
            1, 1, -1, self.model_dim
        )
        batch, num_keypoints, num_local_tokens, _ = local_tokens.shape
        sequence = torch.cat([query.unsqueeze(2), local_tokens], dim=2)
        sequence = sequence.reshape(batch * num_keypoints, num_local_tokens + 1, self.model_dim)
        local_encoded = self.local_transformer(sequence)[:, 0]
        keypoint_latent = self.keypoint_transformer(
            local_encoded.reshape(batch, num_keypoints, self.model_dim)
        )
        delta_px = torch.tanh(self.delta_head(keypoint_latent)) * self.max_residual_px
        reliability_logit = self.reliability_head(keypoint_latent).squeeze(-1)
        reliability = torch.sigmoid(reliability_logit)
        return TransformerGroundingOutput(
            delta_px=delta_px,
            q_corrected_px=q_nominal_px + delta_px,
            reliability=reliability,
            reliability_logit=reliability_logit,
            keypoint_latent=keypoint_latent,
        )


class LeWMAnchorResidualCorrector(nn.Module):
    """LeWM-style anchor-conditioned residual-only grounding corrector."""

    def __init__(
        self,
        visual_dim: int,
        obs_dim: int,
        num_keypoints: int,
        model_dim: int = 256,
        local_grid_size: int = 5,
        local_radius_px: float = 16.0,
        local_layers: int = 4,
        global_layers: int = 2,
        heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_residual_px: float = 20.0,
    ) -> None:
        super().__init__()
        if model_dim % heads != 0:
            raise ValueError(f"model_dim={model_dim} must be divisible by heads={heads}")
        if max_residual_px <= 0:
            raise ValueError("max_residual_px must be positive")
        self.visual_dim = int(visual_dim)
        self.obs_dim = int(obs_dim)
        self.num_keypoints = int(num_keypoints)
        self.model_dim = int(model_dim)
        self.local_grid_size = int(local_grid_size)
        self.local_radius_px = float(local_radius_px)
        self.max_residual_px = float(max_residual_px)

        query_input_dim = 2 + 3 + self.num_keypoints + self.obs_dim
        self.query_projection = nn.Sequential(
            nn.LayerNorm(query_input_dim),
            nn.Linear(query_input_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.visual_projection = nn.Linear(self.visual_dim, self.model_dim)
        self.relative_position = nn.Sequential(
            nn.Linear(2, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        mlp_dim = self.model_dim * int(mlp_ratio)
        self.local_transformer = LeWMTransformer(
            input_dim=self.model_dim,
            hidden_dim=self.model_dim,
            output_dim=self.model_dim,
            depth=int(local_layers),
            heads=int(heads),
            mlp_dim=mlp_dim,
            dropout=dropout,
            causal=False,
            conditional=True,
            condition_dim=self.model_dim,
        )
        self.keypoint_transformer = LeWMTransformer(
            input_dim=self.model_dim,
            hidden_dim=self.model_dim,
            output_dim=self.model_dim,
            depth=int(global_layers),
            heads=int(heads),
            mlp_dim=mlp_dim,
            dropout=dropout,
            causal=False,
            conditional=True,
            condition_dim=self.model_dim,
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, 2),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def _query_vectors(
        self,
        q_nominal_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        batch, num_keypoints, _ = q_nominal_px.shape
        if num_keypoints != self.num_keypoints:
            raise ValueError(f"expected {self.num_keypoints} keypoints, got {num_keypoints}")
        if keypoints_xyz.shape != (batch, num_keypoints, 3):
            raise ValueError(f"keypoints_xyz has incompatible shape {tuple(keypoints_xyz.shape)}")
        if robot_obs.shape != (batch, self.obs_dim):
            raise ValueError(
                f"robot_obs must have shape {(batch, self.obs_dim)}, "
                f"got {tuple(robot_obs.shape)}"
            )
        width, height = image_size
        q_norm = torch.stack(
            [q_nominal_px[..., 0] / float(width), q_nominal_px[..., 1] / float(height)],
            dim=-1,
        )
        onehot = torch.eye(
            self.num_keypoints,
            device=q_nominal_px.device,
            dtype=q_nominal_px.dtype,
        ).unsqueeze(0).expand(batch, -1, -1)
        obs = robot_obs[:, None, :].expand(-1, self.num_keypoints, -1)
        return torch.cat([q_norm, keypoints_xyz, onehot, obs], dim=-1)

    def forward(
        self,
        feature_map: torch.Tensor,
        q_nominal_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> LeWMGroundingOutput:
        local_visual, relative_offsets = sample_local_feature_tokens(
            feature_map,
            q_nominal_px,
            image_size=image_size,
            radius_px=self.local_radius_px,
            grid_size=self.local_grid_size,
        )
        return self.forward_from_local_tokens(
            local_visual,
            relative_offsets,
            q_nominal_px,
            keypoints_xyz,
            robot_obs,
            image_size,
        )

    def forward_from_local_tokens(
        self,
        local_visual: torch.Tensor,
        relative_offsets: torch.Tensor,
        q_nominal_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> LeWMGroundingOutput:
        """Run the conditional LeWM corrector from cached local ViT tokens."""

        batch = q_nominal_px.shape[0]
        expected_tokens = self.local_grid_size**2
        if local_visual.shape != (
            batch,
            self.num_keypoints,
            expected_tokens,
            self.visual_dim,
        ):
            raise ValueError(
                "local_visual must have shape "
                f"[B,{self.num_keypoints},{expected_tokens},{self.visual_dim}]"
            )
        if relative_offsets.shape != (expected_tokens, 2):
            raise ValueError(
                f"relative_offsets must have shape [{expected_tokens},2]"
            )
        query = self.query_projection(
            self._query_vectors(q_nominal_px, keypoints_xyz, robot_obs, image_size)
        )
        local_tokens = self.visual_projection(local_visual)
        local_tokens = local_tokens + self.relative_position(relative_offsets).view(
            1, 1, -1, self.model_dim
        )
        batch, num_keypoints, num_local_tokens, _ = local_tokens.shape
        local_value = torch.cat([query.unsqueeze(2), local_tokens], dim=2)
        local_condition = query.unsqueeze(2).expand(-1, -1, num_local_tokens + 1, -1)
        local_value = local_value.reshape(
            batch * num_keypoints,
            num_local_tokens + 1,
            self.model_dim,
        )
        local_condition = local_condition.reshape_as(local_value)
        local_encoded = self.local_transformer(local_value, local_condition)[:, 0]
        local_encoded = local_encoded.reshape(batch, num_keypoints, self.model_dim)
        keypoint_latent = self.keypoint_transformer(local_encoded, query)
        delta_px = torch.tanh(self.delta_head(keypoint_latent)) * self.max_residual_px
        return LeWMGroundingOutput(
            delta_px=delta_px,
            q_corrected_px=q_nominal_px + delta_px,
            keypoint_latent=keypoint_latent,
        )


class TwoTipStructuredLeWMAnchorCorrector(nn.Module):
    """Predict two fingertip anchors and derive the gripper center exactly.

    SO-101 grounding uses the public anchor order ``[center, left, right]``.
    The center is geometrically redundant, so this corrector learns only the
    two fingertip residuals and reconstructs the center as their midpoint.
    Downstream modules still receive the original three-anchor interface.
    """

    center_index = 0
    tip_indices = (1, 2)

    def __init__(
        self,
        visual_dim: int,
        obs_dim: int,
        model_dim: int = 256,
        local_grid_size: int = 5,
        local_radius_px: float = 16.0,
        local_layers: int = 4,
        global_layers: int = 2,
        heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_residual_px: float = 20.0,
    ) -> None:
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.obs_dim = int(obs_dim)
        self.num_keypoints = 3
        self.model_dim = int(model_dim)
        self.local_grid_size = int(local_grid_size)
        self.local_radius_px = float(local_radius_px)
        self.max_residual_px = float(max_residual_px)
        self.tip_corrector = LeWMAnchorResidualCorrector(
            visual_dim=self.visual_dim,
            obs_dim=self.obs_dim,
            num_keypoints=2,
            model_dim=self.model_dim,
            local_grid_size=self.local_grid_size,
            local_radius_px=self.local_radius_px,
            local_layers=local_layers,
            global_layers=global_layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_residual_px=self.max_residual_px,
        )

    @property
    def delta_head(self) -> nn.Sequential:
        """Expose the residual head for existing initialization diagnostics."""

        return self.tip_corrector.delta_head

    def forward(
        self,
        feature_map: torch.Tensor,
        q_nominal_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> LeWMGroundingOutput:
        if q_nominal_px.ndim != 3 or q_nominal_px.shape[1:] != (3, 2):
            raise ValueError(
                "q_nominal_px must use SO-101 anchor order [center,left,right] "
                "with shape [B,3,2]"
            )
        batch = q_nominal_px.shape[0]
        if keypoints_xyz.shape != (batch, 3, 3):
            raise ValueError("keypoints_xyz must have shape [B,3,3]")

        tip_output = self.tip_corrector(
            feature_map,
            q_nominal_px[:, self.tip_indices],
            keypoints_xyz[:, self.tip_indices],
            robot_obs,
            image_size,
        )
        corrected_tips = tip_output.q_corrected_px
        corrected_center = corrected_tips.mean(dim=1, keepdim=True)
        q_corrected = torch.cat((corrected_center, corrected_tips), dim=1)
        center_latent = tip_output.keypoint_latent.mean(dim=1, keepdim=True)
        keypoint_latent = torch.cat(
            (center_latent, tip_output.keypoint_latent), dim=1
        )
        return LeWMGroundingOutput(
            delta_px=q_corrected - q_nominal_px,
            q_corrected_px=q_corrected,
            keypoint_latent=keypoint_latent,
        )

    def forward_from_local_tokens(
        self,
        tip_local_visual: torch.Tensor,
        relative_offsets: torch.Tensor,
        q_nominal_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> LeWMGroundingOutput:
        """Structured two-tip forward from pre-sampled frozen ViT tokens."""

        if q_nominal_px.ndim != 3 or q_nominal_px.shape[1:] != (3, 2):
            raise ValueError("q_nominal_px must have shape [B,3,2]")
        batch = q_nominal_px.shape[0]
        if keypoints_xyz.shape != (batch, 3, 3):
            raise ValueError("keypoints_xyz must have shape [B,3,3]")
        tip_output = self.tip_corrector.forward_from_local_tokens(
            tip_local_visual,
            relative_offsets,
            q_nominal_px[:, self.tip_indices],
            keypoints_xyz[:, self.tip_indices],
            robot_obs,
            image_size,
        )
        corrected_tips = tip_output.q_corrected_px
        corrected_center = corrected_tips.mean(dim=1, keepdim=True)
        q_corrected = torch.cat((corrected_center, corrected_tips), dim=1)
        center_latent = tip_output.keypoint_latent.mean(dim=1, keepdim=True)
        keypoint_latent = torch.cat(
            (center_latent, tip_output.keypoint_latent), dim=1
        )
        return LeWMGroundingOutput(
            delta_px=q_corrected - q_nominal_px,
            q_corrected_px=q_corrected,
            keypoint_latent=keypoint_latent,
        )


class FCPGViTAdapter(nn.Module):
    """End-to-end ViT feature extractor plus FCPG residual transformer."""

    def __init__(
        self,
        obs_dim: int,
        num_keypoints: int,
        vision_dim: int = 256,
        model_dim: int = 256,
        model_name: str = DEFAULT_VIT_MODEL,
        vision_selected_layers: Sequence[int] = (2, 5, 8, 11),
        pretrained: bool = True,
        freeze_backbone: bool = True,
        local_grid_size: int = 5,
        local_radius_px: float = 16.0,
        local_layers: int = 4,
        global_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
        max_residual_px: float = 20.0,
    ) -> None:
        super().__init__()
        self.vision = ViTDenseEncoder(
            model_name=model_name,
            pretrained=pretrained,
            selected_layers=vision_selected_layers,
            output_dim=vision_dim,
            freeze_backbone=freeze_backbone,
        )
        self.grounding = AnchorConditionedResidualTransformer(
            visual_dim=vision_dim,
            obs_dim=obs_dim,
            num_keypoints=num_keypoints,
            model_dim=model_dim,
            local_grid_size=local_grid_size,
            local_radius_px=local_radius_px,
            local_layers=local_layers,
            global_layers=global_layers,
            heads=heads,
            dropout=dropout,
            max_residual_px=max_residual_px,
        )

    def _ground_from_vision(
        self,
        vision: DenseVisionOutput,
        q_nominal_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> TransformerGroundingOutput:
        result = self.grounding(
            vision.feature_map,
            q_nominal_px,
            keypoints_xyz,
            robot_obs,
            image_size=image_size,
        )
        return TransformerGroundingOutput(
            delta_px=result.delta_px,
            q_corrected_px=result.q_corrected_px,
            reliability=result.reliability,
            reliability_logit=result.reliability_logit,
            keypoint_latent=result.keypoint_latent,
            feature_map=vision.feature_map,
            global_feature=vision.global_feature,
        )

    def forward_from_backbone_features(
        self,
        feature_maps: torch.Tensor,
        prefix: torch.Tensor,
        q_nominal_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> TransformerGroundingOutput:
        vision = self.vision.project_backbone_features(feature_maps, prefix)
        return self._ground_from_vision(
            vision,
            q_nominal_px,
            keypoints_xyz,
            robot_obs,
            image_size=image_size,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        q_nominal_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
    ) -> TransformerGroundingOutput:
        vision = self.vision(rgb)
        return self._ground_from_vision(
            vision,
            q_nominal_px,
            keypoints_xyz,
            robot_obs,
            image_size=(int(rgb.shape[-1]), int(rgb.shape[-2])),
        )


class LeWMFCPGAdapter(nn.Module):
    """Dense ViT feature extractor plus residual-only LeWM corrector."""

    def __init__(
        self,
        obs_dim: int,
        num_keypoints: int,
        vision_dim: int = 256,
        model_dim: int = 256,
        model_name: str = DEFAULT_VIT_MODEL,
        vision_selected_layers: Sequence[int] = (2, 5, 8, 11),
        pretrained: bool = True,
        freeze_backbone: bool = True,
        local_grid_size: int = 5,
        local_radius_px: float = 16.0,
        local_layers: int = 4,
        global_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
        max_residual_px: float = 20.0,
    ) -> None:
        super().__init__()
        self.vision = ViTDenseEncoder(
            model_name=model_name,
            pretrained=pretrained,
            selected_layers=vision_selected_layers,
            output_dim=vision_dim,
            freeze_backbone=freeze_backbone,
        )
        self.grounding = LeWMAnchorResidualCorrector(
            visual_dim=vision_dim,
            obs_dim=obs_dim,
            num_keypoints=num_keypoints,
            model_dim=model_dim,
            local_grid_size=local_grid_size,
            local_radius_px=local_radius_px,
            local_layers=local_layers,
            global_layers=global_layers,
            heads=heads,
            dropout=dropout,
            max_residual_px=max_residual_px,
        )

    def _ground_from_vision(
        self,
        vision: DenseVisionOutput,
        q_nominal_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> LeWMGroundingOutput:
        result = self.grounding(
            vision.feature_map,
            q_nominal_px,
            keypoints_xyz,
            robot_obs,
            image_size=image_size,
        )
        return LeWMGroundingOutput(
            delta_px=result.delta_px,
            q_corrected_px=result.q_corrected_px,
            keypoint_latent=result.keypoint_latent,
            feature_map=vision.feature_map,
            global_feature=vision.global_feature,
        )

    def forward_from_backbone_features(
        self,
        feature_maps: torch.Tensor,
        prefix: torch.Tensor,
        q_nominal_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
        image_size: tuple[int, int],
    ) -> LeWMGroundingOutput:
        vision = self.vision.project_backbone_features(feature_maps, prefix)
        return self._ground_from_vision(
            vision,
            q_nominal_px,
            keypoints_xyz,
            robot_obs,
            image_size=image_size,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        q_nominal_px: torch.Tensor,
        keypoints_xyz: torch.Tensor,
        robot_obs: torch.Tensor,
    ) -> LeWMGroundingOutput:
        vision = self.vision(rgb)
        return self._ground_from_vision(
            vision,
            q_nominal_px,
            keypoints_xyz,
            robot_obs,
            image_size=(int(rgb.shape[-1]), int(rgb.shape[-2])),
        )



def load_fcpg_vit_adapter(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[FCPGViTAdapter | LeWMFCPGAdapter, dict[str, object]]:
    """Reconstruct an FCPG ViT adapter from a training checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = dict(checkpoint.get("args", {}))
    corrector_arch = str(
        checkpoint.get("corrector_arch", args.get("corrector_arch", "legacy"))
    ).lower()
    if corrector_arch == "legacy":
        model_class = FCPGViTAdapter
    elif corrector_arch == "lewm":
        model_class = LeWMFCPGAdapter
    else:
        raise ValueError(f"unknown corrector architecture: {corrector_arch}")

    schema_version = int(checkpoint.get("schema_version", 1))
    if corrector_arch == "lewm" and schema_version >= 2:
        upstream_commit = checkpoint.get("lewm_upstream_commit")
        if upstream_commit != LEWM_UPSTREAM_COMMIT:
            raise ValueError(
                "LeWM schema-v2 checkpoint has missing or mismatched upstream commit"
            )
        if bool(checkpoint.get("has_reliability", False)):
            raise ValueError("LeWM residual-only checkpoint cannot declare reliability output")
    if schema_version >= 2:
        args.setdefault("corrector_arch", corrector_arch)
        args.setdefault("schema_version", schema_version)
        if "lewm_upstream_commit" in checkpoint:
            args.setdefault("lewm_upstream_commit", checkpoint["lewm_upstream_commit"])
    model = model_class(
        obs_dim=int(checkpoint["obs_dim"]),
        num_keypoints=int(checkpoint["num_keypoints"]),
        vision_dim=int(args.get("vision_dim", 256)),
        model_dim=int(args.get("model_dim", 256)),
        model_name=str(args.get("model_name", DEFAULT_VIT_MODEL)),
        vision_selected_layers=tuple(
            int(layer) for layer in args.get("vision_selected_layers", (2, 5, 8, 11))
        ),
        pretrained=False,
        freeze_backbone=not bool(args.get("unfreeze_backbone", False)),
        local_grid_size=int(args.get("local_grid_size", 5)),
        local_radius_px=float(args.get("local_radius_px", 16.0)),
        local_layers=int(args.get("local_layers", 4)),
        global_layers=int(args.get("global_layers", 2)),
        heads=int(args.get("heads", 8)),
        dropout=float(args.get("dropout", 0.1)),
        max_residual_px=float(args.get("max_residual_px", 20.0)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, args
