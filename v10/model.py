"""Unified V10: frozen ViT, fresh FCPG, memory, dynamics, and action head."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
LEWM_ROOT = REPO_ROOT / "third_party" / "lewm"
if str(LEWM_ROOT) not in sys.path:
    sys.path.insert(0, str(LEWM_ROOT))

from stable_pretraining.backbone.utils import vit_hf  # noqa: E402

from fcpg.so101_fcpg_lewm_bridge import SO101FCPGAnchorBridge  # noqa: E402
from fcpg.so101_predictive_action_policy import (  # noqa: E402
    SO101PredictiveActionOutput,
    SO101PredictiveActionPolicy,
)
from fcpg.transformer_grounding import LeWMAnchorResidualCorrector  # noqa: E402
from v10.config import V10Config, VIT_STATE_PREFIX  # noqa: E402


EMBEDDED_V10_VIT_STATE_PREFIX = "vit.encoder."


def _extract_vit_encoder_state(
    state: dict[str, torch.Tensor],
    *,
    expected_parameter_count: int = 5_501_376,
) -> tuple[dict[str, torch.Tensor], str]:
    """Extract one exact ViT namespace from an original V9 or embedded V10+ state."""

    candidates: list[tuple[str, dict[str, torch.Tensor]]] = []
    for prefix in (VIT_STATE_PREFIX, EMBEDDED_V10_VIT_STATE_PREFIX):
        selected = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if isinstance(key, str)
            and key.startswith(prefix)
            and isinstance(value, torch.Tensor)
        }
        if selected:
            candidates.append((prefix, selected))
    if not candidates:
        raise ValueError(
            "checkpoint contains no ViT tensors under either known prefix: "
            f"{VIT_STATE_PREFIX!r}, {EMBEDDED_V10_VIT_STATE_PREFIX!r}"
        )
    if len(candidates) != 1:
        raise ValueError("checkpoint contains ambiguous original and embedded ViT namespaces")
    prefix, selected = candidates[0]
    count = sum(tensor.numel() for tensor in selected.values())
    if count != int(expected_parameter_count):
        raise ValueError(
            f"V10 expected {expected_parameter_count:,} ViT parameters, found {count:,}"
        )
    return selected, prefix


@dataclass(frozen=True)
class GroundedHistory:
    context: torch.Tensor
    patch_tokens: torch.Tensor
    anchor_tokens: torch.Tensor
    q_corrected_px: torch.Tensor
    anchor_valid: torch.Tensor
    view_valid: torch.Tensor
    anchor_reliability_logit: torch.Tensor
    view_reliability_logit: torch.Tensor


@dataclass(frozen=True)
class V10TrajectoryOutput:
    grounded: GroundedHistory
    prediction: SO101PredictiveActionOutput
    target_future_context: torch.Tensor


def select_corrected_q_px(
    q_nominal_px: torch.Tensor,
    q_corrected_px: torch.Tensor,
    *,
    use_fcpg: bool = True,
    zero_residual: bool = False,
) -> torch.Tensor:
    """Choose the 2-D location used for visual re-sampling.

    The default (``use_fcpg=True``, ``zero_residual=False``) is the FCPG
    residual. ``zero_residual`` is opt-in and default-off: it forces the
    residual to zero so features are sampled at ``q_nominal_px`` only.
    Reliability scaling is independent of this choice.
    """

    if q_nominal_px.shape != q_corrected_px.shape:
        raise ValueError("q_nominal_px and q_corrected_px must share a shape")
    if use_fcpg and not zero_residual:
        return q_corrected_px
    return q_nominal_px


class _BoundedResidualProjector(nn.Module):
    """Train a fresh projector without destroying the frozen ViT at step zero."""

    def __init__(self, dimension: int, *, cap: float = 0.25) -> None:
        super().__init__()
        self.cap = float(cap)
        self.residual = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, 2 * dimension),
            nn.GELU(),
            nn.Linear(2 * dimension, dimension),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.cap * torch.tanh(self.residual(value))


class FrozenV9ViT(nn.Module):
    """Load exactly the ViT encoder tensors from V9 and nothing else."""

    def __init__(self, checkpoint: str | Path) -> None:
        super().__init__()
        path = Path(checkpoint).expanduser().resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model")
        if not isinstance(state, dict):
            raise ValueError("ViT source checkpoint must contain a model state mapping")
        selected, source_prefix = _extract_vit_encoder_state(state)
        self.encoder = vit_hf(
            size="tiny",
            patch_size=14,
            image_size=224,
            pretrained=False,
            use_mask_token=False,
        )
        missing, unexpected = self.encoder.load_state_dict(selected, strict=False)
        if missing or unexpected:
            raise ValueError(
                "V9 ViT extraction is not exact: "
                f"missing={list(missing)[:5]}, unexpected={list(unexpected)[:5]}"
            )
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.source_checkpoint = str(path)
        self.source_tensor_prefix = source_prefix
        self.tensor_count = len(selected)
        self.parameter_count = sum(parameter.numel() for parameter in self.encoder.parameters())
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True) -> "FrozenV9ViT":
        super().train(False)
        self.encoder.eval()
        return self

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [N,3,H,W]")
        pixels = F.interpolate(
            rgb.float(),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        pixels = (pixels - self.image_mean) / self.image_std
        with torch.no_grad():
            output = self.encoder(pixels, interpolate_pos_encoding=True)
        hidden = output.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[-1] != 192 or hidden.shape[1] != 257:
            raise ValueError("canonical V9 ViT must return CLS plus a 16x16 patch grid")
        return hidden


class SO101V10(nn.Module):
    """One checkpoint for feedback grounding and action-conditioned control."""

    camera_context_dim = 5

    def __init__(
        self,
        *,
        config: V10Config,
        vit_checkpoint: str | Path,
        dynamics_conditioning: str = "actor_latent",
        anchor_baseline_mode: str = "current",
        state_residual_mode: str = "uniform",
        state_residual_steps: float = 1.0,
        structured_goal: bool = False,
        structured_goal_source: str | None = None,
        goal_query_residual: bool | None = None,
        task_class_conditioning: bool = False,
        max_task_class: int = 8,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        if structured_goal_source is not None:
            structured_goal = True
        if structured_goal and structured_goal_source is None:
            structured_goal_source = "target_future_state_h4"
        if goal_query_residual is None:
            goal_query_residual = structured_goal_source == "target_future_state_h4"
        self.vit = FrozenV9ViT(vit_checkpoint)
        latent = config.latent_dim
        self.context_projector = _BoundedResidualProjector(latent)
        self.patch_projector = _BoundedResidualProjector(latent)
        self.corrector = LeWMAnchorResidualCorrector(
            visual_dim=latent,
            obs_dim=config.state_dim + self.camera_context_dim,
            num_keypoints=config.num_anchors,
            model_dim=latent,
            local_grid_size=5,
            local_radius_px=16.0,
            local_layers=2,
            global_layers=1,
            heads=config.heads,
            dropout=0.0,
            max_residual_px=20.0,
        )
        self.bridge = SO101FCPGAnchorBridge(
            latent_dim=latent,
            num_anchors=config.num_anchors,
            heads=config.heads,
            ffn_dim=2 * latent,
            dropout=0.0,
        )
        self.anchor_reliability_head = nn.Sequential(
            nn.LayerNorm(latent), nn.Linear(latent, 1)
        )
        self.view_reliability_head = nn.Sequential(
            nn.LayerNorm(latent), nn.Linear(latent, 1)
        )
        for head in (self.anchor_reliability_head[-1], self.view_reliability_head[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.policy = SO101PredictiveActionPolicy(
            latent_dim=latent,
            state_dim=config.state_dim,
            num_views=config.num_views,
            num_anchors=config.num_anchors,
            obs_horizon=config.obs_horizon,
            action_horizon=config.action_horizon,
            raw_steps_per_chunk=config.raw_steps_per_chunk,
            image_size=config.image_size,
            model_dim=config.model_dim,
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            dynamics_layers=config.dynamics_layers,
            heads=config.heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            dynamics_conditioning=dynamics_conditioning,
            anchor_baseline_mode=anchor_baseline_mode,
            state_residual_mode=state_residual_mode,
            state_residual_steps=state_residual_steps,
            structured_goal=structured_goal,
            goal_query_residual=bool(goal_query_residual),
            structured_goal_source=structured_goal_source,
            task_class_conditioning=task_class_conditioning,
            max_task_class=max_task_class,
        )

    def train(self, mode: bool = True) -> "SO101V10":
        super().train(mode)
        self.vit.eval()
        return self

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def assert_vit_only_reuse(self) -> None:
        if any(parameter.requires_grad for parameter in self.vit.parameters()):
            raise RuntimeError("V10 ViT must remain frozen")
        if self.vit.parameter_count != 5_501_376:
            raise RuntimeError("V10 loaded an unexpected ViT")

    def set_action_contract(
        self,
        step_cap: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> None:
        self.policy.set_action_contract(step_cap, lower, upper)

    def _encode_flat(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.vit(rgb)
        return self.context_projector(hidden[:, 0]), self.patch_projector(hidden[:, 1:])

    def encode_context_only(self, rgb: torch.Tensor) -> torch.Tensor:
        """Encode [B,T,V,3,H,W] for stop-gradient future visual targets."""

        if rgb.ndim != 6:
            raise ValueError("rgb must have shape [B,T,V,3,H,W]")
        batch, frames, views = rgb.shape[:3]
        context, _ = self._encode_flat(rgb.flatten(0, 2))
        return context.reshape(batch, frames, views, self.config.latent_dim)

    def encode_grounded_history(
        self,
        *,
        rgb: torch.Tensor,
        q_nominal_px: torch.Tensor,
        anchor_xyz_history: torch.Tensor,
        state_history: torch.Tensor,
        camera_context: torch.Tensor,
        anchor_valid: torch.Tensor,
        view_valid: torch.Tensor,
        use_fcpg: bool = True,
        zero_residual: bool = False,
    ) -> GroundedHistory:
        if rgb.ndim != 6:
            raise ValueError("rgb must have shape [B,T,V,3,H,W]")
        batch, frames, views, channels, height, width = rgb.shape
        expected = (batch, frames, views)
        if (frames, views, channels) != (
            self.config.obs_horizon,
            self.config.num_views,
            3,
        ):
            raise ValueError("rgb does not match the V10 history/view contract")
        if q_nominal_px.shape != (*expected, self.config.num_anchors, 2):
            raise ValueError("q_nominal_px has an invalid shape")
        if anchor_xyz_history.shape != (
            batch,
            frames,
            self.config.num_anchors,
            3,
        ):
            raise ValueError("anchor_xyz_history has an invalid shape")
        if state_history.shape != (batch, frames, self.config.state_dim):
            raise ValueError("state_history has an invalid shape")
        if camera_context.shape != (*expected, self.camera_context_dim):
            raise ValueError("camera_context has an invalid shape")
        if anchor_valid.shape != (*expected, self.config.num_anchors):
            raise ValueError("anchor_valid has an invalid shape")
        if view_valid.shape != expected:
            raise ValueError("view_valid has an invalid shape")

        flat_context, flat_patches = self._encode_flat(rgb.flatten(0, 2))
        flat_q = q_nominal_px.flatten(0, 2)
        flat_xyz = (
            anchor_xyz_history[:, :, None]
            .expand(-1, -1, views, -1, -1)
            .flatten(0, 2)
        )
        flat_state = (
            state_history[:, :, None]
            .expand(-1, -1, views, -1)
            .flatten(0, 2)
        )
        flat_camera = camera_context.flatten(0, 2)
        feature_map = (
            flat_patches.reshape(-1, 16, 16, self.config.latent_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        correction = self.corrector(
            feature_map,
            flat_q,
            flat_xyz,
            torch.cat((flat_state, flat_camera), dim=-1),
            image_size=(width, height),
        )
        corrected_flat = select_corrected_q_px(
            flat_q,
            correction.q_corrected_px,
            use_fcpg=use_fcpg,
            zero_residual=zero_residual,
        )
        corrected = corrected_flat.reshape(
            batch, frames, views, self.config.num_anchors, 2
        )
        effective_valid = anchor_valid & view_valid[..., None]

        view_batch = batch * views
        bridge = self.bridge(
            flat_context.reshape(batch, frames, views, -1)
            .permute(0, 2, 1, 3)
            .reshape(view_batch, frames, -1),
            flat_patches.reshape(batch, frames, views, 256, -1)
            .permute(0, 2, 1, 3, 4)
            .reshape(view_batch, frames, 256, -1),
            q_nominal_px.permute(0, 2, 1, 3, 4).reshape(
                view_batch, frames, self.config.num_anchors, 2
            ),
            corrected.permute(0, 2, 1, 3, 4).reshape(
                view_batch, frames, self.config.num_anchors, 2
            ),
            anchor_xyz_history[:, None]
            .expand(-1, views, -1, -1, -1)
            .reshape(view_batch, frames, self.config.num_anchors, 3),
            effective_valid.permute(0, 2, 1, 3).reshape(
                view_batch, frames, self.config.num_anchors
            ),
            image_size=(width, height),
        )
        context = bridge.context_embedding.reshape(batch, views, frames, -1).permute(
            0, 2, 1, 3
        )
        tokens = bridge.anchor_tokens.reshape(
            batch, views, frames, self.config.num_anchors, -1
        ).permute(0, 2, 1, 3, 4)
        anchor_logit = self.anchor_reliability_head(
            correction.keypoint_latent
        ).squeeze(-1).reshape(batch, frames, views, self.config.num_anchors)
        view_logit = self.view_reliability_head(flat_context).squeeze(-1).reshape(
            batch, frames, views
        )
        # Clean measurements retain unit scale. Negative reliability evidence
        # can attenuate a corrupted measurement but can never amplify it.
        if use_fcpg:
            anchor_scale = torch.clamp(2.0 * torch.sigmoid(anchor_logit), max=1.0)
            view_scale = torch.clamp(2.0 * torch.sigmoid(view_logit), max=1.0)
        else:
            anchor_scale = torch.ones_like(anchor_logit)
            view_scale = torch.ones_like(view_logit)
        tokens = tokens * anchor_scale[..., None]
        context = context * view_scale[..., None]
        corrected = torch.where(
            effective_valid[..., None], corrected, q_nominal_px
        )
        return GroundedHistory(
            context=context,
            patch_tokens=flat_patches.reshape(
                batch, frames, views, 256, self.config.latent_dim
            ),
            anchor_tokens=tokens,
            q_corrected_px=corrected,
            anchor_valid=effective_valid,
            view_valid=view_valid,
            anchor_reliability_logit=anchor_logit,
            view_reliability_logit=view_logit,
        )

    def forward_trajectory(
        self,
        *,
        rgb_history: torch.Tensor,
        rgb_future: torch.Tensor,
        q_nominal_px: torch.Tensor,
        anchor_xyz_history: torch.Tensor,
        state_history: torch.Tensor,
        camera_context: torch.Tensor,
        anchor_valid: torch.Tensor,
        view_valid: torch.Tensor,
        past_action_chunks: torch.Tensor,
        forced_dynamics_action: torch.Tensor | None,
        future_anchor_xyz_baseline: torch.Tensor | None = None,
        use_fcpg: bool = True,
        zero_residual: bool = False,
        goal_state: torch.Tensor | None = None,
    ) -> V10TrajectoryOutput:
        grounded = self.encode_grounded_history(
            rgb=rgb_history,
            q_nominal_px=q_nominal_px,
            anchor_xyz_history=anchor_xyz_history,
            state_history=state_history,
            camera_context=camera_context,
            anchor_valid=anchor_valid,
            view_valid=view_valid,
            use_fcpg=use_fcpg,
            zero_residual=zero_residual,
        )
        prediction = self.policy(
            grounded.context,
            grounded.anchor_tokens,
            grounded.anchor_valid,
            grounded.view_valid,
            grounded.q_corrected_px,
            anchor_xyz_history,
            state_history,
            past_action_chunks,
            forced_dynamics_action=forced_dynamics_action,
            future_anchor_xyz_baseline=future_anchor_xyz_baseline,
            goal_state=goal_state,
        )
        target_context = self.encode_context_only(rgb_future).detach()
        return V10TrajectoryOutput(
            grounded=grounded,
            prediction=prediction,
            target_future_context=target_context,
        )

    def fcpg_forward(
        self,
        *,
        rgb: torch.Tensor,
        q_nominal_px: torch.Tensor,
        anchors_xyz: torch.Tensor,
        state: torch.Tensor,
        camera_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context, patches = self._encode_flat(rgb)
        feature_map = (
            patches.reshape(-1, 16, 16, self.config.latent_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        result = self.corrector(
            feature_map,
            q_nominal_px,
            anchors_xyz,
            torch.cat((state, camera_context), dim=-1),
            image_size=(int(rgb.shape[-1]), int(rgb.shape[-2])),
        )
        anchor_logit = self.anchor_reliability_head(result.keypoint_latent).squeeze(-1)
        view_logit = self.view_reliability_head(context).squeeze(-1)
        return result.q_corrected_px, anchor_logit, view_logit, result.keypoint_latent


__all__ = [
    "FrozenV9ViT",
    "GroundedHistory",
    "SO101V10",
    "V10TrajectoryOutput",
]
