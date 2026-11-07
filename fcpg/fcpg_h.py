"""Calibration-free temporal FCPG-H and its LeWM token feedback contract."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from fcpg.modules import sample_feature_map_at_pixels
from fcpg.offline_pipeline_manifest import sha256_file
from fcpg.temporal_corruptions import warp_endpoint_heatmaps
from fcpg.temporal_grounding_protocol import (
    GROUNDING_ENDPOINT_INDICES,
    GROUNDING_HISTORY_SLICES,
)


FCPGH_MAX_TRAINABLE_PARAMETERS = 1_500_000
TEMPORAL_ADAPTER_MAX_TRAINABLE_PARAMETERS = 1_000_000
FCPGH_NCE_TEMPERATURE = 0.10
FEEDBACK_BETA = 0.25
FEEDBACK_TEMPERATURE = 0.10
LEWM_SIGREG_WEIGHT = 0.02
FCPGH_METHOD_REVISION = "fcpg-h-v4-online-student-endpoint"
FCPGH_CHECKPOINT_FORMAT = "so101-fcpg-h-checkpoint-v4"
FCPGH_INFERENCE_STATE_KEY = "core"


def fcpg_h_inference_state(payload: Mapping[str, object]) -> dict[str, torch.Tensor]:
    """Return the fixed online-student endpoint; EMA is a train-only teacher."""

    if payload.get("format") != FCPGH_CHECKPOINT_FORMAT:
        raise ValueError("unexpected FCPG-H checkpoint format")
    if payload.get("method_revision") != FCPGH_METHOD_REVISION:
        raise ValueError("unexpected FCPG-H method revision")
    state = payload.get(FCPGH_INFERENCE_STATE_KEY)
    if not isinstance(state, dict) or not state:
        raise ValueError("FCPG-H checkpoint has no online-student state")
    if not all(isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in state.items()):
        raise ValueError("FCPG-H online-student state is malformed")
    return state


@dataclass(frozen=True)
class TemporalGroundingBatch:
    images: torch.Tensor
    states: torch.Tensor
    completed_actions: torch.Tensor
    view_valid: torch.Tensor
    cad_motion_signatures: torch.Tensor | None = None

    def validate(self) -> None:
        if self.images.ndim != 6 or self.images.shape[1:3] != (16, self.view_valid.shape[1]):
            raise ValueError("images must have shape [B,16,V,3,224,224]")
        if self.images.shape[3:] != (3, 224, 224):
            raise ValueError("images must have shape [B,16,V,3,224,224]")
        batch, _, views = self.images.shape[:3]
        if self.states.shape != (batch, 16, 6):
            raise ValueError("states must have shape [B,16,6]")
        if self.completed_actions.shape != (batch, 15, 6):
            raise ValueError("completed_actions must have shape [B,15,6]")
        if self.view_valid.shape != (batch, views) or self.view_valid.dtype != torch.bool:
            raise ValueError("view_valid must be bool [B,V]")
        if self.cad_motion_signatures is not None and self.cad_motion_signatures.shape != (batch, 16, 64):
            raise ValueError("cad_motion_signatures must have shape [B,16,64]")


@dataclass(frozen=True)
class FCPGHOutput:
    tip_heatmaps: torch.Tensor
    tip_px: torch.Tensor
    anchor_tokens: torch.Tensor
    tip_logits: torch.Tensor
    image_descriptors: torch.Tensor
    motion_queries: torch.Tensor


@dataclass(frozen=True)
class FCPGHLoss:
    loss: torch.Tensor
    sensorimotor_nce: torch.Tensor
    corruption_equivariance: torch.Tensor


@dataclass(frozen=True)
class LeWMPrior:
    tokens: torch.Tensor
    residual: torch.Tensor
    dynamics_latent: torch.Tensor


@dataclass(frozen=True)
class TemporalLeWMLoss:
    loss: torch.Tensor
    prediction_loss: torch.Tensor
    sigreg_loss: torch.Tensor


@dataclass(frozen=True)
class PredictiveQueryPosterior:
    tip_heatmaps: torch.Tensor
    tip_px: torch.Tensor
    anchor_tokens: torch.Tensor
    posterior_logits: torch.Tensor
    prior_similarity: torch.Tensor


@dataclass(frozen=True)
class PredictiveQueryRollout:
    posterior_tokens: torch.Tensor
    posterior_tip_px: torch.Tensor
    prior_tokens: torch.Tensor


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def assert_parameter_budget(module: nn.Module, *, maximum: int, name: str) -> None:
    count = trainable_parameter_count(module)
    if count > maximum:
        raise RuntimeError(f"{name} has {count:,} trainable parameters, above {maximum:,}")


def build_cad_motion_signatures(
    states: torch.Tensor,
    completed_actions: torch.Tensor,
    tip_xyz: torch.Tensor,
    tip_jacobians: torch.Tensor,
) -> torch.Tensor:
    """Build the explicit 64-D CAD motion signature stored by preprocessing.

    Jacobians are derivatives of the two fingertip XYZ positions with respect
    to the six canonical joint coordinates.  The function performs no camera
    projection and consumes no RGB.
    """

    if states.ndim != 3 or states.shape[-1] != 6 or states.shape[1] < 2:
        raise ValueError("states must have shape [B,T>=2,6]")
    batch, time = states.shape[:2]
    if completed_actions.shape != (batch, time - 1, 6):
        raise ValueError("completed_actions must have shape [B,T-1,6]")
    if tip_xyz.shape != (batch, time, 2, 3):
        raise ValueError("tip_xyz must have shape [B,T,2,3]")
    if tip_jacobians.shape != (batch, time, 2, 3, 6):
        raise ValueError("tip_jacobians must have shape [B,T,2,3,6]")

    delta_state = torch.zeros_like(states)
    delta_state[:, 1:] = states[:, 1:] - states[:, :-1]
    command = torch.zeros_like(states)
    command[:, 0] = states[:, 0]
    command[:, 1:] = completed_actions
    servo_error = torch.zeros_like(states)
    servo_error[:, 1:] = completed_actions - states[:, 1:]
    command_increment = torch.zeros_like(states)
    command_increment[:, 1:] = completed_actions - states[:, :-1]
    tip_delta = torch.zeros_like(tip_xyz)
    tip_delta[:, 1:] = tip_xyz[:, 1:] - tip_xyz[:, :-1]
    jacobian_measured = torch.einsum("bticj,btj->btic", tip_jacobians, delta_state)
    jacobian_commanded = torch.einsum("bticj,btj->btic", tip_jacobians, command_increment)
    jacobian_column_norms = torch.linalg.vector_norm(tip_jacobians, dim=3)
    measured_tip_norm = torch.linalg.vector_norm(jacobian_measured, dim=-1)
    commanded_tip_norm = torch.linalg.vector_norm(jacobian_commanded, dim=-1)
    signature = torch.cat(
        (
            states,
            delta_state,
            command,
            servo_error,
            tip_xyz.flatten(-2),
            tip_delta.flatten(-2),
            jacobian_measured.flatten(-2),
            jacobian_commanded.flatten(-2),
            jacobian_column_norms.flatten(-2),
            measured_tip_norm,
            commanded_tip_norm,
        ),
        dim=-1,
    )
    if signature.shape != (batch, time, 64):
        raise AssertionError(f"CAD motion signature drifted to {signature.shape[-1]} dimensions")
    if not bool(torch.isfinite(signature).all()):
        raise ValueError("CAD motion signature contains non-finite values")
    return signature


class FrozenV12PatchEncoder(nn.Module):
    """The exact V12.4 ViT exposed as a frozen 16x16 patch map."""

    def __init__(self, checkpoint: str | Path) -> None:
        super().__init__()
        from v10.model import FrozenV9ViT

        self.vit = FrozenV9ViT(checkpoint)
        self.vit.requires_grad_(False)
        self.vit.eval()
        self.source_checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.source_sha256 = sha256_file(Path(self.source_checkpoint))

    def train(self, mode: bool = True) -> "FrozenV12PatchEncoder":
        super().train(False)
        self.vit.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ValueError("frozen V12.4 ViT input must be [N,3,224,224]")
        pixels = images.float()
        if images.dtype == torch.uint8:
            pixels = pixels / 255.0
        self.vit.eval()
        with torch.no_grad():
            tokens = self.vit(pixels)
        patches = tokens[:, 1:].reshape(-1, 16, 16, 192)
        return patches.permute(0, 3, 1, 2).contiguous()


class _CrossAttentionBlock(nn.Module):
    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dimension)
        self.memory_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, 2 * dimension),
            nn.GELU(),
            nn.Linear(2 * dimension, dimension),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        query = query + attended
        return query + self.ffn(query)


class FCPGHTemporalGrounder(nn.Module):
    """Two isolated towers which meet only through final cosine matching."""

    def __init__(
        self,
        *,
        visual_dim: int = 192,
        motion_dim: int = 64,
        model_dim: int = 96,
        heads: int = 4,
        history_enabled: bool = True,
        logit_temperature: float = FCPGH_NCE_TEMPERATURE,
    ) -> None:
        super().__init__()
        if visual_dim != 192:
            raise ValueError("FCPG-H is bound to the frozen V12.4 192-D ViT")
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        if logit_temperature <= 0:
            raise ValueError("logit temperature must be positive")
        self.visual_dim = visual_dim
        self.motion_dim = motion_dim
        self.model_dim = model_dim
        self.history_enabled = bool(history_enabled)
        self.logit_temperature = float(logit_temperature)

        # Visual tower: no proprioception, action, task, source, role, K, or T.
        self.visual_projection = nn.Linear(visual_dim, model_dim)
        self.history_position = nn.Parameter(torch.zeros(11, 8 * 8, model_dim))
        self.visual_blocks = nn.ModuleList(
            [_CrossAttentionBlock(model_dim, heads) for _ in range(2)]
        )
        self.visual_output = nn.LayerNorm(model_dim)

        # Kinematic tower: no RGB or image-derived tensor.
        self.motion_projection = nn.Linear(motion_dim, model_dim)
        motion_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=2 * model_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.motion_encoder = nn.TransformerEncoder(motion_layer, num_layers=2)
        self.motion_time = nn.Parameter(torch.zeros(16, model_dim))
        self.tip_identity = nn.Embedding(2, model_dim)
        self.motion_output = nn.LayerNorm(model_dim)
        nn.init.trunc_normal_(self.history_position, std=0.02)
        nn.init.trunc_normal_(self.motion_time, std=0.02)
        assert_parameter_budget(self, maximum=FCPGH_MAX_TRAINABLE_PARAMETERS, name="FCPG-H")

    def _visual_tower(self, patch_maps: torch.Tensor) -> torch.Tensor:
        batch, time, views, channels, grid_h, grid_w = patch_maps.shape
        if (time, channels, grid_h, grid_w) != (16, self.visual_dim, 16, 16):
            raise ValueError("patch_maps must have shape [B,16,V,192,16,16]")
        frozen = patch_maps.detach()
        pooled = F.avg_pool2d(
            frozen.reshape(batch * time * views, channels, 16, 16),
            kernel_size=2,
            stride=2,
        ).reshape(batch, time, views, channels, 8, 8)
        outputs: list[torch.Tensor] = []
        for endpoint, (start, stop) in zip(GROUNDING_ENDPOINT_INDICES, GROUNDING_HISTORY_SLICES, strict=True):
            current = frozen[:, endpoint].permute(0, 1, 3, 4, 2).reshape(
                batch * views, 16 * 16, channels
            )
            # Do not add a trainable absolute coordinate lookup to the current
            # grid.  A spatially constant image must stay spatially constant,
            # so the action/motion tower cannot emit coordinates by itself.
            query = self.visual_projection(current)
            if self.history_enabled:
                history = pooled[:, start:stop].permute(0, 2, 1, 4, 5, 3).reshape(
                    batch * views, 11 * 8 * 8, channels
                )
                memory = self.visual_projection(history)
                memory = memory + self.history_position.reshape(1, 11 * 8 * 8, self.model_dim)
            else:
                # Compute-matched H0 sees only the current image.  Repeating
                # its pooled cells preserves tensor shapes and parameter count.
                current_pooled = pooled[:, endpoint].permute(0, 1, 3, 4, 2).reshape(
                    batch * views, 8 * 8, channels
                )
                memory = self.visual_projection(current_pooled).repeat(1, 11, 1)
                memory = memory + self.history_position.reshape(1, 11 * 8 * 8, self.model_dim)
            for block in self.visual_blocks:
                query = block(query, memory)
            outputs.append(self.visual_output(query).reshape(batch, views, 16 * 16, self.model_dim))
        return torch.stack(outputs, dim=1)

    def _kinematic_tower(self, motion_signatures: torch.Tensor) -> torch.Tensor:
        if motion_signatures.ndim != 3 or motion_signatures.shape[1:] != (16, self.motion_dim):
            raise ValueError(f"motion_signatures must have shape [B,16,{self.motion_dim}]")
        signature = motion_signatures
        if not self.history_enabled:
            signature = torch.zeros_like(signature)
        motion = self.motion_projection(signature) + self.motion_time
        causal_mask = torch.triu(
            torch.ones(16, 16, device=motion.device, dtype=torch.bool), diagonal=1
        )
        encoded = self.motion_encoder(motion, mask=causal_mask)
        endpoints = encoded[:, GROUNDING_ENDPOINT_INDICES]
        queries = endpoints[:, :, None, :] + self.tip_identity.weight[None, None]
        return self.motion_output(queries)

    def forward(
        self,
        patch_maps: torch.Tensor,
        motion_signatures: torch.Tensor,
        *,
        view_valid: torch.Tensor | None = None,
        image_size: tuple[int, int] = (224, 224),
    ) -> FCPGHOutput:
        batch, _, views = patch_maps.shape[:3]
        if view_valid is None:
            view_valid = torch.ones(batch, views, dtype=torch.bool, device=patch_maps.device)
        if view_valid.shape != (batch, views) or view_valid.dtype != torch.bool:
            raise ValueError("view_valid must be bool [B,V]")
        visual = self._visual_tower(patch_maps)
        motion = self._kinematic_tower(motion_signatures)
        visual_normalized = F.normalize(visual, dim=-1)
        motion_normalized = F.normalize(motion, dim=-1)
        logits_flat = torch.einsum(
            "bevpd,betd->bevtp", visual_normalized, motion_normalized
        ) / self.logit_temperature
        logits = logits_flat.reshape(batch, 2, views, 2, 16, 16)
        heatmaps = logits.flatten(-2).softmax(dim=-1).reshape_as(logits)
        valid = view_valid[:, None, :, None, None, None]
        heatmaps = heatmaps * valid
        tip_px = heatmap_soft_argmax_px(heatmaps, image_size=image_size) * view_valid[:, None, :, None, None]
        center_px = tip_px.mean(dim=3, keepdim=True)
        anchors_px = torch.cat((center_px, tip_px), dim=3)
        endpoint_maps = patch_maps[:, GROUNDING_ENDPOINT_INDICES]
        flat_maps = endpoint_maps.reshape(batch * 2 * views, self.visual_dim, 16, 16)
        flat_points = anchors_px.reshape(batch * 2 * views, 3, 2)
        # CUDA grid_sample does not accept the fp16-map/bf16-grid mixture that
        # autocast can produce here.  This frozen-feature readout is tiny; keep
        # it explicitly fp32 while the temporal towers remain bf16-autocast.
        with torch.autocast(device_type=flat_maps.device.type, enabled=False):
            anchor_tokens = sample_feature_map_at_pixels(
                flat_maps.float(),
                flat_points.float(),
                image_size=image_size,
            ).reshape(batch, 2, views, 3, self.visual_dim)
        anchor_tokens = anchor_tokens * view_valid[:, None, :, None, None]
        return FCPGHOutput(
            tip_heatmaps=heatmaps,
            tip_px=tip_px,
            anchor_tokens=anchor_tokens,
            tip_logits=logits,
            image_descriptors=visual,
            motion_queries=motion,
        )


class FCPGH(nn.Module):
    """Frozen V12.4 encoder plus the trainable temporal grounding core."""

    def __init__(self, checkpoint: str | Path, *, core: FCPGHTemporalGrounder | None = None) -> None:
        super().__init__()
        self.encoder = FrozenV12PatchEncoder(checkpoint)
        self.core = core or FCPGHTemporalGrounder()
        self.encoder.requires_grad_(False)

    def train(self, mode: bool = True) -> "FCPGH":
        super().train(mode)
        self.encoder.train(False)
        self.core.train(mode)
        return self

    def encode_images(self, images: torch.Tensor, *, chunk_rows: int = 256) -> torch.Tensor:
        if images.ndim != 6 or images.shape[1] != 16 or images.shape[3:] != (3, 224, 224):
            raise ValueError("images must have shape [B,16,V,3,224,224]")
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        batch, time, views = images.shape[:3]
        rows = images.reshape(batch * time * views, 3, 224, 224)
        encoded = [self.encoder(rows[start : start + chunk_rows]) for start in range(0, len(rows), chunk_rows)]
        return torch.cat(encoded).reshape(batch, time, views, 192, 16, 16)

    def forward(self, batch: TemporalGroundingBatch, *, patch_maps: torch.Tensor | None = None) -> FCPGHOutput:
        batch.validate()
        if batch.cad_motion_signatures is None:
            raise ValueError("formal FCPG-H requires cached 64-D CAD motion signatures")
        features = self.encode_images(batch.images) if patch_maps is None else patch_maps
        return self.core(features, batch.cad_motion_signatures, view_valid=batch.view_valid)


def heatmap_soft_argmax_px(
    heatmaps: torch.Tensor,
    *,
    image_size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    if heatmaps.ndim < 3:
        raise ValueError("heatmaps must end in [H,W]")
    height, width = heatmaps.shape[-2:]
    image_width, image_height = image_size
    x = (torch.arange(width, device=heatmaps.device, dtype=heatmaps.dtype) + 0.5) * image_width / width - 0.5
    y = (torch.arange(height, device=heatmaps.device, dtype=heatmaps.dtype) + 0.5) * image_height / height - 0.5
    probability = heatmaps / heatmaps.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    px = (probability.sum(dim=-2) * x).sum(dim=-1)
    py = (probability.sum(dim=-1) * y).sum(dim=-1)
    return torch.stack((px, py), dim=-1)


def _multi_positive_nce(logits: torch.Tensor, positive: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~valid[None], -torch.inf)
    numerator = torch.logsumexp(masked.masked_fill(~positive, -torch.inf), dim=-1)
    denominator = torch.logsumexp(masked, dim=-1)
    anchor_valid = valid & positive.any(dim=-1) & torch.isfinite(numerator)
    if not bool(anchor_valid.any()):
        return logits.sum() * 0.0
    return (denominator[anchor_valid] - numerator[anchor_valid]).mean()


def sensorimotor_nce_loss(
    output: FCPGHOutput,
    *,
    motion_cluster: torch.Tensor,
    source_index: torch.Tensor,
    task_index: torch.Tensor,
    high_motion: torch.Tensor,
    view_valid: torch.Tensor,
    temperature: float = FCPGH_NCE_TEMPERATURE,
) -> torch.Tensor:
    """Multi-positive cross-modal NCE; metadata builds masks, never tower inputs."""

    heatmaps = output.tip_heatmaps
    batch, endpoints, views, tips = heatmaps.shape[:4]
    if motion_cluster.shape == (batch,):
        motion_cluster = motion_cluster[:, None].expand(-1, endpoints)
    if high_motion.shape == (batch,):
        high_motion = high_motion[:, None].expand(-1, endpoints)
    if motion_cluster.shape != (batch, endpoints) or high_motion.shape != (batch, endpoints):
        raise ValueError("motion_cluster/high_motion must have shape [B] or [B,2]")
    if source_index.shape != (batch,) or task_index.shape != (batch,):
        raise ValueError("source_index/task_index must have shape [B]")
    if view_valid.shape != (batch, views):
        raise ValueError("view_valid must have shape [B,V]")

    probability = heatmaps.flatten(-2)
    descriptors = output.image_descriptors[:, :, :, None].expand(-1, -1, -1, tips, -1, -1)
    visual_tube = torch.einsum("bevtp,bevtpd->bevtd", probability, descriptors)
    queries = output.motion_queries[:, :, None].expand(-1, -1, views, -1, -1)
    visual_flat = F.normalize(visual_tube.reshape(-1, visual_tube.shape[-1]), dim=-1)
    query_flat = F.normalize(queries.reshape(-1, queries.shape[-1]), dim=-1)
    logits = visual_flat @ query_flat.T / float(temperature)

    b = torch.arange(batch, device=logits.device)[:, None, None, None].expand(batch, endpoints, views, tips).reshape(-1)
    e = torch.arange(endpoints, device=logits.device)[None, :, None, None].expand(batch, endpoints, views, tips).reshape(-1)
    v = torch.arange(views, device=logits.device)[None, None, :, None].expand(batch, endpoints, views, tips).reshape(-1)
    tip = torch.arange(tips, device=logits.device)[None, None, None, :].expand(batch, endpoints, views, tips).reshape(-1)
    cluster = motion_cluster[:, :, None, None].expand(batch, endpoints, views, tips).reshape(-1)
    source = source_index[:, None, None, None].expand(batch, endpoints, views, tips).reshape(-1)
    task = task_index[:, None, None, None].expand(batch, endpoints, views, tips).reshape(-1)
    active = high_motion[:, :, None, None].expand(batch, endpoints, views, tips).reshape(-1)
    valid = active & view_valid[b, v]
    same_window = (b[:, None] == b[None]) & (e[:, None] == e[None]) & (tip[:, None] == tip[None])
    cluster_positive = (
        (cluster[:, None] == cluster[None])
        & (tip[:, None] == tip[None])
        & ((source[:, None] != source[None]) | (task[:, None] != task[None]))
    )
    positive = (same_window | cluster_positive) & valid[:, None] & valid[None]
    forward = _multi_positive_nce(logits, positive, valid)
    backward = _multi_positive_nce(logits.T, positive.T, valid)
    return 0.5 * (forward + backward)


def corruption_equivariance_loss(
    corrupted_output: FCPGHOutput,
    ema_clean_heatmaps: torch.Tensor,
    source_to_target: torch.Tensor,
) -> torch.Tensor:
    transformed = warp_endpoint_heatmaps(ema_clean_heatmaps.detach(), source_to_target)
    target = transformed.float().clamp_min(0.0)
    target = target / target.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    prediction = corrupted_output.tip_heatmaps.float().clamp_min(1e-8)
    prediction = prediction / prediction.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    # Distributional KL has zero at exact equivariance and a useful scale for
    # 16x16 heatmaps. Per-cell Smooth-L1 was ~1e-7 and numerically erased this
    # supposedly equal-weight objective next to an O(1) NCE term.
    divergence = target * (target.clamp_min(1e-8).log() - prediction.log())
    return divergence.sum(dim=(-2, -1)).mean()


def fcpg_h_objective(
    *,
    sensorimotor_nce: torch.Tensor,
    corruption_equivariance: torch.Tensor,
) -> FCPGHLoss:
    if sensorimotor_nce.ndim or corruption_equivariance.ndim:
        raise ValueError("FCPG-H objectives must be scalar tensors")
    return FCPGHLoss(
        loss=sensorimotor_nce + corruption_equivariance,
        sensorimotor_nce=sensorimotor_nce,
        corruption_equivariance=corruption_equivariance,
    )


class GroundedTemporalAdapter(nn.Module):
    """Shared-view bounded residual for four real future grounded-token targets."""

    def __init__(
        self,
        *,
        latent_dim: int = 192,
        action_dim: int = 30,
        max_latent_residual: float = 0.25,
    ) -> None:
        super().__init__()
        if max_latent_residual <= 0:
            raise ValueError("max_latent_residual must be positive")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.max_latent_residual = float(max_latent_residual)
        self.grounded_fusion = nn.Sequential(
            nn.LayerNorm(2 * latent_dim),
            nn.Linear(2 * latent_dim, latent_dim),
            nn.GELU(),
        )
        self.state_encoder = nn.Sequential(nn.LayerNorm(18), nn.Linear(18, latent_dim), nn.GELU())
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(action_dim), nn.Linear(action_dim, latent_dim), nn.GELU()
        )
        self.anchor_identity = nn.Embedding(3, latent_dim)
        self.horizon_identity = nn.Embedding(4, latent_dim)
        self.dynamics_norm = nn.LayerNorm(latent_dim)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(6 * latent_dim),
            nn.Linear(6 * latent_dim, 2 * latent_dim),
            nn.GELU(),
            nn.Linear(2 * latent_dim, latent_dim),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        assert_parameter_budget(
            self,
            maximum=TEMPORAL_ADAPTER_MAX_TRAINABLE_PARAMETERS,
            name="GroundedTemporalAdapter",
        )

    def forward(
        self,
        *,
        frozen_base_prediction: torch.Tensor,
        global_context: torch.Tensor,
        current_anchor_tokens: torch.Tensor,
        states: torch.Tensor,
        future_action_chunks: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> LeWMPrior:
        if frozen_base_prediction.ndim != 5:
            raise ValueError("frozen_base_prediction must have shape [B,4,V,3,D]")
        batch, horizons, views, anchors, latent = frozen_base_prediction.shape
        if (horizons, anchors, latent) != (4, 3, self.latent_dim):
            raise ValueError("frozen base prediction violates H4/anchor/latent contract")
        if global_context.shape != (batch, views, latent):
            raise ValueError("global_context must have shape [B,V,D]")
        if current_anchor_tokens.shape != (batch, views, 3, latent):
            raise ValueError("current_anchor_tokens must have shape [B,V,3,D]")
        if states.shape != (batch, 16, 6):
            raise ValueError("states must have shape [B,16,6]")
        if future_action_chunks.shape != (batch, 4, self.action_dim):
            raise ValueError("future_action_chunks must have shape [B,4,30]")
        if view_valid.shape != (batch, views) or view_valid.dtype != torch.bool:
            raise ValueError("view_valid must be bool [B,V]")

        base = frozen_base_prediction.detach()
        valid = view_valid[:, None, :, None, None]
        global_detached = global_context.detach() * view_valid[..., None]
        anchors_detached = current_anchor_tokens.detach() * view_valid[..., None, None]
        grounded = self.grounded_fusion(
            torch.cat((anchors_detached, global_detached[:, :, None].expand(-1, -1, 3, -1)), dim=-1)
        )
        grounded = grounded[:, None].expand(-1, 4, -1, -1, -1)
        state_summary = torch.cat(
            (states[:, -1], states[:, -1] - states[:, -2], states.mean(dim=1)), dim=-1
        )
        state = self.state_encoder(state_summary)[:, None, None, None].expand(-1, 4, views, 3, -1)
        action = self.action_encoder(future_action_chunks)[:, :, None, None].expand(-1, -1, views, 3, -1)
        anchor = self.anchor_identity.weight[None, None, None].expand(batch, 4, views, -1, -1)
        horizon = self.horizon_identity.weight[None, :, None, None].expand(batch, -1, views, 3, -1)
        dynamics = self.dynamics_norm(base + grounded + state + action + anchor + horizon)
        residual_input = torch.cat((base, grounded, state, action, anchor, horizon), dim=-1)
        residual = torch.tanh(self.residual_head(residual_input)) * self.max_latent_residual
        residual = residual * valid
        return LeWMPrior(tokens=(base + residual) * valid, residual=residual, dynamics_latent=dynamics * valid)


def _per_source_sigreg_temporal(
    latent: torch.Tensor,
    source_index: torch.Tensor,
    sigreg: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    batch, horizons, views, anchors, dimension = latent.shape
    if source_index.shape != (batch,):
        raise ValueError("source_index must have shape [B]")
    sequence = latent.reshape(batch, horizons * views * anchors, dimension)
    losses = []
    for source in torch.unique(source_index):
        selected = sequence[source_index == source]
        if selected.shape[0] >= 2:
            losses.append(sigreg(selected.transpose(0, 1)))
    return latent.sum() * 0.0 if not losses else torch.stack(losses).mean()


def temporal_lewm_objective(
    prior: LeWMPrior,
    future_measurement_tokens: torch.Tensor,
    source_index: torch.Tensor,
    *,
    sigreg: Callable[[torch.Tensor], torch.Tensor],
    sigreg_weight: float = LEWM_SIGREG_WEIGHT,
) -> TemporalLeWMLoss:
    if sigreg_weight not in {0.0, LEWM_SIGREG_WEIGHT}:
        raise ValueError("temporal LeWM permits only matched SIGReg=0 or production 0.02")
    if future_measurement_tokens.shape != prior.tokens.shape:
        raise ValueError("future measurement target must match LeWM prior tokens")
    prediction = F.smooth_l1_loss(prior.tokens.float(), future_measurement_tokens.detach().float())
    regularization = _per_source_sigreg_temporal(prior.dynamics_latent.float(), source_index, sigreg)
    return TemporalLeWMLoss(
        loss=prediction + float(sigreg_weight) * regularization,
        prediction_loss=prediction,
        sigreg_loss=regularization,
    )


def predictive_query_feedback(
    *,
    measurement_logits: torch.Tensor,
    current_patch_maps: torch.Tensor,
    prior_tokens: torch.Tensor,
    view_valid: torch.Tensor,
    beta: float = FEEDBACK_BETA,
    temperature: float = FEEDBACK_TEMPERATURE,
    image_size: tuple[int, int] = (224, 224),
) -> PredictiveQueryPosterior:
    """Use a stopped-gradient token prior as a query, never as a pixel state."""

    if beta != FEEDBACK_BETA or temperature != FEEDBACK_TEMPERATURE:
        raise ValueError("predictive-query beta=0.25 and temperature=0.1 are frozen")
    if measurement_logits.ndim != 5 or measurement_logits.shape[2:] != (2, 16, 16):
        raise ValueError("measurement_logits must have shape [B,V,2,16,16]")
    batch, views = measurement_logits.shape[:2]
    if current_patch_maps.shape != (batch, views, 192, 16, 16):
        raise ValueError("current_patch_maps must have shape [B,V,192,16,16]")
    if prior_tokens.shape != (batch, views, 3, 192):
        raise ValueError("prior_tokens must have shape [B,V,3,192]")
    if view_valid.shape != (batch, views):
        raise ValueError("view_valid must have shape [B,V]")
    features = F.normalize(current_patch_maps.detach(), dim=2)
    tip_prior = F.normalize(prior_tokens[:, :, 1:].detach(), dim=-1)
    similarity = torch.einsum("bvdhw,bvtd->bvthw", features, tip_prior)
    posterior_logits = measurement_logits + float(beta) * similarity / float(temperature)
    heatmaps = posterior_logits.flatten(-2).softmax(dim=-1).reshape_as(posterior_logits)
    heatmaps = heatmaps * view_valid[:, :, None, None, None]
    tip_px = heatmap_soft_argmax_px(heatmaps, image_size=image_size) * view_valid[:, :, None, None]
    anchor_px = torch.cat((tip_px.mean(dim=2, keepdim=True), tip_px), dim=2)
    tokens = sample_feature_map_at_pixels(
        current_patch_maps.reshape(batch * views, 192, 16, 16),
        anchor_px.reshape(batch * views, 3, 2),
        image_size=image_size,
    ).reshape(batch, views, 3, 192)
    tokens = tokens * view_valid[:, :, None, None]
    return PredictiveQueryPosterior(
        tip_heatmaps=heatmaps,
        tip_px=tip_px,
        anchor_tokens=tokens,
        posterior_logits=posterior_logits,
        prior_similarity=similarity,
    )


def rollout_predictive_query_feedback(
    *,
    measurement_logits: torch.Tensor,
    future_patch_maps: torch.Tensor,
    initial_posterior_tokens: torch.Tensor,
    executed_action_chunks: torch.Tensor,
    view_valid: torch.Tensor,
    predict_one_step: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
) -> PredictiveQueryRollout:
    """Run the locked H1-H4 token feedback loop with no actor or pixel state.

    ``predict_one_step`` receives the previous posterior *token*, the factual
    already-executed action chunk, and the horizon index.  The returned prior is
    stopped inside :func:`predictive_query_feedback`; only the newly measured
    posterior token is written into the next iteration.
    """

    if measurement_logits.ndim != 6 or measurement_logits.shape[1:] != (
        4,
        view_valid.shape[1],
        2,
        16,
        16,
    ):
        raise ValueError("measurement_logits must have shape [B,4,V,2,16,16]")
    batch, horizons, views = measurement_logits.shape[:3]
    if future_patch_maps.shape != (batch, horizons, views, 192, 16, 16):
        raise ValueError("future_patch_maps must have shape [B,4,V,192,16,16]")
    if initial_posterior_tokens.shape != (batch, views, 3, 192):
        raise ValueError("initial posterior token must have shape [B,V,3,192]")
    if executed_action_chunks.shape != (batch, horizons, 30):
        raise ValueError("executed_action_chunks must have shape [B,4,30]")
    history = initial_posterior_tokens
    posterior_tokens = []
    posterior_px = []
    priors = []
    for horizon in range(horizons):
        prior = predict_one_step(history, executed_action_chunks[:, horizon], horizon)
        if prior.shape != history.shape:
            raise ValueError("one-step LeWM prior must have shape [B,V,3,192]")
        posterior = predictive_query_feedback(
            measurement_logits=measurement_logits[:, horizon],
            current_patch_maps=future_patch_maps[:, horizon],
            prior_tokens=prior,
            view_valid=view_valid,
        )
        priors.append(prior)
        posterior_tokens.append(posterior.anchor_tokens)
        posterior_px.append(posterior.tip_px)
        history = posterior.anchor_tokens
    return PredictiveQueryRollout(
        posterior_tokens=torch.stack(posterior_tokens, dim=1),
        posterior_tip_px=torch.stack(posterior_px, dim=1),
        prior_tokens=torch.stack(priors, dim=1),
    )


def matched_temporal_adapter_pair(
    adapter: GroundedTemporalAdapter,
) -> tuple[GroundedTemporalAdapter, GroundedTemporalAdapter]:
    left, right = copy.deepcopy(adapter), copy.deepcopy(adapter)
    for first, second in zip(left.state_dict().values(), right.state_dict().values(), strict=True):
        if not torch.equal(first, second):
            raise AssertionError("matched temporal adapters lost identical initialization")
    return left, right
