"""Candidate-first temporal grounding for the FCPG-H V5 design.

The visual tower creates a dense local candidate bank without reading robot
state, action, CAD, source, task, or camera metadata.  Part-specific CAD
histories can only score this pre-existing bank.  The sensorimotor objective
compares whole image/history pairs, so a motion query never pools the visual
descriptor that is subsequently declared to be its own positive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from fcpg.fcpg_h import (
    FCPGH_MAX_TRAINABLE_PARAMETERS,
    FCPGH_NCE_TEMPERATURE,
    FrozenV12PatchEncoder,
    assert_parameter_budget,
    heatmap_soft_argmax_px,
)
from fcpg.modules import sample_feature_map_at_pixels
from fcpg.so101_kinematics import SO101_CALIBRATION_KEYPOINT_NAMES
from fcpg.temporal_corruptions import causal_trailing_blur
from fcpg.temporal_grounding_protocol import (
    GROUNDING_ENDPOINT_INDICES,
    GROUNDING_HISTORY_SLICES,
)


FCPGH_V5_METHOD_REVISION = "fcpg-h-v5-candidate-first-local-set-matching"
FCPGH_V5_CHECKPOINT_FORMAT = "so101-fcpg-h-candidate-first-checkpoint-v1"
PART_MOTION_DIM = 44
PART_COUNT = len(SO101_CALIBRATION_KEYPOINT_NAMES)
TIP_PART_INDICES = (PART_COUNT - 2, PART_COUNT - 1)


@dataclass(frozen=True)
class CandidateFirstOutput:
    part_heatmaps: torch.Tensor
    tip_heatmaps: torch.Tensor
    tip_px: torch.Tensor
    anchor_tokens: torch.Tensor
    part_logits: torch.Tensor
    image_candidates: torch.Tensor
    part_queries: torch.Tensor


@dataclass(frozen=True)
class CandidateFirstLocalization:
    part_heatmaps: torch.Tensor
    tip_heatmaps: torch.Tensor
    tip_px: torch.Tensor
    part_logits: torch.Tensor
    image_candidates: torch.Tensor
    part_queries: torch.Tensor


@dataclass(frozen=True)
class CandidateFirstLoss:
    loss: torch.Tensor
    sensorimotor_nce: torch.Tensor
    corruption_equivariance: torch.Tensor


def build_corrupted_local_patch_maps(
    clean_local_maps: torch.Tensor,
    corrupted_images: torch.Tensor,
    corruption_ids: torch.Tensor,
    encoder,
    *,
    blur_frames: int,
    chunk_rows: int = 256,
) -> torch.Tensor:
    """Reuse clean local rows; linear causal blur commutes with patch projection.

    Only jitter-containing rows (IDs 2/3) call the frozen patch encoder.  ID 0
    is an exact cache hit and ID 1 is formed by the same causal convex mixture
    in local-feature space.  The identity relies on patch projection and input
    normalization both being affine and blur weights summing to one.
    """

    if clean_local_maps.ndim != 6 or corrupted_images.ndim != 6:
        raise ValueError("clean local maps and corrupted images must be [B,T,V,C,H,W]")
    if clean_local_maps.shape[:3] != corrupted_images.shape[:3]:
        raise ValueError("clean local maps and corrupted images must align")
    if corruption_ids.shape != clean_local_maps.shape[:1]:
        raise ValueError("corruption_ids must have shape [B]")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    result = clean_local_maps.clone()
    blur_rows = corruption_ids == 1
    if bool(blur_rows.any()):
        blurred = causal_trailing_blur(clean_local_maps, frames=blur_frames)
        result[blur_rows] = blurred[blur_rows].to(dtype=result.dtype)
    encoded_rows = corruption_ids >= 2
    if bool(encoded_rows.any()):
        selected = corrupted_images[encoded_rows]
        rows = selected.reshape(-1, *selected.shape[-3:])
        encoded = torch.cat(
            [
                encoder(rows[start : start + chunk_rows])
                for start in range(0, len(rows), chunk_rows)
            ]
        )
        result[encoded_rows] = encoded.reshape_as(result[encoded_rows]).to(
            dtype=result.dtype
        )
    return result


def build_part_motion_signatures(
    states: torch.Tensor,
    completed_actions: torch.Tensor,
    part_xyz: torch.Tensor,
    part_jacobians: torch.Tensor,
) -> torch.Tensor:
    """Build one camera-free 44-D causal signature per fixed CAD part."""

    if states.ndim != 3 or states.shape[-1] != 6 or states.shape[1] < 2:
        raise ValueError("states must have shape [B,T>=2,6]")
    batch, time = states.shape[:2]
    if completed_actions.shape != (batch, time - 1, 6):
        raise ValueError("completed_actions must have shape [B,T-1,6]")
    if part_xyz.ndim != 4 or part_xyz.shape[:2] != (batch, time) or part_xyz.shape[-1] != 3:
        raise ValueError("part_xyz must have shape [B,T,M,3]")
    parts = part_xyz.shape[2]
    if part_jacobians.shape != (batch, time, parts, 3, 6):
        raise ValueError("part_jacobians must have shape [B,T,M,3,6]")

    delta_state = torch.zeros_like(states)
    delta_state[:, 1:] = states[:, 1:] - states[:, :-1]
    command = torch.zeros_like(states)
    command[:, 0] = states[:, 0]
    command[:, 1:] = completed_actions
    servo_error = torch.zeros_like(states)
    servo_error[:, 1:] = completed_actions - states[:, 1:]
    command_increment = torch.zeros_like(states)
    command_increment[:, 1:] = completed_actions - states[:, :-1]
    part_delta = torch.zeros_like(part_xyz)
    part_delta[:, 1:] = part_xyz[:, 1:] - part_xyz[:, :-1]
    jacobian_measured = torch.einsum(
        "btmpj,btj->btmp", part_jacobians, delta_state
    )
    jacobian_commanded = torch.einsum(
        "btmpj,btj->btmp", part_jacobians, command_increment
    )
    jacobian_column_norms = torch.linalg.vector_norm(part_jacobians, dim=3)
    measured_norm = torch.linalg.vector_norm(jacobian_measured, dim=-1, keepdim=True)
    commanded_norm = torch.linalg.vector_norm(jacobian_commanded, dim=-1, keepdim=True)
    shared = torch.cat((states, delta_state, command, servo_error), dim=-1)
    shared = shared[:, :, None].expand(-1, -1, parts, -1)
    signature = torch.cat(
        (
            shared,
            part_xyz,
            part_delta,
            jacobian_measured,
            jacobian_commanded,
            jacobian_column_norms,
            measured_norm,
            commanded_norm,
        ),
        dim=-1,
    )
    if signature.shape != (batch, time, parts, PART_MOTION_DIM):
        raise AssertionError("part motion signature dimension drifted")
    if not bool(torch.isfinite(signature).all()):
        raise ValueError("part motion signature contains non-finite values")
    return signature


def build_part_motion_signature_rows(
    current_states: torch.Tensor,
    previous_states: torch.Tensor,
    completed_actions: torch.Tensor,
    current_xyz: torch.Tensor,
    previous_xyz: torch.Tensor,
    current_jacobians: torch.Tensor,
    has_previous: torch.Tensor,
) -> torch.Tensor:
    """Build sparse per-frame signatures identical to the sequence builder."""

    if current_states.ndim != 2 or current_states.shape[-1] != 6:
        raise ValueError("current_states must be [N,6]")
    rows = len(current_states)
    if previous_states.shape != (rows, 6) or completed_actions.shape != (rows, 6):
        raise ValueError("previous state/action rows must align as [N,6]")
    if current_xyz.ndim != 3 or current_xyz.shape[0] != rows or current_xyz.shape[-1] != 3:
        raise ValueError("current_xyz must be [N,M,3]")
    parts = current_xyz.shape[1]
    if previous_xyz.shape != (rows, parts, 3):
        raise ValueError("previous_xyz must match current_xyz")
    if current_jacobians.shape != (rows, parts, 3, 6):
        raise ValueError("current_jacobians must be [N,M,3,6]")
    if has_previous.shape != (rows,) or has_previous.dtype != torch.bool:
        raise ValueError("has_previous must be bool [N]")
    active = has_previous[:, None]
    delta_state = torch.where(active, current_states - previous_states, 0.0)
    command = torch.where(active, completed_actions, current_states)
    servo_error = torch.where(active, completed_actions - current_states, 0.0)
    command_increment = torch.where(
        active, completed_actions - previous_states, 0.0
    )
    part_active = has_previous[:, None, None]
    part_delta = torch.where(part_active, current_xyz - previous_xyz, 0.0)
    jacobian_measured = torch.einsum(
        "nmpj,nj->nmp", current_jacobians, delta_state
    )
    jacobian_commanded = torch.einsum(
        "nmpj,nj->nmp", current_jacobians, command_increment
    )
    jacobian_column_norms = torch.linalg.vector_norm(current_jacobians, dim=2)
    measured_norm = torch.linalg.vector_norm(
        jacobian_measured, dim=-1, keepdim=True
    )
    commanded_norm = torch.linalg.vector_norm(
        jacobian_commanded, dim=-1, keepdim=True
    )
    shared = torch.cat(
        (current_states, delta_state, command, servo_error), dim=-1
    )[:, None].expand(-1, parts, -1)
    signature = torch.cat(
        (
            shared,
            current_xyz,
            part_delta,
            jacobian_measured,
            jacobian_commanded,
            jacobian_column_norms,
            measured_norm,
            commanded_norm,
        ),
        dim=-1,
    )
    if signature.shape != (rows, parts, PART_MOTION_DIM):
        raise AssertionError("sparse part motion signature dimension drifted")
    if not bool(torch.isfinite(signature).all()):
        raise ValueError("sparse part motion signature contains non-finite values")
    return signature


class FrozenV12LocalContextEncoder(FrozenV12PatchEncoder):
    """Expose frozen local patch projections separately from contextual V12 tokens."""

    @torch.no_grad()
    def encode_local(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ValueError("local V12 patch input must be [N,3,224,224]")
        pixels = images.float()
        if images.dtype == torch.uint8:
            pixels = pixels / 255.0
        pixels = (pixels - self.vit.image_mean) / self.vit.image_std
        projection = self.vit.encoder.embeddings.patch_embeddings.projection(pixels)
        if projection.shape[1:] != (192, 16, 16):
            raise ValueError("frozen V12 local patch projection changed shape")
        return projection


class _LocalCausalBlock(nn.Module):
    """Bounded spatiotemporal mixing with no global spatial summary."""

    def __init__(self, dimension: int, *, temporal_dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(
            dimension,
            dimension,
            kernel_size=3,
            padding=(temporal_dilation, 1, 1),
            dilation=(temporal_dilation, 1, 1),
            groups=dimension,
            padding_mode="replicate",
        )
        self.channel_norm = nn.LayerNorm(dimension)
        self.pointwise = nn.Linear(dimension, dimension)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mixed = self.depthwise(value)
        mixed = mixed.permute(0, 2, 3, 4, 1)
        mixed = self.pointwise(F.gelu(self.channel_norm(mixed)))
        return value + mixed.permute(0, 4, 1, 2, 3)


class CandidateFirstTemporalGrounder(nn.Module):
    """Dense local candidates followed by part-specific CAD matching."""

    def __init__(
        self,
        *,
        visual_dim: int = 192,
        motion_dim: int = PART_MOTION_DIM,
        parts: int = PART_COUNT,
        model_dim: int = 96,
        heads: int = 4,
        temperature: float = FCPGH_NCE_TEMPERATURE,
        history_enabled: bool = True,
    ) -> None:
        super().__init__()
        if visual_dim != 192 or motion_dim != PART_MOTION_DIM:
            raise ValueError("candidate-first dimensions are frozen to V12/CAD contracts")
        if parts != PART_COUNT:
            raise ValueError("candidate-first part set is the fixed six-point CAD set")
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.visual_dim = visual_dim
        self.motion_dim = motion_dim
        self.parts = parts
        self.model_dim = model_dim
        self.temperature = float(temperature)
        self.history_enabled = bool(history_enabled)

        self.local_projection = nn.Conv3d(visual_dim, model_dim, kernel_size=1)
        self.local_blocks = nn.ModuleList(
            [
                _LocalCausalBlock(model_dim, temporal_dilation=dilation)
                for dilation in (1, 2, 4)
            ]
        )
        self.visual_output = nn.LayerNorm(model_dim)

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
        self.part_identity = nn.Embedding(parts, model_dim)
        self.motion_output = nn.LayerNorm(model_dim)
        nn.init.trunc_normal_(self.motion_time, std=0.02)
        nn.init.trunc_normal_(self.part_identity.weight, std=0.02)
        assert_parameter_budget(
            self,
            maximum=FCPGH_MAX_TRAINABLE_PARAMETERS,
            name="FCPG-H V5 candidate-first",
        )

    def visual_candidates(self, local_patch_maps: torch.Tensor) -> torch.Tensor:
        """Return `[B,2,V,256,D]` without accepting any motion tensor."""

        if (
            local_patch_maps.ndim != 6
            or local_patch_maps.shape[1] != 16
            or local_patch_maps.shape[3:] != (self.visual_dim, 16, 16)
        ):
            raise ValueError("local_patch_maps must have shape [B,16,V,192,16,16]")
        frozen = local_patch_maps.detach()
        batch, _, views = frozen.shape[:3]
        endpoints: list[torch.Tensor] = []
        for start, stop in GROUNDING_HISTORY_SLICES:
            history = frozen[:, start:stop].permute(0, 2, 3, 1, 4, 5)
            if not self.history_enabled:
                history = history[:, :, :, -1:].expand(-1, -1, -1, 11, -1, -1)
            value = self.local_projection(
                history.reshape(batch * views, self.visual_dim, 11, 16, 16)
            )
            for block in self.local_blocks:
                value = block(value)
            current = value[:, :, -1].permute(0, 2, 3, 1).reshape(
                batch, views, 16 * 16, self.model_dim
            )
            endpoints.append(self.visual_output(current))
        return torch.stack(endpoints, dim=1)

    def part_motion_queries(self, part_motion: torch.Tensor) -> torch.Tensor:
        """Return `[B,2,M,D]` without accepting any image tensor."""

        if part_motion.ndim != 4 or part_motion.shape[1:] != (
            16,
            self.parts,
            self.motion_dim,
        ):
            raise ValueError(
                f"part_motion must have shape [B,16,{self.parts},{self.motion_dim}]"
            )
        batch = part_motion.shape[0]
        if not self.history_enabled:
            part_motion = torch.zeros_like(part_motion)
        motion = self.motion_projection(part_motion).permute(0, 2, 1, 3)
        motion = motion + self.motion_time[None, None]
        motion = motion + self.part_identity.weight[None, :, None]
        motion = motion.reshape(batch * self.parts, 16, self.model_dim)
        causal_mask = torch.triu(
            torch.ones(16, 16, device=motion.device, dtype=torch.bool), diagonal=1
        )
        encoded = self.motion_encoder(motion, mask=causal_mask)
        encoded = encoded[:, GROUNDING_ENDPOINT_INDICES]
        encoded = encoded.reshape(batch, self.parts, 2, self.model_dim).transpose(1, 2)
        return self.motion_output(encoded)

    @staticmethod
    def _suppress_collisions(probability: torch.Tensor, iterations: int = 3) -> torch.Tensor:
        """Learned-free partial assignment; no camera or reliability head."""

        result = probability
        for _ in range(iterations):
            occupancy = result.sum(dim=-2, keepdim=True)
            result = result / occupancy.clamp_min(1.0)
            result = result / result.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return result

    def localize(
        self,
        image_candidates: torch.Tensor,
        part_queries: torch.Tensor,
        *,
        view_valid: torch.Tensor,
        image_size: tuple[int, int] = (224, 224),
    ) -> CandidateFirstLocalization:
        batch, endpoints, views, pixels, dimension = image_candidates.shape
        if (endpoints, pixels, dimension) != (2, 256, self.model_dim):
            raise ValueError("image candidate bank violates the frozen dense contract")
        if part_queries.shape != (batch, 2, self.parts, self.model_dim):
            raise ValueError("part query shape mismatch")
        if view_valid.shape != (batch, views) or view_valid.dtype != torch.bool:
            raise ValueError("view_valid must be bool [B,V]")
        candidates = F.normalize(image_candidates, dim=-1)
        queries = F.normalize(part_queries, dim=-1)
        logits = torch.einsum("bevpd,bemd->bevmp", candidates, queries)
        logits = logits / self.temperature
        probability = logits.softmax(dim=-1)
        probability = self._suppress_collisions(probability)
        valid = view_valid[:, None, :, None, None]
        probability = probability * valid
        part_heatmaps = probability.reshape(batch, 2, views, self.parts, 16, 16)
        tip_heatmaps = part_heatmaps[:, :, :, list(TIP_PART_INDICES)]
        tip_px = heatmap_soft_argmax_px(tip_heatmaps, image_size=image_size)
        tip_px = tip_px * view_valid[:, None, :, None, None]
        return CandidateFirstLocalization(
            part_heatmaps=part_heatmaps,
            tip_heatmaps=tip_heatmaps,
            tip_px=tip_px,
            part_logits=logits.reshape(batch, 2, views, self.parts, 16, 16),
            image_candidates=image_candidates,
            part_queries=part_queries,
        )

    def match(
        self,
        image_candidates: torch.Tensor,
        part_queries: torch.Tensor,
        *,
        contextual_patch_maps: torch.Tensor,
        view_valid: torch.Tensor,
        image_size: tuple[int, int] = (224, 224),
    ) -> CandidateFirstOutput:
        localized = self.localize(
            image_candidates,
            part_queries,
            view_valid=view_valid,
            image_size=image_size,
        )
        batch, _, views = localized.tip_px.shape[:3]
        if contextual_patch_maps.shape == (batch, 16, views, 192, 16, 16):
            endpoint_maps = contextual_patch_maps.detach()[:, GROUNDING_ENDPOINT_INDICES]
        elif contextual_patch_maps.shape == (batch, 2, views, 192, 16, 16):
            endpoint_maps = contextual_patch_maps.detach()
        else:
            raise ValueError(
                "contextual_patch_maps must be [B,16,V,192,16,16] or "
                "endpoint-only [B,2,V,192,16,16]"
            )
        tip_px = localized.tip_px
        center_px = tip_px.mean(dim=3, keepdim=True)
        anchors_px = torch.cat((center_px, tip_px), dim=3)
        flat_maps = endpoint_maps.reshape(batch * 2 * views, 192, 16, 16)
        flat_points = anchors_px.reshape(batch * 2 * views, 3, 2)
        with torch.autocast(device_type=flat_maps.device.type, enabled=False):
            anchor_tokens = sample_feature_map_at_pixels(
                flat_maps.float(), flat_points.float(), image_size=image_size
            ).reshape(batch, 2, views, 3, 192)
        anchor_tokens = anchor_tokens * view_valid[:, None, :, None, None]
        return CandidateFirstOutput(
            part_heatmaps=localized.part_heatmaps,
            tip_heatmaps=localized.tip_heatmaps,
            tip_px=tip_px,
            anchor_tokens=anchor_tokens,
            part_logits=localized.part_logits,
            image_candidates=image_candidates,
            part_queries=part_queries,
        )

    def forward(
        self,
        local_patch_maps: torch.Tensor,
        contextual_patch_maps: torch.Tensor,
        part_motion: torch.Tensor,
        *,
        view_valid: torch.Tensor,
    ) -> CandidateFirstOutput:
        candidates = self.visual_candidates(local_patch_maps)
        queries = self.part_motion_queries(part_motion)
        return self.match(
            candidates,
            queries,
            contextual_patch_maps=contextual_patch_maps,
            view_valid=view_valid,
        )


class CandidateFirstFCPGH(nn.Module):
    """Frozen V12 local/contextual features plus the trainable V5 core."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        core: CandidateFirstTemporalGrounder | None = None,
    ) -> None:
        super().__init__()
        self.encoder = FrozenV12LocalContextEncoder(checkpoint)
        self.core = core or CandidateFirstTemporalGrounder()
        self.encoder.requires_grad_(False)

    def train(self, mode: bool = True) -> "CandidateFirstFCPGH":
        super().train(mode)
        self.encoder.train(False)
        self.core.train(mode)
        return self

    def encode_images(
        self,
        images: torch.Tensor,
        *,
        chunk_rows: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            images.ndim != 6
            or images.shape[1] != 16
            or images.shape[3:] != (3, 224, 224)
        ):
            raise ValueError("images must have shape [B,16,V,3,224,224]")
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        batch, time, views = images.shape[:3]
        rows = images.reshape(batch * time * views, 3, 224, 224)
        local: list[torch.Tensor] = []
        for start in range(0, len(rows), chunk_rows):
            chunk = rows[start : start + chunk_rows]
            local.append(self.encoder.encode_local(chunk))
        endpoint_rows = images[:, list(GROUNDING_ENDPOINT_INDICES)].reshape(
            batch * 2 * views, 3, 224, 224
        )
        contextual = [
            self.encoder(endpoint_rows[start : start + chunk_rows])
            for start in range(0, len(endpoint_rows), chunk_rows)
        ]
        local_shape = (batch, time, views, 192, 16, 16)
        contextual_shape = (batch, 2, views, 192, 16, 16)
        return (
            torch.cat(local).reshape(local_shape),
            torch.cat(contextual).reshape(contextual_shape),
        )

    def forward(
        self,
        images: torch.Tensor,
        part_motion: torch.Tensor,
        *,
        view_valid: torch.Tensor,
        chunk_rows: int = 256,
    ) -> CandidateFirstOutput:
        local, contextual = self.encode_images(images, chunk_rows=chunk_rows)
        return self.core(
            local,
            contextual,
            part_motion,
            view_valid=view_valid,
        )


def _pair_scores(
    image_candidates: torch.Tensor,
    part_queries: torch.Tensor,
    view_valid: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Return whole-pair scores `[image_batch,motion_batch]`."""

    visual = F.normalize(image_candidates, dim=-1)
    motion = F.normalize(part_queries, dim=-1)
    similarity = torch.einsum("ievpd,jemd->ijevmp", visual, motion)
    evidence = torch.logsumexp(similarity / temperature, dim=-1) - math.log(
        image_candidates.shape[-2]
    )
    valid = view_valid[:, None, None, :, None]
    evidence = evidence * valid
    denominator = valid.expand_as(evidence).sum(dim=(2, 3, 4)).clamp_min(1)
    return evidence.sum(dim=(2, 3, 4)) / denominator


def candidate_first_sensorimotor_nce(
    output: CandidateFirstOutput | CandidateFirstLocalization,
    *,
    view_valid: torch.Tensor,
    motion_cluster: torch.Tensor,
    source_index: torch.Tensor,
    task_index: torch.Tensor,
    high_motion: torch.Tensor,
    wrong_part_queries: torch.Tensor | None,
    temperature: float = FCPGH_NCE_TEMPERATURE,
) -> torch.Tensor:
    """Pair-level NCE with explicit same-image wrong-history negatives."""

    batch = output.image_candidates.shape[0]
    for name, value in {
        "motion_cluster": motion_cluster,
        "source_index": source_index,
        "task_index": task_index,
        "high_motion": high_motion,
    }.items():
        if value.shape != (batch,):
            raise ValueError(f"{name} must have shape [B]")
    if view_valid.shape != (
        batch,
        output.image_candidates.shape[2],
    ):
        raise ValueError("view_valid must have shape [B,V]")
    active = high_motion.bool() & view_valid.any(dim=1)
    factual = _pair_scores(
        output.image_candidates,
        output.part_queries,
        view_valid,
        temperature=temperature,
    )
    columns = factual
    column_valid = active
    if wrong_part_queries is not None:
        if wrong_part_queries.ndim == output.part_queries.ndim:
            wrong_part_queries = wrong_part_queries[None]
        if (
            wrong_part_queries.ndim != output.part_queries.ndim + 1
            or wrong_part_queries.shape[1:] != output.part_queries.shape
        ):
            raise ValueError(
                "wrong_part_queries must be [B,2,M,D] or [K,B,2,M,D]"
            )
        negative_sets = wrong_part_queries.shape[0]
        flat_wrong = wrong_part_queries.reshape(
            negative_sets * batch,
            *output.part_queries.shape[1:],
        )
        wrong = _pair_scores(
            output.image_candidates,
            flat_wrong,
            view_valid,
            temperature=temperature,
        )
        columns = torch.cat((factual, wrong), dim=1)
        column_valid = torch.cat((active, active.repeat(negative_sets)))

    diagonal = torch.eye(batch, device=columns.device, dtype=torch.bool)
    cluster_positive = (
        (motion_cluster[:, None] == motion_cluster[None])
        & ((source_index[:, None] != source_index[None]) | (task_index[:, None] != task_index[None]))
    )
    positive_factual = (diagonal | cluster_positive) & active[:, None] & active[None]
    positive = torch.zeros_like(columns, dtype=torch.bool)
    positive[:, :batch] = positive_factual
    masked = columns.masked_fill(~column_valid[None], -torch.inf)
    numerator = torch.logsumexp(masked.masked_fill(~positive, -torch.inf), dim=1)
    denominator = torch.logsumexp(masked, dim=1)
    row_valid = active & positive.any(dim=1) & torch.isfinite(numerator)
    if not bool(row_valid.any()):
        return columns.sum() * 0.0
    forward = (denominator[row_valid] - numerator[row_valid]).mean()

    backward_masked = factual.T.masked_fill(~active[None], -torch.inf)
    backward_positive = positive_factual.T
    backward_numerator = torch.logsumexp(
        backward_masked.masked_fill(~backward_positive, -torch.inf), dim=1
    )
    backward_denominator = torch.logsumexp(backward_masked, dim=1)
    backward_valid = active & backward_positive.any(dim=1) & torch.isfinite(backward_numerator)
    backward = (
        backward_denominator[backward_valid] - backward_numerator[backward_valid]
    ).mean()
    return 0.5 * (forward + backward)


def candidate_first_objective(
    *,
    sensorimotor_nce: torch.Tensor,
    corruption_equivariance: torch.Tensor,
) -> CandidateFirstLoss:
    if sensorimotor_nce.ndim or corruption_equivariance.ndim:
        raise ValueError("candidate-first objectives must be scalar")
    return CandidateFirstLoss(
        loss=sensorimotor_nce + corruption_equivariance,
        sensorimotor_nce=sensorimotor_nce,
        corruption_equivariance=corruption_equivariance,
    )
