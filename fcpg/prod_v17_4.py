"""Dual-view local-measurement residual-belief FCPG for Prod-V17.4.

The model is a sparse-endpoint prediction--measurement observer.  Analytic
CAD/FK provides an ordered nominal F/M location in each camera.  A recurrent
action model propagates two tip-semantic states and a bounded residual belief;
four view-tip queries then have to remeasure their own current image.  Pixel
evidence cannot cross camera boundaries.  Cross-view fusion happens only
after both per-view posteriors and coordinates have been formed.

Persistent state contains no absolute pixel coordinate.  It contains the two
semantic tip states, four distributions over a fixed residual grid, and the
previous measured joint state.  There is no confidence gate, router, fallback,
occlusion threshold, SE(3) head, optical flow, or CEM in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from fcpg.prod_v17_2 import (
    COMMAND_STEPS,
    FROZEN_V12_DIMENSION,
    IMAGE_SIZE,
    READBACK_STEPS,
    SEMANTIC_CELLS,
    SEMANTIC_GRID,
    STATE_DIMENSION,
    TIPS,
    TIP_NAMES,
    VIEWS,
    VIEW_NAMES,
    _boolean,
    _finite,
    _normalized_rgb_with_contrast,
)


MODEL_DIMENSION = 384
DECODER_LAYERS = 4
DECODER_HEADS = 8
DECODER_FFN = 1536
SAMPLES_PER_HEAD = 8
SEARCH_RADII_PX = (64.0, 32.0, 16.0, 8.0)
VISUAL_GRID = 56
LOW_VISUAL_GRID = 28
RESIDUAL_GRID_SIZE = 65
RESIDUAL_STEP_PX = 2.0
RESIDUAL_BOUND_PX = 64.0
PROCESS_KEEP = 0.95
PROCESS_UNIFORM = 0.05
PROCESS_BLUR_SIGMA_PX = 4.0
INITIAL_BELIEF_SIGMA_PX = 24.0


class V174ContractError(ValueError):
    """Raised when the V17.4 endpoint, view, or belief contract is invalid."""


@dataclass(frozen=True)
class V174FCPGMemory:
    """Only the persistent state authorized by the V17.4 observer contract."""

    tip_state: torch.Tensor
    residual_log_belief: torch.Tensor
    previous_q: torch.Tensor

    def detach(self) -> "V174FCPGMemory":
        return V174FCPGMemory(
            tip_state=self.tip_state.detach(),
            residual_log_belief=self.residual_log_belief.detach(),
            previous_q=self.previous_q.detach(),
        )


@dataclass(frozen=True)
class V174Output:
    fm_log_likelihood: torch.Tensor
    fm_log_prior: torch.Tensor
    fm_log_posterior: torch.Tensor
    fm_px: torch.Tensor
    midpoint_px: torch.Tensor
    residual_px: torch.Tensor
    measurement_entropy: torch.Tensor
    posterior_entropy: torch.Tensor
    prior_posterior_kl: torch.Tensor
    initial_reference_px: torch.Tensor
    reference_px_by_layer: torch.Tensor
    query_by_layer: torch.Tensor
    current_query: torch.Tensor
    fused_visual_56: torch.Tensor
    next_memory: V174FCPGMemory


@dataclass(frozen=True)
class V174FMLoss:
    total: torch.Tensor
    view_tip_nll: torch.Tensor
    mean_epe_px: torch.Tensor
    active_points: torch.Tensor


@dataclass(frozen=True)
class V174TemporalFeatures:
    tip_state: torch.Tensor
    next_memory: V174FCPGMemory


def _coordinate_grid(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, int(size), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def _entropy(log_probability: torch.Tensor) -> torch.Tensor:
    probability = log_probability.exp()
    return -(probability * log_probability).sum(dim=(-1, -2))


class ViewLocalDeformableDecoderLayer(nn.Module):
    """One pre-norm decoder layer with an explicit own-view sampling path.

    The implementation deliberately iterates over two cameras.  This makes a
    cross-camera pixel read impossible by construction and keeps the contract
    inspectable in tests.  F/M may self-attend within a camera, but no query
    self-attention is evaluated across cameras.
    """

    def __init__(self, dimension: int, radius_px: float) -> None:
        super().__init__()
        if dimension % DECODER_HEADS:
            raise ValueError("dimension must be divisible by eight")
        self.dimension = int(dimension)
        self.radius_px = float(radius_px)
        self.head_dim = self.dimension // DECODER_HEADS
        self.query_norm = nn.LayerNorm(self.dimension)
        self.same_view_attention = nn.MultiheadAttention(
            self.dimension,
            DECODER_HEADS,
            dropout=0.0,
            batch_first=True,
        )
        self.sample_norm = nn.LayerNorm(self.dimension)
        self.offset = nn.Linear(
            self.dimension, DECODER_HEADS * SAMPLES_PER_HEAD * 2
        )
        self.sample_weight = nn.Linear(
            self.dimension, DECODER_HEADS * SAMPLES_PER_HEAD
        )
        self.level_weight = nn.Linear(self.dimension, DECODER_HEADS * 2)
        self.high_value = nn.Conv2d(self.dimension, self.dimension, 1)
        self.low_value = nn.Conv2d(self.dimension, self.dimension, 1)
        self.output = nn.Linear(self.dimension, self.dimension)
        self.cross_norm = nn.LayerNorm(self.dimension)
        self.ffn = nn.Sequential(
            nn.Linear(self.dimension, DECODER_FFN),
            nn.GELU(),
            nn.Linear(DECODER_FFN, self.dimension),
        )
        self.reference_delta = nn.Sequential(
            nn.LayerNorm(self.dimension), nn.Linear(self.dimension, 2)
        )

        angles = torch.arange(SAMPLES_PER_HEAD, dtype=torch.float32)
        angles = angles * (2.0 * math.pi / SAMPLES_PER_HEAD)
        ring = torch.stack((angles.cos(), angles.sin()), dim=-1)
        head_scale = torch.linspace(0.35, 1.0, DECODER_HEADS)[:, None, None]
        self.register_buffer(
            "base_offsets", head_scale * ring[None], persistent=True
        )
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.reference_delta[-1].weight)
        nn.init.zeros_(self.reference_delta[-1].bias)

    @staticmethod
    def _sample(
        feature: torch.Tensor, points_px: torch.Tensor
    ) -> torch.Tensor:
        """Sample [B,D,H,W] at [B,T,Hd,K,2] pixel points."""

        batch, dimension, _height, _width = feature.shape
        tips, heads, samples = points_px.shape[1:4]
        normalized = points_px.clone()
        normalized[..., 0] = normalized[..., 0] * (2.0 / (IMAGE_SIZE - 1)) - 1.0
        normalized[..., 1] = normalized[..., 1] * (2.0 / (IMAGE_SIZE - 1)) - 1.0
        grid = normalized.reshape(batch, tips * heads * samples, 1, 2)
        value = F.grid_sample(
            feature,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return value[..., 0].transpose(1, 2).reshape(
            batch, tips, heads, samples, dimension
        )

    def forward(
        self,
        query: torch.Tensor,
        high: torch.Tensor,
        low: torch.Tensor,
        reference_px: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 4 or tuple(query.shape[1:3]) != (VIEWS, TIPS):
            raise V174ContractError("decoder query must be [B,2,2,D]")
        batch = len(query)
        updated_views: list[torch.Tensor] = []
        updated_references: list[torch.Tensor] = []
        for view in range(VIEWS):
            view_query = query[:, view]
            normalized = self.query_norm(view_query)
            attended, _ = self.same_view_attention(
                normalized, normalized, normalized, need_weights=False
            )
            view_query = view_query + attended
            source_query = self.sample_norm(view_query)
            learned = torch.tanh(self.offset(source_query)).reshape(
                batch, TIPS, DECODER_HEADS, SAMPLES_PER_HEAD, 2
            )
            offsets = (
                self.base_offsets[None, None].to(source_query)
                + 0.25 * learned
            ) * self.radius_px
            points = reference_px[:, view, :, None, None] + offsets
            high_sample = self._sample(self.high_value(high[:, view]), points)
            low_sample = self._sample(self.low_value(low[:, view]), points)
            # Each attention head consumes its own value-channel slice.  An
            # explicit stack avoids advanced-index broadcasting changing the
            # semantic [B,T,H,K,C] axis order.
            high_sample = torch.stack(
                tuple(
                    high_sample[:, :, head, :, head * self.head_dim : (head + 1) * self.head_dim]
                    for head in range(DECODER_HEADS)
                ),
                dim=2,
            )
            low_sample = torch.stack(
                tuple(
                    low_sample[:, :, head, :, head * self.head_dim : (head + 1) * self.head_dim]
                    for head in range(DECODER_HEADS)
                ),
                dim=2,
            )
            level = self.level_weight(source_query).reshape(
                batch, TIPS, DECODER_HEADS, 2
            ).softmax(dim=-1)
            sampled = (
                level[..., 0, None, None] * high_sample
                + level[..., 1, None, None] * low_sample
            )
            weight = self.sample_weight(source_query).reshape(
                batch, TIPS, DECODER_HEADS, SAMPLES_PER_HEAD
            ).softmax(dim=-1)
            aggregated = (sampled * weight[..., None]).sum(dim=3).reshape(
                batch, TIPS, self.dimension
            )
            view_query = view_query + self.output(aggregated)
            view_query = view_query + self.ffn(self.cross_norm(view_query))
            delta = torch.tanh(self.reference_delta(view_query)) * (
                self.radius_px / 4.0
            )
            updated_views.append(view_query)
            updated_references.append(
                (reference_px[:, view] + delta).clamp(0.0, IMAGE_SIZE - 1.0)
            )
        return (
            torch.stack(updated_views, dim=1),
            torch.stack(updated_references, dim=1),
        )


class V174ResidualBeliefFCPG(nn.Module):
    """Four own-view measurement queries plus an explicit Bayes observer."""

    def __init__(
        self,
        *,
        semantic_dim: int = FROZEN_V12_DIMENSION,
        dimension: int = MODEL_DIMENSION,
    ) -> None:
        super().__init__()
        if int(dimension) <= 0 or int(dimension) % DECODER_HEADS:
            raise ValueError("dimension must be positive and divisible by eight")
        self.semantic_dim = int(semantic_dim)
        self.dimension = int(dimension)

        self.rgb_stem = nn.Sequential(
            nn.Conv2d(6, 96, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(12, 96),
            nn.GELU(),
            nn.Conv2d(96, self.dimension, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(24, self.dimension),
            nn.GELU(),
            nn.Conv2d(self.dimension, self.dimension, 3, padding=1, bias=False),
        )
        self.semantic_projection = nn.Linear(self.semantic_dim, self.dimension)
        self.visual_fusion = nn.Sequential(
            nn.LayerNorm(2 * self.dimension),
            nn.Linear(2 * self.dimension, self.dimension),
            nn.GELU(),
            nn.Linear(self.dimension, self.dimension),
        )
        self.view_tip_identity = nn.Parameter(
            torch.empty(VIEWS, TIPS, self.dimension)
        )
        self.tip_identity = nn.Parameter(torch.empty(TIPS, self.dimension))
        self.nominal_projection = nn.Sequential(
            nn.Linear(2, self.dimension),
            nn.GELU(),
            nn.Linear(self.dimension, self.dimension),
        )
        # Joint values are degrees in this stack.  Fixed physical scaling keeps
        # magnitude observable; per-token LayerNorm would make a shared shift
        # exactly invisible and positive scaling nearly invisible.
        self.state_projection = nn.Linear(STATE_DIMENSION, self.dimension)
        self.command_projection = nn.Linear(STATE_DIMENSION, self.dimension)
        self.readback_projection = nn.Linear(STATE_DIMENSION, self.dimension)
        self.action_positions = nn.Parameter(
            torch.empty(2 + COMMAND_STEPS + READBACK_STEPS, self.dimension)
        )
        action_layer = nn.TransformerEncoderLayer(
            d_model=self.dimension,
            nhead=DECODER_HEADS,
            dim_feedforward=DECODER_FFN,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_memory = nn.TransformerEncoder(
            copy.deepcopy(action_layer), num_layers=2, norm=nn.LayerNorm(self.dimension)
        )
        self.action_update = nn.Linear(self.dimension, self.dimension)
        self.decoder = nn.ModuleList(
            ViewLocalDeformableDecoderLayer(self.dimension, radius)
            for radius in SEARCH_RADII_PX
        )
        self.measurement_query = nn.Sequential(
            nn.LayerNorm(self.dimension), nn.Linear(self.dimension, self.dimension)
        )
        self.candidate_projection = nn.Conv2d(
            self.dimension, self.dimension, kernel_size=1
        )
        self.residual_position = nn.Sequential(
            nn.Linear(2, self.dimension),
            nn.GELU(),
            nn.Linear(self.dimension, self.dimension),
        )
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=self.dimension,
            nhead=DECODER_HEADS,
            dim_feedforward=DECODER_FFN,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_view_semantic_fusion = nn.TransformerEncoder(
            fusion_layer, num_layers=1, norm=nn.LayerNorm(self.dimension)
        )
        self.semantic_update = nn.Linear(self.dimension, self.dimension)

        offsets = torch.arange(
            -RESIDUAL_BOUND_PX,
            RESIDUAL_BOUND_PX + RESIDUAL_STEP_PX,
            RESIDUAL_STEP_PX,
            dtype=torch.float32,
        )
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        residual_grid = torch.stack((xx, yy), dim=-1)
        self.register_buffer("residual_grid_px", residual_grid, persistent=True)
        self.register_buffer(
            "initial_log_belief",
            self._gaussian_log_belief(INITIAL_BELIEF_SIGMA_PX, residual_grid),
            persistent=True,
        )
        blur_sigma_cells = PROCESS_BLUR_SIGMA_PX / RESIDUAL_STEP_PX
        radius = int(math.ceil(3.0 * blur_sigma_cells))
        axis = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (axis / blur_sigma_cells).square())
        kernel = kernel / kernel.sum()
        self.register_buffer("process_blur_kernel", kernel, persistent=True)

        nn.init.normal_(self.view_tip_identity, std=0.02)
        nn.init.normal_(self.tip_identity, std=0.02)
        nn.init.normal_(self.action_positions, std=0.02)
        nn.init.zeros_(self.action_update.weight)
        nn.init.zeros_(self.action_update.bias)
        nn.init.zeros_(self.semantic_update.weight)
        nn.init.zeros_(self.semantic_update.bias)

    @staticmethod
    def _gaussian_log_belief(sigma_px: float, grid: torch.Tensor) -> torch.Tensor:
        logits = -0.5 * grid.square().sum(dim=-1) / float(sigma_px) ** 2
        return logits - torch.logsumexp(logits.reshape(-1), dim=0)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_memory(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        initial_q: torch.Tensor | None = None,
    ) -> V174FCPGMemory:
        if isinstance(batch, bool) or int(batch) <= 0:
            raise V174ContractError("batch must be a positive integer")
        tip = self.tip_identity.to(device=device, dtype=dtype)[None].expand(
            int(batch), -1, -1
        ).clone()
        belief = self.initial_log_belief.to(device=device, dtype=dtype)
        belief = belief[None, None, None].expand(
            int(batch), VIEWS, TIPS, -1, -1
        ).clone()
        previous = (
            torch.zeros(int(batch), STATE_DIMENSION, device=device, dtype=dtype)
            if initial_q is None
            else torch.as_tensor(initial_q, device=device, dtype=dtype).clone()
        )
        if previous.shape != (int(batch), STATE_DIMENSION):
            raise V174ContractError("initial_q must be [B,6]")
        return V174FCPGMemory(tip, belief, previous)

    def _validate_memory(
        self,
        memory: V174FCPGMemory,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> V174FCPGMemory:
        tip = torch.as_tensor(memory.tip_state, device=device, dtype=dtype)
        belief = torch.as_tensor(
            memory.residual_log_belief, device=device, dtype=dtype
        )
        previous = torch.as_tensor(memory.previous_q, device=device, dtype=dtype)
        if tip.shape != (batch, TIPS, self.dimension):
            raise V174ContractError(
                f"memory.tip_state must be [B,2,{self.dimension}]"
            )
        if belief.shape != (
            batch, VIEWS, TIPS, RESIDUAL_GRID_SIZE, RESIDUAL_GRID_SIZE
        ):
            raise V174ContractError("memory residual belief must be [B,2,2,65,65]")
        if previous.shape != (batch, STATE_DIMENSION):
            raise V174ContractError("memory.previous_q must be [B,6]")
        _finite("memory.tip_state", tip)
        _finite("memory.residual_log_belief", belief)
        _finite("memory.previous_q", previous)
        normalized = belief - torch.logsumexp(belief, dim=(-1, -2), keepdim=True)
        if not torch.allclose(belief, normalized, atol=2.0e-3, rtol=0.0):
            raise V174ContractError("memory residual belief is not normalized")
        return V174FCPGMemory(tip, normalized, previous)

    def encode_visual_grid(
        self, rgb: torch.Tensor, frozen_v12_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.as_tensor(rgb)
        tokens = torch.as_tensor(
            frozen_v12_tokens, device=image.device, dtype=image.dtype
        ).detach()
        batch = len(image)
        if image.shape != (batch, VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE):
            raise V174ContractError("rgb must be [B,2,3,224,224]")
        if tokens.shape != (batch, VIEWS, SEMANTIC_CELLS, self.semantic_dim):
            raise V174ContractError("frozen_v12_tokens must be [B,2,256,D]")
        _finite("rgb", image)
        _finite("frozen_v12_tokens", tokens)
        flat = image.reshape(batch * VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE)
        rgb_feature = self.rgb_stem(_normalized_rgb_with_contrast(flat))
        semantic = self.semantic_projection(tokens).reshape(
            batch * VIEWS, SEMANTIC_GRID, SEMANTIC_GRID, self.dimension
        ).permute(0, 3, 1, 2)
        semantic = F.interpolate(
            semantic,
            size=(VISUAL_GRID, VISUAL_GRID),
            mode="bilinear",
            align_corners=False,
        )
        fused = torch.cat((rgb_feature, semantic), dim=1).permute(0, 2, 3, 1)
        fused = self.visual_fusion(fused).permute(0, 3, 1, 2)
        high = fused.reshape(
            batch, VIEWS, self.dimension, VISUAL_GRID, VISUAL_GRID
        )
        low = F.avg_pool2d(fused, 2).reshape(
            batch, VIEWS, self.dimension, LOW_VISUAL_GRID, LOW_VISUAL_GRID
        )
        return high, low

    def _propagate_tip_state(
        self,
        tip_state: torch.Tensor,
        current_q: torch.Tensor,
        commands: torch.Tensor,
        readback: torch.Tensor,
        previous_q: torch.Tensor,
        transition_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch = len(tip_state)
        state_tokens = torch.stack(
            (
                self.state_projection(current_q / 180.0),
                self.state_projection((current_q - previous_q) / 180.0),
            ),
            dim=1,
        )
        command_tokens = self.command_projection(commands / 180.0)
        readback_tokens = self.readback_projection(readback / 180.0)
        condition = torch.cat((state_tokens, command_tokens, readback_tokens), dim=1)
        condition = condition + self.action_positions[None].to(condition)
        sequence = torch.cat((tip_state, condition), dim=1)
        propagated = self.action_memory(sequence)[:, :TIPS]
        update = self.action_update(propagated)
        active = transition_valid[:, None, None]
        return torch.where(active, tip_state + update, tip_state)

    @staticmethod
    def _validate_transition_clock(
        *,
        current_q: torch.Tensor,
        measured_joint_path: torch.Tensor,
        previous_q: torch.Tensor,
        transition_valid: torch.Tensor,
    ) -> None:
        """Verify the endpoint clock at inference without synchronizing training.

        The archive adapters validate this invariant once on CPU.  Runtime and
        frozen evaluation validate it again here: an active transition starts
        at the memorized endpoint and ends at ``current_q``; an inactive
        transition may not silently advance the state clock.
        """

        start_error = (measured_joint_path[:, 0] - previous_q).abs().amax(dim=-1)
        end_error = (measured_joint_path[:, -1] - current_q).abs().amax(dim=-1)
        hold_error = (current_q - previous_q).abs().amax(dim=-1)
        invalid = torch.where(
            transition_valid,
            (start_error > 1.0e-4) | (end_error > 1.0e-4),
            hold_error > 1.0e-4,
        )
        if bool(invalid.any()):
            raise V174ContractError(
                "transition clock disagrees with previous_q/path/current_q"
            )

    def temporal_pretrain_query(
        self,
        current_tokens: torch.Tensor,
        current_q: torch.Tensor,
        completed_commands: torch.Tensor,
        measured_joint_path: torch.Tensor,
        transition_valid: torch.Tensor,
        view_valid: torch.Tensor,
        previous_memory: V174FCPGMemory | None = None,
    ) -> V174TemporalFeatures:
        tokens = torch.as_tensor(current_tokens)
        batch = len(tokens)
        if tokens.shape != (batch, VIEWS, SEMANTIC_CELLS, self.semantic_dim):
            raise V174ContractError("current temporal tokens must be [B,2,256,D]")
        dtype, device = tokens.dtype, tokens.device
        q = torch.as_tensor(current_q, device=device, dtype=dtype)
        commands = torch.as_tensor(completed_commands, device=device, dtype=dtype)
        path = torch.as_tensor(measured_joint_path, device=device, dtype=dtype)
        transition = _boolean(
            "transition_valid", transition_valid, shape=(batch,), device=device
        )
        views = _boolean("view_valid", view_valid, shape=(batch, VIEWS), device=device)
        if q.shape != (batch, STATE_DIMENSION):
            raise V174ContractError("current_q must be [B,6]")
        if commands.shape != (batch, COMMAND_STEPS, STATE_DIMENSION):
            raise V174ContractError("completed_commands must be [B,5,6]")
        if path.shape != (batch, READBACK_STEPS, STATE_DIMENSION):
            raise V174ContractError("measured_joint_path must be [B,6,6]")
        initial_q = torch.where(transition[:, None], path[:, 0], q)
        memory = (
            self.initial_memory(
                batch, device=device, dtype=dtype, initial_q=initial_q
            )
            if previous_memory is None
            else self._validate_memory(
                previous_memory, batch=batch, device=device, dtype=dtype
            )
        )
        if not self.training:
            self._validate_transition_clock(
                current_q=q,
                measured_joint_path=path,
                previous_q=memory.previous_q,
                transition_valid=transition,
            )
        propagated = self._propagate_tip_state(
            memory.tip_state, q, commands, path, memory.previous_q, transition
        )
        pooled = self.semantic_projection(tokens.detach()).mean(dim=2)
        valid_weight = views.to(dtype=dtype)[..., None]
        pooled = (pooled * valid_weight).sum(dim=1) / valid_weight.sum(
            dim=1
        ).clamp_min(1.0)
        observed = views.any(dim=1)[:, None, None]
        tip = torch.where(observed, propagated + pooled[:, None], propagated)
        next_previous_q = torch.where(
            transition[:, None], q, memory.previous_q
        )
        next_memory = V174FCPGMemory(
            tip, memory.residual_log_belief, next_previous_q
        )
        return V174TemporalFeatures(tip_state=tip, next_memory=next_memory)

    def _process_prior(self, log_belief: torch.Tensor) -> torch.Tensor:
        probability = log_belief.float().exp()
        batch_shape = probability.shape[:-2]
        flat = probability.reshape(-1, 1, RESIDUAL_GRID_SIZE, RESIDUAL_GRID_SIZE)
        kernel = self.process_blur_kernel.float()
        radius = len(kernel) // 2
        flat = F.conv2d(
            F.pad(flat, (radius, radius, 0, 0), mode="replicate"),
            kernel[None, None, None, :],
        )
        flat = F.conv2d(
            F.pad(flat, (0, 0, radius, radius), mode="replicate"),
            kernel[None, None, :, None],
        )
        uniform = 1.0 / float(RESIDUAL_GRID_SIZE**2)
        probability = PROCESS_KEEP * flat + PROCESS_UNIFORM * uniform
        probability = probability.reshape(
            *batch_shape, RESIDUAL_GRID_SIZE, RESIDUAL_GRID_SIZE
        )
        log_prior = probability.clamp_min(1.0e-12).log()
        log_prior = log_prior - torch.logsumexp(
            log_prior, dim=(-1, -2), keepdim=True
        )
        return log_prior.to(log_belief.dtype)

    def _belief_expectation(self, log_belief: torch.Tensor) -> torch.Tensor:
        probability = log_belief.exp()
        return (probability[..., None] * self.residual_grid_px.to(probability)).sum(
            dim=(-3, -2)
        )

    def _candidate_logits(
        self,
        high: torch.Tensor,
        query: torch.Tensor,
        nominal_fm_px: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch = len(high)
        residual_position = self.residual_grid_px.to(high) / RESIDUAL_BOUND_PX
        position_feature = self.residual_position(
            residual_position.reshape(-1, 2)
        ).reshape(RESIDUAL_GRID_SIZE, RESIDUAL_GRID_SIZE, self.dimension)
        projected = self.candidate_projection(
            high.reshape(batch * VIEWS, self.dimension, VISUAL_GRID, VISUAL_GRID)
        ).reshape(batch, VIEWS, self.dimension, VISUAL_GRID, VISUAL_GRID)
        measurement_query = F.normalize(self.measurement_query(query), dim=-1)
        result: list[torch.Tensor] = []
        for view in range(VIEWS):
            tips: list[torch.Tensor] = []
            for tip in range(TIPS):
                points = nominal_fm_px[:, view, tip, None, None] + self.residual_grid_px.to(high)
                normalized = points.clone()
                normalized[..., 0] = normalized[..., 0] * (2.0 / (IMAGE_SIZE - 1)) - 1.0
                normalized[..., 1] = normalized[..., 1] * (2.0 / (IMAGE_SIZE - 1)) - 1.0
                sampled = F.grid_sample(
                    projected[:, view],
                    normalized,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                ).permute(0, 2, 3, 1)
                candidate = F.normalize(sampled + position_feature[None], dim=-1)
                logits = torch.einsum(
                    "bhwd,bd->bhw", candidate, measurement_query[:, view, tip]
                ) * math.sqrt(self.dimension)
                inside = (
                    (points[..., 0] >= 0.0)
                    & (points[..., 0] <= IMAGE_SIZE - 1.0)
                    & (points[..., 1] >= 0.0)
                    & (points[..., 1] <= IMAGE_SIZE - 1.0)
                )
                logits = logits.masked_fill(~inside, -1.0e4)
                logits = torch.where(
                    view_valid[:, view, None, None], logits, torch.zeros_like(logits)
                )
                tips.append(logits)
            result.append(torch.stack(tips, dim=1))
        return torch.stack(result, dim=1)

    def _next_tip_state(
        self,
        propagated: torch.Tensor,
        query: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch = len(query)
        sequences: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for tip in range(TIPS):
            sequences.append(
                torch.stack(
                    (propagated[:, tip], query[:, 0, tip], query[:, 1, tip]), dim=1
                )
            )
            masks.append(
                torch.cat(
                    (
                        torch.zeros(batch, 1, dtype=torch.bool, device=query.device),
                        ~view_valid,
                    ),
                    dim=1,
                )
            )
        sequence = torch.cat(sequences, dim=0)
        mask = torch.cat(masks, dim=0)
        fused = self.cross_view_semantic_fusion(
            sequence, src_key_padding_mask=mask
        )[:, 0]
        fused = fused.reshape(TIPS, batch, self.dimension).transpose(0, 1)
        update = self.semantic_update(fused)
        observed = view_valid.any(dim=1)[:, None, None]
        return torch.where(observed, propagated + update, propagated)

    def forward(
        self,
        rgb: torch.Tensor,
        frozen_v12_tokens: torch.Tensor,
        nominal_fm_px: torch.Tensor,
        current_q: torch.Tensor,
        completed_commands: torch.Tensor,
        measured_joint_path: torch.Tensor,
        transition_valid: torch.Tensor,
        view_valid: torch.Tensor,
        previous_memory: V174FCPGMemory | None = None,
        *,
        _encoded_visual: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> V174Output:
        image = torch.as_tensor(rgb)
        batch, device, dtype = len(image), image.device, image.dtype
        nominal = torch.as_tensor(nominal_fm_px, device=device, dtype=dtype)
        q = torch.as_tensor(current_q, device=device, dtype=dtype)
        commands = torch.as_tensor(completed_commands, device=device, dtype=dtype)
        readback = torch.as_tensor(measured_joint_path, device=device, dtype=dtype)
        transition = _boolean(
            "transition_valid", transition_valid, shape=(batch,), device=device
        )
        views = _boolean("view_valid", view_valid, shape=(batch, VIEWS), device=device)
        if nominal.shape != (batch, VIEWS, TIPS, 2):
            raise V174ContractError("nominal_fm_px must be [B,2,2,2]")
        if q.shape != (batch, STATE_DIMENSION):
            raise V174ContractError("current_q must be [B,6]")
        if commands.shape != (batch, COMMAND_STEPS, STATE_DIMENSION):
            raise V174ContractError("completed_commands must be [B,5,6]")
        if readback.shape != (batch, READBACK_STEPS, STATE_DIMENSION):
            raise V174ContractError("measured_joint_path must be [B,6,6]")
        for name, value in (
            ("nominal_fm_px", nominal),
            ("current_q", q),
            ("completed_commands", commands),
            ("measured_joint_path", readback),
        ):
            _finite(name, value)
        initial_q = torch.where(transition[:, None], readback[:, 0], q)
        memory = (
            self.initial_memory(
                batch, device=device, dtype=dtype, initial_q=initial_q
            )
            if previous_memory is None
            else self._validate_memory(
                previous_memory, batch=batch, device=device, dtype=dtype
            )
        )
        if not self.training:
            self._validate_transition_clock(
                current_q=q,
                measured_joint_path=readback,
                previous_q=memory.previous_q,
                transition_valid=transition,
            )
        propagated = self._propagate_tip_state(
            memory.tip_state, q, commands, readback, memory.previous_q, transition
        )
        log_prior = self._process_prior(memory.residual_log_belief)
        prior_residual = self._belief_expectation(log_prior)
        reference = (nominal + prior_residual).clamp(0.0, IMAGE_SIZE - 1.0)
        initial_reference = reference
        if _encoded_visual is None:
            high, low = self.encode_visual_grid(image, frozen_v12_tokens)
        else:
            high, low = _encoded_visual
            expected_high = (
                batch, VIEWS, self.dimension, VISUAL_GRID, VISUAL_GRID
            )
            expected_low = (
                batch, VIEWS, self.dimension, LOW_VISUAL_GRID, LOW_VISUAL_GRID
            )
            if high.shape != expected_high or low.shape != expected_low:
                raise V174ContractError("precomputed visual-grid shape drifted")
            if high.device != device or low.device != device:
                raise V174ContractError("precomputed visual grid is on the wrong device")
        geometry = self.nominal_projection(
            nominal / float(IMAGE_SIZE - 1) * 2.0 - 1.0
        )
        query = (
            propagated[:, None]
            + self.view_tip_identity[None].to(propagated)
            + geometry
        )
        references = []
        queries = []
        for layer in self.decoder:
            query, reference = layer(query, high, low, reference)
            references.append(reference)
            queries.append(query)
        likelihood_logits = self._candidate_logits(high, query, nominal, views)
        log_likelihood = likelihood_logits - torch.logsumexp(
            likelihood_logits, dim=(-1, -2), keepdim=True
        )
        combined = log_prior + likelihood_logits
        combined = combined - torch.logsumexp(
            combined, dim=(-1, -2), keepdim=True
        )
        # This explicit selection is stronger than relying on a second
        # normalization of prior + zero: an invalid view writes no measurement
        # evidence, so its posterior is bitwise equal to the predicted process
        # prior.  The process model itself still diffuses uncertainty.
        log_posterior = torch.where(
            views[:, :, None, None, None], combined, log_prior
        )
        residual = self._belief_expectation(log_posterior)
        fm = (nominal + residual).clamp(0.0, IMAGE_SIZE - 1.0)
        next_tip = self._next_tip_state(propagated, query, views)
        next_previous_q = torch.where(
            transition[:, None], q, memory.previous_q
        )
        next_memory = V174FCPGMemory(
            next_tip, log_posterior, next_previous_q
        )
        measurement_entropy = _entropy(log_likelihood)
        posterior_entropy = _entropy(log_posterior)
        prior_posterior_kl = (
            log_posterior.exp() * (log_posterior - log_prior)
        ).sum(dim=(-1, -2))
        for name, value in (
            ("fm_log_posterior", log_posterior),
            ("fm_px", fm),
            ("residual_px", residual),
            ("tip_state", next_tip),
        ):
            _finite(name, value)
        if bool((residual.abs() > RESIDUAL_BOUND_PX + 1.0e-4).any()):
            raise V174ContractError("posterior expectation leaves residual bounds")
        return V174Output(
            fm_log_likelihood=log_likelihood,
            fm_log_prior=log_prior,
            fm_log_posterior=log_posterior,
            fm_px=fm,
            midpoint_px=fm.mean(dim=2),
            residual_px=residual,
            measurement_entropy=measurement_entropy,
            posterior_entropy=posterior_entropy,
            prior_posterior_kl=prior_posterior_kl,
            initial_reference_px=initial_reference,
            reference_px_by_layer=torch.stack(references, dim=1),
            query_by_layer=torch.stack(queries, dim=1),
            current_query=query,
            fused_visual_56=high,
            next_memory=next_memory,
        )


def _bilinear_log_mass_nll(
    log_probability: torch.Tensor, coordinate: torch.Tensor
) -> torch.Tensor:
    """Bilinear four-cell categorical NLL for continuous grid coordinates."""

    lower = coordinate.floor()
    fraction = coordinate - lower
    x0 = lower[..., 0].long().clamp(0, RESIDUAL_GRID_SIZE - 1)
    y0 = lower[..., 1].long().clamp(0, RESIDUAL_GRID_SIZE - 1)
    x1 = (x0 + 1).clamp(0, RESIDUAL_GRID_SIZE - 1)
    y1 = (y0 + 1).clamp(0, RESIDUAL_GRID_SIZE - 1)
    flat = log_probability.reshape(*log_probability.shape[:-2], -1)

    def gather(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return flat.gather(-1, (y * RESIDUAL_GRID_SIZE + x)[..., None])[..., 0]

    wx, wy = fraction[..., 0], fraction[..., 1]
    return -(
        (1.0 - wx) * (1.0 - wy) * gather(x0, y0)
        + wx * (1.0 - wy) * gather(x1, y0)
        + (1.0 - wx) * wy * gather(x0, y1)
        + wx * wy * gather(x1, y1)
    )


def v174_ordered_fm_categorical_nll(
    output: V174Output,
    target_fm_px: torch.Tensor,
    target_valid: torch.Tensor,
    nominal_fm_px: torch.Tensor,
    *,
    macro_counts: torch.Tensor | None = None,
) -> V174FMLoss:
    """Strict 0.5-view/0.5-tip reduction; sample counts cannot change it."""

    target = torch.as_tensor(
        target_fm_px, device=output.fm_px.device, dtype=output.fm_px.dtype
    )
    nominal = torch.as_tensor(
        nominal_fm_px, device=output.fm_px.device, dtype=output.fm_px.dtype
    )
    valid = torch.as_tensor(target_valid, device=output.fm_px.device)
    if target.shape != output.fm_px.shape or nominal.shape != target.shape:
        raise V174ContractError("target and nominal F/M must match output [B,2,2,2]")
    if valid.shape != target.shape[:-1] or valid.dtype != torch.bool:
        raise V174ContractError("target_valid must be boolean [B,2,2]")
    _finite("target_fm_px", target)
    residual = target - nominal
    in_grid = (residual.abs() <= RESIDUAL_BOUND_PX + 1.0e-5).all(dim=-1)
    outside = valid & ~in_grid
    if bool(outside.any()):
        raise V174ContractError(
            "valid ordered F/M target leaves the fixed +/-64 px residual grid"
        )
    active = valid
    coordinate = residual / RESIDUAL_STEP_PX + (RESIDUAL_GRID_SIZE - 1) / 2.0
    point_nll = _bilinear_log_mass_nll(output.fm_log_posterior, coordinate)
    local_counts = active.sum(dim=0)
    counts = local_counts if macro_counts is None else torch.as_tensor(
        macro_counts, device=active.device, dtype=local_counts.dtype
    )
    if counts.shape != (VIEWS, TIPS) or bool((counts <= 0).any()):
        missing = [
            f"{VIEW_NAMES[view]}-{TIP_NAMES[tip]}"
            for view in range(VIEWS)
            for tip in range(TIPS)
            if int(counts[view, tip].detach().cpu()) <= 0
        ]
        raise V174ContractError(
            "macro loss lacks active labels for " + ", ".join(missing)
        )
    view_tip_nll = (point_nll * active.to(point_nll.dtype)).sum(dim=0) / counts
    total = view_tip_nll.mean()
    epe = torch.linalg.vector_norm(output.fm_px - target, dim=-1)
    mean_epe = (epe * active.to(epe.dtype)).sum() / counts.sum()
    return V174FMLoss(
        total=total,
        view_tip_nll=view_tip_nll,
        mean_epe_px=mean_epe,
        active_points=active.sum(),
    )


def runtime_parameter_groups(model: V174ResidualBeliefFCPG) -> dict[str, tuple[nn.Parameter, ...]]:
    """Named groups used to enforce source-specific optimizer ownership."""

    visual_modules: Iterable[nn.Module] = (
        model.rgb_stem,
        model.semantic_projection,
        model.visual_fusion,
    )
    temporal_modules: Iterable[nn.Module] = (
        model.semantic_projection,
        model.state_projection,
        model.command_projection,
        model.readback_projection,
        model.action_memory,
        model.action_update,
    )

    def unique(modules: Iterable[nn.Module]) -> tuple[nn.Parameter, ...]:
        values: dict[int, nn.Parameter] = {}
        for module in modules:
            for parameter in module.parameters():
                values[id(parameter)] = parameter
        return tuple(values.values())

    return {
        "visual": unique(visual_modules),
        "temporal": unique(temporal_modules),
        "all": tuple(model.parameters()),
    }


__all__ = [
    "DECODER_HEADS",
    "DECODER_LAYERS",
    "MODEL_DIMENSION",
    "PROCESS_BLUR_SIGMA_PX",
    "PROCESS_KEEP",
    "PROCESS_UNIFORM",
    "RESIDUAL_BOUND_PX",
    "RESIDUAL_GRID_SIZE",
    "RESIDUAL_STEP_PX",
    "SEARCH_RADII_PX",
    "V174ContractError",
    "V174FCPGMemory",
    "V174FMLoss",
    "V174Output",
    "V174ResidualBeliefFCPG",
    "V174TemporalFeatures",
    "ViewLocalDeformableDecoderLayer",
    "runtime_parameter_groups",
    "v174_ordered_fm_categorical_nll",
]
