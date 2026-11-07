"""Query-initialized FCPG-H measurement and its single LeWM transition.

This is the maintained observer architecture.  It deliberately keeps the
historical query-free FCPG-H implementations intact so their No-Go results
remain reproducible, while removing their unobservable motion-query-to-pixel
path from the new model:

* the visual measurement reads only frozen V12.4 patches, a supplied query,
  causal visual history, validity, and the already-resolved displacement clock;
* the one LeWM transition reads posterior part tokens and completed physical
  control, but never produces a pixel;
* a stopped-gradient LeWM prior may only re-rank the current visual
  measurement, after which the posterior is sampled from the current image and
  written into the next transition.

The fixed/moving fingertip order is stable throughout this module.  Anchor
order is centre/fixed/moving; the centre pixel is always the strict midpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from fcpg.modules import sample_feature_map_at_pixels


TOKEN_DIM = 192
PATCH_GRID = 16
TIP_COUNT = 2
ANCHOR_COUNT = 3
MAX_EVENT_FRAMES = 16
MEASUREMENT_TEMPERATURE = 0.10
FEEDBACK_BETA = 0.25
FEEDBACK_TEMPERATURE = 0.10
LEWM_RESIDUAL_CAP = 0.25
LEWM_SIGREG_WEIGHT = 0.02
FCPGH_PARAMETER_LIMIT = 1_300_000
LEWM_PARAMETER_LIMIT = 1_000_000
METHOD_REVISION = "fcpg-h-query-conditioned-observer-v1"
CHECKPOINT_FORMAT = "so101-fcpg-h-query-conditioned-observer-checkpoint-v1"
LEWM_CHECKPOINT_FORMAT = "so101-query-conditioned-part-token-lewm-checkpoint-v1"


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def assert_parameter_limit(module: nn.Module, *, maximum: int, name: str) -> int:
    count = trainable_parameter_count(module)
    if count > maximum:
        raise RuntimeError(
            f"{name} has {count:,} trainable parameters, above {maximum:,}"
        )
    return count


@dataclass(frozen=True)
class QueryConditionedMeasurementBatch:
    """Frozen visual rows for one variable-duration displacement event."""

    patch_maps: torch.Tensor
    query_px: torch.Tensor
    time_valid: torch.Tensor
    displacement_progress: torch.Tensor
    view_valid: torch.Tensor

    def validate(self) -> None:
        maps = self.patch_maps
        if maps.ndim != 6 or maps.shape[3:] != (
            TOKEN_DIM,
            PATCH_GRID,
            PATCH_GRID,
        ):
            raise ValueError(
                "patch_maps must have shape [B,T,V,192,16,16]"
            )
        batch, time, views = maps.shape[:3]
        if not 2 <= time <= MAX_EVENT_FRAMES:
            raise ValueError("query measurement requires 2..16 causal frames")
        if self.query_px.shape != (batch, views, TIP_COUNT, 2):
            raise ValueError("query_px must have shape [B,V,2,2]")
        if self.time_valid.shape != (batch, time) or self.time_valid.dtype != torch.bool:
            raise ValueError("time_valid must be bool [B,T]")
        if self.displacement_progress.shape != (batch, time):
            raise ValueError("displacement_progress must have shape [B,T]")
        if self.view_valid.shape != (batch, views) or self.view_valid.dtype != torch.bool:
            raise ValueError("view_valid must be bool [B,V]")
        floating = (maps, self.query_px, self.displacement_progress)
        if not all(value.is_floating_point() for value in floating):
            raise TypeError("patch/query/progress inputs must be floating point")
        if not all(bool(torch.isfinite(value).all()) for value in floating):
            raise ValueError("query measurement inputs must be finite")
        if not bool(self.time_valid[:, 0].all()):
            raise ValueError("the query frame must be valid")
        # Padding is suffix-only.  This makes the final valid frame causal and
        # prevents a hidden future row from being selected as the endpoint.
        if bool((self.time_valid[:, 1:] & ~self.time_valid[:, :-1]).any()):
            raise ValueError("time_valid must be a contiguous valid prefix")
        progress = self.displacement_progress
        valid_progress = progress[self.time_valid]
        if bool(((valid_progress < 0.0) | (valid_progress > 1.0)).any()):
            raise ValueError("valid displacement progress must lie in [0,1]")
        delta = progress[:, 1:] - progress[:, :-1]
        pair_valid = self.time_valid[:, 1:] & self.time_valid[:, :-1]
        if bool((delta[pair_valid] < 0.0).any()):
            raise ValueError("displacement progress must be non-decreasing")


@dataclass(frozen=True)
class QueryConditionedMeasurement:
    measurement_logits: torch.Tensor
    heatmaps: torch.Tensor
    tip_px: torch.Tensor
    anchor_px: torch.Tensor
    part_tokens: torch.Tensor
    anchor_tokens: torch.Tensor
    visual_slots: torch.Tensor
    endpoint_index: torch.Tensor


@dataclass(frozen=True)
class PartTokenPrior:
    part_tokens: torch.Tensor
    residual: torch.Tensor
    dynamics_latent: torch.Tensor


@dataclass(frozen=True)
class ObserverPosterior:
    posterior_logits: torch.Tensor
    heatmaps: torch.Tensor
    tip_px: torch.Tensor
    anchor_px: torch.Tensor
    part_tokens: torch.Tensor
    anchor_tokens: torch.Tensor
    prior_similarity: torch.Tensor


@dataclass(frozen=True)
class ObserverRollout:
    priors: torch.Tensor
    posterior_part_tokens: torch.Tensor
    posterior_anchor_tokens: torch.Tensor
    posterior_tip_px: torch.Tensor
    posterior_logits: torch.Tensor
    dynamics_latent: torch.Tensor


class _TemporalCrossAttention(nn.Module):
    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dimension)
        self.memory_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, 2 * dimension),
            nn.GELU(),
            nn.Linear(2 * dimension, dimension),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_padding: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=memory_padding,
            need_weights=False,
        )
        query = query + attended
        return query + self.ffn(query)


def _progress_encoding(progress: torch.Tensor, dimension: int) -> torch.Tensor:
    """Fixed displacement-clock encoding; it has no learned pixel mapping."""

    if dimension % 2:
        raise ValueError("progress encoding dimension must be even")
    half = dimension // 2
    frequency = torch.exp(
        torch.linspace(
            0.0,
            torch.log(torch.tensor(128.0, device=progress.device)),
            half,
            device=progress.device,
            dtype=progress.dtype,
        )
    )
    phase = progress[..., None] * frequency * torch.pi
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


def _endpoint_rows(time_valid: torch.Tensor) -> torch.Tensor:
    return time_valid.long().sum(dim=1) - 1


def _patch_centres(
    flat_indices: torch.Tensor,
    *,
    image_size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    width, height = image_size
    y = torch.div(flat_indices, PATCH_GRID, rounding_mode="floor")
    x = flat_indices.remainder(PATCH_GRID)
    px = (x.to(torch.float32) + 0.5) * (float(width) / PATCH_GRID) - 0.5
    py = (y.to(torch.float32) + 0.5) * (float(height) / PATCH_GRID) - 0.5
    return torch.stack((px, py), dim=-1)


def decode_unique_tip_pixels(
    logits: torch.Tensor,
    *,
    image_size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Exact parameter-free two-tip assignment with no shared patch."""

    if logits.ndim != 5 or logits.shape[2:] != (
        TIP_COUNT,
        PATCH_GRID,
        PATCH_GRID,
    ):
        raise ValueError("tip logits must have shape [B,V,2,16,16]")
    batch, views = logits.shape[:2]
    flat = logits.float().flatten(-2)
    joint = flat[:, :, 0, :, None] + flat[:, :, 1, None, :]
    diagonal = torch.eye(
        PATCH_GRID * PATCH_GRID,
        device=logits.device,
        dtype=torch.bool,
    )
    joint = joint.masked_fill(diagonal, -torch.inf)
    selected = joint.reshape(batch, views, -1).argmax(dim=-1)
    fixed = torch.div(
        selected,
        PATCH_GRID * PATCH_GRID,
        rounding_mode="floor",
    )
    moving = selected.remainder(PATCH_GRID * PATCH_GRID)
    indices = torch.stack((fixed, moving), dim=-1)
    return _patch_centres(indices, image_size=image_size).to(logits.dtype)


def _current_maps(
    patch_maps: torch.Tensor,
    endpoint_index: torch.Tensor,
) -> torch.Tensor:
    rows = torch.arange(len(patch_maps), device=patch_maps.device)
    return patch_maps[rows, endpoint_index]


def _sample_current_anchors(
    current_patch_maps: torch.Tensor,
    tip_px: torch.Tensor,
    *,
    image_size: tuple[int, int] = (224, 224),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, views = current_patch_maps.shape[:2]
    center = tip_px.mean(dim=2, keepdim=True)
    anchor_px = torch.cat((center, tip_px), dim=2)
    with torch.autocast(
        device_type=current_patch_maps.device.type,
        enabled=False,
    ):
        anchor_tokens = sample_feature_map_at_pixels(
            current_patch_maps.reshape(batch * views, TOKEN_DIM, PATCH_GRID, PATCH_GRID).float(),
            anchor_px.reshape(batch * views, ANCHOR_COUNT, 2).float(),
            image_size=image_size,
        ).reshape(batch, views, ANCHOR_COUNT, TOKEN_DIM)
    return anchor_px, anchor_tokens[:, :, 1:], anchor_tokens


class QueryConditionedFCPGH(nn.Module):
    """Lightweight causal visual measurement initialized by two real queries."""

    def __init__(
        self,
        *,
        dimension: int = TOKEN_DIM,
        heads: int = 6,
        temperature: float = MEASUREMENT_TEMPERATURE,
    ) -> None:
        super().__init__()
        if dimension != TOKEN_DIM:
            raise ValueError("the maintained observer uses the V12.4 192-D token space")
        if dimension % heads:
            raise ValueError("measurement dimension must be divisible by heads")
        if temperature != MEASUREMENT_TEMPERATURE:
            raise ValueError("measurement temperature is frozen at 0.10")
        self.dimension = dimension
        self.temperature = float(temperature)
        self.patch_projection = nn.Linear(TOKEN_DIM, dimension)
        self.query_projection = nn.Linear(TOKEN_DIM, dimension)
        self.tip_identity = nn.Embedding(TIP_COUNT, dimension)
        # First aggregate causal visual history into every current spatial
        # cell.  A separate part-slot query then reads those aggregated
        # current cells; this is the architecture contract, not a second
        # kinematic world model.
        self.temporal_blocks = nn.ModuleList(
            [_TemporalCrossAttention(dimension, heads) for _ in range(2)]
        )
        self.slot_attention = _TemporalCrossAttention(dimension, heads)
        self.spatial_output = nn.LayerNorm(dimension)
        self.slot_output = nn.LayerNorm(dimension)
        assert_parameter_limit(
            self,
            maximum=FCPGH_PARAMETER_LIMIT,
            name="query-conditioned FCPG-H",
        )

    def _measure_cached(
        self,
        *,
        pooled_history: torch.Tensor,
        current_maps: torch.Tensor,
        initial_query_tokens: torch.Tensor,
        time_valid: torch.Tensor,
        displacement_progress: torch.Tensor,
        view_valid: torch.Tensor,
        endpoint: torch.Tensor,
        image_size: tuple[int, int],
    ) -> QueryConditionedMeasurement:
        if pooled_history.ndim != 6 or pooled_history.shape[3:] != (
            TOKEN_DIM,
            8,
            8,
        ):
            raise ValueError("pooled_history must be [B,T,V,192,8,8]")
        batch_size, time, views = pooled_history.shape[:3]
        if current_maps.shape != (
            batch_size,
            views,
            TOKEN_DIM,
            PATCH_GRID,
            PATCH_GRID,
        ):
            raise ValueError("current_maps must be [B,V,192,16,16]")
        if initial_query_tokens.shape != (
            batch_size,
            views,
            TIP_COUNT,
            TOKEN_DIM,
        ):
            raise ValueError("initial_query_tokens must be [B,V,2,192]")
        if time_valid.shape != (batch_size, time) or time_valid.dtype != torch.bool:
            raise ValueError("time_valid must be bool [B,T]")
        if displacement_progress.shape != (batch_size, time):
            raise ValueError("displacement_progress must be [B,T]")
        if view_valid.shape != (batch_size, views) or view_valid.dtype != torch.bool:
            raise ValueError("view_valid must be bool [B,V]")
        values = (pooled_history, current_maps, initial_query_tokens, displacement_progress)
        if not all(value.is_floating_point() and bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("cached measurement tensors must be finite floating point")

        maps = pooled_history.detach()
        current_maps = current_maps.detach()
        initial_query_tokens = initial_query_tokens.detach()
        history = maps.permute(0, 2, 1, 4, 5, 3).reshape(
            batch_size * views,
            time * 8 * 8,
            TOKEN_DIM,
        )
        memory = self.patch_projection(history)
        progress = _progress_encoding(
            displacement_progress.to(memory.dtype),
            self.dimension,
        )
        progress = progress[:, None, :, None].expand(-1, views, -1, 8 * 8, -1)
        memory = memory + progress.reshape(
            batch_size * views,
            time * 8 * 8,
            self.dimension,
        )
        memory_padding = (~time_valid)[:, None, :, None].expand(
            -1,
            views,
            -1,
            8 * 8,
        ).reshape(batch_size * views, time * 8 * 8)

        current = current_maps.permute(0, 1, 3, 4, 2).reshape(
            batch_size * views,
            PATCH_GRID * PATCH_GRID,
            TOKEN_DIM,
        )
        spatial = self.patch_projection(current)
        slots = self.query_projection(
            initial_query_tokens.reshape(
                batch_size * views,
                TIP_COUNT,
                TOKEN_DIM,
            ).to(spatial.dtype)
        )
        slots = slots + self.tip_identity.weight[None]
        for block in self.temporal_blocks:
            spatial = block(spatial, memory, memory_padding=memory_padding)
        spatial = self.spatial_output(spatial)
        spatial_padding = torch.zeros(
            batch_size * views,
            PATCH_GRID * PATCH_GRID,
            dtype=torch.bool,
            device=spatial.device,
        )
        slots = self.slot_attention(
            slots,
            spatial,
            memory_padding=spatial_padding,
        )
        slots = self.slot_output(slots)
        spatial = F.normalize(spatial.float(), dim=-1)
        slots = F.normalize(slots.float(), dim=-1)
        logits = torch.einsum("bpd,btd->btp", spatial, slots)
        logits = (logits / self.temperature).reshape(
            batch_size,
            views,
            TIP_COUNT,
            PATCH_GRID,
            PATCH_GRID,
        )
        view_mask = view_valid[:, :, None, None, None]
        logits = logits.masked_fill(~view_mask, -torch.finfo(logits.dtype).max)
        heatmaps = logits.flatten(-2).softmax(dim=-1).reshape_as(logits)
        heatmaps = heatmaps * view_mask
        tip_px = decode_unique_tip_pixels(logits, image_size=image_size)
        tip_px = tip_px * view_valid[:, :, None, None]
        anchor_px, part_tokens, anchor_tokens = _sample_current_anchors(
            current_maps,
            tip_px,
            image_size=image_size,
        )
        valid_tokens = view_valid[:, :, None, None]
        return QueryConditionedMeasurement(
            measurement_logits=logits,
            heatmaps=heatmaps,
            tip_px=tip_px,
            anchor_px=anchor_px * view_valid[:, :, None, None],
            part_tokens=part_tokens * valid_tokens,
            anchor_tokens=anchor_tokens * valid_tokens,
            visual_slots=slots.reshape(batch_size, views, TIP_COUNT, self.dimension),
            endpoint_index=endpoint,
        )

    def forward_cached(
        self,
        *,
        pooled_history: torch.Tensor,
        current_patch_maps: torch.Tensor,
        initial_query_tokens: torch.Tensor,
        time_valid: torch.Tensor,
        displacement_progress: torch.Tensor,
        view_valid: torch.Tensor,
        image_size: tuple[int, int] = (224, 224),
    ) -> QueryConditionedMeasurement:
        """Train/infer from the compact one-pass frozen-V12 cache."""

        batch_size, time = pooled_history.shape[:2]
        if not 2 <= time <= MAX_EVENT_FRAMES:
            raise ValueError("cached query measurement requires 2..16 frames")
        if time_valid.shape != (batch_size, time):
            raise ValueError("cached time_valid does not align with history")
        if not bool(time_valid[:, 0].all()) or bool(
            (time_valid[:, 1:] & ~time_valid[:, :-1]).any()
        ):
            raise ValueError("cached time_valid must be a non-empty valid prefix")
        pair_valid = time_valid[:, 1:] & time_valid[:, :-1]
        delta = displacement_progress[:, 1:] - displacement_progress[:, :-1]
        if bool((delta[pair_valid] < 0.0).any()):
            raise ValueError("cached displacement progress must be non-decreasing")
        valid_progress = displacement_progress[time_valid]
        if bool(((valid_progress < 0.0) | (valid_progress > 1.0)).any()):
            raise ValueError("cached displacement progress must lie in [0,1]")
        endpoint = _endpoint_rows(time_valid)
        return self._measure_cached(
            pooled_history=pooled_history,
            current_maps=current_patch_maps,
            initial_query_tokens=initial_query_tokens,
            time_valid=time_valid,
            displacement_progress=displacement_progress,
            view_valid=view_valid,
            endpoint=endpoint,
            image_size=image_size,
        )

    def forward(
        self,
        batch: QueryConditionedMeasurementBatch,
        *,
        image_size: tuple[int, int] = (224, 224),
    ) -> QueryConditionedMeasurement:
        batch.validate()
        maps = batch.patch_maps.detach()
        batch_size, time, views = maps.shape[:3]
        endpoint = _endpoint_rows(batch.time_valid)
        current_maps = _current_maps(maps, endpoint)

        pooled = F.avg_pool2d(
            maps.reshape(
                batch_size * time * views,
                TOKEN_DIM,
                PATCH_GRID,
                PATCH_GRID,
            ),
            kernel_size=2,
            stride=2,
        ).reshape(batch_size, time, views, TOKEN_DIM, 8, 8)
        first_maps = maps[:, 0].reshape(
            batch_size * views,
            TOKEN_DIM,
            PATCH_GRID,
            PATCH_GRID,
        )
        with torch.autocast(device_type=maps.device.type, enabled=False):
            initial = sample_feature_map_at_pixels(
                first_maps.float(),
                batch.query_px.reshape(batch_size * views, TIP_COUNT, 2).float(),
                image_size=image_size,
            )
        initial = initial.reshape(
            batch_size,
            views,
            TIP_COUNT,
            TOKEN_DIM,
        )
        return self._measure_cached(
            pooled_history=pooled,
            current_maps=current_maps,
            initial_query_tokens=initial,
            time_valid=batch.time_valid,
            displacement_progress=batch.displacement_progress,
            view_valid=batch.view_valid,
            endpoint=endpoint,
            image_size=image_size,
        )


def teacher_heatmap_cross_entropy(
    measurement_logits: torch.Tensor,
    teacher_tip_px: torch.Tensor,
    *,
    teacher_visible: torch.Tensor,
    view_valid: torch.Tensor,
    image_size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """The sole FCPG-H training loss: causal teacher patch cross-entropy."""

    if measurement_logits.ndim != 5 or measurement_logits.shape[2:] != (
        TIP_COUNT,
        PATCH_GRID,
        PATCH_GRID,
    ):
        raise ValueError("measurement logits must be [B,V,2,16,16]")
    batch, views = measurement_logits.shape[:2]
    if teacher_tip_px.shape != (batch, views, TIP_COUNT, 2):
        raise ValueError("teacher_tip_px must be [B,V,2,2]")
    if teacher_visible.shape != (batch, views, TIP_COUNT) or teacher_visible.dtype != torch.bool:
        raise ValueError("teacher_visible must be bool [B,V,2]")
    if view_valid.shape != (batch, views) or view_valid.dtype != torch.bool:
        raise ValueError("view_valid must be bool [B,V]")
    width, height = image_size
    finite = torch.isfinite(teacher_tip_px).all(dim=-1)
    in_frame = (
        (teacher_tip_px[..., 0] >= 0.0)
        & (teacher_tip_px[..., 0] < float(width))
        & (teacher_tip_px[..., 1] >= 0.0)
        & (teacher_tip_px[..., 1] < float(height))
    )
    valid = teacher_visible & view_valid[:, :, None] & finite & in_frame
    if not bool(valid.any()):
        return measurement_logits.sum() * 0.0
    x = torch.floor(teacher_tip_px[..., 0] * PATCH_GRID / float(width)).long()
    y = torch.floor(teacher_tip_px[..., 1] * PATCH_GRID / float(height)).long()
    target = (y.clamp(0, PATCH_GRID - 1) * PATCH_GRID + x.clamp(0, PATCH_GRID - 1))
    loss = F.cross_entropy(
        measurement_logits.flatten(-2).reshape(-1, PATCH_GRID * PATCH_GRID),
        target.reshape(-1),
        reduction="none",
    ).reshape(batch, views, TIP_COUNT)
    return loss[valid].mean()


class PartTokenLeWM(nn.Module):
    """The observer's only action/CAD-conditioned transition model."""

    def __init__(
        self,
        *,
        action_dim: int = 6,
        action_hidden: int = 64,
        hidden_dim: int = 256,
        residual_cap: float = LEWM_RESIDUAL_CAP,
    ) -> None:
        super().__init__()
        if action_dim != 6:
            raise ValueError("SO101 completed actions are six-dimensional")
        if residual_cap != LEWM_RESIDUAL_CAP:
            raise ValueError("LeWM token residual cap is frozen at 0.25")
        self.action_dim = action_dim
        self.action_hidden = action_hidden
        self.hidden_dim = hidden_dim
        self.residual_cap = float(residual_cap)
        self.action_encoder = nn.GRU(
            action_dim,
            action_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.tip_identity = nn.Embedding(TIP_COUNT, 32)
        joined_dim = TOKEN_DIM + action_hidden + 6 + 3 + 32
        self.input_norm = nn.LayerNorm(joined_dim)
        self.transition = nn.Sequential(
            nn.Linear(joined_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.output = nn.Linear(hidden_dim, TOKEN_DIM)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        assert_parameter_limit(
            self,
            maximum=LEWM_PARAMETER_LIMIT,
            name="part-token LeWM",
        )

    def forward(
        self,
        previous_part_tokens: torch.Tensor,
        *,
        state_delta: torch.Tensor,
        completed_actions: torch.Tensor,
        cad_tip_displacement: torch.Tensor,
    ) -> PartTokenPrior:
        if previous_part_tokens.ndim != 4 or previous_part_tokens.shape[2:] != (
            TIP_COUNT,
            TOKEN_DIM,
        ):
            raise ValueError("previous_part_tokens must be [B,V,2,192]")
        batch, views = previous_part_tokens.shape[:2]
        if state_delta.shape != (batch, 6):
            raise ValueError("state_delta must be [B,6]")
        if completed_actions.ndim != 3 or completed_actions.shape[0] != batch or completed_actions.shape[2] != 6:
            raise ValueError("completed_actions must be [B,A>=1,6]")
        if completed_actions.shape[1] < 1:
            raise ValueError("LeWM requires at least one completed action")
        if cad_tip_displacement.shape != (batch, TIP_COUNT, 3):
            raise ValueError("cad_tip_displacement must be [B,2,3]")
        values = (
            previous_part_tokens,
            state_delta,
            completed_actions,
            cad_tip_displacement,
        )
        if not all(value.is_floating_point() and bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("LeWM inputs must be finite floating point")
        _, hidden = self.action_encoder(completed_actions)
        action_context = hidden[-1]
        action_context = action_context[:, None, None].expand(
            -1,
            views,
            TIP_COUNT,
            -1,
        )
        delta = state_delta[:, None, None].expand(-1, views, TIP_COUNT, -1)
        cad = cad_tip_displacement[:, None].expand(-1, views, -1, -1)
        identity = self.tip_identity.weight[None, None].expand(
            batch,
            views,
            -1,
            -1,
        )
        joined = torch.cat(
            (previous_part_tokens, action_context, delta, cad, identity),
            dim=-1,
        )
        dynamics = self.transition(self.input_norm(joined))
        residual = self.residual_cap * torch.tanh(self.output(dynamics))
        return PartTokenPrior(
            part_tokens=previous_part_tokens + residual,
            residual=residual,
            dynamics_latent=dynamics,
        )


def part_token_lewm_objective(
    prior: PartTokenPrior,
    target_part_tokens: torch.Tensor,
    source_index: torch.Tensor,
    *,
    sigreg: Callable[[torch.Tensor], torch.Tensor],
    sigreg_weight: float = LEWM_SIGREG_WEIGHT,
    target_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth-L1 prediction plus source-local SIGReg on dynamics only."""

    if sigreg_weight not in {0.0, LEWM_SIGREG_WEIGHT}:
        raise ValueError("LeWM permits only SIGReg=0 or the frozen 0.02")
    if target_part_tokens.shape != prior.part_tokens.shape:
        raise ValueError("target part tokens must match the LeWM prior")
    if source_index.shape != (len(target_part_tokens),):
        raise ValueError("source_index must be [B]")
    elementwise = F.smooth_l1_loss(
        prior.part_tokens.float(),
        target_part_tokens.detach().float(),
        reduction="none",
    )
    if target_valid is None:
        prediction = elementwise.mean()
    else:
        if target_valid.shape != target_part_tokens.shape[:-1]:
            raise ValueError("target_valid must match target tokens without feature dim")
        if target_valid.dtype != torch.bool:
            raise TypeError("target_valid must be boolean")
        if not bool(target_valid.any()):
            raise ValueError("LeWM objective requires at least one valid target")
        prediction = elementwise.mean(dim=-1)[target_valid].mean()
    regularizers = []
    for source in torch.unique(source_index):
        selected = prior.dynamics_latent[source_index == source]
        if selected.shape[0] >= 2:
            sequence = selected.reshape(selected.shape[0], -1, selected.shape[-1])
            regularizers.append(sigreg(sequence.transpose(0, 1)))
    regularization = (
        prior.dynamics_latent.sum() * 0.0
        if not regularizers
        else torch.stack(regularizers).mean()
    )
    return (
        prediction + float(sigreg_weight) * regularization,
        prediction,
        regularization,
    )


def measurement_dominant_feedback(
    *,
    measurement_logits: torch.Tensor,
    current_patch_maps: torch.Tensor,
    prior_part_tokens: torch.Tensor,
    view_valid: torch.Tensor,
    beta: float = FEEDBACK_BETA,
    temperature: float = FEEDBACK_TEMPERATURE,
    image_size: tuple[int, int] = (224, 224),
) -> ObserverPosterior:
    """Re-query the current image; the prior never transports a pixel."""

    if beta != FEEDBACK_BETA or temperature != FEEDBACK_TEMPERATURE:
        raise ValueError("feedback beta=0.25 and temperature=0.10 are frozen")
    if measurement_logits.ndim != 5 or measurement_logits.shape[2:] != (
        TIP_COUNT,
        PATCH_GRID,
        PATCH_GRID,
    ):
        raise ValueError("measurement_logits must be [B,V,2,16,16]")
    batch, views = measurement_logits.shape[:2]
    if current_patch_maps.shape != (
        batch,
        views,
        TOKEN_DIM,
        PATCH_GRID,
        PATCH_GRID,
    ):
        raise ValueError("current_patch_maps must be [B,V,192,16,16]")
    if prior_part_tokens.shape != (batch, views, TIP_COUNT, TOKEN_DIM):
        raise ValueError("prior_part_tokens must be [B,V,2,192]")
    if view_valid.shape != (batch, views) or view_valid.dtype != torch.bool:
        raise ValueError("view_valid must be bool [B,V]")
    feature = F.normalize(current_patch_maps.detach().float(), dim=2)
    prior = F.normalize(prior_part_tokens.detach().float(), dim=-1)
    similarity = torch.einsum("bvdhw,bvtd->bvthw", feature, prior)
    posterior_logits = measurement_logits.float() + float(beta) * similarity / float(temperature)
    posterior_logits = posterior_logits.masked_fill(
        ~view_valid[:, :, None, None, None],
        -torch.finfo(posterior_logits.dtype).max,
    )
    heatmaps = posterior_logits.flatten(-2).softmax(dim=-1).reshape_as(posterior_logits)
    heatmaps = heatmaps * view_valid[:, :, None, None, None]
    tip_px = decode_unique_tip_pixels(posterior_logits, image_size=image_size)
    tip_px = tip_px * view_valid[:, :, None, None]
    anchor_px, part_tokens, anchor_tokens = _sample_current_anchors(
        current_patch_maps,
        tip_px,
        image_size=image_size,
    )
    valid_tokens = view_valid[:, :, None, None]
    return ObserverPosterior(
        posterior_logits=posterior_logits,
        heatmaps=heatmaps,
        tip_px=tip_px,
        anchor_px=anchor_px * view_valid[:, :, None, None],
        part_tokens=part_tokens * valid_tokens,
        anchor_tokens=anchor_tokens * valid_tokens,
        prior_similarity=similarity,
    )


def rollout_query_conditioned_observer(
    *,
    lewm: PartTokenLeWM,
    initial_part_tokens: torch.Tensor,
    measurement_logits: torch.Tensor,
    future_patch_maps: torch.Tensor,
    state_deltas: torch.Tensor,
    completed_actions: torch.Tensor,
    cad_tip_displacements: torch.Tensor,
    view_valid: torch.Tensor,
) -> ObserverRollout:
    """Unroll H1--H4 with posterior-token, never pixel, writeback."""

    if measurement_logits.ndim != 6 or measurement_logits.shape[3:] != (
        TIP_COUNT,
        PATCH_GRID,
        PATCH_GRID,
    ):
        raise ValueError("measurement_logits must be [B,H,V,2,16,16]")
    batch, horizons, views = measurement_logits.shape[:3]
    if not 1 <= horizons <= 4:
        raise ValueError("observer rollout supports H1..H4 only")
    if future_patch_maps.shape != (
        batch,
        horizons,
        views,
        TOKEN_DIM,
        PATCH_GRID,
        PATCH_GRID,
    ):
        raise ValueError("future_patch_maps must be [B,H,V,192,16,16]")
    if initial_part_tokens.shape != (batch, views, TIP_COUNT, TOKEN_DIM):
        raise ValueError("initial_part_tokens must be [B,V,2,192]")
    if state_deltas.shape != (batch, horizons, 6):
        raise ValueError("state_deltas must be [B,H,6]")
    if completed_actions.ndim != 4 or completed_actions.shape[:2] != (
        batch,
        horizons,
    ) or completed_actions.shape[-1] != 6:
        raise ValueError("completed_actions must be [B,H,A,6]")
    if cad_tip_displacements.shape != (batch, horizons, TIP_COUNT, 3):
        raise ValueError("cad_tip_displacements must be [B,H,2,3]")
    if view_valid.shape == (batch, views):
        valid = view_valid[:, None].expand(-1, horizons, -1)
    elif view_valid.shape == (batch, horizons, views):
        valid = view_valid
    else:
        raise ValueError("view_valid must be [B,V] or [B,H,V]")
    if valid.dtype != torch.bool:
        raise TypeError("view_valid must be boolean")

    history = initial_part_tokens
    priors = []
    posterior_parts = []
    posterior_anchors = []
    posterior_px = []
    posterior_logits = []
    dynamics = []
    for horizon in range(horizons):
        prior = lewm(
            history,
            state_delta=state_deltas[:, horizon],
            completed_actions=completed_actions[:, horizon],
            cad_tip_displacement=cad_tip_displacements[:, horizon],
        )
        posterior = measurement_dominant_feedback(
            measurement_logits=measurement_logits[:, horizon],
            current_patch_maps=future_patch_maps[:, horizon],
            prior_part_tokens=prior.part_tokens,
            view_valid=valid[:, horizon],
        )
        priors.append(prior.part_tokens)
        posterior_parts.append(posterior.part_tokens)
        posterior_anchors.append(posterior.anchor_tokens)
        posterior_px.append(posterior.tip_px)
        posterior_logits.append(posterior.posterior_logits)
        dynamics.append(prior.dynamics_latent)
        history = posterior.part_tokens
    return ObserverRollout(
        priors=torch.stack(priors, dim=1),
        posterior_part_tokens=torch.stack(posterior_parts, dim=1),
        posterior_anchor_tokens=torch.stack(posterior_anchors, dim=1),
        posterior_tip_px=torch.stack(posterior_px, dim=1),
        posterior_logits=torch.stack(posterior_logits, dim=1),
        dynamics_latent=torch.stack(dynamics, dim=1),
    )


__all__ = [
    "ANCHOR_COUNT",
    "CHECKPOINT_FORMAT",
    "FEEDBACK_BETA",
    "FCPGH_PARAMETER_LIMIT",
    "LEWM_PARAMETER_LIMIT",
    "LEWM_CHECKPOINT_FORMAT",
    "LEWM_SIGREG_WEIGHT",
    "METHOD_REVISION",
    "ObserverPosterior",
    "ObserverRollout",
    "PartTokenLeWM",
    "PartTokenPrior",
    "QueryConditionedFCPGH",
    "QueryConditionedMeasurement",
    "QueryConditionedMeasurementBatch",
    "assert_parameter_limit",
    "decode_unique_tip_pixels",
    "measurement_dominant_feedback",
    "part_token_lewm_objective",
    "rollout_query_conditioned_observer",
    "teacher_heatmap_cross_entropy",
    "trainable_parameter_count",
]
