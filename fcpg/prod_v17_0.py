"""Full-image recurrent fingertip grounding for Prod-V17.0.

The model carries only two ordered semantic queries (fixed and moving) for
each camera view. A completed command/readback transition may update those
queries, after which a shared three-layer Transformer decoder reads the full
current 56x56 visual grid. Spatial coordinates are expectations of the visible
categorical posterior; there is no coordinate head, local crop, candidate
bank, confidence gate, router, fallback, or planning component.

Frozen V12 tokens are detached at this module boundary. Current RGB remains
trainable through the RGB stem. Invalid views never update recurrent state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


IMAGE_SIZE = 224
VIEWS = 2
TIPS = 2
ANCHORS = 3
SEMANTIC_GRID = 16
VISUAL_GRID = 56
VISUAL_CELLS = VISUAL_GRID * VISUAL_GRID
OCCLUDED_INDEX = VISUAL_CELLS
MODEL_DIMENSION = 192
STATE_DIMENSION = 6
COMMAND_STEPS = 5
DECODER_LAYERS = 3
DECODER_HEADS = 6
DECODER_FFN = 768
NOMINAL_BIAS_SIGMA_PX = 32.0


class V170ContractError(ValueError):
    """Raised when the V17.0 causal or tensor contract is violated."""


@dataclass(frozen=True)
class V170State:
    """The sole persistent state: ordered per-view semantic F/M queries."""

    semantic_query: torch.Tensor


@dataclass(frozen=True)
class V170Output:
    """Full-image categorical measurement and recurrent writeback."""

    q_cfm_px: torch.Tensor
    ordered_fm_px: torch.Tensor
    valid_cfm: torch.Tensor
    spatial_logits: torch.Tensor
    occluded_logit: torch.Tensor
    categorical_logits: torch.Tensor
    spatial_probability: torch.Tensor
    occluded_probability: torch.Tensor
    visible_spatial_probability: torch.Tensor
    transported_query: torch.Tensor
    decoded_query: torch.Tensor
    native_v12_tokens: torch.Tensor
    next_state: V170State
    fm_posterior: torch.Tensor
    fm_px: torch.Tensor
    midpoint_px: torch.Tensor
    next_query: torch.Tensor


@dataclass(frozen=True)
class V170SpatialLoss:
    """Bilinear categorical NLL over 56x56 visible cells plus occluded."""

    total: torch.Tensor
    visible: torch.Tensor
    occluded: torch.Tensor
    active_tips: torch.Tensor
    visible_tips: torch.Tensor
    occluded_tips: torch.Tensor


def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise V170ContractError(f"{name} must be finite")


def _boolean(
    name: str,
    value: torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    result = torch.as_tensor(value, device=device)
    if result.shape != shape or result.dtype != torch.bool:
        raise V170ContractError(f"{name} must be boolean {list(shape)}")
    return result


def _normalized_rgb_with_contrast(rgb: torch.Tensor) -> torch.Tensor:
    mean = rgb.mean(dim=(-2, -1), keepdim=True)
    scale = rgb.std(dim=(-2, -1), keepdim=True).clamp_min(0.05)
    low = F.avg_pool2d(
        F.pad(rgb, (7, 7, 7, 7), mode="reflect"),
        kernel_size=15,
        stride=1,
    )
    return torch.cat(((rgb - mean) / scale, (rgb - low) / scale), dim=1)


def _sine_position_2d(*, grid: int, dimension: int) -> torch.Tensor:
    if dimension % 4:
        raise ValueError("2-D sine position dimension must be divisible by four")
    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, grid),
        torch.linspace(0.0, 1.0, grid),
        indexing="ij",
    )
    frequency = torch.exp(
        torch.arange(dimension // 4, dtype=torch.float32)
        * (-math.log(10_000.0) / max(dimension // 4 - 1, 1))
    )
    x_phase = 2.0 * math.pi * x.reshape(-1, 1) * frequency
    y_phase = 2.0 * math.pi * y.reshape(-1, 1) * frequency
    return torch.cat(
        (x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()),
        dim=1,
    )


class V170RecurrentQueryFCPG(nn.Module):
    """Shared-per-view full-image Transformer fingertip grounder."""

    def __init__(
        self,
        *,
        semantic_dim: int = MODEL_DIMENSION,
        dimension: int = MODEL_DIMENSION,
        nominal_bias_sigma_px: float = NOMINAL_BIAS_SIGMA_PX,
    ) -> None:
        super().__init__()
        if int(semantic_dim) <= 0:
            raise ValueError("semantic_dim must be positive")
        if int(dimension) != MODEL_DIMENSION:
            raise ValueError("V17.0 dimension is fixed at 192")
        if (
            not math.isfinite(float(nominal_bias_sigma_px))
            or float(nominal_bias_sigma_px) <= 0.0
        ):
            raise ValueError("nominal_bias_sigma_px must be finite and positive")
        self.semantic_dim = int(semantic_dim)
        self.dimension = int(dimension)
        self.nominal_bias_sigma_px = float(nominal_bias_sigma_px)

        # These names are stable training boundaries for SAM/temporal stages.
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(6, 96, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(12, 96),
            nn.GELU(),
            nn.Conv2d(
                96,
                self.dimension,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(24, self.dimension),
            nn.GELU(),
            nn.Conv2d(
                self.dimension,
                self.dimension,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )
        self.semantic_projection = nn.Linear(self.semantic_dim, self.dimension)
        self.visual_fusion = nn.Sequential(
            nn.LayerNorm(self.dimension),
            nn.Linear(self.dimension, self.dimension),
            nn.GELU(),
            nn.Linear(self.dimension, self.dimension),
        )
        self.tip_identity = nn.Parameter(torch.empty(TIPS, self.dimension))
        self.view_identity = nn.Parameter(torch.empty(VIEWS, self.dimension))

        self.transition_projection = nn.Sequential(
            nn.LayerNorm(4 * STATE_DIMENSION),
            nn.Linear(4 * STATE_DIMENSION, self.dimension),
        )
        self.action_position = nn.Parameter(
            torch.empty(COMMAND_STEPS, self.dimension)
        )

        layer = nn.TransformerDecoderLayer(
            d_model=self.dimension,
            nhead=DECODER_HEADS,
            dim_feedforward=DECODER_FFN,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=DECODER_LAYERS,
            norm=nn.LayerNorm(self.dimension),
        )
        self.spatial_query_projection = nn.Linear(
            self.dimension, self.dimension, bias=False
        )
        self.spatial_key_projection = nn.Linear(
            self.dimension, self.dimension, bias=False
        )
        self.occluded_head = nn.Linear(self.dimension, 1)

        self.register_buffer(
            "visual_position",
            _sine_position_2d(grid=VISUAL_GRID, dimension=self.dimension),
            persistent=True,
        )
        axis = torch.linspace(0.0, IMAGE_SIZE - 1.0, VISUAL_GRID)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer(
            "visual_xy_px",
            torch.stack((xx, yy), dim=-1).reshape(VISUAL_CELLS, 2),
            persistent=True,
        )
        nn.init.normal_(self.tip_identity, std=0.02)
        nn.init.normal_(self.view_identity, std=0.02)
        nn.init.normal_(self.action_position, std=0.02)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_state(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> V170State:
        if isinstance(batch, bool) or int(batch) <= 0:
            raise V170ContractError("batch must be a positive integer")
        semantic = self.tip_identity[None, None] + self.view_identity[None, :, None]
        semantic = semantic.to(device=device, dtype=dtype).expand(
            int(batch), VIEWS, TIPS, self.dimension
        )
        return V170State(semantic_query=semantic)

    def initial_query(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the ordered learned H0 query tensor ``[B,2,2,192]``."""

        return self.initial_state(
            batch, device=device, dtype=dtype
        ).semantic_query

    def encode_visual_grid(
        self,
        rgb: torch.Tensor,
        frozen_v12_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return memory ``[B,2,3136,192]`` and detached native tokens."""

        images = torch.as_tensor(rgb)
        if images.ndim != 5 or tuple(images.shape[1:]) != (
            VIEWS,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise V170ContractError("rgb must be [B,2,3,224,224]")
        batch = len(images)
        native = torch.as_tensor(frozen_v12_tokens, device=images.device)
        if tuple(native.shape) != (
            batch,
            VIEWS,
            SEMANTIC_GRID * SEMANTIC_GRID,
            self.semantic_dim,
        ):
            raise V170ContractError(
                f"frozen_v12_tokens must be [B,2,256,{self.semantic_dim}]"
            )
        images = images.float()
        native = native.detach().float()
        _finite("rgb", images)
        _finite("frozen_v12_tokens", native)

        rgb_grid = self.rgb_stem(
            _normalized_rgb_with_contrast(
                images.reshape(batch * VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE)
            )
        )
        if tuple(rgb_grid.shape) != (
            batch * VIEWS,
            self.dimension,
            VISUAL_GRID,
            VISUAL_GRID,
        ):
            raise RuntimeError("RGB stem must produce a 56x56 token grid")
        semantic = self.semantic_projection(native).reshape(
            batch * VIEWS,
            SEMANTIC_GRID,
            SEMANTIC_GRID,
            self.dimension,
        )
        semantic = F.interpolate(
            semantic.permute(0, 3, 1, 2),
            size=(VISUAL_GRID, VISUAL_GRID),
            mode="bilinear",
            align_corners=False,
        )
        fused = rgb_grid + semantic
        fused = self.visual_fusion(fused.permute(0, 2, 3, 1))
        memory = fused.reshape(batch, VIEWS, VISUAL_CELLS, self.dimension)
        memory = memory + self.visual_position.to(memory)[None, None]
        return memory, native

    def _transition_tokens(
        self,
        *,
        commands_sent: torch.Tensor,
        measured_joint_path: torch.Tensor,
        transition_valid: torch.Tensor,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        commands = torch.as_tensor(commands_sent, device=device).detach().to(dtype)
        path = torch.as_tensor(measured_joint_path, device=device).detach().to(dtype)
        valid = _boolean(
            "transition_valid",
            transition_valid,
            shape=(batch,),
            device=device,
        )
        if commands.shape != (batch, COMMAND_STEPS, STATE_DIMENSION):
            raise V170ContractError("commands_sent must be [B,5,6]")
        if path.shape != (batch, COMMAND_STEPS + 1, STATE_DIMENSION):
            raise V170ContractError("measured_joint_path must be [B,6,6]")
        _finite("commands_sent", commands)
        _finite("measured_joint_path", path)
        previous_command = torch.cat((path[:, :1], commands[:, :-1]), dim=1)
        features = torch.cat(
            (
                commands,
                commands - previous_command,
                path[:, 1:] - path[:, :-1],
                path[:, 1:] - path[:, :1],
            ),
            dim=-1,
        )
        tokens = self.transition_projection(features)
        tokens = tokens + self.action_position.to(tokens)[None]
        return tokens * valid[:, None, None].to(dtype)

    def step(
        self,
        rgb: torch.Tensor,
        frozen_v12_tokens: torch.Tensor,
        nominal_fm_px: torch.Tensor,
        completed_commands: torch.Tensor,
        measured_joint_path: torch.Tensor,
        previous_query: torch.Tensor,
        transition_valid: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> V170Output:
        memory, detached_native = self.encode_visual_grid(rgb, frozen_v12_tokens)
        batch = len(memory)
        device, dtype = memory.device, memory.dtype
        nominal = torch.as_tensor(nominal_fm_px, device=device).detach().to(dtype)
        if nominal.shape != (batch, VIEWS, TIPS, 2):
            raise V170ContractError("nominal_fm_px must be [B,2,2,2]")
        _finite("nominal_fm_px", nominal)
        if bool(((nominal < 0.0) | (nominal > IMAGE_SIZE - 1.0)).any()):
            raise V170ContractError("nominal_fm_px must lie inside the image")
        views = _boolean(
            "view_valid", view_valid, shape=(batch, VIEWS), device=device
        )
        current_valid = views[:, :, None].expand(-1, -1, TIPS)
        prior = torch.as_tensor(previous_query, device=device).to(dtype)
        if prior.shape != (batch, VIEWS, TIPS, self.dimension):
            raise V170ContractError("previous_query must be [B,2,2,192]")
        _finite("previous_query", prior)

        action_tokens = self._transition_tokens(
            commands_sent=completed_commands,
            measured_joint_path=measured_joint_path,
            transition_valid=transition_valid,
            batch=batch,
            device=device,
            dtype=dtype,
        )
        transported = prior
        action_by_view = action_tokens[:, None].expand(-1, VIEWS, -1, -1)
        target = torch.cat((transported, action_by_view), dim=2)

        # B and V are flattened: weights are shared but views never mix. The
        # two ordered queries self-attend the five completed-transition tokens.
        decoded_all = self.decoder(
            target.reshape(
                batch * VIEWS, TIPS + COMMAND_STEPS, self.dimension
            ),
            memory.reshape(batch * VIEWS, VISUAL_CELLS, self.dimension),
        ).reshape(
            batch, VIEWS, TIPS + COMMAND_STEPS, self.dimension
        )
        decoded = decoded_all[:, :, :TIPS]
        query_key = self.spatial_query_projection(decoded)
        spatial_key = self.spatial_key_projection(memory)
        spatial_logits = torch.einsum(
            "bvtd,bvnd->bvtn", query_key, spatial_key
        ) / math.sqrt(self.dimension)

        displacement = (
            self.visual_xy_px.to(device=device, dtype=dtype)[None, None, None]
            - nominal[:, :, :, None]
        )
        nominal_bias = -0.5 * displacement.square().sum(dim=-1) / (
            self.nominal_bias_sigma_px**2
        )
        spatial_logits = spatial_logits + nominal_bias
        occluded_logit = self.occluded_head(decoded).squeeze(-1)
        categorical_logits = torch.cat(
            (spatial_logits, occluded_logit[..., None]), dim=-1
        )
        categorical_probability = torch.softmax(categorical_logits, dim=-1)
        spatial_probability = categorical_probability[..., :VISUAL_CELLS]
        occluded_probability = categorical_probability[..., OCCLUDED_INDEX]
        visible_mass = spatial_probability.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        visible_probability = spatial_probability / visible_mass

        spatial_probability = torch.where(
            current_valid[..., None],
            spatial_probability,
            torch.zeros_like(spatial_probability),
        )
        visible_probability = torch.where(
            current_valid[..., None],
            visible_probability,
            torch.zeros_like(visible_probability),
        )
        occluded_probability = torch.where(
            current_valid,
            occluded_probability,
            torch.ones_like(occluded_probability),
        )
        ordered = torch.einsum(
            "bvtn,nd->bvtd",
            visible_probability,
            self.visual_xy_px.to(device=device, dtype=dtype),
        )
        ordered = torch.where(
            current_valid[..., None], ordered, torch.zeros_like(ordered)
        )
        center = ordered.mean(dim=2)
        q_cfm = torch.stack((center, ordered[:, :, 0], ordered[:, :, 1]), dim=2)
        valid_cfm = torch.stack(
            (current_valid.all(dim=2), current_valid[:, :, 0], current_valid[:, :, 1]),
            dim=2,
        )
        next_semantic = torch.where(current_valid[..., None], decoded, prior)
        fm_posterior = torch.cat(
            (
                spatial_probability,
                occluded_probability[..., None],
            ),
            dim=-1,
        )
        midpoint = q_cfm[:, :, 0]
        return V170Output(
            q_cfm_px=q_cfm,
            ordered_fm_px=ordered,
            valid_cfm=valid_cfm,
            spatial_logits=spatial_logits.reshape(
                batch, VIEWS, TIPS, VISUAL_GRID, VISUAL_GRID
            ),
            occluded_logit=occluded_logit,
            categorical_logits=categorical_logits,
            spatial_probability=spatial_probability.reshape(
                batch, VIEWS, TIPS, VISUAL_GRID, VISUAL_GRID
            ),
            occluded_probability=occluded_probability,
            visible_spatial_probability=visible_probability.reshape(
                batch, VIEWS, TIPS, VISUAL_GRID, VISUAL_GRID
            ),
            transported_query=transported,
            decoded_query=decoded,
            native_v12_tokens=detached_native,
            next_state=V170State(semantic_query=next_semantic),
            fm_posterior=fm_posterior,
            fm_px=ordered,
            midpoint_px=midpoint,
            next_query=next_semantic,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        frozen_v12_tokens: torch.Tensor,
        nominal_fm_px: torch.Tensor,
        completed_commands: torch.Tensor,
        measured_joint_path: torch.Tensor,
        previous_query: torch.Tensor,
        transition_valid: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> V170Output:
        return self.step(
            rgb,
            frozen_v12_tokens,
            nominal_fm_px,
            completed_commands,
            measured_joint_path,
            previous_query,
            transition_valid,
            view_valid,
        )


def v170_spatial_categorical_nll(
    output: V170Output,
    target_fm_px: torch.Tensor,
    target_visible: torch.Tensor,
    target_valid: torch.Tensor | None = None,
) -> V170SpatialLoss:
    """Compute bilinear NLL on the full visible grid or occluded category."""

    logits = output.categorical_logits
    batch = len(logits)
    device, dtype = logits.device, logits.dtype
    target = torch.as_tensor(target_fm_px, device=device).detach().to(dtype)
    if target.shape != (batch, VIEWS, TIPS, 2):
        raise V170ContractError("target_fm_px must be [B,2,2,2]")
    _finite("target_fm_px", target)
    visible = _boolean(
        "target_visible",
        target_visible,
        shape=(batch, VIEWS, TIPS),
        device=device,
    )
    model_valid = output.valid_cfm[:, :, 1:]
    if target_valid is None:
        active = model_valid
    else:
        requested = _boolean(
            "target_valid",
            target_valid,
            shape=(batch, VIEWS, TIPS),
            device=device,
        )
        active = model_valid & requested
    # The CAD renderer defines a valid projection on the continuous half-open
    # viewport [0, W) x [0, H). Values in [W-1, W) belong to the final pixel
    # cell; the clamped lower/upper indices below fold their bilinear mass into
    # the last spatial bin without changing the source label.
    outside = ((target < 0.0) | (target >= IMAGE_SIZE)).any(dim=-1)
    if bool((outside & active & visible).any()):
        raise V170ContractError("active visible target lies outside the image")

    grid_xy = target * ((VISUAL_GRID - 1.0) / (IMAGE_SIZE - 1.0))
    lower = torch.floor(grid_xy).long().clamp(0, VISUAL_GRID - 1)
    upper = (lower + 1).clamp(0, VISUAL_GRID - 1)
    fraction = (grid_xy - lower.to(dtype)).clamp(0.0, 1.0)
    x0, y0 = lower.unbind(dim=-1)
    x1, y1 = upper.unbind(dim=-1)
    fx, fy = fraction.unbind(dim=-1)
    log_probability = F.log_softmax(logits, dim=-1)

    def log_at(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        index = y * VISUAL_GRID + x
        return torch.gather(log_probability, -1, index[..., None]).squeeze(-1)

    visible_log_likelihood = (
        (1.0 - fx) * (1.0 - fy) * log_at(x0, y0)
        + fx * (1.0 - fy) * log_at(x1, y0)
        + (1.0 - fx) * fy * log_at(x0, y1)
        + fx * fy * log_at(x1, y1)
    )
    occluded_log_likelihood = log_probability[..., OCCLUDED_INDEX]
    selected = torch.where(visible, visible_log_likelihood, occluded_log_likelihood)
    active_count = active.sum()
    visible_active = active & visible
    occluded_active = active & ~visible

    def masked_mean_negative(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if int(mask.sum().item()) == 0:
            return logits.sum() * 0.0
        return -value.masked_select(mask).mean()

    return V170SpatialLoss(
        total=masked_mean_negative(selected, active),
        visible=masked_mean_negative(visible_log_likelihood, visible_active),
        occluded=masked_mean_negative(occluded_log_likelihood, occluded_active),
        active_tips=active_count,
        visible_tips=visible_active.sum(),
        occluded_tips=occluded_active.sum(),
    )


__all__ = [
    "ANCHORS",
    "COMMAND_STEPS",
    "DECODER_FFN",
    "DECODER_HEADS",
    "DECODER_LAYERS",
    "IMAGE_SIZE",
    "MODEL_DIMENSION",
    "OCCLUDED_INDEX",
    "TIPS",
    "VIEWS",
    "VISUAL_CELLS",
    "VISUAL_GRID",
    "V170ContractError",
    "V170Output",
    "V170RecurrentQueryFCPG",
    "V170SpatialLoss",
    "V170State",
    "v170_spatial_categorical_nll",
]
