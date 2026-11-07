"""Dual-view action-conditioned image registration for Prod-V17.2.

The robot's three-dimensional structure remains the responsibility of CAD,
forward kinematics, LeWM, and the planner.  FCPG solves the narrower observable
problem: where the calibrated nominal fingertips land in the two current
images.  One shared Query Transformer predicts a bounded 2-D similarity
registration for each view and applies it jointly to ordered F/M:

    q_corrected = Similarity(delta_uv, complex_gain) @ q_nominal

There is no learned SE(3), direct absolute-pixel head, independent per-tip
residual, confidence gate, router, or CEM.  View order is always
``(column, wrist)``.  Persistent state contains semantic queries, bounded
registration parameters, and frozen-V12 endpoint tokens; it never stores
pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


IMAGE_SIZE = 224
VIEWS = 2
VIEW_NAMES = ("column", "wrist")
TIPS = 2
TIP_NAMES = ("F", "M")
STATE_DIMENSION = 6
COMMAND_STEPS = 5
READBACK_STEPS = 6
SEMANTIC_GRID = 16
SEMANTIC_CELLS = SEMANTIC_GRID * SEMANTIC_GRID
VISUAL_GRID = 56
VISUAL_CELLS = VISUAL_GRID * VISUAL_GRID
HISTORY_ENDPOINTS = 5
FROZEN_V12_DIMENSION = 192
MODEL_DIMENSION = 384
DECODER_LAYERS = 3
DECODER_HEADS = 6
DECODER_FFN = 1536
SIMILARITY_PARAMETERS = 4
TRANSLATION_BOUND_PX = 24.0
COMPLEX_GAIN_BOUND = 2.0
COMPLEX_IDENTITY_LOGIT = math.atanh(0.5)


class V172ContractError(ValueError):
    """Raised when a V17.2 tensor, clock, or unit contract is violated."""


@dataclass(frozen=True)
class V172FCPGMemory:
    """Persistent non-pixel state for sparse H0--H4 endpoint inference."""

    registration_query: torch.Tensor
    similarity_parameters: torch.Tensor
    endpoint_tokens: torch.Tensor
    endpoint_valid: torch.Tensor

    def detach(self) -> "V172FCPGMemory":
        return V172FCPGMemory(
            registration_query=self.registration_query.detach(),
            similarity_parameters=self.similarity_parameters.detach(),
            endpoint_tokens=self.endpoint_tokens.detach(),
            endpoint_valid=self.endpoint_valid.detach(),
        )


@dataclass(frozen=True)
class V172Output:
    """Registered fingertips and the next recurrent state."""

    similarity_parameters: torch.Tensor
    nominal_fm_px: torch.Tensor
    fm_px: torch.Tensor
    midpoint_px: torch.Tensor
    fm_valid: torch.Tensor
    registration_query: torch.Tensor
    layer_registration_tokens: torch.Tensor
    fused_visual_tokens: torch.Tensor
    next_memory: V172FCPGMemory


@dataclass(frozen=True)
class V172FMLoss:
    total: torch.Tensor
    active_points: torch.Tensor
    mean_epe_px: torch.Tensor


@dataclass(frozen=True)
class V172TemporalFeatures:
    registration_query: torch.Tensor
    layer_registration_tokens: torch.Tensor
    next_memory: V172FCPGMemory


def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise V172ContractError(f"{name} must be finite")


def _boolean(
    name: str,
    value: torch.Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    result = torch.as_tensor(value, device=device)
    if result.shape != shape or result.dtype != torch.bool:
        raise V172ContractError(f"{name} must be boolean {list(shape)}")
    return result


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
        (x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()), dim=-1
    )


def _normalized_rgb_with_contrast(rgb: torch.Tensor) -> torch.Tensor:
    mean = rgb.mean(dim=(-2, -1), keepdim=True)
    scale = rgb.std(dim=(-2, -1), keepdim=True).clamp_min(0.05)
    low = F.avg_pool2d(
        F.pad(rgb, (7, 7, 7, 7), mode="reflect"),
        kernel_size=15,
        stride=1,
    )
    return torch.cat(((rgb - mean) / scale, (rgb - low) / scale), dim=1)


def apply_fm_similarity(
    nominal_fm_px: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Apply ``[du,dv,a,b]`` jointly to ordered F/M.

    ``a + i b`` is the bounded complex gain on the ordered F-to-M
    displacement.  Unlike angle/log-scale coordinates it continuously
    represents a collapsed projection (``a=b=0``) and a 180-degree identity
    flip (negative ``a``), both of which occur in physical camera views.
    """

    nominal = torch.as_tensor(nominal_fm_px)
    value = torch.as_tensor(
        parameters, device=nominal.device, dtype=nominal.dtype
    )
    if nominal.ndim != 4 or tuple(nominal.shape[1:]) != (VIEWS, TIPS, 2):
        raise V172ContractError("nominal_fm_px must be [B,2,2,2]")
    if value.shape != (len(nominal), VIEWS, SIMILARITY_PARAMETERS):
        raise V172ContractError("similarity parameters must be [B,2,4]")
    _finite("nominal_fm_px", nominal)
    _finite("similarity_parameters", value)
    center = nominal.mean(dim=2, keepdim=True)
    relative = nominal - center
    real, imaginary = value[..., 2], value[..., 3]
    transform = torch.stack(
        (real, -imaginary, imaginary, real), dim=-1
    ).reshape(len(nominal), VIEWS, 2, 2)
    scaled = torch.einsum("bvij,bvtj->bvti", transform, relative)
    translation = value[..., :2][:, :, None]
    return center + scaled + translation


class V172SimilarityRegistrationFCPG(nn.Module):
    """One shared Query Transformer with bounded per-view similarity heads."""

    def __init__(
        self,
        *,
        semantic_dim: int = FROZEN_V12_DIMENSION,
        dimension: int = MODEL_DIMENSION,
    ) -> None:
        super().__init__()
        if int(semantic_dim) <= 0:
            raise ValueError("semantic_dim must be positive")
        if int(dimension) <= 0 or int(dimension) % DECODER_HEADS:
            raise ValueError("dimension must be positive and divisible by six")
        self.semantic_dim = int(semantic_dim)
        self.dimension = int(dimension)

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
            nn.LayerNorm(2 * self.dimension),
            nn.Linear(2 * self.dimension, self.dimension),
            nn.GELU(),
            nn.Linear(self.dimension, self.dimension),
        )

        self.registration_identity = nn.Parameter(
            torch.empty(VIEWS, self.dimension)
        )
        self.view_identity = nn.Parameter(torch.empty(VIEWS, self.dimension))
        self.tip_identity = nn.Parameter(torch.empty(TIPS, self.dimension))
        self.geometry_projection = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, self.dimension),
            nn.GELU(),
            nn.Linear(self.dimension, self.dimension),
        )
        self.state_projection = nn.Sequential(
            nn.LayerNorm(STATE_DIMENSION),
            nn.Linear(STATE_DIMENSION, self.dimension),
        )
        self.command_projection = nn.Sequential(
            nn.LayerNorm(2 * STATE_DIMENSION),
            nn.Linear(2 * STATE_DIMENSION, self.dimension),
        )
        self.readback_projection = nn.Sequential(
            nn.LayerNorm(2 * STATE_DIMENSION),
            nn.Linear(2 * STATE_DIMENSION, self.dimension),
        )
        self.previous_similarity_projection = nn.Sequential(
            nn.LayerNorm(SIMILARITY_PARAMETERS),
            nn.Linear(SIMILARITY_PARAMETERS, self.dimension),
        )
        self.command_position = nn.Parameter(
            torch.empty(COMMAND_STEPS, self.dimension)
        )
        self.readback_position = nn.Parameter(
            torch.empty(READBACK_STEPS, self.dimension)
        )
        self.history_time = nn.Parameter(
            torch.empty(HISTORY_ENDPOINTS, self.dimension)
        )
        self.null_memory = nn.Parameter(torch.empty(1, self.dimension))

        base_layer = nn.TransformerDecoderLayer(
            d_model=self.dimension,
            nhead=DECODER_HEADS,
            dim_feedforward=DECODER_FFN,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_layers = nn.ModuleList(
            copy.deepcopy(base_layer) for _ in range(DECODER_LAYERS)
        )
        self.query_norm = nn.LayerNorm(self.dimension)
        self.similarity_head = nn.Sequential(
            nn.LayerNorm(self.dimension),
            nn.Linear(self.dimension, self.dimension),
            nn.GELU(),
            nn.Linear(self.dimension, SIMILARITY_PARAMETERS),
        )

        self.register_buffer(
            "visual_position",
            _sine_position_2d(grid=VISUAL_GRID, dimension=self.dimension),
            persistent=True,
        )
        self.register_buffer(
            "semantic_position",
            _sine_position_2d(grid=SEMANTIC_GRID, dimension=self.dimension),
            persistent=True,
        )
        self.register_buffer(
            "identity_similarity",
            torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32),
            persistent=True,
        )

        for parameter in (
            self.registration_identity,
            self.view_identity,
            self.tip_identity,
            self.command_position,
            self.readback_position,
            self.history_time,
            self.null_memory,
        ):
            nn.init.normal_(parameter, std=0.02)
        final = self.similarity_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_memory(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> V172FCPGMemory:
        if isinstance(batch, bool) or int(batch) <= 0:
            raise V172ContractError("batch must be a positive integer")
        registration = (
            self.registration_identity + self.view_identity
        ).to(device=device, dtype=dtype)[None].expand(
            int(batch), VIEWS, self.dimension
        )
        parameters = self.identity_similarity.to(
            device=device, dtype=dtype
        )[None, None].expand(int(batch), VIEWS, -1).clone()
        return V172FCPGMemory(
            registration_query=registration,
            similarity_parameters=parameters,
            endpoint_tokens=torch.zeros(
                int(batch),
                HISTORY_ENDPOINTS,
                VIEWS,
                SEMANTIC_CELLS,
                self.semantic_dim,
                device=device,
                dtype=dtype,
            ),
            endpoint_valid=torch.zeros(
                int(batch),
                HISTORY_ENDPOINTS,
                VIEWS,
                device=device,
                dtype=torch.bool,
            ),
        )

    def _validate_memory(
        self,
        memory: V172FCPGMemory,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> V172FCPGMemory:
        registration = torch.as_tensor(
            memory.registration_query, device=device, dtype=dtype
        )
        parameters = torch.as_tensor(
            memory.similarity_parameters, device=device, dtype=dtype
        )
        endpoint = torch.as_tensor(
            memory.endpoint_tokens, device=device, dtype=dtype
        )
        valid = _boolean(
            "memory.endpoint_valid",
            memory.endpoint_valid,
            shape=(batch, HISTORY_ENDPOINTS, VIEWS),
            device=device,
        )
        if registration.shape != (batch, VIEWS, self.dimension):
            raise V172ContractError(
                f"memory.registration_query must be [B,2,{self.dimension}]"
            )
        if parameters.shape != (batch, VIEWS, SIMILARITY_PARAMETERS):
            raise V172ContractError("memory.similarity_parameters must be [B,2,4]")
        if endpoint.shape != (
            batch,
            HISTORY_ENDPOINTS,
            VIEWS,
            SEMANTIC_CELLS,
            self.semantic_dim,
        ):
            raise V172ContractError(
                "memory.endpoint_tokens must be [B,5,2,256,semantic_dim]"
            )
        _finite("memory.registration_query", registration)
        _finite("memory.similarity_parameters", parameters)
        _finite("memory.endpoint_tokens", endpoint)
        if bool(
            (
                parameters[..., :2].abs()
                > TRANSLATION_BOUND_PX + 1.0e-5
            ).any()
        ):
            raise V172ContractError("memory translation leaves the fixed bounds")
        gain = torch.linalg.vector_norm(parameters[..., 2:], dim=-1)
        if bool((gain > COMPLEX_GAIN_BOUND + 1.0e-5).any()):
            raise V172ContractError("memory complex gain leaves the fixed disk")
        return V172FCPGMemory(registration, parameters, endpoint, valid)

    def _normalized_similarity(self, parameters: torch.Tensor) -> torch.Tensor:
        centered = parameters - self.identity_similarity.to(parameters)[None, None]
        translation = centered[..., :2] / TRANSLATION_BOUND_PX
        gain = centered[..., 2:] / COMPLEX_GAIN_BOUND
        return torch.cat((translation, gain), dim=-1)

    def _bounded_similarity(self, logits: torch.Tensor) -> torch.Tensor:
        translation = torch.tanh(logits[..., :2]) * TRANSLATION_BOUND_PX
        # Smoothly map R^2 into a radius-two disk.  The offset makes zero
        # network logits exactly the identity gain (1 + 0i), preserving the
        # zero-initialized nominal contract.
        shifted = logits[..., 2:] + torch.tensor(
            (COMPLEX_IDENTITY_LOGIT, 0.0),
            device=logits.device,
            dtype=logits.dtype,
        )
        norm = torch.linalg.vector_norm(shifted, dim=-1, keepdim=True)
        factor = (
            COMPLEX_GAIN_BOUND
            * torch.tanh(norm)
            / norm.clamp_min(1.0e-6)
        )
        gain = shifted * factor
        return torch.cat((translation, gain), dim=-1)

    def encode_visual_grid(
        self,
        rgb: torch.Tensor,
        frozen_v12_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.as_tensor(rgb)
        if images.ndim != 5 or tuple(images.shape[1:]) != (
            VIEWS,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise V172ContractError("rgb must be [B,2,3,224,224]")
        batch = len(images)
        device = images.device
        native = torch.as_tensor(frozen_v12_tokens, device=device)
        if native.shape != (
            batch,
            VIEWS,
            SEMANTIC_CELLS,
            self.semantic_dim,
        ):
            raise V172ContractError(
                f"frozen_v12_tokens must be [B,2,256,{self.semantic_dim}]"
            )
        input_was_uint8 = images.dtype == torch.uint8
        images = images.float()
        if input_was_uint8:
            images = images / 255.0
        native = native.detach().float()
        _finite("rgb", images)
        _finite("frozen_v12_tokens", native)

        flat_images = images.reshape(batch * VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE)
        rgb_grid = self.rgb_stem(_normalized_rgb_with_contrast(flat_images))
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
        rgb_tokens = rgb_grid.permute(0, 2, 3, 1)
        fused = self.visual_fusion(
            torch.cat((rgb_tokens, semantic.permute(0, 2, 3, 1)), dim=-1)
        )
        fused = fused.reshape(batch, VIEWS, VISUAL_CELLS, self.dimension)
        fused = fused + self.visual_position.to(fused)[None, None]
        fused = fused + self.view_identity.to(fused)[None, :, None]
        return fused, native

    def _query_tokens(
        self,
        *,
        memory: V172FCPGMemory,
        nominal_fm_px: torch.Tensor,
        current_state: torch.Tensor,
        completed_commands: torch.Tensor,
        measured_joint_path: torch.Tensor,
        transition_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(current_state)
        dtype = current_state.dtype
        center = nominal_fm_px.mean(dim=2, keepdim=True)
        geometry = torch.cat(
            (
                nominal_fm_px / float(IMAGE_SIZE - 1),
                (nominal_fm_px - center) / 32.0,
            ),
            dim=-1,
        ).to(dtype)
        geometry_tokens = self.geometry_projection(geometry)
        geometry_tokens = geometry_tokens + self.view_identity.to(dtype)[
            None, :, None
        ]
        geometry_tokens = geometry_tokens + self.tip_identity.to(dtype)[
            None, None
        ]

        normalized_previous = self._normalized_similarity(
            memory.similarity_parameters
        )
        registration = memory.registration_query + self.previous_similarity_projection(
            normalized_previous
        )
        registration = registration + self.view_identity.to(dtype)[None]

        state_tokens = self.state_projection(current_state / 180.0)[:, None]
        command_delta = completed_commands - measured_joint_path[:, :COMMAND_STEPS]
        command_tokens = self.command_projection(
            torch.cat(
                (completed_commands / 180.0, command_delta / 180.0), dim=-1
            )
        ) + self.command_position.to(dtype)[None]
        readback_delta = measured_joint_path - measured_joint_path[:, :1]
        readback_tokens = self.readback_projection(
            torch.cat(
                (measured_joint_path / 180.0, readback_delta / 180.0), dim=-1
            )
        ) + self.readback_position.to(dtype)[None]

        query = torch.cat(
            (
                registration,
                geometry_tokens.reshape(batch, VIEWS * TIPS, self.dimension),
                state_tokens,
                command_tokens,
                readback_tokens,
            ),
            dim=1,
        )
        padding = torch.zeros(
            batch, query.shape[1], device=current_state.device, dtype=torch.bool
        )
        transition_start = VIEWS + VIEWS * TIPS + 1
        padding[:, transition_start:] = (~transition_valid)[:, None]
        return query, padding

    def _history_memory(
        self,
        previous: V172FCPGMemory,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(previous.endpoint_tokens)
        historical = self.semantic_projection(previous.endpoint_tokens.float()).to(
            dtype
        )
        historical = historical + self.semantic_position.to(dtype)[
            None, None, None
        ]
        historical = historical + self.history_time.to(dtype)[
            None, :, None, None
        ]
        historical = historical + self.view_identity.to(dtype)[
            None, None, :, None
        ]
        historical = historical.reshape(
            batch,
            HISTORY_ENDPOINTS * VIEWS * SEMANTIC_CELLS,
            self.dimension,
        )
        padding = (~previous.endpoint_valid)[:, :, :, None].expand(
            -1, -1, -1, SEMANTIC_CELLS
        ).reshape(batch, HISTORY_ENDPOINTS * VIEWS * SEMANTIC_CELLS)
        return historical, padding

    def _decoder_memory(
        self,
        *,
        fused_current: torch.Tensor,
        previous: V172FCPGMemory,
        view_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(fused_current)
        dtype, device = fused_current.dtype, fused_current.device
        current = fused_current.reshape(
            batch, VIEWS * VISUAL_CELLS, self.dimension
        )
        current_padding = (~view_valid)[:, :, None].expand(
            -1, -1, VISUAL_CELLS
        ).reshape(batch, VIEWS * VISUAL_CELLS)
        null = self.null_memory.to(dtype)[None].expand(batch, 1, -1)
        values = [null, current]
        masks = [
            torch.zeros(batch, 1, device=device, dtype=torch.bool),
            current_padding,
        ]
        if bool(previous.endpoint_valid.any()):
            history, history_padding = self._history_memory(previous, dtype=dtype)
            values.append(history)
            masks.append(history_padding)
        return torch.cat(values, dim=1), torch.cat(masks, dim=1)

    def _semantic_decoder_memory(
        self,
        *,
        current_native: torch.Tensor,
        previous: V172FCPGMemory,
        view_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(current_native)
        dtype, device = current_native.dtype, current_native.device
        current = self.semantic_projection(current_native.float()).to(dtype)
        current = current + self.semantic_position.to(dtype)[None, None]
        current = current + self.view_identity.to(dtype)[None, :, None]
        current = current.reshape(batch, VIEWS * SEMANTIC_CELLS, self.dimension)
        current_padding = (~view_valid)[:, :, None].expand(
            -1, -1, SEMANTIC_CELLS
        ).reshape(batch, VIEWS * SEMANTIC_CELLS)
        null = self.null_memory.to(dtype)[None].expand(batch, 1, -1)
        values = [null, current]
        masks = [
            torch.zeros(batch, 1, device=device, dtype=torch.bool),
            current_padding,
        ]
        if bool(previous.endpoint_valid.any()):
            history, history_padding = self._history_memory(previous, dtype=dtype)
            values.append(history)
            masks.append(history_padding)
        return torch.cat(values, dim=1), torch.cat(masks, dim=1)

    def _decode(
        self,
        query: torch.Tensor,
        decoder_memory: torch.Tensor,
        *,
        query_padding: torch.Tensor,
        memory_padding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = query
        layers = []
        for layer in self.query_layers:
            value = layer(
                value,
                decoder_memory,
                tgt_key_padding_mask=query_padding,
                memory_key_padding_mask=memory_padding,
            )
            layers.append(value[:, :VIEWS])
        value = self.query_norm(value)
        return value, torch.stack(layers, dim=1)

    def _append_endpoint(
        self,
        previous: V172FCPGMemory,
        current_native: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_tokens = torch.cat(
            (previous.endpoint_tokens[:, 1:], current_native[:, None]), dim=1
        )
        candidate_valid = torch.cat(
            (previous.endpoint_valid[:, 1:], view_valid[:, None]), dim=1
        )
        next_tokens = torch.where(
            view_valid[:, None, :, None, None],
            candidate_tokens,
            previous.endpoint_tokens,
        )
        next_valid = torch.where(
            view_valid[:, None, :], candidate_valid, previous.endpoint_valid
        )
        return next_tokens, next_valid

    def temporal_pretrain_query(
        self,
        frozen_v12_tokens: torch.Tensor,
        current_state: torch.Tensor,
        completed_commands: torch.Tensor,
        measured_joint_path: torch.Tensor,
        transition_valid: torch.Tensor,
        view_valid: torch.Tensor,
        previous_memory: V172FCPGMemory | None = None,
    ) -> V172TemporalFeatures:
        native = torch.as_tensor(frozen_v12_tokens).detach().float()
        if native.ndim != 4 or tuple(native.shape[1:]) != (
            VIEWS,
            SEMANTIC_CELLS,
            self.semantic_dim,
        ):
            raise V172ContractError(
                f"frozen_v12_tokens must be [B,2,256,{self.semantic_dim}]"
            )
        batch = len(native)
        device, dtype = native.device, native.dtype
        views = _boolean(
            "view_valid", view_valid, shape=(batch, VIEWS), device=device
        )
        transition = _boolean(
            "transition_valid", transition_valid, shape=(batch,), device=device
        )
        previous = (
            self.initial_memory(batch, device=device, dtype=dtype)
            if previous_memory is None
            else self._validate_memory(
                previous_memory, batch=batch, device=device, dtype=dtype
            )
        )
        state = torch.as_tensor(current_state, device=device, dtype=dtype)
        commands = torch.as_tensor(completed_commands, device=device, dtype=dtype)
        readback = torch.as_tensor(measured_joint_path, device=device, dtype=dtype)
        if state.shape != (batch, STATE_DIMENSION):
            raise V172ContractError("current_state must be [B,6]")
        if commands.shape != (batch, COMMAND_STEPS, STATE_DIMENSION):
            raise V172ContractError("completed_commands must be [B,5,6]")
        if readback.shape != (batch, READBACK_STEPS, STATE_DIMENSION):
            raise V172ContractError("measured_joint_path must be [B,6,6]")
        _finite("current_state", state)
        _finite("completed_commands", commands)
        _finite("measured_joint_path", readback)

        normalized_previous = self._normalized_similarity(
            previous.similarity_parameters
        )
        registration = previous.registration_query + self.previous_similarity_projection(
            normalized_previous
        )
        registration = registration + self.view_identity.to(dtype)[None]
        state_token = self.state_projection(state / 180.0)[:, None]
        command_delta = commands - readback[:, :COMMAND_STEPS]
        command_tokens = self.command_projection(
            torch.cat((commands / 180.0, command_delta / 180.0), dim=-1)
        ) + self.command_position.to(dtype)[None]
        readback_delta = readback - readback[:, :1]
        readback_tokens = self.readback_projection(
            torch.cat((readback / 180.0, readback_delta / 180.0), dim=-1)
        ) + self.readback_position.to(dtype)[None]
        query = torch.cat(
            (registration, state_token, command_tokens, readback_tokens), dim=1
        )
        query_padding = torch.zeros(
            batch, query.shape[1], device=device, dtype=torch.bool
        )
        query_padding[:, VIEWS + 1 :] = (~transition)[:, None]
        decoder_memory, memory_padding = self._semantic_decoder_memory(
            current_native=native,
            previous=previous,
            view_valid=views,
        )
        decoded, layers = self._decode(
            query,
            decoder_memory,
            query_padding=query_padding,
            memory_padding=memory_padding,
        )
        registration_next = torch.where(
            views[..., None], decoded[:, :VIEWS], previous.registration_query
        )
        endpoint, endpoint_valid = self._append_endpoint(previous, native, views)
        next_memory = V172FCPGMemory(
            registration_query=registration_next,
            similarity_parameters=previous.similarity_parameters,
            endpoint_tokens=endpoint,
            endpoint_valid=endpoint_valid,
        )
        return V172TemporalFeatures(
            registration_query=registration_next,
            layer_registration_tokens=layers,
            next_memory=next_memory,
        )

    def step(
        self,
        rgb: torch.Tensor,
        frozen_v12_tokens: torch.Tensor,
        nominal_fm_px: torch.Tensor,
        current_state: torch.Tensor,
        completed_commands: torch.Tensor,
        measured_joint_path: torch.Tensor,
        transition_valid: torch.Tensor,
        view_valid: torch.Tensor,
        previous_memory: V172FCPGMemory | None = None,
    ) -> V172Output:
        fused, native = self.encode_visual_grid(rgb, frozen_v12_tokens)
        batch = len(fused)
        device, dtype = fused.device, fused.dtype
        views = _boolean(
            "view_valid", view_valid, shape=(batch, VIEWS), device=device
        )
        transition = _boolean(
            "transition_valid", transition_valid, shape=(batch,), device=device
        )
        previous = (
            self.initial_memory(batch, device=device, dtype=dtype)
            if previous_memory is None
            else self._validate_memory(
                previous_memory, batch=batch, device=device, dtype=dtype
            )
        )
        nominal = torch.as_tensor(nominal_fm_px, device=device, dtype=dtype)
        state = torch.as_tensor(current_state, device=device, dtype=dtype)
        commands = torch.as_tensor(completed_commands, device=device, dtype=dtype)
        readback = torch.as_tensor(measured_joint_path, device=device, dtype=dtype)
        if nominal.shape != (batch, VIEWS, TIPS, 2):
            raise V172ContractError("nominal_fm_px must be [B,2,2,2]")
        if state.shape != (batch, STATE_DIMENSION):
            raise V172ContractError("current_state must be [B,6]")
        if commands.shape != (batch, COMMAND_STEPS, STATE_DIMENSION):
            raise V172ContractError("completed_commands must be [B,5,6]")
        if readback.shape != (batch, READBACK_STEPS, STATE_DIMENSION):
            raise V172ContractError("measured_joint_path must be [B,6,6]")
        for name, value in (
            ("nominal_fm_px", nominal),
            ("current_state", state),
            ("completed_commands", commands),
            ("measured_joint_path", readback),
        ):
            _finite(name, value)

        query, query_padding = self._query_tokens(
            memory=previous,
            nominal_fm_px=nominal,
            current_state=state,
            completed_commands=commands,
            measured_joint_path=readback,
            transition_valid=transition,
        )
        decoder_memory, decoder_padding = self._decoder_memory(
            fused_current=fused,
            previous=previous,
            view_valid=views,
        )
        decoded, registration_layers = self._decode(
            query,
            decoder_memory,
            query_padding=query_padding,
            memory_padding=decoder_padding,
        )
        decoded_registration = decoded[:, :VIEWS]
        proposed = self._bounded_similarity(
            self.similarity_head(decoded_registration)
        )
        next_parameters = torch.where(
            views[..., None], proposed, previous.similarity_parameters
        )
        next_registration = torch.where(
            views[..., None], decoded_registration, previous.registration_query
        )
        # Keep the low-dimensional pixel transform in float32 even when the
        # visual/Transformer path uses bf16.
        with torch.autocast(device_type=device.type, enabled=False):
            corrected = apply_fm_similarity(
                nominal.float(), next_parameters.float()
            )
        finite = torch.isfinite(corrected).all(dim=-1)
        inside = ((corrected >= 0.0) & (corrected < float(IMAGE_SIZE))).all(
            dim=-1
        )
        valid = finite & inside & views[:, :, None]
        midpoint = corrected.mean(dim=2)

        endpoint_tokens, endpoint_valid = self._append_endpoint(
            previous, native.to(dtype), views
        )
        next_memory = V172FCPGMemory(
            registration_query=next_registration,
            similarity_parameters=next_parameters,
            endpoint_tokens=endpoint_tokens,
            endpoint_valid=endpoint_valid,
        )
        _finite("fm_px", corrected)
        _finite("midpoint_px", midpoint)
        return V172Output(
            similarity_parameters=next_parameters,
            nominal_fm_px=nominal,
            fm_px=corrected,
            midpoint_px=midpoint,
            fm_valid=valid,
            registration_query=next_registration,
            layer_registration_tokens=registration_layers,
            fused_visual_tokens=fused,
            next_memory=next_memory,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        frozen_v12_tokens: torch.Tensor,
        nominal_fm_px: torch.Tensor,
        current_state: torch.Tensor,
        completed_commands: torch.Tensor,
        measured_joint_path: torch.Tensor,
        transition_valid: torch.Tensor,
        view_valid: torch.Tensor,
        previous_memory: V172FCPGMemory | None = None,
    ) -> V172Output:
        return self.step(
            rgb,
            frozen_v12_tokens,
            nominal_fm_px,
            current_state,
            completed_commands,
            measured_joint_path,
            transition_valid,
            view_valid,
            previous_memory,
        )


def v172_ordered_fm_smooth_l1(
    output: V172Output,
    target_fm_px: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    beta_px: float = 2.0,
) -> V172FMLoss:
    target = torch.as_tensor(
        target_fm_px, device=output.fm_px.device, dtype=output.fm_px.dtype
    )
    if target.shape != output.fm_px.shape:
        raise V172ContractError("target_fm_px must be [B,2,2,2]")
    valid = _boolean(
        "target_valid",
        target_valid,
        shape=tuple(output.fm_valid.shape),
        device=target.device,
    )
    _finite("target_fm_px", target)
    if not bool(valid.any()):
        zero = output.fm_px.sum() * 0.0
        return V172FMLoss(zero, valid.sum(), zero)
    coordinate_loss = F.smooth_l1_loss(
        output.fm_px, target, reduction="none", beta=float(beta_px)
    ).sum(dim=-1)
    epe = torch.linalg.vector_norm(output.fm_px - target, dim=-1)
    return V172FMLoss(
        total=coordinate_loss.masked_select(valid).mean(),
        active_points=valid.sum(),
        mean_epe_px=epe.masked_select(valid).mean(),
    )


__all__ = [
    "COMMAND_STEPS",
    "COMPLEX_GAIN_BOUND",
    "DECODER_FFN",
    "DECODER_HEADS",
    "DECODER_LAYERS",
    "FROZEN_V12_DIMENSION",
    "HISTORY_ENDPOINTS",
    "IMAGE_SIZE",
    "MODEL_DIMENSION",
    "READBACK_STEPS",
    "SEMANTIC_CELLS",
    "SIMILARITY_PARAMETERS",
    "TIP_NAMES",
    "TIPS",
    "TRANSLATION_BOUND_PX",
    "V172ContractError",
    "V172FCPGMemory",
    "V172FMLoss",
    "V172Output",
    "V172SimilarityRegistrationFCPG",
    "V172TemporalFeatures",
    "VIEW_NAMES",
    "VIEWS",
    "apply_fm_similarity",
    "v172_ordered_fm_smooth_l1",
]
