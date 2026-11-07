"""Prod-V18.1 tip-free dual-view transition-error controller.

The world coordinate system is a parameter-free 4x6 pooling of frozen
DINOv2-S/14 patch features.  A separate trainable policy adapter produces
dense tokens for the Actor.  LeWM predicts a one-chunk transition in the fixed
observer space; after execution, the next real endpoint produces a transition
error.  Predicted visual state is never written into actual-observation memory.

This module intentionally contains no pixel keypoint, mask, depth, metric pose,
language input, hindsight image, confidence gate, candidate optimizer, or
recursive visual rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Final

import torch
import torch.nn.functional as F
from torch import nn


VIEWS: Final = 2
VIEW_ORDER: Final = ("wrist", "column")
VIT_GRID: Final = 16
VIT_PATCHES: Final = VIT_GRID * VIT_GRID
VIT_DIM: Final = 384
MODEL_DIM: Final = 512
OBSERVER_ROWS: Final = 4
OBSERVER_COLUMNS: Final = 6
OBSERVER_SLOTS: Final = OBSERVER_ROWS * OBSERVER_COLUMNS
OBSERVER_DIM: Final = VIT_DIM
STATE_DIM: Final = 6
COMMAND_STEPS: Final = 5
PATH_STEPS: Final = COMMAND_STEPS + 1
HISTORY_CHUNKS: Final = 3


class V181ContractError(ValueError):
    """Raised before a malformed V18.1 tensor can enter the controller."""


@dataclass(frozen=True)
class V181Config:
    model_dim: int = MODEL_DIM
    heads: int = 8
    ffn_dim: int = 2048
    policy_layers: int = 2
    lewm_layers: int = 4
    actor_layers: int = 6
    dropout: float = 0.0

    def validate(self) -> None:
        positive = (
            self.model_dim,
            self.heads,
            self.ffn_dim,
            self.policy_layers,
            self.lewm_layers,
            self.actor_layers,
        )
        if min(positive) <= 0:
            raise V181ContractError("all V18.1 architecture dimensions must be positive")
        if self.model_dim != MODEL_DIM or self.model_dim % self.heads:
            raise V181ContractError("the V18.1 model width and head partition are fixed")
        if self.dropout != 0.0:
            raise V181ContractError("V18.1 dropout is fixed at zero")


@dataclass(frozen=True)
class V181Measurement:
    policy_dense_tokens: torch.Tensor
    local_observer_slots: torch.Tensor
    policy_by_layer: torch.Tensor


@dataclass(frozen=True)
class V181LeWMPrediction:
    predicted_slot_delta: torch.Tensor
    predicted_q_path: torch.Tensor
    tokens_by_layer: torch.Tensor


@dataclass(frozen=True)
class V181Innovation:
    visual: torch.Tensor
    state: torch.Tensor
    actual_slot_delta: torch.Tensor
    innovation_valid: torch.Tensor
    path_valid: torch.Tensor


@dataclass(frozen=True)
class V181ActorOutput:
    commands: torch.Tensor
    desired_commands: torch.Tensor
    increments: torch.Tensor
    action_queries_by_layer: torch.Tensor


def _finite(name: str, value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise V181ContractError(f"{name} must be finite floating point")
    return tensor


def _bool(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.dtype != torch.bool or tuple(tensor.shape) != shape:
        raise V181ContractError(f"{name} must be bool {list(shape)}")
    return tensor


def _shape(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    tensor = _finite(name, value)
    if tuple(tensor.shape) != shape:
        raise V181ContractError(
            f"{name} shape drifted: observed={list(tensor.shape)} expected={list(shape)}"
        )
    return tensor


def normalize_observer_slots(slots: torch.Tensor) -> torch.Tensor:
    """Use one deployment-identical norm for targets, predictions, and errors."""

    return F.layer_norm(_finite("observer_slots", slots), (OBSERVER_DIM,))


def v181_default_action_contract(
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([0.04, 0.04, 0.04, 0.04, 0.04, 0.08], device=device),
        torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, 0.0], device=device),
        torch.ones(STATE_DIM, device=device),
    )


def _action_contract(
    step_cap: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cap = _finite("step_cap", step_cap).float()
    low = _finite("action_lower", lower).float()
    high = _finite("action_upper", upper).float()
    if any(tuple(value.shape) != (STATE_DIM,) for value in (cap, low, high)):
        raise V181ContractError("action contract tensors must be [6]")
    if bool((cap <= 0).any()) or bool((low >= high).any()):
        raise V181ContractError("action contract is invalid")
    return cap, low, high


def project_v181_action_trajectory(
    current_q: torch.Tensor,
    desired_commands: torch.Tensor,
    *,
    step_cap: torch.Tensor,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
) -> torch.Tensor:
    """Sequential projection with an exact fixed-point property."""

    state = _finite("current_q", current_q).float()
    desired = _finite("desired_commands", desired_commands).to(state)
    if state.ndim != 2 or state.shape[-1] != STATE_DIM:
        raise V181ContractError("current_q must be [B,6]")
    if desired.shape != (len(state), COMMAND_STEPS, STATE_DIM):
        raise V181ContractError("desired_commands must be [B,5,6]")
    cap, low, high = _action_contract(step_cap, action_lower, action_upper)
    cap, low, high = cap.to(state), low.to(state), high.to(state)
    previous = state
    projected: list[torch.Tensor] = []
    for step in range(COMMAND_STEPS):
        feasible_low = torch.maximum(low, previous - cap)
        feasible_high = torch.minimum(high, previous + cap)
        previous = torch.maximum(
            torch.minimum(desired[:, step], feasible_high), feasible_low
        )
        projected.append(previous)
    return torch.stack(projected, dim=1)


class _PreNormEncoderLayer(nn.Module):
    def __init__(self, config: V181Config) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.model_dim)
        self.attention = nn.MultiheadAttention(
            config.model_dim,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(config.model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.model_dim, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.model_dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        tokens = tokens + self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return tokens + self.ffn(self.ffn_norm(tokens))


class _PreNormDecoderLayer(nn.Module):
    def __init__(self, config: V181Config) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(config.model_dim)
        self.self_attention = nn.MultiheadAttention(
            config.model_dim,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(config.model_dim)
        self.memory_norm = nn.LayerNorm(config.model_dim)
        self.cross_attention = nn.MultiheadAttention(
            config.model_dim,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(config.model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.model_dim, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.model_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        *,
        query_mask: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.self_norm(queries)
        queries = queries + self.self_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=query_mask,
            need_weights=False,
        )[0]
        queries = queries + self.cross_attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )[0]
        return queries + self.ffn(self.ffn_norm(queries))


class V181TokenFCPG(nn.Module):
    """Fixed world slots plus a separate trainable policy-side dense adapter."""

    def __init__(self, config: V181Config = V181Config()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.policy_input_norm = nn.LayerNorm(VIT_DIM)
        self.policy_projection = nn.Linear(VIT_DIM, config.model_dim)
        self.spatial_embedding = nn.Parameter(
            torch.empty(1, 1, VIT_PATCHES, config.model_dim)
        )
        self.camera_embedding = nn.Parameter(
            torch.empty(1, VIEWS, 1, config.model_dim)
        )
        self.policy_layers = nn.ModuleList(
            _PreNormEncoderLayer(config) for _ in range(config.policy_layers)
        )
        self.policy_output_norm = nn.LayerNorm(config.model_dim)
        self.register_buffer(
            "observer_pooling_weights",
            self._fixed_gaussian_pooling_weights(),
            persistent=True,
        )
        nn.init.normal_(self.spatial_embedding, std=0.02)
        nn.init.normal_(self.camera_embedding, std=0.02)

    @staticmethod
    def _fixed_gaussian_pooling_weights() -> torch.Tensor:
        """Twenty-four immutable local-attention masks over the 16x16 grid."""

        patch_y, patch_x = torch.meshgrid(
            torch.arange(VIT_GRID, dtype=torch.float32),
            torch.arange(VIT_GRID, dtype=torch.float32),
            indexing="ij",
        )
        centers_y = torch.linspace(1.5, 13.5, OBSERVER_ROWS)
        centers_x = torch.linspace(1.0, 14.0, OBSERVER_COLUMNS)
        weights = []
        for center_y in centers_y:
            for center_x in centers_x:
                squared = (
                    ((patch_y - center_y) / 2.0).square()
                    + ((patch_x - center_x) / 1.5).square()
                )
                support = (
                    (patch_y - center_y).abs() <= 3.5
                ) & ((patch_x - center_x).abs() <= 3.0)
                logits = (-0.5 * squared).masked_fill(~support, -torch.inf)
                weights.append(torch.softmax(logits.reshape(-1), dim=0))
        return torch.stack(weights, dim=0)

    def fixed_local_slots(self, vit_tokens: torch.Tensor) -> torch.Tensor:
        """Parameter-free spatial identity with smooth, strictly local support."""

        normalized = normalize_observer_slots(vit_tokens)
        weights = self.observer_pooling_weights.to(
            device=normalized.device, dtype=normalized.dtype
        )
        return torch.einsum("sp,bvpd->bvsd", weights, normalized)

    def forward(
        self,
        vit_tokens: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> V181Measurement:
        tokens = _finite("vit_tokens", vit_tokens)
        batch = len(tokens)
        if tokens.shape != (batch, VIEWS, VIT_PATCHES, VIT_DIM):
            raise V181ContractError("vit_tokens must be [B,2,256,384]")
        valid = _bool("view_valid", view_valid, (batch, VIEWS)).to(tokens.device)
        slots = self.fixed_local_slots(tokens) * valid[:, :, None, None]

        # Cached DINO features are fp16.  CUDA autocast handles mixed module
        # dtypes during training; plain evaluation/runtime calls do not.
        policy_tokens = tokens
        if not torch.is_autocast_enabled():
            policy_tokens = tokens.to(dtype=self.policy_input_norm.weight.dtype)
        dense = self.policy_projection(self.policy_input_norm(policy_tokens))
        dense = dense + self.spatial_embedding + self.camera_embedding
        dense = dense.reshape(batch * VIEWS, VIT_PATCHES, self.config.model_dim)
        layers = [dense.reshape(batch, VIEWS, VIT_PATCHES, self.config.model_dim)]
        for layer in self.policy_layers:
            dense = layer(dense)
            layers.append(
                dense.reshape(batch, VIEWS, VIT_PATCHES, self.config.model_dim)
            )
        dense = self.policy_output_norm(dense).reshape(
            batch, VIEWS, VIT_PATCHES, self.config.model_dim
        )
        dense = dense * valid[:, :, None, None]
        if not bool(torch.isfinite(dense).all() and torch.isfinite(slots).all()):
            raise V181ContractError("Token-FCPG produced non-finite values")
        return V181Measurement(
            policy_dense_tokens=dense,
            local_observer_slots=slots,
            policy_by_layer=torch.stack(layers, dim=1),
        )


class V181LeWM(nn.Module):
    """Predict one chunk's fixed-space visual transition and measured q path."""

    def __init__(self, config: V181Config = V181Config()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.slot_projection = nn.Linear(OBSERVER_DIM, config.model_dim)
        self.camera_embedding = nn.Parameter(
            torch.empty(1, VIEWS, 1, config.model_dim)
        )
        self.slot_embedding = nn.Parameter(
            torch.empty(1, 1, OBSERVER_SLOTS, config.model_dim)
        )
        self.state_projection = nn.Sequential(
            nn.Linear(STATE_DIM, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.command_projection = nn.Sequential(
            nn.Linear(STATE_DIM, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.motion_time_embedding = nn.Parameter(
            torch.empty(1, PATH_STEPS, config.model_dim)
        )
        self.layers = nn.ModuleList(
            _PreNormEncoderLayer(config) for _ in range(config.lewm_layers)
        )
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.slot_delta_head = nn.Linear(config.model_dim, OBSERVER_DIM)
        self.q_residual_head = nn.Linear(config.model_dim, STATE_DIM)
        nn.init.normal_(self.camera_embedding, std=0.02)
        nn.init.normal_(self.slot_embedding, std=0.02)
        nn.init.normal_(self.motion_time_embedding, std=0.02)
        nn.init.zeros_(self.slot_delta_head.weight)
        nn.init.zeros_(self.slot_delta_head.bias)
        nn.init.zeros_(self.q_residual_head.weight)
        nn.init.zeros_(self.q_residual_head.bias)

    def forward(
        self,
        local_observer_slots: torch.Tensor,
        view_valid: torch.Tensor,
        current_q: torch.Tensor,
        commands: torch.Tensor,
    ) -> V181LeWMPrediction:
        slots = _finite("local_observer_slots", local_observer_slots)
        batch = len(slots)
        if slots.shape != (batch, VIEWS, OBSERVER_SLOTS, OBSERVER_DIM):
            raise V181ContractError("local_observer_slots must be [B,2,24,384]")
        valid = _bool("view_valid", view_valid, (batch, VIEWS)).to(slots.device)
        compute_slots = slots
        if not torch.is_autocast_enabled():
            compute_slots = slots.to(dtype=self.slot_projection.weight.dtype)
        state = _shape("current_q", current_q, (batch, STATE_DIM)).to(
            device=slots.device, dtype=compute_slots.dtype
        )
        actions = _shape(
            "commands", commands, (batch, COMMAND_STEPS, STATE_DIM)
        ).to(device=slots.device, dtype=compute_slots.dtype)

        visual = self.slot_projection(normalize_observer_slots(compute_slots))
        visual = visual + self.camera_embedding + self.slot_embedding
        visual = visual.reshape(batch, VIEWS * OBSERVER_SLOTS, self.config.model_dim)
        motion = torch.cat(
            (self.state_projection(state)[:, None], self.command_projection(actions)),
            dim=1,
        )
        motion = motion + self.motion_time_embedding
        sequence = torch.cat((visual, motion), dim=1)
        visual_padding = (~valid)[:, :, None].expand(
            -1, -1, OBSERVER_SLOTS
        ).reshape(batch, VIEWS * OBSERVER_SLOTS)
        padding = torch.cat(
            (
                visual_padding,
                torch.zeros(batch, PATH_STEPS, dtype=torch.bool, device=slots.device),
            ),
            dim=1,
        )
        layers = [sequence]
        for layer in self.layers:
            sequence = layer(sequence, key_padding_mask=padding)
            layers.append(sequence)
        sequence = self.output_norm(sequence)
        visual_output = sequence[:, : VIEWS * OBSERVER_SLOTS].reshape(
            batch, VIEWS, OBSERVER_SLOTS, self.config.model_dim
        )
        predicted_delta = self.slot_delta_head(visual_output)
        predicted_delta = predicted_delta * valid[:, :, None, None]
        motion_output = sequence[:, VIEWS * OBSERVER_SLOTS :]
        baseline_path = torch.cat((state[:, None], actions), dim=1)
        predicted_path = baseline_path + self.q_residual_head(motion_output)
        predicted_path = torch.cat((state[:, None], predicted_path[:, 1:]), dim=1)
        if not bool(torch.isfinite(predicted_delta).all() and torch.isfinite(predicted_path).all()):
            raise V181ContractError("LeWM produced non-finite predictions")
        return V181LeWMPrediction(
            predicted_slot_delta=predicted_delta,
            predicted_q_path=predicted_path,
            tokens_by_layer=torch.stack(layers, dim=1),
        )


def predicted_observer_slots(
    source_slots: torch.Tensor,
    prediction: V181LeWMPrediction,
) -> torch.Tensor:
    source = normalize_observer_slots(source_slots)
    delta = _shape(
        "prediction.predicted_slot_delta",
        prediction.predicted_slot_delta,
        tuple(source.shape),
    ).to(source)
    return source + delta


def compute_v181_innovation(
    source_slots: torch.Tensor,
    source_view_valid: torch.Tensor,
    actual_target_slots: torch.Tensor,
    target_view_valid: torch.Tensor,
    measured_q_path: torch.Tensor,
    prediction: V181LeWMPrediction,
    *,
    path_valid: torch.Tensor | None = None,
) -> V181Innovation:
    source = _finite("source_slots", source_slots)
    batch = len(source)
    expected = (batch, VIEWS, OBSERVER_SLOTS, OBSERVER_DIM)
    if source.shape != expected:
        raise V181ContractError("source_slots must be [B,2,24,384]")
    target = _shape("actual_target_slots", actual_target_slots, expected).to(source)
    source_valid = _bool(
        "source_view_valid", source_view_valid, (batch, VIEWS)
    ).to(source.device)
    target_valid = _bool(
        "target_view_valid", target_view_valid, (batch, VIEWS)
    ).to(source.device)
    innovation_valid = source_valid & target_valid
    path = _shape(
        "measured_q_path", measured_q_path, (batch, PATH_STEPS, STATE_DIM)
    ).to(source)
    if path_valid is None:
        valid_path = torch.ones(
            batch, PATH_STEPS, dtype=torch.bool, device=source.device
        )
    else:
        valid_path = _bool(
            "path_valid", path_valid, (batch, PATH_STEPS)
        ).to(source.device)
    predicted_delta = _shape(
        "prediction.predicted_slot_delta",
        prediction.predicted_slot_delta,
        expected,
    ).to(source)
    predicted_path = _shape(
        "prediction.predicted_q_path",
        prediction.predicted_q_path,
        (batch, PATH_STEPS, STATE_DIM),
    ).to(source)
    actual_delta = normalize_observer_slots(target) - normalize_observer_slots(source)
    actual_delta = actual_delta * innovation_valid[:, :, None, None]
    visual = (actual_delta - predicted_delta) * innovation_valid[:, :, None, None]
    state = (path - predicted_path) * valid_path[:, :, None]
    return V181Innovation(
        visual=visual,
        state=state,
        actual_slot_delta=actual_delta,
        innovation_valid=innovation_valid,
        path_valid=valid_path,
    )


@dataclass(frozen=True)
class V181Memory:
    """Three completed transitions containing only actual observations."""

    actual_observer_slots: torch.Tensor
    actual_view_valid: torch.Tensor
    completed_commands: torch.Tensor
    measured_paths: torch.Tensor
    visual_innovations: torch.Tensor
    state_innovations: torch.Tensor
    innovation_valid: torch.Tensor
    path_valid: torch.Tensor
    delta_t_s: torch.Tensor
    history_valid: torch.Tensor

    @classmethod
    def empty(
        cls,
        batch: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> "V181Memory":
        if batch <= 0:
            raise V181ContractError("memory batch must be positive")
        return cls(
            actual_observer_slots=torch.zeros(
                batch, HISTORY_CHUNKS, VIEWS, OBSERVER_SLOTS, OBSERVER_DIM,
                device=device, dtype=dtype,
            ),
            actual_view_valid=torch.zeros(
                batch, HISTORY_CHUNKS, VIEWS, device=device, dtype=torch.bool
            ),
            completed_commands=torch.zeros(
                batch, HISTORY_CHUNKS, COMMAND_STEPS, STATE_DIM,
                device=device, dtype=dtype,
            ),
            measured_paths=torch.zeros(
                batch, HISTORY_CHUNKS, PATH_STEPS, STATE_DIM,
                device=device, dtype=dtype,
            ),
            visual_innovations=torch.zeros(
                batch, HISTORY_CHUNKS, VIEWS, OBSERVER_SLOTS, OBSERVER_DIM,
                device=device, dtype=dtype,
            ),
            state_innovations=torch.zeros(
                batch, HISTORY_CHUNKS, PATH_STEPS, STATE_DIM,
                device=device, dtype=dtype,
            ),
            innovation_valid=torch.zeros(
                batch, HISTORY_CHUNKS, VIEWS, device=device, dtype=torch.bool
            ),
            path_valid=torch.zeros(
                batch, HISTORY_CHUNKS, PATH_STEPS, device=device, dtype=torch.bool
            ),
            delta_t_s=torch.zeros(
                batch, HISTORY_CHUNKS, device=device, dtype=dtype
            ),
            history_valid=torch.zeros(
                batch, HISTORY_CHUNKS, device=device, dtype=torch.bool
            ),
        )

    @property
    def batch_size(self) -> int:
        return int(self.history_valid.shape[0])

    def validate(self) -> None:
        batch = self.batch_size
        floating = {
            "actual_observer_slots": (batch, HISTORY_CHUNKS, VIEWS, OBSERVER_SLOTS, OBSERVER_DIM),
            "completed_commands": (batch, HISTORY_CHUNKS, COMMAND_STEPS, STATE_DIM),
            "measured_paths": (batch, HISTORY_CHUNKS, PATH_STEPS, STATE_DIM),
            "visual_innovations": (batch, HISTORY_CHUNKS, VIEWS, OBSERVER_SLOTS, OBSERVER_DIM),
            "state_innovations": (batch, HISTORY_CHUNKS, PATH_STEPS, STATE_DIM),
            "delta_t_s": (batch, HISTORY_CHUNKS),
        }
        boolean = {
            "actual_view_valid": (batch, HISTORY_CHUNKS, VIEWS),
            "innovation_valid": (batch, HISTORY_CHUNKS, VIEWS),
            "path_valid": (batch, HISTORY_CHUNKS, PATH_STEPS),
            "history_valid": (batch, HISTORY_CHUNKS),
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in floating:
                _shape(field.name, value, floating[field.name])
            else:
                _bool(field.name, value, boolean[field.name])
        if bool((self.history_valid[:, 1:] & ~self.history_valid[:, :-1]).any()):
            raise V181ContractError("history_valid must be a left-aligned prefix")
        if bool((self.innovation_valid & ~self.actual_view_valid).any()):
            raise V181ContractError("innovation cannot be valid without a source view")
        if bool((self.path_valid & ~self.history_valid[:, :, None]).any()):
            raise V181ContractError("path_valid cannot leave valid history")
        active_view = self.actual_view_valid & self.history_valid[:, :, None]
        if bool(
            (self.actual_observer_slots * (~active_view)[..., None, None]).abs().any()
            or (self.visual_innovations * (~self.innovation_valid)[..., None, None]).abs().any()
        ):
            raise V181ContractError("invalid visual memory entries must be zero")
        inactive = ~self.history_valid
        numeric = (
            self.completed_commands.abs().sum(dim=(-1, -2))
            + self.measured_paths.abs().sum(dim=(-1, -2))
            + self.state_innovations.abs().sum(dim=(-1, -2))
            + self.delta_t_s.abs()
        )
        if bool((numeric * inactive).any()) or bool((self.delta_t_s < 0).any()):
            raise V181ContractError("inactive history must be zero and delta_t nonnegative")

    def prefix(self, length: int) -> "V181Memory":
        if not 0 <= int(length) <= HISTORY_CHUNKS:
            raise V181ContractError("history prefix must be in [0,3]")
        keep = torch.arange(HISTORY_CHUNKS, device=self.history_valid.device) < int(length)
        history = self.history_valid & keep[None]
        views = self.actual_view_valid & history[:, :, None]
        innovation = self.innovation_valid & history[:, :, None]
        path = self.path_valid & history[:, :, None]
        return V181Memory(
            actual_observer_slots=self.actual_observer_slots * views[..., None, None],
            actual_view_valid=views,
            completed_commands=self.completed_commands * history[..., None, None],
            measured_paths=self.measured_paths * history[..., None, None],
            visual_innovations=self.visual_innovations * innovation[..., None, None],
            state_innovations=self.state_innovations * path[..., None],
            innovation_valid=innovation,
            path_valid=path,
            delta_t_s=self.delta_t_s * history,
            history_valid=history,
        )

    def append(
        self,
        *,
        source_actual_slots: torch.Tensor,
        source_view_valid: torch.Tensor,
        completed_commands: torch.Tensor,
        measured_q_path: torch.Tensor,
        realized_innovation: V181Innovation,
        delta_t_s: torch.Tensor,
    ) -> "V181Memory":
        """Append after target reobservation; batch rows are completed together."""

        self.validate()
        batch = self.batch_size
        slots = _shape(
            "source_actual_slots", source_actual_slots,
            (batch, VIEWS, OBSERVER_SLOTS, OBSERVER_DIM),
        ).to(self.actual_observer_slots)
        views = _bool("source_view_valid", source_view_valid, (batch, VIEWS)).to(
            self.actual_view_valid.device
        )
        commands = _shape(
            "completed_commands", completed_commands,
            (batch, COMMAND_STEPS, STATE_DIM),
        ).to(self.completed_commands)
        path = _shape(
            "measured_q_path", measured_q_path,
            (batch, PATH_STEPS, STATE_DIM),
        ).to(self.measured_paths)
        visual = _shape(
            "realized_innovation.visual", realized_innovation.visual,
            (batch, VIEWS, OBSERVER_SLOTS, OBSERVER_DIM),
        ).to(self.visual_innovations)
        state = _shape(
            "realized_innovation.state", realized_innovation.state,
            (batch, PATH_STEPS, STATE_DIM),
        ).to(self.state_innovations)
        innovation_valid = _bool(
            "realized_innovation.innovation_valid",
            realized_innovation.innovation_valid,
            (batch, VIEWS),
        ).to(self.innovation_valid.device)
        path_valid = _bool(
            "realized_innovation.path_valid",
            realized_innovation.path_valid,
            (batch, PATH_STEPS),
        ).to(self.path_valid.device)
        elapsed = _shape("delta_t_s", delta_t_s, (batch,)).to(self.delta_t_s)
        if bool((elapsed <= 0).any()):
            raise V181ContractError("completed transition delta_t must be positive")
        slots = slots * views[..., None, None]
        visual = visual * innovation_valid[..., None, None]
        state = state * path_valid[..., None]

        def append_value(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
            output = old.new_zeros(old.shape)
            for row in range(batch):
                count = int(self.history_valid[row].sum())
                keep = min(count, HISTORY_CHUNKS - 1)
                if keep:
                    output[row, :keep] = old[row, count - keep : count]
                output[row, keep] = new[row]
            return output

        count_after = torch.minimum(
            self.history_valid.sum(dim=1) + 1,
            torch.tensor(HISTORY_CHUNKS, device=self.history_valid.device),
        )
        history = torch.arange(
            HISTORY_CHUNKS, device=self.history_valid.device
        )[None] < count_after[:, None]
        result = V181Memory(
            actual_observer_slots=append_value(self.actual_observer_slots, slots),
            actual_view_valid=append_value(self.actual_view_valid, views),
            completed_commands=append_value(self.completed_commands, commands),
            measured_paths=append_value(self.measured_paths, path),
            visual_innovations=append_value(self.visual_innovations, visual),
            state_innovations=append_value(self.state_innovations, state),
            innovation_valid=append_value(self.innovation_valid, innovation_valid),
            path_valid=append_value(self.path_valid, path_valid),
            delta_t_s=append_value(self.delta_t_s, elapsed),
            history_valid=history,
        )
        result.validate()
        return result


class V181Actor(nn.Module):
    """Deterministic five-query Actor with late dual-view fusion."""

    def __init__(
        self,
        config: V181Config = V181Config(),
        *,
        step_cap: torch.Tensor | None = None,
        action_lower: torch.Tensor | None = None,
        action_upper: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        if step_cap is None or action_lower is None or action_upper is None:
            default_cap, default_low, default_high = v181_default_action_contract()
            step_cap = default_cap if step_cap is None else step_cap
            action_lower = default_low if action_lower is None else action_lower
            action_upper = default_high if action_upper is None else action_upper
        cap, low, high = _action_contract(step_cap, action_lower, action_upper)
        self.register_buffer("step_cap", cap)
        self.register_buffer("action_lower", low)
        self.register_buffer("action_upper", high)

        self.action_queries = nn.Parameter(
            torch.empty(1, COMMAND_STEPS, config.model_dim)
        )
        self.action_time_embedding = nn.Parameter(
            torch.empty(1, COMMAND_STEPS, config.model_dim)
        )
        self.camera_embedding = nn.Parameter(
            torch.empty(1, VIEWS, 1, config.model_dim)
        )
        self.policy_spatial_embedding = nn.Parameter(
            torch.empty(1, 1, VIT_PATCHES, config.model_dim)
        )
        self.slot_embedding = nn.Parameter(
            torch.empty(1, 1, OBSERVER_SLOTS, config.model_dim)
        )
        self.history_embedding = nn.Parameter(
            torch.empty(1, HISTORY_CHUNKS, 1, 1, config.model_dim)
        )
        # policy dense, current actual, current error, predicted missing,
        # history actual, history error, history event, state error, q.
        self.source_embedding = nn.Parameter(torch.empty(9, config.model_dim))
        self.actual_projection = nn.Linear(OBSERVER_DIM, config.model_dim)
        self.innovation_projection = nn.Linear(OBSERVER_DIM, config.model_dim)
        self.predicted_projection = nn.Linear(OBSERVER_DIM, config.model_dim)
        self.q_projection = nn.Sequential(
            nn.Linear(STATE_DIM, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.state_innovation_projection = nn.Sequential(
            nn.Linear(PATH_STEPS * STATE_DIM + PATH_STEPS, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.model_dim),
        )
        event_dim = (
            COMMAND_STEPS * STATE_DIM
            + PATH_STEPS * STATE_DIM
            + PATH_STEPS * STATE_DIM
            + VIEWS
            + VIEWS
            + PATH_STEPS
            + 1
        )
        self.event_projection = nn.Sequential(
            nn.Linear(event_dim, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.model_dim),
        )
        self.layers = nn.ModuleList(
            _PreNormDecoderLayer(config) for _ in range(config.actor_layers)
        )
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.action_head = nn.Linear(config.model_dim, STATE_DIM)
        for parameter in (
            self.action_queries,
            self.action_time_embedding,
            self.camera_embedding,
            self.policy_spatial_embedding,
            self.slot_embedding,
            self.history_embedding,
            self.source_embedding,
        ):
            nn.init.normal_(parameter, std=0.02)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def _slot_tokens(
        self,
        value: torch.Tensor,
        valid: torch.Tensor,
        *,
        projection: nn.Module,
        source: int,
        history_time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(value)
        tokens = projection(value) + self.camera_embedding + self.slot_embedding
        tokens = tokens + self.source_embedding[source]
        if history_time is not None:
            tokens = tokens + history_time
        tokens = tokens.reshape(batch, VIEWS * OBSERVER_SLOTS, self.config.model_dim)
        padding = (~valid)[:, :, None].expand(-1, -1, OBSERVER_SLOTS).reshape(
            batch, VIEWS * OBSERVER_SLOTS
        )
        return tokens, padding

    def forward(
        self,
        policy_dense_tokens: torch.Tensor,
        current_actual_slots: torch.Tensor,
        current_view_valid: torch.Tensor,
        visual_innovation: torch.Tensor,
        state_innovation: torch.Tensor,
        innovation_valid: torch.Tensor,
        path_valid: torch.Tensor,
        predicted_missing_slots: torch.Tensor,
        prediction_valid: torch.Tensor,
        current_q: torch.Tensor,
        memory: V181Memory,
        *,
        history_prefix: int = HISTORY_CHUNKS,
    ) -> V181ActorOutput:
        dense = _finite("policy_dense_tokens", policy_dense_tokens)
        batch = len(dense)
        if dense.shape != (batch, VIEWS, VIT_PATCHES, self.config.model_dim):
            raise V181ContractError("policy_dense_tokens must be [B,2,256,512]")
        slot_shape = (batch, VIEWS, OBSERVER_SLOTS, OBSERVER_DIM)
        actual = _shape("current_actual_slots", current_actual_slots, slot_shape).to(dense)
        visual_error = _shape("visual_innovation", visual_innovation, slot_shape).to(dense)
        predicted = _shape("predicted_missing_slots", predicted_missing_slots, slot_shape).to(dense)
        state_error = _shape(
            "state_innovation", state_innovation,
            (batch, PATH_STEPS, STATE_DIM),
        ).to(dense)
        current_valid = _bool(
            "current_view_valid", current_view_valid, (batch, VIEWS)
        ).to(dense.device)
        error_valid = _bool(
            "innovation_valid", innovation_valid, (batch, VIEWS)
        ).to(dense.device)
        valid_path = _bool("path_valid", path_valid, (batch, PATH_STEPS)).to(dense.device)
        predicted_valid = _bool(
            "prediction_valid", prediction_valid, (batch, VIEWS)
        ).to(dense.device)
        if bool((predicted_valid & current_valid).any()):
            raise V181ContractError("predicted-source slots are only for missing views")
        state = _shape("current_q", current_q, (batch, STATE_DIM)).to(dense)
        memory.validate()
        if memory.batch_size != batch:
            raise V181ContractError("memory batch differs from current endpoint")
        memory = memory.prefix(history_prefix)

        token_groups: list[torch.Tensor] = []
        padding_groups: list[torch.Tensor] = []
        policy = dense + self.camera_embedding + self.policy_spatial_embedding
        policy = policy + self.source_embedding[0]
        token_groups.append(policy.reshape(batch, VIEWS * VIT_PATCHES, -1))
        padding_groups.append(
            (~current_valid)[:, :, None].expand(-1, -1, VIT_PATCHES).reshape(
                batch, VIEWS * VIT_PATCHES
            )
        )
        for value, valid, projection, source in (
            (actual, current_valid, self.actual_projection, 1),
            (visual_error, error_valid, self.innovation_projection, 2),
            (predicted, predicted_valid, self.predicted_projection, 3),
        ):
            tokens, padding = self._slot_tokens(
                value, valid, projection=projection, source=source
            )
            token_groups.append(tokens)
            padding_groups.append(padding)

        for history in range(HISTORY_CHUNKS):
            history_valid = memory.history_valid[:, history]
            actual_valid = memory.actual_view_valid[:, history] & history_valid[:, None]
            error_history_valid = memory.innovation_valid[:, history] & history_valid[:, None]
            time = self.history_embedding[:, history]
            for value, valid, projection, source in (
                (
                    memory.actual_observer_slots[:, history],
                    actual_valid,
                    self.actual_projection,
                    4,
                ),
                (
                    memory.visual_innovations[:, history],
                    error_history_valid,
                    self.innovation_projection,
                    5,
                ),
            ):
                tokens, padding = self._slot_tokens(
                    value,
                    valid,
                    projection=projection,
                    source=source,
                    history_time=time,
                )
                token_groups.append(tokens)
                padding_groups.append(padding)
            event = torch.cat(
                (
                    memory.completed_commands[:, history].reshape(batch, -1),
                    memory.measured_paths[:, history].reshape(batch, -1),
                    memory.state_innovations[:, history].reshape(batch, -1),
                    actual_valid.to(dense.dtype),
                    error_history_valid.to(dense.dtype),
                    memory.path_valid[:, history].to(dense.dtype),
                    memory.delta_t_s[:, history, None],
                ),
                dim=-1,
            )
            event_token = self.event_projection(event)[:, None]
            event_token = event_token + self.source_embedding[6] + time[:, :, 0]
            token_groups.append(event_token)
            padding_groups.append((~history_valid)[:, None])

        state_event = torch.cat(
            (state_error.reshape(batch, -1), valid_path.to(dense.dtype)), dim=-1
        )
        token_groups.append(
            self.state_innovation_projection(state_event)[:, None]
            + self.source_embedding[7]
        )
        padding_groups.append((~valid_path.any(dim=1))[:, None])
        token_groups.append(self.q_projection(state)[:, None] + self.source_embedding[8])
        padding_groups.append(
            torch.zeros(batch, 1, dtype=torch.bool, device=dense.device)
        )
        actor_memory = torch.cat(token_groups, dim=1)
        memory_padding = torch.cat(padding_groups, dim=1)

        queries = self.action_queries.expand(batch, -1, -1) + self.action_time_embedding
        causal = torch.triu(
            torch.ones(
                COMMAND_STEPS, COMMAND_STEPS, dtype=torch.bool, device=dense.device
            ),
            diagonal=1,
        )
        layers = [queries]
        for layer in self.layers:
            queries = layer(
                queries,
                actor_memory,
                query_mask=causal,
                memory_padding_mask=memory_padding,
            )
            layers.append(queries)
        queries = self.output_norm(queries)
        increments = torch.tanh(self.action_head(queries)) * self.step_cap
        desired_rows: list[torch.Tensor] = []
        previous = state
        for step in range(COMMAND_STEPS):
            previous = previous + increments[:, step]
            desired_rows.append(previous)
        desired = torch.stack(desired_rows, dim=1)
        commands = project_v181_action_trajectory(
            state,
            desired,
            step_cap=self.step_cap,
            action_lower=self.action_lower,
            action_upper=self.action_upper,
        )
        return V181ActorOutput(
            commands=commands,
            desired_commands=desired,
            increments=increments,
            action_queries_by_layer=torch.stack(layers, dim=1),
        )


class V181TipFreeController(nn.Module):
    """Trainable V18.1 modules; the frozen image backbone remains external."""

    def __init__(self, config: V181Config = V181Config()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.observer = V181TokenFCPG(config)
        self.lewm = V181LeWM(config)
        self.actor = V181Actor(config)

    def measure(
        self, vit_tokens: torch.Tensor, view_valid: torch.Tensor
    ) -> V181Measurement:
        return self.observer(vit_tokens, view_valid)

    def predict(
        self,
        local_observer_slots: torch.Tensor,
        view_valid: torch.Tensor,
        current_q: torch.Tensor,
        commands: torch.Tensor,
    ) -> V181LeWMPrediction:
        return self.lewm(local_observer_slots, view_valid, current_q, commands)

    def act(
        self,
        measurement: V181Measurement,
        view_valid: torch.Tensor,
        innovation: V181Innovation,
        predicted_missing_slots: torch.Tensor,
        prediction_valid: torch.Tensor,
        current_q: torch.Tensor,
        memory: V181Memory,
        *,
        history_prefix: int = HISTORY_CHUNKS,
    ) -> V181ActorOutput:
        return self.actor(
            measurement.policy_dense_tokens,
            measurement.local_observer_slots,
            view_valid,
            innovation.visual,
            innovation.state,
            innovation.innovation_valid,
            innovation.path_valid,
            predicted_missing_slots,
            prediction_valid,
            current_q,
            memory,
            history_prefix=history_prefix,
        )


def empty_v181_innovation(
    batch: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> V181Innovation:
    return V181Innovation(
        visual=torch.zeros(
            batch, VIEWS, OBSERVER_SLOTS, OBSERVER_DIM, device=device, dtype=dtype
        ),
        state=torch.zeros(
            batch, PATH_STEPS, STATE_DIM, device=device, dtype=dtype
        ),
        actual_slot_delta=torch.zeros(
            batch, VIEWS, OBSERVER_SLOTS, OBSERVER_DIM, device=device, dtype=dtype
        ),
        innovation_valid=torch.zeros(
            batch, VIEWS, device=device, dtype=torch.bool
        ),
        path_valid=torch.zeros(
            batch, PATH_STEPS, device=device, dtype=torch.bool
        ),
    )


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


__all__ = [
    "COMMAND_STEPS",
    "HISTORY_CHUNKS",
    "MODEL_DIM",
    "OBSERVER_DIM",
    "OBSERVER_SLOTS",
    "PATH_STEPS",
    "STATE_DIM",
    "VIEW_ORDER",
    "VIEWS",
    "VIT_DIM",
    "VIT_PATCHES",
    "V181Actor",
    "V181ActorOutput",
    "V181Config",
    "V181ContractError",
    "V181Innovation",
    "V181LeWM",
    "V181LeWMPrediction",
    "V181Measurement",
    "V181Memory",
    "V181TipFreeController",
    "V181TokenFCPG",
    "compute_v181_innovation",
    "empty_v181_innovation",
    "normalize_observer_slots",
    "predicted_observer_slots",
    "project_v181_action_trajectory",
    "trainable_parameter_count",
    "v181_default_action_contract",
]
