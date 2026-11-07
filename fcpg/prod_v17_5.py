"""Structured, morphology-conditioned FCPG/LeWM/Actor stack for Prod-V17.5.

V17.5 treats kinematics as a proposal and the current image as a measurement.
All public visual anchors are generated from one articulated robot state and a
shared per-view registration; there is no independent fingertip pixel head.
Coordinates are supervision/probe outputs. LeWM and Actor consume ordered
posterior-pooled part tokens rather than hard XY coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Final

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fcpg.robot_kinematic_graph import (
    DifferentiableRobotKinematics,
    RobotSpec,
    project_reference_points,
    so101_robot_spec,
)


VIEWS: Final = 2
VIEW_ORDER: Final = ("wrist", "column")
ANCHORS: Final = 6
ANCHOR_ORDER: Final = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_flex_joint",
    "wrist_flex_joint",
    "fixed_finger_tip",
    "moving_finger_tip",
)
TIP_INDICES: Final = (4, 5)
IMAGE_SIZE: Final = 224
FEATURE_GRID: Final = 56
COARSE_CELLS: Final = FEATURE_GRID * FEATURE_GRID
LOCAL_GRID: Final = 9
LOCAL_CELLS: Final = LOCAL_GRID * LOCAL_GRID
VIT_GRID: Final = 16
VIT_PATCHES: Final = VIT_GRID * VIT_GRID
VIT_DIM: Final = 384
ANCHOR_DIM: Final = 384
ACTOR_DIM: Final = 512
STATE_DIM: Final = 6
COMMAND_STEPS: Final = 5
PATH_STEPS: Final = COMMAND_STEPS + 1
HISTORY_CHUNKS: Final = 4
COMPANION_STORAGE_SCALE: Final = 100.0
JOINT_POSITION_UNIT: Final = "lerobot_calibrated_position"


class V175ContractError(ValueError):
    """Raised before a malformed V17.5 tensor enters the model."""


def restore_companion_joint_units(value: torch.Tensor | np.ndarray):
    """Undo the companion builder's explicit ``/100`` storage transform."""

    if isinstance(value, np.ndarray):
        result = np.asarray(value, dtype=np.float32) * COMPANION_STORAGE_SCALE
        if not np.isfinite(result).all():
            raise V175ContractError("companion joint values must be finite")
        return result
    tensor = torch.as_tensor(value)
    if not bool(torch.isfinite(tensor).all()):
        raise V175ContractError("companion joint values must be finite")
    return tensor * COMPANION_STORAGE_SCALE


def v175_action_contract(
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SO-101 command bounds in calibrated-position units."""

    return (
        torch.tensor((4.0, 4.0, 4.0, 4.0, 4.0, 8.0), device=device),
        torch.tensor((-100.0, -100.0, -100.0, -100.0, -100.0, 0.0), device=device),
        torch.full((STATE_DIM,), 100.0, device=device),
    )


def project_v175_action_trajectory(
    current_state: torch.Tensor,
    desired_trajectory: torch.Tensor,
    *,
    step_cap: torch.Tensor,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
) -> torch.Tensor:
    """Sequentially enforce bounds without snapping an overshot joint."""

    state = _finite("current_state", current_state)
    desired = _finite("desired_trajectory", desired_trajectory).to(state)
    batch = len(state)
    _shape("current_state", state, (batch, STATE_DIM))
    _shape("desired_trajectory", desired, (batch, COMMAND_STEPS, STATE_DIM))
    cap = _shape("step_cap", step_cap, (STATE_DIM,)).to(state)
    lower = _shape("action_lower", action_lower, (STATE_DIM,)).to(state)
    upper = _shape("action_upper", action_upper, (STATE_DIM,)).to(state)
    if bool((cap <= 0).any()) or bool((lower >= upper).any()):
        raise V175ContractError("action projection contract is invalid")
    current = state
    rows: list[torch.Tensor] = []
    for step in range(COMMAND_STEPS):
        bounded = torch.maximum(torch.minimum(desired[:, step], upper), lower)
        current = current + torch.clamp(bounded - current, min=-cap, max=cap)
        rows.append(current)
    return torch.stack(rows, dim=1)


@dataclass(frozen=True)
class V175Config:
    anchor_dim: int = ANCHOR_DIM
    actor_dim: int = ACTOR_DIM
    heads: int = 8
    anchor_ffn_dim: int = 1536
    actor_ffn_dim: int = 2048
    measurement_layers: int = 4
    motion_layers: int = 2
    lewm_layers: int = 4
    actor_layers: int = 6
    dropout: float = 0.0
    max_joint_correction_fraction: float = 0.05
    max_registration_translation_px: float = 80.0
    max_registration_rotation_rad: float = math.radians(25.0)
    max_registration_log_scale: float = math.log(1.5)
    local_radius_px: float = 16.0

    def validate(self) -> None:
        integers = (
            self.anchor_dim,
            self.actor_dim,
            self.heads,
            self.anchor_ffn_dim,
            self.actor_ffn_dim,
            self.measurement_layers,
            self.motion_layers,
            self.lewm_layers,
            self.actor_layers,
        )
        if min(integers) <= 0:
            raise V175ContractError("all V17.5 dimensions must be positive")
        if self.anchor_dim != ANCHOR_DIM or self.actor_dim != ACTOR_DIM:
            raise V175ContractError("V17.5 model widths are fixed")
        if self.anchor_dim % self.heads or self.actor_dim % self.heads:
            raise V175ContractError("attention heads must divide both widths")
        if self.dropout != 0.0:
            raise V175ContractError("V17.5 dropout is fixed at zero")
        bounds = (
            self.max_joint_correction_fraction,
            self.max_registration_translation_px,
            self.max_registration_rotation_rad,
            self.max_registration_log_scale,
            self.local_radius_px,
        )
        if min(bounds) <= 0:
            raise V175ContractError("structured correction bounds must be positive")


@dataclass(frozen=True)
class V175FCPGMemory:
    """Persistent physical correction state; never stores pixels/predictions."""

    joint_correction_fraction: torch.Tensor
    view_registration: torch.Tensor
    previous_q: torch.Tensor
    readback_summary: torch.Tensor
    history_valid: torch.Tensor

    @classmethod
    def empty(
        cls,
        batch: int,
        joints: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> "V175FCPGMemory":
        return cls(
            joint_correction_fraction=torch.zeros(batch, joints, device=device, dtype=dtype),
            view_registration=torch.zeros(batch, VIEWS, 4, device=device, dtype=dtype),
            previous_q=torch.zeros(batch, joints, device=device, dtype=dtype),
            readback_summary=torch.zeros(
                batch, HISTORY_CHUNKS, joints, 5, device=device, dtype=dtype
            ),
            history_valid=torch.zeros(
                batch, HISTORY_CHUNKS, device=device, dtype=torch.bool
            ),
        )

    def validate(self, *, joints: int) -> None:
        batch = len(self.history_valid)
        _shape("joint_correction_fraction", self.joint_correction_fraction, (batch, joints))
        _shape("view_registration", self.view_registration, (batch, VIEWS, 4))
        _shape("previous_q", self.previous_q, (batch, joints))
        _shape(
            "readback_summary",
            self.readback_summary,
            (batch, HISTORY_CHUNKS, joints, 5),
        )
        valid = _bool("history_valid", self.history_valid, (batch, HISTORY_CHUNKS))
        if bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise V175ContractError("history_valid must be a left-aligned prefix")


@dataclass(frozen=True)
class V175Measurement:
    scene_tokens: torch.Tensor
    view_valid: torch.Tensor
    prior_keypoints_px: torch.Tensor
    corrected_keypoints_px: torch.Tensor
    fm_px: torch.Tensor
    midpoint_px: torch.Tensor
    grounded_part_tokens: torch.Tensor
    anchor_valid: torch.Tensor
    correspondence_maps: dict[str, torch.Tensor]
    query_tokens_by_layer: torch.Tensor
    next_memory: V175FCPGMemory

    @property
    def anchor_tokens(self) -> torch.Tensor:
        return self.grounded_part_tokens

    @property
    def grounded_anchor_tokens(self) -> torch.Tensor:
        return self.grounded_part_tokens


V175GroundedMeasurement = V175Measurement


@dataclass(frozen=True)
class V175LeWMPrediction:
    predicted_anchor_tokens: torch.Tensor
    predicted_q_path: torch.Tensor
    tokens_by_layer: torch.Tensor


@dataclass(frozen=True)
class V175Innovation:
    visual: torch.Tensor
    state: torch.Tensor
    anchor_valid: torch.Tensor
    path_valid: torch.Tensor


@dataclass(frozen=True)
class V175ActorOutput:
    commands: torch.Tensor
    desired_commands: torch.Tensor
    increments: torch.Tensor
    action_queries_by_layer: torch.Tensor


def _finite(name: str, value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not bool(torch.isfinite(tensor).all()):
        raise V175ContractError(f"{name} must be finite")
    return tensor


def _shape(name: str, value: torch.Tensor, expected: tuple[int, ...]) -> torch.Tensor:
    tensor = _finite(name, value)
    if tuple(tensor.shape) != tuple(expected):
        raise V175ContractError(
            f"{name} shape drifted: observed={list(tensor.shape)} expected={list(expected)}"
        )
    return tensor


def _bool(name: str, value: torch.Tensor, expected: tuple[int, ...]) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.dtype != torch.bool or tuple(tensor.shape) != tuple(expected):
        raise V175ContractError(f"{name} must be boolean {list(expected)}")
    return tensor


class _PreNormEncoderLayer(nn.Module):
    def __init__(self, dimension: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=0.0, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, ffn), nn.GELU(), nn.Linear(ffn, dimension)
        )

    def forward(
        self, value: torch.Tensor, *, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        normalized = self.attention_norm(value)
        value = value + self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return value + self.ffn(self.ffn_norm(value))


class _PreNormDecoderLayer(nn.Module):
    def __init__(self, dimension: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dimension)
        self.self_attention = nn.MultiheadAttention(
            dimension, heads, dropout=0.0, batch_first=True
        )
        self.query_norm = nn.LayerNorm(dimension)
        self.memory_norm = nn.LayerNorm(dimension)
        self.cross_attention = nn.MultiheadAttention(
            dimension, heads, dropout=0.0, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, ffn), nn.GELU(), nn.Linear(ffn, dimension)
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        *,
        query_mask: torch.Tensor | None = None,
        memory_padding_mask: torch.Tensor | None = None,
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


def _anchor_influence(spec: RobotSpec) -> np.ndarray:
    result = np.zeros((spec.anchors, spec.joints), dtype=np.float32)
    for anchor in range(spec.anchors):
        body = int(spec.anchor_body[anchor])
        ancestors: set[int] = set()
        while body >= 0:
            ancestors.add(body)
            body = int(spec.body_parent[body])
        for joint, joint_body in enumerate(spec.joint_body):
            if int(joint_body) in ancestors:
                result[anchor, joint] = 1.0
    return result


class MorphologyConditionedKinematicFCPG(nn.Module):
    """Prediction-measurement observer with a structurally valid output."""

    def __init__(
        self,
        config: V175Config = V175Config(),
        robot_spec: RobotSpec | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.robot_spec = robot_spec or so101_robot_spec()
        self.robot_spec.validate()
        if self.robot_spec.views != VIEWS:
            raise V175ContractError("V17.5 currently requires two camera references")
        self.kinematics = DifferentiableRobotKinematics(self.robot_spec)
        d = config.anchor_dim

        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, 96, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(12, 96), nn.GELU(),
            nn.Conv2d(96, 192, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(24, 192), nn.GELU(),
            nn.Conv2d(192, d, 3, padding=1, bias=False),
            nn.GroupNorm(24, d), nn.GELU(),
        )
        self.rgb_coarse = nn.Sequential(
            nn.Conv2d(d, d, 3, stride=2, padding=1, groups=d, bias=False),
            nn.Conv2d(d, d, 1), nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * d, d, 1, bias=False),
            nn.GroupNorm(24, d), nn.GELU(),
            nn.Conv2d(d, d, 3, padding=1, groups=d, bias=False),
            nn.Conv2d(d, d, 1),
        )
        self.position_projection = nn.Sequential(
            nn.Linear(2, d), nn.GELU(), nn.Linear(d, d)
        )
        self.view_embedding = nn.Parameter(torch.empty(1, VIEWS, 1, d))
        self.anchor_embedding = nn.Parameter(torch.empty(1, self.robot_spec.anchors, d))
        self.joint_projection = nn.Sequential(
            nn.Linear(11, d), nn.GELU(), nn.Linear(d, d)
        )
        self.history_projection = nn.Sequential(
            nn.Linear(5, d), nn.GELU(), nn.Linear(d, d)
        )
        self.history_embedding = nn.Parameter(
            torch.empty(1, HISTORY_CHUNKS, 1, d)
        )
        self.motion_layers = nn.ModuleList(
            _PreNormEncoderLayer(d, config.heads, config.anchor_ffn_dim)
            for _ in range(config.motion_layers)
        )
        self.motion_joint_head = nn.Linear(d, 1)
        self.measurement_layers = nn.ModuleList(
            _PreNormDecoderLayer(d, config.heads, config.anchor_ffn_dim)
            for _ in range(config.measurement_layers)
        )
        self.query_norm = nn.LayerNorm(d)
        self.pixel_norm = nn.LayerNorm(d)
        self.pixel_projection = nn.Linear(d, d, bias=False)
        self.displacement_projection = nn.Sequential(
            nn.Linear(3, d), nn.GELU(), nn.Linear(d, d)
        )
        self.global_joint_update = nn.Sequential(
            nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.global_view_update = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4)
        )
        self.local_joint_update = nn.Sequential(
            nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.local_view_update = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4)
        )
        self.anchor_output_norm = nn.LayerNorm(d)

        centers = torch.arange(FEATURE_GRID, dtype=torch.float32) * 4.0 + 2.0
        yy, xx = torch.meshgrid(centers, centers, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        local_axis = torch.linspace(-config.local_radius_px, config.local_radius_px, LOCAL_GRID)
        local_y, local_x = torch.meshgrid(local_axis, local_axis, indexing="ij")
        local_offsets = torch.stack((local_x, local_y), dim=-1).reshape(-1, 2)
        self.register_buffer("coarse_coordinates_px", coordinates, persistent=True)
        self.register_buffer("local_offsets_px", local_offsets, persistent=True)
        self.register_buffer(
            "anchor_influence", torch.from_numpy(_anchor_influence(self.robot_spec)), persistent=True
        )
        for value in (self.view_embedding, self.anchor_embedding, self.history_embedding):
            nn.init.normal_(value, std=0.02)
        for head in (
            self.motion_joint_head,
            self.global_joint_update[-1],
            self.global_view_update[-1],
            self.local_joint_update[-1],
            self.local_view_update[-1],
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def rgb_features(self, rgb: torch.Tensor) -> torch.Tensor:
        value = _finite("rgb", rgb)
        batch = len(value)
        if tuple(value.shape) != (batch, VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE):
            raise V175ContractError("rgb must be [B,2,3,224,224]")
        if value.dtype == torch.uint8:
            value = value.float().div(255.0)
        elif value.is_floating_point():
            value = value.float()
            if bool(((value < 0.0) | (value > 1.0)).any()):
                raise V175ContractError("floating RGB must be in [0,1]")
        else:
            raise V175ContractError("rgb must be uint8 or floating point")
        mean = value.mean(dim=(-1, -2), keepdim=True)
        std = value.std(dim=(-1, -2), keepdim=True).clamp_min(1.0e-4)
        result = self.rgb_stem(
            ((value - mean) / std).reshape(batch * VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE)
        )
        return result.reshape(batch, VIEWS, ANCHOR_DIM, FEATURE_GRID, FEATURE_GRID)

    def dino_features(self, dino_tokens: torch.Tensor) -> torch.Tensor:
        tokens = _finite("dino_tokens", dino_tokens)
        batch = len(tokens)
        if tuple(tokens.shape) != (batch, VIEWS, VIT_PATCHES, VIT_DIM):
            raise V175ContractError("dino_tokens must be [B,2,256,384]")
        tokens = F.layer_norm(tokens.float(), (VIT_DIM,))
        feature = tokens.reshape(batch * VIEWS, VIT_GRID, VIT_GRID, VIT_DIM)
        feature = feature.permute(0, 3, 1, 2)
        feature = F.interpolate(feature, size=(FEATURE_GRID, FEATURE_GRID), mode="bilinear", align_corners=False)
        return feature.reshape(batch, VIEWS, VIT_DIM, FEATURE_GRID, FEATURE_GRID)

    def scene_tokens_from_dino(
        self, dino_tokens: torch.Tensor, view_valid: torch.Tensor
    ) -> torch.Tensor:
        tokens = _finite("dino_tokens", dino_tokens)
        batch = len(tokens)
        tokens = _shape("dino_tokens", tokens, (batch, VIEWS, VIT_PATCHES, VIT_DIM)).float()
        valid = _bool("view_valid", view_valid, (batch, VIEWS)).to(tokens.device)
        return F.layer_norm(tokens, (VIT_DIM,)) * valid[:, :, None, None]

    def fused_features(
        self, rgb: torch.Tensor, dino_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fused 56-grid plus explicit 56/28 real-RGB pyramid features."""

        dino = self.dino_features(dino_tokens)
        batch = len(dino)
        rgb_feature = self.rgb_features(rgb).to(dino)
        coarse = self.rgb_coarse(
            rgb_feature.reshape(batch * VIEWS, ANCHOR_DIM, FEATURE_GRID, FEATURE_GRID)
        )
        coarse_up = F.interpolate(
            coarse, size=(FEATURE_GRID, FEATURE_GRID), mode="bilinear", align_corners=False
        )
        fused = self.fusion(
            torch.cat((rgb_feature, dino), dim=2).reshape(
                batch * VIEWS, 2 * ANCHOR_DIM, FEATURE_GRID, FEATURE_GRID
            )
        ) + coarse_up
        return (
            fused.reshape(batch, VIEWS, ANCHOR_DIM, FEATURE_GRID, FEATURE_GRID),
            rgb_feature,
            dino,
        )

    def _history_summary(
        self,
        completed_commands: torch.Tensor,
        measured_paths: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
        span = (self.kinematics.position_upper - self.kinematics.position_lower).to(measured_paths)
        displacement = (measured_paths[:, :, -1] - measured_paths[:, :, 0]) / span
        command_error = (completed_commands[:, :, -1] - measured_paths[:, :, -1]) / span
        increments = measured_paths[:, :, 1:] - measured_paths[:, :, :-1]
        direction = torch.sign(increments.sum(dim=2))
        signs = torch.sign(increments)
        reversals = (
            (signs[:, :, 1:] * signs[:, :, :-1] < 0).float().sum(dim=2)
            / max(1, PATH_STEPS - 2)
        )
        command_displacement = (completed_commands[:, :, -1] - measured_paths[:, :, 0]) / span
        summary = torch.stack(
            (displacement, command_error, direction, reversals, command_displacement), dim=-1
        )
        return summary * history_valid[:, :, None, None]

    def _joint_tokens(
        self,
        current_q: torch.Tensor,
        readback_summary: torch.Tensor,
        history_valid: torch.Tensor,
        joint_state_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, joints = current_q.shape
        normalized = self.kinematics.normalize_position(current_q)
        normalized = normalized * joint_state_valid[:, None]
        axis = self.kinematics.joint_axis.to(current_q)
        position = self.kinematics.joint_position.to(current_q)
        joint_type = F.one_hot(self.kinematics.joint_type, num_classes=2).to(current_q)
        coordinate_span = (self.kinematics.coordinate_upper - self.kinematics.coordinate_lower).to(current_q)
        public_span = (self.kinematics.position_upper - self.kinematics.position_lower).to(current_q)
        static = torch.cat(
            (axis, position, joint_type, coordinate_span[:, None], public_span[:, None] / 200.0), dim=-1
        )
        feature = torch.cat(
            (normalized[..., None], static[None].expand(batch, -1, -1)), dim=-1
        )
        if feature.shape[-1] != 11:
            raise V175ContractError("joint feature construction drifted")
        joint_base = self.joint_projection(feature)
        history = self.history_projection(readback_summary) + self.history_embedding
        sequence = torch.cat((joint_base[:, None], history), dim=1).transpose(1, 2)
        sequence = sequence.reshape(batch * joints, HISTORY_CHUNKS + 1, ANCHOR_DIM)
        padding = torch.cat(
            (torch.zeros(batch, 1, dtype=torch.bool, device=current_q.device), ~history_valid), dim=1
        )[:, None].expand(-1, joints, -1).reshape(batch * joints, -1)
        for layer in self.motion_layers:
            sequence = layer(sequence, key_padding_mask=padding)
        motion_joint = sequence[:, 0].reshape(batch, joints, ANCHOR_DIM)
        has_history = history_valid.any(dim=1)[:, None, None]
        joint = torch.where(has_history, motion_joint, joint_base)
        return joint, sequence.reshape(batch, joints, HISTORY_CHUNKS + 1, ANCHOR_DIM)

    def kinematic_part_tokens(
        self,
        current_q_deg: torch.Tensor,
        view_valid: torch.Tensor,
        *,
        completed_commands_deg: torch.Tensor | None = None,
        measured_q_paths_deg: torch.Tensor | None = None,
        history_valid: torch.Tensor | None = None,
        joint_state_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode ordered robot state without inventing unavailable pixels.

        The 320K archive contains measured joint trajectories but no auditable
        camera model or absolute keypoint labels.  This path shares the runtime
        morphology/action encoder and ordered anchor identities, while
        deliberately skipping the image measurement decoder.
        """

        state = _finite("current_q_deg", current_q_deg).float()
        if state.ndim != 2 or state.shape[1] != self.robot_spec.joints:
            raise V175ContractError("current_q_deg must be [B,RobotSpec.joints]")
        batch, joints = state.shape
        valid_view = _bool("view_valid", view_valid, (batch, VIEWS)).to(state.device)
        commands = (
            torch.zeros(
                batch, HISTORY_CHUNKS, COMMAND_STEPS, joints,
                device=state.device, dtype=state.dtype,
            )
            if completed_commands_deg is None
            else _shape(
                "completed_commands_deg", completed_commands_deg,
                (batch, HISTORY_CHUNKS, COMMAND_STEPS, joints),
            ).to(state)
        )
        paths = (
            state[:, None, None].expand(-1, HISTORY_CHUNKS, PATH_STEPS, -1).clone()
            if measured_q_paths_deg is None
            else _shape(
                "measured_q_paths_deg", measured_q_paths_deg,
                (batch, HISTORY_CHUNKS, PATH_STEPS, joints),
            ).to(state)
        )
        causal = (
            torch.zeros(batch, HISTORY_CHUNKS, dtype=torch.bool, device=state.device)
            if history_valid is None
            else _bool(
                "history_valid", history_valid, (batch, HISTORY_CHUNKS)
            ).to(state.device)
        )
        if bool((causal[:, 1:] & ~causal[:, :-1]).any()):
            raise V175ContractError("history_valid must be a left-aligned prefix")
        summary = self._history_summary(commands, paths, causal)
        state_valid = (
            torch.ones(batch, dtype=torch.bool, device=state.device)
            if joint_state_valid is None
            else _bool(
                "joint_state_valid", joint_state_valid, (batch,)
            ).to(state.device)
        )
        joint_tokens, motion_trace = self._joint_tokens(
            state,
            summary,
            causal,
            state_valid,
        )
        influence = self.anchor_influence.to(joint_tokens)
        anchor_joint = torch.einsum(
            "aj,bjd->bad",
            influence / influence.sum(dim=1, keepdim=True).clamp_min(1.0),
            joint_tokens,
        )
        tokens = self.anchor_output_norm(
            self.anchor_embedding[:, None]
            + self.view_embedding
            + anchor_joint[:, None]
        ).expand(batch, -1, -1, -1)
        anchor_valid = valid_view[:, :, None].expand(
            -1, -1, self.robot_spec.anchors
        )
        return tokens * anchor_valid[..., None], anchor_valid, motion_trace

    def _apply_registration(self, pixel: torch.Tensor, registration: torch.Tensor) -> torch.Tensor:
        translation = registration[..., :2]
        angle = registration[..., 2]
        scale = registration[..., 3].exp()
        cosine, sine = angle.cos(), angle.sin()
        rotation = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(*angle.shape, 2, 2)
        centered = pixel - (IMAGE_SIZE - 1.0) / 2.0
        transformed = torch.einsum("bvij,bvaj->bvai", rotation, centered) * scale[..., None, None]
        return transformed + (IMAGE_SIZE - 1.0) / 2.0 + translation[:, :, None]

    def _project(
        self,
        current_q: torch.Tensor,
        joint_fraction: torch.Tensor,
        registration: torch.Tensor,
        camera_intrinsics: torch.Tensor | None,
        camera_from_reference: torch.Tensor | None,
        image_transform: torch.Tensor | None,
        cached_nominal_keypoints_px: torch.Tensor | None,
        cached_nominal_valid: torch.Tensor | None,
        cached_nominal_jacobian_px: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(current_q)
        if cached_nominal_keypoints_px is not None:
            nominal = _shape(
                "cached_nominal_keypoints_px",
                cached_nominal_keypoints_px,
                (batch, VIEWS, self.robot_spec.anchors, 2),
            ).to(current_q)
            if cached_nominal_jacobian_px is not None:
                jacobian = _shape(
                    "cached_nominal_jacobian_px",
                    cached_nominal_jacobian_px,
                    (
                        batch,
                        VIEWS,
                        self.robot_spec.anchors,
                        2,
                        self.robot_spec.joints,
                    ),
                ).to(current_q)
                span = (
                    self.kinematics.position_upper
                    - self.kinematics.position_lower
                ).to(current_q)
                nominal = nominal + torch.einsum(
                    "bvaij,bj->bvai",
                    jacobian,
                    joint_fraction * span[None],
                )
            valid = (
                (nominal[..., 0] >= 0) & (nominal[..., 0] < IMAGE_SIZE)
                & (nominal[..., 1] >= 0) & (nominal[..., 1] < IMAGE_SIZE)
            )
            if cached_nominal_valid is not None:
                valid = valid & _bool(
                    "cached_nominal_valid",
                    cached_nominal_valid,
                    (batch, VIEWS, self.robot_spec.anchors),
                ).to(valid.device)
            return self._apply_registration(nominal, registration), valid
        if camera_intrinsics is None or camera_from_reference is None:
            raise V175ContractError(
                "camera geometry is required unless an audited cached nominal is supplied"
            )
        span = (self.kinematics.position_upper - self.kinematics.position_lower).to(current_q)
        corrected_q = current_q + joint_fraction * span[None]
        reference_points = self.kinematics.anchors_in_view_references(corrected_q)
        projected, valid = project_reference_points(
            reference_points, camera_from_reference, camera_intrinsics, image_transform
        )
        return self._apply_registration(projected, registration), valid

    def _correspond(
        self,
        query: torch.Tensor,
        fused: torch.Tensor,
        prior_px: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = fused.flatten(start_dim=3).transpose(2, 3)
        pixel = self.pixel_projection(self.pixel_norm(memory))
        logits = torch.einsum("bvad,bvpd->bvap", self.query_norm(query), pixel) / math.sqrt(ANCHOR_DIM)
        coordinate = self.coarse_coordinates_px.to(logits)
        distance = (coordinate[None, None, None] - prior_px[..., None, :]).square().sum(dim=-1)
        logits = logits - 0.25 * distance / float(IMAGE_SIZE**2)
        probability = logits.float().softmax(dim=-1)
        measurement = torch.einsum("bvap,pd->bvad", probability, coordinate)
        measurement = torch.where(view_valid[:, :, None, None], measurement, prior_px)
        return probability, measurement

    def _local_correspond(
        self,
        query: torch.Tensor,
        fused: torch.Tensor,
        center_px: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, views, anchors, _ = center_px.shape
        candidate = center_px[..., None, :] + self.local_offsets_px.to(center_px)
        normalized = 2.0 * candidate / (IMAGE_SIZE - 1.0) - 1.0
        sampled = F.grid_sample(
            fused.reshape(batch * views, ANCHOR_DIM, FEATURE_GRID, FEATURE_GRID),
            normalized.reshape(batch * views, anchors, LOCAL_CELLS, 2),
            mode="bilinear", padding_mode="zeros", align_corners=True,
        ).reshape(batch, views, ANCHOR_DIM, anchors, LOCAL_CELLS).permute(0, 1, 3, 4, 2)
        logits = torch.einsum(
            "bvad,bvapd->bvap", self.query_norm(query), self.pixel_norm(sampled)
        ) / math.sqrt(ANCHOR_DIM)
        probability = logits.float().softmax(dim=-1)
        offset = torch.einsum("bvap,pd->bvad", probability, self.local_offsets_px.to(probability))
        measurement = torch.where(
            view_valid[:, :, None, None], center_px + offset, center_px
        )
        return probability, measurement

    def _structured_update(
        self,
        query: torch.Tensor,
        measured_px: torch.Tensor,
        prior_px: torch.Tensor,
        correspondence: torch.Tensor,
        measurement_valid: torch.Tensor,
        joint_tokens: torch.Tensor,
        joint_fraction: torch.Tensor,
        registration: torch.Tensor,
        *,
        local: bool,
        allow_joint_update: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        entropy = -(
            correspondence.clamp_min(1.0e-8) * correspondence.clamp_min(1.0e-8).log()
        ).sum(dim=-1) / math.log(correspondence.shape[-1])
        displacement = (measured_px - prior_px) / float(IMAGE_SIZE)
        evidence = query + self.displacement_projection(
            torch.cat((displacement, entropy[..., None]), dim=-1)
        )
        valid = measurement_valid.to(evidence)
        view_evidence = (evidence * valid[..., None]).sum(dim=2) / valid.sum(
            dim=2
        ).clamp_min(1.0)[..., None]
        view_head = self.local_view_update if local else self.global_view_update
        raw_view = view_head(view_evidence)
        view_scale = raw_view.new_tensor(
            (
                self.config.max_registration_translation_px,
                self.config.max_registration_translation_px,
                self.config.max_registration_rotation_rad,
                self.config.max_registration_log_scale,
            )
        )
        view_delta = torch.tanh(raw_view) * view_scale
        view_has_measurement = measurement_valid.any(dim=-1)
        updated_view = registration + view_delta * view_has_measurement[..., None]
        updated_view = torch.maximum(torch.minimum(updated_view, view_scale), -view_scale)

        influence = self.anchor_influence.to(evidence)
        denominator = influence.sum(dim=0).clamp_min(1.0)
        anchor_evidence = (evidence * valid[..., None]).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)[..., None]
        joint_evidence = torch.einsum(
            "aj,bad->bjd", influence / denominator[None], anchor_evidence
        )
        joint_head = self.local_joint_update if local else self.global_joint_update
        raw_joint = joint_head(torch.cat((joint_tokens, joint_evidence), dim=-1)).squeeze(-1)
        joint_delta = torch.tanh(raw_joint) * self.config.max_joint_correction_fraction
        if not allow_joint_update:
            joint_delta = torch.zeros_like(joint_delta)
        updated_joint = torch.clamp(
            joint_fraction + joint_delta,
            min=-self.config.max_joint_correction_fraction,
            max=self.config.max_joint_correction_fraction,
        )
        return updated_joint, updated_view

    def forward(
        self,
        rgb: torch.Tensor,
        dino_tokens: torch.Tensor,
        view_valid: torch.Tensor,
        current_q_deg: torch.Tensor,
        camera_intrinsics: torch.Tensor | None,
        camera_from_reference: torch.Tensor | None,
        image_transform: torch.Tensor | None = None,
        completed_commands_deg: torch.Tensor | None = None,
        measured_q_paths_deg: torch.Tensor | None = None,
        history_valid: torch.Tensor | None = None,
        previous_memory: V175FCPGMemory | None = None,
        cached_nominal_keypoints_px: torch.Tensor | None = None,
        cached_nominal_valid: torch.Tensor | None = None,
        cached_nominal_jacobian_px: torch.Tensor | None = None,
        joint_state_valid: torch.Tensor | None = None,
    ) -> V175Measurement:
        fused, _, dino = self.fused_features(rgb, dino_tokens)
        batch = len(dino)
        joints, anchors = self.robot_spec.joints, self.robot_spec.anchors
        current_q = _shape("current_q_deg", current_q_deg, (batch, joints)).to(dino)
        q_valid = (
            torch.ones(batch, dtype=torch.bool, device=dino.device)
            if joint_state_valid is None
            else _bool("joint_state_valid", joint_state_valid, (batch,)).to(dino.device)
        )
        valid = _bool("view_valid", view_valid, (batch, VIEWS)).to(dino.device)
        commands = (
            torch.zeros(batch, HISTORY_CHUNKS, COMMAND_STEPS, joints, device=dino.device, dtype=dino.dtype)
            if completed_commands_deg is None
            else _shape(
                "completed_commands_deg", completed_commands_deg,
                (batch, HISTORY_CHUNKS, COMMAND_STEPS, joints),
            ).to(dino)
        )
        paths = (
            current_q[:, None, None].expand(-1, HISTORY_CHUNKS, PATH_STEPS, -1).clone()
            if measured_q_paths_deg is None
            else _shape(
                "measured_q_paths_deg", measured_q_paths_deg,
                (batch, HISTORY_CHUNKS, PATH_STEPS, joints),
            ).to(dino)
        )
        causal = (
            torch.zeros(batch, HISTORY_CHUNKS, dtype=torch.bool, device=dino.device)
            if history_valid is None
            else _bool("history_valid", history_valid, (batch, HISTORY_CHUNKS)).to(dino.device)
        )
        if bool((causal[:, 1:] & ~causal[:, :-1]).any()):
            raise V175ContractError("history_valid must be a left-aligned prefix")
        if previous_memory is None:
            previous_memory = V175FCPGMemory.empty(batch, joints, device=dino.device, dtype=dino.dtype)
        previous_memory.validate(joints=joints)
        if len(previous_memory.history_valid) != batch:
            raise V175ContractError("FCPG memory batch differs")

        summary = self._history_summary(commands, paths, causal)
        joint_tokens, motion_trace = self._joint_tokens(
            current_q, summary, causal, q_valid.to(current_q)
        )
        del motion_trace
        motion_delta = torch.tanh(self.motion_joint_head(joint_tokens).squeeze(-1))
        motion_delta = motion_delta * self.config.max_joint_correction_fraction
        motion_delta = motion_delta * causal.any(dim=1, keepdim=True)
        joint_fraction = torch.clamp(
            previous_memory.joint_correction_fraction.to(current_q) + motion_delta,
            -self.config.max_joint_correction_fraction,
            self.config.max_joint_correction_fraction,
        )
        registration = previous_memory.view_registration.to(current_q)
        prior_px, geometry_valid = self._project(
            current_q, joint_fraction, registration,
            camera_intrinsics, camera_from_reference, image_transform,
            cached_nominal_keypoints_px, cached_nominal_valid,
            cached_nominal_jacobian_px,
        )

        coordinate = self.coarse_coordinates_px.to(fused)
        position = self.position_projection((coordinate / (IMAGE_SIZE - 1.0)) * 2.0 - 1.0)
        fused = fused + position.T.reshape(1, 1, ANCHOR_DIM, FEATURE_GRID, FEATURE_GRID)
        fused = fused * valid[:, :, None, None, None]
        scene = self.scene_tokens_from_dino(dino_tokens, valid)

        influence = self.anchor_influence.to(joint_tokens)
        anchor_joint = torch.einsum(
            "aj,bjd->bad",
            influence / influence.sum(dim=1, keepdim=True).clamp_min(1.0),
            joint_tokens,
        )
        query = (
            self.anchor_embedding[:, None]
            + self.view_embedding
            + anchor_joint[:, None]
        ).expand(batch, -1, -1, -1).clone()
        visual_memory = fused.flatten(start_dim=3).transpose(2, 3).reshape(
            batch * VIEWS, COARSE_CELLS, ANCHOR_DIM
        )
        query_flat = query.reshape(batch * VIEWS, anchors, ANCHOR_DIM)
        query_layers = [query]
        for layer in self.measurement_layers:
            query_flat = layer(query_flat, visual_memory)
            query = query_flat.reshape(batch, VIEWS, anchors, ANCHOR_DIM)
            query_layers.append(query)

        coarse_map, coarse_measurement = self._correspond(query, fused, prior_px, valid)
        measurement_valid = valid[:, :, None] & geometry_valid
        joint_fraction, registration = self._structured_update(
            query, coarse_measurement, prior_px, coarse_map, measurement_valid,
            joint_tokens, joint_fraction, registration,
            local=False,
            allow_joint_update=(
                cached_nominal_keypoints_px is None
                or cached_nominal_jacobian_px is not None
            ),
        )
        coarse_corrected, _ = self._project(
            current_q, joint_fraction, registration,
            camera_intrinsics, camera_from_reference, image_transform,
            cached_nominal_keypoints_px, cached_nominal_valid,
            cached_nominal_jacobian_px,
        )
        local_map, local_measurement = self._local_correspond(
            query, fused, coarse_corrected, valid
        )
        joint_fraction, registration = self._structured_update(
            query, local_measurement, coarse_corrected, local_map, measurement_valid,
            joint_tokens, joint_fraction, registration,
            local=True,
            allow_joint_update=(
                cached_nominal_keypoints_px is None
                or cached_nominal_jacobian_px is not None
            ),
        )
        corrected, corrected_geometry_valid = self._project(
            current_q, joint_fraction, registration,
            camera_intrinsics, camera_from_reference, image_transform,
            cached_nominal_keypoints_px, cached_nominal_valid,
            cached_nominal_jacobian_px,
        )
        anchor_valid = valid[:, :, None] & geometry_valid & corrected_geometry_valid
        sample_grid = 2.0 * corrected / (IMAGE_SIZE - 1.0) - 1.0
        sampled = F.grid_sample(
            fused.reshape(batch * VIEWS, ANCHOR_DIM, FEATURE_GRID, FEATURE_GRID),
            sample_grid.reshape(batch * VIEWS, anchors, 1, 2),
            mode="bilinear", padding_mode="zeros", align_corners=True,
        ).reshape(batch, VIEWS, ANCHOR_DIM, anchors).permute(0, 1, 3, 2)
        grounded = self.anchor_output_norm(sampled + query) * anchor_valid[..., None]
        fm = corrected[
            :,
            :,
            [self.robot_spec.fixed_tip_index, self.robot_spec.moving_tip_index],
        ]
        midpoint = fm.mean(dim=2)
        next_memory = V175FCPGMemory(
            joint_correction_fraction=joint_fraction,
            view_registration=registration,
            previous_q=current_q,
            readback_summary=summary,
            history_valid=causal,
        )
        outputs = (prior_px, corrected, fm, midpoint, grounded, coarse_map, local_map)
        if not all(bool(torch.isfinite(value).all()) for value in outputs):
            raise V175ContractError("structured FCPG produced non-finite output")
        return V175Measurement(
            scene_tokens=scene,
            view_valid=valid,
            prior_keypoints_px=prior_px,
            corrected_keypoints_px=corrected,
            fm_px=fm,
            midpoint_px=midpoint,
            grounded_part_tokens=grounded,
            anchor_valid=anchor_valid,
            correspondence_maps={"global": coarse_map, "local": local_map},
            query_tokens_by_layer=torch.stack(query_layers, dim=1),
            next_memory=next_memory,
        )


V175GroundedAnchorFCPG = MorphologyConditionedKinematicFCPG


def ordered_anchor_smooth_l1(
    measurement: V175Measurement,
    target_anchor_px: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    beta_px: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Ordered all-anchor loss with exact 0.5/0.5 view macro reduction."""

    predicted = measurement.corrected_keypoints_px.float()
    batch = len(predicted)
    target = _shape("target_anchor_px", target_anchor_px, (batch, VIEWS, ANCHORS, 2)).to(predicted)
    valid = _bool("target_valid", target_valid, (batch, VIEWS, ANCHORS)).to(predicted.device)
    valid = valid & measurement.view_valid[:, :, None]
    coordinate = target[valid]
    if coordinate.numel() and bool(((coordinate < 0.0) | (coordinate >= IMAGE_SIZE)).any()):
        raise V175ContractError("valid anchor targets must lie inside the image")
    element = F.smooth_l1_loss(predicted, target, reduction="none", beta=float(beta_px)).mean(dim=-1)
    per_view = (element * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
    active = valid.any(dim=-1)
    view_losses: list[torch.Tensor] = []
    for view in range(VIEWS):
        view_losses.append(
            per_view[active[:, view], view].mean()
            if bool(active[:, view].any())
            else per_view[:, view].sum() * 0.0
        )
    total = 0.5 * view_losses[0] + 0.5 * view_losses[1]
    return total, {
        "total": total.detach(),
        "wrist": view_losses[0].detach(),
        "column": view_losses[1].detach(),
        "active_anchors": valid.sum().detach(),
    }


def unordered_tip_reprojection_loss(
    predicted_fm_px: torch.Tensor,
    target_fm_px: torch.Tensor,
    view_valid: torch.Tensor,
    *,
    beta_px: float = 2.0,
) -> torch.Tensor:
    """Training-fit loss for legacy real rows whose F/M identity is unknown."""

    predicted = _finite("predicted_fm_px", predicted_fm_px).float()
    batch = len(predicted)
    target = _shape("target_fm_px", target_fm_px, (batch, VIEWS, 2, 2)).to(predicted)
    valid = _bool("view_valid", view_valid, (batch, VIEWS)).to(predicted.device)
    direct = F.smooth_l1_loss(predicted, target, reduction="none", beta=float(beta_px)).mean(dim=(-1, -2))
    swapped = F.smooth_l1_loss(predicted, target[:, :, (1, 0)], reduction="none", beta=float(beta_px)).mean(dim=(-1, -2))
    return (torch.minimum(direct, swapped) * valid).sum() / valid.sum().clamp_min(1)


class V175GroundedLeWM(nn.Module):
    """Action-conditioned one-chunk prediction in ordered part-token space."""

    def __init__(self, config: V175Config = V175Config()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.anchor_dim
        self.anchor_projection = nn.Linear(ANCHOR_DIM, d)
        self.view_embedding = nn.Parameter(torch.empty(1, VIEWS, 1, d))
        self.anchor_identity = nn.Parameter(torch.empty(1, 1, ANCHORS, d))
        self.q_projection = nn.Sequential(
            nn.Linear(STATE_DIM, d), nn.GELU(), nn.Linear(d, d)
        )
        self.command_projection = nn.Sequential(
            nn.Linear(STATE_DIM, d), nn.GELU(), nn.Linear(d, d)
        )
        self.motion_time_embedding = nn.Parameter(torch.empty(1, PATH_STEPS, d))
        self.layers = nn.ModuleList(
            _PreNormEncoderLayer(d, config.heads, config.anchor_ffn_dim)
            for _ in range(config.lewm_layers)
        )
        self.output_norm = nn.LayerNorm(d)
        self.anchor_delta_head = nn.Linear(d, ANCHOR_DIM)
        self.q_residual_head = nn.Linear(d, STATE_DIM)
        for value in (self.view_embedding, self.anchor_identity, self.motion_time_embedding):
            nn.init.normal_(value, std=0.02)
        nn.init.zeros_(self.anchor_delta_head.weight)
        nn.init.zeros_(self.anchor_delta_head.bias)
        nn.init.zeros_(self.q_residual_head.weight)
        nn.init.zeros_(self.q_residual_head.bias)

    def forward(
        self,
        grounded_anchor_tokens: torch.Tensor,
        anchor_valid: torch.Tensor,
        current_q: torch.Tensor,
        commands: torch.Tensor,
    ) -> V175LeWMPrediction:
        source = _finite("grounded_anchor_tokens", grounded_anchor_tokens)
        batch = len(source)
        source = _shape(
            "grounded_anchor_tokens", source, (batch, VIEWS, ANCHORS, ANCHOR_DIM)
        ).to(dtype=self.anchor_projection.weight.dtype)
        valid = _bool("anchor_valid", anchor_valid, (batch, VIEWS, ANCHORS)).to(source.device)
        state = _shape("current_q", current_q, (batch, STATE_DIM)).to(source)
        action = _shape("commands", commands, (batch, COMMAND_STEPS, STATE_DIM)).to(source)
        anchors = self.anchor_projection(F.layer_norm(source, (ANCHOR_DIM,)))
        anchors = anchors + self.view_embedding + self.anchor_identity
        anchors = anchors.reshape(batch, VIEWS * ANCHORS, ANCHOR_DIM)
        motion = torch.cat(
            (
                self.q_projection(state / 100.0)[:, None],
                self.command_projection(action / 100.0),
            ),
            dim=1,
        ) + self.motion_time_embedding
        sequence = torch.cat((anchors, motion), dim=1)
        padding = torch.cat(
            (
                (~valid).reshape(batch, VIEWS * ANCHORS),
                torch.zeros(batch, PATH_STEPS, dtype=torch.bool, device=source.device),
            ),
            dim=1,
        )
        traces = [sequence]
        for layer in self.layers:
            sequence = layer(sequence, key_padding_mask=padding)
            traces.append(sequence)
        sequence = self.output_norm(sequence)
        anchor_hidden = sequence[:, : VIEWS * ANCHORS].reshape(
            batch, VIEWS, ANCHORS, ANCHOR_DIM
        )
        predicted = F.layer_norm(source, (ANCHOR_DIM,)) + self.anchor_delta_head(anchor_hidden)
        predicted = predicted * valid[..., None]
        motion_hidden = sequence[:, VIEWS * ANCHORS :]
        baseline = torch.cat((state[:, None], action), dim=1)
        q_path = baseline + self.q_residual_head(motion_hidden) * 4.0
        q_path = torch.cat((state[:, None], q_path[:, 1:]), dim=1)
        return V175LeWMPrediction(
            predicted_anchor_tokens=predicted,
            predicted_q_path=q_path,
            tokens_by_layer=torch.stack(traces, dim=1),
        )


def compute_v175_innovation(
    actual_anchor_tokens: torch.Tensor,
    actual_anchor_valid: torch.Tensor,
    measured_q_path: torch.Tensor,
    prediction: V175LeWMPrediction,
    *,
    path_valid: torch.Tensor | None = None,
) -> V175Innovation:
    actual = _finite("actual_anchor_tokens", actual_anchor_tokens)
    batch = len(actual)
    actual = _shape(
        "actual_anchor_tokens", actual, (batch, VIEWS, ANCHORS, ANCHOR_DIM)
    )
    valid = _bool(
        "actual_anchor_valid", actual_anchor_valid, (batch, VIEWS, ANCHORS)
    ).to(actual.device)
    predicted = _shape(
        "predicted_anchor_tokens",
        prediction.predicted_anchor_tokens,
        (batch, VIEWS, ANCHORS, ANCHOR_DIM),
    ).to(actual)
    path = _shape(
        "measured_q_path", measured_q_path, (batch, PATH_STEPS, STATE_DIM)
    ).to(actual)
    predicted_path = _shape(
        "predicted_q_path", prediction.predicted_q_path,
        (batch, PATH_STEPS, STATE_DIM),
    ).to(actual)
    valid_path = (
        torch.ones(batch, PATH_STEPS, dtype=torch.bool, device=actual.device)
        if path_valid is None
        else _bool("path_valid", path_valid, (batch, PATH_STEPS)).to(actual.device)
    )
    visual = (
        F.layer_norm(actual, (ANCHOR_DIM,))
        - F.layer_norm(predicted, (ANCHOR_DIM,))
    ) * valid[..., None]
    state = (path - predicted_path) * valid_path[..., None]
    return V175Innovation(
        visual=visual, state=state, anchor_valid=valid, path_valid=valid_path
    )


@dataclass(frozen=True)
class V175Memory:
    """Four completed transitions containing only actual observations."""

    actual_anchor_tokens: torch.Tensor
    actual_anchor_valid: torch.Tensor
    completed_commands: torch.Tensor
    measured_paths: torch.Tensor
    visual_innovations: torch.Tensor
    state_innovations: torch.Tensor
    innovation_valid: torch.Tensor
    path_valid: torch.Tensor
    history_valid: torch.Tensor

    @classmethod
    def empty(
        cls,
        batch: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> "V175Memory":
        if int(batch) <= 0:
            raise V175ContractError("memory batch must be positive")
        return cls(
            actual_anchor_tokens=torch.zeros(
                batch, HISTORY_CHUNKS, VIEWS, ANCHORS, ANCHOR_DIM,
                device=device, dtype=dtype,
            ),
            actual_anchor_valid=torch.zeros(
                batch, HISTORY_CHUNKS, VIEWS, ANCHORS,
                device=device, dtype=torch.bool,
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
                batch, HISTORY_CHUNKS, VIEWS, ANCHORS, ANCHOR_DIM,
                device=device, dtype=dtype,
            ),
            state_innovations=torch.zeros(
                batch, HISTORY_CHUNKS, PATH_STEPS, STATE_DIM,
                device=device, dtype=dtype,
            ),
            innovation_valid=torch.zeros(
                batch, HISTORY_CHUNKS, VIEWS, ANCHORS,
                device=device, dtype=torch.bool,
            ),
            path_valid=torch.zeros(
                batch, HISTORY_CHUNKS, PATH_STEPS,
                device=device, dtype=torch.bool,
            ),
            history_valid=torch.zeros(
                batch, HISTORY_CHUNKS, device=device, dtype=torch.bool
            ),
        )

    def validate(self) -> None:
        batch = len(self.history_valid)
        expected = {
            "actual_anchor_tokens": (batch, HISTORY_CHUNKS, VIEWS, ANCHORS, ANCHOR_DIM),
            "completed_commands": (batch, HISTORY_CHUNKS, COMMAND_STEPS, STATE_DIM),
            "measured_paths": (batch, HISTORY_CHUNKS, PATH_STEPS, STATE_DIM),
            "visual_innovations": (batch, HISTORY_CHUNKS, VIEWS, ANCHORS, ANCHOR_DIM),
            "state_innovations": (batch, HISTORY_CHUNKS, PATH_STEPS, STATE_DIM),
        }
        expected_bool = {
            "actual_anchor_valid": (batch, HISTORY_CHUNKS, VIEWS, ANCHORS),
            "innovation_valid": (batch, HISTORY_CHUNKS, VIEWS, ANCHORS),
            "path_valid": (batch, HISTORY_CHUNKS, PATH_STEPS),
            "history_valid": (batch, HISTORY_CHUNKS),
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in expected:
                _shape(field.name, value, expected[field.name])
            else:
                _bool(field.name, value, expected_bool[field.name])
        if bool((self.history_valid[:, 1:] & ~self.history_valid[:, :-1]).any()):
            raise V175ContractError("history_valid must be a left-aligned prefix")


class V175Actor(nn.Module):
    """Five-query policy over scene, ordered parts, feedback and history."""

    def __init__(self, config: V175Config = V175Config()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.actor_dim
        self.scene_projection = nn.Linear(VIT_DIM, d)
        self.anchor_projection = nn.Linear(ANCHOR_DIM, d)
        self.innovation_projection = nn.Linear(ANCHOR_DIM, d)
        self.q_projection = nn.Sequential(
            nn.Linear(STATE_DIM, d), nn.GELU(), nn.Linear(d, d)
        )
        event_dimension = COMMAND_STEPS * STATE_DIM + 2 * PATH_STEPS * STATE_DIM
        self.event_projection = nn.Sequential(
            nn.Linear(event_dimension, d), nn.GELU(), nn.Linear(d, d)
        )
        self.scene_camera_embedding = nn.Parameter(torch.empty(1, VIEWS, 1, d))
        self.anchor_view_embedding = nn.Parameter(torch.empty(1, VIEWS, 1, d))
        self.anchor_identity = nn.Parameter(torch.empty(1, 1, ANCHORS, d))
        self.history_embedding = nn.Parameter(torch.empty(1, HISTORY_CHUNKS, 1, d))
        self.source_embedding = nn.Parameter(torch.empty(6, d))
        self.action_queries = nn.Parameter(torch.empty(1, COMMAND_STEPS, d))
        self.action_time_embedding = nn.Parameter(torch.empty(1, COMMAND_STEPS, d))
        self.layers = nn.ModuleList(
            _PreNormDecoderLayer(d, config.heads, config.actor_ffn_dim)
            for _ in range(config.actor_layers)
        )
        self.output_norm = nn.LayerNorm(d)
        self.action_head = nn.Linear(d, STATE_DIM)
        cap, lower, upper = v175_action_contract()
        self.register_buffer("step_cap", cap.float(), persistent=True)
        self.register_buffer("action_lower", lower.float(), persistent=True)
        self.register_buffer("action_upper", upper.float(), persistent=True)
        for value in (
            self.scene_camera_embedding,
            self.anchor_view_embedding,
            self.anchor_identity,
            self.history_embedding,
            self.source_embedding,
            self.action_queries,
            self.action_time_embedding,
        ):
            nn.init.normal_(value, std=0.02)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def forward(
        self,
        scene_tokens: torch.Tensor,
        grounded_anchor_tokens: torch.Tensor,
        current_anchor_valid: torch.Tensor,
        visual_innovation: torch.Tensor,
        state_innovation: torch.Tensor,
        innovation_valid: torch.Tensor,
        path_valid: torch.Tensor,
        current_q: torch.Tensor,
        memory: V175Memory,
        *,
        history_prefix: int = HISTORY_CHUNKS,
    ) -> V175ActorOutput:
        memory.validate()
        scene = _finite("scene_tokens", scene_tokens)
        batch = len(scene)
        scene = _shape(
            "scene_tokens", scene, (batch, VIEWS, VIT_PATCHES, VIT_DIM)
        ).to(dtype=self.scene_projection.weight.dtype)
        anchor = _shape(
            "grounded_anchor_tokens", grounded_anchor_tokens,
            (batch, VIEWS, ANCHORS, ANCHOR_DIM),
        ).to(scene)
        anchor_valid = _bool(
            "current_anchor_valid", current_anchor_valid,
            (batch, VIEWS, ANCHORS),
        ).to(scene.device)
        visual_error = _shape(
            "visual_innovation", visual_innovation,
            (batch, VIEWS, ANCHORS, ANCHOR_DIM),
        ).to(scene)
        error_valid = _bool(
            "innovation_valid", innovation_valid, (batch, VIEWS, ANCHORS)
        ).to(scene.device)
        state_error = _shape(
            "state_innovation", state_innovation,
            (batch, PATH_STEPS, STATE_DIM),
        ).to(scene)
        valid_path = _bool(
            "path_valid", path_valid, (batch, PATH_STEPS)
        ).to(scene.device)
        state = _shape("current_q", current_q, (batch, STATE_DIM)).to(scene)
        if not 0 <= int(history_prefix) <= HISTORY_CHUNKS:
            raise V175ContractError(
                f"history_prefix must be in [0,{HISTORY_CHUNKS}]"
            )

        groups: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        scene_projected = self.scene_projection(scene) + self.scene_camera_embedding
        groups.append(scene_projected.reshape(batch, VIEWS * VIT_PATCHES, -1))
        scene_view_valid = anchor_valid.any(dim=-1)
        masks.append(
            (~scene_view_valid)[:, :, None]
            .expand(-1, -1, VIT_PATCHES)
            .reshape(batch, -1)
        )

        identity = self.anchor_identity
        current_anchor = (
            self.anchor_projection(anchor) + self.anchor_view_embedding + identity
        )
        groups.append(current_anchor.reshape(batch, VIEWS * ANCHORS, -1))
        masks.append((~anchor_valid).reshape(batch, VIEWS * ANCHORS))
        current_error = (
            self.innovation_projection(visual_error)
            + self.anchor_view_embedding
            + identity
        )
        groups.append(current_error.reshape(batch, VIEWS * ANCHORS, -1))
        masks.append((~error_valid).reshape(batch, VIEWS * ANCHORS))

        q_token = self.q_projection(state / 100.0)[:, None] + self.source_embedding[0]
        groups.append(q_token)
        masks.append(torch.zeros(batch, 1, dtype=torch.bool, device=scene.device))
        state_event = torch.cat(
            (
                torch.zeros(
                    batch, COMMAND_STEPS * STATE_DIM,
                    device=scene.device, dtype=scene.dtype,
                ),
                torch.zeros(
                    batch, PATH_STEPS * STATE_DIM,
                    device=scene.device, dtype=scene.dtype,
                ),
                (state_error / 100.0).reshape(batch, -1),
            ),
            dim=-1,
        )
        groups.append(
            self.event_projection(state_event)[:, None] + self.source_embedding[1]
        )
        masks.append((~valid_path.any(dim=1))[:, None])

        for history in range(HISTORY_CHUNKS):
            enabled = history < int(history_prefix)
            active = memory.history_valid[:, history] & enabled
            time = self.history_embedding[:, history]
            actual = self.anchor_projection(memory.actual_anchor_tokens[:, history])
            actual = actual + self.anchor_view_embedding + identity + time[:, None]
            groups.append(actual.reshape(batch, VIEWS * ANCHORS, -1))
            masks.append(
                (~(
                    memory.actual_anchor_valid[:, history]
                    & active[:, None, None]
                )).reshape(batch, -1)
            )
            error = self.innovation_projection(memory.visual_innovations[:, history])
            error = error + self.anchor_view_embedding + identity + time[:, None]
            groups.append(error.reshape(batch, VIEWS * ANCHORS, -1))
            masks.append(
                (~(
                    memory.innovation_valid[:, history]
                    & active[:, None, None]
                )).reshape(batch, -1)
            )
            event = torch.cat(
                (
                    (memory.completed_commands[:, history] / 100.0).reshape(batch, -1),
                    (memory.measured_paths[:, history] / 100.0).reshape(batch, -1),
                    (memory.state_innovations[:, history] / 100.0).reshape(batch, -1),
                ),
                dim=-1,
            )
            groups.append(
                self.event_projection(event)[:, None]
                + self.source_embedding[2]
                + time
            )
            masks.append((~active)[:, None])

        actor_memory = torch.cat(groups, dim=1)
        padding = torch.cat(masks, dim=1)
        queries = (
            self.action_queries.expand(batch, -1, -1) + self.action_time_embedding
        )
        causal = torch.triu(
            torch.ones(
                COMMAND_STEPS, COMMAND_STEPS,
                dtype=torch.bool, device=scene.device,
            ),
            diagonal=1,
        )
        traces = [queries]
        for layer in self.layers:
            queries = layer(
                queries,
                actor_memory,
                query_mask=causal,
                memory_padding_mask=padding,
            )
            traces.append(queries)
        queries = self.output_norm(queries)
        increments = torch.tanh(self.action_head(queries)) * self.step_cap
        desired_rows: list[torch.Tensor] = []
        previous = state
        for step in range(COMMAND_STEPS):
            previous = previous + increments[:, step]
            desired_rows.append(previous)
        desired = torch.stack(desired_rows, dim=1)
        commands = project_v175_action_trajectory(
            state,
            desired,
            step_cap=self.step_cap,
            action_lower=self.action_lower,
            action_upper=self.action_upper,
        )
        return V175ActorOutput(
            commands=commands,
            desired_commands=desired,
            increments=increments,
            action_queries_by_layer=torch.stack(traces, dim=1),
        )


class V175Controller(nn.Module):
    def __init__(
        self,
        config: V175Config = V175Config(),
        robot_spec: RobotSpec | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.fcpg = MorphologyConditionedKinematicFCPG(config, robot_spec)
        if (
            self.fcpg.robot_spec.joints != STATE_DIM
            or self.fcpg.robot_spec.anchors != ANCHORS
        ):
            raise V175ContractError(
                "the current LeWM/Actor stack is SO-101-specific; FCPG core remains generic"
            )
        self.lewm = V175GroundedLeWM(config)
        self.actor = V175Actor(config)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


__all__ = [
    "ANCHOR_DIM",
    "ANCHOR_ORDER",
    "ANCHORS",
    "COMMAND_STEPS",
    "COMPANION_STORAGE_SCALE",
    "FEATURE_GRID",
    "HISTORY_CHUNKS",
    "IMAGE_SIZE",
    "JOINT_POSITION_UNIT",
    "LOCAL_GRID",
    "PATH_STEPS",
    "STATE_DIM",
    "TIP_INDICES",
    "VIEW_ORDER",
    "VIEWS",
    "VIT_DIM",
    "VIT_PATCHES",
    "MorphologyConditionedKinematicFCPG",
    "V175Actor",
    "V175ActorOutput",
    "V175Config",
    "V175ContractError",
    "V175Controller",
    "V175FCPGMemory",
    "V175GroundedAnchorFCPG",
    "V175GroundedLeWM",
    "V175GroundedMeasurement",
    "V175Innovation",
    "V175LeWMPrediction",
    "V175Measurement",
    "V175Memory",
    "compute_v175_innovation",
    "ordered_anchor_smooth_l1",
    "project_v175_action_trajectory",
    "restore_companion_joint_units",
    "unordered_tip_reprojection_loss",
    "v175_action_contract",
]
