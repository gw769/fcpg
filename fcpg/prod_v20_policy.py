"""V20 policy and real-readback dynamics, independent of legacy model heads.

All public positions use ``lerobot_calibrated_position``.  They are NOT angles
in degrees.  Physical geometry conversion belongs to the shared observer
adapter, not to either network in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


POSITION_UNIT = "lerobot_calibrated_position"
JOINTS = 6
STEPS = 5
HISTORY = 3
VIEWS = 2
COMPONENTS = 4
from .v21_checks import checking, fail_if
from .v21_encoder import DIM as FEATURE_DIM   # 384 for DINOv2-S; 1024 for DINOv3-L (V21_ENCODER)
MODEL_DIM = 192
DEFAULT_LOWER = (-100.0, -100.0, -100.0, -100.0, -100.0, 0.0)
DEFAULT_UPPER = (100.0, 100.0, 100.0, 100.0, 100.0, 100.0)


class PolicyContractError(ValueError):
    """A V20 input violates a shape, unit declaration, or causal contract."""


def _tensor(name: str, value: Tensor, shape: tuple[int, ...]) -> Tensor:
    if not isinstance(value, Tensor) or tuple(value.shape) != shape:
        raise PolicyContractError(f"{name} must have shape {shape}")
    if not value.is_floating_point():
        raise PolicyContractError(f"{name} must be a finite floating tensor")
    if checking():
        fail_if(~torch.isfinite(value).all(), f"{name} must be a finite floating tensor", PolicyContractError)
    return value


def _mask(name: str, value: Tensor, shape: tuple[int, ...]) -> Tensor:
    if not isinstance(value, Tensor) or tuple(value.shape) != shape or value.dtype != torch.bool:
        raise PolicyContractError(f"{name} must be a bool tensor of shape {shape}")
    return value


@dataclass(frozen=True)
class ActualHistory:
    """Right-aligned, oldest-to-newest completed transitions, never predictions.

    Each readback path includes its source endpoint followed by the five actual
    command-synchronous readbacks.  ``time_deltas`` contains the five measured
    elapsed command intervals in seconds.  Invalid H0 slots are masked before
    encoding; a planned command is not a completed transition.
    """

    commands_position: Tensor  # [B,3,5,6], commands actually transmitted
    readbacks_position: Tensor  # [B,3,6,6], actual source + five endpoints
    scene_pooled: Tensor  # [B,3,2,384], source scene for the completed chunk
    time_deltas: Tensor  # [B,3,5], seconds
    valid: Tensor  # [B,3]

    def to(self, *args: object, **kwargs: object) -> "ActualHistory":
        commands = self.commands_position.to(*args, **kwargs)
        return ActualHistory(
            commands,
            self.readbacks_position.to(*args, **kwargs),
            self.scene_pooled.to(*args, **kwargs),
            self.time_deltas.to(*args, **kwargs),
            self.valid.to(device=commands.device),
        )


def empty_history(
    batch: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> ActualHistory:
    if batch <= 0 or not dtype.is_floating_point:
        raise PolicyContractError("history requires a positive batch and floating dtype")
    return ActualHistory(
        torch.zeros(batch, HISTORY, STEPS, JOINTS, device=device, dtype=dtype),
        torch.zeros(batch, HISTORY, STEPS + 1, JOINTS, device=device, dtype=dtype),
        torch.zeros(batch, HISTORY, VIEWS, FEATURE_DIM, device=device, dtype=dtype),
        torch.zeros(batch, HISTORY, STEPS, device=device, dtype=dtype),
        torch.zeros(batch, HISTORY, device=device, dtype=torch.bool),
    )


def _validate_history(history: ActualHistory, batch: int) -> None:
    if not isinstance(history, ActualHistory):
        raise PolicyContractError("history must be ActualHistory; use empty_history at H0")
    _tensor("history.commands_position", history.commands_position, (batch, HISTORY, STEPS, JOINTS))
    _tensor("history.readbacks_position", history.readbacks_position, (batch, HISTORY, STEPS + 1, JOINTS))
    _tensor("history.scene_pooled", history.scene_pooled, (batch, HISTORY, VIEWS, FEATURE_DIM))
    _tensor("history.time_deltas", history.time_deltas, (batch, HISTORY, STEPS))
    _mask("history.valid", history.valid, (batch, HISTORY))
    if checking():
        fail_if(torch.any(history.valid[..., None] & (history.time_deltas <= 0)),
                "completed readback intervals must be positive seconds", PolicyContractError)
    if checking():
        fail_if(torch.any(history.valid[:, :-1] & ~history.valid[:, 1:]),
                "actual history must be right-aligned without gaps", PolicyContractError)


def append_completed(
    history: ActualHistory,
    commands_position: Tensor,
    readbacks_position: Tensor,
    scene_pooled: Tensor,
    time_deltas: Tensor,
    completed: Tensor | None = None,
) -> ActualHistory:
    """Append one fully completed transaction without mutating existing state.

    ``readbacks_position`` must be measured, not LeWM predictions.  Callers must
    not invoke this function at proposal time.  Per-sample ``completed=False``
    preserves the previous history, including its ordering.  A partial chunk
    cannot be relabeled as a complete five-command transition.
    """
    batch = len(commands_position)
    _validate_history(history, batch)
    _tensor("commands_position", commands_position, (batch, STEPS, JOINTS))
    _tensor("readbacks_position", readbacks_position, (batch, STEPS + 1, JOINTS))
    _tensor("scene_pooled", scene_pooled, (batch, VIEWS, FEATURE_DIM))
    _tensor("time_deltas", time_deltas, (batch, STEPS))
    if completed is None:
        completed = torch.ones(batch, device=commands_position.device, dtype=torch.bool)
    _mask("completed", completed, (batch,))
    if checking():
        fail_if(torch.any(completed[:, None] & (time_deltas <= 0)),
                "a completed transaction needs five positive measured intervals", PolicyContractError)

    def push(old: Tensor, new: Tensor) -> Tensor:
        new = new.to(old).detach()
        shifted = torch.cat((old[:, 1:].detach(), new[:, None]), dim=1)
        condition = completed.to(old.device).reshape(batch, *([1] * (old.ndim - 1)))
        return torch.where(condition, shifted, old.detach()).clone()

    valid = push(history.valid, torch.ones(batch, device=history.valid.device, dtype=torch.bool))
    return ActualHistory(
        push(history.commands_position, commands_position),
        push(history.readbacks_position, readbacks_position),
        push(history.scene_pooled, scene_pooled),
        push(history.time_deltas, time_deltas),
        valid,
    )


class _PositionModule(nn.Module):
    def __init__(self, lower: Sequence[float], upper: Sequence[float], increment_scale: Sequence[float] | None):
        super().__init__()
        lo, hi = torch.as_tensor(lower, dtype=torch.float32), torch.as_tensor(upper, dtype=torch.float32)
        if lo.shape != (JOINTS,) or hi.shape != (JOINTS,) or not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or not (hi > lo).all():
            raise PolicyContractError("joint bounds must contain six finite strictly increasing ranges")
        # This is a feature/loss normalization, NEVER a per-command limiter.
        scale = (hi - lo) * 0.01 if increment_scale is None else torch.as_tensor(increment_scale, dtype=torch.float32)
        if scale.shape != (JOINTS,) or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise PolicyContractError("increment_scale must contain six positive finite values")
        self.register_buffer("position_lower", lo)
        self.register_buffer("position_upper", hi)
        self.register_buffer("position_center", (hi + lo) * 0.5)
        self.register_buffer("position_half_range", (hi - lo) * 0.5)
        self.register_buffer("increment_scale", scale)

    def normalize_position(self, q_position: Tensor) -> Tensor:
        return (q_position - self.position_center.to(q_position)) / self.position_half_range.to(q_position)

    def _motion(self, history: ActualHistory, reference: Tensor) -> Tensor:
        """Physical history summaries in explicit position/increment/error units."""
        commands = history.commands_position.to(reference)
        readback = history.readbacks_position.to(reference)
        times = history.time_deltas.to(reference)
        valid = history.valid.to(reference.device)
        # Invalid slots contribute zero rather than normalized fake gripper=0.
        source = self.normalize_position(readback[:, :, 0])
        command_delta = torch.diff(torch.cat((readback[:, :, :1], commands), dim=2), dim=2)
        measured_delta = torch.diff(readback, dim=2)
        servo_error = readback[:, :, 1:] - commands
        scale = self.increment_scale.to(reference)
        motion = torch.cat(
            (source, (command_delta / scale).flatten(2), (measured_delta / scale).flatten(2),
             (servo_error / scale).flatten(2), torch.log1p(times)), dim=-1,
        )
        return torch.where(valid[..., None], motion, torch.zeros_like(motion))


class _Encoder(nn.Module):
    def __init__(self, dim: int = MODEL_DIM, heads: int = 6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: Tensor, padding: Tensor | None = None, causal: Tensor | None = None) -> Tensor:
        normalized = self.norm1(x)
        x = x + self.attn(normalized, normalized, normalized, key_padding_mask=padding, attn_mask=causal, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(MODEL_DIM)
        self.self_attn = nn.MultiheadAttention(MODEL_DIM, 6, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(MODEL_DIM)
        self.memory_norm = nn.LayerNorm(MODEL_DIM)
        self.cross_attn = nn.MultiheadAttention(MODEL_DIM, 6, batch_first=True, dropout=0.0)
        self.norm3 = nn.LayerNorm(MODEL_DIM)
        self.ffn = nn.Sequential(nn.Linear(MODEL_DIM, 4 * MODEL_DIM), nn.GELU(), nn.Linear(4 * MODEL_DIM, MODEL_DIM))

    def forward(self, query: Tensor, memory: Tensor, causal: Tensor, memory_padding: Tensor) -> Tensor:
        normalized = self.norm1(query)
        query = query + self.self_attn(normalized, normalized, normalized, attn_mask=causal, need_weights=False)[0]
        mem = self.memory_norm(memory)
        query = query + self.cross_attn(self.norm2(query), mem, mem, key_padding_mask=memory_padding, need_weights=False)[0]
        return query + self.ffn(self.norm3(query))


@dataclass(frozen=True)
class LeWMOutput:
    predicted_q_path: Tensor  # [B,6,6], source q is exact
    q_uncertainty: Tensor  # [B,5,6], positive position-unit standard deviation


class V20LeWM(_PositionModule):
    """Two-layer, 192D response model; no image, task, pixel head, IK or FK.

    Context is strictly completed history and current q.  Future command tokens
    are causal: prediction j cannot depend on commands or time deltas after j.
    The readback residual is relative to the actual command, not a free visual
    displacement.  Geometry projection of this result is the observer's job.
    """

    def __init__(self, *, lower: Sequence[float] = DEFAULT_LOWER, upper: Sequence[float] = DEFAULT_UPPER,
                 increment_scale: Sequence[float] | None = None):
        super().__init__(lower, upper, increment_scale)
        motion_dim = JOINTS + 3 * STEPS * JOINTS + STEPS
        self.history_projection = nn.Sequential(nn.Linear(motion_dim, MODEL_DIM), nn.GELU(), nn.Linear(MODEL_DIM, MODEL_DIM))
        self.joint_projection = nn.Sequential(nn.Linear(1, MODEL_DIM), nn.GELU(), nn.Linear(MODEL_DIM, MODEL_DIM))
        self.command_projection = nn.Sequential(nn.Linear(2 * JOINTS + 1, MODEL_DIM), nn.GELU(), nn.Linear(MODEL_DIM, MODEL_DIM))
        self.joint_embedding = nn.Parameter(torch.randn(1, JOINTS, MODEL_DIM) * 0.02)
        self.history_embedding = nn.Parameter(torch.randn(1, HISTORY, MODEL_DIM) * 0.02)
        self.command_embedding = nn.Parameter(torch.randn(1, STEPS, MODEL_DIM) * 0.02)
        self.layers = nn.ModuleList([_Encoder(), _Encoder()])
        self.output_norm = nn.LayerNorm(MODEL_DIM)
        self.response_head = nn.Linear(MODEL_DIM, JOINTS)
        self.uncertainty_head = nn.Linear(MODEL_DIM, JOINTS)
        nn.init.normal_(self.response_head.weight, std=0.005)
        nn.init.zeros_(self.response_head.bias)

    def forward(self, q_position: Tensor, commands_position: Tensor, actual_history: ActualHistory,
                time_deltas: Tensor) -> LeWMOutput:
        batch = len(q_position)
        q = _tensor("q_position", q_position, (batch, JOINTS)).to(self.joint_projection[0].weight)
        commands = _tensor("commands_position", commands_position, (batch, STEPS, JOINTS)).to(q)
        dt = _tensor("time_deltas", time_deltas, (batch, STEPS)).to(q)
        if torch.any(dt <= 0):
            raise PolicyContractError("planned command intervals must be positive seconds")
        _validate_history(actual_history, batch)
        history = self.history_projection(self._motion(actual_history, q)) + self.history_embedding
        joints = self.joint_projection(self.normalize_position(q)[..., None]) + self.joint_embedding
        increments = torch.diff(torch.cat((q[:, None], commands), dim=1), dim=1)
        relative = commands - q[:, None]
        cmd_input = torch.cat((increments / self.increment_scale, relative / self.position_half_range, torch.log1p(dt)[..., None]), dim=-1)
        future = self.command_projection(cmd_input) + self.command_embedding
        tokens = torch.cat((joints, history, future), dim=1)
        context_count = JOINTS + HISTORY
        length = context_count + STEPS
        causal = torch.zeros(length, length, device=q.device, dtype=torch.bool)
        causal[:context_count, context_count:] = True
        causal[context_count:, context_count:] = torch.triu(torch.ones(STEPS, STEPS, device=q.device, dtype=torch.bool), diagonal=1)
        padding = torch.cat((torch.zeros(batch, JOINTS, device=q.device, dtype=torch.bool),
                             ~actual_history.valid.to(q.device), torch.zeros(batch, STEPS, device=q.device, dtype=torch.bool)), dim=1)
        for layer in self.layers:
            tokens = layer(tokens, padding=padding, causal=causal)
        hidden = self.output_norm(tokens[:, -STEPS:])
        predicted = commands + self.response_head(hidden) * self.increment_scale
        uncertainty = (F.softplus(self.uncertainty_head(hidden).float()) + 1e-4).to(q) * self.increment_scale
        # Preserve the original readback exactly, including float32 under AMP.
        predicted = predicted.to(q_position)
        return LeWMOutput(torch.cat((q_position[:, None], predicted), dim=1), uncertainty.to(q_position))

    predict = forward


@dataclass(frozen=True)
class ActorOutput:
    commands_position: Tensor  # [B,5,6], bounded absolute positions
    desired_commands_position: Tensor  # raw cumulative prediction
    increments: Tensor  # raw joint increments, no artificial per-step bound


def project_joint_bounds(commands_position: Tensor, lower: Tensor | Sequence[float] = DEFAULT_LOWER,
                         upper: Tensor | Sequence[float] = DEFAULT_UPPER) -> Tensor:
    """Absolute mechanical range projection only; it is a fixed point on legal commands."""
    lo = torch.as_tensor(lower, device=commands_position.device, dtype=commands_position.dtype)
    hi = torch.as_tensor(upper, device=commands_position.device, dtype=commands_position.dtype)
    if lo.shape != (JOINTS,) or hi.shape != (JOINTS,):
        raise PolicyContractError("invalid projection bounds")
    if checking():
        fail_if(~torch.isfinite(lo).all() | ~torch.isfinite(hi).all() | ~(hi > lo).all(), "invalid projection bounds",
                PolicyContractError)
    if commands_position.shape[-1] != JOINTS:
        raise PolicyContractError("commands must be finite six-joint positions")
    if checking():
        fail_if(~torch.isfinite(commands_position).all(), "commands must be finite six-joint positions", PolicyContractError)
    return torch.maximum(torch.minimum(commands_position, hi), lo)


class V20Actor(_PositionModule):
    """Small historical Actor with paired belief/innovation and frozen task features."""

    def __init__(self, *, lower: Sequence[float] = DEFAULT_LOWER, upper: Sequence[float] = DEFAULT_UPPER,
                 increment_scale: Sequence[float] | None = None):
        super().__init__(lower, upper, increment_scale)
        motion_dim = JOINTS + 3 * STEPS * JOINTS + STEPS
        self.scene_projection = nn.Linear(FEATURE_DIM, MODEL_DIM)
        self.track_projection = nn.Sequential(nn.LayerNorm(2 * MODEL_DIM), nn.Linear(2 * MODEL_DIM, MODEL_DIM), nn.GELU(), nn.Linear(MODEL_DIM, MODEL_DIM))
        self.history_motion_projection = nn.Sequential(nn.Linear(motion_dim, MODEL_DIM), nn.GELU(), nn.Linear(MODEL_DIM, MODEL_DIM))
        self.history_scene_projection = nn.Linear(FEATURE_DIM, MODEL_DIM)
        self.q_projection = nn.Sequential(nn.Linear(JOINTS, MODEL_DIM), nn.GELU(), nn.Linear(MODEL_DIM, MODEL_DIM))
        self.task_projection = nn.Sequential(nn.LayerNorm(FEATURE_DIM), nn.Linear(FEATURE_DIM, MODEL_DIM), nn.GELU(), nn.Linear(MODEL_DIM, MODEL_DIM))
        self.view_embedding = nn.Parameter(torch.randn(1, VIEWS, 1, MODEL_DIM) * 0.02)
        self.scene_position = nn.Parameter(torch.randn(1, 1, 256, MODEL_DIM) * 0.02)
        self.kind_embedding = nn.Parameter(torch.randn(5, MODEL_DIM) * 0.02)
        self.history_time_embedding = nn.Parameter(torch.randn(1, HISTORY, 1, MODEL_DIM) * 0.02)
        self.history_kind_embedding = nn.Parameter(torch.randn(1, 1, 3, MODEL_DIM) * 0.02)
        self.context_layers = nn.ModuleList([_Encoder(), _Encoder()])
        self.context_norm = nn.LayerNorm(MODEL_DIM)
        self.action_queries = nn.Parameter(torch.randn(1, STEPS, MODEL_DIM) * 0.02)
        self.layers = nn.ModuleList([_Decoder() for _ in range(4)])
        self.output_norm = nn.LayerNorm(MODEL_DIM)
        self.action_head = nn.Linear(MODEL_DIM, JOINTS)
        # A small nonzero initialization allows every input route to receive
        # gradients on update one.  Zero initialization hid input-use defects.
        nn.init.normal_(self.action_head.weight, std=0.005)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, scene_tokens: Tensor, grounding_tracks: Tensor, innovation_tracks: Tensor,
                q_position: Tensor, history: ActualHistory, task_embedding: Tensor,
                view_valid: Tensor | None = None) -> ActorOutput:
        batch = len(q_position)
        scene = _tensor("scene_tokens", scene_tokens, (batch, VIEWS, 256, FEATURE_DIM)).to(self.scene_projection.weight)
        grounding = _tensor("grounding_tracks", grounding_tracks, (batch, VIEWS, COMPONENTS, MODEL_DIM)).to(scene)
        innovation = _tensor("innovation_tracks", innovation_tracks, (batch, VIEWS, COMPONENTS, MODEL_DIM)).to(scene)
        q = _tensor("q_position", q_position, (batch, JOINTS)).to(scene)
        task = _tensor("task_embedding", task_embedding, (batch, FEATURE_DIM)).to(scene)
        _validate_history(history, batch)
        if view_valid is None:
            view_valid = torch.ones(batch, VIEWS, device=scene.device, dtype=torch.bool)
        valid = _mask("view_valid", view_valid, (batch, VIEWS)).to(scene.device)

        scene_memory = (self.scene_projection(F.layer_norm(scene, (FEATURE_DIM,))) + self.scene_position
                        + self.view_embedding + self.kind_embedding[0]).flatten(1, 2)
        # No mixture-component index embedding: jointly permuting G and E is
        # invariant.  Concatenation FIRST retains which error belongs to G.
        tracks = (self.track_projection(torch.cat((grounding, innovation), dim=-1))
                  + self.view_embedding + self.kind_embedding[1]).flatten(1, 2)
        motion = self.history_motion_projection(self._motion(history, scene))[:, :, None]
        pooled = F.layer_norm(history.scene_pooled.to(scene), (FEATURE_DIM,))
        historic_scene = self.history_scene_projection(pooled)
        history_tokens = (torch.cat((historic_scene, motion), dim=2) + self.history_time_embedding
                          + self.history_kind_embedding + self.kind_embedding[2]).flatten(1, 2)
        state = (self.q_projection(self.normalize_position(q)) + self.kind_embedding[3])[:, None]
        task_token = (self.task_projection(task) + self.kind_embedding[4])[:, None]
        compact = torch.cat((tracks, history_tokens, state, task_token), dim=1)
        compact_padding = torch.cat((~valid[..., None].expand(-1, -1, COMPONENTS).flatten(1),
                                     ~history.valid.to(scene.device)[..., None].expand(-1, -1, 3).flatten(1),
                                     torch.zeros(batch, 2, device=scene.device, dtype=torch.bool)), dim=1)
        for layer in self.context_layers:
            compact = layer(compact, padding=compact_padding)
        memory = torch.cat((scene_memory, self.context_norm(compact)), dim=1)
        memory_padding = torch.cat((~valid[..., None].expand(-1, -1, 256).flatten(1), compact_padding), dim=1)
        query = self.action_queries.expand(batch, -1, -1)
        causal = torch.triu(torch.ones(STEPS, STEPS, device=scene.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            query = layer(query, memory, causal, memory_padding)
        increments = self.action_head(self.output_norm(query))
        desired = q[:, None] + increments.cumsum(dim=1)
        commands = project_joint_bounds(desired, self.position_lower, self.position_upper)
        return ActorOutput(commands, desired, increments)

    act = forward


def _loss_scale(scale: Tensor | Sequence[float], reference: Tensor) -> Tensor:
    result = torch.as_tensor(scale, device=reference.device, dtype=reference.dtype)
    if result.shape != (JOINTS,) or not torch.isfinite(result).all() or not (result > 0).all():
        raise PolicyContractError("loss scale must be six positive finite archive increment statistics")
    return result


def action_loss_terms(predicted_commands: Tensor, target_commands: Tensor, q_position: Tensor,
                      increment_scale: Tensor | Sequence[float], sample_weights: Tensor | None = None) -> dict[str, Tensor]:
    """Natural-unit action loss; scale is archive statistics, never an actuator cap."""
    batch = len(predicted_commands)
    pred = _tensor("predicted_commands", predicted_commands, (batch, STEPS, JOINTS))
    target = _tensor("target_commands", target_commands, (batch, STEPS, JOINTS)).to(pred).detach()
    q = _tensor("q_position", q_position, (batch, JOINTS)).to(pred).detach()
    scale = _loss_scale(increment_scale, pred)
    action_per_sample = F.smooth_l1_loss(pred / scale, target / scale, reduction="none").mean(dim=(1, 2))
    predicted_velocity = torch.diff(torch.cat((q[:, None], pred), dim=1), dim=1)
    target_velocity = torch.diff(torch.cat((q[:, None], target), dim=1), dim=1)
    velocity_per_sample = F.smooth_l1_loss(predicted_velocity / scale, target_velocity / scale, reduction="none").mean(dim=(1, 2))
    if sample_weights is None:
        weights = torch.ones(batch, device=pred.device, dtype=pred.dtype)
    else:
        weights = torch.as_tensor(sample_weights, device=pred.device, dtype=pred.dtype).reshape(-1)
        if weights.shape != (batch,) or not torch.isfinite(weights).all() or bool((weights <= 0).any()):
            raise PolicyContractError("sample_weights must be finite positive per-sample values")
    weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps) * batch
    action = (action_per_sample * weights).mean()
    velocity = (velocity_per_sample * weights).mean()
    return {"loss": action + 0.2 * velocity, "action": action, "velocity": velocity}


def normalized_action_loss(predicted_commands: Tensor, target_commands: Tensor, q_position: Tensor,
                           increment_scale: Tensor | Sequence[float], sample_weights: Tensor | None = None) -> Tensor:
    return action_loss_terms(predicted_commands, target_commands, q_position, increment_scale, sample_weights)["loss"]


def future_q_path_loss(predicted_q_path: Tensor, measured_q_path: Tensor,
                       increment_scale: Tensor | Sequence[float], q_uncertainty: Tensor | None = None) -> Tensor:
    """Supervise q1..q5 only against detached actual readback, not source q0.

    If uncertainty is supplied, a normalized Gaussian NLL calibrates its scale
    alongside SmoothL1.  This uncertainty is descriptive; it is not a gate.
    """
    batch = len(predicted_q_path)
    pred = _tensor("predicted_q_path", predicted_q_path, (batch, STEPS + 1, JOINTS))[:, 1:]
    target = _tensor("measured_q_path", measured_q_path, (batch, STEPS + 1, JOINTS)).to(pred).detach()[:, 1:]
    scale = _loss_scale(increment_scale, pred)
    base = F.smooth_l1_loss(pred / scale, target / scale)
    if q_uncertainty is None:
        return base
    uncertainty = _tensor("q_uncertainty", q_uncertainty, (batch, STEPS, JOINTS)).to(pred)
    if torch.any(uncertainty <= 0):
        raise PolicyContractError("q uncertainty must be positive")
    sigma = (uncertainty.float() / scale.float()).clamp_min(1e-4)
    residual = (pred.float() - target.float()) / scale.float()
    return base + (0.5 * (residual / sigma).square() + sigma.log()).mean()
