"""Prod-V16.5 physical world model and structured-goal action proposer.

This module contains only the two trainable downstream branches introduced by
Prod-V16.5:

* :class:`StateFKLeWM` predicts measured joint results and evaluates those
  results through fixed robot kinematics; and
* :class:`StructuredGoalFiveCommandActor` proposes a bounded five-command
  residual around an explicit, auditable action baseline.

The frozen ViT and the FK-seeded local FCPG deliberately do not live here.
Every external tensor is detached at the module boundary.  The two branches
have separate stage helpers, and :class:`V165Core` is intentionally *not* an
``nn.Module`` so that a holistic optimizer cannot be constructed by accident.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from fcpg.prod_v16_2 import project_bounded_action_trajectory


STATE_DIM = 6
COMMAND_STEPS = 5
PATH_STEPS = COMMAND_STEPS + 1
VIEWS = 2
PATCHES = 256
TOKEN_DIM = 192
ANCHORS = 3

LEWM_CHECKPOINT_FORMAT = "so101-prod-v16-5-state-fk-lewm-v1"
ACTOR_CHECKPOINT_FORMAT = "so101-prod-v16-5-structured-goal-actor-v1"

TrainingStage = Literal["lewm", "actor", "frozen"]
BaselineKind = Literal["analytic", "repeat", "damped"]
FKCallback = Callable[[torch.Tensor], Any]
DAMPED_BASELINE_DECAY = 0.9


def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _detached_float(value: Any, *, device: torch.device | None = None) -> torch.Tensor:
    return torch.as_tensor(value, device=device).detach().float()


def _validate_action_contract(
    step_cap: torch.Tensor,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cap = torch.as_tensor(step_cap).float()
    lower = torch.as_tensor(action_lower).float()
    upper = torch.as_tensor(action_upper).float()
    for name, value in (
        ("step_cap", cap),
        ("action_lower", lower),
        ("action_upper", upper),
    ):
        if value.shape != (STATE_DIM,):
            raise ValueError(f"{name} must be one six-joint vector")
        _finite(name, value)
    if bool((cap <= 0).any()) or bool((lower >= upper).any()):
        raise ValueError("V16.5 action contract is invalid")
    return cap, lower, upper


def _validate_state_history(
    current_state: torch.Tensor,
    past_completed_commands: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = _detached_float(current_state)
    past = _detached_float(past_completed_commands, device=state.device)
    if state.ndim != 2 or state.shape[-1] != STATE_DIM:
        raise ValueError("current_state must be [B,6]")
    if past.shape != (len(state), 2, STATE_DIM):
        raise ValueError("past_completed_commands must be [B,2,6]")
    _finite("current_state", state)
    _finite("past_completed_commands", past)
    return state, past


def _canonical_candidate_bank(
    candidate_commands: torch.Tensor,
    *,
    batch: int,
    device: torch.device,
) -> tuple[torch.Tensor, bool]:
    commands = _detached_float(candidate_commands, device=device)
    if commands.shape == (batch, COMMAND_STEPS, STATE_DIM):
        return commands[:, None], False
    if (
        commands.ndim == 4
        and commands.shape[0] == batch
        and commands.shape[2:] == (COMMAND_STEPS, STATE_DIM)
        and commands.shape[1] > 0
    ):
        return commands, True
    raise ValueError("candidate_commands must be [B,5,6] or [B,C,5,6]")


def _project_candidate_bank(
    current_state: torch.Tensor,
    candidate_commands: torch.Tensor,
    *,
    step_cap: torch.Tensor,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
) -> torch.Tensor:
    batch, candidates = candidate_commands.shape[:2]
    roots = current_state[:, None].expand(-1, candidates, -1).reshape(
        batch * candidates, STATE_DIM
    )
    desired = candidate_commands.reshape(
        batch * candidates, COMMAND_STEPS, STATE_DIM
    )
    return project_bounded_action_trajectory(
        roots,
        desired,
        step_cap=step_cap,
        action_lower=action_lower,
        action_upper=action_upper,
    ).reshape(batch, candidates, COMMAND_STEPS, STATE_DIM)


def commanded_state_baseline(
    current_state: torch.Tensor,
    candidate_commands: torch.Tensor,
    *,
    step_cap: torch.Tensor,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
) -> torch.Tensor:
    """Return the deterministic command/hold state baseline.

    Commands are absolute targets.  They are projected sequentially through
    the same joint and step limits used at runtime.  An all-hold command path
    is therefore an exact hold path without a learned threshold.
    """

    state = _detached_float(current_state)
    if state.ndim != 2 or state.shape[-1] != STATE_DIM:
        raise ValueError("current_state must be [B,6]")
    commands, had_candidate_axis = _canonical_candidate_bank(
        candidate_commands, batch=len(state), device=state.device
    )
    cap, lower, upper = _validate_action_contract(
        step_cap, action_lower, action_upper
    )
    projected = _project_candidate_bank(
        state,
        commands,
        step_cap=cap.to(state),
        action_lower=lower.to(state),
        action_upper=upper.to(state),
    )
    return projected if had_candidate_axis else projected[:, 0]


def goal_interpolation_five_command_baseline(
    current_state: torch.Tensor,
    desired_joint_goal: torch.Tensor,
    *,
    step_cap: torch.Tensor,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
) -> torch.Tensor:
    """Linearly approach an explicit joint goal under the robot contract."""

    state = _detached_float(current_state)
    goal = _detached_float(desired_joint_goal, device=state.device)
    if state.ndim != 2 or state.shape[-1] != STATE_DIM:
        raise ValueError("current_state must be [B,6]")
    if goal.shape != state.shape:
        raise ValueError("desired_joint_goal must be [B,6]")
    cap, lower, upper = _validate_action_contract(
        step_cap, action_lower, action_upper
    )
    cap, lower, upper = cap.to(state), lower.to(state), upper.to(state)
    _finite("current_state", state)
    _finite("desired_joint_goal", goal)
    if bool(((goal < lower) | (goal > upper)).any()):
        raise ValueError("desired_joint_goal lies outside the action contract")
    fraction = torch.linspace(
        1.0 / COMMAND_STEPS,
        1.0,
        COMMAND_STEPS,
        device=state.device,
        dtype=state.dtype,
    ).view(1, COMMAND_STEPS, 1)
    desired = state[:, None] + fraction * (goal - state)[:, None]
    return project_bounded_action_trajectory(
        state,
        desired,
        step_cap=cap,
        action_lower=lower,
        action_upper=upper,
    )


def repeat_five_command_baseline(
    current_state: torch.Tensor,
    past_completed_commands: torch.Tensor,
    *,
    step_cap: torch.Tensor,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
) -> torch.Tensor:
    """Repeat the last completed command through the canonical projector."""

    state, past = _validate_state_history(
        current_state, past_completed_commands
    )
    desired = past[:, -1:, :].expand(-1, COMMAND_STEPS, -1)
    return project_bounded_action_trajectory(
        state,
        desired,
        step_cap=step_cap,
        action_lower=action_lower,
        action_upper=action_upper,
    )


def damped_five_command_baseline(
    current_state: torch.Tensor,
    past_completed_commands: torch.Tensor,
    *,
    step_cap: torch.Tensor,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
) -> torch.Tensor:
    """Continue the latest completed-command increment with fixed decay."""

    state, past = _validate_state_history(
        current_state, past_completed_commands
    )
    cap, lower, upper = _validate_action_contract(
        step_cap, action_lower, action_upper
    )
    cap, lower, upper = cap.to(state), lower.to(state), upper.to(state)
    velocity = torch.clamp(past[:, -1] - past[:, -2], min=-cap, max=cap)
    current = state
    result = []
    for step in range(COMMAND_STEPS):
        desired = current + velocity * (DAMPED_BASELINE_DECAY**step)
        current = project_bounded_action_trajectory(
            current,
            desired[:, None],
            step_cap=cap,
            action_lower=lower,
            action_upper=upper,
        )[:, 0]
        result.append(current)
    return torch.stack(result, dim=1)


def structured_goal_actor_baseline(
    *,
    baseline_kind: BaselineKind,
    current_state: torch.Tensor,
    past_completed_commands: torch.Tensor,
    baseline_joint_goal: torch.Tensor | None,
    step_cap: torch.Tensor,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
) -> torch.Tensor:
    """Construct one of the three auditable V16.5 actor baselines."""

    if baseline_kind == "analytic":
        if baseline_joint_goal is None:
            raise ValueError("analytic baseline requires baseline_joint_goal")
        return goal_interpolation_five_command_baseline(
            current_state,
            baseline_joint_goal,
            step_cap=step_cap,
            action_lower=action_lower,
            action_upper=action_upper,
        )
    if baseline_joint_goal is not None:
        raise ValueError(
            "baseline_joint_goal is only valid for the analytic baseline"
        )
    if baseline_kind == "repeat":
        return repeat_five_command_baseline(
            current_state,
            past_completed_commands,
            step_cap=step_cap,
            action_lower=action_lower,
            action_upper=action_upper,
        )
    if baseline_kind == "damped":
        return damped_five_command_baseline(
            current_state,
            past_completed_commands,
            step_cap=step_cap,
            action_lower=action_lower,
            action_upper=action_upper,
        )
    raise ValueError("baseline_kind must be analytic, repeat, or damped")


@dataclass(frozen=True)
class StateFKLeWMOutput:
    predicted_state_path: torch.Tensor
    baseline_state_path: torch.Tensor
    residual_state_path: torch.Tensor


class StateFKLeWM(nn.Module):
    """Causal measured-state residual model with fixed-FK readout.

    The model sees a measured ``q0``, two already-completed command endpoints,
    and one or more candidate five-command chunks.  It never accepts future
    images or future measured states.  It has no visual branch.  The command
    baseline obeys the controller's per-command step cap; the predicted
    measured response is a bounded residual around that baseline and is only
    clamped to the physical joint bounds.  FK is kept outside ``forward`` as
    a gradient-disabled evaluation adapter.
    """

    def __init__(
        self,
        *,
        step_cap: torch.Tensor,
        action_lower: torch.Tensor,
        action_upper: torch.Tensor,
        residual_cap: torch.Tensor | float = 0.05,
        dimension: int = TOKEN_DIM,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if int(dimension) <= 0 or int(dimension) % 6:
            raise ValueError("StateFKLeWM dimension must be positive and divisible by 6")
        if int(layers) <= 0:
            raise ValueError("StateFKLeWM layers must be positive")
        cap, lower, upper = _validate_action_contract(
            step_cap, action_lower, action_upper
        )
        residual = torch.as_tensor(residual_cap).float()
        if residual.ndim == 0:
            residual = residual.expand(STATE_DIM).clone()
        if residual.shape != (STATE_DIM,) or not bool(torch.isfinite(residual).all()):
            raise ValueError("residual_cap must be finite scalar or [6]")
        if bool((residual <= 0).any()):
            raise ValueError("residual_cap must be positive")
        self.dimension = int(dimension)
        self.register_buffer("step_cap", cap.clone())
        self.register_buffer("action_lower", lower.clone())
        self.register_buffer("action_upper", upper.clone())
        self.register_buffer("residual_cap", residual.clone())

        # q0, two causal command endpoints, current/previous command and
        # deltas, plus the deterministic state baseline and its increment.
        feature_dimension = 8 * STATE_DIM
        self.input_norm = nn.LayerNorm(feature_dimension)
        self.input_projection = nn.Linear(feature_dimension, self.dimension)
        self.step_embedding = nn.Parameter(
            torch.empty(1, COMMAND_STEPS, self.dimension)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.dimension,
            nhead=6,
            dim_feedforward=4 * self.dimension,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(layers),
            norm=nn.LayerNorm(self.dimension),
            enable_nested_tensor=False,
        )
        self.residual_head = nn.Linear(self.dimension, STATE_DIM)
        nn.init.normal_(self.step_embedding, std=0.02)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        *,
        current_state: torch.Tensor,
        past_completed_commands: torch.Tensor,
        candidate_commands: torch.Tensor,
    ) -> StateFKLeWMOutput:
        state, past = _validate_state_history(
            current_state, past_completed_commands
        )
        commands, had_candidate_axis = _canonical_candidate_bank(
            candidate_commands, batch=len(state), device=state.device
        )
        _finite("candidate_commands", commands)
        batch, candidates = commands.shape[:2]
        baseline = _project_candidate_bank(
            state,
            commands,
            step_cap=self.step_cap,
            action_lower=self.action_lower,
            action_upper=self.action_upper,
        )

        roots = state[:, None].expand(-1, candidates, -1)
        history = past[:, None].expand(-1, candidates, -1, -1)
        previous_command = torch.cat(
            (history[:, :, -1:, :], commands[:, :, :-1, :]), dim=2
        )
        previous_baseline = torch.cat(
            (roots[:, :, None, :], baseline[:, :, :-1, :]), dim=2
        )
        features = torch.cat(
            (
                roots[:, :, None, :].expand(-1, -1, COMMAND_STEPS, -1),
                history.flatten(start_dim=2)[:, :, None, :].expand(
                    -1, -1, COMMAND_STEPS, -1
                ),
                commands,
                previous_command,
                commands - previous_command,
                baseline,
                baseline - previous_baseline,
            ),
            dim=-1,
        )
        hidden = self.input_projection(self.input_norm(features))
        hidden = hidden + self.step_embedding[:, None]
        hidden = hidden.reshape(
            batch * candidates, COMMAND_STEPS, self.dimension
        )
        causal = nn.Transformer.generate_square_subsequent_mask(
            COMMAND_STEPS, device=hidden.device, dtype=hidden.dtype
        )
        hidden = self.encoder(hidden, mask=causal)
        raw = torch.tanh(self.residual_head(hidden)).reshape(
            batch, candidates, COMMAND_STEPS, STATE_DIM
        )
        desired = baseline + raw * self.residual_cap.view(1, 1, 1, -1)
        predicted = torch.maximum(
            torch.minimum(desired, self.action_upper.view(1, 1, 1, -1)),
            self.action_lower.view(1, 1, 1, -1),
        )
        state_root = state[:, None, None].expand(-1, candidates, -1, -1)
        predicted_path = torch.cat((state_root, predicted), dim=2)
        baseline_path = torch.cat((state_root, baseline), dim=2)
        residual_path = predicted - baseline

        if not had_candidate_axis:
            predicted_path = predicted_path[:, 0]
            baseline_path = baseline_path[:, 0]
            residual_path = residual_path[:, 0]
        return StateFKLeWMOutput(
            predicted_state_path=predicted_path,
            baseline_state_path=baseline_path,
            residual_state_path=residual_path,
        )


@dataclass(frozen=True)
class StateFKLeWMLoss:
    total: torch.Tensor
    state: torch.Tensor


def state_fk_lewm_loss(
    output: StateFKLeWMOutput,
    measured_state_path: torch.Tensor,
) -> StateFKLeWMLoss:
    """Factual measured-state loss; FK is never part of the optimizer.

    Candidate-bank outputs are deployment objects and cannot be passed to the
    factual training loss.  The state target is only the measured q0..q5 path.
    """

    prediction = output.predicted_state_path
    truth = torch.as_tensor(
        measured_state_path, device=prediction.device, dtype=prediction.dtype
    ).detach()
    if prediction.ndim != 3 or prediction.shape[1:] != (PATH_STEPS, STATE_DIM):
        raise ValueError("state_fk_lewm_loss requires a single factual chunk")
    if truth.shape != prediction.shape:
        raise ValueError("measured_state_path must match predicted [B,6,6]")
    _finite("measured_state_path", truth)
    if not torch.allclose(prediction[:, 0], truth[:, 0], rtol=0.0, atol=1.0e-6):
        raise ValueError("measured_state_path q0 differs from the model acquisition")
    state_loss = F.smooth_l1_loss(prediction[:, 1:], truth[:, 1:])
    return StateFKLeWMLoss(total=state_loss, state=state_loss)


@dataclass(frozen=True)
class StateFKLeWMFKMetrics:
    base_smooth_l1_m: torch.Tensor
    tool_smooth_l1_m: torch.Tensor
    base_epe_m: torch.Tensor
    tool_epe_m: torch.Tensor
    base_epe_per_row_m: torch.Tensor
    tool_epe_per_row_m: torch.Tensor


def _parse_fk_metric_result(
    result: Any,
    *,
    expected_rows: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(result, Mapping):
        if set(result) != {"base", "tool"}:
            raise ValueError("fk_callback mapping must contain only base and tool")
        base_value, tool_value = result["base"], result["tool"]
    elif isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        if len(result) != 2:
            raise ValueError("fk_callback must return exactly base and tool frames")
        base_value, tool_value = result
    else:
        raise ValueError("fk_callback must return (base, tool) anchor frames")
    base = torch.as_tensor(base_value, device=device, dtype=dtype)
    tool = torch.as_tensor(tool_value, device=device, dtype=dtype)
    expected = (expected_rows, ANCHORS, 3)
    if base.shape != expected or tool.shape != expected:
        raise ValueError(f"fk_callback outputs must both be {expected}")
    _finite("base FK anchors", base)
    _finite("tool FK anchors", tool)
    return base, tool


def _select_fk_metric_anchors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    anchor_indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    truth = torch.as_tensor(
        target, device=prediction.device, dtype=prediction.dtype
    ).detach()
    if (
        truth.ndim != 4
        or truth.shape[:2] != prediction.shape[:2]
        or truth.shape[-1] != 3
    ):
        raise ValueError("measured FK path must be [B,6,K,3]")
    _finite("measured FK path", truth)
    indices = tuple(int(index) for index in anchor_indices)
    if len(indices) != truth.shape[2] or len(set(indices)) != len(indices):
        raise ValueError("anchor_indices must uniquely align every FK target anchor")
    if any(index < 0 or index >= ANCHORS for index in indices):
        raise ValueError("anchor_indices lie outside the three-anchor FK path")
    return prediction[:, :, indices], truth


@torch.no_grad()
def state_fk_lewm_fk_metrics(
    output: StateFKLeWMOutput,
    *,
    fk_callback: FKCallback,
    measured_base_anchor_path: torch.Tensor,
    measured_tool_anchor_path: torch.Tensor,
    anchor_indices: Sequence[int] = (1, 2),
    canonical_to_degrees_scale: float = 100.0,
) -> StateFKLeWMFKMetrics:
    """Evaluate fixed three-anchor FK without adding an optimizer path.

    V16.5 joint states are canonical normalized commands.  The production FK
    implementation consumes LeRobot calibration degrees, so the conversion is
    explicit and defaults to the dataset's fixed ``100`` scale.
    """

    path = output.predicted_state_path.detach()
    if path.ndim != 3 or path.shape[1:] != (PATH_STEPS, STATE_DIM):
        raise ValueError("FK metrics require one factual [B,6,6] state path")
    scale = float(canonical_to_degrees_scale)
    if not torch.isfinite(torch.tensor(scale)) or scale <= 0:
        raise ValueError("canonical_to_degrees_scale must be finite and positive")
    physical = (path * scale).reshape(-1, STATE_DIM)
    base, tool = _parse_fk_metric_result(
        fk_callback(physical),
        expected_rows=len(physical),
        device=path.device,
        dtype=path.dtype,
    )
    base = base.reshape(len(path), PATH_STEPS, ANCHORS, 3)
    tool = tool.reshape(len(path), PATH_STEPS, ANCHORS, 3)
    base, base_truth = _select_fk_metric_anchors(
        base, measured_base_anchor_path, anchor_indices
    )
    tool, tool_truth = _select_fk_metric_anchors(
        tool, measured_tool_anchor_path, anchor_indices
    )
    base_delta = base[:, 1:] - base_truth[:, 1:]
    tool_delta = tool[:, 1:] - tool_truth[:, 1:]
    base_epe_per_row = torch.linalg.vector_norm(base_delta, dim=-1).mean(
        dim=(1, 2)
    )
    tool_epe_per_row = torch.linalg.vector_norm(tool_delta, dim=-1).mean(
        dim=(1, 2)
    )
    return StateFKLeWMFKMetrics(
        base_smooth_l1_m=F.smooth_l1_loss(base[:, 1:], base_truth[:, 1:]),
        tool_smooth_l1_m=F.smooth_l1_loss(tool[:, 1:], tool_truth[:, 1:]),
        base_epe_m=base_epe_per_row.mean(),
        tool_epe_m=tool_epe_per_row.mean(),
        base_epe_per_row_m=base_epe_per_row,
        tool_epe_per_row_m=tool_epe_per_row,
    )


@torch.no_grad()
def state_fk_lewm_eval_controls(
    current_state: torch.Tensor,
    candidate_commands: torch.Tensor,
    *,
    group_ids: torch.Tensor,
    step_cap: torch.Tensor,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
    permutation: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build hold, reverse and matched-row-shuffle evaluation controls.

    The function is gradient-disabled by construction.  None of these controls
    is a contrastive or negative training target.
    """

    state = _detached_float(current_state)
    commands, had_candidate_axis = _canonical_candidate_bank(
        candidate_commands, batch=len(state), device=state.device
    )
    cap, lower, upper = _validate_action_contract(
        step_cap, action_lower, action_upper
    )
    hold = state[:, None, None].expand_as(commands)
    reverse = state[:, None, None] - (commands - state[:, None, None])
    reverse = _project_candidate_bank(
        state,
        reverse,
        step_cap=cap.to(state),
        action_lower=lower.to(state),
        action_upper=upper.to(state),
    )
    order = _group_derangement(
        group_ids, batch=len(state), device=state.device, permutation=permutation
    )
    result = {
        "factual": commands,
        "hold": hold,
        "reverse": reverse,
        "shuffle": commands[order],
    }
    if not had_candidate_axis:
        result = {name: value[:, 0] for name, value in result.items()}
    return result


def _group_derangement(
    group_ids: torch.Tensor,
    *,
    batch: int,
    device: torch.device,
    permutation: torch.Tensor | None,
) -> torch.Tensor:
    groups = torch.as_tensor(group_ids, device=device).detach()
    if groups.shape != (batch,):
        raise ValueError("group_ids must be [B]")
    if groups.is_floating_point():
        _finite("group_ids", groups)
        if not torch.equal(groups, groups.round()):
            raise ValueError("group_ids must be integral")
    groups = groups.long()
    if permutation is None:
        order = torch.empty(batch, device=device, dtype=torch.long)
        for group in torch.unique(groups):
            rows = torch.nonzero(groups == group, as_tuple=False).flatten()
            if len(rows) < 2:
                raise ValueError("every shuffle group needs at least two rows")
            order[rows] = torch.roll(rows, 1)
    else:
        raw_order = torch.as_tensor(permutation, device=device).detach()
        if raw_order.is_floating_point():
            _finite("permutation", raw_order)
            if not torch.equal(raw_order, raw_order.round()):
                raise ValueError("permutation must be integral")
        order = raw_order.long()
    if order.shape != (batch,) or sorted(order.tolist()) != list(range(batch)):
        raise ValueError("permutation must contain every batch row exactly once")
    row = torch.arange(batch, device=device)
    if bool((order == row).any()):
        raise ValueError("permutation must be a row-wise derangement")
    if not torch.equal(groups[order], groups):
        raise ValueError("permutation must stay within group_ids")
    return order


@dataclass(frozen=True)
class StructuredGoalActorOutput:
    action: torch.Tensor
    baseline: torch.Tensor
    residual: torch.Tensor


class StructuredGoalFiveCommandActor(nn.Module):
    """Detached full256 actor with an explicit desired six-joint goal.

    This is a proposal module, not a policy authority.  It cannot consume a
    LeWM score and it cannot bypass the external planner or safety layer.
    ``park_mask`` makes the learned residual an exact identity transform over
    the supplied baseline for rows parked by that external state machine.
    """

    def __init__(
        self,
        *,
        step_cap: torch.Tensor,
        action_lower: torch.Tensor,
        action_upper: torch.Tensor,
        dimension: int = TOKEN_DIM,
    ) -> None:
        super().__init__()
        if int(dimension) <= 0 or int(dimension) % 6:
            raise ValueError("actor dimension must be positive and divisible by 6")
        cap, lower, upper = _validate_action_contract(
            step_cap, action_lower, action_upper
        )
        self.dimension = int(dimension)
        self.register_buffer("step_cap", cap.clone())
        self.register_buffer("action_lower", lower.clone())
        self.register_buffer("action_upper", upper.clone())
        self.scene_key_norm = nn.LayerNorm(TOKEN_DIM)
        self.scene_key_projection = nn.Linear(TOKEN_DIM, self.dimension)
        self.view_embedding = nn.Parameter(
            torch.empty(1, VIEWS, 1, self.dimension)
        )
        query_width = (
            STATE_DIM
            + 2 * STATE_DIM
            + STATE_DIM
            + STATE_DIM
            + STATE_DIM
            + STATE_DIM
        )
        self.query_projection = nn.Sequential(
            nn.LayerNorm(query_width),
            nn.Linear(query_width, self.dimension),
            nn.GELU(),
        )
        self.step_embedding = nn.Parameter(
            torch.empty(1, COMMAND_STEPS, self.dimension)
        )
        self.query_norm = nn.LayerNorm(self.dimension)
        self.cross_attention = nn.MultiheadAttention(
            self.dimension,
            6,
            dropout=0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.dimension)
        self.residual_head = nn.Linear(self.dimension, STATE_DIM)
        nn.init.normal_(self.view_embedding, std=0.02)
        nn.init.normal_(self.step_embedding, std=0.02)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        *,
        current_full256: torch.Tensor,
        view_valid: torch.Tensor,
        current_state: torch.Tensor,
        past_completed_commands: torch.Tensor,
        conditioning_past_completed_commands: torch.Tensor | None = None,
        desired_joint_goal: torch.Tensor,
        baseline_kind: BaselineKind,
        baseline_joint_goal: torch.Tensor | None = None,
        park_mask: torch.Tensor | None = None,
    ) -> StructuredGoalActorOutput:
        state, past = _validate_state_history(
            current_state, past_completed_commands
        )
        goal = _detached_float(desired_joint_goal, device=state.device)
        if goal.shape != (len(state), STATE_DIM):
            raise ValueError("desired_joint_goal must be [B,6]")
        _finite("desired_joint_goal", goal)
        if bool(((goal < self.action_lower) | (goal > self.action_upper)).any()):
            raise ValueError("desired_joint_goal lies outside the action contract")
        baseline_goal = (
            None
            if baseline_joint_goal is None
            else _detached_float(baseline_joint_goal, device=state.device)
        )
        if baseline_goal is not None:
            if baseline_goal.shape != (len(state), STATE_DIM):
                raise ValueError("baseline_joint_goal must be [B,6]")
            _finite("baseline_joint_goal", baseline_goal)
        baseline = structured_goal_actor_baseline(
            baseline_kind=baseline_kind,
            current_state=state,
            past_completed_commands=past,
            baseline_joint_goal=baseline_goal,
            step_cap=self.step_cap,
            action_lower=self.action_lower,
            action_upper=self.action_upper,
        )
        if conditioning_past_completed_commands is None:
            conditioning_past = past
            conditioning_baseline = baseline
        else:
            _conditioning_state, conditioning_past = _validate_state_history(
                current_state, conditioning_past_completed_commands
            )
            conditioning_baseline = structured_goal_actor_baseline(
                baseline_kind=baseline_kind,
                current_state=state,
                past_completed_commands=conditioning_past,
                baseline_joint_goal=baseline_goal,
                step_cap=self.step_cap,
                action_lower=self.action_lower,
                action_upper=self.action_upper,
            )
        if park_mask is None:
            parked = torch.zeros(len(state), device=state.device, dtype=torch.bool)
        else:
            parked = torch.as_tensor(park_mask, device=state.device).detach().bool()
            if parked.shape != (len(state),):
                raise ValueError("park_mask must be [B]")

        scene = _detached_float(current_full256, device=state.device)
        views = torch.as_tensor(view_valid, device=state.device).detach().bool()
        if scene.shape != (len(state), VIEWS, PATCHES, TOKEN_DIM):
            raise ValueError("current_full256 must be [B,2,256,192]")
        if views.shape != (len(state), VIEWS):
            raise ValueError("view_valid must be [B,2]")
        _finite("current_full256", scene)
        if not bool(views.any(dim=1).all()):
            raise ValueError("each V16.5 actor row needs at least one valid view")
        memory = self.scene_key_projection(self.scene_key_norm(scene))
        memory = memory + self.view_embedding
        memory = memory.reshape(len(state), VIEWS * PATCHES, self.dimension)
        memory_mask = (~views)[:, :, None].expand(-1, -1, PATCHES).reshape(
            len(state), VIEWS * PATCHES
        )
        conditioning_previous = torch.cat(
            (state[:, None], conditioning_baseline[:, :-1]), dim=1
        )
        query = self.query_projection(
            torch.cat(
                (
                    state[:, None].expand(-1, COMMAND_STEPS, -1),
                    conditioning_past.flatten(start_dim=1)[:, None].expand(
                        -1, COMMAND_STEPS, -1
                    ),
                    goal[:, None].expand(-1, COMMAND_STEPS, -1),
                    (goal - state)[:, None].expand(-1, COMMAND_STEPS, -1),
                    conditioning_baseline,
                    conditioning_baseline - conditioning_previous,
                ),
                dim=-1,
            )
        )
        query = self.query_norm(query + self.step_embedding)
        attended = self.cross_attention(
            query,
            memory,
            memory,
            key_padding_mask=memory_mask,
            need_weights=False,
        )[0]
        raw = torch.tanh(self.residual_head(self.output_norm(attended)))
        raw = raw * (~parked)[:, None, None].to(raw.dtype)
        desired = baseline + raw * self.step_cap.view(1, 1, -1)
        action = project_bounded_action_trajectory(
            state,
            desired,
            step_cap=self.step_cap,
            action_lower=self.action_lower,
            action_upper=self.action_upper,
        )
        # Contract-valid baselines are fixed points of the projector, so park
        # rows remain bitwise baseline identity even at nonzero actor weights.
        action = torch.where(parked[:, None, None], baseline, action)
        return StructuredGoalActorOutput(
            action=action,
            baseline=baseline,
            residual=action - baseline,
        )


def structured_goal_actor_loss(
    output: StructuredGoalActorOutput,
    completed_commands: torch.Tensor,
    *,
    step_cap: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 imitation loss measured in commandable joint-step units."""

    prediction = output.action
    target = torch.as_tensor(
        completed_commands,
        device=prediction.device,
        dtype=prediction.dtype,
    ).detach()
    cap = torch.as_tensor(step_cap, device=prediction.device, dtype=prediction.dtype)
    if prediction.ndim != 3 or prediction.shape[1:] != (
        COMMAND_STEPS,
        STATE_DIM,
    ):
        raise ValueError("actor prediction must be [B,5,6]")
    if target.shape != prediction.shape:
        raise ValueError("completed_commands must match actor [B,5,6]")
    if cap.shape != (STATE_DIM,) or not bool(torch.isfinite(cap).all()):
        raise ValueError("step_cap must be finite [6]")
    if bool((cap <= 0).any()):
        raise ValueError("step_cap must be positive")
    _finite("completed_commands", target)
    return F.smooth_l1_loss(prediction / cap, target / cap)


@torch.no_grad()
def structured_goal_actor_eval_controls(
    *,
    current_full256: torch.Tensor,
    view_valid: torch.Tensor,
    current_state: torch.Tensor,
    past_completed_commands: torch.Tensor,
    desired_joint_goal: torch.Tensor,
    baseline_kind: BaselineKind,
    group_ids: torch.Tensor,
    permutation: torch.Tensor | None = None,
) -> dict[str, dict[str, Any]]:
    """Return controlled eval inputs with one fixed factual baseline.

    Goal controls alter only the learned query condition.  For the analytic
    baseline, every control retains the factual ``baseline_joint_goal`` so a
    zero-head actor cannot manufacture apparent goal dependence.
    """

    state, past = _validate_state_history(
        current_state, past_completed_commands
    )
    scene = _detached_float(current_full256, device=state.device)
    views = torch.as_tensor(view_valid, device=state.device).detach().bool()
    goal = _detached_float(desired_joint_goal, device=state.device)
    if scene.shape != (len(state), VIEWS, PATCHES, TOKEN_DIM):
        raise ValueError("current_full256 must be [B,2,256,192]")
    if views.shape != (len(state), VIEWS) or goal.shape != state.shape:
        raise ValueError("actor eval control inputs have invalid shapes")
    if baseline_kind not in ("analytic", "repeat", "damped"):
        raise ValueError("baseline_kind must be analytic, repeat, or damped")
    order = _group_derangement(
        group_ids, batch=len(state), device=state.device, permutation=permutation
    )
    factual_baseline_goal = goal if baseline_kind == "analytic" else None

    def make(
        selected_scene: torch.Tensor,
        selected_views: torch.Tensor,
        selected_goal: torch.Tensor,
    ) -> dict[str, Any]:
        return {
            "current_full256": selected_scene,
            "view_valid": selected_views,
            "current_state": state,
            "past_completed_commands": past,
            "desired_joint_goal": selected_goal,
            "baseline_kind": baseline_kind,
            "baseline_joint_goal": factual_baseline_goal,
        }

    return {
        "factual": make(scene, views, goal),
        "blank_visual": make(torch.zeros_like(scene), views, goal),
        "swap_visual": make(scene[order], views[order], goal),
        # The null goal is the measured current joint state: "do not move".
        # Keeping the factual baseline fixed isolates learned goal use without
        # inventing an often out-of-distribution all-zero robot pose.
        "blank_goal": make(scene, views, state),
        "swap_goal": make(scene, views, goal[order]),
    }


@dataclass(frozen=True)
class ParameterInventoryItem:
    branch: Literal["lewm", "actor"]
    name: str
    numel: int
    requires_grad: bool


def v165_parameter_inventory(
    lewm: StateFKLeWM,
    actor: StructuredGoalFiveCommandActor,
) -> tuple[ParameterInventoryItem, ...]:
    items = []
    for branch, module in (("lewm", lewm), ("actor", actor)):
        for name, parameter in module.named_parameters():
            items.append(
                ParameterInventoryItem(
                    branch=branch,
                    name=name,
                    numel=parameter.numel(),
                    requires_grad=parameter.requires_grad,
                )
            )
    return tuple(items)


def configure_v165_training_stage(
    lewm: StateFKLeWM,
    actor: StructuredGoalFiveCommandActor,
    stage: TrainingStage,
) -> tuple[ParameterInventoryItem, ...]:
    if stage not in ("lewm", "actor", "frozen"):
        raise ValueError("stage must be lewm, actor, or frozen; joint training is forbidden")
    for branch, module in (("lewm", lewm), ("actor", actor)):
        enabled = branch == stage
        module.train(enabled)
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)
            parameter.grad = None
    return v165_parameter_inventory(lewm, actor)


def v165_stage_parameters(
    lewm: StateFKLeWM,
    actor: StructuredGoalFiveCommandActor,
    stage: Literal["lewm", "actor"],
) -> tuple[nn.Parameter, ...]:
    if stage not in ("lewm", "actor"):
        raise ValueError("an optimizer must select exactly one V16.5 branch")
    selected = lewm if stage == "lewm" else actor
    other = actor if stage == "lewm" else lewm
    if any(parameter.requires_grad for parameter in other.parameters()):
        raise RuntimeError("the non-selected V16.5 branch is not frozen")
    parameters = tuple(
        parameter for parameter in selected.parameters() if parameter.requires_grad
    )
    if not parameters:
        raise RuntimeError("selected V16.5 branch has no trainable parameters")
    return parameters


@dataclass(frozen=True)
class V165Core:
    """Deployment bundle with no combined ``nn.Module.parameters`` surface."""

    lewm: StateFKLeWM
    actor: StructuredGoalFiveCommandActor

    def predict_physical_candidates(self, **kwargs: Any) -> StateFKLeWMOutput:
        return self.lewm(**kwargs)

    def propose_actions(self, **kwargs: Any) -> StructuredGoalActorOutput:
        return self.actor(**kwargs)

    def parameters(self) -> None:
        raise RuntimeError(
            "V16.5 has no holistic parameter iterator; use v165_stage_parameters"
        )

    def separated_state_dicts(self) -> dict[str, Mapping[str, torch.Tensor]]:
        return {"lewm": self.lewm.state_dict(), "actor": self.actor.state_dict()}


__all__ = [
    "ACTOR_CHECKPOINT_FORMAT",
    "ANCHORS",
    "COMMAND_STEPS",
    "LEWM_CHECKPOINT_FORMAT",
    "PATCHES",
    "PATH_STEPS",
    "STATE_DIM",
    "TOKEN_DIM",
    "VIEWS",
    "ParameterInventoryItem",
    "StateFKLeWM",
    "StateFKLeWMFKMetrics",
    "StateFKLeWMLoss",
    "StateFKLeWMOutput",
    "StructuredGoalActorOutput",
    "StructuredGoalFiveCommandActor",
    "V165Core",
    "commanded_state_baseline",
    "configure_v165_training_stage",
    "damped_five_command_baseline",
    "goal_interpolation_five_command_baseline",
    "repeat_five_command_baseline",
    "state_fk_lewm_eval_controls",
    "state_fk_lewm_fk_metrics",
    "state_fk_lewm_loss",
    "structured_goal_actor_baseline",
    "structured_goal_actor_loss",
    "structured_goal_actor_eval_controls",
    "v165_parameter_inventory",
    "v165_stage_parameters",
]
