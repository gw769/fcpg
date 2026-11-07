"""Prod-V16.8 full-history five-command Actor core.

This module deliberately leaves the maintained V16.5/V16.7 Actor untouched.
It reuses that Actor's scene, query, attention, output-head, and action-contract
state names, while adding two zero-initialized branches:

* a causal completed-command history encoder; and
* an optional input-dependent continuous residual gate.

The ``legacy`` fusion is the old attention-only path.  With a strict-migrated
V16.7 state, a zero-initialized history encoder, and the residual gate disabled,
it is bitwise equivalent to :class:`StructuredGoalFiveCommandActor`.
``query_residual`` uses the standard ``query + attended`` residual fusion before
the existing output norm and residual head.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from fcpg.prod_v16_2 import project_bounded_action_trajectory
from fcpg.prod_v16_5 import (
    COMMAND_STEPS,
    PATCHES,
    STATE_DIM,
    TOKEN_DIM,
    VIEWS,
    BaselineKind,
    StructuredGoalActorOutput,
    StructuredGoalFiveCommandActor,
    _detached_float,
    _finite,
    _validate_state_history,
    structured_goal_actor_baseline,
)


HISTORY_CHUNKS = 3
HISTORY_FEATURES = HISTORY_CHUNKS * COMMAND_STEPS * STATE_DIM + HISTORY_CHUNKS
VELOCITY_WEIGHT = 0.2
REGRET_WEIGHT = 1.0
FUSION_MODES = ("legacy", "query_residual")
FusionMode = Literal["legacy", "query_residual"]
HISTORY_MODES = ("disabled", "full")
HistoryMode = Literal["disabled", "full"]


class V168ContractError(ValueError):
    """Fail-closed V16.8 Actor, migration, or objective contract error."""


@dataclass(frozen=True)
class V168ActorObjective:
    """Action, velocity, and repeat-regret terms for one Actor batch."""

    total: torch.Tensor
    action: torch.Tensor
    velocity: torch.Tensor
    regret: torch.Tensor


class FullHistoryFiveCommandActor(StructuredGoalFiveCommandActor):
    """Five-command residual Actor with complete causal command history.

    Existing V16.7 state names are inherited without renaming.  The additional
    history input is three chronological completed chunks ``[B,3,5,6]``, a
    prefix-valid mask ``[B,3]``, and a transition index ``[B]``.  The mask must
    equal ``slot < transition`` exactly, so H1/H2/H3 expose one/two/three valid
    chunks respectively.  Every invalid (future) slot must be exactly zero.

    ``history_encoder`` maps the masked history to one delta per command-step
    query.  Its linear weight and bias are zero-initialized.  An all-masked row
    is additionally multiplied by an exact zero after the encoder, so disabling
    history remains an identity even if that encoder is later trained.
    ``history_mode=disabled`` still validates the factual transition-aligned
    history, but bypasses the encoder and returns an exact non-gradient zero;
    ``history_mode=full`` encodes it.

    In ``query_residual`` mode, both ``query`` and ``attended`` have shape
    ``[B,5,D]``.  Their elementwise sum is passed through the inherited
    ``output_norm`` and ``residual_head``.  ``legacy`` passes only ``attended``
    through those same inherited modules.  When enabled, the continuous gate
    reads this fused visual-query representation per step and joint.
    """

    def __init__(
        self,
        *,
        step_cap: torch.Tensor,
        action_lower: torch.Tensor,
        action_upper: torch.Tensor,
        dimension: int = TOKEN_DIM,
        fusion_mode: FusionMode = "query_residual",
        history_mode: HistoryMode = "full",
        residual_gate: bool = False,
    ) -> None:
        super().__init__(
            step_cap=step_cap,
            action_lower=action_lower,
            action_upper=action_upper,
            dimension=dimension,
        )
        if fusion_mode not in FUSION_MODES:
            raise V168ContractError(
                "V16.8 fusion_mode must be legacy or query_residual"
            )
        if history_mode not in HISTORY_MODES:
            raise V168ContractError("V16.8 history_mode must be disabled or full")
        if not isinstance(residual_gate, bool):
            raise V168ContractError("V16.8 residual_gate must be boolean")
        self.fusion_mode: FusionMode = fusion_mode
        self.history_mode: HistoryMode = history_mode
        self.residual_gate = residual_gate
        self.history_encoder = nn.Linear(
            HISTORY_FEATURES,
            COMMAND_STEPS * self.dimension,
        )
        self.residual_gate_head = nn.Linear(self.dimension, STATE_DIM)
        self.reset_v168_branches()

    def reset_v168_branches(self) -> None:
        """Restore migration-safe history and gate initialization."""

        nn.init.zeros_(self.history_encoder.weight)
        nn.init.zeros_(self.history_encoder.bias)
        nn.init.zeros_(self.residual_gate_head.weight)
        nn.init.zeros_(self.residual_gate_head.bias)

    def _history_delta(
        self,
        completed_command_history: torch.Tensor,
        history_valid: torch.Tensor,
        transition: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        raw_history = torch.as_tensor(completed_command_history, device=device)
        raw_valid = torch.as_tensor(history_valid, device=device)
        raw_transition = torch.as_tensor(transition, device=device)
        if raw_history.ndim != 4 or raw_history.shape[1:] != (
            HISTORY_CHUNKS,
            COMMAND_STEPS,
            STATE_DIM,
        ):
            raise V168ContractError(
                "completed_command_history must be [B,3,5,6]"
            )
        if raw_valid.shape != (len(raw_history), HISTORY_CHUNKS):
            raise V168ContractError("history_valid must be [B,3]")
        if raw_valid.dtype != torch.bool:
            raise V168ContractError("history_valid must be boolean")
        if raw_transition.shape != (len(raw_history),):
            raise V168ContractError("transition must be [B]")
        if raw_transition.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise V168ContractError("transition must have integer dtype")
        transition_index = raw_transition.detach().long()
        if bool(((transition_index < 0) | (transition_index > HISTORY_CHUNKS)).any()):
            raise V168ContractError("transition must contain only H0-H3 indices")
        history = raw_history.detach().float()
        valid = raw_valid.detach()
        _finite("completed_command_history", history)
        expected_valid = torch.arange(
            HISTORY_CHUNKS, device=device
        ).view(1, -1) < transition_index.view(-1, 1)
        if not torch.equal(valid, expected_valid):
            raise V168ContractError(
                "history_valid must equal slot < transition for every row"
            )
        invalid = ~valid[:, :, None, None]
        if bool((history.masked_select(invalid) != 0).any()):
            raise V168ContractError(
                "invalid or future completed-command history must be exactly zero"
            )
        if self.history_mode == "disabled":
            return history.new_zeros(
                (len(history), COMMAND_STEPS, self.dimension)
            )
        masked = history * valid[:, :, None, None].to(history.dtype)
        features = torch.cat(
            (masked.flatten(start_dim=1), valid.to(history.dtype)),
            dim=1,
        )
        if features.shape != (len(history), HISTORY_FEATURES):
            raise V168ContractError("V16.8 history feature shape drifted")
        delta = self.history_encoder(features).reshape(
            len(history), COMMAND_STEPS, self.dimension
        )
        active = valid.any(dim=1).to(delta.dtype).view(-1, 1, 1)
        return delta * active

    def _fused_residual_gate(self, fused: torch.Tensor) -> torch.Tensor:
        if fused.ndim != 3 or fused.shape[1:] != (
            COMMAND_STEPS,
            self.dimension,
        ):
            raise V168ContractError("V16.8 residual-gate fusion shape drifted")
        gate = torch.sigmoid(self.residual_gate_head(fused))
        if gate.shape != (len(fused), COMMAND_STEPS, STATE_DIM):
            raise V168ContractError("V16.8 residual gate must be [B,5,6]")
        return gate

    def forward(
        self,
        *,
        current_full256: torch.Tensor,
        view_valid: torch.Tensor,
        current_state: torch.Tensor,
        past_completed_commands: torch.Tensor,
        completed_command_history: torch.Tensor,
        history_valid: torch.Tensor,
        transition: torch.Tensor,
        desired_joint_goal: torch.Tensor,
        baseline_kind: BaselineKind,
        baseline_joint_goal: torch.Tensor | None = None,
        park_mask: torch.Tensor | None = None,
    ) -> StructuredGoalActorOutput:
        if self.fusion_mode not in FUSION_MODES:
            raise V168ContractError(
                "V16.8 fusion_mode must be legacy or query_residual"
            )
        if self.history_mode not in HISTORY_MODES:
            raise V168ContractError(
                "V16.8 history_mode must be disabled or full"
            )
        if not isinstance(self.residual_gate, bool):
            raise V168ContractError("V16.8 residual_gate must remain boolean")
        if baseline_kind != "repeat":
            raise V168ContractError("V16.8 baseline_kind must be repeat")
        if baseline_joint_goal is not None:
            raise V168ContractError(
                "V16.8 repeat baseline forbids baseline_joint_goal"
            )
        state, past = _validate_state_history(
            current_state, past_completed_commands
        )
        goal = _detached_float(desired_joint_goal, device=state.device)
        if goal.shape != (len(state), STATE_DIM):
            raise V168ContractError("desired_joint_goal must be [B,6]")
        _finite("desired_joint_goal", goal)
        if bool(((goal < self.action_lower) | (goal > self.action_upper)).any()):
            raise V168ContractError(
                "desired_joint_goal lies outside the action contract"
            )
        baseline = structured_goal_actor_baseline(
            baseline_kind="repeat",
            current_state=state,
            past_completed_commands=past,
            baseline_joint_goal=None,
            step_cap=self.step_cap,
            action_lower=self.action_lower,
            action_upper=self.action_upper,
        )
        if park_mask is None:
            parked = torch.zeros(len(state), device=state.device, dtype=torch.bool)
        else:
            parked = torch.as_tensor(park_mask, device=state.device).detach().bool()
            if parked.shape != (len(state),):
                raise V168ContractError("park_mask must be [B]")

        history_delta = self._history_delta(
            completed_command_history,
            history_valid,
            transition,
            device=state.device,
        )
        if len(history_delta) != len(state):
            raise V168ContractError("history batch must match current_state")
        scene = _detached_float(current_full256, device=state.device)
        views = torch.as_tensor(view_valid, device=state.device).detach().bool()
        if scene.shape != (len(state), VIEWS, PATCHES, TOKEN_DIM):
            raise V168ContractError("current_full256 must be [B,2,256,192]")
        if views.shape != (len(state), VIEWS):
            raise V168ContractError("view_valid must be [B,2]")
        _finite("current_full256", scene)
        if not bool(views.any(dim=1).all()):
            raise V168ContractError("each V16.8 Actor row needs one valid view")
        memory = self.scene_key_projection(self.scene_key_norm(scene))
        memory = memory + self.view_embedding
        memory = memory.reshape(len(state), VIEWS * PATCHES, self.dimension)
        memory_mask = (~views)[:, :, None].expand(-1, -1, PATCHES).reshape(
            len(state), VIEWS * PATCHES
        )
        conditioning_previous = torch.cat(
            (state[:, None], baseline[:, :-1]), dim=1
        )
        query = self.query_projection(
            torch.cat(
                (
                    state[:, None].expand(-1, COMMAND_STEPS, -1),
                    past.flatten(start_dim=1)[:, None].expand(
                        -1, COMMAND_STEPS, -1
                    ),
                    goal[:, None].expand(-1, COMMAND_STEPS, -1),
                    (goal - state)[:, None].expand(-1, COMMAND_STEPS, -1),
                    baseline,
                    baseline - conditioning_previous,
                ),
                dim=-1,
            )
        )
        query = self.query_norm(query + history_delta + self.step_embedding)
        attended = self.cross_attention(
            query,
            memory,
            memory,
            key_padding_mask=memory_mask,
            need_weights=False,
        )[0]
        fused = attended if self.fusion_mode == "legacy" else query + attended
        raw = torch.tanh(self.residual_head(self.output_norm(fused)))
        if self.residual_gate:
            raw = raw * self._fused_residual_gate(fused)
        raw = raw * (~parked)[:, None, None].to(raw.dtype)
        desired = baseline + raw * self.step_cap.view(1, 1, -1)
        action = project_bounded_action_trajectory(
            state,
            desired,
            step_cap=self.step_cap,
            action_lower=self.action_lower,
            action_upper=self.action_upper,
        )
        action = torch.where(parked[:, None, None], baseline, action)
        return StructuredGoalActorOutput(
            action=action,
            baseline=baseline,
            residual=action - baseline,
        )


_V168_ONLY_STATE_NAMES = frozenset(
    {
        "history_encoder.weight",
        "history_encoder.bias",
        "residual_gate_head.weight",
        "residual_gate_head.bias",
    }
)


def load_v167_actor_state_dict_strict(
    actor: FullHistoryFiveCommandActor,
    state_dict: Mapping[str, Any],
) -> None:
    """Strictly migrate one old Actor state and zero every V16.8-only tensor."""

    if not isinstance(actor, FullHistoryFiveCommandActor):
        raise V168ContractError("V16.7 migration target must be a V16.8 Actor")
    if not isinstance(state_dict, Mapping):
        raise V168ContractError("V16.7 Actor state_dict must be a mapping")
    target = actor.state_dict()
    expected_old = set(target) - set(_V168_ONLY_STATE_NAMES)
    received = set(state_dict)
    if received != expected_old:
        missing = sorted(expected_old - received)
        unexpected = sorted(received - expected_old)
        raise V168ContractError(
            "V16.7 Actor state_dict key mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    migrated: dict[str, torch.Tensor] = {}
    for name, expected in target.items():
        if name in _V168_ONLY_STATE_NAMES:
            migrated[name] = torch.zeros_like(expected)
            continue
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            raise V168ContractError(f"V16.7 Actor state[{name}] is not a tensor")
        if value.shape != expected.shape:
            raise V168ContractError(f"V16.7 Actor state[{name}] shape drifted")
        if value.dtype != expected.dtype:
            raise V168ContractError(f"V16.7 Actor state[{name}] dtype drifted")
        migrated[name] = value.detach().clone()
    try:
        result = actor.load_state_dict(migrated, strict=True)
    except RuntimeError as exc:
        raise V168ContractError("V16.7 Actor state failed strict migration") from exc
    if result.missing_keys or result.unexpected_keys:
        raise V168ContractError("V16.7 Actor state failed strict migration")
    for name in _V168_ONLY_STATE_NAMES:
        if bool(torch.count_nonzero(actor.state_dict()[name])):
            raise V168ContractError("V16.8-only migration state is not zero")


def _v168_actor_objective(
    output: StructuredGoalActorOutput,
    completed_commands: torch.Tensor,
    current_state: torch.Tensor,
    *,
    step_cap: torch.Tensor,
    loss_kind: Literal["l1", "smooth_l1"],
) -> V168ActorObjective:
    prediction = output.action
    if prediction.ndim != 3 or prediction.shape[1:] != (
        COMMAND_STEPS,
        STATE_DIM,
    ):
        raise V168ContractError("V16.8 Actor prediction must be [B,5,6]")
    target = torch.as_tensor(
        completed_commands,
        device=prediction.device,
        dtype=prediction.dtype,
    ).detach()
    state = torch.as_tensor(
        current_state,
        device=prediction.device,
        dtype=prediction.dtype,
    ).detach()
    cap = torch.as_tensor(
        step_cap,
        device=prediction.device,
        dtype=prediction.dtype,
    ).detach()
    baseline = torch.as_tensor(
        output.baseline,
        device=prediction.device,
        dtype=prediction.dtype,
    ).detach()
    if target.shape != prediction.shape or baseline.shape != prediction.shape:
        raise V168ContractError(
            "completed_commands and repeat baseline must match Actor [B,5,6]"
        )
    if state.shape != (len(prediction), STATE_DIM):
        raise V168ContractError("current_state must match Actor [B,6]")
    if cap.shape != (STATE_DIM,):
        raise V168ContractError("step_cap must be [6]")
    for name, value in (
        ("V16.8 Actor prediction", prediction),
        ("completed_commands", target),
        ("current_state", state),
        ("step_cap", cap),
        ("repeat baseline", baseline),
    ):
        _finite(name, value)
    if bool((cap <= 0).any()):
        raise V168ContractError("step_cap must be positive")
    if loss_kind not in {"l1", "smooth_l1"}:
        raise V168ContractError("V16.8 objective loss kind drifted")
    scale = cap.view(1, 1, STATE_DIM)
    previous_prediction = torch.cat(
        (state[:, None], prediction[:, :-1]), dim=1
    )
    previous_target = torch.cat((state[:, None], target[:, :-1]), dim=1)
    if loss_kind == "l1":
        action = F.l1_loss(prediction / scale, target / scale)
        velocity = F.l1_loss(
            (prediction - previous_prediction) / scale,
            (target - previous_target) / scale,
        )
    else:
        action = F.smooth_l1_loss(prediction / scale, target / scale)
        velocity = F.smooth_l1_loss(
            (prediction - previous_prediction) / scale,
            (target - previous_target) / scale,
        )
    prediction_row_l1 = ((prediction - target).abs() / scale).mean(dim=(1, 2))
    baseline_row_l1 = ((baseline - target).abs() / scale).mean(dim=(1, 2))
    regret = torch.relu(prediction_row_l1 - baseline_row_l1).mean()
    total = action + VELOCITY_WEIGHT * velocity
    if loss_kind == "l1":
        total = total + REGRET_WEIGHT * regret
    return V168ActorObjective(
        total=total,
        action=action,
        velocity=velocity,
        regret=regret,
    )


def v168_actor_l1_objective(
    output: StructuredGoalActorOutput,
    completed_commands: torch.Tensor,
    current_state: torch.Tensor,
    *,
    step_cap: torch.Tensor,
) -> V168ActorObjective:
    """Normalized L1 action, 0.2 L1 velocity, and fixed repeat regret."""

    return _v168_actor_objective(
        output,
        completed_commands,
        current_state,
        step_cap=step_cap,
        loss_kind="l1",
    )


def v168_actor_smooth_l1_control_objective(
    output: StructuredGoalActorOutput,
    completed_commands: torch.Tensor,
    current_state: torch.Tensor,
    *,
    step_cap: torch.Tensor,
) -> V168ActorObjective:
    """V16.7-compatible SmoothL1 action/velocity control objective."""

    return _v168_actor_objective(
        output,
        completed_commands,
        current_state,
        step_cap=step_cap,
        loss_kind="smooth_l1",
    )


__all__ = [
    "FUSION_MODES",
    "FusionMode",
    "HISTORY_CHUNKS",
    "HISTORY_FEATURES",
    "HISTORY_MODES",
    "HistoryMode",
    "REGRET_WEIGHT",
    "VELOCITY_WEIGHT",
    "FullHistoryFiveCommandActor",
    "V168ActorObjective",
    "V168ContractError",
    "load_v167_actor_state_dict_strict",
    "v168_actor_l1_objective",
    "v168_actor_smooth_l1_control_objective",
]
