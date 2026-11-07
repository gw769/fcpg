"""Prod-V16.6 causal curriculum for the maintained five-command Actor.

V16.6 does not introduce a new policy architecture.  It trains the existing
``StructuredGoalFiveCommandActor`` under one of two fixed causal curricula and
keeps the executable horizon at five commands.  This module owns only pure
curriculum metadata, the fixed action-plus-velocity objective, and a strictly
offline loader for a separately adjudicated winner checkpoint.  It has no
dataset, trainer, TEST, planner, camera, robot, or hardware I/O surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from fcpg.prod_v16_2 import prod_v16_2_actor_action_contract
from fcpg.prod_v16_5 import (
    COMMAND_STEPS,
    STATE_DIM,
    TOKEN_DIM,
    StructuredGoalActorOutput,
    StructuredGoalFiveCommandActor,
)


PROTOCOL_ID = "so101-prod-v16-6-causal-curriculum-v1"
ARM_RECEIPT_FORMAT = "so101-prod-v16-6-actor-curriculum-arm-receipt-v1"
ARM_CHECKPOINT_FORMAT = "so101-prod-v16-6-actor-curriculum-arm-fixed-final-v1"
WINNER_RECEIPT_FORMAT = "so101-prod-v16-6-actor-curriculum-winner-v1"
WINNER_CHECKPOINT_FORMAT = (
    "so101-prod-v16-6-actor-curriculum-winner-checkpoint-v1"
)
V166_ACTOR_MODEL_FORMAT = "so101-prod-v16-6-structured-goal-actor-v1"

H0_THEN_H0_RECIPE = "h0_then_h0"
CAUSAL4_THEN_H0_RECIPE = "causal4_then_h0"
V166_RECIPES = (H0_THEN_H0_RECIPE, CAUSAL4_THEN_H0_RECIPE)

PRETRAIN_PHASE = "pretrain"
FINETUNE_PHASE = "finetune"
V166_PHASES = (PRETRAIN_PHASE, FINETUNE_PHASE)

PRETRAIN_STEPS = 313
FINETUNE_STEPS = 313
TOTAL_STEPS = PRETRAIN_STEPS + FINETUNE_STEPS
BATCH_SIZE = 256
VELOCITY_WEIGHT = 0.2

Recipe = Literal["h0_then_h0", "causal4_then_h0"]
CurriculumPhase = Literal["pretrain", "finetune"]


class V166ContractError(ValueError):
    """A V16.6 curriculum, objective, or offline-checkpoint contract drifted."""


def v166_phase_transitions(recipe: str, phase: str) -> tuple[int, ...]:
    """Return the only transitions admitted by one fixed curriculum phase.

    ``h0_then_h0`` is the control arm: both phases train only H0.  The causal
    curriculum first trains on factual H0--H3 transition examples, then uses the
    same H0-only final phase as the control.  Population selection and sampling
    intentionally live outside this model-core module.
    """

    name = str(recipe)
    stage = str(phase)
    if name not in V166_RECIPES:
        raise V166ContractError(
            "V16.6 recipe must be h0_then_h0 or causal4_then_h0"
        )
    if stage not in V166_PHASES:
        raise V166ContractError("V16.6 phase must be pretrain or finetune")
    if stage == FINETUNE_PHASE or name == H0_THEN_H0_RECIPE:
        return (0,)
    return (0, 1, 2, 3)


@dataclass(frozen=True)
class V166ActorObjective:
    """Step-scaled five-command action and causal velocity losses."""

    total: torch.Tensor
    action: torch.Tensor
    velocity: torch.Tensor


def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise V166ContractError(f"{name} must be finite")


def v166_actor_objective(
    output: StructuredGoalActorOutput,
    completed_commands: torch.Tensor,
    current_state: torch.Tensor,
    *,
    step_cap: torch.Tensor,
) -> V166ActorObjective:
    """Compute the fixed V16.6 action plus velocity imitation objective.

    Both predicted and factual velocity sequences start at the same measured
    ``q0`` and thereafter use their own preceding command.  Targets, q0 and the
    action contract are detached optimizer inputs; gradients can reach only the
    Actor prediction.
    """

    prediction = output.action
    if prediction.ndim != 3 or prediction.shape[1:] != (
        COMMAND_STEPS,
        STATE_DIM,
    ):
        raise V166ContractError("V16.6 Actor prediction must be [B,5,6]")
    if not prediction.is_floating_point():
        raise V166ContractError("V16.6 Actor prediction must be floating point")
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
    if target.shape != prediction.shape:
        raise V166ContractError("completed_commands must match Actor [B,5,6]")
    if state.shape != (len(prediction), STATE_DIM):
        raise V166ContractError("current_state must match Actor [B,6]")
    if cap.shape != (STATE_DIM,):
        raise V166ContractError("step_cap must be [6]")
    _finite("V16.6 Actor prediction", prediction)
    _finite("completed_commands", target)
    _finite("current_state", state)
    _finite("step_cap", cap)
    if bool((cap <= 0).any()):
        raise V166ContractError("step_cap must be positive")

    scale = cap.view(1, 1, STATE_DIM)
    action_loss = F.smooth_l1_loss(prediction / scale, target / scale)
    previous_prediction = torch.cat(
        (state[:, None], prediction[:, :-1]), dim=1
    )
    previous_target = torch.cat((state[:, None], target[:, :-1]), dim=1)
    velocity_loss = F.smooth_l1_loss(
        (prediction - previous_prediction) / scale,
        (target - previous_target) / scale,
    )
    return V166ActorObjective(
        total=action_loss + VELOCITY_WEIGHT * velocity_loss,
        action=action_loss,
        velocity=velocity_loss,
    )


def v166_state_dict_sha256(state: Mapping[str, Any]) -> str:
    """Hash a tensor state mapping using the repository checkpoint convention."""

    if not isinstance(state, Mapping) or not state:
        raise V166ContractError("winner state_dict must be a non-empty mapping")
    digest = hashlib.sha256()
    for raw_name in sorted(state):
        if not isinstance(raw_name, str):
            raise V166ContractError("winner state_dict names must be strings")
        value = state[raw_name]
        if not isinstance(value, torch.Tensor):
            raise V166ContractError("winner state_dict values must be tensors")
        tensor = value.detach().cpu().contiguous()
        _finite(f"winner state_dict[{raw_name}]", tensor)
        digest.update(raw_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(
                list(tensor.shape),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _sha256_text(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise V166ContractError(f"{label} must be one lowercase SHA-256")
    return text


def _winner_action_contract(
    payload: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    raw = payload.get("action_contract")
    if not isinstance(raw, Mapping) or set(raw) != {"step_cap", "lower", "upper"}:
        raise V166ContractError(
            "winner action_contract must contain only step_cap, lower, and upper"
        )
    values = {
        "step_cap": torch.as_tensor(raw["step_cap"]).detach().float().cpu(),
        "action_lower": torch.as_tensor(raw["lower"]).detach().float().cpu(),
        "action_upper": torch.as_tensor(raw["upper"]).detach().float().cpu(),
    }
    for name, value in values.items():
        if value.shape != (STATE_DIM,):
            raise V166ContractError(f"winner {name} must be [6]")
        _finite(f"winner {name}", value)
    if bool((values["step_cap"] <= 0).any()) or bool(
        (values["action_lower"] >= values["action_upper"]).any()
    ):
        raise V166ContractError("winner action contract is invalid")
    return values


@dataclass(frozen=True)
class V166OfflineProposal:
    """One non-deployment five-command proposal from the adjudicated winner."""

    commands: torch.Tensor
    recipe: str
    checkpoint_state_sha256: str
    deployment_authorized: bool = False

    def __post_init__(self) -> None:
        commands = self.commands
        if not isinstance(commands, torch.Tensor) or commands.ndim != 3 or commands.shape[1:] != (
            COMMAND_STEPS,
            STATE_DIM,
        ):
            raise V166ContractError("offline proposal commands must be [B,5,6]")
        _finite("offline proposal commands", commands)
        if self.recipe not in V166_RECIPES:
            raise V166ContractError("offline proposal recipe is not registered")
        _sha256_text(
            self.checkpoint_state_sha256,
            label="offline proposal checkpoint_state_sha256",
        )
        if self.deployment_authorized is not False:
            raise V166ContractError("V16.6 offline proposals are never deployment-authorized")


_OFFLINE_LOADER_TOKEN = object()


class V166OfflineActor(nn.Module):
    """Inference-only wrapper around one strictly loaded V16.6 winner Actor.

    Construction is intentionally available only through
    :meth:`from_checkpoint_payload`.  The wrapper accepts no path, receipt,
    planner, or device-I/O object and exposes only a five-command proposal.
    """

    def __init__(
        self,
        actor: StructuredGoalFiveCommandActor,
        *,
        recipe: str,
        checkpoint_state_sha256: str,
        _loader_token: object | None = None,
    ) -> None:
        super().__init__()
        if _loader_token is not _OFFLINE_LOADER_TOKEN:
            raise V166ContractError(
                "V166OfflineActor must be created from a winner checkpoint payload"
            )
        if recipe not in V166_RECIPES:
            raise V166ContractError("winner recipe is not registered")
        self._actor = actor
        self._recipe = str(recipe)
        self._checkpoint_state_sha256 = _sha256_text(
            checkpoint_state_sha256,
            label="winner state_sha256",
        )
        self._actor.requires_grad_(False)
        self._actor.eval()
        super().train(False)

    @classmethod
    def from_checkpoint_payload(
        cls, payload: Mapping[str, Any]
    ) -> "V166OfflineActor":
        """Strict-load one fixed-final, adjudicator-selected winner on CPU."""

        if not isinstance(payload, Mapping):
            raise V166ContractError("winner checkpoint payload must be a mapping")
        required = {
            "format",
            "protocol_id",
            "model_format",
            "winner_selected",
            "recipe",
            "pretrain_steps",
            "finetune_steps",
            "fixed_final_step",
            "model_kwargs",
            "action_contract",
            "state_dict",
            "state_sha256",
            "deployment_authorized",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise V166ContractError(f"winner checkpoint fields are missing: {missing}")
        if payload.get("format") != WINNER_CHECKPOINT_FORMAT:
            raise V166ContractError("winner checkpoint format drifted")
        if payload.get("protocol_id") != PROTOCOL_ID:
            raise V166ContractError("winner checkpoint protocol drifted")
        if payload.get("model_format") != V166_ACTOR_MODEL_FORMAT:
            raise V166ContractError("winner Actor model format drifted")
        if payload.get("winner_selected") is not True:
            raise V166ContractError("winner checkpoint was not adjudicator-selected")
        recipe = str(payload.get("recipe"))
        if recipe not in V166_RECIPES:
            raise V166ContractError("winner checkpoint recipe is not registered")
        if (
            payload.get("pretrain_steps") != PRETRAIN_STEPS
            or payload.get("finetune_steps") != FINETUNE_STEPS
            or payload.get("fixed_final_step") != TOTAL_STEPS
        ):
            raise V166ContractError("winner checkpoint curriculum endpoint drifted")
        if payload.get("deployment_authorized") is not False:
            raise V166ContractError(
                "winner checkpoint must remain deployment_authorized=false"
            )
        model_kwargs = payload.get("model_kwargs")
        if not isinstance(model_kwargs, Mapping) or set(model_kwargs) != {"dimension"}:
            raise V166ContractError("winner model_kwargs must contain only dimension")
        try:
            dimension = int(model_kwargs["dimension"])
        except (TypeError, ValueError) as exc:
            raise V166ContractError("winner Actor dimension is invalid") from exc
        if dimension != TOKEN_DIM:
            raise V166ContractError(
                f"winner Actor dimension must remain the registered {TOKEN_DIM}"
            )
        contract = _winner_action_contract(payload)
        registered_contract = prod_v16_2_actor_action_contract()
        for name, expected in registered_contract.items():
            if not torch.equal(contract[name], expected.detach().float().cpu()):
                raise V166ContractError(
                    f"winner {name} differs from the registered action contract"
                )
        state = payload.get("state_dict")
        if not isinstance(state, Mapping):
            raise V166ContractError("winner checkpoint has no state_dict mapping")
        expected_sha = _sha256_text(
            payload.get("state_sha256"), label="winner state_sha256"
        )
        actual_sha = v166_state_dict_sha256(state)
        if actual_sha != expected_sha:
            raise V166ContractError("winner state_dict SHA-256 differs from its payload")
        try:
            actor = StructuredGoalFiveCommandActor(
                **contract,
                dimension=dimension,
            )
            actor.load_state_dict(state, strict=True)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise V166ContractError("winner five-command Actor does not strict-load") from exc
        if v166_state_dict_sha256(actor.state_dict()) != expected_sha:
            raise V166ContractError("strict-loaded winner Actor state identity drifted")
        for name, expected in contract.items():
            loaded = getattr(actor, name)
            if not torch.equal(loaded.detach().cpu(), expected):
                raise V166ContractError(
                    f"strict-loaded winner {name} differs from action_contract"
                )
        return cls(
            actor,
            recipe=recipe,
            checkpoint_state_sha256=expected_sha,
            _loader_token=_OFFLINE_LOADER_TOKEN,
        )

    @property
    def recipe(self) -> str:
        return self._recipe

    @property
    def checkpoint_state_sha256(self) -> str:
        return self._checkpoint_state_sha256

    @property
    def deployment_authorized(self) -> bool:
        return False

    def train(self, mode: bool = True) -> "V166OfflineActor":
        if mode:
            raise RuntimeError("V166OfflineActor is inference-only")
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(
        self,
        *,
        current_full256: torch.Tensor,
        view_valid: torch.Tensor,
        current_state: torch.Tensor,
        past_completed_commands: torch.Tensor,
        desired_joint_goal: torch.Tensor,
        baseline_kind: str,
        conditioning_past_completed_commands: torch.Tensor | None = None,
        baseline_joint_goal: torch.Tensor | None = None,
        park_mask: torch.Tensor | None = None,
    ) -> V166OfflineProposal:
        if baseline_kind != "repeat" or baseline_joint_goal is not None:
            raise V166ContractError(
                "V16.6 winner inference must use its trained repeat baseline"
            )
        if conditioning_past_completed_commands is not None:
            raise V166ContractError(
                "V16.6 winner inference does not admit alternate past conditioning"
            )
        output = self._actor(
            current_full256=current_full256,
            view_valid=view_valid,
            current_state=current_state,
            past_completed_commands=past_completed_commands,
            conditioning_past_completed_commands=conditioning_past_completed_commands,
            desired_joint_goal=desired_joint_goal,
            baseline_kind=baseline_kind,
            baseline_joint_goal=baseline_joint_goal,
            park_mask=park_mask,
        )
        commands = output.action
        if commands.shape != (len(current_state), COMMAND_STEPS, STATE_DIM):
            raise V166ContractError("strict-loaded winner emitted a non-[B,5,6] proposal")
        return V166OfflineProposal(
            commands=commands,
            recipe=self._recipe,
            checkpoint_state_sha256=self._checkpoint_state_sha256,
        )

    def propose(self, **kwargs: Any) -> V166OfflineProposal:
        """Named offline proposal entry point; identical to ``forward``."""

        return self(**kwargs)


__all__ = [
    "ARM_CHECKPOINT_FORMAT",
    "ARM_RECEIPT_FORMAT",
    "BATCH_SIZE",
    "CAUSAL4_THEN_H0_RECIPE",
    "CurriculumPhase",
    "FINETUNE_PHASE",
    "FINETUNE_STEPS",
    "H0_THEN_H0_RECIPE",
    "PRETRAIN_PHASE",
    "PRETRAIN_STEPS",
    "PROTOCOL_ID",
    "Recipe",
    "TOTAL_STEPS",
    "V166_ACTOR_MODEL_FORMAT",
    "V166_PHASES",
    "V166_RECIPES",
    "VELOCITY_WEIGHT",
    "WINNER_CHECKPOINT_FORMAT",
    "WINNER_RECEIPT_FORMAT",
    "V166ActorObjective",
    "V166ContractError",
    "V166OfflineActor",
    "V166OfflineProposal",
    "v166_actor_objective",
    "v166_phase_transitions",
    "v166_state_dict_sha256",
]
