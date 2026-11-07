"""Goal-free LeWM rollout on canonical physical actions.

Independent LeWM training must use this path:
    (state/history, action) -> future state / dynamics_latent

No structured goal argument exists. Production V11's goal-attended
action_latent -> dynamics path is legacy coupling and is not this API.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import torch

from fcpg.so101_predictive_action_policy import (
    SO101PredictiveActionOutput,
    SO101PredictiveActionPolicy,
)


def _assert_lewm_api_is_class_free(fn: Callable[..., object]) -> None:
    names = set(inspect.signature(fn).parameters)
    leaked = {
        name
        for name in names
        if "goal" in name.lower() or "task_class" in name.lower()
    }
    if leaked:
        raise RuntimeError(f"LeWM API must not accept goal/task tags: {sorted(leaked)}")


def lewm_canonical_output(
    policy: SO101PredictiveActionPolicy,
    *,
    per_view_context: torch.Tensor,
    per_view_anchor_tokens: torch.Tensor,
    anchor_valid: torch.Tensor,
    view_valid: torch.Tensor,
    q_corrected_px: torch.Tensor,
    anchor_xyz_history: torch.Tensor,
    state_history: torch.Tensor,
    past_action_chunks: torch.Tensor,
    action: torch.Tensor,
) -> SO101PredictiveActionOutput:
    """Goal-free, task-class-free LeWM rollout. Returns the full dynamics output."""

    _assert_lewm_api_is_class_free(lewm_canonical_output)
    held_goal = policy.structured_goal
    held_tag = getattr(policy, "task_class_conditioning", False)
    policy.structured_goal = False
    if hasattr(policy, "task_class_conditioning"):
        policy.task_class_conditioning = False
    try:
        return policy(
            per_view_context,
            per_view_anchor_tokens,
            anchor_valid,
            view_valid,
            q_corrected_px,
            anchor_xyz_history,
            state_history,
            past_action_chunks,
            forced_dynamics_action=action,
            goal_state=None,
        )
    finally:
        policy.structured_goal = held_goal
        if hasattr(policy, "task_class_conditioning"):
            policy.task_class_conditioning = held_tag


def lewm_canonical_forward(
    policy: SO101PredictiveActionPolicy,
    *,
    per_view_context: torch.Tensor,
    per_view_anchor_tokens: torch.Tensor,
    anchor_valid: torch.Tensor,
    view_valid: torch.Tensor,
    q_corrected_px: torch.Tensor,
    anchor_xyz_history: torch.Tensor,
    state_history: torch.Tensor,
    past_action_chunks: torch.Tensor,
    action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll LeWM on a canonical physical action. Goal is not a parameter."""

    _assert_lewm_api_is_class_free(lewm_canonical_forward)
    output = lewm_canonical_output(
        policy,
        per_view_context=per_view_context,
        per_view_anchor_tokens=per_view_anchor_tokens,
        anchor_valid=anchor_valid,
        view_valid=view_valid,
        q_corrected_px=q_corrected_px,
        anchor_xyz_history=anchor_xyz_history,
        state_history=state_history,
        past_action_chunks=past_action_chunks,
        action=action,
    )
    return output.future_state, output.dynamics_latent


def probe_unified_v11_goal_leak(
    policy: SO101PredictiveActionPolicy,
    forward_kwargs: dict[str, torch.Tensor],
    action: torch.Tensor,
    goal_a: torch.Tensor,
    goal_b: torch.Tensor,
) -> dict[str, float]:
    """Measure legacy coupling. Not a license to train independent LeWM."""

    if not policy.structured_goal:
        raise ValueError("legacy probe requires the unified V11 structured_goal path")
    out_a = policy(
        **forward_kwargs,
        forced_dynamics_action=action,
        goal_state=goal_a,
    )
    out_b = policy(
        **forward_kwargs,
        forced_dynamics_action=action,
        goal_state=goal_b,
    )
    pred_delta = (out_a.future_state - out_b.future_state).abs().max()
    latent_delta = (out_a.dynamics_latent - out_b.dynamics_latent).abs().max()
    return {
        "prediction_max_abs_delta": float(pred_delta),
        "dynamics_latent_max_abs_delta": float(latent_delta),
    }


def lewm_goal_free_contract(
    future_a: torch.Tensor,
    future_b: torch.Tensor,
    latent_a: torch.Tensor,
    latent_b: torch.Tensor,
) -> dict[str, object]:
    """Fail-closed: tensor equality via max_abs_delta, not hashes."""

    pred = float((future_a - future_b).abs().max())
    latent = float((latent_a - latent_b).abs().max())
    passed = pred == 0.0 and latent == 0.0
    return {
        "LEWM_GOAL_FREE_CONTRACT": "PASS" if passed else "FAIL",
        "prediction_max_abs_delta": pred,
        "dynamics_latent_max_abs_delta": latent,
    }


def assert_no_goal_parameter(fn: Callable[..., object]) -> None:
    names = set(inspect.signature(fn).parameters)
    leaked = {name for name in names if "goal" in name.lower()}
    if leaked:
        raise AssertionError(f"LeWM API must not accept goal parameters: {sorted(leaked)}")


def assert_no_task_class_parameter(fn: Callable[..., object]) -> None:
    names = set(inspect.signature(fn).parameters)
    leaked = {name for name in names if "task_class" in name.lower()}
    if leaked:
        raise AssertionError(
            f"LeWM API must not accept task_class parameters: {sorted(leaked)}"
        )
