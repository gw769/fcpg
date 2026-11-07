"""V11.6 dual-view LeWM rank rollout. Goal-free: state/history + action → future.

B2 single-view wrist scoring is a backup in the planner, not this path.
Structured goal and task_class never enter this API.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import torch

from fcpg.so101_lewm_rank_views import RANK_VIEWS
from fcpg.so101_predictive_action_policy import SO101PredictiveActionPolicy
from v10.data import JOINT_SCALE
from v10.lewm_canonical_forward import (
    assert_no_goal_parameter,
    assert_no_task_class_parameter,
    lewm_canonical_forward,
)

V11_6_ROLLOUT_VARIANT = "v11_6_dual_view"
# Cap concurrent CUDA candidates. Overnight serve died mid 34-candidate rank
# with empty HTTP replies and no Python traceback; smaller batches cut peak
# native/CUDA surface under ThreadingHTTPServer polling.
V11_6_RANK_BATCH = 8
RANK_INFERENCE_LOCK = threading.RLock()


def require_v11_6_rank_policy(model: Any) -> SO101PredictiveActionPolicy:
    """Accept a V11.6 policy or an SO101V10 wrapper. Require two views."""

    policy = model if isinstance(model, SO101PredictiveActionPolicy) else getattr(
        model, "policy", None
    )
    if not isinstance(policy, SO101PredictiveActionPolicy):
        raise TypeError("V11.6 rank requires SO101PredictiveActionPolicy")
    if int(policy.num_views) != int(RANK_VIEWS):
        raise ValueError(
            f"V11.6 rank requires num_views={RANK_VIEWS}, got {policy.num_views}"
        )
    return policy


def stack_rank_view_contexts(
    wrist_context: np.ndarray,
    column_context: np.ndarray,
) -> np.ndarray:
    """Stack wrist then column into [T, 2, D]."""

    wrist = np.asarray(wrist_context, dtype=np.float32)
    column = np.asarray(column_context, dtype=np.float32)
    if wrist.ndim != 2 or column.ndim != 2:
        raise ValueError("rank view contexts must have shape [T, D]")
    if wrist.shape != column.shape:
        raise ValueError(
            f"wrist/column rank contexts must match, got {wrist.shape} vs {column.shape}"
        )
    if not np.all(np.isfinite(wrist)) or not np.all(np.isfinite(column)):
        raise ValueError("rank view contexts must be finite")
    return np.stack((wrist, column), axis=1)


def _tail_or_pad(values: np.ndarray, length: int) -> np.ndarray:
    if length < 1:
        raise ValueError("history length must be positive")
    if values.shape[0] >= length:
        return values[-length:]
    pad = np.repeat(values[:1], length - values.shape[0], axis=0)
    return np.concatenate((pad, values), axis=0)


def _broadcast_past(
    past_chunks: np.ndarray,
    *,
    n_cand: int,
    need: int,
    flat_dim: int,
) -> np.ndarray:
    past = np.asarray(past_chunks, dtype=np.float32)
    if past.ndim == 3 and past.shape[-1] == flat_dim:
        if past.shape[0] == n_cand:
            seq = past
        else:
            seq = np.repeat(past.reshape(1, -1, flat_dim), n_cand, axis=0)
    elif past.ndim == 2 and past.shape[-1] == flat_dim:
        seq = np.repeat(past[None], n_cand, axis=0)
    else:
        raise ValueError(
            "past_chunks must be [M, F] or [N, T, F] with F="
            f"{flat_dim}, got {past.shape}"
        )
    if seq.shape[1] >= need:
        return seq[:, -need:]
    pad = np.repeat(seq[:, :1], need - seq.shape[1], axis=1)
    return np.concatenate((pad, seq), axis=1)


def _to_v11_action(
    candidate_chunks: np.ndarray,
    *,
    action_horizon: int,
    raw_steps: int,
    state_dim: int,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    chunks = np.asarray(candidate_chunks, dtype=np.float32)
    n_cand, horizon, flat = chunks.shape
    if flat != raw_steps * state_dim:
        raise ValueError(
            f"candidate chunks must be [N, H, {raw_steps * state_dim}], got {chunks.shape}"
        )
    center6 = np.asarray(center, dtype=np.float32).reshape(1, 1, 1, state_dim)
    scale6 = np.asarray(scale, dtype=np.float32).reshape(1, 1, 1, state_dim)
    raw = chunks.reshape(n_cand, horizon, raw_steps, state_dim) * scale6 + center6
    steps = raw.reshape(n_cand, horizon * raw_steps, state_dim)
    if steps.shape[1] < action_horizon:
        pad = np.repeat(steps[:, -1:], action_horizon - steps.shape[1], axis=1)
        steps = np.concatenate((steps, pad), axis=1)
    else:
        steps = steps[:, :action_horizon]
    return (steps / float(JOINT_SCALE)).astype(np.float32)


def _to_planner_states(
    future_state: np.ndarray,
    *,
    horizon: int,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(future_state, dtype=np.float32) * float(JOINT_SCALE)
    if raw.shape[1] < horizon:
        pad = np.repeat(raw[:, -1:], horizon - raw.shape[1], axis=1)
        raw = np.concatenate((raw, pad), axis=1)
    else:
        raw = raw[:, :horizon]
    center6 = np.asarray(center, dtype=np.float32).reshape(1, 1, -1)
    scale6 = np.asarray(scale, dtype=np.float32).reshape(1, 1, -1)
    return ((raw - center6) / scale6).astype(np.float32)


def batched_lewm_canonical_forward(
    policy: SO101PredictiveActionPolicy,
    *,
    action: torch.Tensor,
    **kwargs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Goal-free LeWM forward in small CUDA batches. No goal. No task_class."""

    assert_no_goal_parameter(batched_lewm_canonical_forward)
    assert_no_task_class_parameter(batched_lewm_canonical_forward)
    assert_no_goal_parameter(lewm_canonical_forward)
    assert_no_task_class_parameter(lewm_canonical_forward)
    leaked = [
        name
        for name in kwargs
        if "goal" in name.lower() or "task_class" in name.lower()
    ]
    if leaked:
        raise RuntimeError(f"LeWM rank must not accept goal/task tags: {sorted(leaked)}")
    n_cand = int(action.shape[0])
    batch = max(1, int(V11_6_RANK_BATCH))
    futures: list[torch.Tensor] = []
    latents: list[torch.Tensor] = []
    with RANK_INFERENCE_LOCK:
        with torch.inference_mode():
            for start in range(0, n_cand, batch):
                stop = min(n_cand, start + batch)
                future, latent = lewm_canonical_forward(
                    policy,
                    action=action[start:stop],
                    **{name: value[start:stop] for name, value in kwargs.items()},
                )
                futures.append(future)
                latents.append(latent)
    return torch.cat(futures, dim=0), torch.cat(latents, dim=0)


def snapshot_rank_observations(observations: Any) -> list[Any]:
    """Copy the live deque so encode cannot race an HTTP-thread append."""

    from dataclasses import is_dataclass, replace

    snapped: list[Any] = []
    for item in list(observations):
        frames = getattr(item, "frames", None)
        state = getattr(item, "state", None)
        if (
            is_dataclass(item)
            and isinstance(frames, dict)
            and state is not None
        ):
            copied = {
                key: np.asarray(value).copy() for key, value in frames.items()
            }
            snapped.append(
                replace(
                    item,
                    frames=copied,
                    state=np.asarray(state, dtype=np.float32).copy(),
                )
            )
        else:
            snapped.append(item)
    return snapped


def _anchor_field(
    measured: np.ndarray | None,
    *,
    n_cand: int,
    obs_horizon: int,
    trailing: tuple[int, ...],
    fill: float,
) -> np.ndarray:
    """One anchor input, either as FCPG measured it or as ranking invents it."""

    if measured is None:
        return np.full((n_cand, obs_horizon, *trailing), fill, dtype=np.float32)
    values = np.asarray(measured, dtype=np.float32)
    if values.shape[1:] != trailing:
        raise ValueError(
            f"anchor field has trailing shape {values.shape[1:]}, expected {trailing}"
        )
    return np.repeat(_tail_or_pad(values, obs_horizon)[None], n_cand, axis=0)


def build_v11_6_rank_inputs(
    policy: SO101PredictiveActionPolicy,
    wrist_context: np.ndarray,
    column_context: np.ndarray,
    normalized_states: np.ndarray,
    past_chunks: np.ndarray,
    candidate_chunks: np.ndarray,
    *,
    state_center: np.ndarray,
    state_scale: np.ndarray,
    action_center: np.ndarray,
    action_scale: np.ndarray,
    device: torch.device,
    anchor_px: np.ndarray | None = None,
    anchor_xyz: np.ndarray | None = None,
    anchor_seen: np.ndarray | None = None,
    anchor_tokens: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """Assemble the goal-free LeWM inputs for a bank of candidate chunks.

    Shared so that anything reading this model reads it through exactly the
    inputs ranking used. A residual measured against a differently assembled
    forward pass would be measuring the assembly.

    The four grounded fields are an all-or-none contract. If they are omitted,
    this function constructs an explicitly ungrounded baseline: the visual
    anchor mask is false and zero anchor tokens cannot affect the policy. It
    never labels synthetic zero tokens as valid. Training fed this model FCPG's
    measured anchors, so a caller claiming a grounded rank must pass all four
    fields. Pass
    ``anchor_px`` as ``[T, views, anchors, 2]`` in the view order this module
    stacks contexts in, ``anchor_xyz`` as ``[T, anchors, 3]``, and
    ``anchor_seen`` as ``[T, views, anchors]``. Pass ``anchor_tokens`` as
    ``[T, views, anchors, latent_dim]`` from the same visual/FCPG forward.
    """

    assert_no_goal_parameter(build_v11_6_rank_inputs)
    assert_no_task_class_parameter(build_v11_6_rank_inputs)
    policy = require_v11_6_rank_policy(policy)

    n_cand, _horizon, _ = np.asarray(candidate_chunks).shape
    stacked = stack_rank_view_contexts(wrist_context, column_context)
    if stacked.shape[-1] != policy.latent_dim:
        raise ValueError(
            "V11.6 rank context dim "
            f"{stacked.shape[-1]} != policy.latent_dim {policy.latent_dim}"
        )
    history = _tail_or_pad(stacked, policy.obs_horizon)
    states = _tail_or_pad(np.asarray(normalized_states, dtype=np.float32), policy.obs_horizon)
    raw_states = (
        states * np.asarray(state_scale, dtype=np.float32)
        + np.asarray(state_center, dtype=np.float32)
    ) / float(JOINT_SCALE)
    flat = policy.raw_steps_per_chunk * policy.state_dim
    past = _broadcast_past(
        past_chunks, n_cand=n_cand, need=policy.obs_horizon - 1, flat_dim=flat
    )
    past_raw = past.reshape(n_cand, policy.obs_horizon - 1, policy.raw_steps_per_chunk, 6)
    past_raw = (
        past_raw * np.asarray(action_scale, dtype=np.float32).reshape(1, 1, 1, 6)
        + np.asarray(action_center, dtype=np.float32).reshape(1, 1, 1, 6)
    ) / float(JOINT_SCALE)
    action = _to_v11_action(
        candidate_chunks,
        action_horizon=policy.action_horizon,
        raw_steps=policy.raw_steps_per_chunk,
        state_dim=policy.state_dim,
        center=action_center,
        scale=action_scale,
    )

    per_view = np.repeat(history[None], n_cand, axis=0)
    state_hist = np.repeat(raw_states[None], n_cand, axis=0)
    grounded = (anchor_px, anchor_xyz, anchor_seen, anchor_tokens)
    if any(value is not None for value in grounded) and not all(
        value is not None for value in grounded
    ):
        raise ValueError(
            "anchor_px, anchor_xyz, anchor_seen, and anchor_tokens must be "
            "provided together"
        )
    if anchor_tokens is None:
        tokens = np.zeros(
            (
                n_cand,
                policy.obs_horizon,
                RANK_VIEWS,
                policy.num_anchors,
                policy.latent_dim,
            ),
            dtype=np.float32,
        )
        anchor_valid = np.zeros(
            (n_cand, policy.obs_horizon, RANK_VIEWS, policy.num_anchors),
            dtype=bool,
        )
    else:
        tokens = _anchor_field(
            anchor_tokens,
            n_cand=n_cand,
            obs_horizon=policy.obs_horizon,
            trailing=(RANK_VIEWS, policy.num_anchors, policy.latent_dim),
            fill=0.0,
        )
        anchor_valid = _anchor_field(
            anchor_seen,
            n_cand=n_cand,
            obs_horizon=policy.obs_horizon,
            trailing=(RANK_VIEWS, policy.num_anchors),
            fill=0.0,
        ).astype(bool)
    q_px = _anchor_field(
        anchor_px,
        n_cand=n_cand,
        obs_horizon=policy.obs_horizon,
        trailing=(RANK_VIEWS, policy.num_anchors, 2),
        fill=0.5 * float(policy.image_size - 1),
    )
    anchors = _anchor_field(
        anchor_xyz,
        n_cand=n_cand,
        obs_horizon=policy.obs_horizon,
        trailing=(policy.num_anchors, 3),
        fill=0.0,
    )
    view_valid = np.ones((n_cand, policy.obs_horizon, RANK_VIEWS), dtype=bool)

    def tensor(value: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(value, device=device, dtype=dtype)

    past_flat = past_raw.reshape(n_cand, policy.obs_horizon - 1, flat)
    return {
        "action": tensor(action),
        "per_view_context": tensor(per_view),
        "per_view_anchor_tokens": tensor(tokens),
        "anchor_valid": tensor(anchor_valid, dtype=torch.bool),
        "view_valid": tensor(view_valid, dtype=torch.bool),
        "q_corrected_px": tensor(q_px),
        "anchor_xyz_history": tensor(anchors),
        "state_history": tensor(state_hist),
        "past_action_chunks": tensor(past_flat),
    }


def rollout_v11_6_candidates(
    policy: SO101PredictiveActionPolicy,
    wrist_context: np.ndarray,
    column_context: np.ndarray,
    normalized_states: np.ndarray,
    past_chunks: np.ndarray,
    candidate_chunks: np.ndarray,
    *,
    state_center: np.ndarray,
    state_scale: np.ndarray,
    action_center: np.ndarray,
    action_scale: np.ndarray,
    device: torch.device,
    anchor_px: np.ndarray | None = None,
    anchor_xyz: np.ndarray | None = None,
    anchor_seen: np.ndarray | None = None,
    anchor_tokens: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score candidates with V11.6 dual-view LeWM. No goal. No task_class."""

    assert_no_goal_parameter(rollout_v11_6_candidates)
    assert_no_task_class_parameter(rollout_v11_6_candidates)
    assert_no_goal_parameter(lewm_canonical_forward)
    assert_no_task_class_parameter(lewm_canonical_forward)
    policy = require_v11_6_rank_policy(policy)

    n_cand, horizon, _ = np.asarray(candidate_chunks).shape
    stacked = stack_rank_view_contexts(wrist_context, column_context)
    inputs = build_v11_6_rank_inputs(
        policy,
        wrist_context,
        column_context,
        normalized_states,
        past_chunks,
        candidate_chunks,
        state_center=state_center,
        state_scale=state_scale,
        action_center=action_center,
        action_scale=action_scale,
        device=device,
        anchor_px=anchor_px,
        anchor_xyz=anchor_xyz,
        anchor_seen=anchor_seen,
        anchor_tokens=anchor_tokens,
    )
    future, latent = batched_lewm_canonical_forward(policy, **inputs)
    future_np = future.detach().float().cpu().numpy()
    latent_np = latent.detach().float().cpu().numpy()
    states_out = _to_planner_states(
        future_np,
        horizon=horizon,
        center=state_center,
        scale=state_scale,
    )
    context_dim = int(stacked.shape[-1])
    latent_out = np.zeros((n_cand, horizon, context_dim), dtype=np.float32)
    if latent_np.ndim == 3:
        take = min(latent_np.shape[-1], context_dim)
        latent_out[:, : min(horizon, latent_np.shape[1]), :take] = latent_np[
            :, : min(horizon, latent_np.shape[1]), :take
        ]
    return latent_out, states_out


__all__ = [
    "RANK_INFERENCE_LOCK",
    "V11_6_RANK_BATCH",
    "V11_6_ROLLOUT_VARIANT",
    "batched_lewm_canonical_forward",
    "require_v11_6_rank_policy",
    "rollout_v11_6_candidates",
    "snapshot_rank_observations",
    "stack_rank_view_contexts",
]
