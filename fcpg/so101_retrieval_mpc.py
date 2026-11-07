"""Small, deployment-oriented helpers for SO-101 retrieval MPC.

The action bank comes from successful training demonstrations.  These helpers
only construct and score candidates; they do not access robot hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RetrievalCandidates:
    rows: np.ndarray
    episodes: np.ndarray
    latent_distance: np.ndarray
    state_distance: np.ndarray
    combined_distance: np.ndarray


@dataclass(frozen=True)
class PairedCameraContract:
    dataset_id: str
    frames: int
    primary_camera: str
    secondary_camera: str


@dataclass(frozen=True)
class CandidateSafety:
    safe: np.ndarray
    finite: np.ndarray
    command_finite: np.ndarray
    prediction_finite: np.ndarray
    hard_command_safe: np.ndarray
    prediction_plausible: np.ndarray
    first_command_jump: np.ndarray
    prefix_max_step_jump: np.ndarray
    first_jump_ok: np.ndarray
    prefix_step_ok: np.ndarray
    command_bounds_ok: np.ndarray
    prediction_bounds_ok: np.ndarray


SO101_POSITION_LOWER = np.asarray(
    [-100.0, -100.0, -100.0, -100.0, -100.0, 0.0], dtype=np.float32
)
SO101_POSITION_UPPER = np.asarray(
    [100.0, 100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float32
)


def validate_paired_camera_data(
    primary: Any,
    secondary: Any,
) -> PairedCameraContract:
    """Require two camera caches to describe the same physical rows.

    Camera pixels are intentionally allowed to differ. Everything that defines
    the robot trajectory must be byte-identical so both views use the same
    action candidates, split, and held-out future.
    """

    primary_dataset = str(primary.dataset_id)
    secondary_dataset = str(secondary.dataset_id)
    if primary_dataset != secondary_dataset:
        raise ValueError(
            "paired camera caches must have the same dataset_id: "
            f"{primary_dataset!r} != {secondary_dataset!r}"
        )
    primary_camera = str(primary.camera)
    secondary_camera = str(secondary.camera)
    if primary_camera == secondary_camera:
        raise ValueError("paired camera caches must use different camera keys")

    primary_frames = np.asarray(primary.frames)
    secondary_frames = np.asarray(secondary.frames)
    if primary_frames.shape != secondary_frames.shape:
        raise ValueError(
            "paired camera frame arrays must have matching shapes: "
            f"{primary_frames.shape} != {secondary_frames.shape}"
        )
    if primary_frames.ndim != 4 or primary_frames.shape[-1] != 3:
        raise ValueError("paired camera frames must have shape [N,H,W,3]")

    for name in ("episodes", "frame_indices", "tasks", "actions", "states"):
        primary_value = np.asarray(getattr(primary, name))
        secondary_value = np.asarray(getattr(secondary, name))
        if primary_value.shape != secondary_value.shape:
            raise ValueError(
                f"paired camera {name} shapes differ: "
                f"{primary_value.shape} != {secondary_value.shape}"
            )
        if not np.array_equal(primary_value, secondary_value):
            raise ValueError(f"paired camera {name} rows are not identical")

    return PairedCameraContract(
        dataset_id=primary_dataset,
        frames=len(primary_frames),
        primary_camera=primary_camera,
        secondary_camera=secondary_camera,
    )


def candidate_start_rows(
    episodes: np.ndarray,
    frame_indices: np.ndarray,
    allowed_episodes: set[int],
    *,
    horizon_raw: int,
    stride: int,
) -> np.ndarray:
    """Return stride-aligned rows whose full action horizon stays in one episode."""

    episode_values = np.asarray(episodes, dtype=np.int64)
    frame_values = np.asarray(frame_indices, dtype=np.int64)
    if episode_values.ndim != 1 or frame_values.shape != episode_values.shape:
        raise ValueError("episodes and frame_indices must be matching vectors")
    if horizon_raw <= 0 or stride <= 0:
        raise ValueError("horizon_raw and stride must be positive")

    starts: list[int] = []
    for episode in sorted(int(value) for value in allowed_episodes):
        rows = np.flatnonzero(episode_values == episode)
        if len(rows) <= horizon_raw:
            continue
        if len(rows) > 1 and not np.all(np.diff(rows) == 1):
            raise ValueError(f"episode {episode} is not contiguous in the cache")
        local = frame_values[rows]
        if len(local) > 1 and not np.all(np.diff(local) == 1):
            raise ValueError(f"episode {episode} has non-contiguous frame indices")
        valid_count = len(rows) - horizon_raw
        starts.extend(int(rows[offset]) for offset in range(0, valid_count, stride))
    return np.asarray(starts, dtype=np.int64)


def terminal_goal_rows(
    episodes: np.ndarray,
    allowed_episodes: set[int],
    *,
    tail_frames: int,
) -> np.ndarray:
    """Select a short terminal goal bank from each successful training episode."""

    episode_values = np.asarray(episodes, dtype=np.int64)
    if episode_values.ndim != 1:
        raise ValueError("episodes must be one-dimensional")
    if tail_frames <= 0:
        raise ValueError("tail_frames must be positive")
    selected: list[int] = []
    for episode in sorted(int(value) for value in allowed_episodes):
        rows = np.flatnonzero(episode_values == episode)
        if len(rows) == 0:
            raise ValueError(f"episode {episode} has no rows")
        selected.extend(int(row) for row in rows[-min(tail_frames, len(rows)) :])
    if not selected:
        raise ValueError("terminal goal bank is empty")
    return np.asarray(selected, dtype=np.int64)


def retrieve_diverse_candidates(
    query_latent: np.ndarray,
    query_state: np.ndarray,
    bank_latents: np.ndarray,
    bank_states: np.ndarray,
    bank_rows: np.ndarray,
    bank_episodes: np.ndarray,
    *,
    state_scale: np.ndarray,
    per_episode: int = 2,
    minimum_row_gap: int = 20,
    state_weight: float = 1.0,
    extra_distance: np.ndarray | None = None,
    extra_weight: float = 1.0,
) -> RetrievalCandidates:
    """Retrieve phase-compatible candidates while retaining episode diversity."""

    latent = np.asarray(query_latent, dtype=np.float32)
    state = np.asarray(query_state, dtype=np.float32)
    latents = np.asarray(bank_latents, dtype=np.float32)
    states = np.asarray(bank_states, dtype=np.float32)
    rows = np.asarray(bank_rows, dtype=np.int64)
    episodes = np.asarray(bank_episodes, dtype=np.int64)
    scale = np.asarray(state_scale, dtype=np.float32)
    if latent.ndim != 1 or latents.ndim != 2 or latents.shape[1:] != latent.shape:
        raise ValueError("query and bank latent dimensions do not match")
    if state.shape != (6,) or states.shape != (len(rows), 6):
        raise ValueError("SO-101 states must have six dimensions")
    if latents.shape[0] != len(rows) or episodes.shape != rows.shape:
        raise ValueError("candidate bank arrays do not align")
    if scale.shape != (6,) or np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise ValueError("state_scale must be a finite positive six-vector")
    if (
        per_episode <= 0
        or minimum_row_gap < 0
        or state_weight < 0
        or extra_weight < 0
    ):
        raise ValueError("retrieval settings are invalid")

    latent_distance = np.mean(np.square(latents - latent[None]), axis=1)
    state_distance = np.mean(
        np.square((states - state[None]) / scale[None]), axis=1
    )
    combined = latent_distance + state_weight * state_distance
    if extra_distance is not None:
        extra = np.asarray(extra_distance, dtype=np.float32)
        if extra.shape != rows.shape or not np.all(np.isfinite(extra)):
            raise ValueError("extra_distance must be a finite candidate vector")
        if np.any(extra < 0):
            raise ValueError("extra_distance cannot be negative")
        combined = combined + extra_weight * extra

    selected: list[int] = []
    for episode in sorted(int(value) for value in np.unique(episodes)):
        episode_indices = np.flatnonzero(episodes == episode)
        order = episode_indices[np.argsort(combined[episode_indices], kind="stable")]
        kept: list[int] = []
        for index in order:
            row = int(rows[index])
            if any(abs(row - int(rows[previous])) < minimum_row_gap for previous in kept):
                continue
            kept.append(int(index))
            if len(kept) == per_episode:
                break
        selected.extend(kept)
    if not selected:
        raise ValueError("retrieval produced no candidates")
    selected_array = np.asarray(selected, dtype=np.int64)
    selected_array = selected_array[
        np.argsort(combined[selected_array], kind="stable")
    ]
    return RetrievalCandidates(
        rows=rows[selected_array],
        episodes=episodes[selected_array],
        latent_distance=latent_distance[selected_array],
        state_distance=state_distance[selected_array],
        combined_distance=combined[selected_array],
    )


def raw_action_prefixes(
    actions: np.ndarray,
    start_rows: np.ndarray,
    *,
    horizon_raw: int,
) -> np.ndarray:
    action_values = np.asarray(actions, dtype=np.float32)
    starts = np.asarray(start_rows, dtype=np.int64)
    if action_values.ndim != 2 or action_values.shape[1] != 6:
        raise ValueError("actions must have shape [N, 6]")
    if starts.ndim != 1 or horizon_raw <= 0:
        raise ValueError("start_rows must be a vector and horizon_raw must be positive")
    prefixes = []
    for row in starts:
        prefix = action_values[int(row) : int(row) + horizon_raw]
        if prefix.shape != (horizon_raw, 6):
            raise ValueError(f"incomplete action prefix at row {int(row)}")
        prefixes.append(prefix)
    if not prefixes:
        raise ValueError("action prefix bank is empty")
    return np.stack(prefixes).astype(np.float32, copy=False)


def rebase_action_prefixes(
    raw_prefixes: np.ndarray,
    candidate_states: np.ndarray,
    current_state: np.ndarray,
) -> np.ndarray:
    """Translate absolute demonstration commands to the current joint state."""

    prefixes = np.asarray(raw_prefixes, dtype=np.float32)
    starts = np.asarray(candidate_states, dtype=np.float32)
    current = np.asarray(current_state, dtype=np.float32)
    if prefixes.ndim != 3 or prefixes.shape[-1] != 6:
        raise ValueError("raw_prefixes must have shape [K, R, 6]")
    if starts.shape != (len(prefixes), 6) or current.shape != (6,):
        raise ValueError("candidate and current states do not align with prefixes")
    shifted = prefixes + (current[None, None] - starts[:, None])
    if not np.all(np.isfinite(shifted)):
        raise FloatingPointError("rebased action prefixes are non-finite")
    return shifted.astype(np.float32, copy=False)


def chunk_and_normalize_actions(
    raw_prefixes: np.ndarray,
    *,
    frameskip: int,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    prefixes = np.asarray(raw_prefixes, dtype=np.float32)
    center_values = np.asarray(center, dtype=np.float32)
    scale_values = np.asarray(scale, dtype=np.float32)
    if prefixes.ndim != 3 or prefixes.shape[-1] != 6:
        raise ValueError("raw_prefixes must have shape [K, R, 6]")
    if frameskip <= 0 or prefixes.shape[1] % frameskip:
        raise ValueError("raw action horizon must be divisible by frameskip")
    if center_values.shape != (6,) or scale_values.shape != (6,) or np.any(scale_values <= 0):
        raise ValueError("action normalization must contain positive six-vectors")
    normalized = (prefixes - center_values[None, None]) / scale_values[None, None]
    return normalized.reshape(
        len(prefixes), prefixes.shape[1] // frameskip, frameskip * 6
    ).astype(np.float32, copy=False)


def terminal_goal_cost(
    predicted_latents: np.ndarray,
    goal_latents: np.ndarray,
    *,
    final_steps: int = 3,
) -> np.ndarray:
    """Score each candidate by its late-horizon distance to any demonstrated goal."""

    predictions = np.asarray(predicted_latents, dtype=np.float32)
    goals = np.asarray(goal_latents, dtype=np.float32)
    if predictions.ndim != 3 or goals.ndim != 2:
        raise ValueError("predictions must be [K,H,D] and goals must be [G,D]")
    if predictions.shape[-1] != goals.shape[-1] or len(goals) == 0:
        raise ValueError("prediction and goal latent dimensions do not match")
    if not 1 <= final_steps <= predictions.shape[1]:
        raise ValueError("final_steps must fit inside the prediction horizon")
    tail = predictions[:, -final_steps:]
    squared = np.mean(
        np.square(tail[:, :, None, :] - goals[None, None, :, :]), axis=-1
    )
    return np.mean(np.min(squared, axis=2), axis=1)


def terminal_state_cost(
    predicted_states: np.ndarray,
    goal_states: np.ndarray,
    *,
    final_steps: int = 3,
) -> np.ndarray:
    """Score normalized future states against successful terminal states."""

    predictions = np.asarray(predicted_states, dtype=np.float32)
    goals = np.asarray(goal_states, dtype=np.float32)
    if predictions.ndim != 3 or goals.ndim != 2:
        raise ValueError("predictions must be [K,H,S] and goals must be [G,S]")
    if predictions.shape[-1] != goals.shape[-1] or len(goals) == 0:
        raise ValueError("prediction and goal state dimensions do not match")
    if not 1 <= final_steps <= predictions.shape[1]:
        raise ValueError("final_steps must fit inside the prediction horizon")
    tail = predictions[:, -final_steps:]
    squared = np.mean(
        np.square(tail[:, :, None, :] - goals[None, None, :, :]), axis=-1
    )
    return np.mean(np.min(squared, axis=2), axis=1)


def candidate_safety_checks(
    first_commands: np.ndarray,
    current_state: np.ndarray,
    *,
    max_joint_jump: float,
    raw_prefixes: np.ndarray | None = None,
    predicted_states: np.ndarray | None = None,
    position_lower: np.ndarray | None = None,
    position_upper: np.ndarray | None = None,
    max_prefix_step_jump: float | None = None,
    prediction_bound_margin: float = 0.0,
) -> CandidateSafety:
    """Evaluate deployment safety without clipping an unsafe command."""

    commands = np.asarray(first_commands, dtype=np.float32)
    state = np.asarray(current_state, dtype=np.float32)
    if commands.ndim != 2 or commands.shape[1:] != (6,) or state.shape != (6,):
        raise ValueError("first commands and current state do not align")
    if max_joint_jump <= 0 or not np.isfinite(max_joint_jump):
        raise ValueError("max_joint_jump must be finite and positive")
    if max_prefix_step_jump is None:
        max_prefix_step_jump = max_joint_jump
    if max_prefix_step_jump <= 0 or not np.isfinite(max_prefix_step_jump):
        raise ValueError("max_prefix_step_jump must be finite and positive")
    if prediction_bound_margin < 0 or not np.isfinite(prediction_bound_margin):
        raise ValueError("prediction_bound_margin must be finite and non-negative")

    candidate_count = len(commands)
    prefixes = commands[:, None]
    if raw_prefixes is not None:
        prefixes = np.asarray(raw_prefixes, dtype=np.float32)
        if (
            prefixes.ndim != 3
            or prefixes.shape[0] != candidate_count
            or prefixes.shape[2] != 6
        ):
            raise ValueError("raw_prefixes must have shape [K,R,6]")
        if prefixes.shape[1] == 0:
            raise ValueError("raw_prefixes cannot be empty")
        if not np.array_equal(prefixes[:, 0], commands):
            raise ValueError("first_commands must equal raw_prefixes[:, 0]")

    predictions: np.ndarray | None = None
    if predicted_states is not None:
        predictions = np.asarray(predicted_states, dtype=np.float32)
        if (
            predictions.ndim != 3
            or predictions.shape[0] != candidate_count
            or predictions.shape[2] != 6
        ):
            raise ValueError("predicted_states must have shape [K,H,6]")

    bounds_supplied = position_lower is not None or position_upper is not None
    if bounds_supplied and (position_lower is None or position_upper is None):
        raise ValueError("position_lower and position_upper must be supplied together")
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    if bounds_supplied:
        lower = np.asarray(position_lower, dtype=np.float32)
        upper = np.asarray(position_upper, dtype=np.float32)
        if lower.shape != (6,) or upper.shape != (6,) or np.any(lower >= upper):
            raise ValueError("position bounds must be ordered six-vectors")

    jumps = np.max(np.abs(commands - state[None]), axis=1)
    trajectory = np.concatenate(
        (np.repeat(state[None, None], candidate_count, axis=0), prefixes), axis=1
    )
    prefix_jumps = np.max(np.abs(np.diff(trajectory, axis=1)), axis=(1, 2))
    command_finite = (
        np.all(np.isfinite(prefixes), axis=(1, 2))
        & np.all(np.isfinite(commands), axis=1)
        & bool(np.all(np.isfinite(state)))
    )
    prediction_finite = np.ones(candidate_count, dtype=bool)
    if predictions is not None:
        prediction_finite = np.all(np.isfinite(predictions), axis=(1, 2))
    finite = command_finite & prediction_finite
    first_jump_ok = jumps <= max_joint_jump
    prefix_step_ok = prefix_jumps <= max_prefix_step_jump
    command_bounds_ok = np.ones(candidate_count, dtype=bool)
    prediction_bounds_ok = np.ones(candidate_count, dtype=bool)
    if lower is not None and upper is not None:
        command_bounds_ok = np.all(
            (prefixes >= lower[None, None]) & (prefixes <= upper[None, None]),
            axis=(1, 2),
        )
        command_bounds_ok &= bool(np.all((state >= lower) & (state <= upper)))
        if predictions is not None:
            prediction_lower = lower - prediction_bound_margin
            prediction_upper = upper + prediction_bound_margin
            prediction_bounds_ok = np.all(
                (predictions >= prediction_lower[None, None])
                & (predictions <= prediction_upper[None, None]),
                axis=(1, 2),
            )
    hard_command_safe = (
        command_finite
        & first_jump_ok
        & prefix_step_ok
        & command_bounds_ok
    )
    prediction_plausible = prediction_finite & prediction_bounds_ok
    safe = hard_command_safe & prediction_plausible
    return CandidateSafety(
        safe=safe,
        finite=finite,
        command_finite=command_finite,
        prediction_finite=prediction_finite,
        hard_command_safe=hard_command_safe,
        prediction_plausible=prediction_plausible,
        first_command_jump=jumps,
        prefix_max_step_jump=prefix_jumps,
        first_jump_ok=first_jump_ok,
        prefix_step_ok=prefix_step_ok,
        command_bounds_ok=command_bounds_ok,
        prediction_bounds_ok=prediction_bounds_ok,
    )


def select_safe_candidate(
    costs: np.ndarray,
    first_commands: np.ndarray,
    current_state: np.ndarray,
    *,
    max_joint_jump: float,
    raw_prefixes: np.ndarray | None = None,
    predicted_states: np.ndarray | None = None,
    position_lower: np.ndarray | None = None,
    position_upper: np.ndarray | None = None,
    max_prefix_step_jump: float | None = None,
    prediction_bound_margin: float = 0.0,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Choose the lowest-cost candidate that passes every requested safety check."""

    score = np.asarray(costs, dtype=np.float64)
    commands = np.asarray(first_commands, dtype=np.float32)
    if score.ndim != 1 or commands.shape != (len(score), 6):
        raise ValueError("candidate costs and first commands do not align")
    checks = candidate_safety_checks(
        commands,
        current_state,
        max_joint_jump=max_joint_jump,
        raw_prefixes=raw_prefixes,
        predicted_states=predicted_states,
        position_lower=position_lower,
        position_upper=position_upper,
        max_prefix_step_jump=max_prefix_step_jump,
        prediction_bound_margin=prediction_bound_margin,
    )
    safe = checks.safe
    if np.any(safe):
        masked = np.where(safe, score, np.inf)
        return int(np.argmin(masked)), safe, checks.first_command_jump
    # Diagnostic fallback only. A deployment caller must stop when safe is all false.
    return int(np.argmin(checks.first_command_jump)), safe, checks.first_command_jump


def prefix_rmse(candidate: np.ndarray, target: np.ndarray) -> float:
    candidate_values = np.asarray(candidate, dtype=np.float32)
    target_values = np.asarray(target, dtype=np.float32)
    if candidate_values.shape != target_values.shape:
        raise ValueError("candidate and target action prefixes must have matching shapes")
    return float(np.sqrt(np.mean(np.square(candidate_values - target_values))))
