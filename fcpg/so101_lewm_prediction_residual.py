"""Ask the world model, after the fact, whether the world did what it said.

The live loop consults LeWM only before acting: it rolls candidates out, turns
the rollout into scalar costs, ranks, and throws the prediction away. Nothing
ever compares what the model said would happen against what happened. That gap
is why a grasp could close on nothing and be recorded as a hold -- the question
"am I holding it" was answered by a threshold ladder on pixels in a single view,
and the component whose output space actually contains the object was not asked.

The model does predict the object. ``SO101PredictiveActionPolicy`` emits
``future_context`` alongside ``future_state``: the per-view 192-d frozen-ViT
latent the scene is expected to have after the action, for both views. Ranking
uses the joint states and discards the rest. This module keeps the discarded
half and pairs it with the latent the next observation actually produces.

Nothing here trains or conditions anything. It reads a frozen goal-free model
through the same inputs ranking uses, which is the only way the residual is a
statement about the world rather than about the assembly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fcpg.so101_predictive_action_policy import SO101PredictiveActionPolicy
from fcpg.so101_v11_6_lewm_rank import (
    RANK_INFERENCE_LOCK,
    build_v11_6_rank_inputs,
    require_v11_6_rank_policy,
)
from v10.lewm_canonical_forward import lewm_canonical_output

#: Views in the order ``stack_rank_view_contexts`` produces them.
RESIDUAL_VIEWS = ("wrist", "column")


@dataclass(frozen=True)
class PredictedFuture:
    """What the world model expects to see after the chunk that was executed."""

    #: [horizon, views, latent] predicted per-view scene latent.
    context: np.ndarray
    #: [horizon, state] predicted joint state, in the model's own raw units.
    state: np.ndarray
    #: The round whose observation this prediction was made from.
    round_id: str

    def __post_init__(self) -> None:
        context = np.asarray(self.context, dtype=np.float32)
        if context.ndim != 3 or context.shape[1] != len(RESIDUAL_VIEWS):
            raise ValueError("context must be [horizon, views, latent]")
        if not np.all(np.isfinite(context)):
            raise ValueError("context must be finite")


def predict_future_context(
    policy: SO101PredictiveActionPolicy,
    wrist_context: np.ndarray,
    column_context: np.ndarray,
    normalized_states: np.ndarray,
    past_chunks: np.ndarray,
    executed_chunk: np.ndarray,
    *,
    state_center: np.ndarray,
    state_scale: np.ndarray,
    action_center: np.ndarray,
    action_scale: np.ndarray,
    device: torch.device,
    round_id: str,
    anchor_px: np.ndarray | None = None,
    anchor_xyz: np.ndarray | None = None,
    anchor_seen: np.ndarray | None = None,
) -> PredictedFuture:
    """Roll the frozen world model once, on the chunk that is about to run.

    One candidate rather than the bank: the residual is only meaningful for the
    action the arm actually took, and the others were never realised.

    Passing the anchor arguments asks the model the question it was trained on.
    Leaving them out asks the question ranking asks, which is the same question
    with FCPG's grounding replaced by zeros and image centres. Both are worth
    recording, and only their difference says whether that replacement costs
    anything.
    """

    policy = require_v11_6_rank_policy(policy)
    chunk = np.asarray(executed_chunk, dtype=np.float32)
    if chunk.ndim == 2:
        chunk = chunk[None]
    if chunk.ndim != 3 or chunk.shape[0] != 1:
        raise ValueError("executed_chunk must be a single [horizon, action] chunk")
    inputs = build_v11_6_rank_inputs(
        policy,
        wrist_context,
        column_context,
        normalized_states,
        past_chunks,
        chunk,
        state_center=state_center,
        state_scale=state_scale,
        action_center=action_center,
        action_scale=action_scale,
        device=device,
        anchor_px=anchor_px,
        anchor_xyz=anchor_xyz,
        anchor_seen=anchor_seen,
    )
    with RANK_INFERENCE_LOCK:
        with torch.inference_mode():
            output = lewm_canonical_output(policy, **inputs)
    return PredictedFuture(
        context=output.future_context[0].detach().float().cpu().numpy(),
        state=output.future_state[0].detach().float().cpu().numpy(),
        round_id=str(round_id),
    )


def context_residual(
    predicted: PredictedFuture,
    realized_wrist_context: np.ndarray,
    realized_column_context: np.ndarray,
    *,
    horizon_index: int = -1,
) -> dict[str, float]:
    """Distance between the predicted scene and the observed one, per view.

    Reported both as a raw distance and normalised by how much the scene moved
    at all. A prediction that changed nothing scores well whenever the world
    changed nothing, so the residual alone cannot say whether the model knew
    anything; the ratio against the observed change can.
    """

    realized = np.stack(
        (
            np.asarray(realized_wrist_context, dtype=np.float32).reshape(-1),
            np.asarray(realized_column_context, dtype=np.float32).reshape(-1),
        )
    )
    expected = np.asarray(predicted.context, dtype=np.float32)
    if not -expected.shape[0] <= int(horizon_index) < expected.shape[0]:
        raise ValueError("horizon_index is outside the predicted horizon")
    expected = expected[int(horizon_index)]
    if expected.shape != realized.shape:
        raise ValueError(
            f"predicted context {expected.shape} does not match observed "
            f"{realized.shape}"
        )
    out: dict[str, float] = {}
    for index, view in enumerate(RESIDUAL_VIEWS):
        error = float(np.linalg.norm(expected[index] - realized[index]))
        out[f"{view}_residual"] = error
        out[f"{view}_predicted_norm"] = float(np.linalg.norm(expected[index]))
        out[f"{view}_realized_norm"] = float(np.linalg.norm(realized[index]))
    return out


def observed_change(
    before_wrist: np.ndarray,
    before_column: np.ndarray,
    after_wrist: np.ndarray,
    after_column: np.ndarray,
) -> dict[str, float]:
    """How much each view actually changed. The residual's denominator.

    Kept separate from the residual so that a run where nothing moved is
    visible as such rather than as a model that predicted well.
    """

    pairs = (
        ("wrist", before_wrist, after_wrist),
        ("column", before_column, after_column),
    )
    out: dict[str, float] = {}
    for view, before, after in pairs:
        first = np.asarray(before, dtype=np.float32).reshape(-1)
        second = np.asarray(after, dtype=np.float32).reshape(-1)
        if first.shape != second.shape:
            raise ValueError(f"{view} contexts have different widths")
        out[f"{view}_observed_change"] = float(np.linalg.norm(second - first))
    return out


__all__ = [
    "RESIDUAL_VIEWS",
    "PredictedFuture",
    "context_residual",
    "observed_change",
    "predict_future_context",
]
