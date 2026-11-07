"""Hard split: actor may be goal-sensitive; LeWM may not.

LeWM dynamics is frozen GOAL-FREE:

    (state / history / candidate action) -> future state / residual / consequence

LeWM supervision is ONLY the existing future/state regression plus the frozen
SIGReg contract (weight 0.02 on dynamics_latent).

Structured goal, goal-sensitive losses, and policy action-delta causal losses
belong to the V11 actor. They must not enter LeWM.

MuJoCo paired rollouts may be shared. Supervision stays segregated:

    ACTOR: (state, goal_A) -> action_A ; (state, goal_B) -> action_B
           BC + optional causal direction/delta

    LEWM:  (state, action_A) -> future_A ; (state, action_B) -> future_B
           future/state regression + SIGReg only
"""

from __future__ import annotations

from collections.abc import Mapping

from v10.config import SIGREG_WEIGHT

ACTOR_CAUSAL_PART_KEYS = frozenset(
    {
        "supervised_counterfactual",
        "supervised_counterfactual_direction",
        "supervised_counterfactual_magnitude",
        "supervised_counterfactual_n_pairs",
        "supervised_counterfactual_cosine",
        "supervised_counterfactual_mag_mae_deg",
        "context_swap_hinge",
        "context_swap_mag_hinge",
        "context_swap_dir_hinge",
        "context_swap_flail_tax",
        "g1_swap_effect_deg",
        "g1_swap_cosine",
    }
)

LEWM_REGRESSION_PART_KEYS = frozenset(
    {
        "state_loss",
        "context_loss",
        "anchor_loss",
        "pixel_loss",
    }
)

FORBIDDEN_LEWM_SUPERVISION = frozenset(
    {
        "goal_delta",
        "goal_sensitive",
        "action_delta_causal",
        "pred_delta",
        "gt_delta",
        "structured_goal_loss",
    }
)


def lewm_actor_supervision_contract() -> dict[str, object]:
    """Record the frozen split on every run. Not a V10Config field."""

    return {
        "lewm_dynamics": "goal_free",
        "lewm_supervision": "future_state_regression_and_sigreg_only",
        "lewm_sigreg_weight": float(SIGREG_WEIGHT),
        "structured_goal_enters_lewm": False,
        "goal_sensitive_loss_enters_lewm": False,
        "policy_action_delta_causal_loss_enters_lewm": False,
        "actor_may_use_causal_pair_loss": True,
        "mujoco_pairs_may_be_shared_supervision_must_not": True,
    }


def assert_lewm_parts_are_goal_free(parts: Mapping[str, object]) -> None:
    """LeWM regression dict must not carry actor causal or goal-delta keys."""

    leaked = FORBIDDEN_LEWM_SUPERVISION.intersection(parts)
    if leaked:
        raise ValueError(f"goal-sensitive keys leaked into LeWM parts: {sorted(leaked)}")
    causal = ACTOR_CAUSAL_PART_KEYS.intersection(parts)
    if causal:
        raise ValueError(
            "actor causal keys must not be passed as LeWM supervision: "
            f"{sorted(causal)}"
        )


def assert_sigreg_is_lewm_only(
    *,
    sigreg_input_name: str,
    added_to: str,
) -> None:
    if sigreg_input_name != "dynamics_latent":
        raise ValueError(
            "SIGReg may regularize dynamics_latent only, "
            f"got {sigreg_input_name!r}"
        )
    if added_to != "total_with_lewm_regression":
        raise ValueError("SIGReg must stay on the LeWM side of the split")


def actor_action_only(output) -> object:
    """Read the actor head. Do not use this call's future/dynamics as LeWM loss."""

    return output.action
