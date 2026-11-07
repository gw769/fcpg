"""Rank/score must use the same visual evidence as LeWM dynamics.

V11.6 trains goal-free dual-view dynamics on wrist+column. That is the
live rank path. The V10 B2 ranking backup was trained on single-view
public caches and consumes one 192-d raw CLS; it is backup only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

LEWM_TRAIN_VIEWS = 2
RANK_VIEWS = 2
LEWM_DYNAMICS_VIEWS = ("wrist", "column")
LEWM_DYNAMICS_VIEW = "wrist+column"
RANK_VIEW = "wrist+column"
B2_BACKUP_VIEW = "wrist"
GOAL_IN_LEWM = False
WRIST_JAW_GROUNDING_DISTANCE_M = 0.13
WRIST_JAW_SOURCE = "wrist_jaw_fcpg"
WRIST_JAW_RADIUS_PX = 18.0


def rank_target_views(rank_view: str = RANK_VIEW) -> tuple[str, ...]:
    if rank_view in {"wrist+column", "column+wrist", "dual"}:
        return LEWM_DYNAMICS_VIEWS
    if rank_view in {"wrist", "column"}:
        return (rank_view,)
    raise ValueError(f"unknown rank view {rank_view!r}")


def assert_rank_matches_dynamics(
    rank_view: str = RANK_VIEW,
    dynamics_view: str = LEWM_DYNAMICS_VIEW,
) -> None:
    if int(RANK_VIEWS) != int(LEWM_TRAIN_VIEWS):
        raise ValueError(
            f"RANK_VIEWS={RANK_VIEWS} must match LEWM_TRAIN_VIEWS={LEWM_TRAIN_VIEWS}"
        )
    if rank_view != dynamics_view:
        raise ValueError(
            f"RANK_VIEW={rank_view!r} must match dynamics view {dynamics_view!r}"
        )
    if rank_target_views(rank_view) != rank_target_views(dynamics_view):
        raise ValueError("rank views must match the LeWM dynamics views")


def ranking_pixel_goal_weight(
    *,
    rank_view: str,
    pixel_view: str | None,
    configured_weight: float,
) -> float:
    """Pixel term may enter argmin only when it is a dynamics view."""

    if pixel_view is None or pixel_view not in rank_target_views(rank_view):
        return 0.0
    weight = float(configured_weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("configured pixel_goal_weight must be finite and >= 0")
    return weight


def pixel_rank_view_for_rollout(rollout_variant: str) -> str:
    """V11.6 scores both views. B2 backup stays wrist-only."""

    if rollout_variant in {"v11_6_dual_view", "v11_6_2view_goal_free"}:
        return RANK_VIEW
    return B2_BACKUP_VIEW


def rank_view_contract_fields(
    *,
    rank_pixel_view: str,
    rank_pixel_weight: float,
    wrist_fish_grounded: bool,
) -> dict[str, Any]:
    assert_rank_matches_dynamics()
    return {
        "lewm_train_views": LEWM_TRAIN_VIEWS,
        "rank_views": RANK_VIEWS,
        "lewm_dynamics_view": LEWM_DYNAMICS_VIEW,
        "rank_view": RANK_VIEW,
        "b2_backup_view": B2_BACKUP_VIEW,
        "rank_pixel_view": rank_pixel_view,
        "rank_pixel_goal_weight": float(rank_pixel_weight),
        "wrist_fish_grounded": bool(wrist_fish_grounded),
        "goal_in_lewm": bool(GOAL_IN_LEWM),
    }


def wrist_pickup_is_unconfigured(descriptor: dict[str, Any]) -> bool:
    if descriptor.get("configured"):
        return False
    return descriptor.get("reason") == "target_reference_not_configured"


def should_session_register_wrist_jaw(
    wrist_pickup: dict[str, Any],
    column_pickup: dict[str, Any],
    *,
    wrist_valid: bool,
    planar_error_m: float | None,
    distance_m: float = WRIST_JAW_GROUNDING_DISTANCE_M,
) -> bool:
    """Near-field wrist already sees the object; do not leave it unconfigured."""

    if not wrist_pickup_is_unconfigured(wrist_pickup):
        return False
    if not bool(column_pickup.get("visible", False)):
        return False
    if not wrist_valid:
        return False
    if planar_error_m is None or not np.isfinite(float(planar_error_m)):
        return False
    return float(planar_error_m) <= float(distance_m)


def wrist_jaw_pickup_descriptor(
    gripper_center_px: np.ndarray,
    *,
    source: str = WRIST_JAW_SOURCE,
) -> dict[str, Any]:
    """Session-live wrist pickup at the FCPG jaw center (fish between claws)."""

    gripper = np.asarray(gripper_center_px, dtype=np.float32)
    if gripper.shape != (2,) or not np.all(np.isfinite(gripper)):
        raise ValueError("gripper_center_px must be a finite two-vector")
    return {
        "visible": True,
        "configured": True,
        "role": "pickup",
        "view": "wrist",
        "centroid_px": gripper.copy(),
        "bbox_xyxy": None,
        "area_px": 0,
        "score": 1.0,
        "second_score": -1.0,
        "margin": 1.0,
        "source": source,
        "gripper_center_px": gripper.copy(),
        "relative_centroid_px": np.zeros(2, dtype=np.float32),
        "reason": None,
    }


def wrist_target_px(wrist_pickup: dict[str, Any]) -> np.ndarray | None:
    if not wrist_pickup.get("visible", False):
        return None
    raw = wrist_pickup.get("centroid_px")
    if raw is None:
        return None
    point = np.asarray(raw, dtype=np.float32)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        return None
    return point


__all__ = [
    "B2_BACKUP_VIEW",
    "GOAL_IN_LEWM",
    "LEWM_DYNAMICS_VIEW",
    "LEWM_DYNAMICS_VIEWS",
    "LEWM_TRAIN_VIEWS",
    "RANK_VIEW",
    "RANK_VIEWS",
    "WRIST_JAW_GROUNDING_DISTANCE_M",
    "WRIST_JAW_RADIUS_PX",
    "WRIST_JAW_SOURCE",
    "assert_rank_matches_dynamics",
    "pixel_rank_view_for_rollout",
    "rank_target_views",
    "rank_view_contract_fields",
    "ranking_pixel_goal_weight",
    "should_session_register_wrist_jaw",
    "wrist_jaw_pickup_descriptor",
    "wrist_pickup_is_unconfigured",
    "wrist_target_px",
]
