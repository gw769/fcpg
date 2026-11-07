"""Single-view target visibility logic for SO-101 planner (Agent C).

Enables planning to proceed when only one view has a reliable target detection,
while preserving fail-closed behavior for truly unavailable targets.

Key principles:
- Column view is required for 3D target computation (backprojection)
- Wrist view is optional (used for attachment confirmation and correction)
- Geometry hold ("held_gripper_occlusion") is valid for column, never for wrist
- Per-view reliability based on match quality (margin, source)
- DO NOT fabricate targets or bypass operator registration gates
"""

from __future__ import annotations

from typing import Any

#: Hold frames after a full-margin appearance that still count as "recent"
#: for TRANSPORT→RELEASE. Brief occlusion flicker over the mouth is ok;
#: a long held rim fix is not.
RECENT_APPEARANCE_HOLD_FRAMES = 2

# How much the frozen backbone has to actually agree with the prototype before
# the object counts as seen. This has to be an absolute similarity, because how
# far the peak stands above the rest of the frame does not distinguish a match
# from no match at all: measured on this rig, four live frames holding the fish
# gave 4.1-5.7 background deviations, and a flat grey frame gave 3.8, pure
# noise 4.0, and the synthetic preflight canvas 4.0-4.2. Every similarity map
# has a maximum somewhere, and that statistic only says the map has a peak.
# The peak's own height does separate them, with no overlap: 0.55-0.59 with the
# fish against 0.15-0.24 without it, so the floor sits in the gap between.
FROZEN_FEATURE_MIN_PEAK = 0.35


def gripper_inside_bbox(
    gripper_center_px: Any,
    bbox_xyxy: Any,
) -> bool:
    """True when the gripper pixel lies inside an axis-aligned bbox."""

    if gripper_center_px is None or bbox_xyxy is None:
        return False
    try:
        gx, gy = (float(gripper_center_px[0]), float(gripper_center_px[1]))
        x0, y0, x1, y1 = (float(v) for v in bbox_xyxy[:4])
    except (TypeError, ValueError, IndexError):
        return False
    return x0 <= gx <= x1 and y0 <= gy <= y1


def compute_drop_phase_gate_reliable(
    drop_zone: dict[str, Any],
    *,
    gripper_center_px: Any = None,
    min_margin: float = 0.020,
) -> bool:
    """Whether drop evidence is reliable enough to enter RELEASE / open jaws.

    Appearance (full margin) and operator registration are reliable. Geometry
    hold (``held_gripper_occlusion``) is refused unless a recent full-margin
    appearance refresh is still in budget or the gripper pixel sits inside the
    drop bbox. Callers stamp the result onto the descriptor as
    ``phase_gate_reliable`` so the MPC checks a bool, not source strings.
    """

    if not drop_zone.get("visible", False):
        return False
    if not drop_zone.get("configured", False):
        return False
    source = drop_zone.get("source")
    if source == "operator_reference":
        return True
    if source in {"frozen_vit", "open_vocab", "predicted", "reprojection"}:
        return True
    if source == "frozen_vit_features":
        peak = _optional_float(drop_zone, "score")
        return peak is not None and peak >= FROZEN_FEATURE_MIN_PEAK
    if source == "appearance_template":
        margin = float(drop_zone.get("margin", 0.0))
        return margin >= float(min_margin)
    if source == "held_gripper_occlusion":
        if bool(drop_zone.get("recent_appearance_template", False)):
            return True
        return gripper_inside_bbox(
            gripper_center_px, drop_zone.get("bbox_xyxy")
        )
    return False


def annotate_drop_phase_gate(
    drop_zone: dict[str, Any],
    *,
    gripper_center_px: Any = None,
    min_margin: float = 0.020,
) -> dict[str, Any]:
    """Return a copy of ``drop_zone`` stamped with ``phase_gate_reliable``."""

    reliable = compute_drop_phase_gate_reliable(
        drop_zone,
        gripper_center_px=gripper_center_px,
        min_margin=min_margin,
    )
    return {**drop_zone, "phase_gate_reliable": bool(reliable)}


def assess_target_reliability(
    target_descriptor: dict[str, Any],
    *,
    view: str,
    min_margin: float = 0.020,
    allow_held: bool = True,
) -> bool:
    """Assess whether a target detection is reliable for planning.

    Args:
        target_descriptor: Result from VisualTargetTracker.track()
        view: "column" or "wrist"
        min_margin: Minimum template match margin (best - second)
        allow_held: Whether to accept "held_gripper_occlusion" as reliable

    Returns:
        True if target is reliable for planning, False otherwise

    Reliability criteria:
    1. Target must be visible and configured
    2. Source must be "appearance_template", "operator_reference", or
       "held_gripper_occlusion" (column only, if allow_held=True)
    3. For appearance_template: margin must meet threshold
    4. For held_gripper_occlusion: only valid for column view
    5. For operator_reference: always reliable (initial registration)
    """
    if view not in ("column", "wrist"):
        raise ValueError(f"view must be 'column' or 'wrist', got {view!r}")

    # Not visible or not configured → unreliable
    if not target_descriptor.get("visible", False):
        return False
    if not target_descriptor.get("configured", False):
        return False

    source = target_descriptor.get("source")

    # Operator reference is always reliable (initial registration)
    if source == "operator_reference":
        return True

    # Geometry hold: only reliable for column view when allowed
    if source == "held_gripper_occlusion":
        # Wrist geometry hold would fabricate attachment evidence
        if view == "wrist":
            return False
        return allow_held

    # Appearance template: check margin
    if source == "appearance_template":
        margin = float(target_descriptor.get("margin", 0.0))
        return margin >= min_margin

    # Frozen features: a dense similarity field has no runner-up box to
    # difference against, so margin is not the question to ask of it. What it
    # has instead is how strongly the backbone agreed with the prototype, and
    # that has to be read as an absolute similarity: how far the peak stands
    # above the rest of the frame says only that the map has a peak, which
    # every map does. Measured on this rig, a flat grey frame stood 3.8
    # background deviations above its own background and pure noise 4.0, against
    # 4.1-5.7 for frames actually holding the fish -- indistinguishable. The
    # peak itself separated them cleanly, 0.55-0.59 with the fish against
    # 0.15-0.24 without, so that is what the bar is set on. Both numbers are
    # published on every descriptor so the grounding layer can weigh them
    # against everything else rather than inherit a verdict.
    if source == "frozen_vit_features":
        peak = _optional_float(target_descriptor, "score")
        if peak is None:
            return False
        return peak >= FROZEN_FEATURE_MIN_PEAK
    if source in {"frozen_vit", "open_vocab"}:
        return True
    if source == "predicted":
        return bool(target_descriptor.get("centroid_px") is not None)
    if source == "reprojection":
        return True

    # Unknown or missing source → unreliable
    return False


def compute_pickup_visible_with_fallback(
    column_pickup: dict[str, Any],
    wrist_pickup: dict[str, Any],
    *,
    min_margin: float = 0.020,
    allow_held: bool = True,
) -> dict[str, Any]:
    """Compute pickup_visible with single-view fallback support.

    Replaces the simple AND logic with reliability-gated fallback:
    - BOTH_VISIBLE: both views reliable (existing behavior)
    - COLUMN_ONLY: column reliable, wrist unavailable or unreliable
    - BLOCKED: column unreliable (cannot compute 3D target)

    Args:
        column_pickup: Result from tracker.track("pickup", "column", ...)
        wrist_pickup: Result from tracker.track("pickup", "wrist", ...)
        min_margin: Minimum margin for reliability
        allow_held: Whether to accept geometry hold as reliable

    Returns:
        Dictionary with:
        - pickup_visible: bool (for PhaseEvidence)
        - mode: str (diagnostic: "both_views", "column_only", "blocked_*")
        - column_reliable: bool
        - wrist_reliable: bool
        - column_source: str | None
        - wrist_source: str | None
        - column_margin: float | None
        - wrist_margin: float | None
        - column_reliability_reason: str | None
        - wrist_reliability_reason: str | None
    """
    column_reliable = assess_target_reliability(
        column_pickup, view="column", min_margin=min_margin, allow_held=allow_held
    )
    wrist_reliable = assess_target_reliability(
        wrist_pickup, view="wrist", min_margin=min_margin, allow_held=False
    )

    # Determine mode
    if column_reliable and wrist_reliable:
        mode = "both_views"
        pickup_visible = True
    elif column_reliable and not wrist_reliable:
        mode = "column_only"
        pickup_visible = True
    elif not column_reliable and wrist_reliable:
        mode = "blocked_no_reliable_column"
        pickup_visible = False
    elif not column_pickup.get("visible", False) and not wrist_pickup.get(
        "visible", False
    ):
        mode = "blocked_neither_visible"
        pickup_visible = False
    else:
        mode = "blocked_no_reliable_column"
        pickup_visible = False

    # Compute reliability reasons for diagnostics
    column_reason = _reliability_reason(column_pickup, view="column", min_margin=min_margin, allow_held=allow_held)
    wrist_reason = _reliability_reason(wrist_pickup, view="wrist", min_margin=min_margin, allow_held=False)

    return {
        "pickup_visible": pickup_visible,
        "mode": mode,
        "column_reliable": column_reliable,
        "wrist_reliable": wrist_reliable,
        "column_source": column_pickup.get("source"),
        "wrist_source": wrist_pickup.get("source"),
        # A detector with no runner-up box reports no margin, and the key is
        # present with a None in it rather than absent. Testing for the key
        # alone read that as a number and crashed the planning round.
        "column_margin": _optional_float(column_pickup, "margin"),
        "wrist_margin": _optional_float(wrist_pickup, "margin"),
        "column_separation": _optional_float(column_pickup, "separation"),
        "wrist_separation": _optional_float(wrist_pickup, "separation"),
        "column_reliability_reason": column_reason,
        "wrist_reliability_reason": wrist_reason,
    }


def _optional_float(descriptor: dict[str, Any], key: str) -> float | None:
    if not descriptor.get("visible"):
        return None
    value = descriptor.get(key)
    return None if value is None else float(value)


def _reliability_reason(
    target_descriptor: dict[str, Any],
    *,
    view: str,
    min_margin: float,
    allow_held: bool,
) -> str | None:
    """Compute diagnostic reason for reliability assessment."""
    if not target_descriptor.get("visible", False):
        return "not_visible"
    if not target_descriptor.get("configured", False):
        return "not_configured"

    source = target_descriptor.get("source")

    if source == "operator_reference":
        return None  # Reliable

    if source == "held_gripper_occlusion":
        if view == "wrist":
            return "wrist_hold_invalid"
        if not allow_held:
            return "hold_not_allowed"
        return None  # Reliable

    if source == "appearance_template":
        margin = float(target_descriptor.get("margin", 0.0))
        if margin < min_margin:
            return "margin_below_threshold"
        return None  # Reliable

    if source == "frozen_vit_features":
        peak = _optional_float(target_descriptor, "score")
        if peak is None:
            return "peak_unavailable"
        if peak < FROZEN_FEATURE_MIN_PEAK:
            return "peak_below_threshold"
        return None  # Reliable
    if source in {"frozen_vit", "open_vocab", "predicted", "reprojection"}:
        return None

    return "unknown_source"


__all__ = [
    "FROZEN_FEATURE_MIN_PEAK",
    "RECENT_APPEARANCE_HOLD_FRAMES",
    "annotate_drop_phase_gate",
    "assess_target_reliability",
    "compute_drop_phase_gate_reliable",
    "compute_pickup_visible_with_fallback",
    "gripper_inside_bbox",
]
