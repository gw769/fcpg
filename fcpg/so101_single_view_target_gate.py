"""Single-view target visibility gate for SO-101 realtime planning.

Generalizes the pickup visibility requirement from "both views visible" to
"column view visible OR wrist view visible with sufficient confidence."
Current implementation supports:

- BOTH_VISIBLE: use column (existing behavior, unchanged)
- COLUMN_ONLY: use column when wrist occluded (NEW, enables grasp continuation)
- WRIST_ONLY: not yet supported (fails closed, requires wrist-based xyz)
- NEITHER_VISIBLE: blocked (fail-closed, existing behavior)

Preserves per-view tracking state and fails closed when neither view provides
reliable information. The gate respects the existing reliability/confidence
thresholds from VisualTargetTracker (minimum_score, minimum_margin, continuity
support) and does NOT weaken any individual track's validity requirements.
"""

from __future__ import annotations

from typing import Any


def select_primary_target_view(
    column_descriptor: dict[str, Any],
    wrist_descriptor: dict[str, Any],
    *,
    target_kind: str,
) -> tuple[dict[str, Any] | None, str]:
    """Select the primary view descriptor for target-based planning.

    Returns the most reliable visible target descriptor and its source view.
    Falls back gracefully when only one view is visible, or fails closed when
    neither view provides sufficient information.

    Args:
        column_descriptor: track result from column view
        wrist_descriptor: track result from wrist view (may be empty for
            drop_zone/peg which are column-only)
        target_kind: "pickup", "drop_zone", or "peg"

    Returns:
        (selected_descriptor, view_name) when a reliable track exists, or
        (None, "neither") when both views are unreliable.

    The selection priority for pickup targets:
        1. BOTH_VISIBLE: prefer column (it provides the backprojection xyz)
        2. COLUMN_ONLY: use column if visible
        3. WRIST_ONLY: use wrist if visible (requires wrist-based xyz fallback)
        4. NEITHER_VISIBLE: return None

    For drop_zone and peg (column-only targets), wrist_descriptor is ignored.
    """

    column_visible = bool(column_descriptor.get("visible", False))
    wrist_visible = bool(wrist_descriptor.get("visible", False))

    # Column-only targets: drop_zone and peg never use wrist
    if target_kind in {"drop_zone", "peg"}:
        if column_visible:
            return column_descriptor, "column"
        return None, "neither"

    # Pickup target: prefer column, fall back to wrist
    if target_kind == "pickup":
        if column_visible and wrist_visible:
            # BOTH_VISIBLE: prefer column (provides backprojection)
            return column_descriptor, "column"
        elif column_visible:
            # COLUMN_ONLY: column provides direct backprojection to base xyz
            return column_descriptor, "column"
        elif wrist_visible:
            # WRIST_ONLY: wrist can provide relative correction but requires
            # a base xyz estimate. Current implementation does not support
            # wrist-only xyz computation, so we fail closed here.
            # TODO: Implement wrist-based xyz estimation using fingertip
            # similarity transform or FK-relative positioning.
            return None, "neither"
        else:
            # NEITHER_VISIBLE
            return None, "neither"

    # Unknown target kind: fail closed
    return None, "neither"


def should_block_on_target_visibility(
    target_descriptor: dict[str, Any] | None,
    target_xyz: Any,
    *,
    target_kind: str,
    primary_view: str,
) -> tuple[bool, str]:
    """Decide whether to block planning due to insufficient target visibility.

    Args:
        target_descriptor: selected primary view descriptor (may be None)
        target_xyz: backprojected 3D target position (may be None)
        target_kind: "pickup", "drop_zone", "peg", or None
        primary_view: "column", "wrist", or "neither"

    Returns:
        (should_block, reason) where reason is empty string when not blocking.

    Blocking conditions (fail-closed):
        - No target descriptor selected (neither view visible)
        - Target descriptor selected but xyz unavailable (backprojection failed)
    """

    if target_descriptor is None:
        # Neither view provided a reliable track
        if target_kind:
            return True, f"{target_kind}_not_visible"
        return True, "target_not_visible"

    if target_xyz is None:
        # Track exists but 3D position unavailable
        if target_kind:
            return True, f"{target_kind}_target_xyz_unavailable"
        return True, "target_xyz_unavailable"

    # All gates passed
    return False, ""


__all__ = [
    "select_primary_target_view",
    "should_block_on_target_visibility",
]
