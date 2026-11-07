"""Display-only DINOv2 target localization for the Prod-V17.7 dashboard.

This module deliberately has no dependency on the V17.7 policy, runtime, or
hardware service.  It reads operator-registered appearance profiles through
``VisualTargetTracker`` and exposes frozen-feature locations as JSON-safe
diagnostics.  Callers must not use these records as control input unless they
mark the payload as such (fcpg/v21_live_service.py: the visual alignment phases
and the alignment CEM's wrist drop zone set ``control_input`` and name the
consumer).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from fcpg.so101_visual_targets import VisualTargetTracker


TARGET_SITES = (
    ("pickup", "wrist"),
    ("pickup", "column"),
    ("drop_zone", "column"),
)


def _not_visible(role: str, view: str, reason: str) -> dict[str, Any]:
    """Return the complete, stable schema for one failed localization."""

    return {
        "role": role,
        "view": view,
        "visible": False,
        "centroid_px": None,
        "axis_px": None,
        "width_px": None,
        "length_px": None,
        "score": -1.0,
        "separation": 0.0,
        "source": "frozen_dinov2_profile_prototype",
        "reason": str(reason),
    }


def _finite_vector(value: Any, *, length: int, name: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {length}-vector")
    return [float(item) for item in array]


def _visible_payload(role: str, view: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a tracker hit; keep overlay shape fields, drop the dense mask.

    ``support_mask`` is too large for status JSON and was previously stripped
    along with the contour — so column never drew the object silhouette.
    Precompute ``contour_px`` here and keep ``cog_px`` / ``grasp_axis_px``.
    """

    if not bool(result.get("visible", False)):
        rejected = _not_visible(role, view, str(result.get("reason") or "not_visible"))
        for key in ("score", "separation"):
            value = result.get(key)
            if value is not None and np.isfinite(float(value)):
                rejected[key] = float(value)
        return rejected
    centroid = _finite_vector(result.get("centroid_px"), length=2, name="centroid_px")
    axis = _finite_vector(result.get("axis_px"), length=2, name="axis_px")
    width = float(result.get("width_px"))
    length = float(result.get("length_px"))
    score = float(result.get("score"))
    separation = float(result.get("separation"))
    scalars = np.asarray([width, length, score, separation], dtype=np.float64)
    if not np.all(np.isfinite(scalars)) or width < 0.0 or length < 0.0:
        raise ValueError("target shape and scores must be finite with non-negative extents")
    cog = result.get("cog_px")
    if cog is not None:
        cog = _finite_vector(cog, length=2, name="cog_px")
    else:
        cog = list(centroid)
    grasp_axis = result.get("grasp_axis_px")
    if grasp_axis is not None:
        grasp_axis = _finite_vector(grasp_axis, length=2, name="grasp_axis_px")
    else:
        grasp_axis = [float(-axis[1]), float(axis[0])]
    contour = result.get("contour_px")
    if not contour:
        mask = result.get("support_mask")
        if mask is not None:
            from fcpg.so101_frozen_vit_grounding import support_contour_px
            contour = support_contour_px(mask)
    if contour:
        contour = [[float(p[0]), float(p[1])] for p in contour]
    else:
        contour = []
    return {
        "role": role,
        "view": view,
        "visible": True,
        "centroid_px": centroid,
        "cog_px": cog,
        "axis_px": axis,
        "grasp_axis_px": grasp_axis,
        "width_px": width,
        "length_px": length,
        "score": score,
        "separation": separation,
        # Contour is taken from this view's support at locate time — not from
        # the other camera. Dense mask stays out of the JSON status payload.
        "contour_px": contour,
        "symmetry_iou": float(result.get("symmetry_iou") or 0.0),
        "support_clipped": bool(result.get("support_clipped")),
        "aimable": bool(result.get("aimable", True)),
        "source": str(result.get("source") or "frozen_dinov2_profile_prototype"),
        "reason": None,
    }


def locate_dino_targets(
    frames: Mapping[str, np.ndarray],
    profile_root: str | Path | None = None,
    *,
    tracker: Any | None = None,
    tracker_factory: Callable[..., Any] = VisualTargetTracker,
    jaw_pairs_px: Mapping[str, Any] | None = None,
    sites: tuple[tuple[str, str], ...] = TARGET_SITES,
) -> dict[str, Any]:
    """Locate the registered pickup and drop region with frozen DINOv2 features.

    ``tracker`` is injectable so schema, routing, and failure behavior can be
    tested without loading a backbone.  When omitted, ``profile_root`` is
    passed to ``VisualTargetTracker``, which loads the existing persisted
    registration crops.  Every target is isolated: profile, frame, backbone,
    or localization failures become ``visible=false`` records and never escape
    into the control path.

    ``jaw_pairs_px`` maps view name → ordered [fixed, moving] tip pixels so
    claw occlusion does not join the object support.  ``sites`` are the
    (role, view) pairs located (fcpg/v21_target_tracker.V21_TARGET_SITES adds
    the drop zone in the wrist).
    """

    load_error: str | None = None
    active_tracker = tracker
    if active_tracker is None:
        try:
            if profile_root is None:
                raise ValueError("profile_root is required when tracker is not injected")
            active_tracker = tracker_factory(Path(profile_root).expanduser().resolve())
        except Exception as exc:  # diagnostic path must fail closed
            load_error = f"profile_load_failed: {type(exc).__name__}: {exc}"

    targets: dict[str, dict[str, dict[str, Any]]] = {
        "pickup": {},
        "drop_zone": {},
    }
    pairs = {} if jaw_pairs_px is None else dict(jaw_pairs_px)
    for role, view in sites:
        if load_error is not None:
            record = _not_visible(role, view, load_error)
        else:
            try:
                frame = frames[view]
                locate_kw: dict[str, Any] = {}
                jaw = pairs.get(view)
                if jaw is not None:
                    locate_kw["jaw_pair_px"] = jaw
                result = active_tracker.locate_by_frozen_features(
                    role, view, frame, **locate_kw
                )
                if not isinstance(result, Mapping):
                    raise TypeError("tracker result must be a mapping")
                record = _visible_payload(role, view, result)
            except Exception as exc:  # one failed view must not hide the others
                record = _not_visible(
                    role,
                    view,
                    f"localization_failed: {type(exc).__name__}: {exc}",
                )
        targets[role][view] = record

    return {
        "version": "17.7.0",
        "source": "frozen_dinov2_profile_prototype",
        "display_only": True,
        "control_input": False,
        "targets": targets,
    }


__all__ = ["TARGET_SITES", "locate_dino_targets"]
