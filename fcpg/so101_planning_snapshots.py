"""Render operator-facing planning snapshots for the live SO-101 debug API.

The images are drawn on the 224 planning canvas the planner already scored.
This module does not letterbox, squash, or otherwise re-project a camera
frame: ``column_raw.jpg`` in a round directory is already that canvas.

Column operator marks are FCPG-corrected anchors after the one-shot pulse
has restored support. Metal, OWL gripper boxes, and the recursive observer
are diagnostics only. When the live-anchor transform is not applied the
nominal projection is out of support: FK is a thin X, not a claimed tip.
Marks are geometric: no C / L / R glyphs and no pickup caption on the
workspace.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fcpg.letterbox_geometry import PLANNING_IMAGE_SIZE
from fcpg.so101_column_extrinsic_observer import (
    apply_column_frame_offset_px,
)
from fcpg.so101_column_tip_measurement import (
    column_metal_operator_tips,
    is_patch_pitch_triple,
)
from fcpg.so101_realtime_mpc import detected_tip_separation_px


VIEWS = ("wrist", "column")
ANCHOR_NAMES = ("C", "L", "R")
SNAPSHOT_JPEG_QUALITY = 93

# Same C / L / R language as the existing column target-path overlay.
OPERATOR_COLORS = ((255, 255, 255), (46, 211, 255), (255, 92, 185))
RAW_MEASUREMENT_COLORS = ((160, 160, 160), (120, 170, 190), (180, 120, 150))
PRED_COLORS = ((22, 163, 74), (46, 211, 255), (255, 92, 185))
IK_COLORS = ((37, 99, 235), (85, 135, 235), (30, 68, 170))
LOCATOR_COLORS = ((255, 180, 40), (255, 140, 20), (220, 90, 20))
FK_COLOR = (220, 220, 220)
PICKUP_COLOR = (255, 210, 36)
WARNING_FILL = (176, 28, 28)
WARNING_TEXT = (255, 245, 245)
BANNER_FILL = (20, 20, 20)

COLUMN_UNANCHORED_STAMP = "COLUMN FK UNANCHORED"
MARKS_NOT_ON_TIPS_STAMP = "MARKS ARE NOT ON TIPS"
WRIST_PICKUP_MISSING_STAMP = "WRIST PICKUP MISSING"
PROD_V13_NO_FUTURE_PIXEL_STAMP = "SEED13161 HAS NO FUTURE PIXEL HEAD"
FCPGH_AUTO_H0_PENDING_STAMP = "FCPG-H AUTO H0 PENDING"

# Operator marks must read at 224 px without letter labels.
TIP_MARK_RADIUS = 6


def load_snapshot_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def planning_frame(frame: np.ndarray) -> np.ndarray:
    """Return the last 224 planning RGB frame. Refuse any other canvas."""

    values = np.asarray(frame)
    if values.ndim == 4:
        if values.shape[0] < 1:
            raise ValueError("planning frames are empty")
        values = values[-1]
    if values.shape != (PLANNING_IMAGE_SIZE, PLANNING_IMAGE_SIZE, 3):
        raise ValueError(
            "planning snapshot must use the 224 planning frame; "
            f"got shape {tuple(values.shape)}"
        )
    return np.ascontiguousarray(values, dtype=np.uint8)


def last_anchors(values: Any) -> np.ndarray:
    points = np.asarray(values, dtype=np.float32)
    if points.ndim == 3:
        if points.shape[0] < 1:
            raise ValueError("anchor history is empty")
        points = points[-1]
    if points.shape != (3, 2) or not np.all(np.isfinite(points)):
        raise ValueError("anchors must be a finite [3,2] pixel triple")
    return points.astype(np.float32, copy=False)


def _as_optional_anchors(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    points = np.asarray(values, dtype=np.float32)
    if points.size == 0:
        return None
    if points.ndim == 3:
        points = points[-1]
    if points.shape != (3, 2) or not np.all(np.isfinite(points)):
        return None
    return points.astype(np.float32, copy=False)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def prod_v13_observed_tip_trajectory(
    plan: Any, view: str
) -> dict[str, Any] | None:
    """Read raw query/tracker/posterior pixels without changing coordinates."""

    payload = _mapping(getattr(plan, "prod_v13_tip_trajectory", None))
    if not payload.get("enabled"):
        return None
    view_index = VIEWS.index(view)
    display_rows: list[np.ndarray] = []
    measurement_rows: list[np.ndarray] = []
    posterior_rows: list[np.ndarray] = []
    changed_rows: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for raw in payload.get("rows") or []:
        row = _mapping(raw)
        measurement = np.asarray(row.get("measurement_tip_px"), dtype=np.float32)
        visible = np.asarray(row.get("measurement_visible"), dtype=bool)
        if measurement.shape != (2, 2, 2) or visible.shape != (2, 2):
            continue
        tips = measurement[view_index].copy()
        tip_visible = visible[view_index]
        tips[~tip_visible] = np.nan
        measured_triple = np.concatenate((tips.mean(axis=0, keepdims=True), tips))
        measurement_rows.append(measured_triple)
        posterior = row.get("posterior_tip_px")
        posterior_triple = np.full((3, 2), np.nan, dtype=np.float32)
        if posterior is not None:
            values = np.asarray(posterior, dtype=np.float32)
            if values.shape == (2, 2, 2):
                selected = values[view_index].copy()
                selected[~tip_visible] = np.nan
                posterior_triple = np.concatenate(
                    (selected.mean(axis=0, keepdims=True), selected)
                )
        posterior_rows.append(posterior_triple)
        display_rows.append(
            posterior_triple
            if np.all(np.isfinite(posterior_triple))
            else measured_triple
        )
        changed = row.get("prior_changed_selection")
        changed_values = np.zeros(2, dtype=bool)
        if changed is not None:
            values = np.asarray(changed, dtype=bool)
            if values.shape == (2, 2):
                changed_values = values[view_index]
        changed_rows.append(changed_values)
        metadata.append(
            {
                "kind": row.get("kind"),
                "rollout_index": row.get("rollout_index"),
                "horizon": row.get("horizon"),
            }
        )
    return {
        "enabled": True,
        "initialized": bool(payload.get("initialized")),
        "checkpoint_seed": payload.get("checkpoint_seed"),
        "checkpoint_step": payload.get("checkpoint_step"),
        "pixel_prediction_contract": payload.get("pixel_prediction_contract"),
        "display_px": np.stack(display_rows) if display_rows else None,
        "measurement_px": np.stack(measurement_rows) if measurement_rows else None,
        "posterior_px": np.stack(posterior_rows) if posterior_rows else None,
        "prior_changed_selection": (
            np.stack(changed_rows) if changed_rows else None
        ),
        "rows": metadata,
    }


def grounding_row(plan: Any, view: str) -> dict[str, Any]:
    diagnostics = _mapping(getattr(plan, "grounding_diagnostics", None))
    row = _mapping(diagnostics.get(view))
    summary = _mapping(getattr(plan, "grounding", None)).get(view)
    if isinstance(summary, Mapping):
        merged = dict(summary)
        merged.update(row)
        return merged
    return row


def live_anchor_status(plan: Any, view: str) -> dict[str, Any]:
    row = grounding_row(plan, view)
    transform = row.get("live_anchor_transform")
    if not isinstance(transform, Mapping):
        return {"present": False, "applied": False}
    status = {
        "present": bool(transform.get("present", False)),
        "applied": bool(transform.get("applied", False)),
    }
    for key in ("distance_m", "radius_m", "residual_rms_px", "hysteresis_band"):
        if key in transform:
            status[key] = transform[key]
    return status


def locator_points(plan: Any, view: str) -> np.ndarray | None:
    row = grounding_row(plan, view)
    locator = row.get("global_locator")
    if isinstance(locator, Mapping):
        points = _as_optional_anchors(locator.get("points_px"))
        if points is not None:
            return points
    return _as_optional_anchors(row.get("global_locator_points_px"))


def observer_tip_projection(plan: Any, view: str) -> tuple[np.ndarray, bool] | None:
    """Shadow / null-model column projection. Not operator tip authority."""

    if view != "column":
        return None
    row = grounding_row(plan, view)
    observer = row.get("recursive_observer")
    if not isinstance(observer, Mapping):
        return None
    points = _as_optional_anchors(observer.get("projected_tips_px"))
    if points is None:
        return None
    accepted = bool(observer.get("accepted"))
    # An unconverged prior can project off the letterbox canvas. Motion-tip
    # count is a measurement diagnostic, not an accept. Wait for the update.
    if not accepted:
        return None
    in_content = (
        np.all(np.isfinite(points))
        and np.all(points[:, 0] >= 0.0)
        and np.all(points[:, 0] < float(PLANNING_IMAGE_SIZE))
        and np.all(points[:, 1] >= 28.0)
        and np.all(points[:, 1] < 196.0)
    )
    if not in_content:
        return None
    return points, True


def locator_accepted(plan: Any, view: str) -> bool:
    row = grounding_row(plan, view)
    locator = row.get("global_locator")
    if isinstance(locator, Mapping) and "accepted" in locator:
        return bool(locator.get("accepted"))
    return bool(row.get("global_locator_accepted", False))


def pickup_descriptor(plan: Any, view: str) -> dict[str, Any]:
    descriptors = _mapping(getattr(plan, "target_descriptors", None))
    if view == "wrist":
        return _mapping(descriptors.get("pickup_wrist"))
    target = _mapping(getattr(plan, "target", None))
    if target.get("centroid_px") is not None:
        return target
    return _mapping(descriptors.get("pickup_column"))


def pickup_visible(plan: Any, view: str) -> bool:
    descriptor = pickup_descriptor(plan, view)
    return bool(
        descriptor.get("visible") and descriptor.get("centroid_px") is not None
    )


def pickup_radius_px(descriptor: Mapping[str, Any]) -> float:
    radius = descriptor.get("radius_px")
    if radius is not None and np.isfinite(float(radius)):
        return float(radius)
    bbox = descriptor.get("bbox_xyxy")
    if bbox is not None:
        x0, y0, x1, y1 = (float(value) for value in bbox)
        return 0.5 * max(abs(x1 - x0), abs(y1 - y0), 8.0)
    length = descriptor.get("length_px")
    width = descriptor.get("width_px")
    if length is not None and width is not None:
        return 0.5 * max(float(length), float(width), 8.0)
    return 8.0


def plan_anchor_paths(
    plan: Any, view: str
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Predicted and commanded paths in the same geometry as the round overlay."""

    paths = _mapping(getattr(plan, "anchor_paths_px", None)).get(view) or {}
    paths = _mapping(paths)
    predicted = paths.get("predicted")
    commanded = paths.get("commanded")
    if view == "column":
        if commanded is None:
            commanded = getattr(plan, "commanded_anchor_px", None)
        if predicted is None:
            predicted = getattr(plan, "predicted_anchor_px", None)
        if commanded is None and getattr(plan, "commanded_gripper_px", None) is not None:
            commanded = np.asarray(plan.commanded_gripper_px)[:, None]
        if predicted is None and getattr(plan, "predicted_gripper_px", None) is not None:
            predicted = np.asarray(plan.predicted_gripper_px)[:, None]
    return _normalize_path(predicted), _normalize_path(commanded)


def selected_candidate_row(plan: Any) -> dict[str, Any] | None:
    index = getattr(plan, "selected_index", None)
    rows = getattr(plan, "candidate_rows", None) or []
    if index is None or not rows:
        return None
    try:
        index = int(index)
    except (TypeError, ValueError):
        return None
    if not 0 <= index < len(rows):
        return None
    row = rows[index]
    return dict(row) if isinstance(row, Mapping) else None


def column_commanded_is_losing_arc(plan: Any) -> bool:
    """Losing ``arc_*`` IK must not be drawn on the column snapshot."""

    row = selected_candidate_row(plan)
    if row is None:
        return False
    source = str(row.get("source") or "")
    variant = str(row.get("variant") or "")
    return source == "calibrated_cartesian_ik" and variant.startswith("arc_")


def shift_column_paths(
    paths: np.ndarray | None, offset_px: np.ndarray | None
) -> np.ndarray | None:
    """Translate a column forecast by a hard-match / metal residual."""

    if paths is None or offset_px is None:
        return paths
    return apply_column_frame_offset_px(paths, offset_px)


def align_column_paths_to_operator(
    paths: np.ndarray | None,
    *,
    operator_px: np.ndarray,
    fk_points_px: np.ndarray | None,
) -> np.ndarray | None:
    """Shift a raw-FK forecast onto the metal marks when it still sits on FK.

    Paths already projected through the applied column extrinsic are left
    alone. A leftover pulse/FK curve is translated by the same metal-minus-FK
    residual the operator marks use. Mean error can hide a 12 px R miss after
    a camera-global shift, so the decision and the offset are per-anchor.
    """

    if paths is None or fk_points_px is None:
        return paths
    values = np.asarray(paths, dtype=np.float32)
    operator = np.asarray(operator_px, dtype=np.float32).reshape(3, 2)
    fk = np.asarray(fk_points_px, dtype=np.float32).reshape(3, 2)
    if values.ndim != 3 or values.shape[-1] != 2:
        return values
    start = values[0]
    if start.shape[-1] != 2:
        return values
    if start.ndim == 1:
        start = start.reshape(1, 2)
        metal = np.mean(operator, axis=0, keepdims=True)
        fk_start = np.mean(fk, axis=0, keepdims=True)
        offset = (np.mean(operator, axis=0) - np.mean(fk, axis=0)).astype(np.float32)
    elif start.shape[0] == 3:
        metal = operator
        fk_start = fk
        offset = (operator - fk).astype(np.float32)
    else:
        metal = np.mean(operator, axis=0, keepdims=True)
        fk_start = np.mean(fk, axis=0, keepdims=True)
        start = np.mean(start, axis=0, keepdims=True)
        offset = (np.mean(operator, axis=0) - np.mean(fk, axis=0)).astype(np.float32)
    to_metal = np.linalg.norm(start - metal, axis=-1)
    to_fk = np.linalg.norm(start - fk_start, axis=-1)
    if np.all(to_metal <= 3.0) or np.all(to_fk >= to_metal):
        return values
    return apply_column_frame_offset_px(values, offset)


def _normalize_path(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    paths = np.asarray(values, dtype=np.float32)
    if paths.size == 0:
        return None
    if paths.ndim == 2 and paths.shape[-1] == 2:
        paths = paths[:, None, :]
    if paths.ndim != 3 or paths.shape[-1] != 2:
        raise ValueError("anchor paths must have shape [H,K,2]")
    if paths.shape[1] not in (1, 3):
        raise ValueError("anchor paths must contain one or three anchors")
    return paths


def planner_tip_anchors(plan: Any, view: str) -> tuple[np.ndarray, str]:
    """Anchors the planner is currently using (corrected if applied, else nominal)."""

    nominal = last_anchors(_require_view_points(plan, "nominal", view))
    corrected = last_anchors(_require_view_points(plan, "corrected", view))
    live = live_anchor_status(plan, view)
    if live["applied"] or locator_accepted(plan, view):
        return corrected, "corrected"
    return nominal, "nominal"


def _require_view_points(plan: Any, field: str, view: str) -> Any:
    mapping = getattr(plan, field, None)
    if not isinstance(mapping, Mapping) or view not in mapping:
        raise ValueError(f"plan is missing {field}[{view}]")
    return mapping[view]


def _plan_frame(plan: Any, view: str) -> np.ndarray | None:
    frames = getattr(plan, "frames", None)
    if not isinstance(frames, Mapping) or view not in frames:
        return None
    try:
        return planning_frame(frames[view])
    except ValueError:
        return None


def resolve_operator_tips(
    plan: Any, view: str, frame: np.ndarray | None = None
) -> dict[str, Any]:
    """Show current FCPG-H measurements, never a relabelled V12 prediction."""

    planner, planner_source = planner_tip_anchors(plan, view)
    prod_v13 = prod_v13_observed_tip_trajectory(plan, view)
    if prod_v13 is not None and prod_v13.get("display_px") is not None:
        rows = prod_v13.get("rows") or []
        latest_kind = rows[-1].get("kind") if rows else None
        if latest_kind == "automatic_fcpgh_query":
            source = "fcpgh_active_h0"
        elif latest_kind == "operator_query":
            source = "seed13161_operator_query"
        else:
            source = "seed13161_raw_posterior"
        return {
            "points_px": np.asarray(prod_v13["display_px"][-1], dtype=np.float32),
            "source": source,
            "on_tips": True,
            "planner_points_px": planner,
            "planner_source": planner_source,
            "locator_points_px": locator_points(plan, view),
            "locator_accepted": locator_accepted(plan, view),
            "unanchored": False,
            "live_anchor": live_anchor_status(plan, view),
            "prod_v13_trajectory": prod_v13,
        }
    if prod_v13 is not None:
        # The latest observer is attached but has not completed its automatic
        # H0 pulse.  The legacy V12/CAD anchors are not FCPG-H measurements,
        # so display no fingertip mark until real current-image support exists.
        return {
            "points_px": planner,
            "source": "fcpgh_auto_h0_pending",
            "on_tips": False,
            "planner_points_px": planner,
            "planner_source": planner_source,
            "locator_points_px": locator_points(plan, view),
            "locator_accepted": locator_accepted(plan, view),
            "unanchored": False,
            "live_anchor": live_anchor_status(plan, view),
            "prod_v13_trajectory": prod_v13,
        }
    live = live_anchor_status(plan, view)
    locator = locator_points(plan, view)
    accepted = locator_accepted(plan, view)
    unanchored = view == "column" and not live["applied"]

    if view == "wrist":
        return {
            "points_px": planner if planner_source == "corrected" else last_anchors(
                _require_view_points(plan, "corrected", view)
            ),
            "source": "corrected",
            "on_tips": True,
            "planner_points_px": planner,
            "planner_source": planner_source,
            "locator_points_px": locator,
            "locator_accepted": accepted,
            "unanchored": False,
            "live_anchor": live,
        }

    if frame is None:
        frame = _plan_frame(plan, view)
    raw_measurement = None
    if frame is not None:
        metal = column_metal_operator_tips(frame)
        if metal is not None and not is_patch_pitch_triple(metal):
            raw_measurement = metal
        elif locator is not None and not is_patch_pitch_triple(locator):
            raw_measurement = locator

    if live["applied"]:
        return {
            "points_px": last_anchors(_require_view_points(plan, "corrected", view)),
            "source": "corrected",
            "on_tips": True,
            "planner_points_px": planner,
            "planner_source": planner_source,
            "locator_points_px": locator,
            "locator_accepted": accepted,
            "unanchored": False,
            "live_anchor": live,
            "raw_measurement_px": raw_measurement,
        }
    return {
        "points_px": planner,
        "source": planner_source,
        "on_tips": False,
        "planner_points_px": planner,
        "planner_source": planner_source,
        "locator_points_px": locator,
        "locator_accepted": accepted,
        "unanchored": True,
        "live_anchor": live,
        "raw_measurement_px": raw_measurement,
    }


def snapshot_warnings(plan: Any, view: str, resolution: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    prod_v13 = prod_v13_observed_tip_trajectory(plan, view)
    if prod_v13 is not None:
        warnings.append(PROD_V13_NO_FUTURE_PIXEL_STAMP)
        if prod_v13.get("display_px") is None:
            warnings.append(FCPGH_AUTO_H0_PENDING_STAMP)
    if view == "column" and resolution["unanchored"]:
        warnings.append(COLUMN_UNANCHORED_STAMP)
    if (
        view == "column"
        and not resolution["on_tips"]
        and resolution.get("source") != "fcpgh_auto_h0_pending"
    ):
        warnings.append(MARKS_NOT_ON_TIPS_STAMP)
    if view == "wrist" and not pickup_visible(plan, "wrist"):
        warnings.append(WRIST_PICKUP_MISSING_STAMP)
    return warnings


def render_planning_snapshot(
    frame: np.ndarray, plan: Any, view: str
) -> np.ndarray:
    if view not in VIEWS:
        raise ValueError(f"unsupported snapshot view: {view}")
    canvas = planning_frame(frame)
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = load_snapshot_font(11, bold=True)
    resolution = resolve_operator_tips(plan, view, frame=canvas)
    operator = np.asarray(resolution["points_px"], dtype=np.float32)
    planner = np.asarray(resolution["planner_points_px"], dtype=np.float32)
    predicted, commanded = plan_anchor_paths(plan, view)
    prod_v13 = prod_v13_observed_tip_trajectory(plan, view)
    if prod_v13 is not None:
        predicted = None
        if prod_v13.get("display_px") is None:
            commanded = None
    if view == "column":
        predicted = shift_column_paths(predicted, resolution.get("path_offset_px"))
        commanded = shift_column_paths(commanded, resolution.get("path_offset_px"))
        fk_points = resolution.get("fk_points_px")
        predicted = align_column_paths_to_operator(
            predicted, operator_px=operator, fk_points_px=fk_points
        )
        commanded = align_column_paths_to_operator(
            commanded, operator_px=operator, fk_points_px=fk_points
        )
        if column_commanded_is_losing_arc(plan):
            commanded = None

    path_start = operator if view == "column" else planner
    if commanded is not None:
        _draw_anchor_paths(draw, path_start, commanded, IK_COLORS, width=2)
    if prod_v13 is not None and prod_v13.get("display_px") is not None:
        _draw_observed_tip_paths(
            draw,
            np.asarray(prod_v13["display_px"], dtype=np.float32),
            PRED_COLORS,
            width=3,
        )
    else:
        _draw_anchor_paths(draw, path_start, predicted, PRED_COLORS, width=3)

    if view == "column" and resolution["unanchored"]:
        _draw_fk_triple(draw, planner)

    _draw_pickup(draw, plan, view)
    if resolution["on_tips"]:
        _draw_triple(
            draw,
            operator,
            OPERATOR_COLORS,
            radius=TIP_MARK_RADIUS,
        )
    elif str(resolution.get("source")) == "raw_measurement":
        _draw_hollow_triple(draw, operator, RAW_MEASUREMENT_COLORS)
    _draw_banners(draw, plan, view, resolution, font)
    rendered = np.array(image, dtype=np.uint8)
    if view == "column" and resolution["unanchored"]:
        _paint_pixels(rendered, planner, ((255, 255, 255),) * 3)
    if resolution["on_tips"]:
        _paint_pixels(rendered, operator, OPERATOR_COLORS)
    descriptor = pickup_descriptor(plan, view)
    if descriptor.get("visible") and descriptor.get("centroid_px") is not None:
        _paint_pixels(
            rendered,
            np.asarray(descriptor["centroid_px"], dtype=np.float32).reshape(1, 2),
            (PICKUP_COLOR,),
        )
    return rendered


def render_wrist_planning_snapshot(frame: np.ndarray, plan: Any) -> np.ndarray:
    return render_planning_snapshot(frame, plan, "wrist")


def render_column_planning_snapshot(frame: np.ndarray, plan: Any) -> np.ndarray:
    return render_planning_snapshot(frame, plan, "column")


def _draw_anchor_paths(
    draw: ImageDraw.ImageDraw,
    current_anchors: np.ndarray,
    values: np.ndarray | None,
    colors: tuple[tuple[int, int, int], ...],
    *,
    width: int,
) -> None:
    if values is None:
        return
    paths = np.asarray(values, dtype=np.float32)
    for anchor_index in range(paths.shape[1]):
        start_index = 0 if paths.shape[1] == 1 else anchor_index
        line = [
            tuple(map(float, current_anchors[start_index])),
            *[tuple(map(float, point)) for point in paths[:, anchor_index]],
        ]
        color = colors[start_index]
        if len(line) > 1:
            draw.line(line, fill=color, width=width)
        for point in line[1:]:
            draw.ellipse(
                (point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2),
                fill=color,
            )


def _draw_observed_tip_paths(
    draw: ImageDraw.ImageDraw,
    values: np.ndarray,
    colors: tuple[tuple[int, int, int], ...],
    *,
    width: int,
) -> None:
    """Draw exact observed rows; NaNs break a path instead of being filled."""

    paths = np.asarray(values, dtype=np.float32)
    if paths.ndim != 3 or paths.shape[1:] != (3, 2):
        raise ValueError("observed tip trajectory must have shape [T,3,2]")
    for anchor_index, color in enumerate(colors):
        previous: tuple[float, float] | None = None
        for point in paths[:, anchor_index]:
            if not np.all(np.isfinite(point)):
                previous = None
                continue
            current = tuple(map(float, point))
            if previous is not None:
                draw.line((*previous, *current), fill=color, width=width)
            draw.ellipse(
                (current[0] - 2, current[1] - 2, current[0] + 2, current[1] + 2),
                fill=color,
            )
            previous = current


def _draw_triple(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    colors: tuple[tuple[int, int, int], ...],
    *,
    radius: int,
) -> None:
    for point, color in zip(points, colors, strict=True):
        _draw_tip_mark(draw, float(point[0]), float(point[1]), color, radius=radius)


def _draw_hollow_triple(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    colors: tuple[tuple[int, int, int], ...],
    *,
    radius: int = 4,
) -> None:
    for point, color in zip(points, colors, strict=True):
        x, y = float(point[0]), float(point[1])
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=1,
        )


def _draw_tip_mark(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    color: tuple[int, int, int],
    *,
    radius: int,
) -> None:
    """High-contrast tip mark whose centroid is the supplied tip pixel."""

    mark_radius = float(radius)
    draw.ellipse(
        (
            x - mark_radius,
            y - mark_radius,
            x + mark_radius,
            y + mark_radius,
        ),
        fill=color,
        outline=(0, 0, 0),
        width=2,
    )
    arm = mark_radius + 2.0
    draw.line((x - arm, y, x + arm, y), fill=(0, 0, 0), width=2)
    draw.line((x, y - arm, x, y + arm), fill=(0, 0, 0), width=2)
    core = max(1.5, mark_radius * 0.35)
    draw.ellipse(
        (x - core, y - core, x + core, y + core),
        fill=color,
        outline=(0, 0, 0),
        width=1,
    )
    _stamp_pixel(draw, x, y, color)


def _stamp_pixel(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    color: tuple[int, int, int],
) -> None:
    """Force the claimed tip coordinate itself to carry the mark color.

    PIL ellipses with a 2px outline often miss the integer center. The
    operator-facing contract is that the three tip marks sit on the supplied
    tip pixels.
    """

    ix, iy = int(round(x)), int(round(y))
    draw.rectangle((ix, iy, ix, iy), fill=color)


def _paint_pixels(
    image: np.ndarray,
    points: np.ndarray,
    colors: tuple[tuple[int, int, int], ...],
) -> None:
    for point, color in zip(np.asarray(points, dtype=np.float32), colors, strict=True):
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            image[y, x] = color


def _draw_fk_triple(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
) -> None:
    for point in points:
        x, y = map(float, point)
        draw.line((x - 4, y - 4, x + 4, y + 4), fill=(255, 255, 255), width=1)
        draw.line((x + 4, y - 4, x - 4, y + 4), fill=(255, 255, 255), width=1)
        _stamp_pixel(draw, x, y, (255, 255, 255))


def _draw_pickup(
    draw: ImageDraw.ImageDraw,
    plan: Any,
    view: str,
) -> None:
    descriptor = pickup_descriptor(plan, view)
    if not descriptor.get("visible") or descriptor.get("centroid_px") is None:
        return
    x, y = (float(value) for value in descriptor["centroid_px"])
    radius = pickup_radius_px(descriptor)
    bbox = descriptor.get("bbox_xyxy")
    if view == "wrist" and bbox is not None:
        x0, y0, x1, y1 = (float(value) for value in bbox)
        draw.rectangle((x0, y0, x1, y1), outline=PICKUP_COLOR, width=2)
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=PICKUP_COLOR,
        width=2,
    )
    draw.line((x - radius - 3, y, x + radius + 3, y), fill=PICKUP_COLOR, width=2)
    draw.line((x, y - radius - 3, x, y + radius + 3), fill=PICKUP_COLOR, width=2)
    _stamp_pixel(draw, x, y, PICKUP_COLOR)


def _draw_banners(
    draw: ImageDraw.ImageDraw,
    plan: Any,
    view: str,
    resolution: Mapping[str, Any],
    font: Any,
) -> None:
    source = str(resolution["source"]).upper()
    pickup_source = str(pickup_descriptor(plan, view).get("source") or "unavailable")
    drop_source = "unavailable"
    if view == "column":
        drop = _mapping(_mapping(getattr(plan, "target_descriptors", None)).get("drop_zone_column"))
        drop_source = str(drop.get("source") or "unavailable")
    if view == "wrist":
        title = f"WRIST  tips={source} P={pickup_source}"
    else:
        planner_source = str(resolution["planner_source"]).upper()
        title = f"COL tips={source} P={pickup_source} D={drop_source}"
    prod_v13 = prod_v13_observed_tip_trajectory(plan, view)
    if prod_v13 is not None:
        row_count = len(prod_v13.get("rows") or [])
        title += f" RAW_TRACK T={row_count}"
    elif view == "column":
        predicted, commanded = plan_anchor_paths(plan, view)
        if column_commanded_is_losing_arc(plan):
            commanded = None
        if predicted is not None and commanded is not None:
            title += "  PRED+IK"
        elif predicted is not None:
            title += "  PRED"
        elif commanded is not None:
            title += "  IK"
    draw.rectangle((0, 0, 223, 16), fill=BANNER_FILL)
    draw.text((4, 2), title, fill=(245, 245, 245), font=font)
    warnings = snapshot_warnings(plan, view, resolution)
    for index, warning in enumerate(warnings):
        top = 17 + index * 15
        draw.rectangle((0, top, 223, top + 14), fill=WARNING_FILL)
        draw.text((4, top + 1), warning, fill=WARNING_TEXT, font=font)


def predicted_or_commanded(plan: Any, view: str) -> bool:
    predicted, commanded = plan_anchor_paths(plan, view)
    return predicted is not None or commanded is not None


def build_snapshot_meta(
    plan: Any,
    *,
    files: Mapping[str, str] | None = None,
    rendered_at_unix_s: float | None = None,
) -> dict[str, Any]:
    guard = getattr(plan, "guard", None)
    round_id = getattr(guard, "round_id", None)
    created_monotonic_s = getattr(guard, "created_monotonic_s", None)
    now = time.time() if rendered_at_unix_s is None else float(rendered_at_unix_s)
    anchors: dict[str, Any] = {}
    tip_separation: dict[str, Any] = {}
    pickup: dict[str, Any] = {}
    live_anchor: dict[str, Any] = {}
    warnings: list[str] = []
    for view in VIEWS:
        if not isinstance(getattr(plan, "corrected", None), Mapping):
            continue
        if view not in plan.corrected:
            continue
        resolution = resolve_operator_tips(plan, view, frame=_plan_frame(plan, view))
        pickup_row = pickup_descriptor(plan, view)
        drop_row = _mapping(
            _mapping(getattr(plan, "target_descriptors", None)).get("drop_zone_column")
        )
        anchors[view] = {
            "source": resolution["source"],
            "on_tips": bool(resolution["on_tips"]),
            "points_px": np.asarray(resolution["points_px"], dtype=np.float32),
            "planner_source": resolution["planner_source"],
            "planner_points_px": np.asarray(
                resolution["planner_points_px"], dtype=np.float32
            ),
            "locator_accepted": bool(resolution["locator_accepted"]),
            "pickup_source": pickup_row.get("source") or "unavailable",
            "drop_source": (
                drop_row.get("source") or "unavailable" if view == "column" else None
            ),
        }
        tip_separation[view] = detected_tip_separation_px(resolution["points_px"])
        pickup[view] = pickup_visible(plan, view)
        live_anchor[view] = resolution["live_anchor"]
        warnings.extend(snapshot_warnings(plan, view, resolution))
    return _jsonable(
        {
            "round_id": round_id,
            "timestamps": {
                "created_monotonic_s": created_monotonic_s,
                "rendered_at_unix_s": now,
            },
            "anchors": anchors,
            "live_anchor": live_anchor,
            "live_anchor.applied": bool(
                _mapping(live_anchor.get("column")).get("applied", False)
            ),
            "tip_separation_px": tip_separation,
            "pickup_visible": pickup,
            "warnings": warnings,
            "files": dict(files or {}),
        }
    )


def snapshot_jpeg_bytes(frame: np.ndarray) -> bytes:
    from io import BytesIO

    values = planning_frame(frame)
    output = BytesIO()
    Image.fromarray(values, mode="RGB").save(
        output,
        format="JPEG",
        quality=SNAPSHOT_JPEG_QUALITY,
        optimize=False,
    )
    return output.getvalue()


def write_planning_snapshots(
    round_root: Path,
    plan: Any,
    *,
    latest_dir: Path | None = None,
) -> dict[str, str]:
    """Write the historical pair into the round directory and refresh latest/."""

    round_root = Path(round_root)
    round_root.mkdir(parents=True, exist_ok=True)
    images: dict[str, np.ndarray] = {}
    files: dict[str, str] = {}
    for view in VIEWS:
        frame = planning_frame(_require_view_points(plan, "frames", view))
        images[view] = render_planning_snapshot(frame, plan, view)
        path = round_root / f"{view}_planning_snapshot.jpg"
        save_jpeg_atomic(path, images[view])
        files[view] = str(path)
    meta_path = round_root / "snapshots.json"
    files["meta"] = str(meta_path)
    if latest_dir is not None:
        latest_dir = Path(latest_dir)
        latest_dir.mkdir(parents=True, exist_ok=True)
        for view, image in images.items():
            latest_path = latest_dir / f"latest_{view}.jpg"
            save_jpeg_atomic(latest_path, image)
            files[f"latest_{view}"] = str(latest_path)
        files["latest_meta"] = str(latest_dir / "latest.json")
    meta = build_snapshot_meta(plan, files=files)
    save_json_atomic(meta_path, meta)
    if latest_dir is not None:
        save_json_atomic(latest_dir / "latest.json", meta)
    return files


def load_round_snapshot_jpeg(round_root: Path, view: str) -> bytes:
    if view not in VIEWS:
        raise ValueError(f"unsupported snapshot view: {view}")
    path = Path(round_root) / f"{view}_planning_snapshot.jpg"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_bytes()


def load_round_snapshot_meta(round_root: Path) -> dict[str, Any]:
    path = Path(round_root) / "snapshots.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("snapshots.json must be an object")
    return payload


def load_latest_snapshot_jpeg(latest_dir: Path, view: str) -> bytes:
    if view not in VIEWS:
        raise ValueError(f"unsupported snapshot view: {view}")
    path = Path(latest_dir) / f"latest_{view}.jpg"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_bytes()


def load_latest_snapshot_meta(latest_dir: Path) -> dict[str, Any]:
    path = Path(latest_dir) / "latest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("latest.json must be an object")
    return payload


def save_jpeg_atomic(path: Path, frame: np.ndarray) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        Image.fromarray(planning_frame(frame), mode="RGB").save(
            stream,
            format="JPEG",
            quality=SNAPSHOT_JPEG_QUALITY,
        )
        stream.flush()
        os.fsync(stream.fileno())
    _commit_atomic(temporary, path)


def save_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            _jsonable(payload),
            stream,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _commit_atomic(temporary, path)


def _commit_atomic(temporary: Path, destination: Path) -> None:
    temporary.replace(destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
