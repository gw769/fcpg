"""Fixed-background ORB/AKAZE + RANSAC affine for self-contact ``u_hat``.

Only fixed scene texture may generate inliers.  The arm, jaws, open-close
difference region, and setup-manifest exclude ROIs (fish, cardboard box,
any movable object) are masked out.  Whole-frame phase correlation, FK,
M0/M1, FCPG, and gripper self-features are not used.

Thresholds are conservative offline stand-ins.  A real-camera pilot must
freeze them; this module writes them into the setup manifest and never
treats the defaults as a physical pass.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from fcpg.self_contact_v1 import (
    PLANNING_CANVAS_HW,
    SelfContactSchemaError,
    empty_background_registration,
    is_planning_224_hw,
)

SETUP_SCHEMA_ID = "minimal_physical_grounding_screen.self_contact_setup.v1"
PREFERRED_INCLUDE_ROI_IDS = ("base_or_mount_plate", "table_edge")
MOVABLE_EXCLUDE_KINDS = frozenset(
    {"fish", "cardboard_box", "movable_object", "box", "tube", "hose", "pipe"}
)

# Conservative defaults.  Starred values need a real-camera pilot.
DEFAULT_REGISTRATION_THRESHOLDS: dict[str, Any] = {
    "detector": "ORB",
    "fallback_detector": "AKAZE",
    "min_inliers": 12,  # PILOT
    "min_inliers_per_include_roi": 2,  # PILOT
    "min_spatial_coverage": 0.20,  # PILOT: inlier bbox / include-union area
    "min_rois_with_inliers": 2,
    "max_rms_px": 1.5,  # PILOT @ planning 224
    "max_scale_deviation": 0.15,  # PILOT
    "max_rotation_deg": 8.0,  # PILOT
    "max_translation_px": 40.0,  # PILOT
    "min_roi_separation_px": 40.0,  # PILOT
    "min_include_rois": 2,
    "motion_diff_threshold": 15,  # PILOT
    "motion_dilate_px": 7,  # PILOT
    "coarse_identity_mad": 80.0,  # PILOT; 32×32 include preview
    "coarse_identity_whole_frame_mad": 22.0,  # PILOT; 32×32 whole frame
    "coarse_identity_corner_mad": 25.0,  # PILOT; 16×16 corner patches
    "coarse_identity_min_hot_corners": 3,
    "nfeatures": 800,
    "ransac_reproj_threshold": 2.0,
    "note": (
        "These defaults are conservative offline stand-ins.  A real-camera "
        "pilot must freeze inlier count, spatial coverage, RMS, and "
        "transform bounds before any physical Gate 1 claim."
    ),
    "frozen": False,
}


class BackgroundRegistrationError(ValueError):
    """Fail-closed registration refusal.  Not a screen PASS/FAIL."""


def setup_manifest_template() -> dict[str, Any]:
    """Unfilled dry-run template.  ``valid`` is always false."""

    return {
        "schema": SETUP_SCHEMA_ID,
        "valid": False,
        "gripper_visible": None,
        "include_rois": [
            {
                "id": "base_or_mount_plate",
                "kind": "fixed_background",
                "xyxy": None,
                "filled": False,
                "note": "Existing SO-101 base / mounting plate.  Do not add objects.",
            },
            {
                "id": "table_edge",
                "kind": "fixed_background",
                "xyxy": None,
                "filled": False,
                "note": "Existing table edge / installation lip.  Do not add objects.",
            },
        ],
        "exclude_rois": [
            {
                "id": "fish",
                "kind": "fish",
                "xyxy": None,
                "filled": False,
                "note": "Task fish must not generate background inliers.",
            },
            {
                "id": "cardboard_box",
                "kind": "cardboard_box",
                "xyxy": None,
                "filled": False,
                "note": "Task box must not generate background inliers.",
            },
        ],
        "gripper_roi": None,
        "thresholds": dict(DEFAULT_REGISTRATION_THRESHOLDS),
        "reason": "dry_run_template_unfilled",
        "notes": {
            "column_must_see_gripper": True,
            "column_must_see_two_separated_fixed_rois": False,
            "dual_fixed_roi_is_optional_a_diagnostic": True,
            "do_not_add_objects": True,
            "do_not_move_or_raise_task_objects": True,
            "high_clearance_arm_recheck_allowed": True,
            "dry_run_must_not_claim_valid": True,
            "pre_motion_identity_required_on_live": True,
        },
        "camera_identity_reference": {
            "column": None,
            "wrist": None,
            "canvas": "letterbox_v1",
            "source_size": [640, 480],
            "gripper_roi_xyxy": None,
            "role_fingerprints": None,
            "note": "Planning-224 letterbox_v1 frames from the verified setup FOV. Live handshake refuses without them.",
        },
    }


def _as_mapping(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SelfContactSchemaError(f"{where} must be an object")
    return dict(value)


def _thresholds(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_REGISTRATION_THRESHOLDS)
    if raw:
        merged.update(dict(raw))
    merged["frozen"] = bool(raw.get("frozen")) if raw else False
    return merged


def _parse_xyxy(raw: Any) -> tuple[int, int, int, int] | None:
    if raw is None or raw == "":
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise SelfContactSchemaError("ROI xyxy must be [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (int(round(float(v))) for v in raw)
    if x1 <= x0 or y1 <= y0:
        raise SelfContactSchemaError("ROI xyxy must have positive area")
    return (x0, y0, x1, y1)


def _roi_centroid(xyxy: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = xyxy
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _roi_area(xyxy: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = xyxy
    return float(max(1, x1 - x0) * max(1, y1 - y0))


def _point_in_xyxy(point: Sequence[float], xyxy: tuple[int, int, int, int]) -> bool:
    x, y = float(point[0]), float(point[1])
    x0, y0, x1, y1 = xyxy
    return x0 <= x < x1 and y0 <= y < y1


def parse_setup_manifest(
    raw: Mapping[str, Any],
    *,
    where: str = "self_contact_setup",
) -> dict[str, Any]:
    document = _as_mapping(raw, where=where)
    if document.get("schema") not in {SETUP_SCHEMA_ID, None}:
        raise SelfContactSchemaError(f"{where}.schema is not {SETUP_SCHEMA_ID}")
    include = []
    for index, row in enumerate(document.get("include_rois") or []):
        item = _as_mapping(row, where=f"{where}.include_rois[{index}]")
        include.append(
            {
                "id": str(item.get("id") or f"include_{index}"),
                "kind": str(item.get("kind") or "fixed_background"),
                "xyxy": _parse_xyxy(item.get("xyxy")),
                "filled": bool(item.get("filled", item.get("xyxy") is not None)),
            }
        )
    exclude = []
    for index, row in enumerate(document.get("exclude_rois") or []):
        item = _as_mapping(row, where=f"{where}.exclude_rois[{index}]")
        exclude.append(
            {
                "id": str(item.get("id") or f"exclude_{index}"),
                "kind": str(item.get("kind") or "movable_object"),
                "xyxy": _parse_xyxy(item.get("xyxy")),
                "filled": bool(item.get("filled", item.get("xyxy") is not None)),
            }
        )
    return {
        "schema": SETUP_SCHEMA_ID,
        "valid": bool(document.get("valid", False)),
        "gripper_visible": document.get("gripper_visible"),
        "include_rois": include,
        "exclude_rois": exclude,
        "gripper_roi": _parse_xyxy((document.get("gripper_roi") or {}).get("xyxy"))
        if isinstance(document.get("gripper_roi"), Mapping)
        else _parse_xyxy(document.get("gripper_roi")),
        "thresholds": _thresholds(document.get("thresholds") if isinstance(document.get("thresholds"), Mapping) else None),
        "reason": document.get("reason"),
    }


def _xyxy_overlap(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> bool:
    ax0, ay0, ax1, ay1 = left
    bx0, by0, bx1, by1 = right
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def evaluate_workspace_object_interference(
    raw: Mapping[str, Any],
    motion_roi: Mapping[str, Any] | Sequence[int] | None = None,
    *,
    where: str = "workspace_object_interference",
) -> dict[str, Any]:
    """Block if fish/box/tube exclude ROIs intersect the gripper close path.

    High-clearance is allowed: move the arm so the motion/gripper ROI no
    longer overlaps task objects.  Do not move or raise the objects.
    Missing exclude ROIs cannot invent a collision.
    """

    parsed = parse_setup_manifest(raw, where=where) if "include_rois" in raw or "exclude_rois" in raw else raw
    if not isinstance(parsed, Mapping) or "exclude_rois" not in parsed:
        parsed = parse_setup_manifest(raw, where=where)
    gripper = parsed.get("gripper_roi")
    motion_xyxy = None
    if isinstance(motion_roi, Mapping):
        motion_xyxy = _parse_xyxy(motion_roi.get("xyxy"))
    elif motion_roi is not None:
        motion_xyxy = _parse_xyxy(motion_roi)
    path_roi = motion_xyxy or gripper
    hits: list[str] = []
    for row in parsed.get("exclude_rois") or []:
        kind = str(row.get("kind") or "")
        xyxy = row.get("xyxy")
        if xyxy is None or kind not in MOVABLE_EXCLUDE_KINDS:
            continue
        if path_roi is not None and _xyxy_overlap(tuple(int(v) for v in path_roi), tuple(int(v) for v in xyxy)):
            hits.append(str(row.get("id") or kind))
    blocked = bool(hits)
    return {
        "blocked": blocked,
        "high_clearance": (not blocked) and path_roi is not None,
        "overlapping_objects": hits,
        "path_roi_xyxy": None if path_roi is None else list(path_roi),
        "reason": (
            None
            if not blocked
            else (
                "task object(s) "
                f"{hits} intersect the gripper/closed-path ROI at this pose; "
                "move the arm to a high-clearance configuration.  "
                "Do not move or raise the fish/box/tube"
            )
        ),
        "do_not_move_objects": True,
    }


def evaluate_setup_preflight(
    raw: Mapping[str, Any],
    *,
    where: str = "self_contact_setup",
    motion_roi: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """B preflight: gripper visible and close-path clear of task objects.

    Dual fixed-background ROIs are an optional A diagnostic and do not
    refuse a B session.
    """

    parsed = parse_setup_manifest(raw, where=where)
    thresholds = parsed["thresholds"]
    filled = [row for row in parsed["include_rois"] if row["xyxy"] is not None]
    b_reasons: list[str] = []
    a_reasons: list[str] = []
    if parsed.get("gripper_visible") is not True:
        b_reasons.append("column view does not confirm the gripper is visible")
    if len(filled) < int(thresholds["min_include_rois"]):
        a_reasons.append(
            f"A diagnostic: need ≥{thresholds['min_include_rois']} filled "
            f"fixed-background include ROIs (prefer {PREFERRED_INCLUDE_ROI_IDS}); "
            f"got {len(filled)}.  A unauthorized, not a B blocker"
        )
    if len(filled) >= 2:
        max_sep = 0.0
        for i, left in enumerate(filled):
            for right in filled[i + 1 :]:
                cx0, cy0 = _roi_centroid(left["xyxy"])
                cx1, cy1 = _roi_centroid(right["xyxy"])
                max_sep = max(max_sep, float(np.hypot(cx0 - cx1, cy0 - cy1)))
        if max_sep < float(thresholds["min_roi_separation_px"]):
            a_reasons.append(
                "A diagnostic: fixed-background include ROIs are not spatially "
                f"separated ({max_sep:.1f} < {thresholds['min_roi_separation_px']})"
            )
    interference = evaluate_workspace_object_interference(
        parsed, motion_roi or ({"xyxy": parsed["gripper_roi"]} if parsed.get("gripper_roi") else None)
    )
    if interference["blocked"]:
        b_reasons.append(str(interference["reason"]))
    dry_template = str(raw.get("reason") or "") == "dry_run_template_unfilled" or (
        raw.get("valid") is False and not filled and parsed.get("gripper_visible") is not True
    )
    if dry_template:
        b_reasons.append("dry-run or unfilled setup template cannot claim valid")
    b_valid = (
        not b_reasons
        and parsed.get("gripper_visible") is True
        and not dry_template
    )
    a_available = not a_reasons and len(filled) >= 2
    return {
        "schema": SETUP_SCHEMA_ID,
        "valid": b_valid,
        "b_ready": b_valid,
        "gripper_visible": parsed.get("gripper_visible") is True,
        "n_include_rois_filled": len(filled),
        "include_rois": parsed["include_rois"],
        "exclude_rois": parsed["exclude_rois"],
        "thresholds": thresholds,
        "background_sensitivity_available": a_available,
        "claim_A_intrinsic_fk_unauthorized": True,
        "workspace_object_interference": interference,
        "a_reasons": a_reasons,
        "reason": None if b_valid else "; ".join(b_reasons) or "setup_incomplete",
        "reasons": b_reasons,
    }


def _as_rgb(frame: np.ndarray, *, where: str) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise BackgroundRegistrationError(f"{where} must be uint8 RGB [H,W,3]")
    if image.shape[0] < 16 or image.shape[1] < 16:
        raise BackgroundRegistrationError(f"{where} is too small")
    return image


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def _clip_xyxy(
    xyxy: tuple[int, int, int, int], shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    height, width = int(shape[0]), int(shape[1])
    x0, y0, x1, y1 = xyxy
    return (
        max(0, min(width, x0)),
        max(0, min(height, y0)),
        max(0, min(width, x1)),
        max(0, min(height, y1)),
    )


def _mask_from_rois(
    shape: tuple[int, ...],
    rois: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for row in rois:
        xyxy = row.get("xyxy") if isinstance(row, Mapping) else None
        if xyxy is None:
            continue
        x0, y0, x1, y1 = _clip_xyxy(tuple(int(v) for v in xyxy), shape)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return mask


def motion_exclude_mask(
    reference: np.ndarray,
    contact: np.ndarray,
    *,
    threshold: int,
    dilate_px: int,
) -> np.ndarray:
    """Open-close difference: arm, jaws, and anything that moved."""

    diff = cv2.absdiff(_gray(reference), _gray(contact))
    _ignored, motion = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)
    radius = max(1, int(dilate_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(motion, kernel)


def apply_affine_px(
    point: Sequence[float], matrix_2x3: Sequence[Sequence[float]]
) -> tuple[float, float]:
    """Apply pulse-contact → session ``transform_2x3`` to one image point.

    ``cv2.estimateAffine2D(contact, session)`` yields ``session ≈ M @ [x, y, 1]``.
    Raw knife-edge ``u_hat`` is observed on the contact frame, so this maps
    it into the session / canonical canvas.  The inverse is not applied.
    """

    pixel = np.asarray(point, dtype=np.float64).reshape(2)
    matrix = np.asarray(matrix_2x3, dtype=np.float64)
    if matrix.shape != (2, 3):
        raise BackgroundRegistrationError("transform_2x3 must be 2x3")
    mapped = matrix @ np.asarray([pixel[0], pixel[1], 1.0], dtype=np.float64)
    if not np.all(np.isfinite(mapped)):
        raise BackgroundRegistrationError("affine mapped u_hat is not finite")
    return (float(mapped[0]), float(mapped[1]))


def decompose_affine(matrix_2x3: Sequence[Sequence[float]]) -> dict[str, float]:
    matrix = np.asarray(matrix_2x3, dtype=np.float64)
    if matrix.shape != (2, 3):
        raise BackgroundRegistrationError("transform_2x3 must be 2x3")
    scale_x = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    scale_y = float(np.hypot(matrix[1, 0], matrix[1, 1]))
    rotation_deg = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    return {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "rotation_deg": rotation_deg,
        "translation_x": float(matrix[0, 2]),
        "translation_y": float(matrix[1, 2]),
    }


def assess_transform(
    matrix_2x3: Sequence[Sequence[float]] | None,
    *,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    if matrix_2x3 is None:
        return {"ok": False, "reason": "transform_missing", "decompose": None}
    parts = decompose_affine(matrix_2x3)
    reasons: list[str] = []
    max_scale_dev = float(thresholds["max_scale_deviation"])
    for key in ("scale_x", "scale_y"):
        if abs(parts[key] - 1.0) > max_scale_dev:
            reasons.append(f"{key}={parts[key]:.3f} outside 1±{max_scale_dev}")
    if abs(parts["rotation_deg"]) > float(thresholds["max_rotation_deg"]):
        reasons.append(
            f"rotation={parts['rotation_deg']:.2f} exceeds {thresholds['max_rotation_deg']}"
        )
    translation = float(np.hypot(parts["translation_x"], parts["translation_y"]))
    if translation > float(thresholds["max_translation_px"]):
        reasons.append(
            f"translation={translation:.2f} exceeds {thresholds['max_translation_px']}"
        )
    if not all(np.isfinite(list(parts.values()))):
        reasons.append("transform is not finite")
    return {
        "ok": not reasons,
        "reason": None if not reasons else "; ".join(reasons),
        "decompose": parts,
    }


def _detect(gray: np.ndarray, mask: np.ndarray, *, method: str, nfeatures: int):
    if method == "AKAZE":
        detector = cv2.AKAZE_create()
    else:
        detector = cv2.ORB_create(nfeatures=int(nfeatures))
    return detector.detectAndCompute(gray, mask)


def _match(desc_a: np.ndarray, desc_b: np.ndarray, *, method: str) -> list[cv2.DMatch]:
    if desc_a is None or desc_b is None or len(desc_a) < 4 or len(desc_b) < 4:
        return []
    norm = cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    raw = matcher.knnMatch(desc_a, desc_b, k=2)
    good: list[cv2.DMatch] = []
    for pair in raw:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < 0.75 * second.distance:
            good.append(best)
    return good


def _coarse_identity(a: np.ndarray, b: np.ndarray, include: np.ndarray, *, mad_max: float) -> bool:
    if a.shape != b.shape:
        return False
    masked_a = a.copy()
    masked_b = b.copy()
    masked_a[include == 0] = 0
    masked_b[include == 0] = 0
    small_a = cv2.resize(masked_a, (32, 32), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(masked_b, (32, 32), interpolation=cv2.INTER_AREA)
    mad = float(np.mean(np.abs(small_a.astype(np.int16) - small_b.astype(np.int16))))
    return mad <= float(mad_max)


def assess_camera_identity(
    session: np.ndarray,
    pulse_reference: np.ndarray,
    pulse_contact: np.ndarray | None = None,
    *,
    whole_frame_mad_max: float = 22.0,
    corner_mad_max: float = 25.0,
    min_hot_corners: int = 3,
) -> dict[str, Any]:
    """Coarse framing / hash / large break-frame.  Not ORB.  Not dual-ROI.

    A ~27 px evening-type whole-scene translation of structured content
    is fail-closed.  Uniform / no-texture frames and missing include
    ROIs do not trigger.  Local gripper open-close is not a break.
    """

    session_g = session if session.ndim == 2 else _gray(session)
    pref_g = pulse_reference if pulse_reference.ndim == 2 else _gray(pulse_reference)
    if session_g.shape != pref_g.shape:
        return {
            "ok": False,
            "camera_identity_break": True,
            "reason": "reference_framing_changed",
            "whole_frame_mad": None,
            "n_hot_corners": None,
        }

    def _mad32(a: np.ndarray, b: np.ndarray) -> float:
        small_a = cv2.resize(a, (32, 32), interpolation=cv2.INTER_AREA)
        small_b = cv2.resize(b, (32, 32), interpolation=cv2.INTER_AREA)
        return float(np.mean(np.abs(small_a.astype(np.int16) - small_b.astype(np.int16))))

    height, width = session_g.shape[:2]
    corners = (
        (0, 0, max(1, width // 4), max(1, height // 4)),
        (3 * width // 4, 0, width, max(1, height // 4)),
        (0, 3 * height // 4, max(1, width // 4), height),
        (3 * width // 4, 3 * height // 4, width, height),
    )

    def _hot_corners(a: np.ndarray, b: np.ndarray) -> tuple[int, list[float]]:
        mads: list[float] = []
        for x0, y0, x1, y1 in corners:
            if x1 <= x0 or y1 <= y0:
                continue
            patch_a = cv2.resize(a[y0:y1, x0:x1], (16, 16), interpolation=cv2.INTER_AREA)
            patch_b = cv2.resize(b[y0:y1, x0:x1], (16, 16), interpolation=cv2.INTER_AREA)
            mads.append(
                float(np.mean(np.abs(patch_a.astype(np.int16) - patch_b.astype(np.int16))))
            )
        return sum(mad > float(corner_mad_max) for mad in mads), mads

    whole_mad = _mad32(session_g, pref_g)
    n_hot, corner_mads = _hot_corners(session_g, pref_g)
    if pulse_contact is not None:
        contact_g = pulse_contact if pulse_contact.ndim == 2 else _gray(pulse_contact)
        if contact_g.shape == session_g.shape:
            whole_mad = max(whole_mad, _mad32(session_g, contact_g))
            n_hot_c, corner_c = _hot_corners(session_g, contact_g)
            n_hot = max(n_hot, n_hot_c)
            corner_mads = corner_mads + corner_c
    broke = whole_mad > float(whole_frame_mad_max) and n_hot >= int(min_hot_corners)
    return {
        "ok": not broke,
        "camera_identity_break": bool(broke),
        "reason": None if not broke else "whole_frame_identity_break",
        "whole_frame_mad": whole_mad,
        "n_hot_corners": n_hot,
        "corner_mads": corner_mads,
        "session_include_hash": _include_hash(session_g, np.ones_like(session_g)),
        "pulse_reference_include_hash": _include_hash(pref_g, np.ones_like(pref_g)),
    }


def _include_hash(gray: np.ndarray, include: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(gray.shape, dtype=np.int32).tobytes())
    digest.update(gray[include > 0].tobytes() if np.any(include) else gray.tobytes())
    return digest.hexdigest()


def _invalid(
    reason: str,
    *,
    n_inliers: int = 0,
    coverage: float = 0.0,
    residual: float | None = None,
    transform: list[list[float]] | None = None,
    camera_break: bool = False,
    movable: bool = False,
    exclude_hits: int = 0,
    sufficient: bool = False,
    coverage_ok: bool = False,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = empty_background_registration()
    payload.update(
        {
            "valid": False,
            "n_inliers": int(n_inliers),
            "n_inliers_sufficient": bool(sufficient),
            "spatial_coverage": float(coverage),
            "spatial_coverage_sufficient": bool(coverage_ok),
            "residual_px224": residual,
            "transform_2x3": transform,
            "camera_break_frame": bool(camera_break),
            "camera_identity_break": bool(camera_break),
            "movable_objects_dominated": bool(movable),
            "inliers_in_exclude_rois": int(exclude_hits),
            "reason": reason,
            "diagnostics": {"reason": reason, **dict(diagnostics or {})},
        }
    )
    return payload


def register_background(
    session_reference: np.ndarray,
    pulse_reference: np.ndarray,
    pulse_contact: np.ndarray,
    setup_manifest: Mapping[str, Any],
    *,
    where: str = "background_registration",
) -> dict[str, Any]:
    """Estimate pulse-contact → session affine on fixed-background inliers."""

    session = _as_rgb(session_reference, where=f"{where}.session_reference")
    pref = _as_rgb(pulse_reference, where=f"{where}.pulse_reference")
    contact = _as_rgb(pulse_contact, where=f"{where}.pulse_contact")
    parsed = parse_setup_manifest(setup_manifest, where=f"{where}.setup")
    thresholds = parsed["thresholds"]
    if not (
        is_planning_224_hw(session.shape)
        and is_planning_224_hw(pref.shape)
        and is_planning_224_hw(contact.shape)
    ):
        return _invalid(
            "frame_not_planning_224",
            diagnostics={
                "session_shape": list(session.shape),
                "required_hw": list(PLANNING_CANVAS_HW),
            },
        )
    if session.shape != pref.shape or session.shape != contact.shape:
        return _invalid(
            "reference_framing_changed",
            camera_break=True,
            diagnostics={"session_shape": list(session.shape), "contact_shape": list(contact.shape)},
        )

    identity = assess_camera_identity(
        session,
        pref,
        contact,
        whole_frame_mad_max=float(thresholds.get("coarse_identity_whole_frame_mad", 22.0)),
        corner_mad_max=float(thresholds.get("coarse_identity_corner_mad", 25.0)),
        min_hot_corners=int(thresholds.get("coarse_identity_min_hot_corners", 3)),
    )
    if identity["camera_identity_break"]:
        return _invalid(
            str(identity.get("reason") or "whole_frame_identity_break"),
            camera_break=True,
            diagnostics=identity,
        )

    include = _mask_from_rois(session.shape, parsed["include_rois"])
    exclude = _mask_from_rois(session.shape, parsed["exclude_rois"])
    gripper_mask = np.zeros(session.shape[:2], dtype=np.uint8)
    if parsed["gripper_roi"] is not None:
        gx0, gy0, gx1, gy1 = _clip_xyxy(parsed["gripper_roi"], session.shape)
        gripper_mask[gy0:gy1, gx0:gx1] = 255
        exclude = cv2.bitwise_or(exclude, gripper_mask)
    motion = motion_exclude_mask(
        pref,
        contact,
        threshold=int(thresholds["motion_diff_threshold"]),
        dilate_px=int(thresholds["motion_dilate_px"]),
    )
    # Open-close difference is the jaws/arm, not the camera-drifted background.
    if np.any(gripper_mask):
        radius = max(1, int(thresholds["motion_dilate_px"]))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (4 * radius + 1, 4 * radius + 1)
        )
        grip_zone = cv2.dilate(gripper_mask, kernel)
        motion = cv2.bitwise_and(motion, grip_zone)
    exclude = cv2.bitwise_or(exclude, motion)
    usable = cv2.bitwise_and(include, cv2.bitwise_not(exclude))
    session_gray = _gray(session)
    contact_gray = _gray(contact)
    pref_gray = _gray(pref)
    if int(np.count_nonzero(include)) >= 64 and not _coarse_identity(
        pref_gray,
        session_gray,
        include,
        mad_max=float(thresholds["coarse_identity_mad"]),
    ):
        # Include-ROI hash/framing is an extra identity signal only when
        # dual ROIs exist.  ORB failure below is not an identity break.
        return _invalid(
            "reference_hash_or_framing_changed",
            camera_break=True,
            diagnostics={
                "session_include_hash": _include_hash(session_gray, include),
                "pulse_reference_include_hash": _include_hash(pref_gray, include),
            },
        )
    if int(np.count_nonzero(usable)) < 64:
        return _invalid(
            "insufficient_fixed_background_mask",
            movable=bool(np.any(exclude)),
            diagnostics={
                "usable_px": int(np.count_nonzero(usable)),
                "camera_identity": identity,
                "a_unavailable": True,
            },
        )

    methods = [str(thresholds.get("detector") or "ORB")]
    fallback = str(thresholds.get("fallback_detector") or "AKAZE")
    if fallback not in methods:
        methods.append(fallback)

    best: dict[str, Any] | None = None
    for method in methods:
        kps_a, desc_a = _detect(
            session_gray, usable, method=method, nfeatures=int(thresholds["nfeatures"])
        )
        kps_b, desc_b = _detect(
            contact_gray, usable, method=method, nfeatures=int(thresholds["nfeatures"])
        )
        matches = _match(desc_a, desc_b, method=method)
        if len(matches) < 4:
            continue
        src = np.asarray([kps_b[item.trainIdx].pt for item in matches], dtype=np.float64)
        dst = np.asarray([kps_a[item.queryIdx].pt for item in matches], dtype=np.float64)
        matrix, inlier_mask = cv2.estimateAffine2D(
            src.reshape(-1, 1, 2),
            dst.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=float(thresholds["ransac_reproj_threshold"]),
            maxIters=2000,
            confidence=0.99,
        )
        if matrix is None or inlier_mask is None:
            continue
        inlier_idx = np.flatnonzero(inlier_mask.reshape(-1) > 0)
        src_in = src[inlier_idx]
        dst_in = dst[inlier_idx]
        exclude_hits = 0
        kept_src = []
        kept_dst = []
        for point_src, point_dst in zip(src_in, dst_in):
            hit = any(
                row["xyxy"] is not None
                and (
                    _point_in_xyxy(point_src, row["xyxy"])
                    or _point_in_xyxy(point_dst, row["xyxy"])
                )
                for row in parsed["exclude_rois"]
            )
            if hit:
                exclude_hits += 1
                continue
            kept_src.append(point_src)
            kept_dst.append(point_dst)
        if not kept_src:
            candidate = {
                "n": 0,
                "exclude_hits": exclude_hits,
                "method": method,
                "matrix": None,
                "src": np.zeros((0, 2)),
                "dst": np.zeros((0, 2)),
            }
        else:
            candidate = {
                "n": len(kept_src),
                "exclude_hits": exclude_hits,
                "method": method,
                "matrix": matrix.astype(np.float64),
                "src": np.asarray(kept_src, dtype=np.float64),
                "dst": np.asarray(kept_dst, dtype=np.float64),
            }
        if best is None or candidate["n"] > best["n"]:
            best = candidate

    if best is None or best["n"] == 0 or best["matrix"] is None:
        return _invalid(
            "insufficient_inliers",
            n_inliers=0 if best is None else int(best["n"]),
            exclude_hits=0 if best is None else int(best["exclude_hits"]),
            movable=best is not None and int(best["exclude_hits"]) > 0,
        )

    src_in = best["src"]
    dst_in = best["dst"]
    predicted = np.concatenate(
        [src_in, np.ones((len(src_in), 1), dtype=np.float64)], axis=1
    ) @ best["matrix"].T
    residual = float(np.sqrt(np.mean(np.sum((predicted - dst_in) ** 2, axis=1))))
    include_area = float(max(1, int(np.count_nonzero(include))))
    if len(src_in) >= 2:
        coverage = float(
            max(1.0, float(np.ptp(src_in[:, 0])) * float(np.ptp(src_in[:, 1])))
            / include_area
        )
    else:
        coverage = 0.0
    rois_with = 0
    per_roi: dict[str, int] = {}
    for row in parsed["include_rois"]:
        if row["xyxy"] is None:
            continue
        count = sum(1 for point in src_in if _point_in_xyxy(point, row["xyxy"]))
        per_roi[row["id"]] = count
        if count >= int(thresholds["min_inliers_per_include_roi"]):
            rois_with += 1

    n_inliers = int(best["n"])
    sufficient = n_inliers >= int(thresholds["min_inliers"])
    coverage_ok = (
        coverage >= float(thresholds["min_spatial_coverage"])
        and rois_with >= int(thresholds["min_rois_with_inliers"])
    )
    transform = best["matrix"].tolist()
    transform_ok = assess_transform(transform, thresholds=thresholds)
    movable = int(best["exclude_hits"]) > 0
    rms_ok = residual <= float(thresholds["max_rms_px"])
    reasons: list[str] = []
    if not sufficient:
        reasons.append(f"n_inliers={n_inliers} < {thresholds['min_inliers']}")
    if not coverage_ok:
        reasons.append(
            f"spatial_coverage={coverage:.3f} or rois_with_inliers={rois_with} insufficient"
        )
    if not rms_ok:
        reasons.append(f"rms={residual:.3f} > {thresholds['max_rms_px']}")
    if not transform_ok["ok"]:
        reasons.append(str(transform_ok["reason"]))
    if movable:
        reasons.append("exclude-ROI features reached the inlier set")

    valid = not reasons and sufficient and coverage_ok and rms_ok and transform_ok["ok"] and not movable
    payload = empty_background_registration()
    payload.update(
        {
            "valid": valid,
            "n_inliers": n_inliers,
            "n_inliers_sufficient": sufficient,
            "spatial_coverage": coverage,
            "spatial_coverage_sufficient": coverage_ok,
            "residual_px224": residual,
            "transform_2x3": transform,
            "camera_break_frame": False,
            "camera_identity_break": False,
            "movable_objects_dominated": movable,
            "inliers_in_exclude_rois": int(best["exclude_hits"]),
            "reason": None if valid else "; ".join(reasons),
            "diagnostics": {
                "reason": None if valid else "; ".join(reasons),
                "detector": best["method"],
                "thresholds": dict(thresholds),
                "per_include_roi_inliers": per_roi,
                "inlier_points_px": src_in.tolist(),
                "transform": transform_ok,
                "session_include_hash": _include_hash(session_gray, include),
            },
        }
    )
    return payload


def render_textured_roi(
    frame: np.ndarray,
    xyxy: tuple[int, int, int, int],
    *,
    cell: int,
    phase: int,
    seed: int,
    amplitude: int = 40,
) -> None:
    """Deterministic checker + noise used by offline tests."""

    x0, y0, x1, y1 = xyxy
    height, width = y1 - y0, x1 - x0
    ys, xs = np.indices((height, width))
    checker = ((((xs + int(phase)) // int(cell)) + (ys // int(cell))) % 2) * 180 + 40
    rng = np.random.default_rng(int(seed))
    noise = rng.integers(-int(amplitude), int(amplitude) + 1, (height, width))
    patch = np.clip(checker + noise, 0, 255).astype(np.uint8)
    frame[y0:y1, x0:x1] = np.stack([patch, patch, patch], axis=-1)


def shift_roi(
    frame: np.ndarray,
    xyxy: tuple[int, int, int, int],
    *,
    dx: int,
    dy: int,
) -> None:
    x0, y0, x1, y1 = xyxy
    patch = frame[y0:y1, x0:x1].copy()
    matrix = np.asarray([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    warped = cv2.warpAffine(
        patch,
        matrix,
        (patch.shape[1], patch.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT,
    )
    frame[y0:y1, x0:x1] = warped
