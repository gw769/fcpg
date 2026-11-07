"""Hard pre-motion camera identity handshake.

Pilot must grab several planning frames after opening cameras and before
any ordinary-pulse or seek Goal write.  A role swap, stale/blank frame,
letterbox/crop mismatch, or coarse identity break refuses the attempt
without consuming it.  MAD thresholds are the existing frozen coarse
identity bounds; they are not relaxed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fcpg.letterbox_geometry import letterbox_geometry
from fcpg.self_contact_background_registration import (
    DEFAULT_REGISTRATION_THRESHOLDS,
    assess_camera_identity,
)
from fcpg.self_contact_v1 import is_planning_224_hw

PRE_MOTION_N_FRAMES = 3
PRE_MOTION_BLANK_STD_MAX = 6.0
PRE_MOTION_BLANK_UNIQUE_MAX = 24
LETTERBOX_PAD_TOL_PX = 2
# 640x480 letterbox_v1: pad_y=28, content 224x168.  Not inferred from video index.
EXPECTED_SOURCE_SIZE = (640, 480)

def _expected_letterbox() -> Any:
    return letterbox_geometry(EXPECTED_SOURCE_SIZE[0], EXPECTED_SOURCE_SIZE[1])


def camera_role_fingerprint(
    frame: np.ndarray,
    *,
    role: str,
    source_size: tuple[int, int] = EXPECTED_SOURCE_SIZE,
) -> dict[str, Any]:
    """Stable role/canvas fingerprint.  No video index."""

    rgb = _as_rgb(frame)
    canvas = inspect_letterbox_canvas(rgb)
    geom = letterbox_geometry(int(source_size[0]), int(source_size[1]))
    return {
        "role": str(role),
        "binding": "by-id",
        "protocol": "letterbox_v1",
        "source_size": [int(source_size[0]), int(source_size[1])],
        "canvas_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
        "letterbox_pad_y": int(canvas["pad_y"]),
        "letterbox_content_h": int(canvas["content_h"]),
        "expected_pad_y": int(geom.pad_y),
        "expected_content_h": int(geom.content_height),
    }


def verify_camera_role_fingerprint(
    frame: np.ndarray,
    expected: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    """Compare a live frame to a stored role fingerprint.  Fail closed."""

    live = camera_role_fingerprint(frame, role=role)
    reasons: list[str] = []
    if str(expected.get("role") or role) != str(role):
        reasons.append("role_mismatch")
    if expected.get("binding") not in {None, "by-id"}:
        reasons.append("binding_not_by_id")
    if expected.get("resolved_device") and "/dev/video" in str(expected["resolved_device"]):
        reasons.append("fingerprint_used_volatile_video_node")
    if expected.get("protocol") not in {None, "letterbox_v1"}:
        reasons.append("protocol_mismatch")
    if list(expected.get("canvas_hw") or []) and list(expected["canvas_hw"]) != live["canvas_hw"]:
        reasons.append("canvas_hw_mismatch")
    exp_pad = expected.get("letterbox_pad_y", expected.get("expected_pad_y"))
    exp_h = expected.get("letterbox_content_h", expected.get("expected_content_h"))
    if exp_pad is not None and abs(int(live["letterbox_pad_y"]) - int(exp_pad)) > LETTERBOX_PAD_TOL_PX:
        reasons.append("letterbox_pad_mismatch")
    if exp_h is not None and abs(int(live["letterbox_content_h"]) - int(exp_h)) > LETTERBOX_PAD_TOL_PX:
        reasons.append("letterbox_content_mismatch")
    if expected.get("source_size") and list(expected["source_size"]) != live["source_size"]:
        if list(expected.get("source_size") or []) != list(live["source_size"]):
            reasons.append("source_size_mismatch")
    return {
        "ok": not reasons,
        "reason": None if not reasons else reasons[0],
        "reasons": reasons,
        "live": live,
        "expected": dict(expected),
    }


def inspect_letterbox_canvas(frame: np.ndarray) -> dict[str, Any]:
    """Measure black pad bands on a planning canvas."""

    rgb = _as_rgb(frame)
    height, width = rgb.shape[:2]
    gray = rgb.mean(axis=-1)
    row_mean = gray.mean(axis=1)
    content = np.flatnonzero(row_mean > 2.0)
    if content.size == 0:
        pad_y = int(height)
        content_h = 0
    else:
        pad_y = int(content[0])
        content_h = int(content[-1] - content[0] + 1)
    geom = _expected_letterbox()
    pad_ok = abs(pad_y - int(geom.pad_y)) <= LETTERBOX_PAD_TOL_PX
    height_ok = abs(content_h - int(geom.content_height)) <= LETTERBOX_PAD_TOL_PX
    planning = is_planning_224_hw((height, width))
    return {
        "canvas_hw": [int(height), int(width)],
        "planning_224": bool(planning),
        "pad_y": pad_y,
        "content_h": content_h,
        "expected_pad_y": int(geom.pad_y),
        "expected_content_h": int(geom.content_height),
        "letterbox_ok": bool(planning and pad_ok and height_ok),
        "protocol": "letterbox_v1",
        "source_size": [int(EXPECTED_SOURCE_SIZE[0]), int(EXPECTED_SOURCE_SIZE[1])],
    }


def frame_is_blank_or_stale(
    frames: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Detect uninitialized / saturated / frozen camera buffers."""

    if len(frames) < 1:
        return {"stale": True, "blank": True, "reason": "no_frames"}
    blank = False
    identical = 0
    details: list[dict[str, Any]] = []
    prev = None
    for index, frame in enumerate(frames):
        rgb = _as_rgb(frame)
        canvas = inspect_letterbox_canvas(rgb)
        y0 = max(0, int(canvas["pad_y"]))
        y1 = min(rgb.shape[0], y0 + max(1, int(canvas["content_h"])))
        content = rgb[y0:y1] if y1 > y0 else rgb
        std = float(content.std())
        sample = content[::4, ::4] if min(content.shape[:2]) >= 8 else content
        unique = int(len(np.unique(sample.reshape(-1, 3), axis=0)))
        is_blank = std <= PRE_MOTION_BLANK_STD_MAX and unique <= PRE_MOTION_BLANK_UNIQUE_MAX
        blank = blank or is_blank
        if prev is not None and np.array_equal(rgb, prev):
            identical += 1
        details.append(
            {
                "index": index,
                "std": std,
                "n_unique": unique,
                "blank": is_blank,
            }
        )
        prev = rgb
    # Identical rich scenes are a static workspace.  Frozen buffers that
    # we refuse are blank / saturated or a repeated empty canvas.
    stale = bool(identical >= 1 and blank)
    reason = None
    if blank or stale:
        reason = "stale_or_blank_frame"
    return {
        "stale": bool(stale),
        "blank": bool(blank),
        "identical_consecutive": int(identical),
        "reason": reason,
        "frames": details,
    }


def _as_rgb(frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {getattr(rgb, 'shape', None)}")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _identity_thresholds(setup: Mapping[str, Any] | None) -> dict[str, float]:
    raw = {}
    if isinstance(setup, Mapping):
        thresholds = setup.get("thresholds")
        if isinstance(thresholds, Mapping):
            raw = dict(thresholds)
    return {
        "whole_frame_mad_max": float(
            raw.get(
                "coarse_identity_whole_frame_mad",
                DEFAULT_REGISTRATION_THRESHOLDS["coarse_identity_whole_frame_mad"],
            )
        ),
        "corner_mad_max": float(
            raw.get(
                "coarse_identity_corner_mad",
                DEFAULT_REGISTRATION_THRESHOLDS["coarse_identity_corner_mad"],
            )
        ),
        "min_hot_corners": float(
            raw.get(
                "coarse_identity_min_hot_corners",
                DEFAULT_REGISTRATION_THRESHOLDS["coarse_identity_min_hot_corners"],
            )
        ),
    }


def _roi_patch_mad(
    live: np.ndarray,
    reference: np.ndarray,
    xyxy: Sequence[int],
) -> float | None:
    x0, y0, x1, y1 = (int(v) for v in xyxy)
    if x1 <= x0 or y1 <= y0:
        return None
    if live.shape != reference.shape:
        return None
    height, width = live.shape[:2]
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    a = live[y0:y1, x0:x1].astype(np.int16)
    b = reference[y0:y1, x0:x1].astype(np.int16)
    return float(np.mean(np.abs(a - b)))


def evaluate_pre_motion_identity(
    live_column_frames: Sequence[np.ndarray],
    live_wrist_frames: Sequence[np.ndarray],
    *,
    reference_column: np.ndarray | None,
    reference_wrist: np.ndarray | None,
    setup_manifest: Mapping[str, Any] | None = None,
    gripper_roi_xyxy: Sequence[int] | None = None,
    n_frames: int = PRE_MOTION_N_FRAMES,
) -> dict[str, Any]:
    """Coarse role / identity / canvas / ROI match.  Fail closed on unknown."""

    evidence: dict[str, Any] = {
        "before_first_command": True,
        "n_frames_required": int(n_frames),
        "n_column": len(live_column_frames),
        "n_wrist": len(live_wrist_frames),
    }
    if reference_column is None or reference_wrist is None:
        return _refuse("unknown_missing_setup_reference", evidence)
    if len(live_column_frames) < int(n_frames) or len(live_wrist_frames) < int(n_frames):
        return _refuse("unknown_insufficient_frames", evidence)

    try:
        ref_col = _as_rgb(reference_column)
        ref_wri = _as_rgb(reference_wrist)
        live_cols = [_as_rgb(frame) for frame in live_column_frames[: int(n_frames)]]
        live_wris = [_as_rgb(frame) for frame in live_wrist_frames[: int(n_frames)]]
    except ValueError as exc:
        return _refuse(f"unknown_bad_frame:{exc}", evidence)

    thresholds = _identity_thresholds(setup_manifest)
    evidence["thresholds"] = dict(thresholds)
    evidence["not_relaxed_mad"] = True

    for name, frames, reference in (
        ("column", live_cols, ref_col),
        ("wrist", live_wris, ref_wri),
    ):
        if any(frame.shape != reference.shape for frame in frames):
            return _refuse("crop_or_letterbox_mismatch", {**evidence, "shape_view": name})
        if not is_planning_224_hw(reference.shape):
            return _refuse("unknown_reference_not_planning_224", evidence)

    ref_col_canvas = inspect_letterbox_canvas(ref_col)
    ref_wri_canvas = inspect_letterbox_canvas(ref_wri)
    if not ref_col_canvas["letterbox_ok"] or not ref_wri_canvas["letterbox_ok"]:
        return _refuse("unknown_reference_letterbox", evidence)

    col_stale = frame_is_blank_or_stale(live_cols)
    wri_stale = frame_is_blank_or_stale(live_wris)
    evidence["stale"] = {"column": col_stale, "wrist": wri_stale}
    if col_stale["reason"] or wri_stale["reason"]:
        return _refuse("stale_or_blank_frame", evidence)

    column_canvas = [inspect_letterbox_canvas(frame) for frame in live_cols]
    wrist_canvas = [inspect_letterbox_canvas(frame) for frame in live_wris]
    evidence["canvas"] = {"column": column_canvas, "wrist": wrist_canvas}
    if any(not row["letterbox_ok"] for row in (*column_canvas, *wrist_canvas)):
        return _refuse("crop_or_letterbox_mismatch", evidence)

    fingerprints = {
        "column": camera_role_fingerprint(live_cols[-1], role="column"),
        "wrist": camera_role_fingerprint(live_wris[-1], role="wrist"),
    }
    evidence["role_fingerprints"] = fingerprints
    stored = None
    if isinstance(setup_manifest, Mapping):
        stored = (setup_manifest.get("camera_identity_reference") or {}).get(
            "role_fingerprints"
        )
    if isinstance(stored, Mapping):
        for role in ("column", "wrist"):
            expected = stored.get(role)
            if not isinstance(expected, Mapping):
                continue
            checked = verify_camera_role_fingerprint(
                live_cols[-1] if role == "column" else live_wris[-1],
                expected,
                role=role,
            )
            evidence[f"{role}_fingerprint"] = checked
            if not checked["ok"]:
                return _refuse("crop_or_letterbox_mismatch", evidence)

    ident_kwargs = {
        "whole_frame_mad_max": thresholds["whole_frame_mad_max"],
        "corner_mad_max": thresholds["corner_mad_max"],
        "min_hot_corners": int(thresholds["min_hot_corners"]),
    }
    column_vs_col = [
        assess_camera_identity(frame, ref_col, **ident_kwargs) for frame in live_cols
    ]
    wrist_vs_wri = [
        assess_camera_identity(frame, ref_wri, **ident_kwargs) for frame in live_wris
    ]
    column_vs_wri = [
        assess_camera_identity(frame, ref_wri, **ident_kwargs) for frame in live_cols
    ]
    wrist_vs_col = [
        assess_camera_identity(frame, ref_col, **ident_kwargs) for frame in live_wris
    ]
    evidence["identity"] = {
        "column_vs_column_ref": column_vs_col,
        "wrist_vs_wrist_ref": wrist_vs_wri,
        "column_vs_wrist_ref": column_vs_wri,
        "wrist_vs_column_ref": wrist_vs_col,
    }

    mad_max = thresholds["whole_frame_mad_max"]

    def _matches(row: Mapping[str, Any]) -> bool:
        mad = row.get("whole_frame_mad")
        return mad is not None and float(mad) <= mad_max and not row["camera_identity_break"]

    def _mismatches(row: Mapping[str, Any]) -> bool:
        mad = row.get("whole_frame_mad")
        return mad is not None and float(mad) > mad_max

    swapped = any(
        _matches(vs_other) and _mismatches(vs_own)
        for vs_other, vs_own in zip(column_vs_wri, column_vs_col)
    ) or any(
        _matches(vs_other) and _mismatches(vs_own)
        for vs_other, vs_own in zip(wrist_vs_col, wrist_vs_wri)
    )
    if swapped:
        return _refuse("camera_role_swap", evidence)

    if any(row["camera_identity_break"] for row in (*column_vs_col, *wrist_vs_wri)):
        reason = "whole_frame_identity_break"
        if any(
            int(row.get("n_hot_corners") or 0) >= int(thresholds["min_hot_corners"])
            for row in (*column_vs_col, *wrist_vs_wri)
        ):
            reason = "hot_corner_identity_break"
        return _refuse(reason, evidence)

    roi = gripper_roi_xyxy
    if roi is None and isinstance(setup_manifest, Mapping):
        gripper = setup_manifest.get("gripper_roi")
        if isinstance(gripper, Mapping):
            roi = gripper.get("xyxy")
        elif isinstance(gripper, (list, tuple)):
            roi = gripper
        ref_block = setup_manifest.get("camera_identity_reference")
        if roi is None and isinstance(ref_block, Mapping):
            roi = ref_block.get("gripper_roi_xyxy")
    if roi is not None:
        roi_mads = [_roi_patch_mad(frame, ref_col, roi) for frame in live_cols]
        evidence["roi"] = {"xyxy": list(roi), "column_patch_mads": roi_mads}
        if any(
            mad is None or mad > thresholds["whole_frame_mad_max"] for mad in roi_mads
        ):
            return _refuse("roi_identity_mismatch", evidence)

    return {
        "pre_motion_identity_valid": True,
        "ok": True,
        "reason": None,
        "attempt_consumed": False,
        "first_command_blocked": False,
        "before_first_command": True,
        "visual_truth": None,
        "evidence": evidence,
        "role_fingerprints": fingerprints,
    }


def _refuse(reason: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pre_motion_identity_valid": False,
        "ok": False,
        "reason": str(reason),
        "attempt_consumed": False,
        "first_command_blocked": True,
        "before_first_command": True,
        "visual_truth": False,
        "evidence": dict(evidence),
    }


def evaluate_seek_frame_identity(
    column: np.ndarray,
    wrist: np.ndarray,
    *,
    reference_column: np.ndarray,
    reference_wrist: np.ndarray | None = None,
    setup_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One-frame seek identity.  Break ⇒ no visual truth, ramped restore."""

    if reference_wrist is None:
        thresholds = _identity_thresholds(setup_manifest)
        try:
            live_c = _as_rgb(column)
            live_w = _as_rgb(wrist)
            ref_c = _as_rgb(reference_column)
        except ValueError as exc:
            return _refuse(f"unknown_bad_frame:{exc}", {"before_first_command": False})
        blank = frame_is_blank_or_stale([live_c])
        blank_w = frame_is_blank_or_stale([live_w])
        canvas = inspect_letterbox_canvas(live_c)
        canvas_w = inspect_letterbox_canvas(live_w)
        ident = assess_camera_identity(
            live_c,
            ref_c,
            whole_frame_mad_max=thresholds["whole_frame_mad_max"],
            corner_mad_max=thresholds["corner_mad_max"],
            min_hot_corners=int(thresholds["min_hot_corners"]),
        )
        evidence = {
            "before_first_command": False,
            "stale": {"column": blank, "wrist": blank_w},
            "canvas": {"column": canvas, "wrist": canvas_w},
            "identity": {"column_vs_column_ref": [ident]},
        }
        if blank["reason"] or blank_w["reason"]:
            return _refuse("stale_or_blank_frame", evidence)
        if not canvas["letterbox_ok"] or not canvas_w["letterbox_ok"]:
            return _refuse("crop_or_letterbox_mismatch", evidence)
        if ident["camera_identity_break"]:
            reason = "whole_frame_identity_break"
            if int(ident.get("n_hot_corners") or 0) >= int(thresholds["min_hot_corners"]):
                reason = "hot_corner_identity_break"
            return _refuse(reason, evidence)
        return {
            "pre_motion_identity_valid": True,
            "ok": True,
            "reason": None,
            "attempt_consumed": True,
            "first_command_blocked": False,
            "before_first_command": False,
            "visual_truth": True,
            "evidence": evidence,
        }
    verdict = evaluate_pre_motion_identity(
        [column],
        [wrist],
        reference_column=reference_column,
        reference_wrist=reference_wrist,
        setup_manifest=setup_manifest,
        n_frames=1,
    )
    verdict["before_first_command"] = False
    verdict["attempt_consumed"] = True
    verdict["first_command_blocked"] = False
    return verdict


def load_planning_rgb(path: str | Path) -> np.ndarray:
    """Load a planning-canvas RGB frame from jpg/png/npy."""

    target = Path(path)
    if target.suffix.lower() == ".npy":
        rgb = np.asarray(np.load(target))
        return _as_rgb(rgb)
    import cv2

    bgr = cv2.imread(str(target), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read reference frame {target}")
    return _as_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def load_identity_reference_from_setup(
    setup_manifest: Mapping[str, Any] | None,
) -> dict[str, np.ndarray] | None:
    """Read column/wrist setup reference frames.  Missing → None (unknown)."""

    if not isinstance(setup_manifest, Mapping):
        return None
    block = setup_manifest.get("camera_identity_reference")
    if not isinstance(block, Mapping):
        return None
    column = block.get("column")
    wrist = block.get("wrist")
    if isinstance(column, np.ndarray) and isinstance(wrist, np.ndarray):
        return {"column": _as_rgb(column), "wrist": _as_rgb(wrist)}
    if not column or not wrist:
        return None
    return {
        "column": load_planning_rgb(column),
        "wrist": load_planning_rgb(wrist),
    }


def grab_pre_motion_frames(
    observe,
    *,
    n_frames: int = PRE_MOTION_N_FRAMES,
) -> dict[str, list[np.ndarray]]:
    """Read N planning frames.  Caller must not have written a Goal yet."""

    columns: list[np.ndarray] = []
    wrists: list[np.ndarray] = []
    for _ in range(int(n_frames)):
        observation = observe()
        frames = getattr(observation, "frames", observation)
        if not isinstance(frames, Mapping):
            raise ValueError("observe() did not return column/wrist frames")
        columns.append(_as_rgb(frames["column"]))
        wrists.append(_as_rgb(frames["wrist"]))
    return {"column": columns, "wrist": wrists}
