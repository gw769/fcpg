"""Planner-facing glue for the recursive observer (M4/M5/M6).

Kept out of ``so101_realtime_planner`` so the live class does not grow another
few hundred lines of filter math.  Every entry point here is a no-op unless
the corresponding config flag is on, which is how ``:8877`` stays untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from fcpg.observer_bundle import LETTERBOX_FOV45_FOCAL_PX
from fcpg.observer_excitation import d_optimal_excitation
from fcpg.observer_filter import InformationObserver, motion_type
from fcpg.observer_shadow import ShadowLog
from fcpg.observer_supervisor import (
    DEFAULT_CLOSE_SIGMA_M,
    DropZone3D,
    register_drop_zone_3d,
    reproject_drop_zone,
)
from fcpg.observer_tip_gate import (
    DEFAULT_REJECT_RADIUS_PX,
    DEFAULT_STALE_AGE_FRAMES,
    DEFAULT_STALE_DQ_DEG,
    FingertipHardRejectGate,
)
from fcpg.observer_tips import color_seed_tips, measure_motion_tips

#: Measured L–R span vs FK-predicted span. Open jaws at ~31° project much
#: wider than a 15–17 px knuckle cluster; that cluster is not a tip update.
TIP_SPAN_RATIO_MIN = 0.45
TIP_SPAN_RATIO_MAX = 2.2
TIP_SPAN_MIN_PREDICTED_PX = 8.0


def tip_span_inconsistent(
    measured_px: np.ndarray, predicted_px: np.ndarray
) -> bool:
    """True when the measured jaw width cannot be the predicted tips."""

    measured = np.asarray(measured_px, dtype=np.float64)
    predicted = np.asarray(predicted_px, dtype=np.float64)
    if measured.shape != (3, 2) or predicted.shape != (3, 2):
        return False
    meas_span = float(np.linalg.norm(measured[2] - measured[1]))
    pred_span = float(np.linalg.norm(predicted[2] - predicted[1]))
    if not np.isfinite(meas_span) or not np.isfinite(pred_span):
        return False
    if pred_span < TIP_SPAN_MIN_PREDICTED_PX:
        return False
    ratio = meas_span / pred_span
    return ratio < TIP_SPAN_RATIO_MIN or ratio > TIP_SPAN_RATIO_MAX


def observer_from_bundle(
    path: str | Path | None,
    *,
    session: str | None = None,
    calibration_sha256: str | None = None,
) -> InformationObserver:
    """Seed ``T_cb`` from the task-data bundle; keep the unidentified FOV ``K``.

    A cold identity pose rejects the first metal-tip updates on the chi-square
    gate, so the close supervisor would never arm.  The bundle mean is the
    operating point M1 actually identified.  Prior std stays wide enough that
    close still requires accepted updates (0.08 m > 0.012 m).
    """

    if path is None:
        return InformationObserver()
    bundle_path = Path(path)
    if not bundle_path.is_file():
        return InformationObserver()
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    fit = payload.get("fit") or {}
    groups = list(fit.get("extrinsics_cam_from_base") or [])
    chosen = None
    for group in groups:
        key = group.get("group_key") or {}
        if calibration_sha256 and key.get("live_anchor_calibration_sha256") != calibration_sha256:
            continue
        if session and key.get("session") != session:
            continue
        chosen = group
    if chosen is None:
        for group in groups:
            key = group.get("group_key") or {}
            if session and key.get("session") == session:
                chosen = group
        if chosen is None and groups:
            chosen = groups[-1]
    kwargs: dict[str, Any] = {
        "fx": LETTERBOX_FOV45_FOCAL_PX,
        "fy": LETTERBOX_FOV45_FOCAL_PX,
        "cx": 112.0,
        "cy": 112.0,
        "prior_rot_std_rad": 0.08,
        "prior_t_std_m": 0.08,
    }
    if chosen is not None:
        matrix = np.asarray(chosen["matrix"], dtype=np.float64)
        kwargs["initial_rotvec"] = Rotation.from_matrix(matrix[:3, :3]).as_rotvec()
        kwargs["initial_translation"] = matrix[:3, 3]
    observer = InformationObserver(**kwargs)
    offsets = np.asarray(fit.get("joint_offsets_normalized") or [], dtype=np.float64)
    if offsets.size:
        packed = observer.state.joint_offsets.copy()
        packed[: min(packed.size, offsets.size)] = offsets[: packed.size]
        observer.state.joint_offsets = packed
    return observer


_OBSERVER_ADJACENT_HISTORY_ATTRS = (
    "_observer_prev_gray",
    "_observer_prev_anchors_xyz",
    "_observer_prev_jacobian",
    "_observer_prev_association_px",
    "_observer_prev_joints",
)


def clear_observer_adjacent_history(planner: Any) -> None:
    """Cut adjacent-frame motion association after a live-transform re-anchor.

    The recursive observer's accepted state and information stay intact.
    Pending single-tip measurements are dropped so a pre-pulse tip cannot
    stack with post-pulse evidence from a different association frame.
    """

    for name in _OBSERVER_ADJACENT_HISTORY_ATTRS:
        setattr(planner, name, None)
    planner._observer_pending_measurements = []


def attach_observer(planner: Any) -> None:
    """Install the 3-D observer, shadow log and drop slot on a live planner."""

    if getattr(planner, "_recursive_observer", None) is not None:
        return
    config = getattr(planner, "config", None)
    planner._recursive_observer = observer_from_bundle(
        getattr(config, "observer_bundle_path", None),
        session="v12_1_live_deploy_20260816",
    )
    planner._observer_shadow = ShadowLog()
    planner._observer_drop_zone: DropZone3D | None = None
    planner._observer_prev_joints: np.ndarray | None = None
    planner._observer_prev_gray: np.ndarray | None = None
    planner._observer_prev_anchors_xyz: np.ndarray | None = None
    planner._observer_prev_jacobian: np.ndarray | None = None
    planner._observer_motion_records: list[tuple[str, np.ndarray]] = []
    planner._fingertip_gate = FingertipHardRejectGate(
        reject_radius_px=float(
            getattr(config, "fingertip_reject_radius_px", DEFAULT_REJECT_RADIUS_PX)
        ),
        stale_age_frames=int(
            getattr(config, "fingertip_stale_age_frames", DEFAULT_STALE_AGE_FRAMES)
        ),
        stale_dq_deg=float(
            getattr(config, "fingertip_stale_dq_deg", DEFAULT_STALE_DQ_DEG)
        ),
    )
    planner._fingertip_last_result = None
    persisted = _load_persisted_drop(getattr(config, "observer_drop_3d_path", None))
    if persisted is not None:
        planner._observer_drop_zone = persisted


def shadow_enabled(planner: Any) -> bool:
    config = getattr(planner, "config", None)
    return bool(getattr(config, "recursive_observer_shadow", False))


def close_sigma_limit(planner: Any) -> float | None:
    config = getattr(planner, "config", None)
    value = getattr(config, "observer_close_sigma_m", None)
    if value is None:
        return None
    return float(value)


def excitation_enabled(planner: Any) -> bool:
    config = getattr(planner, "config", None)
    return bool(getattr(config, "observer_active_excitation", False))


def _rgb_to_gray(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    return (
        0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    ).astype(np.uint8)


def _load_persisted_drop(path: str | Path | None) -> DropZone3D | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    xyz = payload.get("xyz_base")
    if xyz is None:
        return None
    covariance = payload.get("covariance")
    if covariance is None:
        covariance = np.eye(3, dtype=np.float64) * 0.01
    try:
        return DropZone3D(
            xyz_base=np.asarray(xyz, dtype=np.float64),
            covariance=np.asarray(covariance, dtype=np.float64),
            registered_round_id=str(payload.get("registered_round_id") or "persisted"),
            source=str(payload.get("source") or "persisted"),
        )
    except (TypeError, ValueError):
        return None


def persist_drop_zone(planner: Any, drop: DropZone3D) -> None:
    path = getattr(getattr(planner, "config", None), "observer_drop_3d_path", None)
    if path is None:
        return
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(drop.to_dict(), indent=2) + "\n", encoding="utf-8"
    )


def _require_tip_gate(planner: Any) -> FingertipHardRejectGate:
    gate = getattr(planner, "_fingertip_gate", None)
    if gate is not None:
        return gate
    config = getattr(planner, "config", None)
    gate = FingertipHardRejectGate(
        reject_radius_px=float(
            getattr(config, "fingertip_reject_radius_px", DEFAULT_REJECT_RADIUS_PX)
        ),
        stale_age_frames=int(
            getattr(config, "fingertip_stale_age_frames", DEFAULT_STALE_AGE_FRAMES)
        ),
        stale_dq_deg=float(
            getattr(config, "fingertip_stale_dq_deg", DEFAULT_STALE_DQ_DEG)
        ),
    )
    planner._fingertip_gate = gate
    return gate


def _tip_gate_diagnostics(planner: Any) -> dict[str, Any]:
    result = getattr(planner, "_fingertip_last_result", None)
    if result is None:
        return {"decision": "unavailable", "reason": "no_update", "available": False}
    return result.to_dict()


def _motion_candidate_px(tips: Any) -> tuple[np.ndarray | None, str | None]:
    """Fresh motion locks only. Held / colour seeds are not candidates."""

    candidate = np.full((3, 2), np.nan, dtype=np.float64)
    reasons = []
    for index, source in enumerate(tips.source):
        if source == "motion" and bool(tips.valid[index]):
            candidate[index] = tips.points_px[index]
        else:
            reason = (tips.diagnostics or {}).get(f"anchor_{index}_reason")
            if reason is not None:
                reasons.append(str(reason))
    if not np.any(np.isfinite(candidate)):
        if any(item.endswith("_hold") for item in reasons):
            return None, "predicted_flow_too_small_hold"
        if "predicted_flow_too_small" in reasons:
            return None, "predicted_flow_too_small"
        if "chi2_gate" in reasons:
            return None, "chi2_gate"
        if "no_candidate" in reasons:
            return None, "no_candidate"
        return None, None
    return candidate, None


def measure_live_column_tips(
    planner: Any,
    *,
    frame_rgb: np.ndarray,
    anchors_base_xyz: np.ndarray,
    joint_jacobian: np.ndarray,
    metal_seed_px: np.ndarray | None,
    joints: np.ndarray | None = None,
    association_offset_px: np.ndarray | None = None,
    association_predicted_px: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Motion-consistent tips. Colour is a search seed, never a measurement.

    A candidate that survives motion-consistency still has to pass the
    hard FK-transport gate.  Idle frames (no informative predicted flow)
    hold the last accepted lock; they do not fall back to metal.
    Only a *fresh* accept is returned for the observer update.
    """

    observer: InformationObserver = planner._recursive_observer
    gate = _require_tip_gate(planner)
    gray = _rgb_to_gray(frame_rgb)
    prev_gray = getattr(planner, "_observer_prev_gray", None)
    prev_anchors = getattr(planner, "_observer_prev_anchors_xyz", None)
    prev_jacobian = getattr(planner, "_observer_prev_jacobian", None)
    seed = color_seed_tips(frame_rgb)
    if seed is None and metal_seed_px is not None:
        seed = np.asarray(metal_seed_px, dtype=np.float64)
    measured = None
    valid = None
    n_motion = 0
    candidate = None
    candidate_reason = "no_previous_frame"
    predicted_now, _ = observer.project(anchors_base_xyz, joint_jacobian)
    association_offset = np.zeros(2, dtype=np.float64)
    if association_offset_px is not None:
        proposed_offset = np.asarray(association_offset_px, dtype=np.float64)
        if proposed_offset.shape == (2,) and np.all(np.isfinite(proposed_offset)):
            association_offset = proposed_offset
    associated_now = predicted_now + association_offset[None]
    if association_predicted_px is not None:
        proposed_prediction = np.asarray(
            association_predicted_px, dtype=np.float64
        )
        if (
            proposed_prediction.shape == predicted_now.shape
            and np.all(np.isfinite(proposed_prediction))
        ):
            associated_now = proposed_prediction
    if prev_gray is not None and prev_anchors is not None:
        predicted0, _ = observer.project(prev_anchors, prev_jacobian)
        associated0 = getattr(planner, "_observer_prev_association_px", None)
        if associated0 is None:
            associated0 = predicted0 + association_offset[None]
        else:
            associated0 = np.asarray(associated0, dtype=np.float64)
            if (
                associated0.shape != predicted0.shape
                or not np.all(np.isfinite(associated0))
            ):
                associated0 = predicted0 + association_offset[None]
        motion_seed = seed
        if association_predicted_px is not None and seed is not None:
            # A colour peak on the box/table used to drag the DIS search tens
            # of pixels away even when pulse+affine transport had a calibrated
            # cold-start prediction. Colour remains only a local search seed:
            # discard each peak unless it is near either endpoint prediction.
            proposed_seed = np.asarray(seed, dtype=np.float64).copy()
            if proposed_seed.shape == associated0.shape:
                distance0 = np.linalg.norm(proposed_seed - associated0, axis=1)
                distance1 = np.linalg.norm(proposed_seed - associated_now, axis=1)
                seed_radius = 2.0 * float(gate.reject_radius_px)
                proposed_seed[np.minimum(distance0, distance1) > seed_radius] = np.nan
                motion_seed = proposed_seed
        tips = measure_motion_tips(
            prev_gray,
            gray,
            associated0,
            associated_now - associated0,
            seed_px=motion_seed,
            held_px=gate.last_accepted_px,
            min_predicted_flow_px=1.5,
        )
        n_motion = tips.n_motion
        candidate, candidate_reason = _motion_candidate_px(tips)
        if n_motion and candidate is not None and tip_span_inconsistent(
            candidate, associated_now
        ):
            candidate = None
            n_motion = 0
            candidate_reason = "tip_span_inconsistent"
    gated = gate.step(
        predicted_px=associated_now,
        joints=joints,
        candidate_px=candidate,
        candidate_reason=candidate_reason,
    )
    planner._fingertip_last_result = gated
    # Fresh accepts only.  A hold is published in diagnostics; feeding it
    # to the filter would treat a stale lock as a new measurement.
    if gated.fresh:
        measured = gated.points_px
        valid = gated.valid
        n_motion = int(np.sum(valid))
    else:
        measured = None
        valid = None
        n_motion = 0
    planner._observer_prev_gray = gray
    planner._observer_prev_anchors_xyz = np.asarray(
        anchors_base_xyz, dtype=np.float64
    ).copy()
    planner._observer_prev_jacobian = np.asarray(
        joint_jacobian, dtype=np.float64
    ).copy()
    planner._observer_prev_association_px = associated_now.copy()
    return measured, valid, n_motion


def shadow_step(
    planner: Any,
    *,
    round_id: str,
    phase: str | None,
    joints: np.ndarray,
    anchors_base_xyz: np.ndarray,
    joint_jacobian: np.ndarray,
    measured_px: np.ndarray | None,
    measured_valid: np.ndarray | None,
    affine_tips_px: np.ndarray | None,
    kf2d_offset_px: np.ndarray | None,
    n_motion_tips: int = 0,
    frame_rgb: np.ndarray | None = None,
) -> dict[str, Any]:
    """One predict/update cycle.  Returns a JSON-ready diagnostics dict."""

    attach_observer(planner)
    observer: InformationObserver = planner._recursive_observer
    prev = planner._observer_prev_joints
    if prev is not None:
        observer.predict()
        planner._observer_motion_records.append(
            (motion_type(np.asarray(joints) - np.asarray(prev)), np.zeros((0, 0)))
        )
    if frame_rgb is not None:
        association_offset = None
        association_prediction = None
        association_source = None
        updates = int(getattr(observer.state, "updates", 0) or 0)
        if updates < 3 and kf2d_offset_px is not None:
            proposed_offset = np.asarray(kf2d_offset_px, dtype=np.float64)
            config = getattr(planner, "config", None)
            maximum_offset = float(
                getattr(config, "column_extrinsic_max_offset_px", 60.0)
            )
            if (
                proposed_offset.shape == (2,)
                and np.all(np.isfinite(proposed_offset))
                and float(np.linalg.norm(proposed_offset)) <= maximum_offset
            ):
                # A pulse/2-D extrinsic estimate is association authority only:
                # it transports the cold-start search window after a camera
                # shift, while the recursive observer still updates against
                # the raw measured pixels and must pass its own chi-square gate.
                association_offset = proposed_offset
                affine = None
                if affine_tips_px is not None:
                    proposed_affine = np.asarray(
                        affine_tips_px, dtype=np.float64
                    )
                    if proposed_affine.ndim == 3:
                        proposed_affine = proposed_affine[-1]
                    if (
                        proposed_affine.shape == (len(anchors_base_xyz), 2)
                        and np.all(np.isfinite(proposed_affine))
                    ):
                        affine = proposed_affine
                if affine is not None:
                    # The bundle pose can be stale after a camera re-aim. Use
                    # the calibrated affine FK transport only to associate the
                    # cold-start motion tips; the raw pixels still update the
                    # 3-D observer and its chi-square gate remains authoritative.
                    association_prediction = affine + proposed_offset[None]
                    association_source = "affine_plus_2d_offset"
                else:
                    association_source = "observer_plus_2d_offset"
        measured_px, measured_valid, n_motion_tips = measure_live_column_tips(
            planner,
            frame_rgb=frame_rgb,
            anchors_base_xyz=anchors_base_xyz,
            joint_jacobian=joint_jacobian,
            metal_seed_px=measured_px,
            joints=joints,
            association_offset_px=association_offset,
            association_predicted_px=association_prediction,
        )
    else:
        gate = _require_tip_gate(planner)
        predicted_now, _ = observer.project(anchors_base_xyz, joint_jacobian)
        gated = gate.step(
            predicted_px=predicted_now,
            joints=joints,
            candidate_px=measured_px,
            candidate_reason=None if measured_px is None else "supplied_measurement",
        )
        planner._fingertip_last_result = gated
        if gated.fresh:
            measured_px = gated.points_px
            measured_valid = gated.valid
        else:
            measured_px = None
            measured_valid = None
            n_motion_tips = 0
    update = None
    if measured_px is not None:
        points = np.asarray(anchors_base_xyz, dtype=np.float64)
        pixels = np.asarray(measured_px, dtype=np.float64)
        jacobian = np.asarray(joint_jacobian, dtype=np.float64)
        valid = (
            np.ones(len(points), dtype=bool)
            if measured_valid is None
            else np.asarray(measured_valid, dtype=bool)
        )
        valid = (
            valid
            & np.all(np.isfinite(points), axis=1)
            & np.all(np.isfinite(pixels), axis=1)
        )
        pending = list(
            getattr(planner, "_observer_pending_measurements", [])
        )
        if np.any(valid):
            pending.append(
                (
                    points[valid].copy(),
                    pixels[valid].copy(),
                    jacobian[valid].copy(),
                )
            )
        # A single fresh fingertip supplies only two equations and remains
        # deliberately insufficient for the 3-D observer. Accumulate fresh,
        # motion-certified points across distinct planning frames and perform
        # one stacked update as soon as at least two are available. Held or
        # colour-only points never enter this buffer.
        total = sum(len(row[0]) for row in pending)
        if total >= 2:
            stacked_points = np.concatenate([row[0] for row in pending], axis=0)
            stacked_pixels = np.concatenate([row[1] for row in pending], axis=0)
            stacked_jacobian = np.concatenate([row[2] for row in pending], axis=0)
            update = observer.update(
                stacked_points,
                stacked_pixels,
                stacked_jacobian,
            )
            # Never replay a rejected batch or count accepted evidence twice.
            pending = []
        else:
            update = observer.update(
                points,
                pixels,
                jacobian,
                valid=valid,
            )
        planner._observer_pending_measurements = pending
    planner._observer_prev_joints = np.asarray(joints, dtype=np.float64).copy()
    close_direction = anchors_base_xyz[2] - anchors_base_xyz[1]
    along = observer.along_close_sigma_m(anchors_base_xyz[0], close_direction)
    projected, depth = observer.project(anchors_base_xyz, joint_jacobian)
    tip_gate = _tip_gate_diagnostics(planner)
    planner._observer_shadow.append(
        round_id=round_id,
        phase=phase,
        observer=observer,
        update=update,
        observer_tips_px=projected,
        affine_tips_px=affine_tips_px,
        kf2d_offset_px=kf2d_offset_px,
        along_close_sigma_m=along,
        n_motion_tips=n_motion_tips,
        tip_gate=tip_gate,
    )
    return {
        "enabled": True,
        "accepted": False if update is None else bool(update.accepted),
        "residual_rms_px": None if update is None else update.residual_rms_px,
        "rejection_reason": None if update is None else update.rejection_reason,
        "trace_p": float(np.trace(observer.state.covariance())),
        "sigma": observer.state.sigma().tolist(),
        "along_close_sigma_m": along,
        "projected_tips_px": projected.tolist(),
        "projected_depth_m": depth.tolist(),
        "n_motion_tips": int(n_motion_tips),
        "pending_motion_measurements": sum(
            len(row[0])
            for row in getattr(planner, "_observer_pending_measurements", [])
        ),
        "tip_gate": tip_gate,
        "association_offset_px": (
            None
            if frame_rgb is None or association_offset is None
            else association_offset.tolist()
        ),
        "association_source": None if frame_rgb is None else association_source,
        "state": observer.state.to_dict(),
    }


def maybe_register_or_reproject_drop(
    planner: Any,
    *,
    centroid_px: np.ndarray | None,
    plane_z_base: float,
    round_id: str,
    measurement_source: str | None = None,
    observer_accepted: bool | None = None,
) -> dict[str, Any] | None:
    """Store a new 3-D drop once, or reproject the one already stored.

    A new zone is registered only from a live FrozenViT or open-vocab
    measurement after the observer has accepted updates. An unconverged prior
    cannot mint a drop, and a later peak cannot replace the stored identity.
    """

    observer = getattr(planner, "_recursive_observer", None)
    if observer is None:
        return None
    existing: DropZone3D | None = getattr(planner, "_observer_drop_zone", None)
    if existing is not None:
        return reproject_drop_zone(existing, observer)
    if centroid_px is None:
        return None
    if str(measurement_source or "") not in {"frozen_vit", "open_vocab"}:
        return None
    updates = int(getattr(getattr(observer, "state", None), "updates", 0) or 0)
    accepted = (
        bool(observer_accepted)
        if observer_accepted is not None
        else updates >= 3
    )
    if not accepted or updates < 3:
        return None
    try:
        drop = register_drop_zone_3d(
            observer, centroid_px, plane_z_base=plane_z_base, round_id=round_id
        )
    except ValueError:
        return None
    planner._observer_drop_zone = drop
    persist_drop_zone(planner, drop)
    return reproject_drop_zone(drop, observer)


def maybe_excitation(
    planner: Any,
    *,
    phase: str | None,
    jaws_holding: bool,
    anchors_base_xyz: np.ndarray,
    joint_jacobian: np.ndarray,
) -> dict[str, Any]:
    observer = getattr(planner, "_recursive_observer", None)
    if observer is None or not excitation_enabled(planner):
        return {"allowed": False, "reason": "disabled"}
    decision = d_optimal_excitation(
        observer,
        anchors_base_xyz,
        joint_jacobian,
        phase=phase,
        jaws_holding=jaws_holding,
    )
    return {
        "allowed": decision.allowed,
        "reason": decision.reason,
        "delta_q": None if decision.delta_q is None else decision.delta_q.tolist(),
        "predicted_logdet": decision.predicted_logdet,
    }
