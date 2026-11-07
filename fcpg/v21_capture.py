"""Per-view actual acquisition bookkeeping, with no hardware owner or side effects.

All I/O is injected by the established transport. A known camera delivery
failure may leave that view invalid; it cannot hide readback, frame geometry,
identity, or clock failures. Invalid slots contain masked zeros, never cached
pixels with new timestamps. The result alone does not authorize robot motion.
"""
from dataclasses import dataclass
from copy import deepcopy
from collections.abc import Mapping
import time
from typing import Callable

import numpy as np


VIEWS = ("wrist", "column")


class CameraDeliveryUnavailable(RuntimeError):
    """Explicit adapter signal for a known missing/disconnected expected camera.

    Convert an established transport error to this type only after establishing
    that its *expected* device is absent. An arbitrary RuntimeError, changed
    device identity, or wrong geometry must never be converted to this class.
    """
    def __init__(self, reason: str, detail: str = ""):
        if reason not in ("camera_timeout", "camera_disconnected", "known_camera_device_missing"):
            raise ValueError("unknown camera delivery reason")
        self.reason = reason
        self.detail = str(detail)
        super().__init__(self.detail or reason)


@dataclass(frozen=True)
class ActualViewCapture:
    rgb: np.ndarray                       # [2,224,224,3], uint8; invalid slots zero
    q_at_image: np.ndarray                # [2,6], calibrated positions
    image_times_s: np.ndarray             # [2], float64 host read-finished times
    measurement_available: np.ndarray     # [2], bool; NOT semantic visibility
    q_position: np.ndarray                # [6], fresh overall actual readback
    q_time_s: float
    q_sequence_id: int
    views: list[dict]
    raw_frames: dict[str, np.ndarray]


def capture_actual_views(read_camera: Callable, read_q: Callable, *,
                         active_views=VIEWS, clock: Callable = time.monotonic,
                         camera_identity_check: Callable | None = None,
                         read_raw_frame: Callable | None = None,
                         maximum_bracket_s: float = .05,
                         maximum_pose_delta: float = .25) -> ActualViewCapture:
    """Capture independent actual views and always obtain current joint readback.

    The caller owns the existing camera/transport locks. ``read_camera(view)``
    must return letterbox-v1 RGB uint8 [224,224,3]; this function does not resize
    or reinterpret a camera. ``camera_identity_check(view)`` must return None,
    True, or an identity mapping whose ``view`` is the requested role. It is
    called before and after a delivered frame, and mapping results must match.
    Only explicit delivery exceptions (TimeoutError,
    ConnectionError, CameraDeliveryUnavailable) are downgraded to missing views.
    A plain RuntimeError from a transport is deliberately not classified here.

    ``read_q()`` is outside every recoverable-exception handler. Each successful
    camera read is enclosed by fresh actual q samples with strictly increasing
    per-capture sequence IDs. The frame is accepted only when those two poses
    differ by at most ``maximum_pose_delta`` on every joint. Host timing,
    including whether the former 50 ms reference was exceeded, remains in the
    record but does not substitute for measured pose stability. One additional
    overall actual q sample follows camera acquisition. ``read_raw_frame`` is optional
    diagnostic access to the raw frame delivered by that same camera callback;
    it must not open/read a second camera or substitute historical pixels.

    There is no DINO invocation here. The caller must skip encoding when
    ``measurement_available.any()`` is false, while preserving actual memory.
    Both views missing is an honest observation record, not a ready/safe plan.
    """
    active = tuple(active_views)
    if len(set(active)) != len(active) or any(view not in VIEWS for view in active):
        raise ValueError("active_views must be a unique subset of wrist,column")
    if not np.isfinite(maximum_bracket_s) or not 0 < maximum_bracket_s <= .05:
        raise ValueError("camera/readback timing reference must remain positive and <=50ms")
    if (isinstance(maximum_pose_delta, bool)
            or not isinstance(maximum_pose_delta, (int, float))
            or not np.isfinite(maximum_pose_delta)
            or not 0 < float(maximum_pose_delta) <= .25):
        raise ValueError("camera/readback pose stability tolerance must remain positive and <=0.25")
    if not callable(read_camera) or not callable(read_q) or not callable(clock):
        raise TypeError("camera, readback and clock callbacks must be callable")
    frames = np.zeros((2,224,224,3), np.uint8)
    image_q = np.zeros((2,6), np.float32)
    image_times = np.zeros(2, np.float64)
    valid = np.zeros(2, bool)
    records, raw_frames = [], {}
    last_clock = None
    readback_sequence_id = 0

    def stamp():
        nonlocal last_clock
        value = float(clock())
        if not np.isfinite(value) or value < 0 or (last_clock is not None and value < last_clock):
            raise RuntimeError("actual acquisition clock is nonfinite or regressed")
        last_clock = value
        return value

    def actual_q():
        nonlocal readback_sequence_id
        # Never catch a TimeoutError/ConnectionError originating at read_q.
        started = stamp()
        value = np.asarray(read_q())
        finished = stamp()
        if value.shape != (6,) or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise RuntimeError("actual readback must be a finite calibrated six-vector")
        with np.errstate(over="ignore", invalid="ignore"):
            value = value.astype(np.float32, copy=True)
        if not np.isfinite(value).all():
            raise RuntimeError("actual readback overflows calibrated position storage")
        readback_sequence_id += 1
        return value, dict(sequence_id=readback_sequence_id, read_started_s=started,
                           read_finished_s=finished, q_position=value.tolist(), fresh=True)

    def checked_identity(view):
        if camera_identity_check is None:
            return None
        identity = camera_identity_check(view)
        if identity is None or identity is True:
            return identity
        if not isinstance(identity, Mapping) or identity.get("view") != view:
            raise RuntimeError(f"camera {view} identity was not verified")
        result = deepcopy(dict(identity))
        if any(not isinstance(result.get(key), str) or not result[key]
               for key in ("view", "stable_device", "resolved_device")):
            raise RuntimeError(f"camera {view} identity record is incomplete")
        return result

    for index, view in enumerate(VIEWS):
        if view not in active:
            records.append(dict(view=view, available=False, reason="view_not_active",
                                image_time_s=None, q_sample_s=None, q_position=None))
            continue
        delivery_error = None
        identity_before = None
        try:
            identity_before = checked_identity(view)
        except CameraDeliveryUnavailable as exc:
            delivery_error = (exc.reason, str(exc))
        except TimeoutError as exc:
            delivery_error = ("camera_timeout", str(exc))
        except ConnectionError as exc:
            delivery_error = ("camera_disconnected", str(exc))
        if delivery_error is not None:
            ended = stamp()
            records.append(dict(view=view, available=False, reason=delivery_error[0],
                                error=delivery_error[1], read_started_s=None,
                                read_finished_s=ended, image_time_s=None,
                                q_sample_s=None, q_position=None,
                                q_before_image=None, camera_identity=None))
            continue
        # Both reads are outside the recoverable camera handlers: a bus failure
        # can never be relabelled as a missing image.
        q_before, q_before_record = actual_q()
        begin = stamp()
        try:
            frame = read_camera(view)
        except CameraDeliveryUnavailable as exc:
            delivery_error = (exc.reason, str(exc))
        except TimeoutError as exc:
            delivery_error = ("camera_timeout", str(exc))
        except ConnectionError as exc:
            delivery_error = ("camera_disconnected", str(exc))
        image_end = stamp()
        if delivery_error is not None:
            records.append(dict(view=view, available=False, reason=delivery_error[0],
                                error=delivery_error[1], read_started_s=begin,
                                read_finished_s=image_end, image_time_s=None,
                                q_sample_s=None, q_position=None,
                                q_before_image=q_before_record,
                                camera_identity=deepcopy(identity_before)))
            continue
        # Frame validation is intentionally outside the recoverable camera block.
        image = np.asarray(frame)
        if image.shape != (224,224,3) or image.dtype != np.uint8:
            raise RuntimeError(f"camera {view} delivered invalid RGB geometry/dtype; expected uint8 [224,224,3]")
        if read_raw_frame is not None:
            raw = np.asarray(read_raw_frame(view))
            if raw.ndim != 3 or raw.shape[2] != 3 or min(raw.shape[:2]) < 1 or raw.dtype != np.uint8:
                raise RuntimeError(f"camera {view} raw diagnostic frame has invalid geometry/dtype")
            raw_frames[view] = raw.copy()
        q, q_after_record = actual_q()
        identity_after = checked_identity(view)
        if isinstance(identity_before, Mapping) and identity_after != identity_before:
            raise RuntimeError(f"camera {view} identity changed during frame acquisition")
        delta = np.abs(q.astype(np.float64) - q_before.astype(np.float64))
        maximum_delta = float(delta.max())
        if maximum_delta > float(maximum_pose_delta) + 1e-12:
            raise RuntimeError("camera/readback pose bracket is not stable")
        bracket_elapsed = q_after_record["read_finished_s"] - q_before_record["read_started_s"]
        frames[index] = image.copy()
        image_q[index], image_times[index], valid[index] = q, image_end, True
        records.append(dict(view=view, available=True, reason=None,
                            read_started_s=begin, read_finished_s=image_end,
                            image_time_s=image_end, q_sample_s=q_after_record["read_finished_s"],
                            q_position=q.tolist(), identity_check_provided=camera_identity_check is not None,
                            camera_identity=deepcopy(identity_after),
                            camera_identity_checked_before_and_after=camera_identity_check is not None,
                            q_before_image=q_before_record, q_after_image=q_after_record,
                            readback_sequence_order=[q_before_record["sequence_id"],
                                                     q_after_record["sequence_id"]],
                            readback_sequence_scope="capture_actual_views_call",
                            readback_fresh=True, image_enclosed_by_readbacks=True, pose_stable=True,
                            pose_stability_tolerance=float(maximum_pose_delta),
                            maximum_pose_delta=maximum_delta,
                            acquisition_bracket_elapsed_s=bracket_elapsed,
                            timing_reference_s=float(maximum_bracket_s),
                            within_timing_reference=bracket_elapsed <= maximum_bracket_s + 1e-12,
                            q_association="post_actual_readback_with_pre_post_stability_envelope"))
    current_q, current_record = actual_q()
    current_time = current_record["read_finished_s"]
    return ActualViewCapture(frames, image_q, image_times, valid, current_q,
                             current_time, current_record["sequence_id"], records, raw_frames)


__all__ = ["ActualViewCapture", "CameraDeliveryUnavailable", "capture_actual_views"]
