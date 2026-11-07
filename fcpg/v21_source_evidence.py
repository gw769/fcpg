"""Exact, non-executable source RGB evidence owned by the existing capturer.

An image is identified by its original view/time, actual q association and
encoded features. Later endpoint pixels are never substituted for missing
source pixels. This module opens no camera, motor, runtime, or network.
"""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import torch

from .v21_encoder import DIM, PATCHES
from .v21_memory import VisualObservation


VIEWS = ("wrist", "column")
SCHEMA = "v21-source-visual-evidence-v1"
OBJECT_SCHEMA = "v21-object-token-provenance-v1"
_RECORD_FIELDS = ("source", "reason", "visible", "support_clipped", "aimable",
                  "body_truncated", "cog_px", "centroid_px", "axis_px", "length_px",
                  "width_px", "score", "area_px", "grasp_kind",
                  "partial_support_reacquisition")


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def _json(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return _array(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    return value


def array_sha256(value):
    """Bind dtype and shape as well as exact contiguous array bytes."""
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(dtype=value.dtype.str, shape=list(value.shape)),
                             sort_keys=True, separators=(",", ":")).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _compact_record(record):
    if not isinstance(record, dict):
        return None
    result = _json({key: record[key] for key in _RECORD_FIELDS if key in record})
    # Diagnostics with malformed geometry must not break persistence itself.
    def clean(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    return clean(result)


def object_token_provenance(vector, *, wrist_measurement_available,
                            encoder_enabled=True, policy_record=None, diagnostic_record=None,
                            image_time_s=None, q_at_image=None):
    """Describe the actual eight-vector without inferring a missing target."""
    if vector is None:
        token = np.zeros(8, np.float32)
        supplied = False
    else:
        token = _array(vector).reshape(-1)
        supplied = True
    if token.shape != (8,) or not np.isfinite(token).all() or token[0] not in (0, 1):
        raise ValueError("object token must be a finite eight-vector with binary availability")
    if token[0] == 0 and np.any(token != 0):
        raise ValueError("unavailable object token must be all zeros")
    available = bool(token[0])
    if available and wrist_measurement_available is False:
        raise ValueError("an available object token requires its actual wrist observation")
    policy = _compact_record(policy_record)
    diagnostic = _compact_record(diagnostic_record)
    source = None
    if available:
        if policy and policy.get("source") == "operator_colour_body":
            source = "registered-fish-body-cog"
            reason = "available_partial_colour_body" if policy.get("body_truncated") else "available_registered_colour_body"
        else:
            source = policy.get("source") if policy else None
            reason = "available_token_source_unverified" if source is None else "available_from_recorded_policy_source"
    elif not encoder_enabled:
        reason = "object_encoder_disabled"
    elif not wrist_measurement_available:
        reason = "current_wrist_observation_unavailable"
    elif not supplied:
        reason = "object_token_not_supplied"
    elif policy is None:
        reason = "registered_body_target_unavailable"
    elif policy.get("support_clipped"):
        reason = "policy_support_clipped"
    elif policy.get("aimable") is False:
        reason = "policy_support_not_aimable"
    else:
        reason = "policy_geometry_not_encodable"
    result = dict(schema=OBJECT_SCHEMA, available=available, reason=reason,
                  source=source, coordinate_view="wrist", coordinate_space="letterbox224/224",
                  vector=token.astype(np.float32).tolist(), policy_record=policy,
                  diagnostic_record=diagnostic,
                  image_time_s=None if image_time_s is None else float(image_time_s),
                  q_at_image=None if q_at_image is None else _array(q_at_image).reshape(6).tolist(),
                  diagnostic_is_policy_target=False, physical_target_verified=False)
    json.dumps(result, allow_nan=False)
    return result


class SourceEvidenceStore:
    """Small per-view cache; frozen plan bundles survive later captures/eviction."""
    def __init__(self, maximum_frames_per_view=16):
        if type(maximum_frames_per_view) is not int or maximum_frames_per_view < 1:
            raise ValueError("positive integer source frame capacity required")
        self.maximum_frames_per_view = maximum_frames_per_view
        self._frames = {view: OrderedDict() for view in VIEWS}
        self._lock = threading.RLock()

    def remember(self, rgb224, observation, *, acquisition=None, object_provenance=None):
        if not isinstance(observation, VisualObservation) or observation.source != "actual_rgb":
            raise ValueError("source evidence accepts actual RGB observations only")
        if observation.view_order != VIEWS:
            raise ValueError("source evidence requires wrist,column view order")
        rgb = _array(rgb224)
        features, q, times, available = map(_array, (
            observation.features, observation.q_at_image, observation.image_times,
            observation.measurement_available))
        if rgb.shape != (2, 224, 224, 3) or rgb.dtype != np.uint8:
            raise ValueError("source evidence requires exact RGB uint8 [2,224,224,3]")
        if (features.shape != (1, 2, PATCHES, DIM) or q.shape != (1, 2, 6)
                or not np.issubdtype(features.dtype, np.floating) or not np.issubdtype(q.dtype, np.floating)
                or times.shape != (1, 2) or times.dtype != np.float64
                or available.shape != (1, 2) or available.dtype != np.bool_):
            raise ValueError("unaligned source observation fields")
        if (not np.isfinite(features[available]).all() or not np.isfinite(q[available]).all()
                or not np.isfinite(times[available]).all() or (times[available] < 0).any()):
            raise ValueError("nonfinite or invalid source evidence")
        provenance = deepcopy(object_provenance)
        if provenance is not None and provenance.get("schema") != OBJECT_SCHEMA:
            raise ValueError("invalid object token provenance schema")
        if provenance is not None:
            expected_token = (np.zeros(8, np.float32) if observation.object_relative is None else
                              _array(observation.object_relative).reshape(8).astype(np.float32))
            if not np.array_equal(np.asarray(provenance.get("vector"), np.float32), expected_token):
                raise ValueError("object provenance differs from the encoded observation")
            if available[0, 0] and (
                    provenance.get("image_time_s") != float(times[0, 0]) or
                    not np.array_equal(np.asarray(provenance.get("q_at_image"), dtype=q.dtype), q[0, 0])):
                raise ValueError("object provenance is not paired to the wrist exposure")
        staged = []
        for i, view in enumerate(VIEWS):
            if not available[0, i]:
                continue
            stamp = float(times[0, i])
            frame = dict(rgb=rgb[i].copy(), features=features[0, i].copy(),
                         q_at_image=q[0, i].copy(), image_time_s=stamp,
                         acquisition=deepcopy(_json(acquisition)),
                         object_provenance=provenance if view == "wrist" else None)
            frame["identity"] = {name: array_sha256(frame[name])
                                 for name in ("rgb", "features", "q_at_image")}
            staged.append((view, stamp, frame))
        with self._lock:
            # Validate all views before committing any of the paired capture.
            for view, stamp, frame in staged:
                old = self._frames[view].get(stamp)
                if old is not None and old["identity"] != frame["identity"]:
                    raise ValueError("same source exposure has changed RGB, q, or features")
            for view, stamp, frame in staged:
                bank = self._frames[view]
                bank[stamp] = frame
                bank.move_to_end(stamp)
                while len(bank) > self.maximum_frames_per_view:
                    bank.popitem(last=False)

    def _lookup(self, view, stamp):
        with self._lock:
            frame = self._frames[view].get(stamp)
            return None if frame is None else deepcopy(frame)


@dataclass(frozen=True)
class SourceEvidenceBundle:
    manifest: dict
    arrays: dict
    manifest_sha256: str = field(init=False)

    def __post_init__(self):
        payload = json.dumps(self.manifest, sort_keys=True, allow_nan=False).encode()
        object.__setattr__(self, "manifest_sha256", hashlib.sha256(payload).hexdigest())


def snapshot_source_evidence(state, store=None):
    """Freeze the latest valid source frame per view at the actual plan state."""
    if store is not None and not isinstance(store, SourceEvidenceStore):
        raise TypeError("source evidence requires its capture-owned SourceEvidenceStore")
    memory = state.memory
    valid, features, positions, times = map(_array, (
        memory.valid, memory.features, memory.q_at_image, memory.image_times))
    now = _array(state.time)
    if (valid.shape != (1, 2, 3) or valid.dtype != np.bool_
            or features.shape != (1, 2, 3, PATCHES, DIM) or positions.shape != (1, 2, 3, 6)
            or times.shape != (1, 2, 3) or times.dtype != np.float64
            or now.shape != (1,) or now.dtype != np.float64
            or not np.isfinite(now).all()):
        raise ValueError("source evidence requires one complete actual state")
    if not np.isfinite(times[valid]).all() or (times[valid] > now[0]).any():
        raise ValueError("source evidence cannot contain future image clocks")
    arrays, rows = {}, []
    matched_wrist = None
    for view_index, view in enumerate(VIEWS):
        slots = np.flatnonzero(valid[0, view_index])
        row = dict(view=view, memory_available=bool(len(slots)),
                   raw_rgb_available=False, source="actual_visual_memory",
                   reason="memory_view_unavailable", slot=None)
        if not len(slots):
            rows.append(row); continue
        slot = int(slots[-1]); stamp = float(times[0, view_index, slot])
        expected_q = positions[0, view_index, slot]
        expected_features = features[0, view_index, slot]
        if not np.isfinite(expected_q).all() or not np.isfinite(expected_features).all():
            raise ValueError("nonfinite actual source memory")
        row.update(slot=slot, image_time_s=stamp, age_at_source_s=float(now[0] - stamp),
                   q_at_image=expected_q.tolist(), reason="raw_source_not_retained",
                   model_feature_sha256=array_sha256(expected_features),
                   model_q_sha256=array_sha256(expected_q))
        frame = None if store is None else store._lookup(view, stamp)
        if frame is not None:
            # Match after exactly the dtype conversion used by memory.write.
            if not np.array_equal(frame["q_at_image"].astype(expected_q.dtype), expected_q):
                row["reason"] = "source_q_pairing_mismatch"
            elif not np.array_equal(frame["features"].astype(expected_features.dtype), expected_features):
                row["reason"] = "source_feature_pairing_mismatch"
            else:
                for name in ("rgb", "features", "q_at_image"):
                    arrays[f"{view}_{name}"] = frame[name].copy()
                arrays[f"{view}_image_time_s"] = np.asarray(stamp, dtype=np.float64)
                row.update(raw_rgb_available=True, reason=None,
                           array_keys={name: f"{view}_{name}" for name in
                                       ("rgb", "features", "q_at_image", "image_time_s")},
                           acquisition=frame["acquisition"],
                           captured_checksums=frame["identity"],
                           clock="host-camera-read-finished-not-sensor-exposure")
                if view == "wrist":
                    matched_wrist = frame
        rows.append(row)
    token = getattr(state, "object_relative", None)
    current_wrist = bool(_array(state.measurement_available)[0, 0])
    token_available = token is not None and bool(_array(token).reshape(-1)[0])
    # Core duplicate exposures can carry a current cue with insertion=False.
    # Preserve the state token; do not turn an insertion mask into a camera claim.
    delivered = current_wrist if not token_available else (True if current_wrist else None)
    object_contract = object_token_provenance(token, wrist_measurement_available=delivered)
    stored = None if matched_wrist is None else matched_wrist["object_provenance"]
    if stored is not None and (current_wrist or token_available):
        if np.array_equal(np.asarray(stored["vector"], np.float32), np.asarray(object_contract["vector"], np.float32)):
            object_contract = deepcopy(stored)
        else:
            object_contract.update(source=None, reason="state_token_differs_from_paired_wrist_capture")
    elif stored is None and (current_wrist or token_available):
        object_contract.update(source=None, reason=("available_token_source_evidence_missing" if token_available
                                                    else "unavailable_token_source_evidence_missing"))
    manifest = dict(schema=SCHEMA, executable=False, evidence_helper_hardware_accessed=False, weights_updated=False,
                    source_time_s=float(now[0]), source_q_position=_array(state.q)[0].tolist(),
                    position_unit="lerobot_calibrated_position", rgb_colour_space="RGB",
                    rgb_geometry=[224, 224, 3], rgb_transform="existing-letterbox224-no-additional-transform",
                    view_order=list(VIEWS), scope="latest_valid_actual_memory_frame_per_view",
                    views=rows, object_token=object_contract,
                    state_measurement_available=_array(state.measurement_available).tolist(),
                    missing_raw_source_is_not_replaced=True,
                    array_checksums={name: array_sha256(value) for name, value in arrays.items()})
    indices = getattr(memory, 'readback_indices', None)
    current_index = getattr(memory, 'current_readback_index', None)
    if indices is not None and current_index is not None:
        manifest['readback_progress'] = dict(
            unit='completed_policy_readback_sample_index',
            current_index=_array(current_index).tolist(),
            memory_capture_indices=_array(indices).tolist(),
            progress_session_id=getattr(state,'progress_session_id',None),
            progress_origin=getattr(state,'progress_origin',None),
            actual_timestamps_retained=True)
    json.dumps(manifest, allow_nan=False)
    return SourceEvidenceBundle(manifest, arrays)


def save_source_evidence(output, stem, bundle):
    """Persist a frozen source bundle once; return JSON/NPZ checksum references."""
    if not isinstance(bundle, SourceEvidenceBundle):
        raise TypeError("a frozen source evidence bundle is required")
    if not stem or Path(stem).name != stem or stem in (".", ".."):
        raise ValueError("source evidence stem must be a plain filename")
    output = Path(output)
    json_path = output / f"{stem}-source-evidence.json"
    array_path = output / f"{stem}-source-evidence.npz"
    if json_path.exists() or array_path.exists():
        raise FileExistsError("source evidence already exists")
    expected = bundle.manifest["array_checksums"]
    manifest_digest = hashlib.sha256(json.dumps(bundle.manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if manifest_digest != bundle.manifest_sha256:
        raise ValueError("frozen source evidence manifest changed")
    if set(expected) != set(bundle.arrays) or any(array_sha256(bundle.arrays[k]) != v for k, v in expected.items()):
        raise ValueError("frozen source evidence arrays changed")
    manifest = deepcopy(bundle.manifest)
    with array_path.open("xb") as stream:
        np.savez_compressed(stream, **bundle.arrays)
    manifest["array_file"] = array_path.name
    manifest["array_file_sha256"] = hashlib.sha256(array_path.read_bytes()).hexdigest()
    payload = json.dumps(manifest, indent=2, allow_nan=False).encode()
    with json_path.open("xb") as stream:
        stream.write(payload)
    return dict(schema=SCHEMA, executable=False, json=json_path.name, npz=array_path.name,
                json_sha256=hashlib.sha256(payload).hexdigest(),
                npz_sha256=manifest["array_file_sha256"],
                complete_raw_views=[row["view"] for row in manifest["views"] if row["raw_rgb_available"]],
                object_token=deepcopy(manifest["object_token"]))
