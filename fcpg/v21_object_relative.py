"""Object-relative wrist grounding for the V21 Actor.

Eight numbers per exposure, from the wrist camera and the actual q:

    [available, score, dx, dy, length, width, cos 2theta, sin 2theta]

``dx, dy`` is the object's centre in the wrist picture minus the jaw-pair midpoint (letterbox224 / 224); length, width
and axis come from the same outline, with orientation encoded modulo pi. The outline comes from the live object finder
(fcpg/v21_object_finder.py: the learned finder on the exposure's own DINO tokens, or fcpg/v21_sam3_finder.py), which
makes the token with these helpers' layout. No object / wrist miss means an unavailable (zero) token. The operator
colour model and its encoder that lived here are gone from the code (10/31, James 「不要的废弃代码 这种颜色函数删掉」).
"""
from __future__ import annotations

import numpy as np
import torch

DIM = 8
FIELDS = ("available", "score", "dx", "dy", "length", "width", "cos2theta", "sin2theta")
TRACKER_MINIMUM_SCORE = .05  # Legacy attribute; the frozen-feature path does not read it.


def zeros(batch: int, *, device=None) -> torch.Tensor:
    return torch.zeros(batch, DIM, device=device, dtype=torch.float32)


def encode_vector(record, jaw_pair_px) -> np.ndarray:
    """Pure function from one finder record and the jaw pair [2,2] to the 8-vector.

    Explicit clipping / aimability gates make its token unavailable.
    """
    out = np.zeros(DIM, np.float32)
    if not record or (record.get("cog_px") is None and record.get("centroid_px") is None):
        return out
    if record.get("support_clipped") or record.get("aimable") is False:
        return out
    # C is the same CoG used by the overlay; centroid is a legacy record fallback.
    centroid = np.asarray(record["cog_px"] if record.get("cog_px") is not None
                          else record["centroid_px"], np.float64)
    axis_value = record.get("axis_px")
    axis = np.asarray((1., 0.) if axis_value is None else axis_value, np.float64).reshape(-1)
    length = float(record.get("length_px") or 0.); width = float(record.get("width_px") or 0.)
    score = float(record.get("score") if record.get("score") is not None else 0.)
    pair = np.asarray(jaw_pair_px, np.float64)
    if centroid.shape != (2,) or pair.shape != (2, 2) or not np.isfinite(centroid).all() or not np.isfinite(pair).all():
        return out
    if (axis.shape != (2,) or not np.isfinite(axis).all()
            or not np.isfinite([length, width, score]).all()):
        return out
    jaw = pair[1] - pair[0]
    if np.linalg.norm(axis) < 1e-8 or np.linalg.norm(jaw) < 1e-6:
        return out
    theta = np.arctan2(axis[1], axis[0]) - np.arctan2(jaw[1], jaw[0])
    mid = pair.mean(0)
    out[:] = (1., max(0., min(1., score)), (centroid[0] - mid[0]) / 224., (centroid[1] - mid[1]) / 224.,
              length / 224., width / 224., np.cos(2 * theta), np.sin(2 * theta))
    return out


def encode_place_vector(record, jaw_pair_px) -> np.ndarray:
    """encode_vector for where the object goes (the registered drop zone in the wrist picture): the same 8 numbers
    against the same jaw pair, available whenever it is seen. No clipping or aimability gate: near the jaws a box fills the picture
    and its support always touches the border (docs/V21_FCPG_ACTOR_TRAINING_OVERHAUL_20261012.md section 17)."""
    if not isinstance(record, dict) or record.get("visible") is not True:
        return np.zeros(DIM, np.float32)
    return encode_vector({k: v for k, v in record.items() if k not in ("support_clipped", "aimable")}, jaw_pair_px)


def assistance_cue(owner):
    """Separate current display/helper cue, never the Actor's input channel."""
    endpoint=getattr(owner,'last_endpoint',None)
    observation=getattr(endpoint,'observation',None)
    if observation is not None and not bool(observation.measurement_available[0,0]):
        return np.zeros(DIM,np.float32)
    value=getattr(owner,'last_object_diagnostic',None)
    return value if value is not None else getattr(owner,'last_object_relative',None)


__all__ = ["DIM", "FIELDS", "TRACKER_MINIMUM_SCORE", "zeros", "encode_vector", "encode_place_vector", "assistance_cue"]
