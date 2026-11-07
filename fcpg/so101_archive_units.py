"""Canonicalize heterogeneous archived SO-101 joint representations.

The public archive contains post-PR-777 normalized positions, legacy physical
degrees, and a small number of exports in MuJoCo radians.  Geometry and the
frozen V12.4 endpoint must not silently interpret those three encodings as the
same six numbers.  This module provides the fixed, provenance-recorded affine
maps used by the canonical manifest and every downstream archive reader.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


CANONICAL_JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

# Official SO-101 limits from the vendored upstream MuJoCo models.  The old
# model expresses joints in degrees; the new model expresses them in radians.
OLD_CALIB_DEG = np.asarray(
    [
        [-110.00000000000001, 110.00000000000013],
        [-190.00000000000284, 10.00000000000017],
        [-9.999999999999405, 180.00000000000085],
        [-95.00000245400055, 94.99999958700116],
        [-160.0000107309983, 159.9999999990015],
        [-10.000000000000607, 100.00000000000199],
    ],
    dtype=np.float64,
)
NEW_CALIB_RAD = np.asarray(
    [
        [-1.9198621771937616, 1.9198621771937634],
        [-1.7453292519943224, 1.7453292519943366],
        [-1.69, 1.69],
        [-1.6580628494556928, 1.6580627293335335],
        [-2.7438472969992493, 2.841206309382605],
        [-0.17453297762778586, 1.7453291995659765],
    ],
    dtype=np.float64,
)
LEGACY_POSITIVE_DEG = OLD_CALIB_DEG.copy()
LEGACY_POSITIVE_DEG[1] = -OLD_CALIB_DEG[1, ::-1]

SUPPORTED_UNIT_FAMILIES = frozenset(
    {
        "lerobot_normalized",
        "old_calib_degrees",
        "legacy_positive_degrees",
        "new_calib_radians",
    }
)


def canonical_joint_order(feature: Mapping[str, object], *, name: str) -> tuple[int, ...]:
    """Return source-column indices in canonical SO-101 joint order."""

    raw_names = feature.get("names")
    if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
        raise ValueError(f"{name} metadata must declare six joint names")
    aliases: dict[str, int] = {}
    for index, raw_name in enumerate(raw_names):
        normalized = str(raw_name).removeprefix("main_")
        if not normalized.endswith(".pos"):
            normalized += ".pos"
        if normalized in aliases:
            raise ValueError(f"{name} repeats joint name {normalized!r}")
        aliases[normalized] = index
    missing = [joint for joint in CANONICAL_JOINT_NAMES if joint not in aliases]
    if missing or len(raw_names) != 6:
        raise ValueError(
            f"{name} must name exactly the canonical SO-101 joints; missing={missing}"
        )
    return tuple(aliases[joint] for joint in CANONICAL_JOINT_NAMES)


def canonical_affine(unit_family: str) -> tuple[np.ndarray, np.ndarray]:
    """Map one declared raw unit family into V12/FK canonical coordinates.

    Arm joints map to ``[-100,100]`` over their official mechanical ranges;
    the gripper maps to ``[0,100]``.  Post-PR-777 normalized archives are
    identity-mapped so small measured calibration overshoot remains observable.
    """

    if unit_family not in SUPPORTED_UNIT_FAMILIES:
        raise ValueError(f"unsupported SO-101 joint unit family {unit_family!r}")
    if unit_family == "lerobot_normalized":
        return np.ones(6, dtype=np.float64), np.zeros(6, dtype=np.float64)
    ranges = {
        "old_calib_degrees": OLD_CALIB_DEG,
        "legacy_positive_degrees": LEGACY_POSITIVE_DEG,
        "new_calib_radians": NEW_CALIB_RAD,
    }[unit_family]
    span = ranges[:, 1] - ranges[:, 0]
    scale = np.empty(6, dtype=np.float64)
    offset = np.empty(6, dtype=np.float64)
    scale[:5] = 200.0 / span[:5]
    offset[:5] = -100.0 - scale[:5] * ranges[:5, 0]
    if unit_family == "legacy_positive_degrees":
        # These exports negate the old-calib shoulder-lift coordinate while
        # leaving the other five joint axes unchanged.
        scale[1] *= -1.0
        offset[1] = 100.0 - scale[1] * ranges[1, 0]
    scale[5] = 100.0 / span[5]
    offset[5] = -scale[5] * ranges[5, 0]
    return scale, offset


def canonicalize_joint_values(
    values: np.ndarray,
    *,
    order: Sequence[int],
    scale: Sequence[float],
    offset: Sequence[float],
) -> np.ndarray:
    """Reorder and affine-map raw joint rows without clipping."""

    raw = np.asarray(values)
    if raw.ndim < 1 or raw.shape[-1] != 6:
        raise ValueError(f"SO-101 joint values must end in six dimensions, got {raw.shape}")
    indices = np.asarray(order, dtype=np.int64)
    gains = np.asarray(scale, dtype=np.float64)
    biases = np.asarray(offset, dtype=np.float64)
    if sorted(indices.tolist()) != list(range(6)):
        raise ValueError("joint order must be a permutation of range(6)")
    if gains.shape != (6,) or biases.shape != (6,):
        raise ValueError("joint affine scale/offset must each contain six values")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(gains)) or not np.all(
        np.isfinite(biases)
    ):
        raise ValueError("joint values and affine must be finite")
    return raw[..., indices].astype(np.float64, copy=False) * gains + biases


def row_joint_transform(row: Mapping[str, object], column: str) -> tuple[list[int], list[float], list[float]]:
    """Read a manifest transform, defaulting to identity for legacy fixtures."""

    if column not in {"state", "action"}:
        raise ValueError("column must be state or action")
    order = row.get(f"{column}_joint_order", list(range(6)))
    scale = row.get("joint_affine_scale", [1.0] * 6)
    offset = row.get("joint_affine_offset", [0.0] * 6)
    return list(order), list(scale), list(offset)


__all__ = [
    "CANONICAL_JOINT_NAMES",
    "LEGACY_POSITIVE_DEG",
    "NEW_CALIB_RAD",
    "OLD_CALIB_DEG",
    "SUPPORTED_UNIT_FAMILIES",
    "canonical_affine",
    "canonical_joint_order",
    "canonicalize_joint_values",
    "row_joint_transform",
]
