"""Nominal camera-projection fits for real SO-101 recordings.

Real SO-101 recordings do not ship camera calibration. This module fits only
the nominal pixel locations needed by FCPG from clicked anchor correspondences
and MuJoCo FK features. It is a projection diagnostic, not metric calibration:
the fixed column view uses base-frame XYZ, while the eye-in-hand wrist view
uses gripper-frame XYZ plus gripper opening.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SO101_CALIBRATION_VIEWS = ("wrist", "column")


@dataclass(frozen=True)
class ProjectionFit:
    """A linear feature-to-pixel projection and its feature convention."""

    view: str
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    ridge: float

    def __post_init__(self) -> None:
        if self.view not in SO101_CALIBRATION_VIEWS:
            raise ValueError(f"unsupported calibration view: {self.view!r}")
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if coefficients.ndim != 2 or coefficients.shape[1] != 2:
            raise ValueError("projection coefficients must have shape [features, 2]")
        if coefficients.shape[0] != len(self.feature_names):
            raise ValueError("feature names do not match projection coefficients")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("projection coefficients must be finite")
        if self.ridge < 0 or not np.isfinite(self.ridge):
            raise ValueError("ridge must be finite and non-negative")


def projection_features(
    view: str,
    anchors_base_xyz: np.ndarray,
    anchors_gripper_xyz: np.ndarray,
    gripper: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build per-anchor features, flattened sample-major then anchor-major."""

    if view not in SO101_CALIBRATION_VIEWS:
        raise ValueError(f"unsupported calibration view: {view!r}")
    base = np.asarray(anchors_base_xyz, dtype=np.float64)
    local = np.asarray(anchors_gripper_xyz, dtype=np.float64)
    opening = np.asarray(gripper, dtype=np.float64)
    if base.ndim != 3 or base.shape[-1] != 3:
        raise ValueError("anchors_base_xyz must have shape [N, A, 3]")
    if local.shape != base.shape:
        raise ValueError("base and gripper anchor arrays must have equal shape")
    if opening.shape != (len(base),):
        raise ValueError("gripper must have shape [N]")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(local)):
        raise ValueError("anchor coordinates must be finite")
    if not np.all(np.isfinite(opening)):
        raise ValueError("gripper values must be finite")

    if view == "column":
        values = np.concatenate(
            (base, np.ones((*base.shape[:2], 1), dtype=np.float64)), axis=-1
        )
        names = ("x_base", "y_base", "z_base", "bias")
    else:
        normalized_opening = np.clip(opening, 0.0, 100.0) / 100.0
        opening_column = np.broadcast_to(
            normalized_opening[:, None, None], (*base.shape[:2], 1)
        )
        values = np.concatenate(
            (
                local,
                opening_column,
                np.ones((*base.shape[:2], 1), dtype=np.float64),
            ),
            axis=-1,
        )
        names = ("x_gripper", "y_gripper", "z_gripper", "opening", "bias")
    return values.reshape(-1, values.shape[-1]), names


def fit_projection(
    view: str,
    anchors_base_xyz: np.ndarray,
    anchors_gripper_xyz: np.ndarray,
    gripper: np.ndarray,
    pixels: np.ndarray,
    *,
    ridge: float = 1e-8,
) -> ProjectionFit:
    """Fit a ridge-stabilized linear pixel projection."""

    features, names = projection_features(
        view, anchors_base_xyz, anchors_gripper_xyz, gripper
    )
    points = np.asarray(pixels, dtype=np.float64)
    expected = np.asarray(anchors_base_xyz).shape[:2] + (2,)
    if points.shape != expected:
        raise ValueError(f"pixels must have shape {expected}, got {points.shape}")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")
    flat_points = points.reshape(-1, 2)
    if not np.all(np.isfinite(flat_points)):
        raise ValueError("clicked pixels must be finite")
    normal = features.T @ features
    regularizer = np.eye(features.shape[1], dtype=np.float64) * float(ridge)
    coefficients = np.linalg.solve(normal + regularizer, features.T @ flat_points)
    return ProjectionFit(view, tuple(names), coefficients, float(ridge))


def predict_projection(
    fit: ProjectionFit,
    anchors_base_xyz: np.ndarray,
    anchors_gripper_xyz: np.ndarray,
    gripper: np.ndarray,
) -> np.ndarray:
    features, names = projection_features(
        fit.view, anchors_base_xyz, anchors_gripper_xyz, gripper
    )
    if tuple(names) != tuple(fit.feature_names):
        raise ValueError("projection feature convention does not match fit")
    shape = np.asarray(anchors_base_xyz).shape[:2] + (2,)
    return (features @ np.asarray(fit.coefficients, dtype=np.float64)).reshape(shape)


def approximate_column_backprojection_on_plane(
    fit: ProjectionFit,
    pixels: np.ndarray,
    *,
    z_base: float,
    max_condition_number: float = 1e4,
) -> np.ndarray:
    """Invert the column affine fit on a known base-frame horizontal plane.

    This is a temporary RGB-only substitute for metric depth. It is suitable
    for MuJoCo/IK shadow planning on a tabletop, but the affine projection fit
    is not a calibrated pinhole camera and must not be presented as accurate
    3-D sensing.
    """

    points = np.asarray(pixels, dtype=np.float64)
    if points.shape[-1:] != (2,) or not np.all(np.isfinite(points)):
        raise ValueError("pixels must end in one finite UV pair")
    if fit.view != "column" or tuple(fit.feature_names) != (
        "x_base",
        "y_base",
        "z_base",
        "bias",
    ):
        raise ValueError("planar backprojection requires the column base-XYZ fit")
    if not np.isfinite(z_base):
        raise ValueError("z_base must be finite")
    if max_condition_number <= 1 or not np.isfinite(max_condition_number):
        raise ValueError("max_condition_number must be finite and greater than one")

    coefficients = np.asarray(fit.coefficients, dtype=np.float64)
    xy_to_uv = coefficients[:2]
    condition = float(np.linalg.cond(xy_to_uv))
    if not np.isfinite(condition) or condition > max_condition_number:
        raise ValueError(
            "column affine fit is not invertible on the requested table plane: "
            f"condition_number={condition:.3g}"
        )
    offset = float(z_base) * coefficients[2] + coefficients[3]
    flat = points.reshape(-1, 2)
    xy = np.linalg.solve(xy_to_uv.T, (flat - offset[None]).T).T
    xyz = np.concatenate(
        (xy, np.full((len(xy), 1), float(z_base), dtype=np.float64)), axis=1
    )
    return xyz.reshape(*points.shape[:-1], 3)


def projection_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    image_size: tuple[int, int] | None = None,
) -> dict[str, float | list[float]]:
    """Return aggregate and per-anchor pixel errors for a fit."""

    prediction = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(observed, dtype=np.float64)
    if (
        prediction.shape != target.shape
        or prediction.ndim != 3
        or prediction.shape[-1] != 2
    ):
        raise ValueError("predicted and observed must both have shape [N, A, 2]")
    error = np.linalg.norm(prediction - target, axis=-1)
    result: dict[str, float | list[float]] = {
        "mean_epe_px": float(np.mean(error)),
        "median_epe_px": float(np.median(error)),
        "p95_epe_px": float(np.percentile(error, 95)),
        "max_epe_px": float(np.max(error)),
        "per_anchor_mean_epe_px": np.mean(error, axis=0).tolist(),
    }
    if image_size is not None:
        width, height = (int(image_size[0]), int(image_size[1]))
        in_bounds = (
            (prediction[..., 0] >= 0)
            & (prediction[..., 0] < width)
            & (prediction[..., 1] >= 0)
            & (prediction[..., 1] < height)
        )
        result["predicted_in_bounds_fraction"] = float(np.mean(in_bounds))
    return result


def valid_projection_mask(
    pixels: np.ndarray,
    *,
    image_size: tuple[int, int],
    margin: float = 1.0,
) -> np.ndarray:
    """Mark projections inside the image with a small safety margin."""

    points = np.asarray(pixels, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("pixels must have shape [N, A, 2]")
    width, height = map(float, image_size)
    if width <= 0 or height <= 0 or margin < 0:
        raise ValueError("image size and margin are invalid")
    return (
        np.all(np.isfinite(points), axis=-1)
        & (points[..., 0] >= margin)
        & (points[..., 0] < width - margin)
        & (points[..., 1] >= margin)
        & (points[..., 1] < height - margin)
    )
