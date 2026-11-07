"""Small image-space camera perturbations for real SO-101 FCPG development."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SO101ImageAffineDrift:
    """A center rotation followed by image-plane translation."""

    rotation_degrees: float = 0.0
    translation_x_px: float = 0.0
    translation_y_px: float = 0.0

    def matrix(self, width: int, height: int) -> np.ndarray:
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        center = ((float(width) - 1.0) / 2.0, (float(height) - 1.0) / 2.0)
        matrix = cv2.getRotationMatrix2D(center, self.rotation_degrees, 1.0)
        matrix[0, 2] += self.translation_x_px
        matrix[1, 2] += self.translation_y_px
        return np.asarray(matrix, dtype=np.float64)

    def as_array(self) -> np.ndarray:
        return np.asarray(
            (
                self.rotation_degrees,
                self.translation_x_px,
                self.translation_y_px,
            ),
            dtype=np.float32,
        )


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    affine = np.asarray(matrix, dtype=np.float64)
    if values.shape[-1:] != (2,):
        raise ValueError("points must end in two pixel coordinates")
    if affine.shape != (2, 3):
        raise ValueError("affine matrix must have shape (2, 3)")
    homogeneous = np.concatenate(
        (values, np.ones((*values.shape[:-1], 1), dtype=np.float64)), axis=-1
    )
    return (homogeneous @ affine.T).astype(np.float32)


def points_in_bounds(points: np.ndarray, width: int, height: int) -> np.ndarray:
    values = np.asarray(points)
    if values.shape[-1:] != (2,):
        raise ValueError("points must end in two pixel coordinates")
    return (
        np.all(np.isfinite(values), axis=-1)
        & (values[..., 0] >= 0.0)
        & (values[..., 0] <= float(width - 1))
        & (values[..., 1] >= 0.0)
        & (values[..., 1] <= float(height - 1))
    )


def warp_rgb(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    affine = np.asarray(matrix, dtype=np.float64)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be a uint8 HxWx3 image")
    if affine.shape != (2, 3):
        raise ValueError("affine matrix must have shape (2, 3)")
    height, width = image.shape[:2]
    return cv2.warpAffine(
        image,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def sample_bounded_drift(
    rng: np.random.Generator,
    points_by_view: tuple[np.ndarray, ...],
    valid_by_view: tuple[np.ndarray, ...],
    *,
    width: int,
    height: int,
    max_rotation_degrees: float,
    max_translation_px: float,
    min_mean_residual_px: float,
    max_residual_px: float,
    attempts: int = 256,
) -> SO101ImageAffineDrift:
    """Sample one perturbation shared by all views at a physical timestamp."""

    if len(points_by_view) != len(valid_by_view) or not points_by_view:
        raise ValueError("points and valid masks must contain the same views")
    if min(max_rotation_degrees, max_translation_px, min_mean_residual_px) < 0:
        raise ValueError("drift limits must be non-negative")
    if max_residual_px <= min_mean_residual_px:
        raise ValueError("max residual must exceed the minimum mean residual")
    if attempts <= 0:
        raise ValueError("attempts must be positive")

    for _ in range(attempts):
        drift = SO101ImageAffineDrift(
            rotation_degrees=float(
                rng.uniform(-max_rotation_degrees, max_rotation_degrees)
            ),
            translation_x_px=float(
                rng.uniform(-max_translation_px, max_translation_px)
            ),
            translation_y_px=float(
                rng.uniform(-max_translation_px, max_translation_px)
            ),
        )
        matrix = drift.matrix(width, height)
        residuals: list[np.ndarray] = []
        supported_targets: list[np.ndarray] = []
        for points, valid in zip(points_by_view, valid_by_view, strict=True):
            values = np.asarray(points, dtype=np.float32)
            support = np.asarray(valid, dtype=bool)
            if values.shape != (3, 2) or support.shape != (3,):
                raise ValueError("each view must provide three anchors")
            target = transform_points(values, matrix)
            support = support & points_in_bounds(target, width, height)
            if np.any(support):
                residuals.append(np.linalg.norm(target[support] - values[support], axis=-1))
                supported_targets.append(target[support])
        if not residuals:
            continue
        residual = np.concatenate(residuals)
        if (
            float(residual.mean()) >= min_mean_residual_px
            and float(residual.max()) <= max_residual_px
            and sum(len(values) for values in supported_targets) >= 3
        ):
            return drift
    raise RuntimeError("could not sample a bounded affine drift")


__all__ = [
    "SO101ImageAffineDrift",
    "points_in_bounds",
    "sample_bounded_drift",
    "transform_points",
    "warp_rgb",
]
