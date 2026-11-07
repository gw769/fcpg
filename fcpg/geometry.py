"""Geometry helpers for FCPG stage-0/1 experiments.

The functions in this file intentionally avoid simulator-specific APIs. They
operate on camera matrices and image coordinates so they can be tested before
MetaWorld/RLBench integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    """Pinhole camera parameters.

    `extrinsic_cam_from_world` maps homogeneous world coordinates into the
    camera frame.
    """

    intrinsic: np.ndarray
    extrinsic_cam_from_world: np.ndarray
    width: int
    height: int


def as_homogeneous(points_xyz: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    if points_xyz.ndim == 1:
        points_xyz = points_xyz[None, :]
    ones = np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)
    return np.concatenate([points_xyz, ones], axis=1)


def project_points(points_xyz: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D world points to image pixels.

    Returns:
        pixels: shape [N, 2], in image pixel coordinates `(x, y)`.
        valid: shape [N], true if point is in front of camera and image bounds.
    """

    points_h = as_homogeneous(points_xyz)
    cam_h = (camera.extrinsic_cam_from_world @ points_h.T).T
    cam_xyz = cam_h[:, :3]
    z = cam_xyz[:, 2]
    eps = 1e-8
    uvw = (camera.intrinsic @ cam_xyz.T).T
    pixels = uvw[:, :2] / np.maximum(uvw[:, 2:3], eps)
    valid = (
        (z > eps)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < camera.width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < camera.height)
    )
    return pixels, valid


def pixels_to_grid(pixels_xy: np.ndarray, width: int, height: int, align_corners: bool = False) -> np.ndarray:
    """Convert image pixels `(x, y)` to PyTorch `grid_sample` coordinates."""

    pixels_xy = np.asarray(pixels_xy, dtype=np.float64)
    if align_corners:
        x = 2.0 * pixels_xy[..., 0] / max(width - 1, 1) - 1.0
        y = 2.0 * pixels_xy[..., 1] / max(height - 1, 1) - 1.0
    else:
        x = (2.0 * pixels_xy[..., 0] + 1.0) / width - 1.0
        y = (2.0 * pixels_xy[..., 1] + 1.0) / height - 1.0
    return np.stack([x, y], axis=-1)


def grid_to_pixels(grid_xy: np.ndarray, width: int, height: int, align_corners: bool = False) -> np.ndarray:
    """Convert PyTorch `grid_sample` coordinates to image pixels `(x, y)`."""

    grid_xy = np.asarray(grid_xy, dtype=np.float64)
    if align_corners:
        x = (grid_xy[..., 0] + 1.0) * max(width - 1, 1) / 2.0
        y = (grid_xy[..., 1] + 1.0) * max(height - 1, 1) / 2.0
    else:
        x = ((grid_xy[..., 0] + 1.0) * width - 1.0) / 2.0
        y = ((grid_xy[..., 1] + 1.0) * height - 1.0) / 2.0
    return np.stack([x, y], axis=-1)


def make_gaussian_heatmap(width: int, height: int, center_xy: np.ndarray, sigma: float) -> np.ndarray:
    """Create a 2D Gaussian heatmap centered at image pixel `center_xy`."""

    center_xy = np.asarray(center_xy, dtype=np.float64)
    yy, xx = np.mgrid[0:height, 0:width]
    dist2 = (xx - center_xy[0]) ** 2 + (yy - center_xy[1]) ** 2
    return np.exp(-0.5 * dist2 / max(sigma, 1e-6) ** 2)


def crop_patch(image_hw: np.ndarray, center_xy: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Crop a square patch with zero padding.

    Returns:
        patch: shape `[2 * radius + 1, 2 * radius + 1]`.
        origin_xy: image pixel coordinate of patch index `(0, 0)`.
    """

    center_xy = np.asarray(center_xy, dtype=np.float64)
    cx = int(round(center_xy[0]))
    cy = int(round(center_xy[1]))
    size = 2 * radius + 1
    patch = np.zeros((size, size), dtype=image_hw.dtype)
    x0 = cx - radius
    y0 = cy - radius
    x1 = cx + radius + 1
    y1 = cy + radius + 1
    src_x0 = max(x0, 0)
    src_y0 = max(y0, 0)
    src_x1 = min(x1, image_hw.shape[1])
    src_y1 = min(y1, image_hw.shape[0])
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    patch[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = image_hw[
        src_y0:src_y1, src_x0:src_x1
    ]
    return patch, np.array([x0, y0], dtype=np.float64)


def soft_argmax_patch(patch: np.ndarray, origin_xy: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Estimate image-space point from a local heatmap patch."""

    patch = np.asarray(patch, dtype=np.float64)
    logits = patch / max(temperature, 1e-6)
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    weights = weights / np.maximum(np.sum(weights), 1e-12)
    yy, xx = np.mgrid[0 : patch.shape[0], 0 : patch.shape[1]]
    x = float(np.sum(weights * xx) + origin_xy[0])
    y = float(np.sum(weights * yy) + origin_xy[1])
    return np.array([x, y], dtype=np.float64)


def rotation_matrix_z(degrees: float) -> np.ndarray:
    rad = math.radians(degrees)
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def perturb_extrinsic(extrinsic_cam_from_world: np.ndarray, translation_xyz: np.ndarray, yaw_degrees: float = 0.0) -> np.ndarray:
    """Apply a small camera-frame perturbation to an extrinsic matrix."""

    pert = np.eye(4, dtype=np.float64)
    pert[:3, :3] = rotation_matrix_z(yaw_degrees)
    pert[:3, 3] = np.asarray(translation_xyz, dtype=np.float64)
    return pert @ np.asarray(extrinsic_cam_from_world, dtype=np.float64)
