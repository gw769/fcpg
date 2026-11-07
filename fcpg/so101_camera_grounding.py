"""Camera projection helpers for SO-101 feedback-corrected grounding."""

from __future__ import annotations

import numpy as np

from fcpg.geometry import Camera
from fcpg.mujoco_camera import camera_intrinsic_from_fovy


def free_camera_extrinsic_cv(
    position_world: np.ndarray,
    forward_world: np.ndarray,
    up_world: np.ndarray,
) -> np.ndarray:
    """Convert MuJoCo free-camera axes into an OpenCV world-to-camera matrix."""

    position = np.asarray(position_world, dtype=np.float64)
    forward = np.asarray(forward_world, dtype=np.float64)
    up = np.asarray(up_world, dtype=np.float64)
    if position.shape != (3,) or forward.shape != (3,) or up.shape != (3,):
        raise ValueError("camera position, forward and up must each have shape (3,)")
    if not np.all(np.isfinite(np.concatenate((position, forward, up)))):
        raise ValueError("camera axes must be finite")
    forward_norm = float(np.linalg.norm(forward))
    up_norm = float(np.linalg.norm(up))
    if forward_norm <= 1.0e-9 or up_norm <= 1.0e-9:
        raise ValueError("camera forward and up axes must be non-zero")
    forward = forward / forward_norm
    up = up / up_norm
    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1.0e-9:
        raise ValueError("camera forward and up axes must not be collinear")
    right = right / right_norm
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)

    rotation = np.stack((right, -true_up, forward), axis=0)
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ position
    return extrinsic


def free_camera_projection(
    position_world: np.ndarray,
    forward_world: np.ndarray,
    up_world: np.ndarray,
    *,
    width: int,
    height: int,
    fovy_degrees: float,
) -> Camera:
    """Build the pinhole camera used by a MuJoCo free-camera render."""

    if width <= 0 or height <= 0:
        raise ValueError("camera image dimensions must be positive")
    return Camera(
        intrinsic=camera_intrinsic_from_fovy(width, height, fovy_degrees),
        extrinsic_cam_from_world=free_camera_extrinsic_cv(
            position_world, forward_world, up_world
        ),
        width=int(width),
        height=int(height),
    )


def renderer_free_camera_projection(renderer, *, width: int, height: int, fovy_degrees: float) -> Camera:
    """Read the centered camera pose from an updated MuJoCo renderer scene."""

    cameras = renderer.scene.camera
    positions = np.stack(
        [np.asarray(cameras[index].pos, dtype=np.float64) for index in (0, 1)]
    )
    forwards = np.stack(
        [np.asarray(cameras[index].forward, dtype=np.float64) for index in (0, 1)]
    )
    ups = np.stack(
        [np.asarray(cameras[index].up, dtype=np.float64) for index in (0, 1)]
    )
    return free_camera_projection(
        positions.mean(axis=0),
        forwards.mean(axis=0),
        ups.mean(axis=0),
        width=width,
        height=height,
        fovy_degrees=fovy_degrees,
    )


__all__ = [
    "free_camera_extrinsic_cv",
    "free_camera_projection",
    "renderer_free_camera_projection",
]
