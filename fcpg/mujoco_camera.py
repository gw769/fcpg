"""MuJoCo camera helpers for projection sanity checks."""

from __future__ import annotations

import math

import numpy as np


def sample_camera_pose_offset(
    rng: np.random.Generator,
    translation_cm_max: float,
    rotation_deg_max: float,
    clean_probability: float = 0.0,
    *,
    translation_cm_min: float = 0.0,
    rotation_deg_min: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample a continuous world-frame SE(3) offset for a fixed camera."""

    if not 0.0 <= translation_cm_min <= translation_cm_max:
        raise ValueError("translation_cm_min must be in [0, translation_cm_max]")
    if not 0.0 <= rotation_deg_min <= rotation_deg_max:
        raise ValueError("rotation_deg_min must be in [0, rotation_deg_max]")
    if not 0.0 <= clean_probability <= 1.0:
        raise ValueError("clean_probability must be in [0, 1]")

    zero_translation = np.zeros(3, dtype=np.float64)
    zero_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if translation_cm_max <= 0.0 and rotation_deg_max <= 0.0:
        return zero_translation, zero_axis, 0.0
    if clean_probability > 0.0 and rng.random() < clean_probability:
        return zero_translation, zero_axis, 0.0

    translation = zero_translation.copy()
    if translation_cm_max > 0.0:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        magnitude_m = rng.uniform(translation_cm_min, translation_cm_max) / 100.0
        translation = direction * magnitude_m

    rotation_axis = zero_axis.copy()
    rotation_degrees = 0.0
    if rotation_deg_max > 0.0:
        rotation_axis = rng.normal(size=3)
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_magnitude = float(rng.uniform(rotation_deg_min, rotation_deg_max))
        rotation_sign = -1.0 if rng.random() < 0.5 else 1.0
        rotation_degrees = rotation_sign * rotation_magnitude
    return translation, rotation_axis, rotation_degrees


def camera_intrinsic_from_fovy(width: int, height: int, fovy_degrees: float) -> np.ndarray:
    """Return a pinhole intrinsic matrix from MuJoCo vertical field-of-view."""

    fy = 0.5 * height / math.tan(math.radians(fovy_degrees) / 2.0)
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_extrinsic_cv_from_mujoco(data, camera_id: int) -> np.ndarray:
    """Build an OpenCV-style world-to-camera matrix from MuJoCo camera state.

    MuJoCo's camera frame follows OpenGL convention: x right, y up, and the
    camera looks along negative z. We convert it to an OpenCV-like frame:
    x right, y down, z forward. This convention matches the projection helper
    in `fcpg.geometry.project_points`.
    """

    camera_pos = np.asarray(data.cam_xpos[camera_id], dtype=np.float64)
    camera_rot = np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
    world_to_mjcam = np.eye(4, dtype=np.float64)
    world_to_mjcam[:3, :3] = camera_rot.T
    world_to_mjcam[:3, 3] = -camera_rot.T @ camera_pos

    mjcam_to_cv = np.eye(4, dtype=np.float64)
    mjcam_to_cv[1, 1] = -1.0
    mjcam_to_cv[2, 2] = -1.0
    return mjcam_to_cv @ world_to_mjcam


def get_camera_id(model, camera_name: str) -> int:
    import mujoco

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"Unknown camera: {camera_name}")
    return int(camera_id)


def set_fixed_camera_pose_offset(
    model,
    data,
    camera_id: int,
    base_position: np.ndarray,
    base_quaternion: np.ndarray,
    translation_world: np.ndarray,
    rotation_axis_world: np.ndarray,
    rotation_degrees: float,
) -> None:
    """Set a world-attached MuJoCo camera from a fixed base pose plus an SE(3) offset."""

    import mujoco

    if int(model.cam_bodyid[camera_id]) != 0:
        raise ValueError("Camera pose randomization currently requires a world-attached fixed camera")

    base_position = np.asarray(base_position, dtype=np.float64)
    base_quaternion = np.asarray(base_quaternion, dtype=np.float64)
    translation_world = np.asarray(translation_world, dtype=np.float64)
    rotation_axis_world = np.asarray(rotation_axis_world, dtype=np.float64)

    model.cam_pos[camera_id] = base_position + translation_world
    if abs(float(rotation_degrees)) <= 1e-12:
        model.cam_quat[camera_id] = base_quaternion
    else:
        axis_norm = float(np.linalg.norm(rotation_axis_world))
        if axis_norm <= 1e-12:
            raise ValueError("rotation_axis_world must be non-zero when rotation_degrees is non-zero")
        delta_quaternion = np.zeros(4, dtype=np.float64)
        composed_quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_axisAngle2Quat(
            delta_quaternion,
            rotation_axis_world / axis_norm,
            math.radians(float(rotation_degrees)),
        )
        mujoco.mju_mulQuat(composed_quaternion, delta_quaternion, base_quaternion)
        model.cam_quat[camera_id] = composed_quaternion / np.linalg.norm(composed_quaternion)
    mujoco.mj_forward(model, data)


def get_site_id(model, site_name: str) -> int:
    import mujoco

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Unknown site: {site_name}")
    return int(site_id)


def list_mujoco_names(model, obj_type, count: int) -> list[str]:
    import mujoco

    names: list[str] = []
    for idx in range(count):
        name = mujoco.mj_id2name(model, obj_type, idx)
        if name:
            names.append(name)
    return names
