"""Synthetic SO-101 observations with exact nominal and drifted anchor labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fcpg.geometry import project_points
from fcpg.so101_camera_grounding import renderer_free_camera_projection
from fcpg.so101_kinematics import (
    SO101_MUJOCO_JOINT_NAMES,
    SO101MuJoCoKinematics,
    normalized_positions_to_mujoco_qpos,
)


@dataclass(frozen=True)
class SO101CameraDrift:
    azimuth_degrees: float = 0.0
    elevation_degrees: float = 0.0
    distance_meters: float = 0.0


@dataclass(frozen=True)
class SO101SyntheticGroundingFrame:
    rgb: np.ndarray
    anchors_xyz: np.ndarray
    q_nominal_px: np.ndarray
    q_true_px: np.ndarray
    valid_nominal: np.ndarray
    valid_true: np.ndarray
    drift: SO101CameraDrift


class SO101SyntheticGroundingRenderer:
    """Render one physical SO-101 state under a perturbed free camera."""

    def __init__(
        self,
        scene: str | Path,
        *,
        width: int = 224,
        height: int = 224,
        azimuth: float = 135.0,
        elevation: float = -25.0,
        distance: float = 0.72,
        lookat: tuple[float, float, float] = (0.0, 0.0, 0.18),
    ) -> None:
        if width <= 0 or height <= 0 or distance <= 0:
            raise ValueError("render dimensions and camera distance must be positive")
        import mujoco

        self.mujoco = mujoco
        self.scene = Path(scene).resolve()
        self.width = int(width)
        self.height = int(height)
        self.azimuth = float(azimuth)
        self.elevation = float(elevation)
        self.distance = float(distance)
        self.lookat = np.asarray(lookat, dtype=np.float64)
        if self.lookat.shape != (3,) or not np.all(np.isfinite(self.lookat)):
            raise ValueError("lookat must contain three finite values")
        self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model,
            width=self.width,
            height=self.height,
        )
        joint_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in SO101_MUJOCO_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(joint_ids < 0):
            raise ValueError("SO-101 scene is missing one or more required joints")
        self.qpos_addresses = np.asarray(
            self.model.jnt_qposadr[joint_ids], dtype=np.int32
        )
        self.joint_ranges = np.asarray(
            self.model.jnt_range[joint_ids], dtype=np.float64
        ).copy()
        self.kinematics = SO101MuJoCoKinematics(self.scene)

    def _camera(
        self,
        *,
        azimuth: float,
        elevation: float,
        distance: float,
        lookat: np.ndarray,
    ):
        if distance <= 0:
            raise ValueError("camera distance must be positive")
        camera = self.mujoco.MjvCamera()
        camera.type = self.mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = lookat
        camera.distance = float(distance)
        camera.azimuth = float(azimuth)
        camera.elevation = float(elevation)
        return camera

    def _scene_camera_axes(self) -> tuple[np.ndarray, np.ndarray]:
        cameras = self.renderer.scene.camera
        position = np.stack(
            [np.asarray(cameras[index].pos, dtype=np.float64) for index in (0, 1)]
        ).mean(axis=0)
        forward = np.stack(
            [
                np.asarray(cameras[index].forward, dtype=np.float64)
                for index in (0, 1)
            ]
        ).mean(axis=0)
        forward /= np.linalg.norm(forward)
        return position, forward

    def _true_camera(
        self,
        drift: SO101CameraDrift,
        nominal_position: np.ndarray,
    ):
        distance = self.distance + float(drift.distance_meters)
        if distance <= 0:
            raise ValueError("camera drift produces a non-positive distance")
        azimuth = self.azimuth + float(drift.azimuth_degrees)
        elevation = self.elevation + float(drift.elevation_degrees)
        probe = self._camera(
            azimuth=azimuth,
            elevation=elevation,
            distance=self.distance,
            lookat=self.lookat,
        )
        self.renderer.update_scene(self.data, camera=probe)
        _, forward = self._scene_camera_axes()
        desired_position = nominal_position - float(drift.distance_meters) * forward
        lookat = desired_position + distance * forward
        return self._camera(
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            lookat=lookat,
        )

    def _projection(self, camera):
        self.renderer.update_scene(self.data, camera=camera)
        return renderer_free_camera_projection(
            self.renderer,
            width=self.width,
            height=self.height,
            fovy_degrees=float(self.model.vis.global_.fovy),
        )

    def render(
        self,
        state: np.ndarray,
        drift: SO101CameraDrift,
    ) -> SO101SyntheticGroundingFrame:
        values = np.asarray(state, dtype=np.float64)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("state must be one finite six-joint vector")
        self.data.qpos[self.qpos_addresses] = normalized_positions_to_mujoco_qpos(
            values,
            self.joint_ranges,
        )
        self.mujoco.mj_forward(self.model, self.data)
        anchors = self.kinematics.forward_anchors(values[None])[0]
        nominal_camera = self._camera(
            azimuth=self.azimuth,
            elevation=self.elevation,
            distance=self.distance,
            lookat=self.lookat,
        )
        nominal_projection = self._projection(nominal_camera)
        nominal_position, _ = self._scene_camera_axes()
        true_camera = self._true_camera(drift, nominal_position)
        true_projection = self._projection(true_camera)
        rgb = self.renderer.render().copy()
        q_nominal, valid_nominal = project_points(anchors, nominal_projection)
        q_true, valid_true = project_points(anchors, true_projection)
        return SO101SyntheticGroundingFrame(
            rgb=np.asarray(rgb, dtype=np.uint8),
            anchors_xyz=np.asarray(anchors, dtype=np.float32),
            q_nominal_px=np.asarray(q_nominal, dtype=np.float32),
            q_true_px=np.asarray(q_true, dtype=np.float32),
            valid_nominal=np.asarray(valid_nominal, dtype=bool),
            valid_true=np.asarray(valid_true, dtype=bool),
            drift=drift,
        )

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self) -> "SO101SyntheticGroundingRenderer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = [
    "SO101CameraDrift",
    "SO101SyntheticGroundingFrame",
    "SO101SyntheticGroundingRenderer",
]
