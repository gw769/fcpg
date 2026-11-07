"""Differentiable morphology graph used by the structured V17.5 observer.

The graph deliberately separates *robot morphology* from a fixed-width policy
state.  A :class:`RobotSpec` may contain any number of one-DoF hinge or slide
joints, bodies and ordered visual anchors.  SO-101 is the only morphology used
for training in V17.5; the generic representation is an implementation
contract, not evidence of cross-robot generalisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import torch
from torch import nn


HINGE: Final = 0
SLIDE: Final = 1
DEFAULT_SO101_MODEL: Final = (
    Path(__file__).resolve().parents[1]
    / "third_party/so_arm100/Simulation/SO101/so101_new_calib.xml"
)


class RobotSpecError(ValueError):
    """Raised when a morphology or kinematic tensor violates its contract."""


@dataclass(frozen=True)
class RobotSpec:
    """Plain, serialisable description of a one-DoF articulated robot.

    ``position_lower/upper`` describe the public joint-state convention used
    at data/runtime boundaries.  ``coordinate_lower/upper`` are the matching
    hinge radians or slide metres used by FK.  This explicit affine conversion
    is what prevents a cached ``/100`` value from silently being normalised a
    second time inside the network.
    """

    name: str
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    anchor_names: tuple[str, ...]
    view_names: tuple[str, ...]
    body_parent: np.ndarray
    body_translation: np.ndarray
    body_quaternion_wxyz: np.ndarray
    body_joint: np.ndarray
    joint_body: np.ndarray
    joint_type: np.ndarray
    joint_axis: np.ndarray
    joint_position: np.ndarray
    position_lower: np.ndarray
    position_upper: np.ndarray
    coordinate_lower: np.ndarray
    coordinate_upper: np.ndarray
    anchor_body: np.ndarray
    anchor_local_xyz: np.ndarray
    base_reference_body: int
    base_reference_translation: np.ndarray
    base_reference_quaternion_wxyz: np.ndarray
    view_reference_body: np.ndarray
    view_reference_translation: np.ndarray
    view_reference_quaternion_wxyz: np.ndarray
    fixed_tip_index: int
    moving_tip_index: int

    @property
    def joints(self) -> int:
        return len(self.joint_names)

    @property
    def bodies(self) -> int:
        return len(self.body_names)

    @property
    def anchors(self) -> int:
        return len(self.anchor_names)

    @property
    def views(self) -> int:
        return len(self.view_names)

    def validate(self) -> None:
        j, n, a, v = self.joints, self.bodies, self.anchors, self.views
        if min(j, n, a, v) <= 0:
            raise RobotSpecError("robot spec must have joints, bodies, anchors and views")
        shapes = {
            "body_parent": (n,),
            "body_translation": (n, 3),
            "body_quaternion_wxyz": (n, 4),
            "body_joint": (n,),
            "joint_body": (j,),
            "joint_type": (j,),
            "joint_axis": (j, 3),
            "joint_position": (j, 3),
            "position_lower": (j,),
            "position_upper": (j,),
            "coordinate_lower": (j,),
            "coordinate_upper": (j,),
            "anchor_body": (a,),
            "anchor_local_xyz": (a, 3),
            "base_reference_translation": (3,),
            "base_reference_quaternion_wxyz": (4,),
            "view_reference_body": (v,),
            "view_reference_translation": (v, 3),
            "view_reference_quaternion_wxyz": (v, 4),
        }
        for name, expected in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != expected:
                raise RobotSpecError(
                    f"{name} shape drifted: observed={list(value.shape)} expected={list(expected)}"
                )
            if value.dtype.kind in "fc" and not np.isfinite(value).all():
                raise RobotSpecError(f"{name} must be finite")
        if int(self.body_parent[0]) != -1:
            raise RobotSpecError("body zero must be the graph root")
        if np.any(self.body_parent[1:] < 0) or np.any(
            self.body_parent[1:] >= np.arange(1, n)
        ):
            raise RobotSpecError("bodies must be in parent-before-child order")
        if not np.array_equal(
            np.sort(np.asarray(self.joint_body, dtype=np.int64)),
            np.sort(np.flatnonzero(np.asarray(self.body_joint) >= 0)),
        ):
            raise RobotSpecError("joint_body and body_joint disagree")
        if np.any(np.asarray(self.position_upper) <= np.asarray(self.position_lower)):
            raise RobotSpecError("public joint upper limits must exceed lower limits")
        if np.any(np.asarray(self.coordinate_upper) <= np.asarray(self.coordinate_lower)):
            raise RobotSpecError("FK coordinate upper limits must exceed lower limits")
        if not set(np.unique(self.joint_type)).issubset({HINGE, SLIDE}):
            raise RobotSpecError("only one-DoF hinge and slide joints are supported")
        if np.any(self.anchor_body < 0) or np.any(self.anchor_body >= n):
            raise RobotSpecError("anchor body index leaves morphology")
        if not 0 <= int(self.base_reference_body) < n:
            raise RobotSpecError("base reference body index leaves morphology")
        if np.any(self.view_reference_body < 0) or np.any(
            self.view_reference_body >= n
        ):
            raise RobotSpecError("view reference body index leaves morphology")
        if not (0 <= self.fixed_tip_index < a and 0 <= self.moving_tip_index < a):
            raise RobotSpecError("tip indices leave ordered anchors")
        axis_norm = np.linalg.norm(self.joint_axis, axis=-1)
        if np.any(axis_norm < 1.0e-8):
            raise RobotSpecError("joint axes must be non-zero")

    def morphology_arrays(self) -> dict[str, np.ndarray]:
        """Return numeric graph fields suitable for ordinary checkpoint config."""

        self.validate()
        return {
            name: np.asarray(getattr(self, name)).copy()
            for name in (
                "body_parent",
                "body_translation",
                "body_quaternion_wxyz",
                "body_joint",
                "joint_body",
                "joint_type",
                "joint_axis",
                "joint_position",
                "position_lower",
                "position_upper",
                "coordinate_lower",
                "coordinate_upper",
                "anchor_body",
                "anchor_local_xyz",
                "base_reference_translation",
                "base_reference_quaternion_wxyz",
                "view_reference_body",
                "view_reference_translation",
                "view_reference_quaternion_wxyz",
            )
        }


def _quat_wxyz_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def _axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*axis.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).expand_as(skew)
    outer = axis[..., :, None] * axis[..., None, :]
    cosine = angle.cos()[..., None, None]
    sine = angle.sin()[..., None, None]
    return cosine * eye + (1.0 - cosine) * outer + sine * skew


def _transform(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(
        *rotation.shape[:-2], 4, 4, dtype=rotation.dtype, device=rotation.device
    )
    output[..., :3, :3] = rotation
    output[..., :3, 3] = translation
    output[..., 3, 3] = 1.0
    return output


def _inverse_transform(value: torch.Tensor) -> torch.Tensor:
    rotation = value[..., :3, :3]
    translation = value[..., :3, 3]
    inverse_rotation = rotation.transpose(-1, -2)
    return _transform(
        inverse_rotation,
        -torch.einsum("...ij,...j->...i", inverse_rotation, translation),
    )


class DifferentiableRobotKinematics(nn.Module):
    """Batched FK for a validated :class:`RobotSpec`."""

    def __init__(self, spec: RobotSpec) -> None:
        super().__init__()
        spec.validate()
        self.spec = spec
        for name, value in spec.morphology_arrays().items():
            array = np.asarray(value)
            tensor = torch.from_numpy(array.copy())
            if array.dtype.kind in "iu":
                tensor = tensor.long()
            else:
                tensor = tensor.float()
            self.register_buffer(name, tensor, persistent=True)

    @property
    def joint_count(self) -> int:
        return self.spec.joints

    @property
    def anchor_count(self) -> int:
        return self.spec.anchors

    @property
    def view_count(self) -> int:
        return self.spec.views

    def normalize_position(self, position: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(position)
        if value.shape[-1] != self.joint_count or not bool(torch.isfinite(value).all()):
            raise RobotSpecError("joint position must be finite and end in RobotSpec.joints")
        lower = self.position_lower.to(value)
        upper = self.position_upper.to(value)
        # Match the source SO-101 FK convention exactly: archived support can
        # extend beyond the public calibration interval, while MuJoCo geometry
        # is evaluated at the nearest physical joint limit.
        clipped = torch.maximum(torch.minimum(value, upper), lower)
        return 2.0 * (clipped - lower) / (upper - lower) - 1.0

    def position_to_coordinate(self, position: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_position(position)
        lower = self.coordinate_lower.to(normalized)
        upper = self.coordinate_upper.to(normalized)
        return lower + 0.5 * (normalized + 1.0) * (upper - lower)

    def forward(
        self, position: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return base-frame anchors, base-from-body and base-from-view-reference."""

        value = torch.as_tensor(position)
        if value.ndim != 2 or value.shape[1] != self.joint_count:
            raise RobotSpecError("joint position must be [B,J]")
        if not bool(torch.isfinite(value).all()):
            raise RobotSpecError("joint position must be finite")
        coordinate = self.position_to_coordinate(value)
        batch = len(value)
        dtype = value.dtype if value.is_floating_point() else torch.float32
        coordinate = coordinate.to(dtype=dtype)
        fixed_rotation = _quat_wxyz_to_matrix(
            self.body_quaternion_wxyz.to(device=value.device, dtype=dtype)
        )
        fixed = _transform(
            fixed_rotation,
            self.body_translation.to(device=value.device, dtype=dtype),
        )
        body_world: list[torch.Tensor] = []
        for body_index in range(self.spec.bodies):
            local = fixed[body_index].expand(batch, -1, -1)
            joint_index = int(self.body_joint[body_index])
            if joint_index >= 0:
                axis = self.joint_axis[joint_index].to(value).expand(batch, -1)
                origin = self.joint_position[joint_index].to(value).expand(batch, -1)
                if int(self.joint_type[joint_index]) == HINGE:
                    motion_rotation = _axis_angle_matrix(
                        axis, coordinate[:, joint_index]
                    )
                    rotated_origin = torch.einsum(
                        "bij,bj->bi", motion_rotation, origin
                    )
                    motion = _transform(
                        motion_rotation, origin - rotated_origin
                    )
                else:
                    eye = torch.eye(3, dtype=dtype, device=value.device).expand(
                        batch, -1, -1
                    )
                    motion = _transform(
                        eye, axis * coordinate[:, joint_index, None]
                    )
                local = local @ motion
            parent = int(self.body_parent[body_index])
            body_world.append(local if parent < 0 else body_world[parent] @ local)
        world = torch.stack(body_world, dim=1)
        # View zero's reference is the canonical base frame by construction.
        reference_local = _transform(
            _quat_wxyz_to_matrix(
                self.view_reference_quaternion_wxyz.to(value)
            ),
            self.view_reference_translation.to(value),
        )
        world_from_reference = torch.stack(
            [
                world[:, int(self.view_reference_body[index])]
                @ reference_local[index]
                for index in range(self.view_count)
            ],
            dim=1,
        )
        base_reference_local = _transform(
            _quat_wxyz_to_matrix(
                self.base_reference_quaternion_wxyz.to(value)
            ),
            self.base_reference_translation.to(value),
        )
        world_from_base = (
            world[:, int(self.spec.base_reference_body)] @ base_reference_local
        )
        base_from_world = _inverse_transform(world_from_base)
        base_from_body = base_from_world[:, None] @ world
        base_from_reference = base_from_world[:, None] @ world_from_reference
        homogeneous_anchor = torch.cat(
            (
                self.anchor_local_xyz.to(value),
                torch.ones(self.anchor_count, 1, dtype=dtype, device=value.device),
            ),
            dim=-1,
        )
        base_anchor = torch.stack(
            [
                torch.einsum(
                    "bij,j->bi",
                    base_from_body[:, int(self.anchor_body[index])],
                    homogeneous_anchor[index],
                )[..., :3]
                for index in range(self.anchor_count)
            ],
            dim=1,
        )
        return base_anchor, base_from_body, base_from_reference

    def anchors_in_view_references(self, position: torch.Tensor) -> torch.Tensor:
        anchors, _, base_from_reference = self(position)
        reference_from_base = _inverse_transform(base_from_reference)
        homogeneous = torch.cat(
            (anchors, torch.ones_like(anchors[..., :1])), dim=-1
        )
        return torch.einsum(
            "bvij,baj->bvai", reference_from_base, homogeneous
        )[..., :3]


def project_reference_points(
    points_reference: torch.Tensor,
    camera_from_reference: torch.Tensor,
    intrinsic: torch.Tensor,
    image_transform: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project ``[B,V,A,3]`` reference points to the 224-pixel input image."""

    points = torch.as_tensor(points_reference)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise RobotSpecError("reference points must be [B,V,A,3]")
    batch, views, anchors, _ = points.shape
    transform = torch.as_tensor(camera_from_reference, device=points.device, dtype=points.dtype)
    camera = torch.as_tensor(intrinsic, device=points.device, dtype=points.dtype)
    if transform.shape != (batch, views, 4, 4):
        raise RobotSpecError("camera_from_reference must be [B,V,4,4]")
    if camera.shape != (batch, views, 3, 3):
        raise RobotSpecError("intrinsic must be [B,V,3,3]")
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    camera_xyz = torch.einsum("bvij,bvaj->bvai", transform, homogeneous)[..., :3]
    depth = camera_xyz[..., 2]
    projected = torch.einsum("bvij,bvaj->bvai", camera, camera_xyz)
    pixel = projected[..., :2] / projected[..., 2:3].clamp_min(1.0e-6)
    if image_transform is not None:
        image = torch.as_tensor(image_transform, device=points.device, dtype=points.dtype)
        if image.shape != (batch, views, 3, 3):
            raise RobotSpecError("image_transform must be [B,V,3,3]")
        pixel_h = torch.cat((pixel, torch.ones_like(pixel[..., :1])), dim=-1)
        mapped = torch.einsum("bvij,bvaj->bvai", image, pixel_h)
        pixel = mapped[..., :2] / mapped[..., 2:3].clamp_min(1.0e-6)
    valid = (
        (depth > 1.0e-6)
        & (pixel[..., 0] >= 0.0)
        & (pixel[..., 0] < 224.0)
        & (pixel[..., 1] >= 0.0)
        & (pixel[..., 1] < 224.0)
    )
    return pixel, valid


def so101_robot_spec(
    model_path: str | Path = DEFAULT_SO101_MODEL,
    *,
    view_names: Sequence[str] = ("wrist", "column"),
) -> RobotSpec:
    """Build the SO-101 spec from the same MuJoCo file used by CAD/FK."""

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("building the SO-101 RobotSpec requires mujoco") from exc
    from fcpg.so101_kinematics import (
        SO101_CALIBRATION_KEYPOINT_NAMES,
        SO101_MUJOCO_JOINT_NAMES,
        SO101MuJoCoKinematics,
        normalized_positions_to_mujoco_qpos,
    )

    kinematics = SO101MuJoCoKinematics(model_path)
    model = kinematics.model
    canonical_zero = np.zeros((1, len(SO101_MUJOCO_JOINT_NAMES)), dtype=np.float64)
    qpos = normalized_positions_to_mujoco_qpos(
        canonical_zero, kinematics.all_joint_ranges
    )[0]
    kinematics._set_state(qpos)

    joint_ids = kinematics.all_joint_ids
    body_joint = np.full(model.nbody, -1, dtype=np.int64)
    joint_body = np.asarray(model.jnt_bodyid[joint_ids], dtype=np.int64)
    for ordinal, body in enumerate(joint_body):
        if body_joint[body] >= 0:
            raise RobotSpecError("SO-101 body unexpectedly has multiple controlled joints")
        body_joint[body] = ordinal
    base_site = kinematics.baseframe_site_id
    tool_site = kinematics.gripperframe_site_id
    base_body = int(model.site_bodyid[base_site])
    tool_body = int(model.site_bodyid[tool_site])
    if tuple(view_names) != ("wrist", "column"):
        raise RobotSpecError("V17.5 canonical SO-101 view order is wrist,column")

    anchor_body = np.concatenate(
        (
            np.asarray(model.jnt_bodyid[kinematics.joint_ids[:4]], dtype=np.int64),
            np.asarray(
                [kinematics.gripper_body_id, kinematics.moving_jaw_body_id],
                dtype=np.int64,
            ),
        )
    )
    anchor_local = np.concatenate(
        (
            np.asarray(model.jnt_pos[kinematics.joint_ids[:4]], dtype=np.float64),
            kinematics.fixed_tip_local[None],
            kinematics.moving_tip_local[None],
        ),
        axis=0,
    )
    public_lower = np.asarray([-100.0] * 5 + [0.0], dtype=np.float64)
    public_upper = np.asarray([100.0] * 6, dtype=np.float64)
    result = RobotSpec(
        name="so101",
        joint_names=tuple(SO101_MUJOCO_JOINT_NAMES),
        body_names=tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
            or f"body_{index}"
            for index in range(model.nbody)
        ),
        anchor_names=tuple(SO101_CALIBRATION_KEYPOINT_NAMES),
        view_names=tuple(view_names),
        body_parent=np.concatenate(
            ([-1], np.asarray(model.body_parentid[1:], dtype=np.int64))
        ),
        body_translation=np.asarray(model.body_pos, dtype=np.float64).copy(),
        body_quaternion_wxyz=np.asarray(model.body_quat, dtype=np.float64).copy(),
        body_joint=body_joint,
        joint_body=joint_body,
        joint_type=np.asarray(
            [HINGE if int(value) == int(mujoco.mjtJoint.mjJNT_HINGE) else SLIDE for value in model.jnt_type[joint_ids]],
            dtype=np.int64,
        ),
        joint_axis=np.asarray(model.jnt_axis[joint_ids], dtype=np.float64).copy(),
        joint_position=np.asarray(model.jnt_pos[joint_ids], dtype=np.float64).copy(),
        position_lower=public_lower,
        position_upper=public_upper,
        coordinate_lower=np.asarray(model.jnt_range[joint_ids, 0], dtype=np.float64).copy(),
        coordinate_upper=np.asarray(model.jnt_range[joint_ids, 1], dtype=np.float64).copy(),
        anchor_body=anchor_body,
        anchor_local_xyz=anchor_local,
        base_reference_body=base_body,
        base_reference_translation=np.asarray(
            model.site_pos[base_site], dtype=np.float64
        ).copy(),
        base_reference_quaternion_wxyz=np.asarray(
            model.site_quat[base_site], dtype=np.float64
        ).copy(),
        # Canonical view order is wrist then column.  CAD's camera transform
        # for wrist is camera<-gripperframe; column is camera<-baseframe.
        view_reference_body=np.asarray([tool_body, base_body], dtype=np.int64),
        view_reference_translation=np.stack(
            (
                np.asarray(model.site_pos[tool_site], dtype=np.float64),
                np.asarray(model.site_pos[base_site], dtype=np.float64),
            )
        ),
        view_reference_quaternion_wxyz=np.stack(
            (
                np.asarray(model.site_quat[tool_site], dtype=np.float64),
                np.asarray(model.site_quat[base_site], dtype=np.float64),
            )
        ),
        fixed_tip_index=4,
        moving_tip_index=5,
    )
    result.validate()
    return result


__all__ = [
    "DEFAULT_SO101_MODEL",
    "DifferentiableRobotKinematics",
    "HINGE",
    "RobotSpec",
    "RobotSpecError",
    "SLIDE",
    "project_reference_points",
    "so101_robot_spec",
]
