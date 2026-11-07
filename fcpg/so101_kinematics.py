"""Forward kinematics for LeRobot's normalized SO-101 joint convention."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

import numpy as np


SO101_JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
SO101_ARM_JOINT_NAMES = tuple(name.removesuffix(".pos") for name in SO101_JOINT_NAMES[:5])
SO101_MUJOCO_JOINT_NAMES = tuple(
    name.removesuffix(".pos") for name in SO101_JOINT_NAMES
)
SO101_ANCHOR_NAMES = (
    "gripper_center",
    "fixed_finger_tip",
    "moving_finger_tip",
)
SO101_CALIBRATION_KEYPOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_flex_joint",
    "wrist_flex_joint",
    "fixed_finger_tip",
    "moving_finger_tip",
)


@dataclass(frozen=True)
class SO101IKResult:
    """Result of one position-only inverse-kinematics solve."""

    state: np.ndarray
    achieved_xyz: np.ndarray
    target_xyz: np.ndarray
    error_m: float
    converged: bool
    iterations: int


@dataclass(frozen=True)
class SO101AnchorIKResult:
    """Result of fitting all three physical gripper anchors."""

    state: np.ndarray
    achieved_anchors: np.ndarray
    target_anchors: np.ndarray
    rms_error_m: float
    max_error_m: float
    converged: bool
    iterations: int


@dataclass(frozen=True)
class SO101TipIKFamily:
    """Every seed's fingertip solve, plus the distinct accepted subset."""

    results: tuple[SO101IKResult, ...]
    distinct: tuple[SO101IKResult, ...]


def normalized_arm_positions_to_radians(
    positions: np.ndarray,
    joint_ranges: np.ndarray,
) -> np.ndarray:
    """Map LeRobot SO-101 ``[-100, 100]`` arm positions to joint radians.

    The post-PR-777 SO-101 calibration places zero at the middle of each
    measured joint range. The gripper is intentionally excluded because
    LeRobot represents it as a linear ``[0, 100]`` command while the upstream
    MuJoCo model still uses a revolute approximation.
    """

    values = np.asarray(positions, dtype=np.float64)
    ranges = np.asarray(joint_ranges, dtype=np.float64)
    if values.shape[-1:] != (5,):
        raise ValueError(f"SO-101 arm positions must end in five joints, got {values.shape}")
    if ranges.shape != (5, 2):
        raise ValueError(f"SO-101 joint ranges must have shape (5, 2), got {ranges.shape}")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(ranges)):
        raise ValueError("SO-101 positions and joint ranges must be finite")
    if np.any(ranges[:, 1] <= ranges[:, 0]):
        raise ValueError("SO-101 joint range maxima must exceed minima")

    clipped = np.clip(values, -100.0, 100.0)
    fraction = (clipped + 100.0) / 200.0
    return ranges[:, 0] + fraction * (ranges[:, 1] - ranges[:, 0])


def radians_to_normalized_arm_positions(
    radians: np.ndarray,
    joint_ranges: np.ndarray,
) -> np.ndarray:
    """Map five SO-101 hinge angles to LeRobot's ``[-100, 100]`` convention."""

    values = np.asarray(radians, dtype=np.float64)
    ranges = np.asarray(joint_ranges, dtype=np.float64)
    if values.shape[-1:] != (5,):
        raise ValueError(f"SO-101 arm radians must end in five joints, got {values.shape}")
    if ranges.shape != (5, 2):
        raise ValueError(f"SO-101 joint ranges must have shape (5, 2), got {ranges.shape}")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(ranges)):
        raise ValueError("SO-101 radians and joint ranges must be finite")
    span = ranges[:, 1] - ranges[:, 0]
    if np.any(span <= 0):
        raise ValueError("SO-101 joint range maxima must exceed minima")
    clipped = np.clip(values, ranges[:, 0], ranges[:, 1])
    return 200.0 * (clipped - ranges[:, 0]) / span - 100.0


def normalized_positions_to_mujoco_qpos(
    positions: np.ndarray,
    joint_ranges: np.ndarray,
) -> np.ndarray:
    """Map six LeRobot SO-101 positions to the official MuJoCo hinges.

    The five arm joints use LeRobot's ``[-100, 100]`` calibration. The public
    SO-101 MuJoCo model still represents the gripper as a hinge, so its
    ``[0, 100]`` linear command is mapped across that hinge range only for
    pose visualization. End-effector XYZ does not depend on this approximation.
    """

    values = np.asarray(positions, dtype=np.float64)
    ranges = np.asarray(joint_ranges, dtype=np.float64)
    if values.shape[-1:] != (6,):
        raise ValueError(f"SO-101 positions must end in six joints, got {values.shape}")
    if ranges.shape != (6, 2):
        raise ValueError(f"SO-101 joint ranges must have shape (6, 2), got {ranges.shape}")
    result = np.empty_like(values, dtype=np.float64)
    result[..., :5] = normalized_arm_positions_to_radians(values[..., :5], ranges[:5])
    gripper_fraction = np.clip(values[..., 5], 0.0, 100.0) / 100.0
    result[..., 5] = ranges[5, 0] + gripper_fraction * (
        ranges[5, 1] - ranges[5, 0]
    )
    return result


class SO101MuJoCoKinematics:
    """Evaluate the official SO-101 ``gripperframe`` site in base coordinates."""

    base_frame_id = "baseframe"
    tool_frame_id = "gripperframe"

    def __init__(self, model_path: str | Path) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - environment-specific dependency
            raise RuntimeError("SO-101 FK requires the mujoco Python package") from exc

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SO-101 MuJoCo model not found: {path}")
        self.model_path = path
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self._forward_lock = threading.RLock()

        all_joint_ids = []
        all_qpos_addresses = []
        for name in SO101_MUJOCO_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"SO-101 MuJoCo model is missing joint {name!r}")
            all_joint_ids.append(joint_id)
            all_qpos_addresses.append(int(self.model.jnt_qposadr[joint_id]))
        self.all_joint_ids = np.asarray(all_joint_ids, dtype=np.int32)
        self.all_qpos_addresses = np.asarray(all_qpos_addresses, dtype=np.int32)
        self.joint_ids = self.all_joint_ids[:5]
        self.qpos_addresses = self.all_qpos_addresses[:5]
        self.dof_addresses = np.asarray(
            self.model.jnt_dofadr[self.joint_ids], dtype=np.int32
        ).copy()
        self.all_dof_addresses = np.asarray(
            self.model.jnt_dofadr[self.all_joint_ids], dtype=np.int32
        ).copy()
        self.all_joint_ranges = np.asarray(
            self.model.jnt_range[self.all_joint_ids], dtype=np.float64
        ).copy()
        self.joint_ranges = np.asarray(
            self.model.jnt_range[self.joint_ids], dtype=np.float64
        ).copy()
        self.baseframe_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.base_frame_id
        )
        if self.baseframe_site_id < 0:
            raise ValueError(
                f"SO-101 MuJoCo model is missing site {self.base_frame_id!r}"
            )
        self.gripperframe_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.tool_frame_id
        )
        if self.gripperframe_site_id < 0:
            raise ValueError(
                f"SO-101 MuJoCo model is missing site {self.tool_frame_id!r}"
            )
        self.gripper_body_id = self._body_id("gripper")
        self.moving_jaw_body_id = self._body_id("moving_jaw_so101_v1")
        mujoco.mj_forward(self.model, self.data)
        tool_direction = (
            np.asarray(self.data.site_xpos[self.gripperframe_site_id], dtype=np.float64)
            - np.asarray(self.data.xpos[self.gripper_body_id], dtype=np.float64)
        )
        tool_direction /= np.linalg.norm(tool_direction)
        self.fixed_tip_local = self._mesh_tip_local_offset(
            "wrist_roll_follower_so101_v1",
            self.gripper_body_id,
            tool_direction,
        )
        self.moving_tip_local = self._mesh_tip_local_offset(
            "moving_jaw_so101_v1",
            self.moving_jaw_body_id,
            tool_direction,
        )
        self.robot_body_ids = self._robot_body_subtree()

    def _robot_body_subtree(self) -> frozenset[int]:
        """Collect every body belonging to the arm's own kinematic subtree.

        The model may be a full scene, so self-collision has to be separated from
        contact with scene objects.  A body is arm-owned when walking up
        ``body_parentid`` reaches the body that carries the first arm joint.
        """

        first_moving = int(self.model.jnt_bodyid[int(self.all_joint_ids[0])])
        # Start at the fixed base link when there is one, so base-vs-arm contact
        # counts as self-collision too.
        parent = int(self.model.body_parentid[first_moving])
        root = first_moving if parent == 0 else parent
        parent_of_root = int(self.model.body_parentid[root])
        owned = {root}
        for body_id in range(int(self.model.nbody)):
            cursor = body_id
            while cursor != 0 and cursor != parent_of_root:
                if cursor in owned:
                    owned.add(body_id)
                    break
                cursor = int(self.model.body_parentid[cursor])
        return frozenset(owned)

    def self_collision_penetration_m(self, state: np.ndarray) -> float:
        """Return the deepest arm-vs-arm penetration at one normalized state.

        Zero means the pose is self-collision free.  This is a property of the
        robot's own geometry, not of the scene, so it is equally available to an
        offline planner and to a deployed controller.
        """

        values = np.asarray(state, dtype=np.float64)
        if values.shape != (6,):
            raise ValueError(f"SO-101 state must have six joints, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("SO-101 state must be finite")
        # MjData is not thread-safe; serialize with forward/IK under ThreadingHTTPServer.
        with self._forward_lock:
            self._set_state(
                normalized_positions_to_mujoco_qpos(values, self.all_joint_ranges)
            )
            worst = 0.0
            for index in range(int(self.data.ncon)):
                contact = self.data.contact[index]
                first = int(self.model.geom_bodyid[int(contact.geom1)])
                second = int(self.model.geom_bodyid[int(contact.geom2)])
                if first in self.robot_body_ids and second in self.robot_body_ids:
                    worst = max(worst, -float(contact.dist))
        return worst

    def _body_id(self, name: str) -> int:
        body_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, name
        )
        if body_id < 0:
            raise ValueError(f"SO-101 MuJoCo model is missing body {name!r}")
        return int(body_id)

    def _mesh_tip_local_offset(
        self,
        mesh_name: str,
        body_id: int,
        tool_direction_world: np.ndarray,
    ) -> np.ndarray:
        """Locate a jaw tip from the distal surface of its official visual mesh."""

        mesh_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_MESH, mesh_name
        )
        if mesh_id < 0:
            raise ValueError(f"SO-101 MuJoCo model is missing mesh {mesh_name!r}")
        candidates = np.flatnonzero(
            (np.asarray(self.model.geom_dataid) == mesh_id)
            & (np.asarray(self.model.geom_bodyid) == body_id)
            & (np.asarray(self.model.geom_group) == 2)
        )
        if len(candidates) != 1:
            raise ValueError(
                f"expected one visual geom for mesh {mesh_name!r}, got {len(candidates)}"
            )
        geom_id = int(candidates[0])
        vertex_start = int(self.model.mesh_vertadr[mesh_id])
        vertex_stop = vertex_start + int(self.model.mesh_vertnum[mesh_id])
        vertices = np.asarray(
            self.model.mesh_vert[vertex_start:vertex_stop], dtype=np.float64
        )
        geom_rotation = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        world_vertices = (
            vertices @ geom_rotation.T
            + np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64)
        )
        scores = world_vertices @ np.asarray(tool_direction_world, dtype=np.float64)
        distal = world_vertices[scores >= np.quantile(scores, 0.995)]
        if len(distal) < 3:
            raise ValueError(f"mesh {mesh_name!r} has insufficient distal vertices")
        tip_world = np.median(distal, axis=0)
        body_rotation = np.asarray(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        return body_rotation.T @ (
            tip_world - np.asarray(self.data.xpos[body_id], dtype=np.float64)
        )

    def _set_state(self, qpos: np.ndarray) -> None:
        self.data.qpos[self.all_qpos_addresses] = qpos
        self._mujoco.mj_forward(self.model, self.data)

    def _set_kinematic_state(self, qpos: np.ndarray) -> None:
        """Update position-dependent transforms without collision/contact work.

        ``forward_anchors`` uses only body positions and rotations.  MuJoCo's
        kinematics pass produces those same fields as ``mj_forward`` for the
        pinned SO-101 model, while self-contact paths continue to use the full
        forward pass above.
        """
        self.data.qpos[self.all_qpos_addresses] = qpos
        self._mujoco.mj_kinematics(self.model, self.data)

    def _body_local_to_world(self, body_id: int, local_point: np.ndarray) -> np.ndarray:
        rotation = np.asarray(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        return (
            np.asarray(self.data.xpos[body_id], dtype=np.float64)
            + rotation @ np.asarray(local_point, dtype=np.float64)
        )

    def forward(self, states: np.ndarray) -> np.ndarray:
        """Return base-frame gripper-center XYZ for normalized six-joint states."""

        values = np.asarray(states, dtype=np.float64)
        if values.shape[-1:] != (6,):
            raise ValueError(f"SO-101 states must end in six joints, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("SO-101 states must be finite")
        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, 6)
        all_qpos = normalized_positions_to_mujoco_qpos(flat, self.all_joint_ranges)
        xyz = np.empty((len(flat), 3), dtype=np.float64)
        with self._forward_lock:
            for index, qpos in enumerate(all_qpos):
                self._set_state(qpos)
                xyz[index] = self.data.site_xpos[self.gripperframe_site_id]
        return xyz.reshape(*leading_shape, 3)

    def forward_gripper_poses(self, states: np.ndarray) -> np.ndarray:
        """Return ``T_base_gripperframe`` for each normalized joint row.

        This is the full-SE(3) counterpart of :meth:`forward`.  It exposes only
        known robot FK and is suitable for composing a measured hand-eye
        transform; it does not infer whether an external entity is attached.
        """

        values = np.asarray(states, dtype=np.float64)
        if values.shape[-1:] != (6,):
            raise ValueError(f"SO-101 states must end in six joints, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("SO-101 states must be finite")
        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, 6)
        all_qpos = normalized_positions_to_mujoco_qpos(flat, self.all_joint_ranges)
        poses = np.broadcast_to(
            np.eye(4, dtype=np.float64), (len(flat), 4, 4)
        ).copy()
        with self._forward_lock:
            for index, qpos in enumerate(all_qpos):
                self._set_state(qpos)
                rotation_world_base = np.asarray(
                    self.data.site_xmat[self.baseframe_site_id], dtype=np.float64
                ).reshape(3, 3)
                position_world_base = np.asarray(
                    self.data.site_xpos[self.baseframe_site_id], dtype=np.float64
                )
                rotation_world_tool = np.asarray(
                    self.data.site_xmat[self.gripperframe_site_id], dtype=np.float64
                ).reshape(3, 3)
                position_world_tool = np.asarray(
                    self.data.site_xpos[self.gripperframe_site_id], dtype=np.float64
                )
                poses[index, :3, :3] = rotation_world_base.T @ rotation_world_tool
                poses[index, :3, 3] = rotation_world_base.T @ (
                    position_world_tool - position_world_base
                )
        return poses.reshape(*leading_shape, 4, 4)

    def forward_anchors(self, states: np.ndarray) -> np.ndarray:
        """Return center, fixed-tip and moving-tip XYZ in the MuJoCo base frame."""

        values = np.asarray(states, dtype=np.float64)
        if values.shape[-1:] != (6,):
            raise ValueError(f"SO-101 states must end in six joints, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("SO-101 states must be finite")
        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, 6)
        anchors = self._anchors(normalized_positions_to_mujoco_qpos(flat, self.all_joint_ranges))
        return anchors.reshape(*leading_shape, len(SO101_ANCHOR_NAMES), 3)

    def forward_anchors_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """``forward_anchors`` from MuJoCo hinge positions [..., 6] (the arm's radians, the gripper hinge) as given: no
        joint range clips them (a robot's calibrated range can be wider than the model's, as the wrist roll's)."""

        values = np.asarray(qpos, dtype=np.float64)
        if values.shape[-1:] != (6,):
            raise ValueError(f"SO-101 hinge positions must end in six joints, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("SO-101 hinge positions must be finite")
        anchors = self._anchors(values.reshape(-1, 6))
        return anchors.reshape(*values.shape[:-1], len(SO101_ANCHOR_NAMES), 3)

    def _anchors(self, all_qpos: np.ndarray) -> np.ndarray:
        anchors = np.empty((len(all_qpos), len(SO101_ANCHOR_NAMES), 3), dtype=np.float64)
        with self._forward_lock:
            for index, qpos in enumerate(all_qpos):
                self._set_kinematic_state(qpos)
                fixed_rotation = self.data.xmat[self.gripper_body_id].reshape(3, 3)
                moving_rotation = self.data.xmat[self.moving_jaw_body_id].reshape(3, 3)
                fixed_tip = self.data.xpos[self.gripper_body_id] + fixed_rotation @ self.fixed_tip_local
                moving_tip = self.data.xpos[self.moving_jaw_body_id] + moving_rotation @ self.moving_tip_local
                anchors[index, 0] = 0.5 * (fixed_tip + moving_tip)
                anchors[index, 1] = fixed_tip
                anchors[index, 2] = moving_tip
        return anchors

    def forward_tip_jacobians(self, states: np.ndarray) -> np.ndarray:
        """Return analytic fingertip XYZ Jacobians in canonical joint units.

        The final axis follows the six LeRobot coordinates rather than MuJoCo
        radians.  This makes ``J @ (state[t] - state[t-1])`` directly usable by
        the cached FCPG-H CAD motion signature without numerical differencing.
        No camera model is involved.
        """

        values = np.asarray(states, dtype=np.float64)
        if values.shape[-1:] != (6,):
            raise ValueError(f"SO-101 states must end in six joints, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("SO-101 states must be finite")
        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, 6)
        all_qpos = normalized_positions_to_mujoco_qpos(flat, self.all_joint_ranges)
        result = np.empty((len(flat), 2, 3, 6), dtype=np.float64)
        qpos_per_canonical = np.empty(6, dtype=np.float64)
        qpos_per_canonical[:5] = (
            self.all_joint_ranges[:5, 1] - self.all_joint_ranges[:5, 0]
        ) / 200.0
        qpos_per_canonical[5] = (
            self.all_joint_ranges[5, 1] - self.all_joint_ranges[5, 0]
        ) / 100.0
        with self._forward_lock:
            for index, qpos in enumerate(all_qpos):
                self._set_state(qpos)
                base_rotation = np.asarray(
                    self.data.site_xmat[self.baseframe_site_id], dtype=np.float64
                ).reshape(3, 3)
                tips = (
                    (
                        self.gripper_body_id,
                        self._body_local_to_world(self.gripper_body_id, self.fixed_tip_local),
                    ),
                    (
                        self.moving_jaw_body_id,
                        self._body_local_to_world(self.moving_jaw_body_id, self.moving_tip_local),
                    ),
                )
                for tip_index, (body_id, point) in enumerate(tips):
                    jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
                    jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
                    self._mujoco.mj_jac(
                        self.model,
                        self.data,
                        jacobian_position,
                        jacobian_rotation,
                        point,
                        body_id,
                    )
                    selected = jacobian_position[:, self.all_dof_addresses]
                    result[index, tip_index] = (
                        base_rotation.T @ selected
                    ) * qpos_per_canonical[None]
        return result.reshape(*leading_shape, 2, 3, 6)

    def forward_tips_and_jacobians(
        self, states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return fingertip XYZ and Jacobians from one MuJoCo forward pass.

        This is the preprocessing fast path for FCPG-H.  Its outputs are
        numerically identical to ``forward_anchors(states)[..., 1:, :]`` and
        :meth:`forward_tip_jacobians`, but it avoids setting each simulator
        state twice across the multi-million-frame canonical index.
        """

        values = np.asarray(states, dtype=np.float64)
        if values.shape[-1:] != (6,):
            raise ValueError(f"SO-101 states must end in six joints, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("SO-101 states must be finite")
        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, 6)
        all_qpos = normalized_positions_to_mujoco_qpos(flat, self.all_joint_ranges)
        tips_out = np.empty((len(flat), 2, 3), dtype=np.float64)
        jacobians_out = np.empty((len(flat), 2, 3, 6), dtype=np.float64)
        qpos_per_canonical = np.empty(6, dtype=np.float64)
        qpos_per_canonical[:5] = (
            self.all_joint_ranges[:5, 1] - self.all_joint_ranges[:5, 0]
        ) / 200.0
        qpos_per_canonical[5] = (
            self.all_joint_ranges[5, 1] - self.all_joint_ranges[5, 0]
        ) / 100.0
        with self._forward_lock:
            for index, qpos in enumerate(all_qpos):
                self._set_state(qpos)
                base_position = np.asarray(
                    self.data.site_xpos[self.baseframe_site_id], dtype=np.float64
                )
                base_rotation = np.asarray(
                    self.data.site_xmat[self.baseframe_site_id], dtype=np.float64
                ).reshape(3, 3)
                world_tips = (
                    (
                        self.gripper_body_id,
                        self._body_local_to_world(self.gripper_body_id, self.fixed_tip_local),
                    ),
                    (
                        self.moving_jaw_body_id,
                        self._body_local_to_world(self.moving_jaw_body_id, self.moving_tip_local),
                    ),
                )
                for tip_index, (body_id, point) in enumerate(world_tips):
                    tips_out[index, tip_index] = base_rotation.T @ (point - base_position)
                    jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
                    jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
                    self._mujoco.mj_jac(
                        self.model,
                        self.data,
                        jacobian_position,
                        jacobian_rotation,
                        point,
                        body_id,
                    )
                    selected = jacobian_position[:, self.all_dof_addresses]
                    jacobians_out[index, tip_index] = (
                        base_rotation.T @ selected
                    ) * qpos_per_canonical[None]
        return (
            tips_out.reshape(*leading_shape, 2, 3),
            jacobians_out.reshape(*leading_shape, 2, 3, 6),
        )

    def forward_calibration_keypoints(self, states: np.ndarray) -> np.ndarray:
        """Return four chain joints plus two finger tips in the base frame.

        These six fixed-identity CAD points are intended for the preprocessing
        heatmap teacher and metric camera fit.  They are not extra FCPG output
        heads; the deployed FCPG predicts residuals for the final two points
        only.  Including both tips also makes the teacher itself an explicit
        baseline for the FCPG labels instead of hiding tip error behind a
        camera-fit projection.
        """

        values = np.asarray(states, dtype=np.float64)
        if values.shape[-1:] != (6,):
            raise ValueError(f"SO-101 states must end in six joints, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("SO-101 states must be finite")
        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, 6)
        all_qpos = normalized_positions_to_mujoco_qpos(flat, self.all_joint_ranges)
        points = np.empty(
            (len(flat), len(SO101_CALIBRATION_KEYPOINT_NAMES), 3),
            dtype=np.float64,
        )
        with self._forward_lock:
            for index, qpos in enumerate(all_qpos):
                self._set_state(qpos)
                base_position = np.asarray(
                    self.data.site_xpos[self.baseframe_site_id], dtype=np.float64
                )
                base_rotation = np.asarray(
                    self.data.site_xmat[self.baseframe_site_id], dtype=np.float64
                ).reshape(3, 3)
                joint_centers_world = np.asarray(
                    self.data.xanchor[self.joint_ids[:4]], dtype=np.float64
                )
                fixed_tip = self._body_local_to_world(
                    self.gripper_body_id, self.fixed_tip_local
                )
                moving_tip = self._body_local_to_world(
                    self.moving_jaw_body_id, self.moving_tip_local
                )
                world = np.concatenate(
                    (joint_centers_world, fixed_tip[None], moving_tip[None]), axis=0
                )
                points[index] = (world - base_position[None]) @ base_rotation
        return points.reshape(
            *leading_shape, len(SO101_CALIBRATION_KEYPOINT_NAMES), 3
        )

    def forward_calibration_keypoints_and_jacobians(
        self, states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return all six fixed CAD parts and canonical-unit XYZ Jacobians.

        This is the schedule-bound preprocessing path for candidate-first
        FCPG-H.  It uses only the official robot model and measured joint state;
        no image, camera calibration, source identity, or task metadata enters
        either output.
        """

        values = np.asarray(states, dtype=np.float64)
        if values.shape[-1:] != (6,):
            raise ValueError(f"SO-101 states must end in six joints, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("SO-101 states must be finite")
        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, 6)
        all_qpos = normalized_positions_to_mujoco_qpos(flat, self.all_joint_ranges)
        part_count = len(SO101_CALIBRATION_KEYPOINT_NAMES)
        points_out = np.empty((len(flat), part_count, 3), dtype=np.float64)
        jacobians_out = np.empty((len(flat), part_count, 3, 6), dtype=np.float64)
        qpos_per_canonical = np.empty(6, dtype=np.float64)
        qpos_per_canonical[:5] = (
            self.all_joint_ranges[:5, 1] - self.all_joint_ranges[:5, 0]
        ) / 200.0
        qpos_per_canonical[5] = (
            self.all_joint_ranges[5, 1] - self.all_joint_ranges[5, 0]
        ) / 100.0
        with self._forward_lock:
            for index, qpos in enumerate(all_qpos):
                self._set_state(qpos)
                base_position = np.asarray(
                    self.data.site_xpos[self.baseframe_site_id], dtype=np.float64
                )
                base_rotation = np.asarray(
                    self.data.site_xmat[self.baseframe_site_id], dtype=np.float64
                ).reshape(3, 3)
                joint_points = np.asarray(
                    self.data.xanchor[self.joint_ids[:4]], dtype=np.float64
                )
                part_specs = [
                    (int(self.model.jnt_bodyid[joint_id]), point)
                    for joint_id, point in zip(self.joint_ids[:4], joint_points, strict=True)
                ]
                part_specs.extend(
                    (
                        (
                            self.gripper_body_id,
                            self._body_local_to_world(
                                self.gripper_body_id, self.fixed_tip_local
                            ),
                        ),
                        (
                            self.moving_jaw_body_id,
                            self._body_local_to_world(
                                self.moving_jaw_body_id, self.moving_tip_local
                            ),
                        ),
                    )
                )
                for part_index, (body_id, world_point) in enumerate(part_specs):
                    points_out[index, part_index] = base_rotation.T @ (
                        world_point - base_position
                    )
                    jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
                    jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
                    self._mujoco.mj_jac(
                        self.model,
                        self.data,
                        jacobian_position,
                        jacobian_rotation,
                        world_point,
                        body_id,
                    )
                    selected = jacobian_position[:, self.all_dof_addresses]
                    jacobians_out[index, part_index] = (
                        base_rotation.T @ selected
                    ) * qpos_per_canonical[None]
        return (
            points_out.reshape(*leading_shape, part_count, 3),
            jacobians_out.reshape(*leading_shape, part_count, 3, 6),
        )

    def forward_anchor_frames(
        self, states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return base-frame and gripperframe-local anchor coordinates.

        Local coordinates are the correct nominal geometry for an eye-in-hand
        camera. The moving jaw still changes with the gripper command.
        """

        values = np.asarray(states, dtype=np.float64)
        if values.shape[-1:] != (6,):
            raise ValueError(f"SO-101 states must end in six joints, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("SO-101 states must be finite")
        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, 6)
        all_qpos = normalized_positions_to_mujoco_qpos(flat, self.all_joint_ranges)
        base = np.empty((len(flat), len(SO101_ANCHOR_NAMES), 3), dtype=np.float64)
        local = np.empty_like(base)
        with self._forward_lock:
            for index, qpos in enumerate(all_qpos):
                self._set_state(qpos)
                fixed_tip = self._body_local_to_world(
                    self.gripper_body_id, self.fixed_tip_local
                )
                moving_tip = self._body_local_to_world(
                    self.moving_jaw_body_id, self.moving_tip_local
                )
                anchors = np.stack(
                    (0.5 * (fixed_tip + moving_tip), fixed_tip, moving_tip), axis=0
                )
                site_position = np.asarray(
                    self.data.site_xpos[self.gripperframe_site_id], dtype=np.float64
                )
                site_rotation = np.asarray(
                    self.data.site_xmat[self.gripperframe_site_id], dtype=np.float64
                ).reshape(3, 3)
                base[index] = anchors
                local[index] = (anchors - site_position[None]) @ site_rotation
        return (
            base.reshape(*leading_shape, len(SO101_ANCHOR_NAMES), 3),
            local.reshape(*leading_shape, len(SO101_ANCHOR_NAMES), 3),
        )

    def clipping_fraction(self, states: np.ndarray) -> float:
        values = np.asarray(states, dtype=np.float64)
        if values.shape[-1:] != (6,):
            raise ValueError(f"SO-101 states must end in six joints, got {values.shape}")
        return float(np.mean(np.any(np.abs(values[..., :5]) > 100.0, axis=-1)))

    def inverse_position(
        self,
        target_xyz: np.ndarray,
        initial_state: np.ndarray,
        *,
        max_iterations: int = 100,
        damping: float = 1e-3,
        tolerance_m: float = 1e-3,
        max_step_normalized: float = 8.0,
        posture_weight: float = 1e-4,
    ) -> SO101IKResult:
        """Solve a bounded position-only IK problem for ``gripperframe``.

        The first five normalized LeRobot joints are optimized. The gripper
        command is copied bit-for-bit from ``initial_state`` because arm IK
        cannot decide when an object should be grasped or released.
        """

        target = np.asarray(target_xyz, dtype=np.float64)
        initial = np.asarray(initial_state, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_xyz must be one finite XYZ vector")
        if initial.shape != (6,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_state must be one finite six-joint vector")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if damping <= 0 or not np.isfinite(damping):
            raise ValueError("damping must be finite and positive")
        if tolerance_m <= 0 or not np.isfinite(tolerance_m):
            raise ValueError("tolerance_m must be finite and positive")
        if max_step_normalized <= 0 or not np.isfinite(max_step_normalized):
            raise ValueError("max_step_normalized must be finite and positive")
        if posture_weight < 0 or not np.isfinite(posture_weight):
            raise ValueError("posture_weight must be finite and non-negative")

        # Hold the FK lock for the whole solve. inverse_position mutates MjData
        # via _set_state/mj_jacSite; releasing between steps races concurrent
        # forward_* callers under ThreadingHTTPServer and has SIGSEGV'd live serve.
        with self._forward_lock:
            state = initial.copy()
            state[:5] = np.clip(state[:5], -100.0, 100.0)
            reference_radians = normalized_arm_positions_to_radians(
                state[:5], self.joint_ranges
            )
            best_state = state.copy()
            best_xyz = self.forward(state)
            best_error = float(np.linalg.norm(target - best_xyz))
            iterations = 0

            for iteration in range(max_iterations):
                iterations = iteration + 1
                qpos = normalized_positions_to_mujoco_qpos(
                    state, self.all_joint_ranges
                )
                self._set_state(qpos)
                current_xyz = np.asarray(
                    self.data.site_xpos[self.gripperframe_site_id], dtype=np.float64
                ).copy()
                residual = target - current_xyz
                error = float(np.linalg.norm(residual))
                if error < best_error:
                    best_error = error
                    best_xyz = current_xyz
                    best_state = state.copy()
                if error <= tolerance_m:
                    break

                jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
                jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
                self._mujoco.mj_jacSite(
                    self.model,
                    self.data,
                    jacobian_position,
                    jacobian_rotation,
                    self.gripperframe_site_id,
                )
                jacobian = jacobian_position[:, self.dof_addresses]
                current_radians = qpos[:5]
                regularizer = damping * damping + posture_weight
                system = jacobian.T @ jacobian + regularizer * np.eye(5)
                rhs = jacobian.T @ residual
                if posture_weight:
                    rhs += posture_weight * (reference_radians - current_radians)
                try:
                    delta_radians = np.linalg.solve(system, rhs)
                except np.linalg.LinAlgError:
                    delta_radians = np.linalg.lstsq(system, rhs, rcond=None)[0]

                normalized_scale = 200.0 / (
                    self.joint_ranges[:, 1] - self.joint_ranges[:, 0]
                )
                delta_normalized = delta_radians * normalized_scale
                largest = float(np.max(np.abs(delta_normalized)))
                if largest > max_step_normalized:
                    delta_normalized *= max_step_normalized / largest
                state[:5] = np.clip(state[:5] + delta_normalized, -100.0, 100.0)

            final_error = float(np.linalg.norm(target - best_xyz))
            return SO101IKResult(
                state=best_state.astype(np.float32),
                achieved_xyz=np.asarray(best_xyz, dtype=np.float64),
                target_xyz=target.copy(),
                error_m=final_error,
                converged=bool(final_error <= tolerance_m),
                iterations=iterations,
            )

    def inverse_anchors(
        self,
        target_anchors: np.ndarray,
        initial_state: np.ndarray,
        *,
        max_iterations: int = 80,
        damping: float = 1e-4,
        tolerance_m: float = 1e-3,
        finite_difference_step: float = 0.25,
        max_step_normalized: float = 8.0,
        posture_weight: float = 0.0,
        position_lower: np.ndarray | None = None,
        position_upper: np.ndarray | None = None,
    ) -> SO101AnchorIKResult:
        """Fit center and both fingertip targets in normalized joint space.

        The sixth coordinate is optimized from fingertip separation, so the
        output retains the LeWM-predicted open/close trajectory without a
        hand-written grasp or release phase.
        """

        target = np.asarray(target_anchors, dtype=np.float64)
        initial = np.asarray(initial_state, dtype=np.float64)
        if target.shape != (len(SO101_ANCHOR_NAMES), 3) or not np.all(
            np.isfinite(target)
        ):
            raise ValueError("target_anchors must be finite [3,3]")
        if initial.shape != (6,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_state must be one finite six-joint vector")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        positive = (
            damping,
            tolerance_m,
            finite_difference_step,
            max_step_normalized,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("anchor IK values must be finite and positive")
        if not np.isfinite(posture_weight) or posture_weight < 0:
            raise ValueError("posture_weight must be finite and non-negative")

        lower = np.asarray(
            [-100.0] * 5 + [0.0]
            if position_lower is None
            else position_lower,
            dtype=np.float64,
        )
        upper = np.asarray(
            [100.0] * 6 if position_upper is None else position_upper,
            dtype=np.float64,
        )
        if lower.shape != (6,) or upper.shape != (6,):
            raise ValueError("anchor IK position bounds must be six-vectors")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("anchor IK position bounds must be finite")
        if np.any(upper < lower):
            raise ValueError("anchor IK upper bounds cannot be below lower bounds")
        state = np.clip(initial, lower, upper)
        reference = state.copy()
        best_state = state.copy()
        best_anchors = np.asarray(self.forward_anchors(state), dtype=np.float64)
        best_errors = np.linalg.norm(best_anchors - target, axis=-1)
        best_max_error = float(np.max(best_errors))
        best_rms_error = float(np.sqrt(np.mean(np.square(best_anchors - target))))
        iterations = 0

        for iteration in range(max_iterations):
            iterations = iteration + 1
            current = np.asarray(self.forward_anchors(state), dtype=np.float64)
            errors = np.linalg.norm(current - target, axis=-1)
            max_error = float(np.max(errors))
            rms_error = float(np.sqrt(np.mean(np.square(current - target))))
            if max_error < best_max_error:
                best_state = state.copy()
                best_anchors = current.copy()
                best_max_error = max_error
                best_rms_error = rms_error
            if max_error <= tolerance_m:
                break

            jacobian = np.empty((target.size, 6), dtype=np.float64)
            for joint in range(6):
                plus = state.copy()
                minus = state.copy()
                plus[joint] = min(upper[joint], plus[joint] + finite_difference_step)
                minus[joint] = max(lower[joint], minus[joint] - finite_difference_step)
                denominator = plus[joint] - minus[joint]
                if denominator <= 0:
                    jacobian[:, joint] = 0.0
                    continue
                plus_anchors = np.asarray(
                    self.forward_anchors(plus), dtype=np.float64
                )
                minus_anchors = np.asarray(
                    self.forward_anchors(minus), dtype=np.float64
                )
                jacobian[:, joint] = (
                    (plus_anchors - minus_anchors) / denominator
                ).reshape(-1)

            residual = (target - current).reshape(-1)
            regularizer = damping * damping + posture_weight
            system = jacobian.T @ jacobian + regularizer * np.eye(6)
            rhs = jacobian.T @ residual
            if posture_weight:
                rhs += posture_weight * (reference - state)
            try:
                delta = np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(system, rhs, rcond=None)[0]
            largest = float(np.max(np.abs(delta)))
            if largest > max_step_normalized:
                delta *= max_step_normalized / largest
            state = np.clip(state + delta, lower, upper)

        return SO101AnchorIKResult(
            state=best_state.astype(np.float32),
            achieved_anchors=best_anchors,
            target_anchors=target.copy(),
            rms_error_m=best_rms_error,
            max_error_m=best_max_error,
            converged=bool(best_max_error <= tolerance_m),
            iterations=iterations,
        )

    def fixed_tip_joint_lever_arms_m(self, states: np.ndarray) -> np.ndarray:
        """Perpendicular distance from each arm joint axis to the fixed fingertip.

        The lever is ``|| (tip - xanchor) × xaxis ||``, the radius that converts
        a small joint-angle error into fingertip arc length.  The gripper hinge
        is omitted because it does not move the fixed fingertip.
        """

        values = np.asarray(states, dtype=np.float64)
        if values.shape[-1:] != (6,):
            raise ValueError(f"SO-101 states must end in six joints, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("SO-101 states must be finite")
        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, 6)
        levers = np.empty((len(flat), 5), dtype=np.float64)
        all_qpos = normalized_positions_to_mujoco_qpos(flat, self.all_joint_ranges)
        with self._forward_lock:
            for index, qpos in enumerate(all_qpos):
                self._set_state(qpos)
                tip = self._body_local_to_world(
                    self.gripper_body_id, self.fixed_tip_local
                )
                for joint in range(5):
                    joint_id = int(self.joint_ids[joint])
                    anchor = np.asarray(
                        self.data.xanchor[joint_id], dtype=np.float64
                    )
                    axis = np.asarray(self.data.xaxis[joint_id], dtype=np.float64)
                    axis_norm = float(np.linalg.norm(axis))
                    if axis_norm <= 0:
                        raise ValueError(f"joint {joint} has a zero axis")
                    axis = axis / axis_norm
                    levers[index, joint] = float(
                        np.linalg.norm(np.cross(tip - anchor, axis))
                    )
        return levers.reshape(*leading_shape, 5)

    def inverse_tip_position(
        self,
        target_xyz: np.ndarray,
        initial_state: np.ndarray,
        *,
        max_iterations: int = 100,
        damping: float = 1e-3,
        tolerance_m: float = 1e-3,
        max_step_normalized: float = 8.0,
        posture_weight: float = 1e-4,
        nullspace_preference: np.ndarray | None = None,
    ) -> SO101IKResult:
        """Solve a bounded position-only IK problem for the fixed fingertip.

        Additive counterpart of :meth:`inverse_position`, which still targets
        ``gripperframe`` and is left bit-identical.  ``nullspace_preference``
        (five or six normalized joints) replaces the seed as the posture pull
        when the arm is redundant, so a caller can grow a family of
        configurations that share one 3D datum.  ``error_m`` is the achieved
        fingertip residual so the caller can reject a bad solve.
        """

        target = np.asarray(target_xyz, dtype=np.float64)
        initial = np.asarray(initial_state, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_xyz must be one finite XYZ vector")
        if initial.shape != (6,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_state must be one finite six-joint vector")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if damping <= 0 or not np.isfinite(damping):
            raise ValueError("damping must be finite and positive")
        if tolerance_m <= 0 or not np.isfinite(tolerance_m):
            raise ValueError("tolerance_m must be finite and positive")
        if max_step_normalized <= 0 or not np.isfinite(max_step_normalized):
            raise ValueError("max_step_normalized must be finite and positive")
        if posture_weight < 0 or not np.isfinite(posture_weight):
            raise ValueError("posture_weight must be finite and non-negative")
        if nullspace_preference is None:
            preference = initial[:5].copy()
        else:
            preferred = np.asarray(nullspace_preference, dtype=np.float64)
            if preferred.shape not in {(5,), (6,)} or not np.all(
                np.isfinite(preferred)
            ):
                raise ValueError(
                    "nullspace_preference must be a finite five- or six-vector"
                )
            preference = preferred[:5]

        with self._forward_lock:
            state = initial.copy()
            state[:5] = np.clip(state[:5], -100.0, 100.0)
            reference_radians = normalized_arm_positions_to_radians(
                np.clip(preference, -100.0, 100.0), self.joint_ranges
            )
            best_state = state.copy()
            self._set_state(
                normalized_positions_to_mujoco_qpos(state, self.all_joint_ranges)
            )
            best_xyz = self._body_local_to_world(
                self.gripper_body_id, self.fixed_tip_local
            ).copy()
            best_error = float(np.linalg.norm(target - best_xyz))
            iterations = 0

            for iteration in range(max_iterations):
                iterations = iteration + 1
                qpos = normalized_positions_to_mujoco_qpos(
                    state, self.all_joint_ranges
                )
                self._set_state(qpos)
                current_xyz = self._body_local_to_world(
                    self.gripper_body_id, self.fixed_tip_local
                ).copy()
                residual = target - current_xyz
                error = float(np.linalg.norm(residual))
                if error < best_error:
                    best_error = error
                    best_xyz = current_xyz
                    best_state = state.copy()
                if error <= tolerance_m:
                    break

                jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
                jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
                self._mujoco.mj_jac(
                    self.model,
                    self.data,
                    jacobian_position,
                    jacobian_rotation,
                    current_xyz,
                    self.gripper_body_id,
                )
                jacobian = jacobian_position[:, self.dof_addresses]
                current_radians = qpos[:5]
                regularizer = damping * damping + posture_weight
                system = jacobian.T @ jacobian + regularizer * np.eye(5)
                rhs = jacobian.T @ residual
                if posture_weight:
                    rhs += posture_weight * (reference_radians - current_radians)
                try:
                    delta_radians = np.linalg.solve(system, rhs)
                except np.linalg.LinAlgError:
                    delta_radians = np.linalg.lstsq(system, rhs, rcond=None)[0]

                normalized_scale = 200.0 / (
                    self.joint_ranges[:, 1] - self.joint_ranges[:, 0]
                )
                delta_normalized = delta_radians * normalized_scale
                largest = float(np.max(np.abs(delta_normalized)))
                if largest > max_step_normalized:
                    delta_normalized *= max_step_normalized / largest
                state[:5] = np.clip(state[:5] + delta_normalized, -100.0, 100.0)

            final_error = float(np.linalg.norm(target - best_xyz))
            return SO101IKResult(
                state=best_state.astype(np.float32),
                achieved_xyz=np.asarray(best_xyz, dtype=np.float64),
                target_xyz=target.copy(),
                error_m=final_error,
                converged=bool(final_error <= tolerance_m),
                iterations=iterations,
            )

    def inverse_tip_positions(
        self,
        target_xyz: np.ndarray,
        seed_states: np.ndarray,
        *,
        min_joint_separation_l2: float = 12.0,
        max_accept_error_m: float | None = None,
        **solve_kwargs: object,
    ) -> SO101TipIKFamily:
        """Reach one fingertip datum from many seeds; keep distinct configurations.

        Each seed is solved independently.  ``results`` reports every seed's
        achieved ``error_m`` so the caller can reject bad solves.
        ``distinct`` keeps accepted solutions whose first five joints are at
        least ``min_joint_separation_l2`` from an already kept configuration.
        """

        seeds = np.asarray(seed_states, dtype=np.float64)
        if seeds.ndim != 2 or seeds.shape[1] != 6:
            raise ValueError("seed_states must have shape [N, 6]")
        if min_joint_separation_l2 <= 0 or not np.isfinite(min_joint_separation_l2):
            raise ValueError("min_joint_separation_l2 must be finite and positive")
        if max_accept_error_m is not None and (
            not np.isfinite(max_accept_error_m) or max_accept_error_m <= 0
        ):
            raise ValueError("max_accept_error_m must be finite and positive")

        results: list[SO101IKResult] = []
        distinct: list[SO101IKResult] = []
        for seed in seeds:
            result = self.inverse_tip_position(target_xyz, seed, **solve_kwargs)
            results.append(result)
            accepted = (
                bool(result.converged)
                if max_accept_error_m is None
                else bool(result.error_m <= max_accept_error_m)
            )
            if not accepted:
                continue
            if any(
                float(np.linalg.norm(result.state[:5] - other.state[:5]))
                < min_joint_separation_l2
                for other in distinct
            ):
                continue
            distinct.append(result)
        return SO101TipIKFamily(results=tuple(results), distinct=tuple(distinct))
