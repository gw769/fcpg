"""Generic dual-jaw trajectory targets for the general task-space actor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


GENERAL_TASK_SPACE_DUAL_JAW_FORMAT = (
    "so101-v10-general-task-space-dual-jaw-trajectory-v1"
)
GENERAL_DAGGER_COLLECTION_FORMAT_V3 = (
    "so101-v10-general-task-space-dagger-v3"
)
GENERAL_DAGGER_DUAL_JAW_FIELDS = frozenset(
    {
        "current_jaw_xyz",
        "target_jaw_delta_trajectory_m",
    }
)
DEFAULT_DUAL_JAW_TRAJECTORY_STEPS = 5
GENERAL_TASK_SPACE_JOINT_ACTION_CONTRACT_FORMAT = (
    "so101-v10-general-task-space-normalized-joint-action-v1"
)
GENERAL_TASK_SPACE_ACTION_STEP_CAP = np.asarray(
    (0.04, 0.04, 0.04, 0.04, 0.04, 0.08), dtype=np.float32
)
GENERAL_TASK_SPACE_ACTION_LOWER = np.asarray(
    (-1.0, -1.0, -1.0, -1.0, -1.0, 0.0), dtype=np.float32
)
GENERAL_TASK_SPACE_ACTION_UPPER = np.ones(6, dtype=np.float32)

# Executable gripper positions behind the three-mode CLOSE/HOLD/OPEN vocabulary.
# A deployed student emits only a phase class, which the joint-plan projector turns
# into one of these positions, so every teacher gate that reads ``state[5]`` back
# has to be expressed against them. Gating on an object-derived opening the class
# vocabulary cannot express deadlocks the phase machine.
EXECUTABLE_CLOSE_GRIPPER = 5.0
EXECUTABLE_OPEN_GRIPPER = 50.0


def general_task_space_joint_action_contract() -> dict[str, object]:
    """Return the robot-wide command contract used by anonymous task-space control."""

    return {
        "format": GENERAL_TASK_SPACE_JOINT_ACTION_CONTRACT_FORMAT,
        "units": "normalized_absolute_joint",
        "physical_scale": [100.0] * 6,
        "step_cap": GENERAL_TASK_SPACE_ACTION_STEP_CAP.tolist(),
        "lower": GENERAL_TASK_SPACE_ACTION_LOWER.tolist(),
        "upper": GENERAL_TASK_SPACE_ACTION_UPPER.tolist(),
        "projection": "direction_preserving_sequential_step_projection",
        "phase_or_category_conditioned": False,
        "physical_command_bounds_relaxed": False,
    }


class DualJawKinematics(Protocol):
    """Robot-only geometry needed to transport measured pad locations."""

    def forward_anchors(self, states: np.ndarray) -> np.ndarray: ...

    def forward_gripper_poses(self, states: np.ndarray) -> np.ndarray: ...


def project_normalized_joint_trajectory_to_step_contract(
    *,
    current_state_physical: np.ndarray,
    target_trajectory_normalized: np.ndarray,
    step_cap: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    physical_scale: float | np.ndarray = 100.0,
) -> np.ndarray:
    """Project absolute joint targets onto a bounded direction-preserving path."""

    current = np.asarray(current_state_physical, dtype=np.float64)
    targets = np.asarray(target_trajectory_normalized, dtype=np.float64)
    cap = np.asarray(step_cap, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    scale = np.asarray(physical_scale, dtype=np.float64)
    if scale.shape == ():
        scale = np.full(6, float(scale), dtype=np.float64)
    if current.shape != (6,) or not np.isfinite(current).all():
        raise ValueError("current_state_physical must be finite [6]")
    if targets.ndim != 2 or targets.shape[1:] != (6,) or not len(targets):
        raise ValueError("target_trajectory_normalized must be finite non-empty [K,6]")
    if not np.isfinite(targets).all():
        raise ValueError("target_trajectory_normalized must be finite non-empty [K,6]")
    if any(value.shape != (6,) for value in (cap, low, high, scale)):
        raise ValueError("joint trajectory contract vectors must have shape [6]")
    if not all(np.isfinite(value).all() for value in (cap, low, high, scale)):
        raise ValueError("joint trajectory contract vectors must be finite")
    if np.any(cap <= 0.0) or np.any(low >= high) or np.any(scale <= 0.0):
        raise ValueError("joint trajectory contract is invalid")

    state = current / scale
    tolerance = 1.0e-6
    if np.any(state < low - tolerance) or np.any(state > high + tolerance):
        raise ValueError("current state lies outside the joint trajectory contract")
    state = np.clip(state, low, high)
    projected = np.empty_like(targets, dtype=np.float64)
    for index, target in enumerate(targets):
        bounded_target = np.clip(target, low, high)
        delta = bounded_target - state
        moving = np.abs(delta) > 1.0e-12
        fraction = (
            min(1.0, float(np.min(cap[moving] / np.abs(delta[moving]))))
            if np.any(moving)
            else 1.0
        )
        state = np.clip(state + fraction * delta, low, high)
        projected[index] = state
    return projected.astype(np.float32)


@dataclass(frozen=True)
class DualJawTrajectoryTarget:
    """Two ordered physical jaw trajectories relative to their current pads."""

    current_jaw_xyz: np.ndarray
    delta_trajectory_m: np.ndarray

    def __post_init__(self) -> None:
        current = np.asarray(self.current_jaw_xyz, dtype=np.float32).copy()
        delta = np.asarray(self.delta_trajectory_m, dtype=np.float32).copy()
        if current.shape != (2, 3) or not np.isfinite(current).all():
            raise ValueError("current_jaw_xyz must be finite [2,3]")
        if (
            delta.ndim != 3
            or delta.shape[1:] != (2, 3)
            or len(delta) == 0
            or not np.isfinite(delta).all()
        ):
            raise ValueError("delta_trajectory_m must be finite non-empty [K,2,3]")
        current.flags.writeable = False
        delta.flags.writeable = False
        object.__setattr__(self, "current_jaw_xyz", current)
        object.__setattr__(self, "delta_trajectory_m", delta)

    @property
    def target_jaw_xyz(self) -> np.ndarray:
        return self.current_jaw_xyz[None] + self.delta_trajectory_m

    @property
    def separation_trajectory_m(self) -> np.ndarray:
        target = self.target_jaw_xyz
        return np.linalg.norm(target[:, 1] - target[:, 0], axis=-1)


def dual_jaw_trajectory_contract(steps: int) -> dict[str, object]:
    """Return the cache/output contract without task-instance semantics."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("dual-jaw trajectory steps must be a positive integer")
    return {
        "format": GENERAL_TASK_SPACE_DUAL_JAW_FORMAT,
        "ordered_jaws": ["fixed", "moving"],
        "trajectory_steps": int(steps),
        "frame": "robot_base_m",
        "output_tensor": "jaw_delta_trajectory_m",
        "output_shape": ["B", int(steps), 2, 3],
        "cache_current_tensor": "current_jaw_xyz",
        "cache_target_tensor": "target_jaw_delta_trajectory_m",
        "cache_target_shape": ["N", int(steps), 2, 3],
        "target_source": (
            "current measured ordered pad centers transported by robot FK to "
            "teacher target-action states"
        ),
        "separation_is_derived_from_two_predicted_trajectories": True,
        "category_appearance_seed_or_task_name_used": False,
        "simulator_contact_force_is_actor_input": False,
    }


def dual_jaw_target_from_teacher_action(
    kinematics: DualJawKinematics,
    *,
    current_state: np.ndarray,
    current_jaw_xyz: np.ndarray,
    target_action_normalized: np.ndarray,
    joint_scale: np.ndarray,
    trajectory_steps: int = DEFAULT_DUAL_JAW_TRAJECTORY_STEPS,
) -> DualJawTrajectoryTarget:
    """Transport measured pad centers along the teacher's robot-only FK path.

    The measured pad-to-nominal-tip offsets are first expressed in the current
    gripper frame, then rotated into every target gripper frame.  This retains
    physical pad extension without a fixed extension constant or scene label.
    """

    state = np.asarray(current_state, dtype=np.float64)
    current_jaws = np.asarray(current_jaw_xyz, dtype=np.float64)
    actions = np.asarray(target_action_normalized, dtype=np.float64)
    scale = np.asarray(joint_scale, dtype=np.float64)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError("current_state must be finite [6]")
    if current_jaws.shape != (2, 3) or not np.isfinite(current_jaws).all():
        raise ValueError("current_jaw_xyz must be finite [2,3]")
    if actions.ndim != 2 or actions.shape[1:] != (6,) or not np.isfinite(actions).all():
        raise ValueError("target_action_normalized must be finite [H,6]")
    if scale.shape == ():
        scale = np.full(6, float(scale), dtype=np.float64)
    if scale.shape != (6,) or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("joint_scale must be a positive scalar or six-vector")
    if (
        isinstance(trajectory_steps, bool)
        or not isinstance(trajectory_steps, int)
        or not 1 <= trajectory_steps <= len(actions)
    ):
        raise ValueError("trajectory_steps must lie inside the action horizon")

    current_anchors = np.asarray(
        kinematics.forward_anchors(state[None]), dtype=np.float64
    )
    current_pose = np.asarray(
        kinematics.forward_gripper_poses(state[None]), dtype=np.float64
    )
    target_states = actions[:trajectory_steps] * scale[None]
    target_anchors = np.asarray(
        kinematics.forward_anchors(target_states), dtype=np.float64
    )
    target_poses = np.asarray(
        kinematics.forward_gripper_poses(target_states), dtype=np.float64
    )
    if current_anchors.shape != (1, 3, 3) or target_anchors.shape != (
        trajectory_steps,
        3,
        3,
    ):
        raise ValueError("kinematics returned an invalid ordered anchor shape")
    if current_pose.shape != (1, 4, 4) or target_poses.shape != (
        trajectory_steps,
        4,
        4,
    ):
        raise ValueError("kinematics returned an invalid gripper pose shape")
    if not all(
        np.isfinite(value).all()
        for value in (current_anchors, target_anchors, current_pose, target_poses)
    ):
        raise ValueError("kinematics returned non-finite geometry")

    current_rotation = current_pose[0, :3, :3]
    target_rotation = target_poses[:, :3, :3]
    local_pad_offsets = (
        current_jaws - current_anchors[0, 1:3]
    ) @ current_rotation
    rotated_offsets = np.einsum(
        "kij,sj->ksi", target_rotation, local_pad_offsets
    )
    target_jaws = target_anchors[:, 1:3] + rotated_offsets
    return DualJawTrajectoryTarget(
        current_jaw_xyz=current_jaws,
        delta_trajectory_m=target_jaws - current_jaws[None],
    )


__all__ = [
    "DEFAULT_DUAL_JAW_TRAJECTORY_STEPS",
    "EXECUTABLE_CLOSE_GRIPPER",
    "EXECUTABLE_OPEN_GRIPPER",
    "GENERAL_DAGGER_COLLECTION_FORMAT_V3",
    "GENERAL_DAGGER_DUAL_JAW_FIELDS",
    "GENERAL_TASK_SPACE_ACTION_LOWER",
    "GENERAL_TASK_SPACE_ACTION_STEP_CAP",
    "GENERAL_TASK_SPACE_ACTION_UPPER",
    "GENERAL_TASK_SPACE_DUAL_JAW_FORMAT",
    "GENERAL_TASK_SPACE_JOINT_ACTION_CONTRACT_FORMAT",
    "DualJawKinematics",
    "DualJawTrajectoryTarget",
    "dual_jaw_target_from_teacher_action",
    "dual_jaw_trajectory_contract",
    "general_task_space_joint_action_contract",
    "project_normalized_joint_trajectory_to_step_contract",
]
