"""Shared actor sample contract for real SO-101 and MuJoCo.

This is the only mixing surface for the FINAL DATA EXPERIMENT.
A/B/C may change data and actor supervision. They may not change this schema.

LeWM must consume only (state/history, canonical physical action) -> future.
Structured goal and causal pair fields are actor-side. They never enter LeWM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from fcpg.so101_kinematics import SO101_ANCHOR_NAMES, SO101_JOINT_NAMES
from v10.config import FUTURE_OFFSETS
from v10.data import CAMERA_CONTEXT, JOINT_SCALE

SOURCE = Literal["real", "mujoco"]

CONTROL_HZ = 30.0
V11_HISTORY_OFFSETS = (-5, 0)
ACTION_HORIZON = 20
RAW_STEPS_PER_CHUNK = 5
IMAGE_SIZE = 224
NUM_VIEWS = 2
NUM_ANCHORS = 3
STATE_DIM = 6
WRIST_VIEW = 0
COLUMN_VIEW = 1
VIEW_NAMES = ("wrist", "column")

# Raw LeRobot units before / JOINT_SCALE.
ARM_RAW_LIMITS = (-100.0, 100.0)
GRIPPER_RAW_LIMITS = (0.0, 100.0)
# Higher gripper command = more open on both real and MuJoCo.
GRIPPER_HIGHER_IS_MORE_OPEN = True


@dataclass(frozen=True)
class CanonicalActorSample:
    """One V11-shaped actor window. Values are already / JOINT_SCALE except RGB/px/m."""

    source: SOURCE
    rgb_history: np.ndarray
    rgb_future: np.ndarray
    q_nominal_px: np.ndarray
    anchor_xyz_history: np.ndarray
    state_history: np.ndarray
    camera_context: np.ndarray
    anchor_valid: np.ndarray
    view_valid: np.ndarray
    past_action_chunks: np.ndarray
    target_action: np.ndarray
    target_action_mask: np.ndarray
    target_future_view_valid: np.ndarray
    target_future_state: np.ndarray
    target_future_anchor_xyz: np.ndarray
    target_future_anchor_px: np.ndarray
    target_future_anchor_valid: np.ndarray
    episode_terminal_state: np.ndarray
    is_dual_camera: bool
    grounded_target_xyz: np.ndarray | None = None
    pair_id: str | None = None
    variant_id: str | None = None
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def obs_horizon(self) -> int:
        return int(self.state_history.shape[0])


def empty_camera_context(obs_horizon: int) -> np.ndarray:
    return np.broadcast_to(CAMERA_CONTEXT[None], (obs_horizon, NUM_VIEWS, 5)).copy()


def scale_joints(raw: np.ndarray) -> np.ndarray:
    return np.asarray(raw, dtype=np.float32) / np.float32(JOINT_SCALE)


def unscale_joints(normalized: np.ndarray) -> np.ndarray:
    return np.asarray(normalized, dtype=np.float32) * np.float32(JOINT_SCALE)


def action_landmarks(actions_raw: np.ndarray, current: int, offsets: tuple[int, ...]) -> np.ndarray:
    """Same five-command encoding as V10TrajectoryDataset._action_landmarks."""

    chunks = []
    for begin_offset, end_offset in zip(offsets[:-1], offsets[1:], strict=True):
        begin = current + begin_offset
        end = current + end_offset
        if end <= begin:
            raise ValueError("history offsets must be strictly increasing")
        indices = np.linspace(begin, end - 1, RAW_STEPS_PER_CHUNK).round().astype(np.int64)
        chunks.append(scale_joints(actions_raw[indices]).reshape(-1))
    return np.stack(chunks).astype(np.float32)


def contract_constants() -> dict[str, object]:
    return {
        "joint_names": list(SO101_JOINT_NAMES),
        "anchor_names": list(SO101_ANCHOR_NAMES),
        "joint_scale": JOINT_SCALE,
        "control_hz": CONTROL_HZ,
        "history_offsets": list(V11_HISTORY_OFFSETS),
        "future_offsets": list(FUTURE_OFFSETS),
        "action_horizon": ACTION_HORIZON,
        "raw_steps_per_chunk": RAW_STEPS_PER_CHUNK,
        "image_size": IMAGE_SIZE,
        "wrist_view": WRIST_VIEW,
        "column_view": COLUMN_VIEW,
        "gripper_higher_is_more_open": GRIPPER_HIGHER_IS_MORE_OPEN,
        "action_semantics": "absolute_joint_position",
        "state_units_after_scale": "lerobot_normalized_div_100",
        "anchor_xyz_units": "metres",
        "q_units": "pixels_in_224",
        "goal_pathway": "episode_terminal_state_memory_only",
        "lewm_may_read_goal": False,
    }
