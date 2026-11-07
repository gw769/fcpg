"""Layer 2 — Robot control and kinematics.

Unified interface for robot state, command execution, and forward/inverse
kinematics.  Upper layers see only joint-space commands and Cartesian poses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import numpy as np


# ── data types ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RobotState:
    """Normalized robot state: positions in [0,100] per joint convention."""
    positions: np.ndarray           # (N,) joint positions
    velocities: np.ndarray | None = None
    loads: np.ndarray | None = None
    timestamp_s: float | None = None


@dataclass(frozen=True)
class Pose:
    """Cartesian pose: position + orientation (quaternion or euler)."""
    position: np.ndarray            # (3,) [x, y, z] in meters
    orientation: np.ndarray | None = None  # (4,) quat or (3,) euler


@dataclass(frozen=True)
class CommandResult:
    """Result of executing a command chunk."""
    success: bool
    executed_count: int
    duration_s: float
    final_state: RobotState
    error: str | None = None


@dataclass(frozen=True)
class PreflightResult:
    """Safety check result before hardware execution."""
    safe: bool
    reason: str | None = None
    minimum_clearance_m: float | None = None


# ── interfaces ────────────────────────────────────────────────────────────────

class RobotControllerInterface(ABC):
    """High-level robot control: state queries and command execution with safety."""

    @abstractmethod
    def get_state(self) -> RobotState:
        """Return current normalized joint state."""

    @abstractmethod
    def send_commands(self, commands: np.ndarray, *, enable_motion: bool = False) -> CommandResult:
        """Execute a chunk of commands.  Shape (K, N) for K steps, N joints.

        If enable_motion=False, performs dry-run (preflight only).
        """

    @abstractmethod
    def enable_torque(self, joints: list[str] | None = None) -> None:
        """Enable torque for named joints (all if None)."""

    @abstractmethod
    def disable_torque(self, joints: list[str] | None = None) -> None:
        """Disable torque for named joints (all if None)."""

    @abstractmethod
    def home(self) -> None:
        """Move to home position (if defined)."""

    @abstractmethod
    def close(self) -> None:
        """Release hardware resources."""


class KinematicsInterface(ABC):
    """Forward and inverse kinematics for the robot.

    Implementations typically wrap MuJoCo, PyBullet, or analytic solvers.
    """

    @abstractmethod
    def forward_kinematics(self, state: np.ndarray) -> Pose:
        """Compute end-effector pose from joint state."""

    @abstractmethod
    def forward_anchors(self, states: np.ndarray) -> np.ndarray:
        """Compute anchor points (e.g., fingertip centers) from joint states.

        Args:
            states: (N, J) or (J,) joint states

        Returns:
            (N, A, 3) or (A, 3) anchor xyz positions, A = number of anchors
        """

    @abstractmethod
    def inverse_kinematics(
        self,
        target: Pose,
        current_state: np.ndarray,
        *,
        max_iterations: int = 100,
    ) -> np.ndarray | None:
        """Solve for joint state that reaches target pose.

        Returns None if no solution found.
        """

    @property
    @abstractmethod
    def joint_names(self) -> tuple[str, ...]:
        """Ordered joint names."""

    @property
    @abstractmethod
    def joint_limits(self) -> tuple[tuple[float, float], ...]:
        """Per-joint (min, max) normalized limits."""
