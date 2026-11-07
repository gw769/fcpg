"""Generic robotics manipulation framework interfaces.

This package defines abstract interfaces for:
- Hardware: motor buses, cameras
- Robot: control, kinematics, state
- Perception: visual grounding, coordinate projection, tracking
- Planning: motion generation, safety checking, phase management
- Task: definitions and execution

All hardware/algorithm-specific code belongs in adapters/, not here.
"""

from .hardware import RobotBusInterface, CameraInterface
from .robot import RobotControllerInterface, KinematicsInterface, RobotState
from .perception import (
    ProjectionInterface, GripperGroundingInterface,
    TargetDetectorInterface, TargetRegistrationInterface, TargetTrackerInterface,
)
from .planning import MotionPlannerInterface, SafetyCheckerInterface, PhaseTrackerInterface
from .task import TaskDefinition, TaskExecutorInterface

__all__ = [
    # Hardware layer
    "RobotBusInterface",
    "CameraInterface",
    # Robot layer
    "RobotControllerInterface",
    "KinematicsInterface",
    "RobotState",
    # Perception layer
    "ProjectionInterface",
    "GripperGroundingInterface",
    "TargetDetectorInterface",
    "TargetRegistrationInterface",
    "TargetTrackerInterface",
    # Planning layer
    "MotionPlannerInterface",
    "SafetyCheckerInterface",
    "PhaseTrackerInterface",
    # Task layer
    "TaskDefinition",
    "TaskExecutorInterface",
]
