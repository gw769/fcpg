"""Mock adapters for CI and offline logic testing.

No hardware, no GPU, no external dependencies.  Every interface is satisfied
with minimal in-memory state so tests can exercise the full planning pipeline
without a physical robot or camera.

Usage::

    from adapters.mock.adapters import (
        MockRobotBus, MockCamera, MockRobotController,
        MockKinematics, MockProjection, MockGripperGrounding,
        MockTargetDetector, MockTargetRegistration, MockTargetTracker,
        MockSafetyChecker, MockMotionPlanner,
        MockPhaseTracker,
    )
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from interfaces.hardware import RobotBusInterface, CameraInterface, MotorReading, Frame
from interfaces.robot import (
    RobotControllerInterface, KinematicsInterface,
    RobotState, Pose, CommandResult, PreflightResult,
)
from interfaces.perception import (
    ProjectionInterface, GripperGroundingInterface,
    TargetDetectorInterface, TargetRegistrationInterface, TargetTrackerInterface,
    GripperLocalization, TargetObservation, ProjectionCalibration,
    DetectionCandidate, AppearanceDescriptor,
)
from interfaces.planning import (
    MotionPlannerInterface, SafetyCheckerInterface, PhaseTrackerInterface,
    Observation, ActionCandidate, PhaseLabel, PhaseStatus, WorkspaceConfig,
    TargetState,
)


# ── hardware layer ─────────────────────────────────────────────────────────────

class MockRobotBus(RobotBusInterface):
    """In-memory motor bus; positions default to zero."""

    def __init__(self, motor_ids: list[int] | None = None) -> None:
        self._ids = list(motor_ids or [1, 2, 3, 4, 5, 6])
        self._positions: dict[int, float] = {i: 0.0 for i in self._ids}
        self._torque: dict[int, bool] = {i: False for i in self._ids}
        self.write_log: list[tuple[list[int], list[float]]] = []

    def read(self, motor_ids: list[int]) -> MotorReading:
        positions = tuple(self._positions.get(mid, 0.0) for mid in motor_ids)
        return MotorReading(
            motor_ids=tuple(motor_ids),
            positions=positions,
        )

    def write(self, motor_ids: list[int], positions: list[float]) -> None:
        self.write_log.append((motor_ids, positions))
        for mid, pos in zip(motor_ids, positions):
            self._positions[mid] = pos

    def enable_torque(self, motor_ids: list[int] | None = None) -> None:
        for mid in (motor_ids or self._ids):
            self._torque[mid] = True

    def disable_torque(self, motor_ids: list[int] | None = None) -> None:
        for mid in (motor_ids or self._ids):
            self._torque[mid] = False

    def ping(self, motor_id: int) -> bool:
        return motor_id in self._ids

    def close(self) -> None:
        pass


class MockCamera(CameraInterface):
    """Returns a solid-colour frame; role is set at construction time."""

    def __init__(self, role: str = "primary", width: int = 224, height: int = 224) -> None:
        self._role = role
        self._width = width
        self._height = height
        self._running = False
        self.capture_count = 0

    @property
    def role(self) -> str:
        return self._role

    def start(self) -> None:
        self._running = True

    def capture(self) -> Frame:
        if not self._running:
            raise RuntimeError("camera not started")
        self.capture_count += 1
        data = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        return Frame(
            data=data,
            width=self._width,
            height=self._height,
            timestamp_s=time.monotonic(),
            camera_id=f"mock-{self._role}",
            role=self._role,
        )

    def stop(self) -> None:
        self._running = False

    def is_alive(self) -> bool:
        return self._running


# ── robot layer ────────────────────────────────────────────────────────────────

class MockRobotController(RobotControllerInterface):
    """Stateful mock: tracks joint state in memory, all commands succeed."""

    def __init__(self, n_joints: int = 6, home_state: np.ndarray | None = None) -> None:
        self._state = np.zeros(n_joints, dtype=np.float32)
        self._home = home_state if home_state is not None else np.zeros(n_joints, dtype=np.float32)
        self._torque_enabled = False
        self.command_log: list[np.ndarray] = []

    def get_state(self) -> RobotState:
        return RobotState(
            positions=self._state.copy(),
            timestamp_s=time.monotonic(),
        )

    def send_commands(self, commands: np.ndarray, *, enable_motion: bool = False) -> CommandResult:
        start = time.monotonic()
        self.command_log.append(commands.copy())
        if enable_motion:
            self._state = commands[-1].copy()
        return CommandResult(
            success=True,
            executed_count=len(commands),
            duration_s=time.monotonic() - start,
            final_state=self.get_state(),
        )

    def enable_torque(self, joints: list[str] | None = None) -> None:
        self._torque_enabled = True

    def disable_torque(self, joints: list[str] | None = None) -> None:
        self._torque_enabled = False

    def home(self) -> None:
        self._state = self._home.copy()

    def close(self) -> None:
        pass


class MockKinematics(KinematicsInterface):
    """Returns trivial FK results; anchors are always at (0,0,0.1)."""

    def __init__(self, n_joints: int = 6) -> None:
        self._n_joints = n_joints

    def forward_kinematics(self, state: np.ndarray) -> Pose:
        return Pose(position=np.array([0.0, 0.0, 0.1]))

    def forward_anchors(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states)
        if states.ndim == 1:
            return np.array([[0.0, 0.0, 0.1], [0.0, 0.01, 0.1], [0.0, -0.01, 0.1]])
        n = states.shape[0]
        anchors = np.zeros((n, 3, 3), dtype=np.float64)
        anchors[..., 2] = 0.1
        return anchors

    def inverse_kinematics(
        self,
        target: Pose,
        current_state: np.ndarray,
        *,
        max_iterations: int = 100,
    ) -> np.ndarray | None:
        return current_state.copy()

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(f"joint_{i}" for i in range(self._n_joints))

    @property
    def joint_limits(self) -> tuple[tuple[float, float], ...]:
        return tuple((-100.0, 100.0) for _ in range(self._n_joints))


# ── perception layer ───────────────────────────────────────────────────────────

class MockProjection(ProjectionInterface):
    """Identity projection: world→pixel treats (x,y) as (u,v) directly."""

    def pixel_to_world(
        self,
        pixel: np.ndarray,
        z_plane: float,
        *,
        live_transform: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.array([pixel[0], pixel[1], z_plane])

    def world_to_pixel(
        self,
        world_point: np.ndarray,
        *,
        live_transform: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.array([world_point[0], world_point[1]])

    def load_calibration(self, calibration_path: str) -> None:
        pass


class MockGripperGrounding(GripperGroundingInterface):
    """Returns a fixed gripper location at image centre."""

    def __init__(self, width: int = 224, height: int = 224) -> None:
        self._cx = width / 2.0
        self._cy = height / 2.0

    def locate(self, frame: Frame, *, prior: GripperLocalization | None = None) -> GripperLocalization:
        center = np.array([self._cx, self._cy])
        return GripperLocalization(
            center_px=center,
            fixed_tip_px=center + np.array([0.0, 5.0]),
            moving_tip_px=center + np.array([0.0, -5.0]),
            confidence=1.0,
            method="mock",
        )

    def update_calibration(self, live_transform: np.ndarray) -> None:
        pass


class MockTargetDetector(TargetDetectorInterface):
    """Returns a configurable candidate list — no real image analysis.

    Construct with `candidates=[]` to simulate a detector that finds nothing,
    or with several entries to simulate an ambiguous scene.
    """

    def __init__(self, candidates: list[dict[str, Any]] | None = None) -> None:
        if candidates is None:
            candidates = [{"centroid_px": [112.0, 112.0], "area_px": 500,
                           "solidity": 0.8, "score": 500.0}]
        self._specs = candidates

    def detect(self, frame: Frame) -> list[DetectionCandidate]:
        out: list[DetectionCandidate] = []
        for spec in self._specs:
            u, v = spec["centroid_px"]
            r = float(spec.get("radius_px", 14.0))
            out.append(
                DetectionCandidate(
                    centroid_px=np.asarray([u, v], dtype=np.float64),
                    bbox_xyxy=np.asarray([u - r, v - r, u + r, v + r], dtype=np.float64),
                    area_px=int(spec["area_px"]),
                    solidity=float(spec["solidity"]),
                    score=float(spec["score"]),
                    descriptor=AppearanceDescriptor(
                        histogram=np.full(48, 1.0 / 48.0), n_channels=3, n_bins=16
                    ),
                )
            )
        out.sort(key=lambda c: c.score, reverse=True)
        return out

    def describe(self, frame: Frame, region_mask: np.ndarray) -> AppearanceDescriptor:
        return AppearanceDescriptor(
            histogram=np.full(48, 1.0 / 48.0), n_channels=3, n_bins=16
        )


class MockTargetRegistration(TargetRegistrationInterface):
    """In-memory reference store keyed by (role, view)."""

    def __init__(self) -> None:
        self._refs: dict[str, dict[str, Any]] = {}

    def register(
        self,
        frame: Frame,
        centroid_px: np.ndarray,
        role: str,
        view: str,
        *,
        radius_px: float = 28.0,
    ) -> str:
        target_id = f"{role}-{view}"
        self._refs[target_id] = {
            "role": role,
            "view": view,
            "centroid_px": np.asarray(centroid_px, dtype=np.float64).copy(),
            "radius_px": radius_px,
        }
        return target_id

    def clear(self, target_id: str) -> None:
        self._refs.pop(target_id, None)

    def is_registered(self, role: str, view: str) -> bool:
        return f"{role}-{view}" in self._refs

    def list_registered(self) -> dict[str, dict[str, Any]]:
        return dict(self._refs)


class MockTargetTracker(TargetTrackerInterface):
    """Tracks references held by a MockTargetRegistration instance."""

    def __init__(self, registration: MockTargetRegistration) -> None:
        self._registration = registration

    def track(self, frame: Frame, target_id: str) -> TargetObservation:
        refs = self._registration.list_registered()
        t = refs.get(target_id)
        if t is None:
            return TargetObservation(visible=False, centroid_px=None,
                                     bbox_xyxy=None, score=None,
                                     reason="unknown target_id")
        cx, cy = t["centroid_px"]
        r = t["radius_px"]
        return TargetObservation(
            visible=True,
            centroid_px=np.asarray(t["centroid_px"]).copy(),
            bbox_xyxy=np.asarray([cx - r, cy - r, cx + r, cy + r]),
            score=1.0,
            reason=None,
        )


# ── planning layer ─────────────────────────────────────────────────────────────

class MockSafetyChecker(SafetyCheckerInterface):
    """Always passes; optionally configured to fail for testing."""

    def __init__(self, always_safe: bool = True) -> None:
        self._always_safe = always_safe

    def check(
        self,
        commands: np.ndarray,
        current_state: np.ndarray,
        workspace: WorkspaceConfig,
    ) -> PreflightResult:
        if self._always_safe:
            return PreflightResult(safe=True, minimum_clearance_m=0.1)
        return PreflightResult(safe=False, reason="mock failure", minimum_clearance_m=-0.01)


class MockMotionPlanner(MotionPlannerInterface):
    """Returns a single candidate that holds the current state."""

    def plan(
        self,
        observation: Observation,
        goal: np.ndarray,
        *,
        n_candidates: int = 8,
    ) -> list[ActionCandidate]:
        state = observation.current_state
        chunk = np.tile(state, (5, 1))
        return [ActionCandidate(commands=chunk, cost=0.0)]

    def select(
        self,
        candidates: list[ActionCandidate],
        observation: Observation,
    ) -> ActionCandidate | None:
        return candidates[0] if candidates else None


class MockPhaseTracker(PhaseTrackerInterface):
    """Stays in APPROACH unless explicitly advanced."""

    def __init__(self) -> None:
        self._phase = PhaseLabel.APPROACH

    def advance(self, observation: Observation, thresholds: dict[str, float]) -> PhaseStatus:
        return PhaseStatus(
            current=self._phase,
            pending=None,
            confirmations_needed=0,
            confirmations_received=0,
        )

    def current_phase(self) -> PhaseLabel:
        return self._phase

    def reset(self) -> None:
        self._phase = PhaseLabel.APPROACH
