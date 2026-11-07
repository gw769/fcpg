"""Layer 4 — Planning: candidate generation, safety checking, phase management.

All planner implementations operate on abstract observations and produce
abstract action candidates.  Robot-specific details are hidden behind the
interfaces in lower layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import numpy as np

from .robot import RobotState, PreflightResult


# ── data types ────────────────────────────────────────────────────────────────

class PhaseLabel(str, Enum):
    """Generic manipulation phases.  Task configs may use a subset.

    The NEAR_FIELD_* phases form the autonomous near-field acquisition stage.
    When a safety gate demands a near-field reference that does not exist yet,
    the executor traverses OBSERVATION → DETECTION → REGISTRATION and then
    resumes, instead of blocking on an external agent.
    """
    IDLE                      = "idle"
    APPROACH                  = "approach"
    NEAR_FIELD_OBSERVATION    = "near_field_observation"
    NEAR_FIELD_DETECTION      = "near_field_detection"
    NEAR_FIELD_REGISTRATION   = "near_field_registration"
    APPROACH_RESUME           = "approach_resume"
    CONTACT                   = "contact"
    CLOSE                     = "close"
    LIFT                      = "lift"
    TRANSPORT                 = "transport"
    RELEASE                   = "release"
    RECOVER                   = "recover"


@dataclass(frozen=True)
class PhaseStatus:
    current: PhaseLabel
    pending: PhaseLabel | None
    confirmations_needed: int
    confirmations_received: int


@dataclass(frozen=True)
class TargetState:
    """State of one tracked target (role name lives in the enclosing dict key)."""
    visible: bool
    centroid_px: np.ndarray | None  # (2,) [u, v] if visible
    world_xyz: np.ndarray | None    # (3,) [x, y, z] in metres if visible


@dataclass(frozen=True)
class Observation:
    """Structured world observation passed to the planner.

    Targets are keyed by role name (e.g. ``"primary"``, ``"secondary"``);
    the role vocabulary comes from task config, not from this interface.
    """
    current_state: np.ndarray               # (N,) joint positions
    gripper_center_px: np.ndarray | None    # (2,) or None
    targets: dict[str, TargetState]         # role → target state
    phase: PhaseLabel
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionCandidate:
    """A candidate action chunk with associated cost."""
    commands: np.ndarray                # (K, N) joint commands
    cost: float
    preflight: PreflightResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkspaceConfig:
    """Robot workspace and safety limits — loaded from config, never hardcoded."""
    lower_bound_m: tuple[float, float, float]
    upper_bound_m: tuple[float, float, float]
    table_z_m: float
    minimum_anchor_clearance_m: float   # hardware gate
    workspace_margin_m: float = 0.04


# ── interfaces ────────────────────────────────────────────────────────────────

class MotionPlannerInterface(ABC):
    """Generate action candidates from an observation and a goal.

    The planner is model-agnostic: LeWM, BC, analytic IK, retrieval — all
    implement this interface.  Model checkpoints and hyperparameters come
    from the adapter's config, not from this interface.
    """

    @abstractmethod
    def plan(
        self,
        observation: Observation,
        goal: np.ndarray,           # goal in world coordinates
        *,
        n_candidates: int = 8,
    ) -> list[ActionCandidate]:
        """Return candidates sorted by cost (lowest first)."""

    @abstractmethod
    def select(
        self,
        candidates: list[ActionCandidate],
        observation: Observation,
    ) -> ActionCandidate | None:
        """Select the best candidate, or None if none are safe."""


class SafetyCheckerInterface(ABC):
    """Validate a command chunk before hardware execution.

    Checks may include: anchor clearance, joint limits, workspace bounds,
    velocity limits, and any additional robot-specific constraints.
    """

    @abstractmethod
    def check(
        self,
        commands: np.ndarray,
        current_state: np.ndarray,
        workspace: WorkspaceConfig,
    ) -> PreflightResult:
        """Return a PreflightResult; safe=False blocks execution."""


class PhaseTrackerInterface(ABC):
    """Manage task phase transitions based on observations.

    Phase thresholds (distance, gripper width, etc.) come from TaskDefinition,
    not hardcoded here.
    """

    @abstractmethod
    def advance(self, observation: Observation, thresholds: dict[str, float]) -> PhaseStatus:
        """Evaluate transition conditions and update phase state.

        Args:
            observation: current world state
            thresholds: named thresholds from TaskDefinition
                        (e.g. 'near_target_m', 'gripper_closed', ...)
        """

    @abstractmethod
    def current_phase(self) -> PhaseLabel:
        """Return the active phase label."""

    @abstractmethod
    def reset(self) -> None:
        """Reset to initial phase (IDLE or APPROACH)."""
