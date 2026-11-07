"""Layer 5 — Task definition and execution.

A TaskDefinition is a pure data object loaded from config.  The executor
orchestrates all lower layers without knowing which robot, camera, or model
is in use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .planning import PhaseLabel, WorkspaceConfig


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class TaskDefinition:
    """Describes a manipulation task entirely from config — no hardcoded values.

    All fields are loaded from a task YAML file.  The executor uses these
    values to drive phase transitions and success evaluation.
    """
    name: str

    # Which phases are active for this task (ordered)
    phases: list[PhaseLabel]

    # Workspace bounds for this task instance
    workspace: WorkspaceConfig

    # Phase-transition thresholds (keyed by name; executor is threshold-name-agnostic)
    thresholds: dict[str, float] = field(default_factory=dict)

    # Grasp geometry (set to None = derive from training data at runtime)
    contact_height_m: float | None = None
    transport_height_m: float | None = None
    release_height_m: float | None = None

    # Maximum autonomous execution chunks before requiring operator confirmation
    max_autonomous_chunks: int = 1

    # Whether to stop after the lift phase (for validation trials)
    stop_after_lift: bool = False

    # Free-form metadata (robot model, camera roles, experiment id, …)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialResult:
    """Result of a single execution trial."""
    success: bool
    final_phase: PhaseLabel
    final_state: np.ndarray | None
    contact_signal: float | None
    object_moved: bool | None
    chunks_executed: int
    duration_s: float
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ── interfaces ────────────────────────────────────────────────────────────────

class TaskExecutorInterface(ABC):
    """Orchestrates all layers to execute a manipulation task.

    Concrete executors receive references to robot, perception, and planning
    components through their constructor (dependency injection).  They must
    not create hardware resources themselves.
    """

    @abstractmethod
    def run_trial(
        self,
        task: TaskDefinition,
        *,
        max_chunks: int | None = None,
        dry_run: bool = False,
    ) -> TrialResult:
        """Execute one trial of the task.

        Args:
            task: task parameters loaded from config
            max_chunks: override task.max_autonomous_chunks if set
            dry_run: plan only, do not send commands to hardware
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop execution immediately and disable torque."""

    @abstractmethod
    def reset(self) -> None:
        """Return robot to home and reset phase tracker."""
