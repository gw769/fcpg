"""Runtime contracts for human-confirmed SO-101 receding-horizon control."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from fcpg.letterbox_geometry import (
    LETTERBOX_V1,
    PLANNING_IMAGE_SIZE,
    SQUASH_V0,
    PlanningGeometry,
    planning_geometry_from_protocol,
)
from fcpg.observer_runtime import clear_observer_adjacent_history
from fcpg.so101_active_gripper import (
    independent_open_frame_metal_tip_hint,
    pulse_tracks_allowed,
    role_boxes_for_pulse_tracks,
)
from fcpg.so101_pulse_excitation import (
    excitation_preflight,
    solve_excitation_pose,
)
from fcpg.so101_realtime_mpc import ManipulationPhase, validate_command_chunk
from fcpg.servo_load_telemetry import (
    attach_servo_load_telemetry,
    servo_load_enabled,
)
from fcpg.so101_realtime_planner import (
    CausalPlanningMemory,
    ContextRegistrationError,
    LiveFingertipMeasurement,
    RealtimePlan,
    plan_json,
)
from fcpg.so101_retrieval_mpc import SO101_POSITION_LOWER, SO101_POSITION_UPPER
from fcpg.so101_shadow_session import SO101ShadowSession
from fcpg.prod_v16_5_runtime import (
    V165RuntimeContractError,
    conservative_joint_sweeps,
    enforce_no_contact_selected_chunk,
)
from interfaces.grounding import GroundingVerdict


VIEWS = ("wrist", "column")
CAMERA_BINDING_STABLE_FIELDS = (
    "role",
    "requested_alias",
    "physical_device",
    "binding",
)
#: The commanded joint endpoint was reached inside the settle tolerance.
ENDPOINT_SETTLED = "settled"
#: Motion demonstrably stopped with the joint target still missed.  The pose
#: reported alongside this outcome is the measured one, never the commanded one.
ENDPOINT_MISS_WITH_ACTUAL_STATE = "endpoint_miss_with_actual_state"


@dataclass(frozen=True)
class LiveObservation:
    timestamp_s: float
    state: np.ndarray
    frames: dict[str, np.ndarray]
    native_frames: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float32)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("live state must be a finite six-vector")
        if not np.isfinite(self.timestamp_s):
            raise ValueError("live timestamp must be finite")
        if set(self.frames) != set(VIEWS):
            raise ValueError("live observation must contain wrist and column frames")
        for view in VIEWS:
            frame = np.asarray(self.frames[view])
            if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
                raise ValueError(f"{view} frame must be uint8 [224,224,3]")
        if self.native_frames is not None:
            if not isinstance(self.native_frames, dict):
                raise ValueError("native_frames must be a dict or None")
            for view, frame in self.native_frames.items():
                rgb = np.asarray(frame)
                if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                    raise ValueError(f"{view} native frame must be uint8 HxWx3")


@dataclass(frozen=True)
class EndpointEnvelope:
    """Bounds a stalled endpoint must satisfy to be replanned from instead of aborted.

    None of these numbers are new tuning knobs.  ``max_joint_miss`` is the
    caller's already-declared per-step command limit: the safety layer refuses
    to *command* a larger single-step motion, so a rest pose further than that
    from its command is not a load residual but a divergence.
    ``max_tcp_miss_m`` is measured off the robot model rather than chosen -- see
    :func:`gripper_capture_radius_m`.
    """

    #: LeRobot normalized position units, the same unit the settle tolerances
    #: use.  Close to degrees on SO-101 but not identical, so it is not named
    #: after either.
    max_joint_miss: float
    max_tcp_miss_m: float
    forward: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_joint_miss) or self.max_joint_miss <= 0:
            raise ValueError("max_joint_miss must be finite and positive")
        if not np.isfinite(self.max_tcp_miss_m) or self.max_tcp_miss_m <= 0:
            raise ValueError("max_tcp_miss_m must be finite and positive")
        if self.forward is not None and not callable(self.forward):
            raise ValueError("forward must be callable when supplied")

    def evaluate(
        self,
        *,
        commanded_state: np.ndarray,
        actual_state: np.ndarray,
    ) -> tuple[tuple[str, ...], float | None]:
        """Return the violated bound names and the measured TCP miss."""

        commanded = np.asarray(commanded_state, dtype=np.float32)
        actual = np.asarray(actual_state, dtype=np.float32)
        violations: list[str] = []
        if np.any(actual < SO101_POSITION_LOWER) or np.any(
            actual > SO101_POSITION_UPPER
        ):
            violations.append("hard_position_bounds")
        joint_miss = float(np.max(np.abs(actual[:5] - commanded[:5])))
        if joint_miss > self.max_joint_miss:
            violations.append("joint_miss")
        tcp_miss_m: float | None = None
        if self.forward is not None:
            commanded_xyz = np.asarray(self.forward(commanded), dtype=np.float64)
            actual_xyz = np.asarray(self.forward(actual), dtype=np.float64)
            if commanded_xyz.shape != (3,) or actual_xyz.shape != (3,):
                raise ValueError("envelope forward kinematics must return XYZ")
            if not np.all(np.isfinite(commanded_xyz)) or not np.all(
                np.isfinite(actual_xyz)
            ):
                raise ValueError("envelope forward kinematics must be finite")
            tcp_miss_m = float(np.linalg.norm(actual_xyz - commanded_xyz))
            if tcp_miss_m > self.max_tcp_miss_m:
                violations.append("tcp_miss")
        return tuple(violations), tcp_miss_m


@dataclass(frozen=True)
class EndpointSettlement:
    """Measured evidence for how one chunk's endpoint came to rest."""

    outcome: str
    commanded_state: np.ndarray
    actual_state: np.ndarray
    joint_residual: float
    step_delta: float
    residual_improvement: float
    quiet_samples: int
    samples: int
    tcp_miss_m: float | None = None
    envelope_violations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.outcome not in {ENDPOINT_SETTLED, ENDPOINT_MISS_WITH_ACTUAL_STATE}:
            raise ValueError(f"unsupported endpoint outcome: {self.outcome!r}")
        for name in ("commanded_state", "actual_state"):
            values = np.asarray(getattr(self, name), dtype=np.float32)
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a finite six-vector")
        for name in ("joint_residual", "step_delta", "residual_improvement"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.quiet_samples < 0 or self.samples <= 0:
            raise ValueError("settlement sample counts are invalid")
        if self.tcp_miss_m is not None and (
            not np.isfinite(self.tcp_miss_m) or self.tcp_miss_m < 0
        ):
            raise ValueError("tcp_miss_m must be finite and non-negative")
        # A settlement that reports a miss without proving motion stopped would
        # be indistinguishable from "we gave up waiting", which is the very
        # thing this contract exists to stop the runtime from doing.
        if self.outcome == ENDPOINT_MISS_WITH_ACTUAL_STATE and self.quiet_samples <= 0:
            raise ValueError("an endpoint miss must be backed by quiet samples")

    @property
    def reached_commanded_endpoint(self) -> bool:
        return self.outcome == ENDPOINT_SETTLED


@dataclass(frozen=True)
class ChunkExecution:
    requested: np.ndarray
    sent: np.ndarray
    observations: tuple[LiveObservation, ...]
    duration_s: float
    command_journal: str | None = None
    settlement: EndpointSettlement | None = None

    def __post_init__(self) -> None:
        requested = np.asarray(self.requested, dtype=np.float32)
        sent = np.asarray(self.sent, dtype=np.float32)
        if requested.shape != (5, 6) or sent.shape != requested.shape:
            raise ValueError("chunk execution commands must have shape [5,6]")
        if not np.all(np.isfinite(requested)) or not np.all(np.isfinite(sent)):
            raise ValueError("chunk execution commands must be finite")
        if len(self.observations) != 5:
            raise ValueError("chunk execution must return five post-command observations")
        if not np.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError("chunk duration must be finite and non-negative")
        if self.command_journal is not None and (
            not isinstance(self.command_journal, str) or not self.command_journal
        ):
            raise ValueError("command_journal must be a non-empty path string")
        if self.settlement is not None and not isinstance(
            self.settlement, EndpointSettlement
        ):
            raise ValueError("settlement must be an EndpointSettlement")

    @property
    def reached_commanded_endpoint(self) -> bool:
        """False only when the arm provably stopped short of its command."""

        return self.settlement is None or self.settlement.reached_commanded_endpoint


class RealtimeObservationIO(Protocol):
    is_hardware: bool

    def capture_initial(self, count: int, period_s: float) -> list[LiveObservation]: ...

    def observe(self) -> LiveObservation: ...

    def preview_frame(self, view: str) -> np.ndarray: ...

    def execute_chunk(
        self,
        chunk: np.ndarray,
        *,
        period_s: float,
        stop_event: threading.Event,
    ) -> ChunkExecution: ...

    def calibrate_gripper(
        self,
        *,
        stop_event: threading.Event,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


class RealtimePlanner(Protocol):
    phase_tracker: Any

    def plan(
        self,
        frames: dict[str, np.ndarray],
        states: np.ndarray,
        *,
        past_raw_actions: np.ndarray | None = None,
        causal_memory: CausalPlanningMemory | None = None,
        current_fingertips: LiveFingertipMeasurement | None = None,
    ) -> RealtimePlan: ...

    def detect_live_target(
        self,
        column_frame: np.ndarray,
        phase: ManipulationPhase,
        *,
        update_track: bool = False,
    ) -> tuple[str, dict[str, Any]]: ...

    def set_target_reference(
        self,
        role: str,
        view: str,
        frame: np.ndarray,
        centroid_px: np.ndarray,
        *,
        radius_px: float,
    ) -> dict[str, Any]: ...

    def set_context_reference(
        self,
        frame: np.ndarray,
        centroid_px: np.ndarray,
        *,
        radius_px: float,
        registration: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    def clear_context_reference(self) -> None: ...

    def detect_live_context(
        self,
        column_frame: np.ndarray,
        *,
        update_track: bool = False,
    ) -> dict[str, Any]: ...

    def context_clearance(
        self,
        current_state: np.ndarray,
        command_prefixes: np.ndarray,
        context_xy_m: np.ndarray,
    ) -> dict[str, np.ndarray]: ...

    def set_live_anchor_center(
        self,
        view: str,
        center_px: np.ndarray,
        state: np.ndarray,
        *,
        anchor_index: int = 0,
    ) -> np.ndarray: ...

    def set_live_anchor_candidates(
        self,
        view: str,
        candidates_px: np.ndarray,
        state: np.ndarray,
        *,
        anchor_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def set_live_moving_tip_track(
        self,
        view: str,
        moving_tip_track_candidates_px: np.ndarray,
        open_state: np.ndarray,
        closed_state: np.ndarray,
        *,
        center_anchor_index: int = 0,
        fixed_tip_anchor_index: int = 1,
        moving_tip_anchor_index: int = 2,
        maximum_center_offset_px: float | None = None,
        maximum_residual_px: float | None = None,
    ) -> dict[str, Any]: ...

    def commit_executed_plan(
        self,
        plan: RealtimePlan,
        *,
        command_count: int,
    ) -> dict[str, Any] | None: ...

    def reset_task_progress(self) -> None: ...


ArtifactWriter = Callable[[RealtimePlan], dict[str, str]]


class PlanRejectedError(RuntimeError):
    """A pre-motion safety rejection that permits a fresh observation and replan."""


class GroundingPulseRejectedError(RuntimeError):
    """A visual pulse was rejected, while hardware motion itself stayed healthy."""


def is_recoverable_pulse_track_fit_error(exc: BaseException) -> bool:
    """True only for an explicit moving-tip track-fit / translation-bound refusal."""

    if not isinstance(exc, ValueError):
        return False
    text = str(exc)
    return (
        "no moving-tip track fit" in text
        and "translation" in text
        and "bound" in text
    )


class SafeHoldVerificationError(RuntimeError):
    """A safe hold was commanded but the servo did not confirm the new goal.

    Distinct from a plain fault because the arm is in an *unverified* command
    state: it may still be chasing the goal the hold was meant to cancel.  It is
    never raised silently -- the per-joint mismatch travels with the message and
    is journalled before the raise.
    """


def _endpoint_check(result: ChunkExecution) -> dict[str, Any]:
    """Summarize how a chunk's endpoint came to rest, for the operator receipt."""

    settlement = result.settlement
    if settlement is None:
        return {"reported": False, "outcome": None}
    return {
        "reported": True,
        "outcome": settlement.outcome,
        "reached_commanded_endpoint": settlement.reached_commanded_endpoint,
        "commanded_state": settlement.commanded_state,
        "actual_state": settlement.actual_state,
        "joint_residual": settlement.joint_residual,
        "step_delta": settlement.step_delta,
        "residual_improvement": settlement.residual_improvement,
        "quiet_samples": settlement.quiet_samples,
        "settle_samples_seen": settlement.samples,
        "tcp_miss_m": settlement.tcp_miss_m,
    }


def gripper_capture_radius_m(kinematics: Any) -> float:
    """Measure the open-jaw capture radius off the robot model.

    Half the distance between the two jaw tips with the gripper fully open: the
    largest tool-center displacement for which a commanded point still lies
    within the jaw span.  It is read from the MuJoCo model rather than chosen,
    so it tracks the modelled hardware instead of becoming a tuning knob.
    """

    state = np.zeros(6, dtype=np.float32)
    state[5] = 100.0
    anchors = np.asarray(kinematics.forward_anchors(state), dtype=np.float64)
    if anchors.shape != (3, 3):
        raise ValueError("forward_anchors must return three anchor XYZ rows")
    radius = 0.5 * float(np.linalg.norm(anchors[1] - anchors[2]))
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("measured gripper capture radius is not positive")
    return radius


def _observation_at(session: dict[str, SO101ShadowSession], index: int) -> LiveObservation:
    first = session[VIEWS[0]]
    return LiveObservation(
        timestamp_s=float(first.timestamps_s[index]),
        state=np.asarray(first.states[index], dtype=np.float32).copy(),
        frames={
            view: np.asarray(session[view].frames[index], dtype=np.uint8).copy()
            for view in VIEWS
        },
    )


#: Optional shadow-session manifest key naming the pixel geometry the stored
#: frames were preprocessed with.  Absent from every historical recording.
SESSION_GEOMETRY_PROTOCOL_KEY = "geometry_protocol"


def _declared_session_geometry(session: Any) -> str | None:
    """Return the geometry protocol a session *declares*, or ``None``.

    Geometry is never inferred from the stored array shape.  Both ``squash_v0``
    and ``letterbox_v1`` produce the same ``224x224`` canvas, so the shape
    carries no information about which one preprocessed the pixels; a
    shape-based decision would be a fail-open.  Only an explicit manifest
    declaration counts as a declaration.
    """

    manifest = getattr(session, "manifest", None)
    if not isinstance(manifest, Mapping):
        return None
    declared = manifest.get(SESSION_GEOMETRY_PROTOCOL_KEY)
    return None if declared is None else str(declared)


class ReplaySO101IO:
    """Deterministic paired-session I/O for a no-hardware web dry-run.

    Replay reads cached frames directly and never calls the hardware camera
    path, so it is a second, fully independent pixel-ingestion point.  Its
    geometry is therefore declared explicitly here rather than shared with the
    live path or guessed from the frames.
    """

    is_hardware = False

    def __init__(
        self,
        sessions: dict[str, SO101ShadowSession],
        *,
        start_index: int = 10,
        geometry_protocol: str = SQUASH_V0,
        source_image_size: tuple[int, int] | None = None,
        planning_image_size: int = PLANNING_IMAGE_SIZE,
    ) -> None:
        if set(sessions) != set(VIEWS):
            raise ValueError("replay requires wrist and column sessions")
        count = sessions[VIEWS[0]].frame_count
        if count < 16:
            raise ValueError("replay requires at least 16 paired frames")
        for view in VIEWS[1:]:
            if sessions[view].frame_count != count:
                raise ValueError("paired replay frame counts differ")
        self.geometry_protocol = self._require_session_geometry(
            sessions, geometry_protocol
        )
        self.source_image_size = (
            None
            if source_image_size is None
            else (int(source_image_size[0]), int(source_image_size[1]))
        )
        self.planning_image_size = int(planning_image_size)
        # Build once so an unusable declaration fails at construction time
        # rather than on the first replayed frame.
        self.planning_geometry = self._build_planning_geometry()
        self.sessions = sessions
        self.count = count
        self.index = max(10, int(start_index))
        if self.index >= count:
            raise ValueError("replay start index is outside the session")
        if self.index + 5 >= count:
            raise ValueError(
                "replay start index must leave a complete next five-frame chunk"
            )

    @staticmethod
    def _require_session_geometry(
        sessions: Mapping[str, Any], geometry_protocol: Any
    ) -> str:
        """Reconcile the caller's declared geometry with each session's own.

        The rules, all of them explicit-declaration based:

        * a session that declares the *same* protocol passes;
        * a session that declares a *different* protocol fails closed -- the
          cached pixels were produced by one preprocessing pipeline and
          reinterpreting them under another silently biases every derived
          pixel;
        * a session that declares *nothing* fails closed under a strict
          ``letterbox_v1`` requirement, because an undeclared recording cannot
          be shown to be letterboxed and the ``224x224`` shape proves nothing;
        * a session that declares nothing stays permissive under the legacy
          ``squash_v0`` default, so every existing replay caller keeps working
          unchanged -- historical recordings were written by
          ``so101_shadow_session.resize_rgb``, which is exactly ``squash_v0``.
        """

        if geometry_protocol not in (SQUASH_V0, LETTERBOX_V1):
            raise ValueError(
                f"replay geometry_protocol must be {SQUASH_V0!r} or "
                f"{LETTERBOX_V1!r}, got {geometry_protocol!r}. Geometry is "
                "never inferred from the cached frame shape: a 224x224 frame "
                "is consistent with both protocols, so replay must declare it."
            )
        mismatched: dict[str, str] = {}
        undeclared: list[str] = []
        for view in VIEWS:
            declared = _declared_session_geometry(sessions[view])
            if declared is None:
                undeclared.append(view)
            elif declared != geometry_protocol:
                mismatched[view] = declared
        if mismatched:
            detail = "; ".join(
                f"{view} declares {declared!r}"
                for view, declared in sorted(mismatched.items())
            )
            raise ValueError(
                f"replay geometry mismatch: requested {geometry_protocol!r} but "
                f"{detail}. Cached frames are only meaningful in the geometry "
                "they were preprocessed with; re-record or replay under the "
                "declared protocol instead of reinterpreting the pixels."
            )
        if undeclared and geometry_protocol != SQUASH_V0:
            raise ValueError(
                f"replay requires geometry_protocol={geometry_protocol!r} but "
                f"session(s) {', '.join(sorted(undeclared))} declare no "
                f"{SESSION_GEOMETRY_PROTOCOL_KEY!r}. An undeclared recording "
                "cannot be assumed to be letterboxed: both protocols produce "
                "the same 224x224 canvas, so accepting it would be a "
                "fail-open. Re-record the session or declare its geometry."
            )
        return str(geometry_protocol)

    def _build_planning_geometry(self) -> PlanningGeometry:
        """Resolve the declared protocol to a canvas/content descriptor."""

        source = self.source_image_size
        if self.geometry_protocol == LETTERBOX_V1 and source is None:
            raise ValueError(
                f"replay geometry_protocol={LETTERBOX_V1!r} requires "
                "source_image_size: the padding band is a function of the raw "
                "camera aspect ratio and cannot be recovered from the "
                f"{self.planning_image_size}x{self.planning_image_size} "
                "cached canvas alone"
            )
        return planning_geometry_from_protocol(
            self.geometry_protocol,
            source_width=None if source is None else source[0],
            source_height=None if source is None else source[1],
            image_size=self.planning_image_size,
        )

    def capture_initial(self, count: int, period_s: float) -> list[LiveObservation]:
        del period_s
        if count != 11 or self.index < count - 1:
            raise ValueError("replay initial history requires exactly 11 frames")
        return [
            _observation_at(self.sessions, index)
            for index in range(self.index - count + 1, self.index + 1)
        ]

    def observe(self) -> LiveObservation:
        return _observation_at(self.sessions, self.index)

    def preview_frame(self, view: str) -> np.ndarray:
        if view not in VIEWS:
            raise ValueError(f"unsupported preview view: {view}")
        return np.asarray(
            self.sessions[view].frames[self.index], dtype=np.uint8
        ).copy()

    def can_execute_chunk(self) -> bool:
        return self.index + 5 < self.count

    def execute_chunk(
        self,
        chunk: np.ndarray,
        *,
        period_s: float,
        stop_event: threading.Event,
    ) -> ChunkExecution:
        del period_s
        requested = np.asarray(chunk, dtype=np.float32)
        if requested.shape != (5, 6):
            raise ValueError("replay chunk must have shape [5,6]")
        if not self.can_execute_chunk():
            raise RuntimeError("replay session has no complete next five-frame chunk")
        observations: list[LiveObservation] = []
        started = time.perf_counter()
        for _ in range(5):
            if stop_event.is_set():
                raise RuntimeError("execution stopped by operator")
            self.index += 1
            observations.append(_observation_at(self.sessions, self.index))
        return ChunkExecution(
            requested=requested,
            sent=requested.copy(),
            observations=tuple(observations),
            duration_s=time.perf_counter() - started,
        )

    def close(self) -> None:
        return None


class RealtimeMPCController:
    """Coordinate fresh sensing, planning, and bounded closed-loop execution."""

    # Non-formal bounded autonomy may execute multiple freshly supervised
    # tokens after one explicit run authorization. Formal trials keep it off.
    MAX_CLOSED_LOOP_CHUNKS = 16

    def __init__(
        self,
        planner: RealtimePlanner,
        io: RealtimeObservationIO,
        *,
        artifact_writer: ArtifactWriter | None = None,
        execute_enabled: bool = False,
        calibration_enabled: bool | None = None,
        calibration_committer: (
            Callable[[dict[str, Any]], Mapping[str, Any]] | None
        ) = None,
        control_hz: float = 30.0,
        max_plan_age_s: float = 15.0,
        max_state_drift: float = 2.0,
        max_target_drift_px: float = 15.0,
        max_command_jump: float = 8.0,
        minimum_expected_motion: float = 0.5,
        minimum_observed_motion: float = 0.1,
        wrist_registration_distance_m: float = 0.13,
        home_state: np.ndarray | None = None,
        home_control_hz: float = 10.0,
        home_max_step: float = 3.0,
        autonomous_run_enabled: bool = False,
        max_autonomous_chunks: int = 16,
        max_autonomous_duration_s: float = 180.0,
        grounding_bridge: Any | None = None,
        task_class: int | None = None,
    ) -> None:
        positive = (
            control_hz,
            max_state_drift,
            max_target_drift_px,
            max_command_jump,
            minimum_expected_motion,
            minimum_observed_motion,
            wrist_registration_distance_m,
            home_control_hz,
            home_max_step,
            max_autonomous_duration_s,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("runtime thresholds must be finite and positive")
        if not np.isfinite(max_plan_age_s) or max_plan_age_s < 0:
            raise ValueError("max_plan_age_s must be finite and non-negative")
        self.planner = planner
        self.io = io
        self._v165_runtime_adapter = getattr(planner, "v165_runtime_adapter", None)
        self._v168_runtime_adapter = getattr(planner, "v168_runtime_adapter", None)
        self._v170_runtime_adapter = getattr(planner, "v170_runtime_adapter", None)
        # Compatibility alias for existing status/tests.  Mode, rather than
        # the attribute name, determines whether V17 has candidate influence.
        self._v170_shadow_adapter = self._v170_runtime_adapter
        planner_config = getattr(planner, "config", None)
        self._v165_no_contact_approach = bool(
            getattr(planner_config, "v165_no_contact_approach", False)
        )
        self._v168_supervised_debug = bool(
            getattr(planner_config, "v168_supervised_debug", False)
        )
        self._v165_contact_hardware_authorized = bool(
            self._v165_runtime_adapter is not None
            and self._v165_runtime_adapter.contact_hardware_authorized
        )
        self._v165_no_contact_consumed = False
        if self._v165_no_contact_approach and self._v165_runtime_adapter is None:
            raise ValueError("V16.5 no-contact mode requires a frozen runtime bundle")
        if self._v165_no_contact_approach and self._v165_contact_hardware_authorized:
            raise ValueError("V16.5 no-contact and contact-authorized modes conflict")
        if self._v168_supervised_debug and self._v168_runtime_adapter is None:
            raise ValueError("V16.8 supervised debug requires the V16.8 runtime")
        if self._v168_supervised_debug and self._v165_no_contact_approach:
            raise ValueError("V16.8 supervised debug conflicts with no-contact mode")
        if (
            self._v165_runtime_adapter is not None
            and bool(getattr(io, "is_hardware", False))
            and bool(execute_enabled)
            and not self._v165_no_contact_approach
            and not self._v168_supervised_debug
        ):
            raise ValueError(
                "V16.5 contact hardware remains sealed until post-training integration review"
            )
        if self._v168_runtime_adapter is not None:
            mode = str(getattr(self._v168_runtime_adapter, "mode", ""))
            if mode == "shadow" and bool(execute_enabled):
                raise ValueError("V16.8 shadow requires commands_sent=0")
            if (
                bool(getattr(io, "is_hardware", False))
                and bool(execute_enabled)
                and not bool(getattr(planner_config, "v168_live_execute", False))
            ):
                raise ValueError("V16.8 hardware execute needs its explicit live flag")
        if self._v170_runtime_adapter is not None:
            v170_mode = str(getattr(self._v170_runtime_adapter, "mode", ""))
            if v170_mode == "shadow" and bool(execute_enabled):
                raise ValueError("V17.0 shadow requires commands_sent=0")
            if (
                v170_mode == "candidate"
                and bool(getattr(io, "is_hardware", False))
                and bool(execute_enabled)
                and not bool(getattr(planner_config, "v170_live_execute", False))
            ):
                raise ValueError("V17.0 hardware execute needs its explicit live flag")
        self.artifact_writer = artifact_writer
        self.execute_enabled = bool(execute_enabled)
        self.calibration_enabled = (
            self.execute_enabled
            if calibration_enabled is None
            else bool(calibration_enabled)
        )
        if (
            self._v170_runtime_adapter is not None
            and str(getattr(self._v170_runtime_adapter, "mode", "")) == "shadow"
        ):
            self.calibration_enabled = False
        if (
            self._v165_runtime_adapter is not None
            and not self._v168_supervised_debug
        ):
            # V16.5 evaluates one frozen calibration/bundle.  A contact task
            # authorization does not also authorize jaw pulses or re-fitting.
            self.calibration_enabled = False
        if calibration_committer is not None and not callable(
            calibration_committer
        ):
            raise ValueError("calibration_committer must be callable")
        self.calibration_committer = calibration_committer
        self.period_s = 1.0 / float(control_hz)
        self.max_plan_age_s = float(max_plan_age_s)
        self.max_state_drift = float(max_state_drift)
        self.max_target_drift_px = float(max_target_drift_px)
        self.max_command_jump = float(max_command_jump)
        self.minimum_expected_motion = float(minimum_expected_motion)
        self.minimum_observed_motion = float(minimum_observed_motion)
        self.wrist_registration_distance_m = float(
            wrist_registration_distance_m
        )
        if home_state is None:
            self.home_state = None
        else:
            home = np.asarray(home_state, dtype=np.float32)
            if home.shape != (6,) or not np.all(np.isfinite(home)):
                raise ValueError("home state must be a finite six-vector")
            self.home_state = home.copy()
        self.home_period_s = 1.0 / float(home_control_hz)
        self.home_max_step = float(home_max_step)
        if (
            isinstance(max_autonomous_chunks, bool)
            or not isinstance(max_autonomous_chunks, int)
            or not 1 <= max_autonomous_chunks <= self.MAX_CLOSED_LOOP_CHUNKS
        ):
            raise ValueError(
                "max_autonomous_chunks must be an integer in [1,16]"
            )
        self.autonomous_run_enabled = bool(autonomous_run_enabled)
        self.max_autonomous_chunks = int(max_autonomous_chunks)
        self.max_autonomous_duration_s = float(max_autonomous_duration_s)
        # FCPG grounding feedback.  None reproduces the previous behaviour
        # exactly: the near-field acquisition window halts and waits for an
        # operator.  Supplying a bridge replaces that wait with autonomous
        # verification and registration.  Default None so no existing
        # deployment changes behaviour without being asked to.
        self.grounding_bridge = grounding_bridge
        self.last_grounding_feedback: dict[str, Any] | None = None
        if task_class is None:
            self.task_class = None
            self.task_class_record: dict[str, Any] | None = None
        else:
            from v10.task_class import parse_task_class, task_class_record

            self.task_class = parse_task_class(task_class)
            self.task_class_record = task_class_record(self.task_class)

        self.observations: deque[LiveObservation] = deque(maxlen=71)
        self.past_chunks: deque[np.ndarray] = deque(maxlen=14)
        self.plan: RealtimePlan | None = None
        # Optional non-actuating V11 shadow runner.  Set after construction via
        # ``controller.v11_shadow = V11ShadowRunner(...)``.  When set, every
        # _replan_locked call runs a shadow forward pass and logs the result to
        # self.assets["v11_shadow"].  The production plan is never affected.
        self.v11_shadow: Any | None = None
        # Optional default-off Prod-V13 observer. After automatic H0, its
        # current fixed/moving posterior is the planner's pixel measurement;
        # the observer still never emits a command itself.
        self.prod_v13_shadow: Any | None = None
        self.last_prod_v13_shadow_error: str | None = None
        # Optional post-plan V14 ranker. It records a counterfactual over the
        # exact candidate bank and cannot alter selected_chunk.
        self.prod_v14_shadow: Any | None = None
        self.last_prod_v14_shadow_error: str | None = None
        # Optional causal V16.1 system-identification shadow. It observes only
        # completed actions and emits a bounded proposal for inspection; it
        # cannot change the selected command.
        self.prod_v16_secant_shadow: Any | None = None
        self.last_prod_v16_secant_shadow_error: str | None = None
        # Optional V11 production controller.  Set after construction via
        # ``controller.v11_controller = V11Controller(...)``.  When set, the
        # runtime can use V11 as the action source instead of phase_target_ik.
        # V11 proposals flow through the same supervisor → validation → execute
        # pipeline as MPC candidates.
        self.v11_controller: Any | None = None
        self.assets: dict[str, str] = {}
        self.receipts: list[dict[str, Any]] = []
        self.stopped = False
        self.executing = False
        self.active_operation: str | None = None
        self.error: str | None = None
        self.calibration: dict[str, Any] | None = None
        self.last_execution_batch: dict[str, Any] | None = None
        self.execution_armed = False
        self.stop_event = threading.Event()
        self.operation_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self._fresh_seeded_ik_cache: tuple[tuple[Any, ...], dict[str, Any] | None] | None = (
            None
        )

    def _determine_action_source(self, goal_conditioned: bool) -> str:
        """Determine the current action source based on configuration."""
        if not goal_conditioned:
            return "retrieved_demonstration_prefix"
        if self._v170_runtime_adapter is not None:
            return (
                "v17_0_full_stack_shadow_commands_sent_0"
                if str(getattr(self._v170_runtime_adapter, "mode", "")) == "shadow"
                else "v17_0_actor_plus_common_bank_v17_state_fk_lewm_ranked"
            )
        if self._v168_runtime_adapter is not None:
            return (
                "v16_8_shadow_original_fcpg_state_fk_lewm"
                if self._v168_runtime_adapter.mode == "shadow"
                else "v16_8_actor_plus_common_bank_state_fk_lewm_ranked"
            )
        if self._v165_runtime_adapter is not None:
            return "v16_5_actor_plus_common_candidate_bank"
        if self.v11_controller is not None:
            return "v11_and_ik_lewm_ranked"
        return "phase_target_ik_candidates"

    def _assert_v165_session_binding(self) -> None:
        """Reject enabling, replacing, or switching the startup bundle."""

        current_v168 = getattr(self.planner, "v168_runtime_adapter", None)
        if current_v168 is not getattr(self, "_v168_runtime_adapter", None):
            raise RuntimeError("V16.8 Actor overlay changed after session bootstrap")
        if current_v168 is not None:
            config = getattr(self.planner, "config", None)
            if str(getattr(config, "v168_mode", "")) != current_v168.mode:
                raise RuntimeError("V16.8 mode changed after session bootstrap")
            if getattr(config, "anchor_authority", None) != "original_fcpg":
                raise RuntimeError("V16.8 original-FCPG authority changed")
            configured_debug = bool(
                getattr(config, "v168_supervised_debug", False)
            )
            if configured_debug != bool(
                getattr(self, "_v168_supervised_debug", configured_debug)
            ):
                raise RuntimeError(
                    "V16.8 supervised-debug permission changed after bootstrap"
                )
        current_v170 = getattr(self.planner, "v170_runtime_adapter", None)
        if current_v170 is not getattr(self, "_v170_runtime_adapter", None):
            raise RuntimeError("V17.0 runtime changed after session bootstrap")
        if current_v170 is not None:
            config = getattr(self.planner, "config", None)
            configured_mode = str(getattr(config, "v170_mode", ""))
            if configured_mode != str(getattr(current_v170, "mode", "")):
                raise RuntimeError("V17.0 mode changed after bootstrap")
            if configured_mode == "shadow" and bool(
                getattr(self, "execute_enabled", False)
            ):
                raise RuntimeError("V17.0 shadow session gained execution authority")
            if configured_mode == "candidate" and bool(
                getattr(self, "execute_enabled", False)
            ) != bool(getattr(config, "v170_live_execute", False)):
                raise RuntimeError("V17.0 live permission changed after bootstrap")
        current = getattr(self.planner, "v165_runtime_adapter", None)
        if current is not self._v165_runtime_adapter:
            raise RuntimeError("V16.5 runtime bundle changed after session bootstrap")
        if current is not None:
            config = getattr(self.planner, "config", None)
            configured = getattr(config, "anchor_authority", None)
            if configured != current.anchor_authority:
                raise RuntimeError("V16.5 anchor authority changed after bootstrap")
            configured_no_contact = bool(
                getattr(config, "v165_no_contact_approach", False)
            )
            frozen_no_contact = bool(
                getattr(self, "_v165_no_contact_approach", configured_no_contact)
            )
            if configured_no_contact != frozen_no_contact:
                raise RuntimeError("V16.5 no-contact mode changed after bootstrap")
            current_authorized = bool(
                getattr(current, "contact_hardware_authorized", False)
            )
            frozen_authorized = bool(
                getattr(
                    self,
                    "_v165_contact_hardware_authorized",
                    current_authorized,
                )
            )
            if current_authorized != frozen_authorized:
                raise RuntimeError(
                    "V16.5 contact authorization changed after bootstrap"
                )
            if current_authorized:
                verifier = getattr(
                    self.planner, "assert_v165_contact_runtime_binding", None
                )
                if not callable(verifier):
                    raise RuntimeError(
                        "V16.5 contact runtime binding verifier is absent"
                    )
                verifier()
            if (
                bool(getattr(getattr(self, "io", None), "is_hardware", False))
                and bool(getattr(self, "execute_enabled", False))
                and not frozen_no_contact
                and not bool(getattr(self, "_v168_supervised_debug", False))
            ):
                raise RuntimeError(
                    "V16.5 contact hardware remains sealed until post-training integration review"
                )

    def _reset_v168_window(self, reason: str) -> dict[str, Any] | None:
        """Keep V16.8 causal state aligned with every discarded command log."""

        if self._v168_runtime_adapter is None:
            return None
        reset = getattr(self.planner, "reset_v168_window", None)
        if not callable(reset):
            raise RuntimeError("V16.8 planner lacks causal-window reset")
        result = reset(reason)
        if result is not None and not isinstance(result, Mapping):
            raise RuntimeError("V16.8 causal-window reset returned invalid receipt")
        return None if result is None else dict(result)

    def _commit_v168_completed_chunk(
        self,
        *,
        sent_chunk: np.ndarray,
        round_id: str,
        selected_source: str | None,
    ) -> dict[str, Any] | None:
        """Advance V16.8 only after transport confirmed ``result.sent``."""

        if self._v168_runtime_adapter is None:
            return None
        if self._v168_runtime_adapter.mode == "shadow":
            # A shadow session has an intentionally empty execution history.
            return None
        commit = getattr(self.planner, "commit_v168_completed_chunk", None)
        if not callable(commit):
            raise RuntimeError("V16.8 planner lacks completed-chunk commit")
        result = commit(
            sent_chunk=np.asarray(sent_chunk, dtype=np.float32),
            round_id=round_id,
            selected_source=selected_source,
        )
        if result is not None and not isinstance(result, Mapping):
            raise RuntimeError("V16.8 completed-chunk commit returned invalid receipt")
        return None if result is None else dict(result)

    def _reset_v170_window(self, reason: str) -> dict[str, Any] | None:
        """Discard V17 causal state whenever commands/observations are refreshed."""

        if self._v170_runtime_adapter is None:
            return None
        reset = getattr(self.planner, "reset_v170_window", None)
        if not callable(reset):
            raise RuntimeError("V17.0 planner lacks causal-window reset")
        result = reset(reason)
        if result is not None and not isinstance(result, Mapping):
            raise RuntimeError("V17.0 causal-window reset returned invalid state")
        return None if result is None else dict(result)

    def _commit_v170_completed_chunk(
        self,
        *,
        requested_chunk: np.ndarray,
        sent_chunk: np.ndarray,
        measured_joint_path: np.ndarray,
        round_id: str,
    ) -> dict[str, Any] | None:
        """Advance V17 only after the sole transport confirms all five commands."""

        adapter = self._v170_runtime_adapter
        if adapter is None or str(getattr(adapter, "mode", "")) == "shadow":
            return None
        commit = getattr(self.planner, "commit_v170_completed_chunk", None)
        if not callable(commit):
            raise RuntimeError("V17.0 planner lacks completed-chunk commit")
        result = commit(
            requested_commands_deg=np.asarray(requested_chunk, dtype=np.float32),
            sent_commands_deg=np.asarray(sent_chunk, dtype=np.float32),
            measured_joint_path_deg=np.asarray(measured_joint_path, dtype=np.float32),
            round_id=round_id,
        )
        if result is not None and not isinstance(result, Mapping):
            raise RuntimeError("V17.0 completed-chunk commit returned invalid state")
        return None if result is None else dict(result)

    def _reject_no_contact_side_motion(self, operation: str) -> None:
        """Reject non-plan actuator entrypoints before they can touch I/O."""

        v170 = getattr(self, "_v170_runtime_adapter", None)
        if (
            v170 is not None
            and str(getattr(v170, "mode", "")) == "shadow"
        ):
            raise PlanRejectedError(
                f"V17.0 shadow forbids {operation}; no I/O was attempted"
            )
        if bool(getattr(self, "_v165_no_contact_approach", False)):
            raise PlanRejectedError(
                f"V16.5 no-contact approach forbids {operation}; no I/O was attempted"
            )
        if self._v165_runtime_adapter is not None:
            # Public side-motion methods do not enter execute().  Recheck the
            # immutable session here so a programmatic permission flip cannot
            # reach home/open/calibration I/O first.
            self._assert_v165_session_binding()
        if (
            self._v168_runtime_adapter is not None
            and bool(getattr(self, "_v168_supervised_debug", False))
        ):
            return
        if self._v165_runtime_adapter is not None and not bool(
            getattr(self, "_v165_contact_hardware_authorized", False)
        ):
            raise PlanRejectedError(
                f"V16.5 contact authorization forbids {operation}; no I/O was attempted"
            )

    @staticmethod
    def _selected_candidate_row(plan: Any) -> dict[str, Any] | None:
        rows = getattr(plan, "candidate_rows", None) or []
        index = getattr(plan, "selected_index", None)
        if index is None:
            return None
        try:
            index = int(index)
        except (TypeError, ValueError):
            return None
        if not 0 <= index < len(rows):
            return None
        row = rows[index]
        return row if isinstance(row, Mapping) else None

    def _log_lewm_rank_decision(self, plan: Any) -> None:
        rows = list(getattr(plan, "candidate_rows", None) or [])
        v11_rows = [
            row
            for row in rows
            if str(row.get("source", "")).startswith("v11_")
        ]
        selected = self._selected_candidate_row(plan)
        contract = getattr(plan, "model_contract", None) or {}
        influence = contract.get("lewm_decision_influence")
        selected_source = None if selected is None else selected.get("source")
        print(
            f"[V11-RANK] v11_in_candidates={bool(v11_rows)} "
            f"n_candidates={len(rows)} "
            f"selected={selected_source} "
            f"lewm_scored={None if selected is None else selected.get('lewm_scored')} "
            f"predicted_last_lift={None if selected is None else selected.get('predicted_terminal_lift_deg')} "
            f"commanded_last_lift={None if selected is None else selected.get('commanded_terminal_lift_deg')} "
            f"lewm_decision_influence={influence}",
            flush=True,
        )
        for row in v11_rows:
            print(
                f"[V11-RANK] v11_candidate index={row.get('index')} "
                f"source={row.get('source')} safe={row.get('safe')} "
                f"cost={row.get('cost')} "
                f"predicted_last_lift={row.get('predicted_terminal_lift_deg')} "
                f"commanded_last_lift={row.get('commanded_terminal_lift_deg')} "
                f"lewm_scored={row.get('lewm_scored')}",
                flush=True,
            )

    def _v11_actor_prefix_provider(
        self,
        current_state: np.ndarray,
        *,
        relative_goal_xyz: np.ndarray | None = None,
        phase: Any = None,
    ) -> dict[str, Any] | None:
        """Build one hold-filled V11 prefix for LeWM ranking. Goal stays actor-only."""

        if self.v11_controller is None or self.task_class is None:
            return None
        if not self.observations:
            return None
        state = np.asarray(current_state, dtype=np.float32)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            return None
        horizon_raw = int(getattr(self.planner, "horizon_raw", 40))
        command_chunk = int(
            getattr(getattr(self.planner, "config", None), "command_chunk", 5)
        )
        target: dict[str, Any] = {}
        if relative_goal_xyz is not None:
            fish = np.asarray(relative_goal_xyz, dtype=np.float32)
            if fish.shape == (3,) and np.all(np.isfinite(fish)):
                target = {"relative_goal_xyz": fish}
        elif self.plan is not None and isinstance(getattr(self.plan, "target", None), dict):
            target = dict(self.plan.target)
        from types import SimpleNamespace

        stub = SimpleNamespace(
            current_state=state,
            target=target,
            candidate_rows=(
                list(getattr(self.plan, "candidate_rows", None) or [])
                if self.plan is not None
                else []
            ),
            guard=None if self.plan is None else getattr(self.plan, "guard", None),
        )
        try:
            live_goal = self._resolve_v11_live_goal(stub, self.task_class, state)
            goal_raw = np.asarray(live_goal["goal_state"], dtype=np.float32)
            print(
                f"[V11-RANK] actor_predict n_obs={len(self.observations)} "
                f"goal={live_goal.get('source')} "
                f"goal_lift={live_goal.get('goal_lift_deg')} "
                f"horizon_raw={horizon_raw}",
                flush=True,
            )
            v11_result = self.v11_controller.predict_chunk(
                self.observations,
                state,
                goal_state_raw=goal_raw,
                task_class=self.task_class,
            )
        except Exception as exc:
            print(
                f"[V11-RANK] actor_predict failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return None
        if not v11_result.get("executable", False):
            print(
                f"[V11-RANK] skip_prefix reason={v11_result.get('reason', 'not_executable')}",
                flush=True,
            )
            return None
        chunk = v11_result.get("selected_chunk")
        if chunk is None:
            print("[V11-RANK] skip_prefix reason=missing_chunk", flush=True)
            return None
        from fcpg.so101_v11_controller import (
            V11_CANDIDATE_SOURCE,
            hold_fill_v11_prefix,
            rate_limit_command_chunk,
        )

        raw_chunk = np.asarray(chunk, dtype=np.float32)
        limited = rate_limit_command_chunk(raw_chunk, state, self.max_command_jump)
        try:
            prefix = hold_fill_v11_prefix(
                limited,
                horizon_raw,
                command_chunk=command_chunk,
            )
        except ValueError as exc:
            print(f"[V11-RANK] skip_prefix reason={exc}", flush=True)
            return None
        phase_value = (
            phase.value
            if phase is not None and hasattr(phase, "value")
            else None if phase is None else str(phase)
        )
        goal_role = "pickup"
        if phase_value == "transport":
            goal_role = "drop_zone"
        elif phase_value == "grasp":
            goal_role = "grasp"
        print(
            f"[V11-RANK] prefix_ready source={V11_CANDIDATE_SOURCE} "
            f"shape={prefix.shape} "
            f"lift_now={float(state[1]):.3f} "
            f"lift_chunk_last={float(prefix[command_chunk - 1, 1]):.3f} "
            f"lewm_will_score=True goal_in_lewm=False",
            flush=True,
        )
        return {
            "prefixes": prefix[None],
            "metadata": [
                {
                    "source": V11_CANDIDATE_SOURCE,
                    "variant": "hold_filled_actor_prefix",
                    "goal_role": goal_role,
                    "live_goal_source": live_goal.get("source"),
                    "actor_chunk_steps": command_chunk,
                    "lewm_scored": True,
                }
            ],
        }

    @staticmethod
    def _serialize_v11_live_goal(resolved: Mapping[str, Any]) -> dict[str, Any]:
        goal = np.asarray(resolved["goal_state"], dtype=np.float32)
        fish = resolved.get("fish_xyz")
        return {
            "source": resolved["source"],
            "phase": resolved.get("phase"),
            "goal_state": goal.tolist(),
            "l2_arm_deg": float(resolved["l2_arm_deg"]),
            "goal_lift_deg": float(resolved["goal_lift_deg"]),
            "state_lift_deg": float(resolved["state_lift_deg"]),
            "workspace_goal": bool(resolved["workspace_goal"]),
            "needs_fresh_seeded_ik": bool(resolved.get("needs_fresh_seeded_ik", False)),
            "fresh_ik_source": resolved.get("fresh_ik_source"),
            "fk_to_fish_mm": resolved.get("fk_to_fish_mm"),
            "fish_xyz": None if fish is None else list(fish),
        }

    def _fresh_seeded_ik_cache_key(
        self,
        plan: Any,
        fish: np.ndarray,
        gripper: float,
    ) -> tuple[Any, ...]:
        guard = getattr(plan, "guard", None)
        return (
            None if guard is None else getattr(guard, "round_id", None),
            tuple(np.round(np.asarray(fish, dtype=np.float64), 5)),
            round(float(gripper), 2),
        )

    def _solve_fresh_seeded_ik_for_plan(
        self,
        plan: Any,
        current_state: np.ndarray,
    ) -> dict[str, Any] | None:
        from fcpg.so101_v11_live_goal import (
            plan_relative_goal_xyz,
            solve_fresh_seeded_ik,
        )
        from v10.fish_to_box_task import WORKSPACE_LIFT_DEG

        state = np.asarray(current_state, dtype=np.float32)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            return None
        if float(state[1]) <= WORKSPACE_LIFT_DEG:
            return None
        fish = plan_relative_goal_xyz(plan)
        if fish is None:
            return None
        kinematics = getattr(self.planner, "kinematics", None)
        if kinematics is None:
            return None
        key = self._fresh_seeded_ik_cache_key(plan, fish, float(state[5]))
        if self._fresh_seeded_ik_cache is not None and self._fresh_seeded_ik_cache[0] == key:
            return self._fresh_seeded_ik_cache[1]
        solver = getattr(self, "fresh_seeded_ik_solver", solve_fresh_seeded_ik)
        try:
            solved = solver(
                fish_xyz=fish,
                kinematics=kinematics,
                live_gripper=float(state[5]),
            )
        except Exception as exc:
            print(
                f"[V11] fresh_seeded_ik failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            solved = None
        if solved is not None:
            goal = np.asarray(solved.get("goal_state"), dtype=np.float32)
            print(
                f"[V11] fresh_ik source={solved.get('source')} "
                f"lift={float(goal[1]):.3f} "
                f"fk_to_fish_mm={solved.get('fk_to_fish_mm')} "
                f"fish={solved.get('fish_xyz')}",
                flush=True,
            )
        self._fresh_seeded_ik_cache = (key, solved)
        return solved

    def _cached_fresh_seeded_ik(
        self,
        plan: Any,
        current_state: np.ndarray,
    ) -> dict[str, Any] | None:
        from fcpg.so101_v11_live_goal import plan_relative_goal_xyz

        state = np.asarray(current_state, dtype=np.float32)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            return None
        fish = plan_relative_goal_xyz(plan)
        if fish is None or self._fresh_seeded_ik_cache is None:
            return None
        key = self._fresh_seeded_ik_cache_key(plan, fish, float(state[5]))
        if self._fresh_seeded_ik_cache[0] != key:
            return None
        return self._fresh_seeded_ik_cache[1]

    def _resolve_v11_live_goal(
        self,
        plan: Any,
        task_class: int,
        current_state: np.ndarray,
        *,
        allow_ik_solve: bool = True,
    ) -> dict[str, Any]:
        from fcpg.so101_v11_live_goal import (
            plan_relative_goal_xyz,
            resolve_v11_live_goal,
        )

        if allow_ik_solve:
            solved = self._solve_fresh_seeded_ik_for_plan(plan, current_state)
        else:
            solved = self._cached_fresh_seeded_ik(plan, current_state)
        fresh = None if solved is None else solved.get("goal_state")
        resolved = resolve_v11_live_goal(
            plan,
            task_class=task_class,
            current_state=current_state,
            fresh_seeded_ik=fresh,
        )
        fish = plan_relative_goal_xyz(plan)
        resolved["fish_xyz"] = None if fish is None else fish.tolist()
        if solved is not None and not resolved.get("needs_fresh_seeded_ik", True):
            resolved["fresh_ik_source"] = solved.get("source")
            resolved["fk_to_fish_mm"] = solved.get("fk_to_fish_mm")
            resolved["fish_xyz"] = solved.get("fish_xyz")
        else:
            resolved["fresh_ik_source"] = None
            resolved["fk_to_fish_mm"] = None
        return resolved

    def _v11_live_goal_status(self) -> dict[str, Any] | None:
        if self.v11_controller is None or self.plan is None or self.task_class is None:
            return None
        # Status polls share ThreadingHTTPServer with asset GETs.  Overnight
        # native deaths overlapped those polls with 34-candidate FK/rank, so
        # status may reuse a cached IK but must not call MuJoCo itself.
        resolved = self._resolve_v11_live_goal(
            self.plan,
            self.task_class,
            self.plan.current_state,
            allow_ik_solve=False,
        )
        return self._serialize_v11_live_goal(resolved)

    def preview_v11_chunk(self) -> dict[str, Any]:
        """Non-actuating V11 proposal. Does not observe, command, or arm."""

        with self.state_lock:
            if self.executing:
                raise RuntimeError("another planning or execution request is active")
        self._acquire_operation("v11_preview")
        try:
            with self.state_lock:
                if self.executing:
                    raise RuntimeError(
                        "another planning or execution request is active"
                    )
                if self.v11_controller is None:
                    raise ValueError("V11 preview requires v11_controller")
                if self.plan is None:
                    raise ValueError("V11 preview requires a live plan")
                if self.task_class is None:
                    raise ValueError("V11 preview requires a human-set task_class")
                plan = self.plan
                state = np.asarray(plan.current_state, dtype=np.float32).copy()
                observations = self.observations
                v11 = self.v11_controller
                task_class = self.task_class
            if state.shape != (6,) or not np.all(np.isfinite(state)):
                raise ValueError("plan.current_state must be a finite 6-vector")

            live_goal = self._resolve_v11_live_goal(plan, task_class, state)
            goal_raw = np.asarray(live_goal["goal_state"], dtype=np.float32)
            v11_result = v11.predict_chunk(
                observations,
                state,
                goal_state_raw=goal_raw,
                task_class=task_class,
            )
            selected = v11_result.get("selected_chunk")
            chunk_list: list[list[float]] | None = None
            lift_chunk: list[float] | None = None
            lift_delta_last: float | None = None
            toward_workspace_entry = False
            max_abs_joint_delta: float | None = None
            if selected is not None:
                chunk = np.asarray(selected, dtype=np.float32)
                if chunk.shape == (5, 6) and np.all(np.isfinite(chunk)):
                    chunk_list = chunk.tolist()
                    lift_chunk = [float(row[1]) for row in chunk]
                    lift_delta_last = float(chunk[-1, 1] - state[1])
                    toward_workspace_entry = lift_delta_last > 0.0
                    max_abs_joint_delta = float(np.max(np.abs(chunk - state)))
            return {
                "v11_live_goal": self._serialize_v11_live_goal(live_goal),
                "executable": bool(v11_result.get("executable", False)),
                "reason": v11_result.get("reason"),
                "chunk": chunk_list,
                "state": state.tolist(),
                "lift_now": float(state[1]),
                "lift_goal": float(live_goal["goal_lift_deg"]),
                "lift_chunk": lift_chunk,
                "lift_delta_last": lift_delta_last,
                "toward_workspace_entry": toward_workspace_entry,
                "max_abs_joint_delta": max_abs_joint_delta,
            }
        finally:
            self._release_operation()

    def _wrist_registration_required(
        self, plan: RealtimePlan | None
    ) -> bool:
        """Stop in the eye-in-hand acquisition window until pickup is marked."""

        if plan is None or plan.phase != ManipulationPhase.APPROACH:
            return False
        profile = getattr(plan, "target_profile", None)
        target = getattr(plan, "target", None)
        state = np.asarray(getattr(plan, "current_state", ()), dtype=np.float32)
        if not isinstance(profile, dict) or not isinstance(target, dict):
            return False
        if not profile.get("pickup_ready") or profile.get("wrist_pickup_ready"):
            return False
        planar_error = target.get("planar_error_m")
        if planar_error is None or not np.isfinite(planar_error):
            return False
        if float(planar_error) > self.wrist_registration_distance_m:
            return False
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            return False
        thresholds = getattr(self.planner, "thresholds", None)
        gripper_closed = getattr(thresholds, "gripper_closed", None)
        gripper_open = getattr(thresholds, "gripper_open", None)
        if gripper_open is None:
            return True
        if not np.isfinite(gripper_open):
            return False

        # The learned open threshold is a dataset percentile, not a motor
        # settle contract. Pause once the measured aperture has crossed half
        # of the closed-to-open range so modest execution lag cannot skip the
        # eye-in-hand acquisition window.
        minimum_open = float(gripper_open)
        if (
            gripper_closed is not None
            and np.isfinite(gripper_closed)
            and float(gripper_closed) < minimum_open
        ):
            minimum_open = 0.5 * (float(gripper_closed) + minimum_open)
        return bool(float(state[5]) >= minimum_open)

    def _run_grounding_cycle_locked(self) -> str | None:
        """Verify grounding, observe if warranted, and report whether to stop.

        Returns None to continue execution, or a stop reason.

        Called from inside ``execute`` with the operation lock held and the
        state lock released.  The bridge is given the frozen plan and may only
        verify, register, and report; every motion decision stays with the
        planner and with ``_execute_one_locked``.

        Fail-closed: any outcome other than a clean PROCEED or a successful
        registration returns a stop reason.  There is deliberately no path back
        to operator registration, because a recovery that needs a human is not
        a recovery.
        """
        bridge = self.grounding_bridge
        with self.state_lock:
            plan = self.plan
        if plan is None:
            return "wrist_registration_required"

        feedback, record = bridge.review(plan)
        if feedback is None or record is None:
            # No reviewable proposal: the plan carries no usable target pose, so
            # grounding cannot speak to it.  Preserve the previous halt rather
            # than inventing a verdict.
            return "wrist_registration_required"

        with self.state_lock:
            self.last_grounding_feedback = record.as_dict()

        if feedback.verdict is GroundingVerdict.PROCEED:
            # Uncertainty already fits inside the grasp tolerance, so the
            # near-field observation is not needed.  This is the case the
            # deployed distance gate could not express: it halted on geometry
            # alone, whatever the grounding quality was.
            bridge.finalise(record, "execute_without_observation")
            with self.state_lock:
                self.last_grounding_feedback = record.as_dict()
            return None

        if feedback.verdict is GroundingVerdict.REQUEST_OBSERVATION:
            result = bridge.observe_and_register(plan, record)
            if not result.ok:
                bridge.finalise(record, "halt_registration_failed")
                with self.state_lock:
                    self.last_grounding_feedback = record.as_dict()
                return f"grounding_registration_failed:{result.outcome.value}"
            # A new reference changes what the planner sees, so the next command
            # intent must come from a fresh plan rather than the frozen one that
            # triggered the observation.
            self._replan_locked(write_artifacts=True)
            bridge.finalise(record, "observed_registered_and_resumed")
            with self.state_lock:
                self.last_grounding_feedback = record.as_dict()
            return None

        # HALT, or any verdict this runtime does not implement.
        bridge.finalise(record, "halt")
        with self.state_lock:
            self.last_grounding_feedback = record.as_dict()
        return f"grounding_halt:{feedback.verdict.value}"

    def _acquire_operation(self, name: str) -> None:
        if not self.operation_lock.acquire(blocking=False):
            raise RuntimeError("another planning or execution request is active")
        with self.state_lock:
            self.active_operation = name

    def _release_operation(self) -> None:
        with self.state_lock:
            self.active_operation = None
        self.operation_lock.release()

    def bootstrap(self) -> dict[str, Any]:
        with self.operation_lock:
            reset_progress = getattr(self.planner, "reset_task_progress", None)
            if callable(reset_progress):
                reset_progress()
            initial = self.io.capture_initial(11, self.period_s)
            if len(initial) != 11:
                raise RuntimeError("I/O did not return eleven initial observations")
            self.observations.extend(initial)
            self._replan_locked()
            self.execution_armed = True
        return self.status()

    def _history(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        if len(self.observations) < 11:
            raise RuntimeError("runtime does not have a complete 11-frame history")
        selected = [self.observations[index] for index in (-11, -6, -1)]
        frames = {
            view: np.stack([row.frames[view] for row in selected]) for view in VIEWS
        }
        states = np.stack([row.state for row in selected]).astype(np.float32)
        return frames, states

    def _prod_v13_current_fingertips(
        self,
        frames: Mapping[str, np.ndarray],
        states: np.ndarray,
    ) -> LiveFingertipMeasurement | None:
        """Return the automatic fixed/moving posterior before planning.

        Manual/unordered query slots are intentionally not promoted. Before
        automatic H0 there is no Prod-V13 control measurement and the legacy
        planner remains available solely to acquire that H0 pulse.
        """

        observer = self.prod_v13_shadow
        if observer is None:
            return None
        status = observer.status()
        if not bool(status.get("initialized")):
            return None
        identity = status.get("identity")
        mapping = identity.get("mapping") if isinstance(identity, Mapping) else None
        if mapping != {"tip_a": "fixed", "tip_b": "moving"}:
            return None
        refresh = getattr(observer, "observe_measurement", None)
        if callable(refresh):
            latest_state = np.asarray(states[-1], dtype=np.float32)
            refresh(
                frames={view: np.asarray(frames[view][-1]) for view in VIEWS},
                state=latest_state,
                timestamp_s=(
                    self.observations[-1].timestamp_s
                    if self.observations
                    else time.monotonic()
                ),
            )
            status = observer.status()
        measurement = status.get("measurement")
        if not isinstance(measurement, Mapping):
            return None
        raw_points = measurement.get("tracker_tip_px")
        raw_visible = measurement.get("tracker_visible")
        source = "prod_v13_fcpgh_measurement"
        posterior = status.get("posterior")
        if isinstance(posterior, Mapping) and posterior.get("tip_px") is not None:
            raw_points = posterior.get("tip_px")
            source = "prod_v13_fcpgh_lewm_posterior"
        points = np.asarray(raw_points, dtype=np.float32)
        visible = np.asarray(raw_visible, dtype=bool)
        if points.shape != (2, 2, 2) or visible.shape != (2, 2):
            raise ValueError("Prod-V13 current fingertip payload has invalid shape")
        if isinstance(posterior, Mapping):
            view_valid = posterior.get("view_valid")
            if view_valid is not None:
                view_mask = np.asarray(view_valid, dtype=bool)
                if view_mask.shape != (2,):
                    raise ValueError("Prod-V13 posterior view_valid must be [2]")
                visible = visible & view_mask[:, None]
            missing = posterior.get("measurement_missing")
            if missing is not None:
                missing_mask = np.asarray(missing, dtype=bool)
                if missing_mask.shape != (2,):
                    raise ValueError(
                        "Prod-V13 posterior measurement_missing must be [2]"
                    )
                visible = visible & ~missing_mask[:, None]
        return LiveFingertipMeasurement(
            tip_px={view: points[index] for index, view in enumerate(VIEWS)},
            visible={view: visible[index] for index, view in enumerate(VIEWS)},
            source=source,
            generation=int(status.get("generation") or 0),
            horizon=int(status.get("rollout_horizon_completed") or 0),
        )

    def _causal_memory(self) -> CausalPlanningMemory | None:
        offsets = tuple(
            int(value) for value in getattr(self.planner, "memory_offsets_raw", ())
        )
        if not offsets:
            return None
        selected_observations: list[LiveObservation] = []
        selected_actions: list[np.ndarray] = []
        selected_offsets: list[int] = []
        for offset in offsets:
            if offset <= 0 or offset % 5:
                raise RuntimeError("planner memory offsets must be positive multiples of 5")
            required_chunks = 2 + offset // 5
            required_observations = 11 + offset
            if (
                len(self.past_chunks) < required_chunks
                or len(self.observations) < required_observations
            ):
                continue
            selected_observations.append(self.observations[-required_observations])
            selected_actions.append(self.past_chunks[-required_chunks])
            selected_offsets.append(offset)
        if not selected_observations:
            return None
        return CausalPlanningMemory(
            frames={
                view: np.stack([row.frames[view] for row in selected_observations])
                for view in VIEWS
            },
            states=np.stack([row.state for row in selected_observations]).astype(
                np.float32
            ),
            raw_actions=np.stack(selected_actions).astype(np.float32),
            offsets_raw=np.asarray(selected_offsets, dtype=np.int64),
        )

    def _replan_locked(self, *, write_artifacts: bool = True) -> None:
        if self.stopped:
            raise RuntimeError("controller is stopped")
        self._assert_v165_session_binding()
        frames, states = self._history()
        recent_chunks = tuple(self.past_chunks)[-2:]
        past = np.stack(recent_chunks) if recent_chunks else None
        if self.v11_controller is not None and self._v165_runtime_adapter is None:
            self.planner.extra_actor_prefix_provider = self._v11_actor_prefix_provider
        if self._v165_runtime_adapter is None:
            current_fingertips = self._prod_v13_current_fingertips(frames, states)
        else:
            try:
                current_fingertips = self._prod_v13_current_fingertips(
                    frames, states
                )
                self.last_prod_v13_shadow_error = None
            except Exception as exc:
                # A V16.5 shadow is not allowed to turn into an availability
                # gate for the authoritative planner.
                current_fingertips = None
                self.last_prod_v13_shadow_error = (
                    f"{type(exc).__name__}: {exc}"
                )
        plan_kwargs: dict[str, Any] = {
            "past_raw_actions": past,
            "causal_memory": self._causal_memory(),
        }
        if current_fingertips is not None and self._v165_runtime_adapter is None:
            plan_kwargs["current_fingertips"] = current_fingertips
        plan = self.planner.plan(frames, states, **plan_kwargs)
        observer = self.prod_v13_shadow
        if observer is not None:
            try:
                observer_status = observer.status()
                plan.prod_v13_tip_trajectory = {
                    "enabled": bool(observer_status.get("enabled")),
                    "initialized": bool(observer_status.get("initialized")),
                    "view_order": observer_status.get("view_order"),
                    "checkpoint_seed": observer_status.get("checkpoint_seed"),
                    "checkpoint_step": observer_status.get("checkpoint_step"),
                    "model_revision": observer_status.get("model_revision"),
                    "pixel_prediction_contract": observer_status.get(
                        "pixel_prediction_contract"
                    ),
                    "measurement": observer_status.get("measurement"),
                    "prior": observer_status.get("prior"),
                    "posterior": observer_status.get("posterior"),
                    "rows": observer_status.get("tip_trajectory") or [],
                    "control_influence": bool(
                        current_fingertips is not None
                        and self._v165_runtime_adapter is None
                    ),
                }
            except Exception as exc:
                if self._v165_runtime_adapter is None:
                    raise
                self.last_prod_v13_shadow_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                plan.prod_v13_tip_trajectory = {
                    "enabled": True,
                    "actuated": False,
                    "control_influence": False,
                    "reason": "shadow_error",
                    "error": self.last_prod_v13_shadow_error,
                }
        v14 = self.prod_v14_shadow
        if v14 is not None:
            try:
                plan.prod_v14_shadow = v14.rank_plan(plan, observer)
                self.last_prod_v14_shadow_error = None
            except Exception as exc:
                self.last_prod_v14_shadow_error = f"{type(exc).__name__}: {exc}"
                plan.prod_v14_shadow = {
                    "enabled": True,
                    "ranked": False,
                    "actuated": False,
                    "reason": "shadow_error",
                    "error": self.last_prod_v14_shadow_error,
                }
        v16_secant = self.prod_v16_secant_shadow
        if v16_secant is not None:
            try:
                plan.prod_v16_secant_shadow = v16_secant.rank_plan(plan, observer)
                self.last_prod_v16_secant_shadow_error = None
            except Exception as exc:
                self.last_prod_v16_secant_shadow_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                plan.prod_v16_secant_shadow = {
                    "enabled": True,
                    "non_actuating": True,
                    "proposed": False,
                    "reason": "shadow_error",
                    "error": self.last_prod_v16_secant_shadow_error,
                }
        # After plan() so load/current cannot change the commanded chunk.
        # Flag off: this call is skipped and the bus is not touched.
        if write_artifacts and servo_load_enabled(self.planner):
            attach_servo_load_telemetry(plan, self.io)
        self._log_lewm_rank_decision(plan)
        # Non-actuating V11 shadow pass — never modifies plan
        shadow_result: dict[str, Any] | None = None
        if self.v11_shadow is not None:
            current_state = self.observations[-1].state if self.observations else None
            if current_state is not None:
                shadow_result = self.v11_shadow.predict(
                    self.observations,
                    current_state,
                    task_class=self.task_class,
                )
        assets = (
            self.artifact_writer(plan)
            if write_artifacts and self.artifact_writer
            else None
        )
        with self.state_lock:
            if self.stopped or self.stop_event.is_set():
                raise RuntimeError("controller stopped while planning")
            self.plan = plan
            if assets is not None:
                self.assets = dict(assets)
            if shadow_result is not None:
                # ``predict`` above is deliberately called without
                # ``goal_state_raw``, so the shadow runs on its own current-state
                # goal proxy while the control path resolves a real live goal.
                # These numbers therefore describe a different input than the
                # chunk that actuated -- and on a two-view family the same runner
                # is also the LeWM ranker, which makes it easy to misread these
                # as the ranker's contribution. Label the artifact, because
                # ``live_goal_proxy`` alone has been read as the control goal.
                self.assets["v11_shadow"] = {
                    **shadow_result,
                    "actuated": False,
                    "goal_is_control_goal": False,
                    "evidence_scope": (
                        "proxy_goal_monitor_only; not evidence about the "
                        "executed chunk or about LeWM candidate ranking"
                    ),
                }
            self.error = None

    def replan(self) -> dict[str, Any]:
        self._acquire_operation("replan")
        try:
            live = self.io.observe()
            if self.observations:
                self.observations[-1] = live
            else:
                self.observations.append(live)
            self._replan_locked()
            with self.state_lock:
                self.execution_armed = True
            return self.status()
        except BaseException as exc:
            self._latch_fault(exc)
            raise
        finally:
            self._release_operation()

    def _operator_click_geometry(self) -> PlanningGeometry:
        """Declared planning geometry used to validate operator clicks.

        Taken from the planner's declared contract, never inferred from a frame
        shape.  A planner that predates the geometry contract (test doubles and
        stubs) is by construction from the ``squash_v0`` era, where the content
        rectangle *is* the whole canvas -- so the fallback reproduces the
        historical bound exactly rather than relaxing it.
        """

        geometry = getattr(self.planner, "planning_geometry", None)
        if isinstance(geometry, PlanningGeometry):
            return geometry
        return PlanningGeometry.squash_v0()

    def _require_content_click(self, point: np.ndarray, *, label: str) -> None:
        """Fail closed unless an operator click lands on real camera content.

        Both call sites feed ``column_live_pixel_to_base_xyz``, so the click is
        a claim that a physical object is visible at that pixel.  Under
        ``letterbox_v1`` the padding band contains no camera data at all, and
        backprojecting it would fabricate a metric position from black pixels.
        Under the ``squash_v0`` default the content rect is the full canvas and
        this reduces to the historical ``0 <= p < 224`` check exactly.
        """

        geometry = self._operator_click_geometry()
        if bool(geometry.contains_content_pixel(point)):
            return
        left, top, right, bottom = geometry.content_rect
        raise ValueError(
            f"{label} centroid lies outside the planning frame: "
            f"({float(point[0])!r}, {float(point[1])!r}) is not on camera "
            f"content for geometry_protocol={geometry.protocol!r} "
            f"(content rect [{left}, {right}) x [{top}, {bottom}) of a "
            f"{geometry.image_size}x{geometry.image_size} canvas; "
            f"{geometry.pad_rows} padded row(s) carry no camera data)"
        )

    def set_target_reference(
        self,
        role: str,
        view: str,
        centroid_px: np.ndarray,
        *,
        radius_px: float,
        round_id: str,
    ) -> dict[str, Any]:
        """Bind appearance to one exact frozen planning round, then replan."""

        if not isinstance(round_id, str) or not round_id:
            raise ValueError("target round_id must be a non-empty string")
        if role not in {"pickup", "drop_zone"}:
            raise ValueError("target role must be pickup or drop_zone")
        if view not in VIEWS:
            raise ValueError("target reference view must be wrist or column")
        if role == "drop_zone" and view != "column":
            raise ValueError("drop_zone must be configured in the column view")
        point = np.asarray(centroid_px, dtype=np.float32)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("target centroid must be a finite two-vector")
        self._require_content_click(point, label="target")
        if not np.isfinite(radius_px) or not 6 <= radius_px <= 112:
            raise ValueError("target radius must be between 6 and 112 pixels")

        self._acquire_operation("set_target_reference")
        try:
            with self.state_lock:
                if self.stopped:
                    raise RuntimeError("controller is stopped")
                plan = self.plan
                if plan is None:
                    raise PlanRejectedError(
                        "no frozen planning frame is available"
                    )
                if round_id != plan.guard.round_id:
                    raise PlanRejectedError(
                        "stale planning round; reload the current target frame"
                    )
                frame = np.asarray(plan.frames[view][-1], dtype=np.uint8).copy()
            self.planner.set_target_reference(
                role,
                view,
                frame,
                point,
                radius_px=float(radius_px),
            )
            reset_progress = getattr(self.planner, "reset_task_progress", None)
            if callable(reset_progress):
                reset_progress()
            self._replan_locked()
            with self.state_lock:
                self.execution_armed = True
            return self.status()
        except (ValueError, PlanRejectedError):
            raise
        except BaseException as exc:
            self._latch_fault(exc)
            raise
        finally:
            self._release_operation()

    def set_context_reference(
        self,
        round_id: str,
        centroid_px: np.ndarray,
        *,
        radius_px: float,
    ) -> dict[str, Any]:
        """Bind one session-only peg to an exact frozen planning round."""

        if not isinstance(round_id, str) or not round_id:
            raise ValueError("context round_id must be a non-empty string")
        point = np.asarray(centroid_px, dtype=np.float32)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("context centroid must be a finite two-vector")
        self._require_content_click(point, label="context")
        if not np.isfinite(radius_px) or not 6 <= radius_px <= 112:
            raise ValueError("context radius must be between 6 and 112 pixels")

        self._acquire_operation("set_context_reference")
        try:
            with self.state_lock:
                if self.stopped:
                    raise RuntimeError("controller is stopped")
                plan = self.plan
                if plan is None:
                    raise PlanRejectedError(
                        "no frozen planning frame is available"
                    )
                if round_id != plan.guard.round_id:
                    raise PlanRejectedError(
                        "stale planning round; reload the current column frame"
                    )
                frame = np.asarray(
                    plan.frames["column"][-1], dtype=np.uint8
                ).copy()

            if self.io.is_hardware:
                manifest = getattr(self.io, "camera_manifest", None)
                if not isinstance(manifest, Mapping):
                    raise PlanRejectedError(
                        "hardware context registration requires a trusted "
                        "camera manifest"
                    )
                if manifest.get("schema") != "so101-camera-role-manifest-v1":
                    raise PlanRejectedError(
                        "hardware context registration camera manifest has an "
                        "unsupported schema"
                    )
                cameras = manifest.get("cameras")
                if not isinstance(cameras, Mapping):
                    raise PlanRejectedError(
                        "hardware context registration camera manifest lacks roles"
                    )
                column_binding = cameras.get("column")
                if not isinstance(column_binding, Mapping):
                    raise PlanRejectedError(
                        "hardware context registration lacks a trusted column "
                        "camera binding"
                    )
                if column_binding.get("role") != "column":
                    raise PlanRejectedError(
                        "hardware context registration camera binding has the "
                        "wrong role"
                    )
                for field in CAMERA_BINDING_STABLE_FIELDS:
                    value = column_binding.get(field)
                    if not isinstance(value, str) or not value:
                        raise PlanRejectedError(
                            "hardware context registration camera binding lacks "
                            f"a stable {field}"
                        )
                camera_binding = dict(column_binding)
                camera_binding_source = "io.camera_manifest.cameras.column"
            else:
                camera_binding = {
                    "role": "column",
                    "binding": "replay_session",
                }
                camera_binding_source = "replay_session"

            setter = getattr(self.planner, "set_context_reference", None)
            if not callable(setter):
                raise PlanRejectedError(
                    "planner does not expose session context registration"
                )
            try:
                setter(
                    frame,
                    point,
                    radius_px=float(radius_px),
                    registration={
                        "round_id": round_id,
                        "view": "column",
                        "registered_monotonic_s": time.monotonic(),
                        "camera_binding": camera_binding,
                        "camera_binding_source": camera_binding_source,
                    },
                )
            except ContextRegistrationError as exc:
                raise PlanRejectedError(
                    f"peg context registration rejected: {exc}"
                ) from exc
            self._replan_locked()
            with self.state_lock:
                self.execution_armed = True
            return self.status()
        except (ValueError, PlanRejectedError):
            raise
        except BaseException as exc:
            self._latch_fault(exc)
            raise
        finally:
            self._release_operation()

    def _column_reanchor_required(self, plan: RealtimePlan | None) -> bool:
        """Whether the column pulse fit has been left behind and must be remade.

        The support ball is a measurement, not a workspace. Leaving it and
        falling back to the raw FK projection is how a moved column camera
        (or an arm that walked off the last pulse pose) puts the overlay on
        the fish while the jaws sit at the top of the frame. APPROACH is the
        only phase that may pulse: GRASP and TRANSPORT may be holding.
        """

        if self._v165_runtime_adapter is not None or self._v170_runtime_adapter is not None:
            return False
        if plan is None or plan.phase != ManipulationPhase.APPROACH:
            return False
        blocked = getattr(plan, "blocked_reasons", None)
        if not isinstance(blocked, (list, tuple, set)):
            return False
        return "column_live_anchor_out_of_support" in blocked

    def _prod_v13_automatic_initialization_required(self) -> bool:
        # V16.5 promises that ProdV13 is a passive observer. Its historical H0
        # initialization pulse is an actuator command, so it is unavailable in
        # these immutable sessions rather than being hidden inside execute().
        if self._v165_runtime_adapter is not None or self._v170_runtime_adapter is not None:
            return False
        observer = self.prod_v13_shadow
        if observer is None or not callable(
            getattr(observer, "initialize_from_pulse", None)
        ):
            return False
        status = observer.status()
        return not bool(status.get("initialized"))

    def _pulse_open_column_frame(
        self, calibration: Mapping[str, Any]
    ) -> np.ndarray | None:
        """Exact pulse open frame from the same capture. None if unavailable."""

        capture = calibration.get("capture")
        open_index = calibration.get("open_frame_index")
        if not isinstance(capture, Mapping) or open_index is None:
            return None
        npz_path = capture.get("npz")
        if npz_path is None:
            return None
        payload = np.load(str(npz_path))
        frames = payload["column_frames"] if "column_frames" in payload.files else None
        if frames is None:
            return None
        index = int(open_index)
        if index < 0 or index >= len(frames):
            return None
        return np.asarray(frames[index], dtype=np.uint8)

    def _pulse_role_boxes_xyxy(
        self, calibration: Mapping[str, Any]
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
        """Task-table role boxes on the pulse open frame. Not tip authority."""

        frame = self._pulse_open_column_frame(calibration)
        if frame is None:
            return (), ()
        lock = getattr(self.planner, "_lock_column_roles", None)
        if not callable(lock):
            return (), ()
        hits = lock(frame)
        return role_boxes_for_pulse_tracks(hits or {})

    def _pulse_metal_tip_hint(
        self, calibration: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Independent distal identity from the same pulse's exact open frame."""

        frame = self._pulse_open_column_frame(calibration)
        if frame is None:
            return None
        column = calibration.get("views", {}).get("column") if isinstance(
            calibration.get("views"), Mapping
        ) else None
        bbox = None
        tracks = None
        if isinstance(column, Mapping):
            bbox = column.get("bbox_xyxy", calibration.get("bbox_xyxy"))
            tracks = column.get("moving_tip_track_candidates_px")
        else:
            bbox = calibration.get("bbox_xyxy")
            tracks = calibration.get("moving_tip_track_candidates_px")
        return independent_open_frame_metal_tip_hint(
            frame,
            support_bbox_xyxy=bbox,
            candidate_tracks_px=tracks,
        )

    def _drop_observer_adjacency(self) -> None:
        """Forget pre-pulse neighbour frames after an independent re-anchor.

        The 3-D posterior is kept. Only the adjacent-frame motion buffer is
        dropped, including any pending single-tip stack that still belongs to
        the old association frame.
        """

        clear_observer_adjacent_history(self.planner)

    def _initialize_prod_v13_from_pulse_capture(
        self, calibration: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Feed the exact saved dual-view pulse to the automatic FCPG-H H0."""

        observer = self.prod_v13_shadow
        initialize = getattr(observer, "initialize_from_pulse", None)
        if not callable(initialize):
            return None
        capture = calibration.get("capture")
        if not isinstance(capture, Mapping) or capture.get("npz") is None:
            raise RuntimeError(
                "automatic Prod-V13 initialization requires the saved pulse.npz"
            )
        path = Path(str(capture["npz"])).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"automatic initialization pulse is missing: {path}")
        with np.load(path, allow_pickle=False) as payload:
            required = {"wrist_frames", "column_frames", "states"}
            if not required.issubset(payload.files):
                raise ValueError(
                    "automatic initialization pulse lacks dual-view RGB/states"
                )
            frames = {
                "wrist": np.asarray(payload["wrist_frames"], dtype=np.uint8),
                "column": np.asarray(payload["column_frames"], dtype=np.uint8),
            }
            states = np.asarray(payload["states"], dtype=np.float32)
        return initialize(frames_by_view=frames, states=states)

    def _apply_prod_v13_auto_initialization_locked(self) -> dict[str, Any]:
        """Run only the local FCPG-H excitation path, then refresh the plan."""

        self._reject_no_contact_side_motion("ProdV13 automatic initialization pulse")

        capture = getattr(self.io, "capture_gripper_for_fcpgh", None)
        if not callable(capture):
            raise RuntimeError(
                "runtime I/O does not expose automatic FCPG-H pulse capture"
            )
        result = capture(stop_event=self.stop_event)
        automatic = self._initialize_prod_v13_from_pulse_capture(result)
        if automatic is None or not bool(automatic.get("initialized")):
            raise RuntimeError("automatic FCPG-H pulse did not initialize the observer")
        result = {
            **result,
            "prod_v13_auto_initialization": automatic,
            "legacy_pixel_localizer_used": False,
        }
        refreshed = self.io.capture_initial(11, self.period_s)
        if len(refreshed) != 11:
            raise RuntimeError(
                "automatic FCPG-H refresh did not return eleven observations"
            )
        self.observations.clear()
        self.observations.extend(refreshed)
        self.past_chunks.clear()
        self._reset_v168_window("prod_v13_automatic_initialization")
        self._reset_v170_window("prod_v13_automatic_initialization")
        self._replan_locked()
        with self.state_lock:
            self.execution_armed = True
            self.calibration = _jsonable(result)
        return result

    def initialize_prod_v13_automatically(
        self, *, confirmed: bool
    ) -> dict[str, Any]:
        """Run automatic H0 and replan without sending a task action chunk."""

        self._reject_no_contact_side_motion("ProdV13 automatic initialization pulse")

        if self._v165_runtime_adapter is not None:
            raise RuntimeError(
                "ProdV13 initialization pulses are disabled in V16.5 shadow-only sessions"
            )
        if not confirmed:
            raise RuntimeError(
                "automatic FCPG-H initialization requires explicit confirmation"
            )
        if not self.execute_enabled:
            raise RuntimeError(
                "automatic FCPG-H initialization requires command execution"
            )
        self._acquire_operation("prod_v13_auto_initialization")
        try:
            with self.state_lock:
                if self.stopped:
                    raise RuntimeError("controller is stopped")
                if self.plan is None or self.plan.phase != ManipulationPhase.APPROACH:
                    raise PlanRejectedError(
                        "automatic FCPG-H initialization requires APPROACH"
                    )
                self.stop_event.clear()
            if not self._prod_v13_automatic_initialization_required():
                self._replan_locked()
                return self.status()
            self._apply_prod_v13_auto_initialization_locked()
            return self.status()
        except (ValueError, PlanRejectedError):
            raise
        except BaseException as exc:
            self._latch_fault(exc)
            raise
        finally:
            self._release_operation()

    def _apply_column_pulse_locked(self) -> dict[str, Any]:
        """Pulse the jaws, fit the column translation, refresh, and replan.

        Caller holds the operation lock and has already decided this pulse is
        allowed. A failed track fit raises; the wrappers latch that as a fault.
        """

        self._reject_no_contact_side_motion("grounding calibration pulse")

        result = self.io.calibrate_gripper(stop_event=self.stop_event)
        automatic_initialization = self._initialize_prod_v13_from_pulse_capture(
            result
        )
        views = result.get("views")
        if not isinstance(views, dict) or set(views) != {"column"}:
            raise RuntimeError("gripper calibration did not return the column view")
        exclude_bboxes, prefer_bboxes = self._pulse_role_boxes_xyxy(result)
        tip_hint = self._pulse_metal_tip_hint(result)
        hint_px = None if tip_hint is None else tip_hint.get("hint_px")
        hint_radius = None if tip_hint is None else tip_hint.get("radius_px")
        queries = getattr(self.planner, "_task_visual_queries", None)
        configured = queries() if callable(queries) else {}
        column_row = views["column"]
        moving_tip_tracks = np.asarray(
            column_row["moving_tip_track_candidates_px"], dtype=np.float32
        )
        decision = pulse_tracks_allowed(
            moving_tip_tracks,
            gripper_role_configured="gripper" in configured,
            prefer_bboxes_xyxy=prefer_bboxes,
            exclude_bboxes_xyxy=exclude_bboxes,
            opening_span=result.get("opening_span", column_row.get("opening_span")),
            metal_tip_hint_px=hint_px,
            metal_tip_hint_radius_px=hint_radius,
        )
        if not decision.allowed:
            raise GroundingPulseRejectedError(f"PULSE REFUSED: {decision.reason}")
        geometry: dict[str, dict[str, Any]] = {}
        for view in ("column",):
            row = views[view]
            moving_tip_tracks = np.asarray(
                row["moving_tip_track_candidates_px"], dtype=np.float32
            )
            open_state = np.asarray(row["open_state"], dtype=np.float32)
            closed_state = np.asarray(row["closed_state"], dtype=np.float32)
            try:
                geometry[view] = self.planner.set_live_moving_tip_track(
                    view,
                    moving_tip_tracks,
                    open_state,
                    closed_state,
                    exclude_bboxes_xyxy=exclude_bboxes,
                    prefer_bboxes_xyxy=prefer_bboxes,
                    metal_tip_hint_px=hint_px,
                    metal_tip_hint_radius_px=hint_radius,
                )
            except ValueError as exc:
                if not is_recoverable_pulse_track_fit_error(exc):
                    raise
                raise GroundingPulseRejectedError(str(exc)) from exc
            if tip_hint is not None:
                geometry[view] = {
                    **geometry[view],
                    "metal_tip_hint_source": tip_hint.get("source"),
                }
        # The pulse is an explicit, independently checked re-acquisition of
        # column geometry. A lock transported from before this re-anchor is in
        # the old association frame; retaining it can make every fresh
        # motion-certified return tip fail the 10 px hard gate forever.
        # Drop adjacency history too: the post-pulse refresh is not a neighbour
        # of the pre-pulse frame, so jaw-open optical flow must not enter the
        # 3-D observer. The recursive posterior itself stays intact.
        fingertip_gate = getattr(self.planner, "_fingertip_gate", None)
        reset_fingertip_gate = getattr(fingertip_gate, "reset", None)
        if callable(reset_fingertip_gate):
            reset_fingertip_gate()
            self.planner._fingertip_last_result = None
        self._drop_observer_adjacency()
        result = {
            **result,
            "observed_anchor": "moving_tip_open_closed_track",
            "geometry": geometry,
        }
        if automatic_initialization is not None:
            result["prod_v13_auto_initialization"] = automatic_initialization
        refreshed = self.io.capture_initial(11, self.period_s)
        if len(refreshed) != 11:
            raise RuntimeError(
                "calibration refresh did not return eleven observations"
            )
        self.observations.clear()
        self.observations.extend(refreshed)
        self.past_chunks.clear()
        self._reset_v168_window("grounding_calibration_refresh")
        self._reset_v170_window("grounding_calibration_refresh")
        clear_context = getattr(self.planner, "clear_context_reference", None)
        if callable(clear_context):
            clear_context()
        if self.calibration_committer is not None:
            metadata = self.calibration_committer(result)
            if not isinstance(metadata, Mapping):
                raise TypeError("calibration_committer must return a mapping")
            result = {**result, **dict(metadata)}
        self._replan_locked()
        with self.state_lock:
            self.execution_armed = True
            self.calibration = _jsonable(result)
        return result

    def calibrate_grounding(self, *, confirmed: bool) -> dict[str, Any]:
        """Run one gripper-only pulse and re-anchor the external column view."""

        self._reject_no_contact_side_motion("grounding calibration")

        if not confirmed:
            raise RuntimeError("grounding calibration requires explicit human confirmation")
        if not self.calibration_enabled:
            raise RuntimeError("gripper calibration is disabled for this server")
        if not self.io.is_hardware:
            raise RuntimeError("grounding calibration is only available on real hardware")
        self._acquire_operation("calibrate_grounding")
        try:
            with self.state_lock:
                if self.stopped:
                    raise RuntimeError("controller is stopped")
                plan = self.plan
                executed = any(
                    receipt.get("kind", "execute_chunk") == "execute_chunk"
                    for receipt in self.receipts
                )
                if plan is None or plan.phase not in {
                    ManipulationPhase.APPROACH,
                    ManipulationPhase.TRAJECTORY,
                }:
                    raise PlanRejectedError(
                        "grounding calibration is only allowed before task execution"
                    )
                if executed:
                    raise PlanRejectedError(
                        "grounding calibration must run before the first task command"
                    )
                self.executing = True
                self.execution_armed = False
            self.stop_event.clear()
            self._apply_column_pulse_locked()
            with self.state_lock:
                self.executing = False
            return self.status()
        except PlanRejectedError:
            with self.state_lock:
                self.executing = False
            raise
        except BaseException as exc:
            self._latch_fault(exc)
            raise
        finally:
            self._release_operation()

    def reanchor_grounding(self, *, confirmed: bool) -> dict[str, Any]:
        """Remake the column pulse fit after the arm leaves its support ball.

        Unlike ``calibrate_grounding`` this is allowed after task chunks have
        already run. It is still refused in GRASP and TRANSPORT: those phases
        may be holding, and a jaw pulse there is not a calibration.
        """

        self._reject_no_contact_side_motion("grounding re-anchor")

        if not confirmed:
            raise RuntimeError("grounding re-anchor requires explicit human confirmation")
        if not self.calibration_enabled:
            raise RuntimeError("gripper calibration is disabled for this server")
        if not self.io.is_hardware:
            raise RuntimeError("grounding re-anchor is only available on real hardware")
        self._acquire_operation("reanchor_grounding")
        try:
            with self.state_lock:
                if self.stopped:
                    raise RuntimeError("controller is stopped")
                plan = self.plan
                if plan is None or plan.phase != ManipulationPhase.APPROACH:
                    raise PlanRejectedError(
                        "grounding re-anchor is only allowed during approach"
                    )
                self.executing = True
                self.execution_armed = False
            self.stop_event.clear()
            self._apply_column_pulse_locked()
            with self.state_lock:
                self.executing = False
            return self.status()
        except PlanRejectedError:
            with self.state_lock:
                self.executing = False
            raise
        except BaseException as exc:
            self._latch_fault(exc)
            raise
        finally:
            self._release_operation()

    def _calibration_gripper_stops(self) -> tuple[float, float]:
        opened = getattr(self.io, "gripper_calibration_open", None)
        if opened is None:
            opened = getattr(self.planner, "gripper_calibration_open", None)
        closed = getattr(self.io, "gripper_calibration_closed", None)
        if closed is None:
            closed = getattr(self.planner, "gripper_calibration_closed", None)
        if opened is None or closed is None:
            raise PlanRejectedError(
                "calibrated gripper open/closed stops are missing"
            )
        return float(opened), float(closed)

    def _excitation_planar_xy(self, state: np.ndarray) -> np.ndarray | None:
        """Optional task-plane retreat from a locked pickup. +Z if unavailable."""

        measurements = getattr(self.planner, "_column_role_measurements", None) or {}
        pickup = measurements.get("pickup") or {}
        gripper = measurements.get("gripper") or {}
        if not pickup.get("visible") or not gripper.get("visible"):
            return None
        gripper_box = gripper.get("bbox_xyxy")
        if gripper_box is None:
            return None
        box = tuple(float(value) for value in gripper_box)
        grip_px = np.asarray(
            (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])), dtype=np.float64
        )
        pickup_centroid = pickup.get("centroid_px")
        pickup_box = pickup.get("bbox_xyxy")
        if pickup_centroid is not None:
            pickup_px = np.asarray(pickup_centroid, dtype=np.float64).reshape(2)
        elif pickup_box is not None:
            pbox = tuple(float(value) for value in pickup_box)
            pickup_px = np.asarray(
                (0.5 * (pbox[0] + pbox[2]), 0.5 * (pbox[1] + pbox[3])),
                dtype=np.float64,
            )
        else:
            return None
        fits = getattr(self.planner, "fits", None) or {}
        column_fit = fits.get("column")
        kinematics = getattr(self.planner, "kinematics", None)
        if column_fit is None or kinematics is None:
            return None
        from fcpg.so101_camera_calibration import (
            approximate_column_backprojection_on_plane,
        )

        try:
            current_xyz = np.asarray(kinematics.forward(state), dtype=np.float64).reshape(
                3
            )
            pickup_xyz = approximate_column_backprojection_on_plane(
                column_fit, pickup_px, z_base=float(current_xyz[2])
            )
            grip_xyz = approximate_column_backprojection_on_plane(
                column_fit, grip_px, z_base=float(current_xyz[2])
            )
        except ValueError:
            return None
        return np.asarray(grip_xyz[:2] - pickup_xyz[:2], dtype=np.float64)

    def calibrate_with_excitation(self, *, confirmed: bool) -> dict[str, Any]:
        """Hop to free space, pulse, then return. No in-place fallback."""

        self._reject_no_contact_side_motion("excitation calibration")

        if not confirmed:
            raise RuntimeError(
                "excitation calibration requires explicit human confirmation"
            )
        if not self.calibration_enabled:
            raise RuntimeError("gripper calibration is disabled for this server")
        if not self.io.is_hardware:
            raise RuntimeError(
                "excitation calibration is only available on real hardware"
            )
        self._acquire_operation("calibrate_with_excitation")
        pre_grasp: np.ndarray | None = None
        previously_armed = False
        try:
            with self.state_lock:
                previously_armed = self.execution_armed
                if self.stopped:
                    raise RuntimeError("controller is stopped")
                plan = self.plan
                if plan is None or plan.phase != ManipulationPhase.APPROACH:
                    raise PlanRejectedError(
                        "excitation is only allowed during APPROACH"
                    )
                self.executing = True
                self.execution_armed = False
            self.stop_event.clear()
            live = self.io.observe()
            pre_grasp = np.asarray(live.state, dtype=np.float32).copy()
            open_gripper, closed_gripper = self._calibration_gripper_stops()
            preflight = excitation_preflight(
                phase=plan.phase.value,
                current_gripper=float(pre_grasp[5]),
                open_gripper=open_gripper,
                closed_gripper=closed_gripper,
            )
            if not preflight.allowed:
                raise PlanRejectedError(preflight.reason)
            kinematics = getattr(self.planner, "kinematics", None)
            if kinematics is None or not hasattr(kinematics, "inverse_position"):
                raise PlanRejectedError("excitation requires planner kinematics")
            try:
                pose = solve_excitation_pose(
                    kinematics,
                    pre_grasp,
                    open_gripper=open_gripper,
                    closed_gripper=closed_gripper,
                    planar_xy=self._excitation_planar_xy(pre_grasp),
                )
            except ValueError as exc:
                raise PlanRejectedError(str(exc)) from exc
            hop_receipt = self._interpolate_to_state(
                pose.target_state, kind="excitation_hop"
            )
            excursion_error: BaseException | None = None
            result: dict[str, Any] | None = None
            observer_checkpoints: dict[str, Any] = {}
            try:
                # The recursive observer only updates on adjacent planning
                # frames with real arm motion.  Previously the whole hop was
                # hidden inside interpolation and the next plan happened only
                # after the jaw pulse, so the D-optimal motion could never
                # reduce the close-direction covariance.  Observe and plan at
                # the hop endpoint before moving the jaw; this preserves the
                # existing motion-tip and hard-reject gates.
                self.observations.append(self.io.observe())
                self._replan_locked(write_artifacts=False)
                observer_checkpoints["hop"] = _jsonable(
                    getattr(self.planner, "_observer_last_diagnostics", None)
                )
                result = self._apply_column_pulse_locked()
            except BaseException as exc:
                excursion_error = exc
            try:
                return_receipt = self._interpolate_to_state(
                    pre_grasp, kind="excitation_return"
                )
            except BaseException as return_exc:
                if excursion_error is not None:
                    raise RuntimeError(
                        "excitation return failed after excursion error "
                        f"({excursion_error}): {return_exc}"
                    ) from return_exc
                raise
            pulse_rejected = isinstance(
                excursion_error, GroundingPulseRejectedError
            )
            if excursion_error is not None and not pulse_rejected:
                raise excursion_error
            # Pair the return endpoint with the pulse's hop-frame plan before
            # replacing history with static refresh frames.  This gives the
            # observer a second independent, real-motion update opportunity;
            # no colour-only or held tip is admitted.
            self.observations.append(self.io.observe())
            self._replan_locked(write_artifacts=False)
            observer_checkpoints["return"] = _jsonable(
                getattr(self.planner, "_observer_last_diagnostics", None)
            )
            refreshed = self.io.capture_initial(11, self.period_s)
            if len(refreshed) != 11:
                raise RuntimeError(
                    "excitation return refresh did not return eleven observations"
                )
            self.observations.clear()
            self.observations.extend(refreshed)
            if pulse_rejected:
                # The arm returned successfully and only the independent visual
                # fit was refused. Restore task phase bookkeeping so the operator
                # can retry excitation without restarting (and losing the
                # recursive observer posterior). History and past chunks must
                # stay paired: the refresh replaced observations, so any
                # pre-pulse chunks would mix with the new 11-frame window.
                tracker = getattr(self.planner, "phase_tracker", None)
                if tracker is not None and hasattr(tracker, "reset"):
                    tracker.reset()
                reset_progress = getattr(self.planner, "reset_task_progress", None)
                if callable(reset_progress):
                    reset_progress()
                self.past_chunks.clear()
                self._reset_v168_window("excitation_refresh_rejected")
                self._reset_v170_window("excitation_refresh_rejected")
            self._replan_locked()
            with self.state_lock:
                self.executing = False
                # A refused visual fit is not an execute grant. Restore the
                # pre-call arming bit so a previously disarmed controller
                # cannot be re-armed by this recovery path.
                self.execution_armed = (
                    True if result is not None and not pulse_rejected
                    else previously_armed
                )
                if result is not None:
                    self.calibration = {
                        **_jsonable(result),
                        "excitation": {
                            "hop": _jsonable(hop_receipt),
                            "return": _jsonable(return_receipt),
                            "sweep_radius_m": pose.sweep_radius_m,
                            "hop_m": pose.hop_m,
                            "hop_fraction": pose.hop_fraction,
                            "delta_xyz": pose.delta_xyz.tolist(),
                            "observer_checkpoints": observer_checkpoints,
                        },
                    }
            if pulse_rejected:
                raise PlanRejectedError(
                    "excitation pulse rejected after safe return: "
                    f"{excursion_error}"
                ) from excursion_error
            return self.status()
        except PlanRejectedError:
            with self.state_lock:
                self.executing = False
                self.execution_armed = previously_armed
            raise
        except BaseException as exc:
            self._latch_fault(exc)
            raise
        finally:
            self._release_operation()

    def _latch_fault(self, exc: BaseException) -> None:
        self.stop_event.set()
        with self.state_lock:
            if self.stopped and self.error is not None:
                self.executing = False
                return
            self.stopped = True
            self.executing = False
            self.error = f"{type(exc).__name__}: {exc}"
        fault = getattr(self.planner, "phase_tracker", None)
        if fault is not None and hasattr(fault, "fault"):
            fault.fault()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        with self.state_lock:
            self.stopped = True
            self.executing = False
            if self.error is None:
                self.error = "operator stop latched"
        fault = getattr(self.planner, "phase_tracker", None)
        if fault is not None and hasattr(fault, "fault"):
            fault.fault()
        return self.status()

    def preview_frame(self, view: str) -> np.ndarray:
        """Return a monitor frame without advancing the LeWM observation clock."""

        if view not in VIEWS:
            raise ValueError(f"unsupported preview view: {view}")
        getter = getattr(self.io, "preview_frame", None)
        if not callable(getter):
            raise RuntimeError("runtime I/O does not expose camera previews")
        frame = np.asarray(getter(view))
        if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
            raise RuntimeError(f"{view} preview must be uint8 [224,224,3]")
        return frame.copy()

    def _interpolate_to_state(
        self,
        target_state: np.ndarray,
        *,
        kind: str,
    ) -> dict[str, Any]:
        """Joint-space interpolation with the same discipline as return-home."""

        self._reject_no_contact_side_motion(kind)

        target = np.asarray(target_state, dtype=np.float32)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise RuntimeError(f"{kind} target must be a finite six-vector")
        live = self.io.observe()
        start = np.asarray(live.state, dtype=np.float32)
        maximum_delta = float(np.max(np.abs(target - start)))
        minimum_steps = max(1, int(np.ceil(maximum_delta / self.home_max_step)))
        command_count = 5 * int(np.ceil(minimum_steps / 5))
        path = np.linspace(start, target, command_count + 1, dtype=np.float32)[1:]
        sent_chunks: list[np.ndarray] = []
        observed_rows: list[LiveObservation] = []
        final_observation = live
        observer = self.prod_v13_shadow
        observer_status_before = observer.status() if observer is not None else {}
        transition_count_before = int(
            observer_status_before.get("recorded_transition_count") or 0
        )
        artifacts_before = tuple(
            str(value)
            for value in observer_status_before.get("transition_artifacts") or ()
        )
        for offset in range(0, command_count, 5):
            chunk = path[offset : offset + 5]
            previous_planned = start if offset == 0 else path[offset - 1]
            planned = np.concatenate((previous_planned[None], chunk), axis=0)
            planned_step = float(np.max(np.abs(np.diff(planned, axis=0))))
            if planned_step > self.home_max_step + 1e-5:
                raise RuntimeError(
                    f"{kind} interpolation exceeds the per-step limit"
                )
            check = validate_command_chunk(
                chunk,
                final_observation.state,
                max_first_jump=self.max_command_jump,
                max_step_jump=self.max_command_jump,
                outward_tolerance=self.home_max_step + 1e-5,
            )
            if not check.safe:
                raise RuntimeError(
                    f"{kind} interpolation failed safety checks: "
                    + ", ".join(check.reasons)
                )
            if self.stop_event.is_set():
                raise RuntimeError(f"{kind} stopped by operator")
            # Return-home and calibration hops are real robot motion, not an
            # invisible teleport between two planning frames.  Feed every
            # exact five-command interval and its five dual-view observations
            # through the same Prod-V13 transition path used by task chunks.
            # This keeps the query/tracker/posterior history causal across a
            # large interpolation instead of asking CoTracker to bridge the
            # whole move from one stale endpoint image.
            if observer is not None and self.plan is not None:
                self._start_prod_v13_shadow_rollout(
                    self.plan,
                    planned_actions=chunk,
                    current_observation=final_observation,
                )
            result = self.io.execute_chunk(
                chunk,
                period_s=self.home_period_s,
                stop_event=self.stop_event,
            )
            if not np.allclose(result.sent, result.requested, atol=1e-4, rtol=0.0):
                raise RuntimeError(f"LeRobot changed a {kind} command")
            sent_chunks.append(result.sent.copy())
            observed_rows.extend(result.observations)
            self._advance_prod_v13_shadow(result)
            final_observation = result.observations[-1]
        observer_status_after = observer.status() if observer is not None else {}
        transition_count_after = int(
            observer_status_after.get("recorded_transition_count") or 0
        )
        artifacts_after = tuple(
            str(value)
            for value in observer_status_after.get("transition_artifacts") or ()
        )
        new_artifacts = (
            artifacts_after[len(artifacts_before) :]
            if artifacts_after[: len(artifacts_before)] == artifacts_before
            else ()
        )
        return {
            "kind": kind,
            "before_state": start.copy(),
            "requested_state": target.copy(),
            "after_state": final_observation.state.copy(),
            "maximum_initial_delta": maximum_delta,
            "command_count": command_count,
            "sent": np.concatenate(sent_chunks, axis=0),
            "observed_timestamps_s": np.asarray(
                [row.timestamp_s for row in observed_rows], dtype=np.float64
            ),
            "observed_states": np.stack(
                [row.state for row in observed_rows]
            ).astype(np.float32),
            "observer_transition_count_before": transition_count_before,
            "observer_transition_count_after": transition_count_after,
            "observer_transitions_recorded": (
                transition_count_after - transition_count_before
            ),
            "observer_transition_artifacts": list(new_artifacts),
            "observer_transition_error": self.last_prod_v13_shadow_error,
            "hardware": True,
        }

    def return_home(self, *, confirmed: bool) -> dict[str, Any]:
        self._reject_no_contact_side_motion("return home")
        if not confirmed:
            raise RuntimeError("return home requires explicit human confirmation")
        if not self.execute_enabled:
            raise RuntimeError("command execution is disabled for this server")
        if self.home_state is None:
            raise RuntimeError("no task-start home state is configured")
        if not self.io.is_hardware:
            raise RuntimeError("return home is only available on real hardware")
        self._acquire_operation("return_home")

        started = time.perf_counter()
        try:
            with self.state_lock:
                if self.stopped:
                    raise RuntimeError("controller is stopped")
                self.executing = True
            self.stop_event.clear()

            receipt = self._interpolate_to_state(
                self.home_state, kind="return_home"
            )
            receipt["requested_home_state"] = receipt["requested_state"]
            receipt["started_monotonic_s"] = started
            receipt["duration_s"] = time.perf_counter() - started
            self.receipts.append(receipt)

            refreshed = self.io.capture_initial(11, self.period_s)
            if len(refreshed) != 11:
                raise RuntimeError("home refresh did not return eleven observations")
            self.observations.clear()
            self.observations.extend(refreshed)
            self.past_chunks.clear()
            self._reset_v168_window("return_home_refresh")
            self._reset_v170_window("return_home_refresh")
            tracker = getattr(self.planner, "phase_tracker", None)
            if tracker is not None and hasattr(tracker, "reset"):
                tracker.reset()
            reset_grounding = getattr(
                self.planner, "reset_grounding_posterior", None
            )
            if callable(reset_grounding):
                reset_grounding()
            reset_progress = getattr(self.planner, "reset_task_progress", None)
            if callable(reset_progress):
                reset_progress()
            self._replan_locked()
            with self.state_lock:
                self.execution_armed = True
                self.executing = False
            return self.status()
        except BaseException as exc:
            self._latch_fault(exc)
            raise
        finally:
            self._release_operation()

    def prod_v13_observer_status(self, *, refresh: bool = False) -> dict[str, Any]:
        observer = self.prod_v13_shadow
        if observer is None:
            return {
                "enabled": False,
                "non_actuating": True,
                "reason": "server_started_without_prod_v13_observer_shadow",
            }
        if refresh:
            self._acquire_operation("prod_v13_observer_refresh")
            try:
                live = self.io.observe()
                self.observations.append(live)
                observer.observe_measurement(
                    frames=live.frames,
                    state=live.state,
                    timestamp_s=live.timestamp_s,
                )
                self.last_prod_v13_shadow_error = None
            except BaseException as exc:
                self.last_prod_v13_shadow_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                raise
            finally:
                self._release_operation()
        status = dict(observer.status())
        status["last_shadow_error"] = self.last_prod_v13_shadow_error
        status["prod_v14_shadow"] = self.prod_v14_shadow_status()
        if self._v165_runtime_adapter is not None:
            status.update(
                {
                    "pulse_entrypoints_available": False,
                    "automatic_initialization_available": False,
                    "identity_probe_available": False,
                    "control_influence": False,
                    "shadow_only": True,
                }
            )
        return status

    def prod_v14_shadow_status(self) -> dict[str, Any]:
        shadow = self.prod_v14_shadow
        if shadow is None:
            return {
                "enabled": False,
                "non_actuating": True,
                "reason": "server_started_without_prod_v14_shadow",
            }
        getter = getattr(shadow, "status", None)
        if not callable(getter):
            return {
                "enabled": True,
                "non_actuating": True,
                "reason": "prod_v14_shadow_has_no_status_interface",
                "last_shadow_error": self.last_prod_v14_shadow_error,
            }
        result = dict(getter())
        result["last_shadow_error"] = self.last_prod_v14_shadow_error
        return result

    def prod_v16_secant_shadow_status(self) -> dict[str, Any]:
        shadow = self.prod_v16_secant_shadow
        if shadow is None:
            return {
                "enabled": False,
                "non_actuating": True,
                "reason": "server_started_without_prod_v16_secant_shadow",
            }
        getter = getattr(shadow, "status", None)
        if not callable(getter):
            return {
                "enabled": True,
                "non_actuating": True,
                "reason": "prod_v16_secant_shadow_has_no_status_interface",
                "last_shadow_error": self.last_prod_v16_secant_shadow_error,
            }
        result = dict(getter())
        result["last_shadow_error"] = self.last_prod_v16_secant_shadow_error
        return result

    def initialize_prod_v13_query(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        observer = self.prod_v13_shadow
        if observer is None:
            raise RuntimeError("Prod-V13 observer shadow is disabled")
        self._acquire_operation("prod_v13_query_init")
        try:
            with self.state_lock:
                if self.stopped:
                    raise RuntimeError("controller stop-lock is latched")
            live = self.io.observe()
            self.observations.append(live)
            result = observer.initialize_query(
                payload,
                frames=live.frames,
                state=live.state,
                timestamp_s=live.timestamp_s,
            )
            self.last_prod_v13_shadow_error = None
            return result
        finally:
            self._release_operation()

    def run_prod_v13_identity_probe(
        self,
        *,
        confirmed: bool,
        empty_gripper_confirmed: bool,
        workspace_clear_confirmed: bool,
    ) -> dict[str, Any]:
        """Run the explicit ten-unit jaw-only identity pulse and restore."""

        self._reject_no_contact_side_motion("ProdV13 identity pulse")

        if self._v165_runtime_adapter is not None:
            raise RuntimeError(
                "ProdV13 identity pulses are disabled in V16.5 shadow-only sessions"
            )
        if not confirmed:
            raise RuntimeError("identity probe requires explicit human confirmation")
        if not empty_gripper_confirmed:
            raise RuntimeError("identity probe requires an explicitly empty gripper")
        if not workspace_clear_confirmed:
            raise RuntimeError("identity probe requires workspace-clear confirmation")
        if not self.execute_enabled:
            raise RuntimeError("command execution is disabled for this server")
        if not self.io.is_hardware:
            raise RuntimeError("identity probe is only available on real hardware")
        observer = self.prod_v13_shadow
        if observer is None or not bool(observer.status().get("initialized")):
            raise RuntimeError("identity probe requires Prod-V13 query initialization")
        if not callable(getattr(observer, "apply_identity_probe", None)):
            raise RuntimeError(
                "this Prod-V13 observer uses unordered A/B query slots and "
                "does not run a fixed/moving identity probe"
            )
        probe = getattr(self.io, "fcpg_identity_probe", None)
        if not callable(probe):
            raise RuntimeError("runtime I/O does not expose the FCPG identity probe")
        self._acquire_operation("prod_v13_identity_probe")
        started = time.perf_counter()
        try:
            with self.state_lock:
                if self.stopped:
                    raise RuntimeError("controller stop-lock is latched")
                if self.plan is None or self.plan.phase != ManipulationPhase.APPROACH:
                    raise PlanRejectedError(
                        "identity probe is allowed only during APPROACH"
                    )
                self.executing = True
            self.stop_event.clear()
            result = probe(stop_event=self.stop_event)
            identity_status = observer.apply_identity_probe(
                moved_frames=result["moved_frames"],
                restored_frames=result["restored_frames"],
                moved_state=result["moved_state"],
                restored_state=result["restored_state"],
                command_record={
                    "direction": result["direction"],
                    "initial_aperture": result["initial_aperture"],
                    "target_aperture": result["target_aperture"],
                    "displacement_units": result["displacement_units"],
                    "outward_commands": result["outward_commands"],
                    "return_commands": result["return_commands"],
                    "maximum_observed_arm_drift": result[
                        "maximum_observed_arm_drift"
                    ],
                },
            )
            restored_observation = result["restored_observation"]
            self.observations.append(restored_observation)
            receipt = {
                "kind": "prod_v13_fcpg_identity_probe",
                "started_monotonic_s": started,
                "duration_s": time.perf_counter() - started,
                "direction": result["direction"],
                "before_state": result["before_state"],
                "moved_state": result["moved_state"],
                "restored_state": result["restored_state"],
                "pinned_arm": result["pinned_arm"],
                "initial_aperture": result["initial_aperture"],
                "target_aperture": result["target_aperture"],
                "displacement_units": result["displacement_units"],
                "maximum_observed_arm_drift": result[
                    "maximum_observed_arm_drift"
                ],
                "settlement_outcome": result["settlement_outcome"],
                "identity": identity_status["identity"],
                "command_journal": result["command_journal"],
                "hardware": True,
                "shadow_only": True,
            }
            self.receipts.append(receipt)
            # A real pulse, even though restored, invalidates the old plan.
            # The operator must request a fresh plan before any safe chunk.
            with self.state_lock:
                self.plan = None
                self.execution_armed = False
                self.executing = False
            self.last_prod_v13_shadow_error = None
            return self.prod_v13_observer_status(refresh=False)
        except PlanRejectedError:
            with self.state_lock:
                self.executing = False
            raise
        except BaseException as exc:
            self.last_prod_v13_shadow_error = f"{type(exc).__name__}: {exc}"
            self._latch_fault(exc)
            raise
        finally:
            self._release_operation()

    def _start_prod_v13_shadow_rollout(
        self,
        plan: RealtimePlan,
        *,
        planned_actions: np.ndarray | None = None,
        current_observation: LiveObservation | None = None,
    ) -> None:
        observer = self.prod_v13_shadow
        if observer is None:
            return
        try:
            if current_observation is not None:
                observer.observe_measurement(
                    frames=current_observation.frames,
                    state=current_observation.state,
                    timestamp_s=current_observation.timestamp_s,
                )
            status = observer.status()
            rollout_ready = status.get("rollout_ready")
            if rollout_ready is None:
                rollout_ready = (
                    status.get("identity", {}).get("status")
                    == "identity_resolved"
                )
            if not bool(rollout_ready):
                return
            if bool(status.get("rollout_active")):
                return
            prefix = planned_actions
            if prefix is None:
                prefix = plan.selected_prefix
            if prefix is None:
                prefix = plan.selected_chunk
            if prefix is None:
                return
            kinematics = getattr(self.planner, "kinematics", None)
            if kinematics is None:
                raise RuntimeError("planner lacks live FK for Prod-V13 shadow")
            observer.begin_rollout(
                planned_actions=np.asarray(prefix, dtype=np.float32),
                past_chunks=tuple(self.past_chunks),
                kinematics=kinematics,
            )
            self.last_prod_v13_shadow_error = None
        except BaseException as exc:
            # Observer is explicitly non-actuating: a shadow failure is logged
            # but cannot alter the already-supervised command path.
            self.last_prod_v13_shadow_error = f"{type(exc).__name__}: {exc}"

    def _advance_prod_v13_shadow(self, result: ChunkExecution) -> None:
        observer = self.prod_v13_shadow
        if observer is None or not result.observations:
            return
        try:
            status = observer.status()
            if not bool(status.get("rollout_active")):
                return
            live = result.observations[-1]
            advance_kwargs: dict[str, Any] = {
                "frames": live.frames,
                "state": live.state,
                "timestamp_s": live.timestamp_s,
                "executed_actions": result.sent,
            }
            if bool(
                getattr(observer, "requires_observation_sequence", False)
            ):
                advance_kwargs["observation_sequence"] = tuple(
                    row.frames for row in result.observations
                )
            observer.advance_after_chunk(
                **advance_kwargs,
            )
            self.last_prod_v13_shadow_error = None
        except BaseException as exc:
            self.last_prod_v13_shadow_error = f"{type(exc).__name__}: {exc}"

    def open_gripper_in_place(self, *, confirmed: bool) -> dict[str, Any]:
        """Open the jaws to the calibrated open stop without moving the arm."""

        self._reject_no_contact_side_motion("in-place gripper motion")

        if not confirmed:
            raise RuntimeError(
                "in-place gripper open requires explicit human confirmation"
            )
        if not self.execute_enabled:
            raise RuntimeError("command execution is disabled for this server")
        if not self.io.is_hardware:
            raise RuntimeError(
                "in-place gripper open is only available on real hardware"
            )
        opener = getattr(self.io, "open_gripper_in_place", None)
        if not callable(opener):
            raise RuntimeError(
                "runtime I/O does not expose in-place gripper open"
            )
        self._acquire_operation("open_gripper_in_place")
        started = time.perf_counter()
        try:
            with self.state_lock:
                if self.stopped:
                    raise RuntimeError("controller is stopped")
                plan = self.plan
                if plan is None or plan.phase != ManipulationPhase.APPROACH:
                    raise PlanRejectedError(
                        "in-place gripper open is only allowed during APPROACH"
                    )
                self.executing = True
            self.stop_event.clear()
            result = opener(stop_event=self.stop_event)
            live = self.io.observe()
            self.observations.append(live)
            receipt = {
                "kind": "open_gripper_in_place",
                "started_monotonic_s": started,
                "duration_s": time.perf_counter() - started,
                "before_state": result["before_state"],
                "after_state": result["after_state"],
                "opened_aperture": result["opened_aperture"],
                "pinned_arm": result["pinned_arm"],
                "commanded_aperture": result["commanded_aperture"],
                "commanded_state": result["commanded_state"],
                "maximum_observed_arm_drift": result[
                    "maximum_observed_arm_drift"
                ],
                "settlement_outcome": result["settlement_outcome"],
                "settlement_joint_residual": result[
                    "settlement_joint_residual"
                ],
                "settlement_actual_state": result["settlement_actual_state"],
                "hardware": True,
            }
            self.receipts.append(receipt)
            self._replan_locked()
            with self.state_lock:
                self.execution_armed = True
                self.executing = False
            return self.status()
        except PlanRejectedError:
            with self.state_lock:
                self.executing = False
            raise
        except BaseException as exc:
            self._latch_fault(exc)
            raise
        finally:
            self._release_operation()

    def _execute_one_locked(self, *, round_id: str) -> dict[str, Any]:
        """Execute and record one five-command token."""

        self._assert_v165_session_binding()
        if self._v165_no_contact_approach:
            with self.state_lock:
                preflight_plan = self.plan
            if (
                preflight_plan is None
                or preflight_plan.phase != ManipulationPhase.APPROACH
            ):
                raise PlanRejectedError(
                    "V16.5 no-contact mode permits only APPROACH; no I/O was attempted"
                )
        can_execute = getattr(self.io, "can_execute_chunk", None)
        if callable(can_execute) and not can_execute():
            raise PlanRejectedError(
                "replay session has no complete next five-frame chunk"
            )
        with self.state_lock:
            plan = self.plan
            if self.stopped:
                raise RuntimeError("controller is stopped")
            if not self.execution_armed:
                raise PlanRejectedError(
                    "next chunk is locked; request a fresh plan before execution"
                )
            if plan is None or not plan.executable or plan.selected_chunk is None:
                raise PlanRejectedError(
                    "current planning round is not executable; request a fresh plan"
                )
            self.executing = True

        selected_row = self._selected_candidate_row(plan)
        selected_source = None if selected_row is None else selected_row.get("source")
        lewm_scored = None if selected_row is None else bool(selected_row.get("lewm_scored"))
        print(
            f"[V11-RANK] execute action_source={selected_source} "
            f"selected={selected_source} lewm_scored={lewm_scored} "
            f"uses_selected_chunk=True",
            flush=True,
        )

        live = self.io.observe()
        if self._v165_no_contact_approach:
            try:
                selected_sweep = conservative_joint_sweeps(
                    live.state,
                    np.asarray(plan.selected_chunk, dtype=np.float32)[None],
                )[0]
                selected_xyz = self.planner.kinematics.forward_anchors(
                    selected_sweep
                )
                enforce_no_contact_selected_chunk(
                    phase=plan.phase,
                    current_state=live.state,
                    selected_chunk=plan.selected_chunk,
                    swept_xyz=selected_xyz,
                    contact_height_m=float(self.planner.contact_height_m),
                )
            except (ValueError, V165RuntimeContractError) as exc:
                raise PlanRejectedError(f"{exc}; no command was sent") from exc
        if plan.phase == ManipulationPhase.TRAJECTORY:
            if self.io.is_hardware:
                raise PlanRejectedError(
                    "TRAJECTORY plans are replay-only demonstration "
                    "continuation and cannot execute on hardware"
                )
            # TRAJECTORY has no live target measurement; skip the target-drift
            # check honestly instead of feeding the plan's own target back in.
            descriptor = None
        else:
            try:
                # Pass the plan-time grounded gripper pixel so the tracker's
                # occlusion hold can engage symmetrically with plan time; a
                # confident plan-time match clears the hold, and without a
                # gripper estimate this re-check could then only fail.
                plan_gripper_px = None
                corrected = getattr(plan, "corrected", None)
                if isinstance(corrected, dict) and "column" in corrected:
                    plan_gripper_px = np.asarray(
                        corrected["column"][-1, 0], dtype=np.float32
                    )
                kind, descriptor = self.planner.detect_live_target(
                    live.frames["column"],
                    plan.phase,
                    update_track=False,
                    gripper_center_px=plan_gripper_px,
                )
            except RuntimeError as exc:
                raise PlanRejectedError(f"{exc}; request a fresh plan") from exc
            if kind != plan.guard.target_kind:
                raise PlanRejectedError(
                    "live target kind changed after planning; request a fresh plan"
                )
        try:
            plan.guard.validate_live(
                round_id=round_id,
                now_monotonic_s=time.monotonic(),
                live_state=live.state,
                live_target_px=(
                    None
                    if descriptor is None
                    else np.asarray(descriptor["centroid_px"], dtype=np.float32)
                ),
                max_age_s=self.max_plan_age_s,
                max_state_drift=self.max_state_drift,
                max_target_drift_px=self.max_target_drift_px,
            )
        except (RuntimeError, ValueError) as exc:
            raise PlanRejectedError(f"{exc}; request a fresh plan") from exc

        context_check: dict[str, Any] = {"configured": False}
        target_descriptors = getattr(plan, "target_descriptors", None)
        planned_context = (
            target_descriptors.get("peg_column")
            if isinstance(target_descriptors, Mapping)
            else None
        )
        if isinstance(planned_context, Mapping) and bool(
            planned_context.get("configured", False)
        ):
            if self.io.is_hardware:
                registration = planned_context.get("registration")
                registered_binding = (
                    registration.get("camera_binding")
                    if isinstance(registration, Mapping)
                    else None
                )
                manifest = getattr(self.io, "camera_manifest", None)
                current_cameras = (
                    manifest.get("cameras")
                    if isinstance(manifest, Mapping)
                    and manifest.get("schema")
                    == "so101-camera-role-manifest-v1"
                    else None
                )
                current_binding = (
                    current_cameras.get("column")
                    if isinstance(current_cameras, Mapping)
                    else None
                )
                if not isinstance(registered_binding, Mapping) or not isinstance(
                    current_binding, Mapping
                ):
                    raise PlanRejectedError(
                        "configured peg camera binding cannot be verified; "
                        "request a fresh plan"
                    )
                for field in CAMERA_BINDING_STABLE_FIELDS:
                    registered_value = registered_binding.get(field)
                    current_value = current_binding.get(field)
                    if (
                        field not in registered_binding
                        or field not in current_binding
                        or not isinstance(registered_value, str)
                        or not registered_value
                        or not isinstance(current_value, str)
                        or not current_value
                        or registered_value != current_value
                    ):
                        raise PlanRejectedError(
                            "configured peg camera binding changed after "
                            f"registration ({field}); register the peg again"
                        )
            detector = getattr(self.planner, "detect_live_context", None)
            clearance = getattr(self.planner, "context_clearance", None)
            if not callable(detector) or not callable(clearance):
                raise PlanRejectedError(
                    "configured peg context cannot be revalidated; request a "
                    "fresh plan"
                )
            try:
                fresh_context = detector(
                    live.frames["column"], update_track=False
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                raise PlanRejectedError(
                    f"live peg context check failed: {exc}; request a fresh plan"
                ) from exc
            if not isinstance(fresh_context, Mapping):
                raise PlanRejectedError(
                    "live peg context descriptor is invalid; request a fresh plan"
                )
            if not fresh_context.get("configured", False):
                raise PlanRejectedError(
                    "configured peg context disappeared; request a fresh plan"
                )
            if not fresh_context.get("visible", False):
                raise PlanRejectedError(
                    "configured peg context is not freshly visible; request a "
                    "fresh plan"
                )
            try:
                planned_context_px = np.asarray(
                    planned_context.get("centroid_px"), dtype=np.float32
                )
                fresh_context_px = np.asarray(
                    fresh_context.get("centroid_px"), dtype=np.float32
                )
            except (TypeError, ValueError) as exc:
                raise PlanRejectedError(
                    "peg context pixels are invalid; request a fresh plan"
                ) from exc
            if (
                planned_context_px.shape != (2,)
                or fresh_context_px.shape != (2,)
                or not np.all(np.isfinite(planned_context_px))
                or not np.all(np.isfinite(fresh_context_px))
            ):
                raise PlanRejectedError(
                    "peg context pixels are invalid; request a fresh plan"
                )
            context_drift_px = float(
                np.linalg.norm(fresh_context_px - planned_context_px)
            )
            if context_drift_px > self.max_target_drift_px:
                raise PlanRejectedError(
                    f"live peg context drift {context_drift_px:.3f}px exceeds "
                    f"{self.max_target_drift_px:.3f}px; request a fresh plan"
                )
            try:
                fresh_base_xyz = np.asarray(
                    fresh_context.get("base_xyz"), dtype=np.float32
                )
            except (TypeError, ValueError) as exc:
                raise PlanRejectedError(
                    "live peg context backprojection is invalid; request a "
                    "fresh plan"
                ) from exc
            if fresh_base_xyz.shape != (3,) or not np.all(
                np.isfinite(fresh_base_xyz)
            ):
                raise PlanRejectedError(
                    "live peg context backprojection is invalid; request a "
                    "fresh plan"
                )
            try:
                planned_clearance_m = float(planned_context.get("clearance_m"))
                fresh_clearance_m = float(fresh_context.get("clearance_m"))
            except (TypeError, ValueError) as exc:
                raise PlanRejectedError(
                    "peg clearance contract is invalid; request a fresh plan"
                ) from exc
            if (
                not np.isfinite(planned_clearance_m)
                or planned_clearance_m <= 0.0
                or not np.isfinite(fresh_clearance_m)
                or fresh_clearance_m <= 0.0
            ):
                raise PlanRejectedError(
                    "peg clearance contract is invalid; request a fresh plan"
                )
            if not np.isclose(
                planned_clearance_m,
                fresh_clearance_m,
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise PlanRejectedError(
                    "peg clearance contract changed after planning; request a "
                    "fresh plan"
                )
            required_clearance_m = planned_clearance_m
            try:
                clearance_result = clearance(
                    live.state,
                    np.asarray(plan.selected_chunk, dtype=np.float32)[None],
                    fresh_base_xyz[:2],
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                raise PlanRejectedError(
                    f"live peg clearance check failed: {exc}; request a fresh plan"
                ) from exc
            if not isinstance(clearance_result, Mapping):
                raise PlanRejectedError(
                    "live peg clearance result is invalid; request a fresh plan"
                )
            try:
                clearance_safe = np.asarray(clearance_result.get("safe"))
                minimum_distance_m = np.asarray(
                    clearance_result.get("minimum_distance_m"), dtype=np.float32
                )
                per_anchor_distance_m = np.asarray(
                    clearance_result.get("per_anchor_minimum_distance_m"),
                    dtype=np.float32,
                )
            except (TypeError, ValueError) as exc:
                raise PlanRejectedError(
                    "live peg clearance result is invalid; request a fresh plan"
                ) from exc
            if (
                clearance_safe.shape != (1,)
                or clearance_safe.dtype != np.bool_
                or minimum_distance_m.shape != (1,)
                or per_anchor_distance_m.shape != (1, 3)
                or not np.all(np.isfinite(minimum_distance_m))
                or not np.all(np.isfinite(per_anchor_distance_m))
            ):
                raise PlanRejectedError(
                    "live peg clearance result has invalid shapes; request a "
                    "fresh plan"
                )
            if (
                not bool(clearance_safe[0])
                or float(minimum_distance_m[0]) < required_clearance_m
            ):
                raise PlanRejectedError(
                    "live peg clearance failed: "
                    f"{float(minimum_distance_m[0]):.4f}m is below "
                    f"{required_clearance_m:.4f}m; request a fresh plan"
                )
            context_check = {
                "configured": True,
                "fresh_descriptor": dict(fresh_context),
                "pixel_drift_px": context_drift_px,
                "minimum_distance_m": float(minimum_distance_m[0]),
                "per_anchor_minimum_distance_m": per_anchor_distance_m[0],
                "required_clearance_m": required_clearance_m,
                "safe": True,
            }
        check = validate_command_chunk(
            plan.selected_chunk,
            live.state,
            max_first_jump=self.max_command_jump,
            max_step_jump=self.max_command_jump,
            # One STS3215 encoder count (360/4096 ~ 0.088): absorbs read-back
            # quantization between the plan-time state sample and this fresh
            # reading; planner-side validation stays strict at 0.0.
            outward_tolerance=0.1,
        )
        if not check.safe:
            raise PlanRejectedError(
                "live command safety failed: "
                + ", ".join(check.reasons)
                + "; request a fresh plan"
            )

        self._start_prod_v13_shadow_rollout(
            plan,
            current_observation=live,
        )
        started = time.perf_counter()
        if self._v165_no_contact_approach:
            # All planner/fresh-observation gates have passed. From the moment
            # command I/O is attempted, this one-shot physical screen is
            # consumed even if the transport later faults and motion is
            # uncertain. Rejections above this line remain retryable.
            self._v165_no_contact_consumed = True
        result = self.io.execute_chunk(
            plan.selected_chunk,
            period_s=self.period_s,
            stop_event=self.stop_event,
        )
        if not np.allclose(result.sent, result.requested, atol=1e-4, rtol=0.0):
            raise RuntimeError("LeRobot clipped or changed a command")
        # Arm joints only: the gripper legitimately stalls against a grasped
        # object without reaching its command, and that stall must not read
        # as "robot did not follow".
        expected_motion = float(
            np.max(np.abs(result.requested[-1][:5] - live.state[:5]))
        )
        observed_motion = float(
            np.max(np.abs(result.observations[-1].state[:5] - live.state[:5]))
        )
        if (
            self.io.is_hardware
            and expected_motion >= self.minimum_expected_motion
            and observed_motion < self.minimum_observed_motion
        ):
            raise RuntimeError(
                "robot did not follow the five-command chunk: "
                f"expected={expected_motion:.3f}, observed={observed_motion:.3f}"
            )

        # Only completed commands advance model time. Every next token is
        # planned from the endpoint observations returned by this execution.
        self.observations.extend(result.observations)
        self.past_chunks.append(result.sent.copy())
        v168_commit = self._commit_v168_completed_chunk(
            sent_chunk=result.sent,
            round_id=plan.guard.round_id,
            selected_source=selected_source,
        )
        v170_commit: dict[str, Any] | None = None
        if (
            self._v170_runtime_adapter is not None
            and str(getattr(self._v170_runtime_adapter, "mode", "")) == "candidate"
        ):
            try:
                measured_path = np.concatenate(
                    (
                        np.asarray(live.state, dtype=np.float32)[None],
                        np.stack([row.state for row in result.observations]).astype(
                            np.float32
                        ),
                    ),
                    axis=0,
                )
                v170_commit = self._commit_v170_completed_chunk(
                    requested_chunk=result.requested,
                    sent_chunk=result.sent,
                    measured_joint_path=measured_path,
                    round_id=plan.guard.round_id,
                )
            except Exception as exc:
                # Transport already happened. Never retry it. Freeze further
                # execution because V17's action/query clocks are now missing
                # a physical transition.
                setattr(
                    self.planner,
                    "_last_v170_shadow_error",
                    f"post_execution_{type(exc).__name__}: {exc}",
                )
                with self.state_lock:
                    self.stopped = True
                    self.stop_event.set()
        self._advance_prod_v13_shadow(result)
        secant_commit: dict[str, Any] | None = None
        secant_shadow = self.prod_v16_secant_shadow
        if secant_shadow is not None:
            try:
                secant_commit = secant_shadow.commit_execution(
                    plan=plan,
                    sent_chunk=result.sent,
                )
                self.last_prod_v16_secant_shadow_error = None
            except Exception as exc:
                # This observer runs after a physical chunk completed. A
                # bookkeeping failure cannot rewrite that fact as an execution
                # failure or trigger a second command.
                self.last_prod_v16_secant_shadow_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                secant_commit = {
                    "committed": False,
                    "reason": "shadow_error_after_completed_action",
                    "error": self.last_prod_v16_secant_shadow_error,
                }
        progress: dict[str, Any] | None = None
        commit_progress = getattr(self.planner, "commit_executed_plan", None)
        if callable(commit_progress):
            progress = commit_progress(plan, command_count=len(result.sent))
        receipt = {
            "kind": "execute_chunk",
            "round_id": plan.guard.round_id,
            "phase": plan.phase.value,
            "target_kind": plan.guard.target_kind,
            "started_monotonic_s": started,
            "duration_s": result.duration_s,
            "requested": result.requested,
            "sent": result.sent,
            "before_timestamp_s": live.timestamp_s,
            "before_state": live.state,
            "observed_timestamps_s": np.asarray(
                [row.timestamp_s for row in result.observations],
                dtype=np.float64,
            ),
            "observed_states": np.stack(
                [row.state for row in result.observations]
            ).astype(np.float32),
            "after_state": result.observations[-1].state,
            "expected_motion": expected_motion,
            "observed_motion": observed_motion,
            "task_progress": progress,
            "hardware": bool(self.io.is_hardware),
            "target_check": (
                "skipped_no_live_target_in_trajectory"
                if descriptor is None
                else "live_detected"
            ),
            "context_check": context_check,
            "command_journal": result.command_journal,
            "v170_commit": v170_commit,
            # The commanded joint endpoint and the pose the arm actually holds
            # are reported separately.  `after_state` above is always the
            # measured one, so a miss recorded here never claims the commanded
            # endpoint was reached.
            "endpoint_check": _endpoint_check(result),
            "action_source": selected_source
            or self._determine_action_source(
                getattr(getattr(self.planner, "config", None), "control_mode", None)
                == "phase_target"
            ),
            "selected_candidate_source": selected_source,
            "lewm_scored": lewm_scored,
            "prod_v16_secant_commit": secant_commit,
            "anchor_authority": (
                None
                if self._v165_runtime_adapter is None
                else self._v165_runtime_adapter.anchor_authority
            ),
            "v165_runtime_bundle": (
                None
                if self._v165_runtime_adapter is None
                else self._v165_runtime_adapter.receipt
            ),
            "v165_no_contact_approach": self._v165_no_contact_approach,
            "v165_contact_hardware_authorized": self._v165_contact_hardware_authorized,
            "v165_no_contact_consumed": bool(
                getattr(self, "_v165_no_contact_consumed", False)
            ),
            "v168_completed_chunk_commit": v168_commit,
        }
        self.receipts.append(receipt)
        execution_writer = getattr(self.artifact_writer, "write_execution", None)
        if callable(execution_writer):
            artifacts = execution_writer(
                plan=plan,
                receipt=receipt,
                fresh_observation=live,
                observations=result.observations,
            )
            if not isinstance(artifacts, Mapping):
                raise TypeError("execution artifact writer must return a mapping")
            receipt["artifacts"] = dict(artifacts)
        return receipt

    def _execute_one_v11_locked(self, *, round_id: str) -> dict[str, Any]:
        """Legacy V11-only execute path. Execute no longer calls this.

        Ranked V11 prefixes are sent through ``plan.selected_chunk`` so PRED
        and motors share the same LeWM-ranked candidate.
        """
        if self.v11_controller is None:
            raise RuntimeError("V11 controller not configured")

        print("[V11] observe", flush=True)
        live = self.io.observe()
        print(f"[V11] predict n_obs={len(self.observations)}", flush=True)
        if self.task_class is None:
            raise RuntimeError("V11 control requires a human-set task_class")
        if self.plan is None:
            raise RuntimeError("V11 control requires a live plan")
        live_goal = self._resolve_v11_live_goal(
            self.plan,
            self.task_class,
            live.state,
        )
        goal_raw = np.asarray(live_goal["goal_state"], dtype=np.float32)
        v11_result = self.v11_controller.predict_chunk(
            self.observations,
            live.state,
            goal_state_raw=goal_raw,
            task_class=self.task_class,
        )
        diag = {
            **(v11_result.get("diagnostics") or {}),
            "live_goal_source": live_goal["source"],
            "live_goal_state": goal_raw.tolist(),
            "live_goal_l2_arm_deg": live_goal["l2_arm_deg"],
            "live_goal_lift_deg": live_goal["goal_lift_deg"],
            "fresh_ik_source": live_goal.get("fresh_ik_source"),
            "fk_to_fish_mm": live_goal.get("fk_to_fish_mm"),
            "fish_xyz": live_goal.get("fish_xyz"),
        }
        v11_result["diagnostics"] = diag
        print(
            f"[V11] predict_done executable={v11_result.get('executable')} "
            f"reason={v11_result.get('reason')} "
            f"jump_norm={diag.get('max_arm_joint_jump')} "
            f"jump_deg={diag.get('max_arm_joint_jump_deg')} "
            f"grip_deg={diag.get('max_gripper_jump_deg')} "
            f"goal={live_goal['source']} "
            f"goal_lift={live_goal['goal_lift_deg']:.2f} "
            f"l2_arm={live_goal['l2_arm_deg']:.2f}",
            flush=True,
        )

        if not v11_result.get("executable", False):
            raise PlanRejectedError(
                f"V11 prediction not executable: {v11_result.get('reason', 'unknown')}"
            )

        chunk = v11_result["selected_chunk"]
        if chunk is None or chunk.shape != (5, 6):
            raise RuntimeError("V11 returned invalid chunk shape")

        from fcpg.so101_v11_controller import rate_limit_command_chunk

        raw_first = float(np.max(np.abs(chunk[0] - live.state)))
        chunk = rate_limit_command_chunk(chunk, live.state, self.max_command_jump)
        limited_first = float(np.max(np.abs(chunk[0] - live.state)))
        if limited_first + 1e-6 < raw_first:
            print(
                f"[V11] rate_limited first_jump {raw_first:.3f} -> {limited_first:.3f}",
                flush=True,
            )

        # Supervisor validation: same safety checks as MPC path
        check = validate_command_chunk(
            chunk,
            live.state,
            max_first_jump=self.max_command_jump,
            max_step_jump=self.max_command_jump,
            outward_tolerance=0.1,
        )
        if not check.safe:
            raise PlanRejectedError(
                "V11 chunk failed supervisor safety: "
                + ", ".join(check.reasons)
            )

        print("[V11] supervisor_ok; sending 5 commands", flush=True)
        self._start_prod_v13_shadow_rollout(
            self.plan,
            planned_actions=chunk,
            current_observation=live,
        )
        started = time.perf_counter()
        result = self.io.execute_chunk(
            chunk,
            period_s=self.period_s,
            stop_event=self.stop_event,
        )
        if not np.allclose(result.sent, result.requested, atol=1e-4, rtol=0.0):
            raise RuntimeError("LeRobot clipped or changed a V11 command")

        # Motion validation (same as MPC path)
        expected_motion = float(
            np.max(np.abs(result.requested[-1][:5] - live.state[:5]))
        )
        observed_motion = float(
            np.max(np.abs(result.observations[-1].state[:5] - live.state[:5]))
        )
        if (
            self.io.is_hardware
            and expected_motion >= self.minimum_expected_motion
            and observed_motion < self.minimum_observed_motion
        ):
            raise RuntimeError(
                "robot did not follow V11 chunk: "
                f"expected={expected_motion:.3f}, observed={observed_motion:.3f}"
            )

        # Advance model time
        self.observations.extend(result.observations)
        self.past_chunks.append(result.sent.copy())
        self._advance_prod_v13_shadow(result)

        # Build receipt
        receipt = {
            "kind": "execute_chunk",
            "round_id": round_id,
            "phase": "v11_control",
            "target_kind": "v11_dual_view_policy",
            "action_source": "v11_dual_view_policy",
            "started_monotonic_s": started,
            "duration_s": result.duration_s,
            "requested": result.requested,
            "sent": result.sent,
            "before_timestamp_s": live.timestamp_s,
            "before_state": live.state,
            "observed_timestamps_s": np.asarray(
                [row.timestamp_s for row in result.observations],
                dtype=np.float64,
            ),
            "observed_states": np.stack(
                [row.state for row in result.observations]
            ).astype(np.float32),
            "after_state": result.observations[-1].state,
            "expected_motion": expected_motion,
            "observed_motion": observed_motion,
            "hardware": bool(self.io.is_hardware),
            "v11_diagnostics": v11_result.get("diagnostics", {}),
            "command_journal": result.command_journal,
        }
        self.receipts.append(receipt)
        return receipt

    @staticmethod
    def _terminal_plan_stop_reason(plan: RealtimePlan | None) -> str | None:
        if plan is None:
            return None
        phase_status = getattr(plan, "phase_status", {})
        tracker_phase = (
            phase_status.get("phase")
            if isinstance(phase_status, Mapping)
            else None
        )
        phase = getattr(plan, "phase", None)
        phase_value = phase.value if isinstance(phase, ManipulationPhase) else str(phase)
        effective_phase = tracker_phase or phase_value
        if effective_phase == ManipulationPhase.DONE.value:
            return "task_done"
        if effective_phase == ManipulationPhase.FAULT.value:
            return "task_fault"
        return None

    def execute(
        self,
        *,
        round_id: str,
        confirmed: bool,
        max_chunks: int = 1,
        task_class: object = None,
    ) -> dict[str, Any]:
        """Run a bounded sequence of fresh-plan, supervised command tokens."""

        if self.task_class is None:
            if task_class is not None:
                raise ValueError("this server was not started with --task-class")
        else:
            from v10.task_class import require_matching_task_class

            require_matching_task_class(task_class, self.task_class)
        if not confirmed:
            raise RuntimeError("execution requires explicit human confirmation")
        if not self.execute_enabled:
            raise RuntimeError("command execution is disabled for this server")
        # This must precede operation acquisition, auto-initialization and
        # re-anchor pulses.  A programmatic permission flip may never reach an
        # actuator before the session-frozen V16.5 contract is rechecked.
        self._assert_v165_session_binding()
        if self._v165_no_contact_approach and bool(
            getattr(self, "_v165_no_contact_consumed", False)
        ):
            raise RuntimeError(
                "V16.5 no-contact approach was already consumed for this session"
            )
        if isinstance(max_chunks, bool) or not isinstance(max_chunks, int):
            raise ValueError("max_chunks must be an integer")
        if not 1 <= max_chunks <= self.max_autonomous_chunks:
            raise ValueError(
                f"max_chunks must be in [1,{self.max_autonomous_chunks}]"
            )
        if self._v165_no_contact_approach and max_chunks != 1:
            raise ValueError("V16.5 no-contact approach permits exactly one chunk")
        if max_chunks > 1 and not self.autonomous_run_enabled:
            raise ValueError(
                "max_chunks > 1 requires bounded autonomy to be enabled"
            )

        operation = "execute_chunk"
        self._acquire_operation(operation)
        batch_started = time.perf_counter()
        command_journal_before = getattr(self.io, "last_command_journal", None)
        completed: list[dict[str, Any]] = []
        stop_reason = "command_budget_exhausted"
        try:
            with self.state_lock:
                if self.stopped:
                    raise RuntimeError("controller is stopped")
                self.stop_event.clear()
            current_round_id = round_id
            if self._prod_v13_automatic_initialization_required():
                with self.state_lock:
                    plan = self.plan
                    if plan is None or plan.phase != ManipulationPhase.APPROACH:
                        raise PlanRejectedError(
                            "automatic FCPG-H initialization requires APPROACH"
                        )
                # The execution authorization covers this gripper-only pulse.
                # No pixel query or second UI interaction is needed.
                self._apply_prod_v13_auto_initialization_locked()
                with self.state_lock:
                    plan = self.plan
                    if (
                        plan is None
                        or not plan.executable
                        or plan.selected_chunk is None
                    ):
                        raise PlanRejectedError(
                            "plan is not executable after automatic FCPG-H initialization"
                        )
                    current_round_id = plan.guard.round_id
            for index in range(max_chunks):
                if self.stop_event.is_set():
                    stop_reason = "operator_stop"
                    break
                if (
                    index > 0
                    and time.perf_counter() - batch_started
                    >= self.max_autonomous_duration_s
                ):
                    stop_reason = "wall_time_budget_exhausted"
                    break
                if index:
                    can_execute = getattr(self.io, "can_execute_chunk", None)
                    if callable(can_execute) and not can_execute():
                        stop_reason = "input_exhausted"
                        break
                    with self.state_lock:
                        plan = self.plan
                        if (
                            plan is None
                            or not plan.executable
                            or plan.selected_chunk is None
                        ):
                            stop_reason = "next_plan_not_executable"
                            break
                        current_round_id = plan.guard.round_id
                with self.state_lock:
                    terminal_reason = self._terminal_plan_stop_reason(self.plan)
                if terminal_reason is not None:
                    stop_reason = terminal_reason
                    break
                with self.state_lock:
                    reanchor_required = self._column_reanchor_required(self.plan)
                if reanchor_required:
                    if self._v165_no_contact_approach:
                        raise PlanRejectedError(
                            "V16.5 no-contact approach forbids calibration/reanchor pulses"
                        )
                    # The last pulse no longer describes this pose. Remake it
                    # here rather than executing on the raw FK projection: the
                    # pulse is jaws-only, and APPROACH is the only phase that
                    # reaches this branch.
                    self._apply_column_pulse_locked()
                    with self.state_lock:
                        plan = self.plan
                        if self._column_reanchor_required(plan):
                            stop_reason = "column_live_anchor_reanchor_failed"
                            break
                        if (
                            plan is None
                            or not plan.executable
                            or plan.selected_chunk is None
                        ):
                            stop_reason = "next_plan_not_executable"
                            break
                        current_round_id = plan.guard.round_id
                with self.state_lock:
                    registration_required = self._wrist_registration_required(
                        self.plan
                    )
                if registration_required:
                    # The distance gate says an observation is now FEASIBLE.
                    # Whether one is NECESSARY is a grounding question, and the
                    # bridge answers it.  Without a bridge there is no
                    # autonomous path, so the previous operator wait stands.
                    if self.grounding_bridge is None:
                        stop_reason = "wrist_registration_required"
                        break
                    grounding_reason = self._run_grounding_cycle_locked()
                    if grounding_reason is not None:
                        stop_reason = grounding_reason
                        break
                try:
                    receipt = self._execute_one_locked(round_id=current_round_id)
                except PlanRejectedError as exc:
                    if max_chunks <= 1 or "live target drift" not in str(exc):
                        raise
                    self._replan_locked(write_artifacts=True)
                    with self.state_lock:
                        plan = self.plan
                        if (
                            plan is None
                            or not plan.executable
                            or plan.selected_chunk is None
                        ):
                            raise
                        current_round_id = plan.guard.round_id
                    receipt = self._execute_one_locked(round_id=current_round_id)
                completed.append(receipt)
                # Every autonomous continuation needs its exact fresh plan on
                # disk before it can become the next command intent.
                self._replan_locked(write_artifacts=True)
                with self.state_lock:
                    terminal_reason = self._terminal_plan_stop_reason(self.plan)
                if terminal_reason is not None:
                    stop_reason = terminal_reason
                    break

            batch = {
                "requested_chunks": max_chunks,
                "requested_commands": max_chunks * 5,
                "executed_chunks": len(completed),
                "executed_commands": len(completed) * 5,
                "stop_reason": stop_reason,
                "round_ids": [row["round_id"] for row in completed],
                "phases": [row["phase"] for row in completed],
                "duration_s": time.perf_counter() - batch_started,
            }
            with self.state_lock:
                self.last_execution_batch = batch
                self.execution_armed = False
                self.executing = False
            return self.status()
        except PlanRejectedError as exc:
            with self.state_lock:
                rejected_plan = self.plan
            rejection = {
                "kind": "execution_attempt",
                "disposition": "rejected_before_motion",
                "requested_round_id": round_id,
                "active_round_id": (
                    None
                    if rejected_plan is None
                    else rejected_plan.guard.round_id
                ),
                "recorded_monotonic_s": time.monotonic(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "hardware": bool(self.io.is_hardware),
                "requested_chunks": max_chunks,
                "requested_commands": max_chunks * 5,
                "completed_chunks": len(completed),
                "sent_commands": len(completed) * 5,
                "motion_sent": bool(completed),
            }
            rejection_artifacts = self._persist_execution_attempt(
                rejected_plan,
                rejection,
            )
            with self.state_lock:
                self.execution_armed = False
                self.executing = False
                self.last_execution_batch = {
                    "requested_chunks": max_chunks,
                    "requested_commands": max_chunks * 5,
                    "executed_chunks": len(completed),
                    "executed_commands": len(completed) * 5,
                    "stop_reason": "plan_rejected",
                    "round_ids": [row["round_id"] for row in completed],
                    "phases": [row["phase"] for row in completed],
                    "duration_s": time.perf_counter() - batch_started,
                    "attempt_record": rejection_artifacts,
                }
            raise
        except BaseException as exc:
            with self.state_lock:
                fault_plan = self.plan
                operator_stopped = (
                    self.stop_event.is_set()
                    and self.stopped
                    and self.error == "operator stop latched"
                )
            latest_command_journal = getattr(
                self.io, "last_command_journal", None
            )
            command_journal = (
                latest_command_journal
                if latest_command_journal != command_journal_before
                else None
            )
            partial_command_count_unknown = command_journal is not None
            fault = {
                "kind": "execution_attempt",
                "disposition": (
                    "operator_stop" if operator_stopped else "runtime_fault"
                ),
                "requested_round_id": round_id,
                "active_round_id": (
                    None if fault_plan is None else fault_plan.guard.round_id
                ),
                "recorded_monotonic_s": time.monotonic(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "hardware": bool(self.io.is_hardware),
                "requested_chunks": max_chunks,
                "requested_commands": max_chunks * 5,
                "completed_chunks": len(completed),
                "verified_completed_commands": len(completed) * 5,
                "partial_command_count_unknown": partial_command_count_unknown,
                "motion_sent": (
                    None
                    if partial_command_count_unknown
                    else bool(completed)
                ),
                "command_journal": command_journal,
            }
            fault_artifacts = self._persist_execution_attempt(fault_plan, fault)
            with self.state_lock:
                self.last_execution_batch = {
                    "requested_chunks": max_chunks,
                    "requested_commands": max_chunks * 5,
                    "executed_chunks": len(completed),
                    "executed_commands": len(completed) * 5,
                    "stop_reason": (
                        "operator_stop" if operator_stopped else "runtime_fault"
                    ),
                    "round_ids": [row["round_id"] for row in completed],
                    "phases": [row["phase"] for row in completed],
                    "duration_s": time.perf_counter() - batch_started,
                    "attempt_record": fault_artifacts,
                }
            if not operator_stopped:
                self._latch_fault(exc)
            raise
        finally:
            self._release_operation()

    def _persist_execution_attempt(
        self,
        plan: RealtimePlan | None,
        record: Mapping[str, Any],
    ) -> dict[str, str] | None:
        """Durably record rejected/faulted confirmations without claiming motion."""

        writer = getattr(self.artifact_writer, "write_execution_attempt", None)
        if not callable(writer):
            return None
        if plan is None:
            raise RuntimeError(
                "cannot persist an execution attempt without an active plan"
            )
        artifacts = writer(plan=plan, record=dict(record))
        if not isinstance(artifacts, Mapping):
            raise TypeError("execution-attempt writer must return a mapping")
        return {str(key): str(value) for key, value in artifacts.items()}

    def status(self) -> dict[str, Any]:
        with self.state_lock:
            plan = self.plan
            can_execute = getattr(self.io, "can_execute_chunk", None)
            io_ready = not callable(can_execute) or bool(can_execute())
            planner_config = getattr(self.planner, "config", None)
            control_mode = getattr(planner_config, "control_mode", None)
            goal_conditioned = control_mode == "phase_target"
            selected_row = self._selected_candidate_row(plan)
            payload = {
                "mode": "hardware" if self.io.is_hardware else "replay",
                "execute_enabled": self.execute_enabled and io_ready,
                "camera_manifest": getattr(self.io, "camera_manifest", None),
                "control_mode": control_mode,
                "goal_conditioned": goal_conditioned,
                "goal_contract": (
                    "operator_click_pickup_drop_zone_optional_peg_clearance"
                    if goal_conditioned
                    else "replay_demonstration_trajectory"
                ),
                "action_source": self._determine_action_source(goal_conditioned),
                "selected_action_source": (
                    None if selected_row is None else selected_row.get("source")
                ),
                "anchor_authority": (
                    "v17_0_query_transformer_fcpg"
                    if (
                        self._v170_runtime_adapter is not None
                        and str(getattr(self._v170_runtime_adapter, "mode", ""))
                        == "candidate"
                    )
                    else (
                        None
                        if self._v165_runtime_adapter is None
                        else self._v165_runtime_adapter.anchor_authority
                    )
                ),
                "v165_runtime_bundle": (
                    None
                    if self._v165_runtime_adapter is None
                    else self._v165_runtime_adapter.receipt
                ),
                "v168_runtime": (
                    None
                    if self._v168_runtime_adapter is None
                    else self._v168_runtime_adapter.status()
                ),
                "v170_runtime": (
                    self.planner.v170_shadow_status()
                    if callable(getattr(self.planner, "v170_shadow_status", None))
                    else None
                ),
                "v170_shadow": (
                    self.planner.v170_shadow_status()
                    if (
                        self._v170_runtime_adapter is not None
                        and str(getattr(self._v170_runtime_adapter, "mode", ""))
                        == "shadow"
                        and callable(getattr(self.planner, "v170_shadow_status", None))
                    )
                    else None
                ),
                "v168_supervised_debug": bool(
                    getattr(self, "_v168_supervised_debug", False)
                ),
                "v165_no_contact_approach": self._v165_no_contact_approach,
                "v165_contact_hardware_authorized": self._v165_contact_hardware_authorized,
                "v165_no_contact_consumed": bool(
                    getattr(self, "_v165_no_contact_consumed", False)
                ),
                "v11_in_candidates": bool(
                    plan is not None
                    and any(
                        str(row.get("source", "")).startswith("v11_")
                        for row in (getattr(plan, "candidate_rows", None) or [])
                    )
                ),
                "task_class": self.task_class,
                "task_class_name": (
                    None
                    if self.task_class_record is None
                    else self.task_class_record["task_class_name"]
                ),
                "task_class_title": (
                    None
                    if self.task_class_record is None
                    else self.task_class_record["task_class_title"]
                ),
                "calibration_enabled": bool(
                    self.calibration_enabled and not self._v165_no_contact_approach
                ),
                "home_available": bool(
                    not self._v165_no_contact_approach
                    and self.io.is_hardware
                    and self.home_state is not None
                ),
                "input_exhausted": not io_ready,
                "stopped": self.stopped,
                "executing": self.executing,
                "execution_armed": self.execution_armed,
                "bounded_autonomy_enabled": self.autonomous_run_enabled,
                "max_autonomous_chunks": self.max_autonomous_chunks,
                "max_autonomous_duration_s": self.max_autonomous_duration_s,
                "wrist_registration_required": (
                    self._wrist_registration_required(plan)
                ),
                "wrist_registration_distance_m": (
                    self.wrist_registration_distance_m
                ),
                "column_reanchor_required": (
                    False
                    if self._v165_no_contact_approach
                    else self._column_reanchor_required(plan)
                ),
                # Present only when a bridge is attached; absent means the
                # deployment is still on the operator-registration path.
                "grounding_feedback_enabled": self.grounding_bridge is not None,
                "last_grounding_feedback": self.last_grounding_feedback,
                "active_operation": self.active_operation,
                "error": self.error,
                "calibration_available": bool(
                    not self._v165_no_contact_approach
                    and self.io.is_hardware
                    and self.calibration_enabled
                    and not self.stopped
                    and self.active_operation is None
                    and plan is not None
                    and plan.phase in {
                        ManipulationPhase.APPROACH,
                        ManipulationPhase.TRAJECTORY,
                    }
                    and not any(
                        receipt.get("kind", "execute_chunk") == "execute_chunk"
                        for receipt in self.receipts
                    )
                ),
                "reanchor_available": bool(
                    not self._v165_no_contact_approach
                    and self.io.is_hardware
                    and self.calibration_enabled
                    and not self.stopped
                    and self.active_operation is None
                    and plan is not None
                    and plan.phase == ManipulationPhase.APPROACH
                ),
                "excitation_available": bool(
                    not self._v165_no_contact_approach
                    and self.io.is_hardware
                    and self.calibration_enabled
                    and not self.stopped
                    and self.active_operation is None
                    and plan is not None
                    and plan.phase == ManipulationPhase.APPROACH
                ),
                "calibration": self.calibration,
                "v165_side_motion_entrypoints_available": bool(
                    not self._v165_no_contact_approach
                ),
                "legacy_observer_sigma_controls_close": False
                if self._v165_runtime_adapter is not None
                else None,
                "observation_clock": "task_event",
                "monitor_frames_feed_model": False,
                "model_history_frames": len(self.observations),
                "model_executed_chunks": len(self.past_chunks),
                "model_memory_tokens": (
                    0
                    if (memory := self._causal_memory()) is None
                    else len(memory.offsets_raw)
                ),
                "assets": dict(self.assets),
                "completed_chunks": sum(
                    receipt.get("kind", "execute_chunk") == "execute_chunk"
                    for receipt in self.receipts
                ),
                "latest_receipt": self.receipts[-1] if self.receipts else None,
                "last_execution_batch": self.last_execution_batch,
                "plan": None if plan is None else plan_json(plan),
            }
        payload["v11_live_goal"] = self._v11_live_goal_status()
        payload["prod_v14_shadow_status"] = self.prod_v14_shadow_status()
        payload["prod_v16_secant_shadow_status"] = (
            self.prod_v16_secant_shadow_status()
        )
        return _jsonable(payload)

    def read_servo_debug(self) -> dict[str, Any]:
        """Read servo registers for debugging (non-actuating, read-only)."""
        if not self.io.is_hardware:
            return {"error": "servo_debug only available in hardware mode"}

        try:
            from fcpg.so101_kinematics import SO101_JOINT_NAMES

            # Read only from hardware IO
            hardware_io = self.io
            bus = hardware_io.follower.bus

            # Focus on arm joints (j0-j4), especially j1 and j2
            arm_joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

            result = {}
            for joint_name in arm_joints:
                joint_data = {}

                # Always available registers
                try:
                    joint_data["Torque_Enable"] = bus.read("Torque_Enable", joint_name)
                except Exception as e:
                    joint_data["Torque_Enable"] = f"error: {e}"

                try:
                    joint_data["Goal_Position"] = bus.read("Goal_Position", joint_name)
                except Exception as e:
                    joint_data["Goal_Position"] = f"error: {e}"

                try:
                    joint_data["Present_Position"] = bus.read("Present_Position", joint_name)
                except Exception as e:
                    joint_data["Present_Position"] = f"error: {e}"

                # Optional registers (may not be supported by all motor models)
                optional_registers = [
                    "Present_Velocity",
                    "Present_Load",
                    "Present_Current",
                    "Moving",
                    "Moving_Status",
                    "Hardware_Error_Status",
                    "Profile_Velocity",
                    "Profile_Acceleration",
                    "Position_P_Gain",
                    "Position_I_Gain",
                    "Position_D_Gain",
                    "Moving_Threshold",
                    "Velocity_Limit",
                    "Current_Limit",
                    "Operating_Mode",
                ]

                for reg in optional_registers:
                    try:
                        joint_data[reg] = bus.read(reg, joint_name)
                    except Exception as e:
                        # Log which registers are not supported
                        joint_data[f"{reg}_error"] = str(e)

                result[joint_name] = joint_data

            return {"joints": result}

        except Exception as e:
            return {"error": f"Failed to read servo registers: {e}"}

    def close(self) -> None:
        self.stop_event.set()
        self.io.close()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "ChunkExecution",
    "LiveObservation",
    "PlanRejectedError",
    "RealtimeMPCController",
    "RealtimeObservationIO",
    "ReplaySO101IO",
]
