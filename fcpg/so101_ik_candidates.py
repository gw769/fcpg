"""Bounded Cartesian IK candidate generation for SO-101 shadow planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fcpg.so101_kinematics import SO101MuJoCoKinematics


@dataclass(frozen=True)
class IKCandidateBatch:
    names: tuple[str, ...]
    raw_prefixes: np.ndarray
    horizon_states: np.ndarray
    horizon_xyz: np.ndarray
    target_surface_xyz: np.ndarray
    target_approach_xyz: np.ndarray
    terminal_ik_error_m: np.ndarray
    converged: np.ndarray

    def __post_init__(self) -> None:
        candidates = len(self.names)
        if self.raw_prefixes.ndim != 3 or self.raw_prefixes.shape[0] != candidates:
            raise ValueError("raw_prefixes must have shape [K,R,6]")
        if self.raw_prefixes.shape[-1] != 6:
            raise ValueError("raw_prefixes must contain six-joint commands")
        if self.horizon_states.ndim != 3 or self.horizon_states.shape != (
            candidates,
            self.horizon_xyz.shape[1],
            6,
        ):
            raise ValueError("horizon_states must align with horizon_xyz")
        if self.horizon_xyz.shape != (candidates, self.horizon_states.shape[1], 3):
            raise ValueError("horizon_xyz must have shape [K,H,3]")
        if self.target_surface_xyz.shape != (3,):
            raise ValueError("target_surface_xyz must be one XYZ vector")
        if self.target_approach_xyz.shape != (candidates, 3):
            raise ValueError("target_approach_xyz must have shape [K,3]")
        if self.terminal_ik_error_m.shape != (candidates,):
            raise ValueError("terminal_ik_error_m must have shape [K]")
        if self.converged.shape != (candidates,):
            raise ValueError("converged must have shape [K]")


@dataclass(frozen=True)
class AnchorTrajectoryIKResult:
    """Joint command prefix recovered from a three-anchor future trajectory."""

    raw_prefix: np.ndarray
    horizon_states: np.ndarray
    achieved_anchors: np.ndarray
    target_anchors: np.ndarray
    per_horizon_max_error_m: np.ndarray
    converged: bool

    def __post_init__(self) -> None:
        if self.horizon_states.ndim != 2 or self.horizon_states.shape[1:] != (6,):
            raise ValueError("horizon_states must have shape [H,6]")
        horizon = len(self.horizon_states)
        if self.raw_prefix.ndim != 2 or self.raw_prefix.shape[1:] != (6,):
            raise ValueError("raw_prefix must have shape [R,6]")
        expected_anchor_shape = (horizon, 3, 3)
        if self.achieved_anchors.shape != expected_anchor_shape:
            raise ValueError("achieved_anchors must have shape [H,3,3]")
        if self.target_anchors.shape != expected_anchor_shape:
            raise ValueError("target_anchors must have shape [H,3,3]")
        if self.per_horizon_max_error_m.shape != (horizon,):
            raise ValueError("per_horizon_max_error_m must have shape [H]")


@dataclass(frozen=True)
class IKPathSpec:
    name: str
    clearance_m: float
    arc_lift_m: float
    lateral_m: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("IK path name must be non-empty")
        values = np.asarray(
            (self.clearance_m, self.arc_lift_m, self.lateral_m), dtype=np.float64
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("IK path values must be finite")
        if self.clearance_m < 0.0 or self.arc_lift_m < 0.0:
            raise ValueError("IK path clearance and arc lift must be non-negative")


DEFAULT_PATH_SPECS = (
    IKPathSpec("direct_low", 0.025, 0.000, 0.000),
    IKPathSpec("direct_mid", 0.035, 0.000, 0.000),
    IKPathSpec("direct_high", 0.050, 0.000, 0.000),
    IKPathSpec("arc_mid", 0.035, 0.025, 0.000),
    IKPathSpec("arc_high", 0.050, 0.030, 0.000),
    IKPathSpec("arc_left", 0.035, 0.020, 0.015),
    IKPathSpec("arc_right", 0.035, 0.020, -0.015),
    IKPathSpec("conservative", 0.045, 0.040, 0.000),
)


REALTIME_CONTACT_PATH_SPECS = (
    IKPathSpec("direct_contact", 0.000, 0.000, 0.000),
    IKPathSpec("direct_10mm", 0.010, 0.000, 0.000),
    IKPathSpec("arc_contact", 0.000, 0.025, 0.000),
    IKPathSpec("arc_left_contact", 0.000, 0.020, 0.012),
    IKPathSpec("arc_right_contact", 0.000, 0.020, -0.012),
)

#: After a confirmed hold the 20–25 mm ``arc_*`` midpoints bob on every
#: replan. TRANSPORT / RELEASE / held-GRASP keep only the flat contact path.
HELD_CONTACT_PATH_SPECS = (IKPathSpec("direct_contact", 0.000, 0.000, 0.000),)


def realtime_path_specs_for_phase(
    phase: str,
    *,
    three_point_verdict: str | None = None,
) -> tuple[IKPathSpec, ...]:
    """IK paths for one manipulation phase.

    APPROACH keeps the full contact bank (including arcs). Once the object is
    held — TRANSPORT, RELEASE, or GRASP with ``three_point_verdict=="held"`` —
    only ``direct_contact`` remains so a ``--chunks 2`` replan cannot lift and
    drop again.
    """

    name = str(phase)
    held = three_point_verdict == "held"
    if name in {"transport", "release"} or (name == "grasp" and held):
        return HELD_CONTACT_PATH_SPECS
    return REALTIME_CONTACT_PATH_SPECS


def _joint_segment(start: np.ndarray, stop: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("joint interpolation count must be positive")
    fractions = np.arange(1, count + 1, dtype=np.float64) / float(count)
    return (start[None] + fractions[:, None] * (stop - start)[None]).astype(
        np.float32
    )


def _solve_position_with_limit_fallback(
    kinematics: SO101MuJoCoKinematics,
    target: np.ndarray,
    initial_state: np.ndarray,
    *,
    max_iterations: int,
    tolerance_m: float,
    max_step_normalized: float,
):
    """Retry a near-limit local minimum without the posture regularizer."""

    result = kinematics.inverse_position(
        target,
        initial_state,
        max_iterations=max_iterations,
        tolerance_m=tolerance_m,
        max_step_normalized=max_step_normalized,
    )
    if result.converged:
        return result
    fallback = kinematics.inverse_position(
        target,
        initial_state,
        max_iterations=max_iterations,
        tolerance_m=tolerance_m,
        max_step_normalized=max_step_normalized,
        posture_weight=0.0,
    )
    return fallback if fallback.error_m < result.error_m else result


def generate_ik_candidate_prefixes(
    kinematics: SO101MuJoCoKinematics,
    current_state: np.ndarray,
    target_surface_xyz: np.ndarray,
    *,
    frameskip: int = 5,
    horizon: int = 8,
    tolerance_m: float = 3e-3,
    max_step_normalized: float = 8.0,
    arrival_raw_steps: int | None = None,
    max_arrival_step: float | None = None,
    path_specs: tuple[IKPathSpec, ...] = DEFAULT_PATH_SPECS,
) -> IKCandidateBatch:
    """Generate diverse, smooth approach paths without learning an action decoder."""

    current = np.asarray(current_state, dtype=np.float64)
    surface = np.asarray(target_surface_xyz, dtype=np.float64)
    if current.shape != (6,) or not np.all(np.isfinite(current)):
        raise ValueError("current_state must be one finite six-joint vector")
    if surface.shape != (3,) or not np.all(np.isfinite(surface)):
        raise ValueError("target_surface_xyz must be one finite XYZ vector")
    if frameskip <= 0 or horizon <= 0:
        raise ValueError("frameskip and horizon must be positive")
    if not path_specs:
        raise ValueError("at least one IK path specification is required")
    if len({spec.name for spec in path_specs}) != len(path_specs):
        raise ValueError("IK path names must be unique")
    if max_arrival_step is not None and (
        not np.isfinite(max_arrival_step) or max_arrival_step <= 0
    ):
        raise ValueError("max_arrival_step must be finite and positive")

    start_xyz = np.asarray(kinematics.forward(current), dtype=np.float64)
    all_prefixes: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    all_xyz: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_errors: list[float] = []
    all_converged: list[bool] = []
    raw_count = frameskip * horizon
    if arrival_raw_steps is None:
        arrival_count = raw_count
    else:
        if (
            isinstance(arrival_raw_steps, bool)
            or not isinstance(arrival_raw_steps, int)
            or not 1 <= arrival_raw_steps <= raw_count
        ):
            raise ValueError("arrival_raw_steps must be within the raw horizon")
        arrival_count = arrival_raw_steps
    for spec in path_specs:
        target = surface.copy()
        target[2] += spec.clearance_m
        direction = target[:2] - start_xyz[:2]
        length = float(np.linalg.norm(direction))
        normal = (
            np.asarray((-direction[1], direction[0]), dtype=np.float64) / length
            if length > 1e-9
            else np.zeros(2, dtype=np.float64)
        )
        use_midpoint = bool(spec.arc_lift_m or spec.lateral_m)
        midpoint_result = None
        solve_start = current
        if use_midpoint:
            midpoint = 0.5 * (start_xyz + target)
            midpoint[2] += spec.arc_lift_m
            midpoint[:2] += spec.lateral_m * normal
            midpoint_result = _solve_position_with_limit_fallback(
                kinematics,
                midpoint,
                current,
                max_iterations=100,
                tolerance_m=tolerance_m,
                max_step_normalized=max_step_normalized,
            )
            solve_start = np.asarray(midpoint_result.state, dtype=np.float64)
        terminal_result = _solve_position_with_limit_fallback(
            kinematics,
            target,
            solve_start,
            max_iterations=150,
            tolerance_m=tolerance_m,
            max_step_normalized=max_step_normalized,
        )
        terminal_state = np.asarray(terminal_result.state, dtype=np.float64)
        terminal_state[5] = current[5]
        midpoint_state = None
        if midpoint_result is not None:
            midpoint_state = np.asarray(midpoint_result.state, dtype=np.float64)
            midpoint_state[5] = current[5]
        if max_arrival_step is None:
            path_arrival = arrival_count
            first_count = path_arrival // 2
        else:
            # Adaptive arrival: take steps near the allowed size instead of
            # stretching the remaining travel across the full arrival window.
            # Short moves finish inside one command chunk instead of decaying
            # geometrically across replans.
            if midpoint_state is None:
                needed = max(
                    1,
                    int(
                        np.ceil(
                            float(np.max(np.abs(terminal_state - current)))
                            / max_arrival_step
                            - 1e-9
                        )
                    ),
                )
                aligned = int(np.ceil(needed / frameskip)) * frameskip
                path_arrival = int(min(arrival_count, max(frameskip, aligned)))
                first_count = 0
            else:
                needed_first = max(
                    1,
                    int(
                        np.ceil(
                            float(np.max(np.abs(midpoint_state - current)))
                            / max_arrival_step
                            - 1e-9
                        )
                    ),
                )
                needed_second = max(
                    1,
                    int(
                        np.ceil(
                            float(
                                np.max(np.abs(terminal_state - midpoint_state))
                            )
                            / max_arrival_step
                            - 1e-9
                        )
                    ),
                )
                # Each arc segment is aligned to the frameskip grid separately
                # so the lifted midpoint lands on a subsampled horizon state
                # and stays visible to the planner's Cartesian workspace check.
                first_aligned = (
                    int(np.ceil(needed_first / frameskip)) * frameskip
                )
                second_aligned = (
                    int(np.ceil(needed_second / frameskip)) * frameskip
                )
                path_arrival = int(
                    min(arrival_count, first_aligned + second_aligned)
                )
                if path_arrival == first_aligned + second_aligned:
                    first_count = first_aligned
                elif path_arrival >= 2 * frameskip:
                    needed_total = needed_first + needed_second
                    first_count = (
                        int(
                            round(
                                path_arrival
                                * needed_first
                                / needed_total
                                / frameskip
                            )
                        )
                        * frameskip
                    )
                    first_count = min(
                        max(first_count, frameskip), path_arrival - frameskip
                    )
                else:
                    needed_total = needed_first + needed_second
                    first_count = int(
                        round(path_arrival * needed_first / needed_total)
                    )
                    first_count = min(max(first_count, 1), path_arrival - 1)
        if midpoint_state is None:
            moving_prefix = _joint_segment(current, terminal_state, path_arrival)
            path_converged = bool(terminal_result.converged)
        else:
            if path_arrival < 2:
                raise ValueError("arc paths require at least two arrival steps")
            moving_prefix = np.concatenate(
                (
                    _joint_segment(current, midpoint_state, first_count),
                    _joint_segment(
                        midpoint_state,
                        terminal_state,
                        path_arrival - first_count,
                    ),
                ),
                axis=0,
            )
            path_converged = bool(
                midpoint_result.converged and terminal_result.converged
            )
        if path_arrival < raw_count:
            raw_prefix = np.concatenate(
                (
                    moving_prefix,
                    np.repeat(
                        terminal_state[None], raw_count - path_arrival, axis=0
                    ).astype(np.float32),
                ),
                axis=0,
            )
        else:
            raw_prefix = moving_prefix
        horizon_states = raw_prefix[frameskip - 1 :: frameskip]
        horizon_xyz = kinematics.forward(horizon_states)
        all_prefixes.append(raw_prefix)
        all_states.append(horizon_states.astype(np.float32))
        all_xyz.append(horizon_xyz)
        all_targets.append(target)
        all_errors.append(float(terminal_result.error_m))
        all_converged.append(path_converged)

    return IKCandidateBatch(
        names=tuple(spec.name for spec in path_specs),
        raw_prefixes=np.stack(all_prefixes),
        horizon_states=np.stack(all_states),
        horizon_xyz=np.stack(all_xyz),
        target_surface_xyz=surface.copy(),
        target_approach_xyz=np.stack(all_targets),
        terminal_ik_error_m=np.asarray(all_errors, dtype=np.float64),
        converged=np.asarray(all_converged, dtype=bool),
    )


def track_anchor_trajectory(
    kinematics: SO101MuJoCoKinematics,
    current_state: np.ndarray,
    target_anchors: np.ndarray,
    *,
    frameskip: int = 5,
    tolerance_m: float = 3e-3,
    max_step_normalized: float = 8.0,
    limit_guard: float = 3.0,
) -> AnchorTrajectoryIKResult:
    """Turn LeWM center/fixed-tip/moving-tip forecasts into joint commands."""

    current = np.asarray(current_state, dtype=np.float64)
    targets = np.asarray(target_anchors, dtype=np.float64)
    if current.shape != (6,) or not np.all(np.isfinite(current)):
        raise ValueError("current_state must be one finite six-joint vector")
    if targets.ndim != 3 or targets.shape[1:] != (3, 3):
        raise ValueError("target_anchors must have shape [H,3,3]")
    if len(targets) == 0 or not np.all(np.isfinite(targets)):
        raise ValueError("target_anchors must be finite and non-empty")
    if frameskip <= 0:
        raise ValueError("frameskip must be positive")
    if not np.isfinite(limit_guard) or limit_guard < 0:
        raise ValueError("limit_guard must be finite and non-negative")

    previous = current.copy()
    hard_lower = np.asarray([-100.0] * 5 + [0.0], dtype=np.float64)
    hard_upper = np.asarray([100.0] * 6, dtype=np.float64)
    segments: list[np.ndarray] = []
    states: list[np.ndarray] = []
    achieved: list[np.ndarray] = []
    errors: list[float] = []
    converged = True
    for target in targets:
        solve_lower = hard_lower.copy()
        solve_upper = hard_upper.copy()
        near_lower = previous <= hard_lower + limit_guard
        near_upper = previous >= hard_upper - limit_guard
        solve_lower[near_lower] = previous[near_lower]
        solve_upper[near_upper] = previous[near_upper]
        result = kinematics.inverse_anchors(
            target,
            previous,
            tolerance_m=tolerance_m,
            max_step_normalized=max_step_normalized,
            position_lower=solve_lower,
            position_upper=solve_upper,
        )
        waypoint = np.asarray(result.state, dtype=np.float64)
        segments.append(_joint_segment(previous, waypoint, frameskip))
        states.append(waypoint.astype(np.float32))
        achieved.append(np.asarray(result.achieved_anchors, dtype=np.float64))
        errors.append(float(result.max_error_m))
        converged = converged and bool(result.converged)
        previous = waypoint

    return AnchorTrajectoryIKResult(
        raw_prefix=np.concatenate(segments, axis=0).astype(np.float32),
        horizon_states=np.stack(states).astype(np.float32),
        achieved_anchors=np.stack(achieved),
        target_anchors=targets.copy(),
        per_horizon_max_error_m=np.asarray(errors, dtype=np.float64),
        converged=bool(converged),
    )


__all__ = [
    "AnchorTrajectoryIKResult",
    "DEFAULT_PATH_SPECS",
    "HELD_CONTACT_PATH_SPECS",
    "IKCandidateBatch",
    "IKPathSpec",
    "REALTIME_CONTACT_PATH_SPECS",
    "generate_ik_candidate_prefixes",
    "realtime_path_specs_for_phase",
    "track_anchor_trajectory",
]
