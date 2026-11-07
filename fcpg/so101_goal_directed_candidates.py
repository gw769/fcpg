"""FCPG-grounded image-servo proposals for SO-101 receding-horizon control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from fcpg.so101_camera_calibration import ProjectionFit


class AnchorKinematics(Protocol):
    def forward_anchors(self, states: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ImageServoVariant:
    name: str
    gain: float
    damping: float
    maximum_joint_step: float
    transit_height_m: float

    def __post_init__(self) -> None:
        values = (
            self.gain,
            self.damping,
            self.maximum_joint_step,
            self.transit_height_m,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("image-servo variant values must be finite")
        if self.gain <= 0 or self.damping <= 0 or self.maximum_joint_step <= 0:
            raise ValueError("image-servo gain, damping and step must be positive")


DEFAULT_APPROACH_VARIANTS = (
    ImageServoVariant("servo_slow", 0.18, 1.0, 0.65, 0.030),
    ImageServoVariant("servo_balanced", 0.24, 0.75, 0.80, 0.035),
    ImageServoVariant("servo_direct", 0.30, 0.50, 0.95, 0.040),
    ImageServoVariant("servo_high_clearance", 0.24, 0.75, 0.80, 0.050),
)

DEFAULT_TRANSPORT_VARIANTS = (
    ImageServoVariant("transport_slow", 0.16, 1.00, 0.60, 0.075),
    ImageServoVariant("transport_balanced", 0.22, 0.80, 0.75, 0.085),
    ImageServoVariant("transport_direct", 0.27, 0.60, 0.90, 0.095),
)


def scale_image_servo_variants(
    variants: tuple[ImageServoVariant, ...],
    scale: float,
    *,
    max_joint_step: float | None = None,
) -> tuple[ImageServoVariant, ...]:
    """Scale image-servo gain and per-step joint cap; leave damping and height.

    Live FCPG prefixes win most rounds and sit on the 0.65–0.95°/step caps, so
    a single scale is what makes an executed 5-step chunk large enough to
    approach.  Damping and transit height stay put so this does not invent
    post-grasp vertical sway.
    """

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("image-servo scale must be finite and positive")
    if max_joint_step is not None and (
        not np.isfinite(max_joint_step) or max_joint_step <= 0.0
    ):
        raise ValueError("max_joint_step must be finite and positive")
    if scale == 1.0 and max_joint_step is None:
        return variants
    scaled: list[ImageServoVariant] = []
    for variant in variants:
        step = float(variant.maximum_joint_step) * float(scale)
        if max_joint_step is not None:
            step = min(step, float(max_joint_step))
        scaled.append(
            ImageServoVariant(
                variant.name,
                float(variant.gain) * float(scale),
                variant.damping,
                step,
                variant.transit_height_m,
            )
        )
    return tuple(scaled)


def use_transit_height(
    pixel_error: float,
    near_target_px: float,
    *,
    previously_transit: bool | None = None,
    hysteresis_band: float = 0.0,
) -> bool:
    """Whether this servo prefix should command transit height vs contact.

    A hard ``error > near`` step flips every replan when the pixel error rides
    the threshold. After the first look, stay in transit until well inside
    ``near * (1 - band)`` and stay in contact until well outside
    ``near * (1 + band)``. A zero band restores the historical hard step.
    """

    if (
        not np.isfinite(pixel_error)
        or pixel_error < 0.0
        or not np.isfinite(near_target_px)
        or near_target_px <= 0.0
        or not np.isfinite(hysteresis_band)
        or hysteresis_band < 0.0
    ):
        raise ValueError("transit-height gate distances must be finite")
    if previously_transit is None or hysteresis_band == 0.0:
        return float(pixel_error) > float(near_target_px)
    if previously_transit:
        return float(pixel_error) > float(near_target_px) * (
            1.0 - float(hysteresis_band)
        )
    return float(pixel_error) > float(near_target_px) * (
        1.0 + float(hysteresis_band)
    )


def _column_center_task(
    state: np.ndarray,
    fit: ProjectionFit,
    kinematics: AnchorKinematics,
    *,
    height_scale_px_per_m: float,
    live_transform_px: np.ndarray | None = None,
) -> np.ndarray:
    if fit.view != "column" or fit.feature_names != (
        "x_base",
        "y_base",
        "z_base",
        "bias",
    ):
        raise ValueError("image servo requires the calibrated column projection")
    anchors = np.asarray(kinematics.forward_anchors(state), dtype=np.float64)
    if anchors.shape != (3, 3) or not np.all(np.isfinite(anchors)):
        raise ValueError("kinematics must return three finite XYZ anchors")
    center = anchors[0]
    pixel = np.concatenate((center, [1.0])) @ np.asarray(
        fit.coefficients, dtype=np.float64
    )
    if live_transform_px is not None:
        transform = np.asarray(live_transform_px, dtype=np.float64)
        if transform.shape != (2, 3) or not np.all(np.isfinite(transform)):
            raise ValueError("live_transform_px must be a finite [2,3] matrix")
        pixel = np.concatenate((pixel, [1.0])) @ transform.T
    return np.asarray(
        [pixel[0], pixel[1], center[2] * height_scale_px_per_m],
        dtype=np.float64,
    )


def _bounded_damped_step(
    state: np.ndarray,
    error: np.ndarray,
    fit: ProjectionFit,
    kinematics: AnchorKinematics,
    variant: ImageServoVariant,
    *,
    finite_difference_step: float,
    height_scale_px_per_m: float,
    position_lower: np.ndarray,
    position_upper: np.ndarray,
    live_transform_px: np.ndarray | None,
    limit_guard: float = 3.0,
) -> np.ndarray:
    joint_indices = (0, 1, 2, 3)
    jacobian = np.empty((3, len(joint_indices)), dtype=np.float64)
    for column, joint in enumerate(joint_indices):
        low = max(
            float(position_lower[joint]),
            float(state[joint] - finite_difference_step),
        )
        high = min(
            float(position_upper[joint]),
            float(state[joint] + finite_difference_step),
        )
        if high <= low:
            raise ValueError("finite-difference image Jacobian has no support")
        before = state.copy()
        after = state.copy()
        before[joint] = low
        after[joint] = high
        jacobian[:, column] = (
            _column_center_task(
                after,
                fit,
                kinematics,
                height_scale_px_per_m=height_scale_px_per_m,
                live_transform_px=live_transform_px,
            )
            - _column_center_task(
                before,
                fit,
                kinematics,
                height_scale_px_per_m=height_scale_px_per_m,
                live_transform_px=live_transform_px,
            )
        ) / (high - low)

    regularized = (
        jacobian.T @ jacobian
        + np.square(variant.damping) * np.eye(len(joint_indices))
    )
    delta = variant.gain * np.linalg.solve(regularized, jacobian.T @ error)
    joint_state = state[list(joint_indices)]
    joint_lower = position_lower[list(joint_indices)]
    joint_upper = position_upper[list(joint_indices)]
    outward = ((joint_state <= joint_lower + limit_guard) & (delta < 0.0)) | (
        (joint_state >= joint_upper - limit_guard) & (delta > 0.0)
    )
    delta[outward] = 0.0
    maximum = float(np.max(np.abs(delta)))
    if maximum > variant.maximum_joint_step:
        delta *= variant.maximum_joint_step / maximum

    result = state.copy()
    result[list(joint_indices)] += delta
    result = np.clip(result, position_lower, position_upper)
    return result


def generate_fcpg_goal_prefixes(
    current_state: np.ndarray,
    current_corrected_center_px: np.ndarray,
    target_px: np.ndarray,
    fit: ProjectionFit,
    kinematics: AnchorKinematics,
    *,
    contact_height_m: float,
    position_lower: np.ndarray,
    position_upper: np.ndarray,
    horizon_raw: int = 40,
    near_target_px: float = 24.0,
    finite_difference_step: float = 0.25,
    height_scale_px_per_m: float = 500.0,
    variants: tuple[ImageServoVariant, ...] = DEFAULT_APPROACH_VARIANTS,
    source: str = "fcpg_goal_servo",
    transit_height_m: float | None = None,
    live_transform_px: np.ndarray | None = None,
    previously_transit: bool | None = None,
    hysteresis_band: float = 0.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate absolute-state proposals that move the FCPG center toward a target.

    The current FCPG correction is retained as a local pixel offset while each
    proposal is integrated. Wrist roll and gripper commands remain unchanged.
    """

    state = np.asarray(current_state, dtype=np.float64)
    corrected = np.asarray(current_corrected_center_px, dtype=np.float64)
    target = np.asarray(target_px, dtype=np.float64)
    lower = np.asarray(position_lower, dtype=np.float64)
    upper = np.asarray(position_upper, dtype=np.float64)
    if state.shape != (6,) or corrected.shape != (2,) or target.shape != (2,):
        raise ValueError("state must be [6] and image points must be [2]")
    if lower.shape != (6,) or upper.shape != (6,):
        raise ValueError("position bounds must be six-vectors")
    if not all(
        np.all(np.isfinite(value))
        for value in (state, corrected, target, lower, upper)
    ):
        raise ValueError("image-servo inputs must be finite")
    if np.any(upper <= lower):
        raise ValueError("position upper bounds must exceed lower bounds")
    positive = (
        horizon_raw,
        near_target_px,
        finite_difference_step,
        height_scale_px_per_m,
    )
    if any(not np.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("image-servo settings must be finite and positive")
    if not variants:
        raise ValueError("at least one image-servo variant is required")
    if transit_height_m is not None and (
        not np.isfinite(transit_height_m) or transit_height_m <= 0
    ):
        raise ValueError("transit_height_m must be finite and positive")
    if live_transform_px is not None:
        live_transform_px = np.asarray(live_transform_px, dtype=np.float64)
        if live_transform_px.shape != (2, 3) or not np.all(
            np.isfinite(live_transform_px)
        ):
            raise ValueError("live_transform_px must be a finite [2,3] matrix")

    nominal_task = _column_center_task(
        state,
        fit,
        kinematics,
        height_scale_px_per_m=height_scale_px_per_m,
        live_transform_px=live_transform_px,
    )
    correction_offset = corrected - nominal_task[:2]
    initial_distance = float(np.linalg.norm(corrected - target))
    use_transit = use_transit_height(
        initial_distance,
        float(near_target_px),
        previously_transit=previously_transit,
        hysteresis_band=hysteresis_band,
    )
    prefixes: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []

    for variant in variants:
        desired_height = (
            (
                variant.transit_height_m
                if transit_height_m is None
                else float(transit_height_m)
            )
            if use_transit
            else float(contact_height_m)
        )
        desired = np.asarray(
            [target[0], target[1], desired_height * height_scale_px_per_m],
            dtype=np.float64,
        )
        proposal = state.copy()
        rows = np.empty((int(horizon_raw), 6), dtype=np.float64)
        for step in range(int(horizon_raw)):
            task = _column_center_task(
                proposal,
                fit,
                kinematics,
                height_scale_px_per_m=height_scale_px_per_m,
                live_transform_px=live_transform_px,
            )
            task[:2] += correction_offset
            proposal = _bounded_damped_step(
                proposal,
                desired - task,
                fit,
                kinematics,
                variant,
                finite_difference_step=finite_difference_step,
                height_scale_px_per_m=height_scale_px_per_m,
                position_lower=lower,
                position_upper=upper,
                live_transform_px=live_transform_px,
            )
            rows[step] = proposal

        final_task = _column_center_task(
            rows[-1],
            fit,
            kinematics,
            height_scale_px_per_m=height_scale_px_per_m,
            live_transform_px=live_transform_px,
        )
        final_task[:2] += correction_offset
        prefixes.append(rows.astype(np.float32))
        reports.append(
            {
                "source": "fcpg_goal_servo",
                "goal_role": source,
                "variant": variant.name,
                "initial_pixel_error": initial_distance,
                "kinematic_final_pixel_error": float(
                    np.linalg.norm(final_task[:2] - target)
                ),
                "desired_height_m": desired_height,
                "use_transit": use_transit,
                "kinematic_final_height_m": float(
                    final_task[2] / height_scale_px_per_m
                ),
                "maximum_joint_step": variant.maximum_joint_step,
            }
        )

    return np.stack(prefixes), reports


def generate_fcpg_approach_prefixes(*args: Any, **kwargs: Any) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Compatibility wrapper for the pickup-approach role."""

    kwargs.setdefault("source", "pickup")
    return generate_fcpg_goal_prefixes(*args, **kwargs)


def generate_gripper_phase_prefixes(
    current_state: np.ndarray,
    target_positions: tuple[float, ...],
    *,
    position_lower: np.ndarray,
    position_upper: np.ndarray,
    horizon_raw: int = 40,
    maximum_step: float = 1.5,
    source: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Generate arm-stationary close/open proposals for grasp and release."""

    state = np.asarray(current_state, dtype=np.float64)
    lower = np.asarray(position_lower, dtype=np.float64)
    upper = np.asarray(position_upper, dtype=np.float64)
    if state.shape != (6,) or lower.shape != (6,) or upper.shape != (6,):
        raise ValueError("gripper candidate states and bounds must be six-vectors")
    if not all(np.all(np.isfinite(value)) for value in (state, lower, upper)):
        raise ValueError("gripper candidate inputs must be finite")
    if horizon_raw <= 0 or not np.isfinite(maximum_step) or maximum_step <= 0:
        raise ValueError("gripper candidate horizon and step must be positive")
    if not target_positions:
        raise ValueError("at least one gripper target is required")

    prefixes: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    for index, target in enumerate(target_positions):
        if not np.isfinite(target):
            raise ValueError("gripper target must be finite")
        bounded_target = float(np.clip(target, lower[-1], upper[-1]))
        proposal = state.copy()
        rows = np.empty((int(horizon_raw), 6), dtype=np.float64)
        for step in range(int(horizon_raw)):
            delta = float(
                np.clip(
                    bounded_target - proposal[-1],
                    -maximum_step,
                    maximum_step,
                )
            )
            proposal[-1] += delta
            rows[step] = proposal
        prefixes.append(rows.astype(np.float32))
        reports.append(
            {
                "source": "phase_gripper",
                "goal_role": source,
                "variant": f"{source}_{index}",
                "target_gripper_position": bounded_target,
                "maximum_joint_step": maximum_step,
            }
        )
    return np.stack(prefixes), reports


__all__ = [
    "DEFAULT_APPROACH_VARIANTS",
    "DEFAULT_TRANSPORT_VARIANTS",
    "ImageServoVariant",
    "generate_fcpg_approach_prefixes",
    "generate_fcpg_goal_prefixes",
    "generate_gripper_phase_prefixes",
    "scale_image_servo_variants",
    "use_transit_height",
]
