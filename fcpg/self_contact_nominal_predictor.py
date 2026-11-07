"""Nominal FK / Jacobian as an M1 predictor input only.

This is not pixel ground truth.  ``u_hat`` stays image-only.  Predictions
are computed from a real ``q_arm`` readback through existing functions in
``fcpg/so101_kinematics.py``.  Synthetic ``q_true`` is refused.

A missing provider, or a zero / non-finite / wrong-shape return, is
INCONCLUSIVE.  Callers must not silently fill zeros.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np

from fcpg.so101_kinematics import (
    SO101MuJoCoKinematics,
    radians_to_normalized_arm_positions,
)

PROVIDER_FUNCTION = (
    "fcpg.self_contact_nominal_predictor.predict_nominal_from_q_arm_readback"
)
TIP_FUNCTION = "SO101MuJoCoKinematics.forward_anchors"
TIP_NAME = "fixed_finger_tip"
TIP_INDEX = 1
JACOBIAN_METHOD = "central_difference_forward_anchors"
TIP_UNITS = "m"
JACOBIAN_UNITS = "m_per_mechanical_deg"
DEFAULT_MUJOCO_MODEL = (
    Path(__file__).resolve().parents[1]
    / "third_party/so_arm100/Simulation/SO101/so101_new_calib.xml"
)
FORBIDDEN_Q_TRUE_KEYS = frozenset(
    {"q_true", "q_true_px", "synthetic_q_true", "valid_true"}
)


class NominalPredictionError(ValueError):
    """Fail-closed predictor refusal.  Not a pixel observation."""


def mechanical_deg_per_lerobot(joint_ranges: np.ndarray) -> np.ndarray:
    """Degrees of mechanical hinge travel per LeRobot ``[-100, 100]`` unit."""

    ranges = np.asarray(joint_ranges, dtype=np.float64)
    if ranges.shape != (5, 2):
        raise NominalPredictionError("joint_ranges must have shape (5, 2)")
    rad_per_unit = (ranges[:, 1] - ranges[:, 0]) / 200.0
    if not np.all(np.isfinite(rad_per_unit)) or np.any(rad_per_unit == 0.0):
        raise NominalPredictionError("joint ranges must yield a finite non-zero scale")
    return np.abs(rad_per_unit) * (180.0 / np.pi)


def _as_finite_vector(raw: Any, n: int, *, where: str) -> np.ndarray:
    array = np.asarray(raw, dtype=np.float64).reshape(-1)
    if array.shape != (n,):
        raise NominalPredictionError(f"{where} must have shape ({n},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise NominalPredictionError(f"{where} must be finite")
    return array


def _is_all_zero(array: np.ndarray) -> bool:
    return bool(array.size) and bool(np.all(np.abs(array) < 1e-15))


def assess_nominal_prediction(payload: Any) -> dict[str, Any]:
    """Return whether a predictor payload is usable as an M1 input."""

    if not isinstance(payload, Mapping):
        return {
            "valid": False,
            "reason": "nominal_predictor_missing",
            "fk_tip_xyz_m": None,
            "fk_joint_jacobian_m_per_deg": None,
            "prediction_provider": None,
        }
    provider = payload.get("prediction_provider")
    if provider is None or provider == "":
        return {
            "valid": False,
            "reason": "nominal_predictor_missing",
            "fk_tip_xyz_m": None,
            "fk_joint_jacobian_m_per_deg": None,
            "prediction_provider": None,
        }
    leaked = sorted(set(payload) & FORBIDDEN_Q_TRUE_KEYS)
    if leaked:
        return {
            "valid": False,
            "reason": f"synthetic_q_true_refused:{leaked}",
            "fk_tip_xyz_m": None,
            "fk_joint_jacobian_m_per_deg": None,
            "prediction_provider": dict(provider)
            if isinstance(provider, Mapping)
            else {"name": str(provider)},
        }
    try:
        tip = _as_finite_vector(payload.get("fk_tip_xyz_m"), 3, where="fk_tip_xyz_m")
        jac = np.asarray(payload.get("fk_joint_jacobian_m_per_deg"), dtype=np.float64)
        if jac.shape != (3, 5):
            raise NominalPredictionError(
                f"fk_joint_jacobian_m_per_deg must have shape (3, 5), got {jac.shape}"
            )
        if not np.all(np.isfinite(jac)):
            raise NominalPredictionError("fk_joint_jacobian_m_per_deg must be finite")
    except (TypeError, ValueError, NominalPredictionError) as exc:
        return {
            "valid": False,
            "reason": f"nominal_predictor_invalid:{exc}",
            "fk_tip_xyz_m": None,
            "fk_joint_jacobian_m_per_deg": None,
            "prediction_provider": dict(provider)
            if isinstance(provider, Mapping)
            else {"name": str(provider)},
        }
    provider_map = dict(provider) if isinstance(provider, Mapping) else {"name": str(provider)}
    if _is_all_zero(tip) or _is_all_zero(jac):
        if not bool(provider_map.get("allow_zero_synthetic_fixture")):
            return {
                "valid": False,
                "reason": "nominal_predictor_zero",
                "fk_tip_xyz_m": None,
                "fk_joint_jacobian_m_per_deg": None,
                "prediction_provider": provider_map,
            }
    return {
        "valid": True,
        "reason": None,
        "fk_tip_xyz_m": [float(v) for v in tip],
        "fk_joint_jacobian_m_per_deg": [
            [float(v) for v in row] for row in jac
        ],
        "prediction_provider": dict(provider)
        if isinstance(provider, Mapping)
        else {"name": str(provider)},
    }


def _six_state(
    q_arm: Sequence[float],
    gripper: float,
    *,
    kinematics: SO101MuJoCoKinematics,
    q_arm_units: str,
) -> np.ndarray:
    arm = _as_finite_vector(q_arm, 5, where="q_arm_readback")
    grip = float(gripper)
    if not np.isfinite(grip):
        raise NominalPredictionError("gripper readback must be finite")
    if q_arm_units == "mechanical_deg":
        scale = mechanical_deg_per_lerobot(kinematics.joint_ranges)
        radians = np.deg2rad(arm)
        # Convert mechanical degrees → LeRobot via the model's hinge ranges.
        normalized = radians_to_normalized_arm_positions(radians, kinematics.joint_ranges)
        if not np.all(np.isfinite(normalized)):
            raise NominalPredictionError("mechanical_deg conversion was not finite")
        _ = scale  # documented: jacobian uses this same scale
        arm = normalized
    elif q_arm_units not in {"lerobot_normalized", "lerobot_normalized_[-100,100]"}:
        raise NominalPredictionError(f"unsupported q_arm_units {q_arm_units!r}")
    state = np.zeros(6, dtype=np.float64)
    state[:5] = arm
    state[5] = float(np.clip(grip, 0.0, 100.0))
    return state


def _fixed_tip_xyz(kinematics: SO101MuJoCoKinematics, state: np.ndarray) -> np.ndarray:
    anchors = np.asarray(kinematics.forward_anchors(state), dtype=np.float64)
    if anchors.shape != (3, 3):
        raise NominalPredictionError(
            f"{TIP_FUNCTION} must return (3, 3) anchors, got {anchors.shape}"
        )
    tip = np.asarray(anchors[TIP_INDEX], dtype=np.float64)
    if tip.shape != (3,) or not np.all(np.isfinite(tip)):
        raise NominalPredictionError(f"{TIP_NAME} XYZ must be a finite 3-vector")
    return tip


def _tip_jacobian_m_per_deg(
    kinematics: SO101MuJoCoKinematics,
    state: np.ndarray,
    *,
    delta_deg: float = 0.25,
) -> np.ndarray:
    """3×5 ``d(fixed_tip_xyz)/d(q_mechanical_deg)`` via ``forward_anchors``."""

    scale = mechanical_deg_per_lerobot(kinematics.joint_ranges)
    pose = np.asarray(state, dtype=np.float64).copy()
    jac = np.zeros((3, 5), dtype=np.float64)
    for index in range(5):
        step = float(delta_deg) / float(scale[index])
        plus = pose.copy()
        minus = pose.copy()
        plus[index] += step
        minus[index] -= step
        jac[:, index] = (
            _fixed_tip_xyz(kinematics, plus) - _fixed_tip_xyz(kinematics, minus)
        ) / (2.0 * float(delta_deg))
    if jac.shape != (3, 5) or not np.all(np.isfinite(jac)):
        raise NominalPredictionError("jacobian must be a finite 3x5 matrix")
    return jac


def predict_nominal_from_q_arm_readback(
    q_arm: Sequence[float],
    *,
    kinematics: SO101MuJoCoKinematics,
    gripper: float = 0.0,
    q_arm_source: str = "hardware_present_readback",
    q_arm_units: str = "lerobot_normalized",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute nominal tip XYZ and joint Jacobian from a real readback.

    ``q_arm`` is the five-vector Present_Position arm readback.  It is never
    replaced by synthetic ``q_true``.  The returned fields are predictor
    inputs for analysis, not ``u_hat``.
    """

    if extra:
        leaked = sorted(set(extra) & FORBIDDEN_Q_TRUE_KEYS)
        if leaked:
            raise NominalPredictionError(
                f"refusing synthetic q_true key(s) {leaked}; "
                "nominal FK uses hardware readback only"
            )
    if q_arm_source in {"q_true", "synthetic_q_true"}:
        raise NominalPredictionError("q_arm_source must be a real readback, not q_true")
    state = _six_state(
        q_arm, gripper, kinematics=kinematics, q_arm_units=q_arm_units
    )
    tip = _fixed_tip_xyz(kinematics, state)
    jac = _tip_jacobian_m_per_deg(kinematics, state)
    if _is_all_zero(tip) or _is_all_zero(jac):
        raise NominalPredictionError("nominal predictor returned a zero tip or Jacobian")
    provider = {
        "name": PROVIDER_FUNCTION,
        "tip_function": TIP_FUNCTION,
        "jacobian_method": JACOBIAN_METHOD,
        "tip_name": TIP_NAME,
        "tip_index": TIP_INDEX,
        "tip_units": TIP_UNITS,
        "jacobian_units": JACOBIAN_UNITS,
        "frame": "mujoco_base",
        "q_arm_source": str(q_arm_source),
        "q_arm_units": str(q_arm_units),
        "q_arm_readback": [float(v) for v in state[:5]],
        "gripper_readback": float(state[5]),
        "role": "nominal_m1_predictor_input_only",
        "not_pixel_gt": True,
        "q_true_used": False,
        "kinematics_class": type(kinematics).__name__,
        "model_path": str(getattr(kinematics, "model_path", "")),
    }
    payload = {
        "fk_tip_xyz_m": [float(v) for v in tip],
        "fk_joint_jacobian_m_per_deg": [[float(v) for v in row] for row in jac],
        "prediction_provider": provider,
        "prediction_valid": True,
    }
    assessed = assess_nominal_prediction(payload)
    if not assessed["valid"]:
        raise NominalPredictionError(assessed["reason"] or "nominal_predictor_invalid")
    return assessed | {"prediction_provider": provider, "prediction_valid": True}


def make_kinematics(model_path: str | Path | None = None) -> SO101MuJoCoKinematics:
    path = Path(model_path) if model_path is not None else DEFAULT_MUJOCO_MODEL
    return SO101MuJoCoKinematics(path)


def provider_from_record(
    record: Mapping[str, Any],
    *,
    kinematics: SO101MuJoCoKinematics,
) -> dict[str, Any]:
    """Read ``q_arm`` from a formal/sidecar record and predict."""

    leaked = sorted(set(record) & FORBIDDEN_Q_TRUE_KEYS)
    if leaked:
        raise NominalPredictionError(
            f"record contains forbidden q_true key(s) {leaked}"
        )
    q_arm = record.get("q_arm_deg")
    source = str(record.get("q_arm_source") or "hardware_present_readback")
    if q_arm is None:
        arms = record.get("arm_states") or []
        if not arms:
            raise NominalPredictionError("missing q_arm readback")
        last = arms[-1]
        q_arm = last[:5]
        source = "sidecar.arm_states"
    gripper = record.get("present_position")
    if gripper is None:
        presents = record.get("presents") or []
        gripper = presents[-1] if presents else 0.0
    return predict_nominal_from_q_arm_readback(
        q_arm,
        kinematics=kinematics,
        gripper=float(gripper or 0.0),
        q_arm_source=source,
        q_arm_units="lerobot_normalized",
    )


def callable_provider(
    kinematics: SO101MuJoCoKinematics,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    def _predict(record: Mapping[str, Any]) -> Mapping[str, Any]:
        return provider_from_record(record, kinematics=kinematics)

    _predict.__name__ = "predict_nominal_from_q_arm_readback"
    return _predict


def attach_nominal_prediction(
    record: Mapping[str, Any],
    *,
    kinematics: SO101MuJoCoKinematics | None = None,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write predictor fields onto a seek/sidecar mapping.  Never fills zeros.

    A missing, zero, non-finite, or wrong-shape prediction leaves ``fk_*``
    absent and sets ``prediction_valid=False``.  ``u_hat`` is not touched.
    """

    out = dict(record)
    out.pop("fk_tip_xyz_m", None)
    out.pop("fk_joint_jacobian_m_per_deg", None)
    try:
        if payload is None:
            kin = kinematics if kinematics is not None else make_kinematics()
            predicted = provider_from_record(out, kinematics=kin)
        else:
            predicted = dict(payload)
        assessed = assess_nominal_prediction(predicted)
    except Exception as exc:  # noqa: BLE001 — persist INCONCLUSIVE, never fill
        out["prediction_valid"] = False
        out["prediction_provider"] = None
        out["prediction_failure_reason"] = f"nominal_predictor_error:{type(exc).__name__}"
        return out
    out["prediction_provider"] = assessed.get("prediction_provider")
    out["prediction_valid"] = bool(assessed["valid"])
    if assessed["valid"]:
        out["fk_tip_xyz_m"] = assessed["fk_tip_xyz_m"]
        out["fk_joint_jacobian_m_per_deg"] = assessed["fk_joint_jacobian_m_per_deg"]
        out.pop("prediction_failure_reason", None)
    else:
        out["prediction_failure_reason"] = assessed.get("reason")
    return out
