"""V11 dual-view action source for SO-101 production control.

Wraps V11ShadowRunner to provide production action chunks that flow through
the existing supervisor → validation → execute pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections import deque
    from fcpg.so101_realtime_planner import SO101RealtimePlanner

logger = logging.getLogger(__name__)

# Same safety thresholds as shadow mode (normalized units = raw / 100)
_MAX_ARM_JUMP = 0.35
_MAX_GRIPPER_JUMP = 0.35
V11_CANDIDATE_SOURCE = "v11_dual_view_policy"


def rate_limit_command_chunk(
    chunk: np.ndarray,
    current_state: np.ndarray,
    max_jump: float,
) -> np.ndarray:
    """Clip each command toward the V11 target so no step exceeds max_jump."""
    values = np.asarray(chunk, dtype=np.float32)
    prev = np.asarray(current_state, dtype=np.float32)
    limited = np.empty_like(values)
    for index, command in enumerate(values):
        delta = command - prev
        magnitude = float(np.max(np.abs(delta)))
        if magnitude > max_jump:
            delta = delta * np.float32(max_jump / magnitude)
        limited[index] = prev + delta
        prev = limited[index]
    return limited


def hold_fill_v11_prefix(
    chunk: np.ndarray,
    horizon_raw: int,
    *,
    command_chunk: int = 5,
) -> np.ndarray:
    """Pad a V11 command chunk to the planner prefix horizon by holding the last step."""

    values = np.asarray(chunk, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6 or not np.all(np.isfinite(values)):
        raise ValueError("V11 chunk must be a finite [T, 6] array")
    if int(command_chunk) < 1 or values.shape[0] != int(command_chunk):
        raise ValueError(
            f"V11 chunk must have shape [{int(command_chunk)}, 6], got {values.shape}"
        )
    horizon = int(horizon_raw)
    if horizon < int(command_chunk):
        raise ValueError("horizon_raw must be at least command_chunk")
    prefix = np.empty((horizon, 6), dtype=np.float32)
    prefix[: int(command_chunk)] = values
    prefix[int(command_chunk) :] = values[-1]
    return prefix


class V11Controller:
    """Production V11 action source.

    Generates action chunks from V11 dual-view policy for real control.
    Output flows through existing safety supervisor before execution.

    Parameters
    ----------
    checkpoint_path:
        Path to the V11 best checkpoint.
    vit_checkpoint_path:
        Path to the frozen V9 ViT init checkpoint (may be unused if V11
        checkpoint already contains ViT weights).
    planner:
        The live production planner. Used for FK kinematics only.
    device:
        Torch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        vit_checkpoint_path: str | Path,
        planner: "SO101RealtimePlanner",
        *,
        device: str = "cuda:0",
    ) -> None:
        from fcpg.so101_v11_shadow import V11ShadowRunner

        # Reuse V11ShadowRunner's model loading and forward pass logic
        self.shadow = V11ShadowRunner(
            checkpoint_path=checkpoint_path,
            vit_checkpoint_path=vit_checkpoint_path,
            planner=planner,
            device=device,
        )

        logger.info(
            "V11Controller ready for production control: device=%s",
            device,
        )

    def predict_chunk(
        self,
        observations: deque,
        current_state: np.ndarray,
        goal_state_raw: np.ndarray | None = None,
        task_class: int | None = None,
    ) -> dict[str, Any]:
        """Generate one V11 action chunk for production execution.

        Parameters
        ----------
        observations:
            Runtime's ``self.observations`` deque (maxlen=71).
            Each entry is a LiveObservation with .state in LeRobot raw
            units and .frames as uint8 [224,224,3].
        current_state:
            Current arm state [6] in LeRobot raw units.
        goal_state_raw:
            Optional structured-goal proprio in the same raw units.

        Returns
        -------
        dict with:
            - ``executable``: bool, whether the chunk passed safety checks
            - ``selected_chunk``: np.ndarray [5, 6] in LeRobot raw units
            - ``reason``: str if not executable
            - ``diagnostics``: dict with prediction details
        """
        result = self.shadow.predict(
            observations,
            current_state,
            goal_state_raw=goal_state_raw,
            task_class=task_class,
        )

        if not result.get("shadow_ok", False):
            return {
                "executable": False,
                "selected_chunk": None,
                "reason": result.get("reason", result.get("error", "unknown")),
                "diagnostics": result,
            }

        # Check safety gates
        if not result.get("v11_shadow_pass", False):
            reasons = []
            if result.get("nan_inf", False):
                reasons.append("nan_or_inf_in_prediction")
            if not result.get("arm_safe", True):
                reasons.append(
                    f"arm_jump={result.get('max_arm_joint_jump', 0.0):.3f}"
                    f">{_MAX_ARM_JUMP}"
                )
            if not result.get("gripper_safe", True):
                reasons.append(
                    f"gripper_jump={result.get('max_gripper_jump', 0.0):.3f}"
                    f">{_MAX_GRIPPER_JUMP}"
                )

            return {
                "executable": False,
                "selected_chunk": None,
                "reason": "; ".join(reasons) if reasons else "safety_check_failed",
                "diagnostics": result,
            }

        if "first_chunk" not in result:
            return {
                "executable": False,
                "selected_chunk": None,
                "reason": "v11_shadow_missing_first_chunk",
                "diagnostics": result,
            }
        chunk = np.asarray(result["first_chunk"], dtype=np.float32)
        if chunk.shape != (5, 6) or not np.all(np.isfinite(chunk)):
            return {
                "executable": False,
                "selected_chunk": None,
                "reason": f"v11_first_chunk_invalid_shape={getattr(chunk, 'shape', None)}",
                "diagnostics": result,
            }

        return {
            "executable": True,
            "selected_chunk": chunk,
            "reason": None,
            "diagnostics": result,
        }


__all__ = [
    "V11Controller",
    "V11_CANDIDATE_SOURCE",
    "hold_fill_v11_prefix",
    "rate_limit_command_chunk",
]
