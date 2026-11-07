"""Minimal real-endpoint adapter for the tip-free Prod-V18.1 controller."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from fcpg.prod_v18_1 import (
    PATH_STEPS,
    STATE_DIM,
    VIEW_ORDER,
    project_v181_action_trajectory,
)
from fcpg.prod_v18_1_runtime import V181RuntimeController
from fcpg.prod_v18_1_training import (
    extract_dinov2_tokens,
    load_frozen_dinov2,
    load_v181_checkpoint,
)


JOINT_SCALE = 100.0
STEP_CAP_DEG = np.asarray([4.0, 4.0, 4.0, 4.0, 4.0, 8.0], dtype=np.float32)


class V181LiveError(RuntimeError):
    """Raised before malformed live inputs can reach V18.1 or the robot."""


def _inward_only_action_bounds(
    current_q_position: np.ndarray,
    base_lower_position: np.ndarray,
    base_upper_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep an already guarded arm joint stationary or move it inward.

    The deployment projector normally stays three normalized units away from
    each calibrated arm endpoint.  A measured state can settle just outside
    that guard without violating the electrical limit.  Forcing such a state
    immediately onto the guard creates an artificial state innovation.  The
    effective bound is therefore widened only to the measured state and only
    for the five arm joints; it can never authorize additional outward travel.
    """

    current = np.asarray(current_q_position, dtype=np.float32)
    lower = np.asarray(base_lower_position, dtype=np.float32)
    upper = np.asarray(base_upper_position, dtype=np.float32)
    if (
        current.shape != (STATE_DIM,)
        or lower.shape != (STATE_DIM,)
        or upper.shape != (STATE_DIM,)
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower >= upper)
    ):
        raise V181LiveError("action-bound inputs must be finite ordered six-vectors")
    effective_lower = lower.copy()
    effective_upper = upper.copy()
    effective_lower[:5] = np.minimum(effective_lower[:5], current[:5])
    effective_upper[:5] = np.maximum(effective_upper[:5], current[:5])
    return effective_lower, effective_upper


class V181LivePolicy:
    """Own DINO extraction and the actual-observation V18.1 runtime clock."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        dino_repository: str | Path,
        dino_weights: str | Path,
        device: torch.device | str,
        action_lower_position: np.ndarray | None = None,
        action_upper_position: np.ndarray | None = None,
        actor_conditioning: str = "full",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise V181LiveError("CUDA was requested but is unavailable")
        model, payload = load_v181_checkpoint(checkpoint, device=self.device)
        if (action_lower_position is None) != (action_upper_position is None):
            raise V181LiveError(
                "deployment action lower/upper bounds must be supplied together"
            )
        if action_lower_position is not None:
            lower = np.asarray(action_lower_position, dtype=np.float32)
            upper = np.asarray(action_upper_position, dtype=np.float32)
            if (
                lower.shape != (STATE_DIM,)
                or upper.shape != (STATE_DIM,)
                or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(lower >= upper)
            ):
                raise V181LiveError(
                    "deployment action bounds must be finite ordered six-vectors"
                )
            self.action_lower_position = lower.copy()
            self.action_upper_position = upper.copy()
        else:
            self.action_lower_position = (
                model.actor.action_lower.detach().cpu().numpy() * JOINT_SCALE
            ).astype(np.float32)
            self.action_upper_position = (
                model.actor.action_upper.detach().cpu().numpy() * JOINT_SCALE
            ).astype(np.float32)
        self.effective_action_lower_position = self.action_lower_position.copy()
        self.effective_action_upper_position = self.action_upper_position.copy()
        self.backbone = load_frozen_dinov2(
            dino_repository, dino_weights, device=self.device
        )
        self.runtime = V181RuntimeController(
            model,
            device=self.device,
            actor_conditioning=actor_conditioning,
        )
        self.action_source = f"v18_1_tip_free_{actor_conditioning}_controller"
        self._install_action_bounds(
            self.effective_action_lower_position,
            self.effective_action_upper_position,
        )
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.training_phase = str(payload.get("phase", "unknown"))
        self.seed = int(payload.get("seed", -1))
        self.last_proposal: dict[str, Any] | None = None
        self.last_completion: dict[str, Any] | None = None

    def _install_action_bounds(
        self, lower_position: np.ndarray, upper_position: np.ndarray
    ) -> None:
        with torch.no_grad():
            self.runtime.model.actor.action_lower.copy_(
                torch.from_numpy(lower_position / JOINT_SCALE).to(
                    self.runtime.model.actor.action_lower.device,
                    dtype=self.runtime.model.actor.action_lower.dtype,
                )
            )
            self.runtime.model.actor.action_upper.copy_(
                torch.from_numpy(upper_position / JOINT_SCALE).to(
                    self.runtime.model.actor.action_upper.device,
                    dtype=self.runtime.model.actor.action_upper.dtype,
                )
            )

    def _tokens(self, frames: Mapping[str, np.ndarray]) -> torch.Tensor:
        if set(frames) != set(VIEW_ORDER):
            raise V181LiveError(
                f"live frames must have exactly {list(VIEW_ORDER)}"
            )
        ordered: list[np.ndarray] = []
        for view in VIEW_ORDER:
            frame = np.asarray(frames[view])
            if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
                raise V181LiveError(f"{view} must be uint8 [224,224,3]")
            ordered.append(frame)
        tokens = extract_dinov2_tokens(
            self.backbone,
            np.stack(ordered),
            device=self.device,
            batch_size=2,
        )
        # Float32 avoids relying on half-precision LayerNorm support on V100.
        return torch.from_numpy(tokens).to(self.device, dtype=torch.float32)

    def propose(
        self,
        frames: Mapping[str, np.ndarray],
        current_q_deg: np.ndarray,
        *,
        replace_pending: bool = False,
    ) -> dict[str, Any]:
        state_deg = np.asarray(current_q_deg, dtype=np.float32)
        if state_deg.shape != (STATE_DIM,) or not np.all(np.isfinite(state_deg)):
            raise V181LiveError("current robot state must be finite [6]")
        if replace_pending:
            self.runtime.cancel_pending_proposal()
        elif self.runtime.has_pending_chunk:
            raise V181LiveError("an unexecuted V18.1 proposal is already pending")

        effective_lower, effective_upper = _inward_only_action_bounds(
            state_deg,
            self.action_lower_position,
            self.action_upper_position,
        )
        self._install_action_bounds(effective_lower, effective_upper)
        self.effective_action_lower_position = effective_lower
        self.effective_action_upper_position = effective_upper

        started = time.perf_counter()
        tokens = self._tokens(frames)
        valid = torch.ones(2, dtype=torch.bool, device=self.device)
        proposal = self.runtime.observe_and_propose(
            tokens,
            valid,
            torch.from_numpy(state_deg / JOINT_SCALE).to(self.device),
        )
        commands = (
            proposal.actor_output.commands[0].float().cpu().numpy()
            * JOINT_SCALE
        ).astype(np.float32)
        predicted_path = (
            proposal.prediction.predicted_q_path[0].float().cpu().numpy()
            * JOINT_SCALE
        ).astype(np.float32)
        increments = np.diff(
            np.concatenate((state_deg[None], commands), axis=0), axis=0
        )
        maximum_per_joint = np.max(np.abs(increments), axis=0)
        if commands.shape != (5, 6) or not np.all(np.isfinite(commands)):
            raise V181LiveError("V18.1 produced an invalid command chunk")
        if np.any(maximum_per_joint > STEP_CAP_DEG + 1.0e-4):
            raise V181LiveError(
                "V18.1 output violates its fixed per-joint step caps"
            )

        # The public command must already be a fixed point of the model's own
        # sequential projector; runtime never adds a second policy or fallback.
        normalized_state = torch.from_numpy(state_deg / JOINT_SCALE)[None]
        normalized_commands = torch.from_numpy(commands / JOINT_SCALE)[None]
        projected = project_v181_action_trajectory(
            normalized_state,
            normalized_commands,
            step_cap=self.runtime.model.actor.step_cap.cpu(),
            action_lower=self.runtime.model.actor.action_lower.cpu(),
            action_upper=self.runtime.model.actor.action_upper.cpu(),
        )
        fixed_point_error = float(
            torch.max(torch.abs(projected - normalized_commands)).item()
            * JOINT_SCALE
        )
        if fixed_point_error > 1.0e-4:
            raise V181LiveError("V18.1 command is not a projector fixed point")
        record = {
            "action_source": self.action_source,
            "commands_deg": commands,
            "predicted_q_path_deg": predicted_path,
            "maximum_step_per_joint_deg": maximum_per_joint,
            "projector_fixed_point_error_deg": fixed_point_error,
            "effective_action_lower": effective_lower,
            "effective_action_upper": effective_upper,
            "inference_s": time.perf_counter() - started,
            "memory_chunks": int(
                self.runtime.memory.history_valid.sum().item()
            ),
        }
        self.last_proposal = record
        return record

    def complete(
        self,
        frames: Mapping[str, np.ndarray],
        measured_q_path_deg: np.ndarray,
        *,
        delta_t_s: float,
    ) -> dict[str, Any]:
        path_deg = np.asarray(measured_q_path_deg, dtype=np.float32)
        if path_deg.shape != (PATH_STEPS, STATE_DIM) or not np.all(
            np.isfinite(path_deg)
        ):
            raise V181LiveError("measured path must be finite [6,6]")
        tokens = self._tokens(frames)
        innovation = self.runtime.complete_and_reobserve(
            tokens,
            torch.ones(2, dtype=torch.bool, device=self.device),
            torch.from_numpy(path_deg / JOINT_SCALE).to(self.device),
            delta_t_s=float(delta_t_s),
        )
        record = {
            "visual_innovation_rms": float(
                innovation.visual.float().square().mean().sqrt().item()
            ),
            "state_innovation_rms_deg": float(
                innovation.state.float().square().mean().sqrt().item()
                * JOINT_SCALE
            ),
            "memory_chunks": int(
                self.runtime.memory.history_valid.sum().item()
            ),
        }
        self.last_completion = record
        return record

    def status(self) -> dict[str, Any]:
        return {
            "version": "18.1",
            "tip_free": True,
            "view_order": list(VIEW_ORDER),
            "checkpoint": self.checkpoint,
            "training_phase": self.training_phase,
            "seed": self.seed,
            "actor_conditioning": self.runtime.actor_conditioning,
            "deployment_action_lower": self.action_lower_position,
            "deployment_action_upper": self.action_upper_position,
            "effective_action_lower": self.effective_action_lower_position,
            "effective_action_upper": self.effective_action_upper_position,
            "pending_chunk": self.runtime.has_pending_chunk,
            "memory_chunks": int(
                self.runtime.memory.history_valid.sum().item()
            ),
            "last_proposal": self.last_proposal,
            "last_completion": self.last_completion,
        }

    def reset(self) -> None:
        """Start a new H0 clock after an external/manual robot operation."""

        self.runtime.reset()
        self.last_proposal = None
        self.last_completion = None


__all__ = ["JOINT_SCALE", "STEP_CAP_DEG", "V181LiveError", "V181LivePolicy"]
