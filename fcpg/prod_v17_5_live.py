"""Real dual-view adapter for the complete Prod-V17.5 stack."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from fcpg.prod_v17_5 import (
    ANCHORS,
    PATH_STEPS,
    STATE_DIM,
    VIEW_ORDER,
    V175Controller,
    project_v175_action_trajectory,
)
from fcpg.prod_v17_5_runtime import V175RuntimeController
from fcpg.prod_v17_5_training import V175AffineProjectionProvider, load_component
from fcpg.prod_v18_1_training import extract_dinov2_tokens, load_frozen_dinov2


# Public V17.5 tensors are already in LeRobot calibrated-position units.
JOINT_SCALE = 1.0
STEP_CAP_DEG = np.asarray([4.0, 4.0, 4.0, 4.0, 4.0, 8.0], dtype=np.float32)


class V175LiveError(RuntimeError):
    """Raised before malformed live data can reach V17.5 or the robot."""


def _inward_only_action_bounds(
    current_q_position: np.ndarray,
    base_lower_position: np.ndarray,
    base_upper_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
        raise V175LiveError("action bounds must be finite ordered six-vectors")
    effective_lower = lower.copy()
    effective_upper = upper.copy()
    effective_lower[:5] = np.minimum(effective_lower[:5], current[:5])
    effective_upper[:5] = np.maximum(effective_upper[:5], current[:5])
    return effective_lower, effective_upper


class V175LivePolicy:
    """Own DINO extraction and the actual-observation V17.5 runtime clock."""

    def __init__(
        self,
        *,
        fcpg_checkpoint: str | Path,
        lewm_checkpoint: str | Path,
        actor_checkpoint: str | Path,
        dino_repository: str | Path,
        dino_weights: str | Path,
        real_calibration: str | Path,
        device: torch.device | str,
        action_lower_position: np.ndarray,
        action_upper_position: np.ndarray,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise V175LiveError("CUDA was requested but is unavailable")
        model = V175Controller()
        payloads = {
            "fcpg": load_component(
                fcpg_checkpoint, model.fcpg, component="fcpg", device=self.device
            ),
            "lewm": load_component(
                lewm_checkpoint, model.lewm, component="lewm", device=self.device
            ),
            "actor": load_component(
                actor_checkpoint, model.actor, component="actor", device=self.device
            ),
        }
        lower = np.asarray(action_lower_position, dtype=np.float32)
        upper = np.asarray(action_upper_position, dtype=np.float32)
        if (
            lower.shape != (STATE_DIM,)
            or upper.shape != (STATE_DIM,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)
        ):
            raise V175LiveError("deployment action bounds are invalid")
        self.action_lower_position = lower.copy()
        self.action_upper_position = upper.copy()
        self.effective_action_lower_position = lower.copy()
        self.effective_action_upper_position = upper.copy()
        self.runtime = V175RuntimeController(model, device=self.device)
        self.projection = V175AffineProjectionProvider(real_calibration, model.fcpg)
        self._install_action_bounds(lower, upper)
        self.backbone = load_frozen_dinov2(
            dino_repository, dino_weights, device=self.device
        )
        self.action_source = "v17_5_grounded_anchor_innovation_controller"
        self.checkpoints = {
            name: str(Path(path).expanduser().resolve())
            for name, path in (
                ("fcpg", fcpg_checkpoint),
                ("lewm", lewm_checkpoint),
                ("actor", actor_checkpoint),
            )
        }
        self.seed = int(payloads["actor"].get("seed", -1))
        self.last_proposal: dict[str, Any] | None = None
        self.last_completion: dict[str, Any] | None = None

    def _install_action_bounds(
        self, lower_position: np.ndarray, upper_position: np.ndarray
    ) -> None:
        actor = self.runtime.model.actor
        with torch.no_grad():
            actor.action_lower.copy_(
                torch.from_numpy(lower_position).to(
                    actor.action_lower.device, dtype=actor.action_lower.dtype
                )
            )
            actor.action_upper.copy_(
                torch.from_numpy(upper_position).to(
                    actor.action_upper.device, dtype=actor.action_upper.dtype
                )
            )

    def _endpoint_tensors(
        self, frames: Mapping[str, np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if set(frames) != set(VIEW_ORDER):
            raise V175LiveError(f"live frames must contain exactly {list(VIEW_ORDER)}")
        ordered: list[np.ndarray] = []
        for view in VIEW_ORDER:
            frame = np.asarray(frames[view])
            if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
                raise V175LiveError(f"{view} must be uint8 [224,224,3]")
            ordered.append(frame)
        images = np.stack(ordered)
        dino = extract_dinov2_tokens(
            self.backbone, images, device=self.device, batch_size=2
        )
        rgb = torch.from_numpy(np.ascontiguousarray(images)).to(self.device)
        rgb = rgb.permute(0, 3, 1, 2)
        tokens = torch.from_numpy(dino).to(self.device, dtype=torch.float32)
        return rgb, tokens

    def propose(
        self,
        frames: Mapping[str, np.ndarray],
        current_q_deg: np.ndarray,
        *,
        replace_pending: bool = False,
    ) -> dict[str, Any]:
        state_deg = np.asarray(current_q_deg, dtype=np.float32)
        if state_deg.shape != (STATE_DIM,) or not np.all(np.isfinite(state_deg)):
            raise V175LiveError("current robot state must be finite [6]")
        if replace_pending:
            self.runtime.cancel_pending_proposal()
        elif self.runtime.has_pending_chunk:
            raise V175LiveError("an unexecuted V17.5 proposal is already pending")

        effective_lower, effective_upper = _inward_only_action_bounds(
            state_deg, self.action_lower_position, self.action_upper_position
        )
        self._install_action_bounds(effective_lower, effective_upper)
        self.effective_action_lower_position = effective_lower
        self.effective_action_upper_position = effective_upper

        started = time.perf_counter()
        rgb, tokens = self._endpoint_tensors(frames)
        valid = torch.ones(2, dtype=torch.bool, device=self.device)
        nominal, nominal_valid, nominal_jacobian = self.projection(
            torch.from_numpy(state_deg)[None].to(self.device)
        )
        proposal = self.runtime.observe_and_propose(
            rgb,
            tokens,
            valid,
            torch.from_numpy(state_deg).to(self.device),
            nominal_keypoints_px=nominal[0],
            nominal_valid=nominal_valid[0],
            nominal_jacobian_px=nominal_jacobian[0],
        )
        commands = proposal.actor_output.commands[0].float().cpu().numpy().astype(np.float32)
        predicted_path = (
            proposal.prediction.predicted_q_path[0].float().cpu().numpy().astype(np.float32)
        )
        increments = np.diff(
            np.concatenate((state_deg[None], commands), axis=0), axis=0
        )
        maximum_per_joint = np.max(np.abs(increments), axis=0)
        if commands.shape != (5, 6) or not np.all(np.isfinite(commands)):
            raise V175LiveError("V17.5 produced an invalid command chunk")
        if np.any(maximum_per_joint > STEP_CAP_DEG + 1.0e-4):
            raise V175LiveError("V17.5 output violates fixed per-joint step caps")

        normalized_state = torch.from_numpy(state_deg)[None]
        normalized_commands = torch.from_numpy(commands)[None]
        actor = self.runtime.model.actor
        projected = project_v175_action_trajectory(
            normalized_state,
            normalized_commands,
            step_cap=actor.step_cap.cpu(),
            action_lower=actor.action_lower.cpu(),
            action_upper=actor.action_upper.cpu(),
        )
        fixed_point_error = float(
            torch.max(torch.abs(projected - normalized_commands)).item()
        )
        if fixed_point_error > 1.0e-4:
            raise V175LiveError("V17.5 command is not a projector fixed point")

        probability = proposal.measurement.correspondence_maps["local"].float()
        entropy = -(
            probability.clamp_min(1.0e-8)
            * probability.clamp_min(1.0e-8).log()
        ).sum(dim=-1) / np.log(probability.shape[-1])
        record = {
            "action_source": self.action_source,
            "commands_deg": commands,
            "predicted_q_path_deg": predicted_path,
            "maximum_step_per_joint_deg": maximum_per_joint,
            "projector_fixed_point_error_deg": fixed_point_error,
            "effective_action_lower": effective_lower,
            "effective_action_upper": effective_upper,
            "inference_s": time.perf_counter() - started,
            "memory_chunks": int(self.runtime.memory.history_valid.sum().item()),
            "anchor_positions_px": (
                proposal.measurement.corrected_keypoints_px[0].float().cpu().numpy()
            ),
            "fm_positions_px": (
                proposal.measurement.fm_px[0].float().cpu().numpy()
            ),
            "midpoint_px": (
                proposal.measurement.midpoint_px[0].float().cpu().numpy()
            ),
            "anchor_valid": proposal.measurement.anchor_valid[0].cpu().numpy(),
            "anchor_posterior_entropy": entropy[0].float().cpu().numpy(),
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
            raise V175LiveError("measured path must be finite [6,6]")
        if not np.isfinite(delta_t_s) or float(delta_t_s) <= 0.0:
            raise V175LiveError("transition duration must be positive and finite")
        rgb, tokens = self._endpoint_tensors(frames)
        nominal, nominal_valid, nominal_jacobian = self.projection(
            torch.from_numpy(path_deg[-1:]).to(self.device)
        )
        innovation = self.runtime.complete_and_reobserve(
            rgb,
            tokens,
            torch.ones(2, dtype=torch.bool, device=self.device),
            torch.from_numpy(path_deg).to(self.device),
            nominal_keypoints_px=nominal[0],
            nominal_valid=nominal_valid[0],
            nominal_jacobian_px=nominal_jacobian[0],
        )
        record = {
            "visual_innovation_rms": float(
                innovation.visual.float().square().mean().sqrt().item()
            ),
            "state_innovation_rms_deg": float(
                innovation.state.float().square().mean().sqrt().item()
            ),
            "memory_chunks": int(self.runtime.memory.history_valid.sum().item()),
        }
        self.last_completion = record
        return record

    def status(self) -> dict[str, Any]:
        return {
            "version": "17.5",
            "grounded_anchor_state": True,
            "control_uses_point_coordinates": False,
            "view_order": list(VIEW_ORDER),
            "checkpoints": self.checkpoints,
            "seed": self.seed,
            "deployment_action_lower": self.action_lower_position,
            "deployment_action_upper": self.action_upper_position,
            "effective_action_lower": self.effective_action_lower_position,
            "effective_action_upper": self.effective_action_upper_position,
            "pending_chunk": self.runtime.has_pending_chunk,
            "memory_chunks": int(self.runtime.memory.history_valid.sum().item()),
            "last_proposal": self.last_proposal,
            "last_completion": self.last_completion,
        }

    def reset(self) -> None:
        self.runtime.reset()
        self.last_proposal = None
        self.last_completion = None


__all__ = ["JOINT_SCALE", "STEP_CAP_DEG", "V175LiveError", "V175LivePolicy"]
