"""Actual-endpoint runtime clock for the Prod-V17.5 controller.

The runtime mirrors the training contract: each completed chunk is followed by
one real dual-view endpoint measurement.  Ordered structural anchors remove the
old prediction-based slot matching, and only actual measurements enter memory.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from fcpg.prod_v17_5 import (
    ANCHOR_DIM,
    ANCHORS,
    COMMAND_STEPS,
    HISTORY_CHUNKS,
    IMAGE_SIZE,
    PATH_STEPS,
    STATE_DIM,
    VIEWS,
    VIT_DIM,
    VIT_PATCHES,
    V175ActorOutput,
    V175Controller,
    V175FCPGMemory,
    V175Innovation,
    V175LeWMPrediction,
    V175Measurement,
    V175Memory,
    compute_v175_innovation,
)


class V175RuntimeError(RuntimeError):
    """Raised before an invalid endpoint transition reaches V17.5."""


@dataclass(frozen=True)
class V175PendingChunk:
    source_measurement: V175Measurement
    source_anchor_valid: torch.Tensor
    source_q: torch.Tensor
    actor_output: V175ActorOutput
    prediction: V175LeWMPrediction
    input_innovation: V175Innovation


@dataclass(frozen=True)
class V175RuntimeProposal:
    measurement: V175Measurement
    actor_output: V175ActorOutput
    prediction: V175LeWMPrediction


def empty_v175_innovation(
    batch: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> V175Innovation:
    return V175Innovation(
        visual=torch.zeros(
            batch, VIEWS, ANCHORS, ANCHOR_DIM, device=device, dtype=dtype
        ),
        state=torch.zeros(
            batch, PATH_STEPS, STATE_DIM, device=device, dtype=dtype
        ),
        anchor_valid=torch.zeros(
            batch, VIEWS, ANCHORS, device=device, dtype=torch.bool
        ),
        path_valid=torch.zeros(
            batch, PATH_STEPS, device=device, dtype=torch.bool
        ),
    )


def _append_actual_transition(
    memory: V175Memory,
    *,
    actual_anchor_tokens: torch.Tensor,
    actual_anchor_valid: torch.Tensor,
    completed_commands: torch.Tensor,
    measured_path: torch.Tensor,
    innovation: V175Innovation,
) -> V175Memory:
    """Append one actual transition, dropping the oldest after four."""

    memory.validate()
    count = int(memory.history_valid[0].sum().item())
    values: dict[str, torch.Tensor] = {
        field.name: getattr(memory, field.name).clone() for field in fields(memory)
    }
    if count == HISTORY_CHUNKS:
        for name, value in values.items():
            value[:, :-1].copy_(value[:, 1:].clone())
        slot = HISTORY_CHUNKS - 1
    else:
        slot = count
    values["actual_anchor_tokens"][:, slot] = actual_anchor_tokens
    values["actual_anchor_valid"][:, slot] = actual_anchor_valid
    values["completed_commands"][:, slot] = completed_commands
    values["measured_paths"][:, slot] = measured_path
    values["visual_innovations"][:, slot] = innovation.visual
    values["state_innovations"][:, slot] = innovation.state
    values["innovation_valid"][:, slot] = innovation.anchor_valid
    values["path_valid"][:, slot] = innovation.path_valid
    values["history_valid"][:, slot] = True
    result = V175Memory(**values)
    result.validate()
    return result


class V175RuntimeController:
    """Batch-one state machine for supervised physical execution."""

    def __init__(
        self,
        model: V175Controller,
        *,
        device: torch.device | str,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.reset()

    @property
    def has_pending_chunk(self) -> bool:
        return self._pending is not None

    def reset(self) -> None:
        self.memory = V175Memory.empty(1, device=self.device)
        self._pending: V175PendingChunk | None = None
        self._carry_innovation: V175Innovation | None = None
        self._fcpg_memory: V175FCPGMemory | None = None

    def _fcpg_history(
        self,
        *,
        pending_commands: torch.Tensor | None = None,
        pending_path: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        commands = self.memory.completed_commands.clone()
        paths = self.memory.measured_paths.clone()
        valid = self.memory.history_valid.clone()
        if pending_commands is None:
            return commands, paths, valid
        if pending_path is None:
            raise V175RuntimeError("pending commands require a measured path")
        count = int(valid[0].sum().item())
        if count == HISTORY_CHUNKS:
            commands[:, :-1].copy_(commands[:, 1:].clone())
            paths[:, :-1].copy_(paths[:, 1:].clone())
            valid[:, :-1].copy_(valid[:, 1:].clone())
            slot = HISTORY_CHUNKS - 1
        else:
            slot = count
        commands[:, slot] = pending_commands
        paths[:, slot] = pending_path
        valid[:, slot] = True
        return commands, paths, valid

    def _geometry_inputs(
        self,
        *,
        camera_intrinsics: torch.Tensor | None,
        camera_from_reference: torch.Tensor | None,
        image_transform: torch.Tensor | None,
        nominal_keypoints_px: torch.Tensor | None,
        nominal_valid: torch.Tensor | None,
        nominal_jacobian_px: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        def batched(value: torch.Tensor | None, shape: tuple[int, ...], name: str):
            if value is None:
                return None
            tensor = torch.as_tensor(value, device=self.device)
            if tensor.shape == shape:
                tensor = tensor[None]
            if tensor.shape != (1, *shape):
                raise V175RuntimeError(f"{name} must be {list(shape)} or batch-one")
            return tensor

        intrinsic = batched(camera_intrinsics, (VIEWS, 3, 3), "camera intrinsics")
        transform = batched(
            camera_from_reference, (VIEWS, 4, 4), "camera transform"
        )
        image = batched(image_transform, (VIEWS, 3, 3), "image transform")
        nominal = batched(
            nominal_keypoints_px, (VIEWS, ANCHORS, 2), "nominal keypoints"
        )
        valid = batched(nominal_valid, (VIEWS, ANCHORS), "nominal validity")
        jacobian = batched(
            nominal_jacobian_px,
            (VIEWS, ANCHORS, 2, STATE_DIM),
            "nominal Jacobian",
        )
        if nominal is None and (intrinsic is None or transform is None):
            raise V175RuntimeError(
                "runtime needs pinhole camera geometry or calibrated nominal/Jacobian"
            )
        if nominal is not None and valid is None:
            raise V175RuntimeError("calibrated nominal requires its validity mask")
        return intrinsic, transform, image, nominal, valid, jacobian

    def _endpoint_inputs(
        self,
        rgb: torch.Tensor,
        dino_tokens: torch.Tensor,
        view_valid: torch.Tensor,
        current_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = torch.as_tensor(rgb, device=self.device)
        tokens = torch.as_tensor(dino_tokens, device=self.device)
        valid = torch.as_tensor(view_valid, device=self.device)
        state = torch.as_tensor(current_q, device=self.device)
        if image.shape == (VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = image[None]
        if tokens.shape == (VIEWS, VIT_PATCHES, VIT_DIM):
            tokens = tokens[None]
        if valid.shape == (VIEWS,):
            valid = valid[None]
        if state.shape == (STATE_DIM,):
            state = state[None]
        if image.shape != (1, VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE):
            raise V175RuntimeError("runtime RGB must be [1,2,3,224,224]")
        if tokens.shape != (1, VIEWS, VIT_PATCHES, VIT_DIM):
            raise V175RuntimeError("runtime DINO tokens must be [1,2,256,384]")
        if valid.shape != (1, VIEWS) or valid.dtype != torch.bool:
            raise V175RuntimeError("runtime view_valid must be bool [1,2]")
        if state.shape != (1, STATE_DIM) or not state.is_floating_point():
            raise V175RuntimeError("runtime current_q must be floating [1,6]")
        if not bool(torch.isfinite(tokens).all() and torch.isfinite(state).all()):
            raise V175RuntimeError("runtime endpoint contains non-finite values")
        return image, tokens, valid, state

    @torch.inference_mode()
    def observe_and_propose(
        self,
        rgb: torch.Tensor,
        dino_tokens: torch.Tensor,
        view_valid: torch.Tensor,
        current_q: torch.Tensor,
        *,
        camera_intrinsics: torch.Tensor | None = None,
        camera_from_reference: torch.Tensor | None = None,
        image_transform: torch.Tensor | None = None,
        nominal_keypoints_px: torch.Tensor | None = None,
        nominal_valid: torch.Tensor | None = None,
        nominal_jacobian_px: torch.Tensor | None = None,
    ) -> V175RuntimeProposal:
        if self._pending is not None:
            raise V175RuntimeError(
                "complete and reobserve the previous chunk before proposing again"
            )
        image, tokens, valid, state = self._endpoint_inputs(
            rgb, dino_tokens, view_valid, current_q
        )
        geometry = self._geometry_inputs(
            camera_intrinsics=camera_intrinsics,
            camera_from_reference=camera_from_reference,
            image_transform=image_transform,
            nominal_keypoints_px=nominal_keypoints_px,
            nominal_valid=nominal_valid,
            nominal_jacobian_px=nominal_jacobian_px,
        )
        commands, paths, history_valid = self._fcpg_history()
        measurement = self.model.fcpg(
            image,
            tokens,
            valid,
            state,
            geometry[0],
            geometry[1],
            image_transform=geometry[2],
            completed_commands_deg=commands,
            measured_q_paths_deg=paths,
            history_valid=history_valid,
            previous_memory=self._fcpg_memory,
            cached_nominal_keypoints_px=geometry[3],
            cached_nominal_valid=geometry[4],
            cached_nominal_jacobian_px=geometry[5],
        )
        self._fcpg_memory = measurement.next_memory
        anchor_valid = measurement.anchor_valid
        innovation = self._carry_innovation
        if innovation is None:
            innovation = empty_v175_innovation(
                1, device=self.device, dtype=measurement.anchor_tokens.dtype
            )
        actor_innovation = V175Innovation(
            visual=innovation.visual * anchor_valid[..., None],
            state=innovation.state,
            anchor_valid=innovation.anchor_valid & anchor_valid,
            path_valid=innovation.path_valid,
        )
        actor_output = self.model.actor(
            measurement.scene_tokens,
            measurement.anchor_tokens,
            anchor_valid,
            actor_innovation.visual,
            actor_innovation.state,
            actor_innovation.anchor_valid,
            actor_innovation.path_valid,
            state,
            self.memory,
        )
        prediction = self.model.lewm(
            measurement.anchor_tokens,
            anchor_valid,
            state,
            actor_output.commands,
        )
        self._pending = V175PendingChunk(
            source_measurement=measurement,
            source_anchor_valid=anchor_valid.clone(),
            source_q=state.clone(),
            actor_output=actor_output,
            prediction=prediction,
            input_innovation=innovation,
        )
        self._carry_innovation = None
        return V175RuntimeProposal(
            measurement=measurement,
            actor_output=actor_output,
            prediction=prediction,
        )

    def cancel_pending_proposal(self) -> None:
        pending = self._pending
        if pending is None:
            return
        self._carry_innovation = pending.input_innovation
        self._pending = None

    @torch.inference_mode()
    def complete_and_reobserve(
        self,
        actual_rgb: torch.Tensor,
        actual_dino_tokens: torch.Tensor,
        actual_view_valid: torch.Tensor,
        measured_q_path: torch.Tensor,
        *,
        path_valid: torch.Tensor | None = None,
        camera_intrinsics: torch.Tensor | None = None,
        camera_from_reference: torch.Tensor | None = None,
        image_transform: torch.Tensor | None = None,
        nominal_keypoints_px: torch.Tensor | None = None,
        nominal_valid: torch.Tensor | None = None,
        nominal_jacobian_px: torch.Tensor | None = None,
    ) -> V175Innovation:
        pending = self._pending
        if pending is None:
            raise V175RuntimeError("no proposed chunk is waiting for completion")
        path = torch.as_tensor(measured_q_path, device=self.device)
        if path.shape == (PATH_STEPS, STATE_DIM):
            path = path[None]
        if path.shape != (1, PATH_STEPS, STATE_DIM):
            raise V175RuntimeError("measured q path must be [1,6,6]")
        if not bool(torch.isfinite(path).all()):
            raise V175RuntimeError("measured q path contains non-finite values")
        # Hardware is read again immediately before transport. Permit only the
        # configured sub-step settlement scale of encoder jitter/drift; a
        # genuinely different source pose still fails closed.
        if not torch.allclose(path[:, 0], pending.source_q, atol=2.5e-1, rtol=0.0):
            raise V175RuntimeError("measured q0 differs from the proposed source state")
        image, tokens, valid, _ = self._endpoint_inputs(
            actual_rgb,
            actual_dino_tokens,
            actual_view_valid,
            path[:, -1],
        )
        valid_path = (
            torch.ones(1, PATH_STEPS, dtype=torch.bool, device=self.device)
            if path_valid is None
            else torch.as_tensor(path_valid, device=self.device)
        )
        if valid_path.shape == (PATH_STEPS,):
            valid_path = valid_path[None]
        if valid_path.shape != (1, PATH_STEPS) or valid_path.dtype != torch.bool:
            raise V175RuntimeError("path_valid must be bool [1,6]")

        geometry = self._geometry_inputs(
            camera_intrinsics=camera_intrinsics,
            camera_from_reference=camera_from_reference,
            image_transform=image_transform,
            nominal_keypoints_px=nominal_keypoints_px,
            nominal_valid=nominal_valid,
            nominal_jacobian_px=nominal_jacobian_px,
        )
        commands, paths, fcpg_history_valid = self._fcpg_history(
            pending_commands=pending.actor_output.commands,
            pending_path=path,
        )
        target = self.model.fcpg(
            image,
            tokens,
            valid,
            path[:, -1],
            geometry[0],
            geometry[1],
            image_transform=geometry[2],
            completed_commands_deg=commands,
            measured_q_paths_deg=paths,
            history_valid=fcpg_history_valid,
            previous_memory=self._fcpg_memory,
            cached_nominal_keypoints_px=geometry[3],
            cached_nominal_valid=geometry[4],
            cached_nominal_jacobian_px=geometry[5],
        )
        self._fcpg_memory = target.next_memory
        target_valid = target.anchor_valid
        innovation = compute_v175_innovation(
            target.anchor_tokens,
            target_valid,
            path,
            pending.prediction,
            path_valid=valid_path,
        )
        self.memory = _append_actual_transition(
            self.memory,
            actual_anchor_tokens=target.anchor_tokens,
            actual_anchor_valid=target_valid,
            completed_commands=pending.actor_output.commands,
            measured_path=path,
            innovation=innovation,
        )
        self._carry_innovation = innovation
        self._pending = None
        return innovation

    def actual_memory_state(self) -> V175Memory:
        self.memory.validate()
        return self.memory


__all__ = [
    "V175PendingChunk",
    "V175RuntimeController",
    "V175RuntimeError",
    "V175RuntimeProposal",
    "empty_v175_innovation",
]
