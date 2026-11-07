"""Two-Hz post-chunk runtime clock for Prod-V18.1.

The wrapper owns no camera or robot.  It enforces exactly one order:
actual observation -> proposal/prediction -> completed five-command chunk ->
actual reobservation -> transition innovation.  Only actual spatial observer
tokens enter persistent memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fcpg.prod_v18_1 import (
    OBSERVER_DIM,
    OBSERVER_SLOTS,
    PATH_STEPS,
    STATE_DIM,
    VIEWS,
    VIT_DIM,
    VIT_PATCHES,
    V181ActorOutput,
    V181Innovation,
    V181LeWMPrediction,
    V181Measurement,
    V181Memory,
    V181TipFreeController,
    compute_v181_innovation,
    empty_v181_innovation,
    predicted_observer_slots,
)


class V181RuntimeError(RuntimeError):
    """Raised when execution/reobservation ordering is violated."""


ACTOR_CONDITIONING_MODES = ("full", "actual_history")


def _without_actor_innovation(memory: V181Memory) -> V181Memory:
    """Keep factual endpoint/action history while nulling learned innovations."""

    return V181Memory(
        actual_observer_slots=memory.actual_observer_slots,
        actual_view_valid=memory.actual_view_valid,
        completed_commands=memory.completed_commands,
        measured_paths=memory.measured_paths,
        visual_innovations=torch.zeros_like(memory.visual_innovations),
        state_innovations=torch.zeros_like(memory.state_innovations),
        innovation_valid=memory.innovation_valid,
        path_valid=memory.path_valid,
        delta_t_s=memory.delta_t_s,
        history_valid=memory.history_valid,
    )


@dataclass(frozen=True)
class V181PendingChunk:
    source_measurement: V181Measurement
    source_view_valid: torch.Tensor
    source_q: torch.Tensor
    actor_output: V181ActorOutput
    prediction: V181LeWMPrediction
    input_innovation: V181Innovation
    input_predicted_slots: torch.Tensor
    input_prediction_valid: torch.Tensor


@dataclass(frozen=True)
class V181RuntimeProposal:
    actor_output: V181ActorOutput
    prediction: V181LeWMPrediction


class V181RuntimeController:
    """Batch-one state machine for supervised physical execution."""

    def __init__(
        self,
        model: V181TipFreeController,
        *,
        device: torch.device | str,
        actor_conditioning: str = "full",
    ) -> None:
        if actor_conditioning not in ACTOR_CONDITIONING_MODES:
            raise V181RuntimeError(
                f"actor_conditioning must be one of {ACTOR_CONDITIONING_MODES}"
            )
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.actor_conditioning = actor_conditioning
        self.reset()

    @property
    def has_pending_chunk(self) -> bool:
        return self._pending is not None

    def reset(self) -> None:
        self.memory = V181Memory.empty(1, device=self.device)
        self._pending: V181PendingChunk | None = None
        self._carry_innovation: V181Innovation | None = None
        self._carry_predicted_slots: torch.Tensor | None = None
        self._carry_prediction_valid: torch.Tensor | None = None

    def _endpoint_inputs(
        self,
        vit_tokens: torch.Tensor,
        view_valid: torch.Tensor,
        current_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.as_tensor(vit_tokens, device=self.device)
        valid = torch.as_tensor(view_valid, device=self.device)
        state = torch.as_tensor(current_q, device=self.device)
        if tokens.shape == (VIEWS, VIT_PATCHES, VIT_DIM):
            tokens = tokens[None]
        if valid.shape == (VIEWS,):
            valid = valid[None]
        if state.shape == (STATE_DIM,):
            state = state[None]
        if tokens.shape != (1, VIEWS, VIT_PATCHES, VIT_DIM):
            raise V181RuntimeError("runtime ViT tokens must be [1,2,256,384]")
        if valid.shape != (1, VIEWS) or valid.dtype != torch.bool:
            raise V181RuntimeError("runtime view_valid must be bool [1,2]")
        if state.shape != (1, STATE_DIM) or not state.is_floating_point():
            raise V181RuntimeError("runtime current_q must be floating [1,6]")
        if not bool(torch.isfinite(tokens).all() and torch.isfinite(state).all()):
            raise V181RuntimeError("runtime endpoint contains non-finite values")
        return tokens, valid, state

    @torch.inference_mode()
    def observe_and_propose(
        self,
        vit_tokens: torch.Tensor,
        view_valid: torch.Tensor,
        current_q: torch.Tensor,
    ) -> V181RuntimeProposal:
        if self._pending is not None:
            raise V181RuntimeError(
                "complete and reobserve the previous chunk before proposing again"
            )
        tokens, valid, state = self._endpoint_inputs(
            vit_tokens, view_valid, current_q
        )
        measurement = self.model.measure(tokens, valid)
        if self._carry_innovation is None:
            innovation = empty_v181_innovation(
                1, device=self.device, dtype=measurement.local_observer_slots.dtype
            )
            predicted = torch.zeros_like(measurement.local_observer_slots)
            prediction_valid = torch.zeros_like(valid)
        else:
            innovation = self._carry_innovation
            if self._carry_predicted_slots is None or self._carry_prediction_valid is None:
                raise V181RuntimeError("carried transition state is incomplete")
            predicted = self._carry_predicted_slots
            # A newly valid real view always supersedes the predicted-source channel.
            prediction_valid = self._carry_prediction_valid & ~valid
        actor_innovation = innovation
        actor_predicted = predicted
        actor_prediction_valid = prediction_valid
        actor_memory = self.memory
        if self.actor_conditioning == "actual_history":
            actor_innovation = empty_v181_innovation(
                1, device=self.device, dtype=measurement.local_observer_slots.dtype
            )
            actor_predicted = torch.zeros_like(predicted)
            actor_prediction_valid = torch.zeros_like(prediction_valid)
            actor_memory = _without_actor_innovation(self.memory)
        actor_output = self.model.act(
            measurement,
            valid,
            actor_innovation,
            actor_predicted,
            actor_prediction_valid,
            state,
            actor_memory,
        )
        prediction = self.model.predict(
            measurement.local_observer_slots,
            valid,
            state,
            actor_output.commands,
        )
        self._pending = V181PendingChunk(
            source_measurement=measurement,
            source_view_valid=valid.clone(),
            source_q=state.clone(),
            actor_output=actor_output,
            prediction=prediction,
            input_innovation=innovation,
            input_predicted_slots=predicted,
            input_prediction_valid=prediction_valid,
        )
        self._carry_innovation = None
        self._carry_predicted_slots = None
        self._carry_prediction_valid = None
        return V181RuntimeProposal(actor_output=actor_output, prediction=prediction)

    def cancel_pending_proposal(self) -> None:
        """Discard an unexecuted proposal without consuming transition context.

        A hardware command is refreshed from the observation taken immediately
        before transport.  Replacing the earlier UI/preview proposal must not
        advance memory or lose the innovation produced by the last completed
        chunk.
        """

        pending = self._pending
        if pending is None:
            return
        self._carry_innovation = pending.input_innovation
        self._carry_predicted_slots = pending.input_predicted_slots
        self._carry_prediction_valid = pending.input_prediction_valid
        self._pending = None

    @torch.inference_mode()
    def complete_and_reobserve(
        self,
        actual_vit_tokens: torch.Tensor,
        actual_view_valid: torch.Tensor,
        measured_q_path: torch.Tensor,
        *,
        delta_t_s: float,
        path_valid: torch.Tensor | None = None,
    ) -> V181Innovation:
        pending = self._pending
        if pending is None:
            raise V181RuntimeError("no proposed chunk is waiting for completion")
        tokens = torch.as_tensor(actual_vit_tokens, device=self.device)
        valid = torch.as_tensor(actual_view_valid, device=self.device)
        path = torch.as_tensor(measured_q_path, device=self.device)
        if tokens.shape == (VIEWS, VIT_PATCHES, VIT_DIM):
            tokens = tokens[None]
        if valid.shape == (VIEWS,):
            valid = valid[None]
        if path.shape == (PATH_STEPS, STATE_DIM):
            path = path[None]
        if tokens.shape != (1, VIEWS, VIT_PATCHES, VIT_DIM):
            raise V181RuntimeError("reobserved ViT tokens must be [1,2,256,384]")
        if valid.shape != (1, VIEWS) or valid.dtype != torch.bool:
            raise V181RuntimeError("reobserved view_valid must be bool [1,2]")
        if path.shape != (1, PATH_STEPS, STATE_DIM):
            raise V181RuntimeError("measured path must be [1,6,6]")
        if not bool(torch.isfinite(tokens).all() and torch.isfinite(path).all()):
            raise V181RuntimeError("reobserved transition contains non-finite values")
        if not 0.0 < float(delta_t_s) < 5.0:
            raise V181RuntimeError("post-chunk elapsed time must be in (0,5) seconds")
        if path_valid is None:
            valid_path = torch.ones(
                1, PATH_STEPS, dtype=torch.bool, device=self.device
            )
        else:
            valid_path = torch.as_tensor(path_valid, device=self.device)
            if valid_path.shape == (PATH_STEPS,):
                valid_path = valid_path[None]
            if valid_path.shape != (1, PATH_STEPS) or valid_path.dtype != torch.bool:
                raise V181RuntimeError("path_valid must be bool [1,6]")
        if not torch.allclose(path[:, 0], pending.source_q, atol=2.0e-3, rtol=0.0):
            raise V181RuntimeError("measured q0 differs from the proposed source state")

        target_measurement = self.model.measure(tokens, valid)
        innovation = compute_v181_innovation(
            pending.source_measurement.local_observer_slots,
            pending.source_view_valid,
            target_measurement.local_observer_slots,
            valid,
            path,
            pending.prediction,
            path_valid=valid_path,
        )
        self.memory = self.memory.append(
            source_actual_slots=pending.source_measurement.local_observer_slots,
            source_view_valid=pending.source_view_valid,
            completed_commands=pending.actor_output.commands,
            measured_q_path=path,
            realized_innovation=innovation,
            delta_t_s=torch.tensor(
                [float(delta_t_s)], device=self.device, dtype=path.dtype
            ),
        )
        self._carry_innovation = innovation
        self._carry_predicted_slots = predicted_observer_slots(
            pending.source_measurement.local_observer_slots,
            pending.prediction,
        )
        self._carry_prediction_valid = pending.source_view_valid & ~valid
        self._pending = None
        return innovation

    def actual_memory_state(self) -> V181Memory:
        """Return actual-observation memory; transient predictions are excluded."""

        self.memory.validate()
        return self.memory


__all__ = [
    "ACTOR_CONDITIONING_MODES",
    "V181PendingChunk",
    "V181RuntimeController",
    "V181RuntimeError",
    "V181RuntimeProposal",
]
