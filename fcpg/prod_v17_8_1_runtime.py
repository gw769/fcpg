"""Settled-clock runtime adapter for Prod-V17.8.1.

The hardware service supplies six command-clock readbacks (q0..q5), then waits
for the observation camera and samples one additional settled joint state at
the same endpoint as that image.  V17.8 conflated these two clocks.  This
adapter preserves both: LeWM innovation uses the timed path, while the next
Actor/LeWM source state is the settled image-synchronous state.

No command filtering, target coordinate, phase rule or trajectory rewrite is
performed here.
"""

from __future__ import annotations

import torch

from fcpg.prod_v17_8 import (
    COMMAND_STEPS,
    PATH_STEPS,
    STATE_DIM,
    pool_scene_tokens,
)
from fcpg.prod_v17_8_runtime import (
    V178FilterUpdate,
    V178RuntimeController,
    V178RuntimeError,
    V178RuntimeProposal,
    _batch_one,
    _clone_assigned,
    _clone_measurement,
    _clone_update,
)
from fcpg.prod_v17_8_1_clock import (
    DEFAULT_SOURCE_ATOL_POSITION,
    V1781ClockError,
    V1781SettledHistory,
    V1781StateInnovation,
    append_settled_history,
    empty_settled_history,
    make_settled_observation,
    state_innovation_from_observation,
)


# Keep one catchable runtime error family across inherited validation helpers
# and the settled-clock additions.
V1781RuntimeError = V178RuntimeError


class V1781RuntimeController(V178RuntimeController):
    """V17.8 runtime with a distinct post-q5 observation endpoint."""

    def _clear_actual_state(self) -> None:
        super()._clear_actual_state()
        self._settled_history = empty_settled_history(
            1,
            device=self.device,
            source_atol_position=self.source_state_atol_position,
        )
        self._latest_settled_innovation: V1781StateInnovation | None = None

    def actual_settled_history(self) -> V1781SettledHistory:
        value = self._settled_history
        return V1781SettledHistory(
            raw=self.actual_history(),
            settled_q=value.settled_q.detach().clone(),
            settle_delta=value.settle_delta.detach().clone(),
            timed_duration_s=value.timed_duration_s.detach().clone(),
            settle_duration_s=value.settle_duration_s.detach().clone(),
            source_atol_position=value.source_atol_position,
        )

    def latest_settled_state_innovation(self) -> V1781StateInnovation:
        value = self._latest_settled_innovation
        if value is None:
            raise V1781RuntimeError("no completed settled observation exists")
        return V1781StateInnovation(
            timed_path_residual=value.timed_path_residual.detach().clone(),
            timed_endpoint_residual=value.timed_endpoint_residual.detach().clone(),
            settled_endpoint_residual=(
                value.settled_endpoint_residual.detach().clone()
            ),
            settle_delta=value.settle_delta.detach().clone(),
            timed_duration_s=value.timed_duration_s.detach().clone(),
            settle_duration_s=value.settle_duration_s.detach().clone(),
        )

    @torch.inference_mode()
    def propose_from_latest(
        self, current_q: torch.Tensor | None = None
    ) -> V178RuntimeProposal:
        """Plan only from the q sampled with the retained endpoint image.

        A later hardware readback without a new image is not a new visual
        endpoint.  Accepting it as ``current_q`` would pair stale pixels with
        a different robot state.  Callers that observe a changed q must first
        acquire a new synchronized image/measurement transaction.
        """

        self._reject_existing_pending()
        if current_q is not None:
            if self._latest_q is None:
                raise V1781RuntimeError("no actual endpoint is available")
            supplied = _batch_one(
                current_q,
                (STATE_DIM,),
                device=self.device,
                name="current_q",
                floating=True,
            ).to(self._latest_q)
            if not bool(
                torch.allclose(
                    supplied,
                    self._latest_q,
                    atol=DEFAULT_SOURCE_ATOL_POSITION,
                    rtol=0.0,
                )
            ):
                raise V1781RuntimeError(
                    "current_q has no synchronized retained image; reobserve first"
                )
        return super().propose_from_latest(None)

    @torch.inference_mode()
    def complete_and_reobserve(
        self,
        actual_dino_tokens: torch.Tensor,
        actual_view_valid: torch.Tensor,
        executed_commands: torch.Tensor,
        measured_q_path: torch.Tensor,
        *,
        generation: int,
        actual_rgb: torch.Tensor,
        actual_endpoint_q: torch.Tensor | None = None,
        settled_q: torch.Tensor | None = None,
        timed_duration_s: float | torch.Tensor | None = None,
        settle_duration_s: float | torch.Tensor | None = None,
        path_valid: torch.Tensor | None = None,
    ) -> V178FilterUpdate:
        """Commit one command-clock path and its later actual observation."""

        pending = self._require_pending(generation)
        executed = _batch_one(
            executed_commands,
            (COMMAND_STEPS, STATE_DIM),
            device=self.device,
            name="executed commands",
            floating=True,
        ).to(pending.source_q)
        if not bool(
            torch.allclose(
                executed,
                pending.actor_output.commands.to(executed),
                atol=self.command_atol_position,
                rtol=0.0,
            )
        ):
            raise V1781RuntimeError(
                "executed commands differ from pending Actor proposal"
            )
        path = _batch_one(
            measured_q_path,
            (PATH_STEPS, STATE_DIM),
            device=self.device,
            name="measured q path",
            floating=True,
        ).to(pending.source_q)
        if path_valid is not None:
            valid_path = _batch_one(
                path_valid,
                (PATH_STEPS,),
                device=self.device,
                name="path_valid",
                boolean=True,
            )
            if not bool(valid_path.all()):
                raise V1781RuntimeError(
                    "all six measured q path samples must be valid"
                )
        # Settled history and execution use one numerical calibrated-position
        # metric.  Do not let an optional quantized cell encoder approve a q0
        # that the actual-history clock would subsequently reject.
        if not bool(
            torch.allclose(
                pending.source_q.to(path),
                path[:, 0],
                atol=self.source_state_atol_position,
                rtol=0.0,
            )
        ):
            raise V1781RuntimeError(
                "measured q0 differs from proposed source state"
            )
        if settled_q is None:
            settled_q = actual_endpoint_q
        elif actual_endpoint_q is not None:
            raise V1781RuntimeError(
                "provide settled_q or actual_endpoint_q, not both"
            )
        if settled_q is None:
            raise V1781RuntimeError("settled_q is required")
        if timed_duration_s is None or settle_duration_s is None:
            raise V1781RuntimeError(
                "timed_duration_s and settle_duration_s are required"
            )
        endpoint = _batch_one(
            settled_q,
            (STATE_DIM,),
            device=self.device,
            name="settled_q",
            floating=True,
        ).to(pending.source_q)
        try:
            observation = make_settled_observation(
                path,
                endpoint,
                timed_duration_s=timed_duration_s,
                settle_duration_s=settle_duration_s,
            )
            state_innovation = state_innovation_from_observation(
                pending.prediction,
                observation,
                source_atol_position=self.source_state_atol_position,
            )
        except V1781ClockError as error:
            raise V1781RuntimeError(str(error)) from error

        tokens, valid, final_q, image = self._endpoint_inputs(
            actual_dino_tokens,
            actual_view_valid,
            observation.settled_q,
            actual_rgb,
        )
        measurement = self.model.measure(tokens, valid, image)
        self._check_measurement(measurement, valid)
        assigned = self.model.assign_identity(
            measurement, pending.prediction.prior_log_belief
        )
        self._check_assigned(assigned, valid)
        # Belief and q-path innovations retain the command-clock target.  The
        # additional settled residual above is separately auditable and the
        # settled q becomes the next source state.
        update = self.model.update(assigned, pending.prediction, path)
        self._check_filter_update(
            update, expected_valid=valid, expected_prediction=True
        )

        source_scene = pool_scene_tokens(
            pending.source_measurement.scene_tokens,
            pending.source_measurement.view_valid,
        )
        history_tolerance = max(
            self.source_state_atol_position,
            self._settled_history.source_atol_position,
        )
        try:
            settled_history = append_settled_history(
                self._settled_history,
                executed,
                observation,
                source_scene,
                source_atol_position=history_tolerance,
            )
        except V1781ClockError as error:
            raise V1781RuntimeError(str(error)) from error
        readback_delta = path[:, 1:] - path[:, :-1]
        readback_error = path[:, 1:] - executed.to(path)

        # Finish every allocation before mutating controller state.  The
        # assignments below are reference swaps and cannot leave a half-cloned
        # actual endpoint if an allocation fails.
        staged_measurement = _clone_measurement(measurement)
        staged_assigned = _clone_assigned(assigned)
        staged_update = _clone_update(update)
        staged_q = final_q.detach().clone()
        staged_readback_delta = readback_delta.detach().clone()
        staged_readback_error = readback_error.detach().clone()
        staged_history = settled_history
        staged_raw_history = settled_history.raw
        staged_innovation = state_innovation

        self._latest_measurement = staged_measurement
        self._latest_assigned = staged_assigned
        self._latest_update = staged_update
        self._latest_q = staged_q
        self._recent_readback_deltas = staged_readback_delta
        self._recent_readback_errors = staged_readback_error
        self._settled_history = staged_history
        self._history = staged_raw_history
        self._latest_settled_innovation = staged_innovation
        self._completed_chunks += 1
        self._pending = None
        return update


__all__ = ["V1781RuntimeController", "V1781RuntimeError"]
