"""V20.3: actual proprioception localizes; ARM forecasts; vision corrects once.

No hardware I/O, online weight update, new policy head, or pixel action target.
The public joint unit is LeRobot calibrated position, NOT degrees.
"""
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from .prod_v20_1 import (
    V201Config, V201Core, V201ActualState, V201Prediction, V201ControlEncoding,
    POSITION_UNIT, VIEW_ORDER, INNOVATION_MODES, OBSERVATION_CONTRACT,
    detach_actual, belief_statistics, matched_prior_statistics,
)
from .prod_v20_1_arm import V201LeWM, ARMProjector
from .prod_v20_policy import V20Actor, empty_history, append_completed
from .prod_v20_3_observer import V203PairObserver

VERSION = "20.3.0"
CONTRACT = "proprioceptive-pair-observer-v1-arm-sigreg"


@dataclass(frozen=True)
class V203Config(V201Config):
    # Retain V20.1 seed/capacity/schedule for a matched mechanism comparison.
    state_representation: str = "mechanical-reference-correction-and-joint-pair"

    def __post_init__(self):
        super().__post_init__()
        if self.state_representation != "mechanical-reference-correction-and-joint-pair":
            raise ValueError("V20.3 requires actual-readback correction memory")


@dataclass(frozen=True)
class V203ActualState(V201ActualState):
    correction_memory: Any = None
    correction_time_s: torch.Tensor | None = None  # [B,V], per-view last update
    localization_time_s: torch.Tensor | None = None
    localization_prior: Any = None  # prior recomputed from ACTUAL q, not forecast


V203Prediction = V201Prediction


def _time(value, q):
    time = torch.as_tensor(value, device=q.device, dtype=torch.float64)
    if time.shape != (len(q),) or not torch.isfinite(time).all():
        raise ValueError("endpoint/readback time requires finite seconds [B]")
    return time


class V203Core(V201Core):
    def __init__(self, config=V203Config()):
        nn.Module.__init__(self)
        if not isinstance(config, V203Config):
            raise TypeError("V20.3 cannot reinterpret an old model configuration")
        self.config = config
        self.observer = V203PairObserver()
        self.lewm = V201LeWM()
        self.actor = V20Actor()
        self.control_encoding = V201ControlEncoding()
        self.register_buffer("q_loss_scale", torch.ones(6))
        self.arm_sigreg_projector = ARMProjector()

    def initial(self, rgb, dino_tokens, q_position, view_valid, endpoint_time_s, geometry=None):
        prior = geometry.make_prior(q_position) if geometry is not None else None
        observation = self.observer(rgb, dino_tokens, view_valid, prior=prior)
        return self.initial_from_measurement(observation, q_position, view_valid,
                                             endpoint_time_s, geometry=geometry)

    def initial_from_measurement(self, observation, q_position, view_valid, endpoint_time_s, *, geometry=None):
        """Input is the already-once-updated H0 observation, never fuse again."""
        if q_position.shape != (len(q_position), 6) or not torch.isfinite(q_position).all():
            raise ValueError("current q must be finite calibrated positions [B,6]")
        time = _time(endpoint_time_s, q_position)
        valid = view_valid.bool()
        grounding, innovation = self.control_encoding(observation, None, None, None,
                                                       self.q_loss_scale, view_valid=valid)
        evidence = valid & (observation.belief.visible_probability.amax((-1, -2)) > 0)
        memory = None if geometry is None else geometry.capture_correction(
            observation.belief.detach(), q_position.detach(), view_valid=evidence)
        return V203ActualState(observation, q_position,
            empty_history(len(q_position), device=q_position.device, dtype=q_position.dtype),
            grounding, innovation, time, valid, correction_memory=memory,
            correction_time_s=time[:, None].expand(-1, 2).clone(), localization_time_s=time)

    def _actual_prior(self, state, q_actual, time, geometry, readback_std=None):
        if state.localization_time_s is None or (time < state.localization_time_s).any():
            raise ValueError("actual readback cannot travel backwards in time")
        if geometry is not None and state.correction_memory is not None:
            return geometry.project_correction(state.correction_memory, q_actual,
                time[:, None] - state.correction_time_s, readback_std=readback_std)
        if geometry is not None:
            return geometry.propagate_actual(state.observation.belief.detach(), state.q_position,
                q_actual, time - state.localization_time_s, readback_std=readback_std)
        # No camera geometry -> cannot invent a q-to-pixel map. Missing geometry
        # remains explicit, with time-scaled uncertainty, not confidence in a
        # stationary point. This path is NOT evidence of occlusion tracking.
        belief = state.observation.belief.detach()
        elapsed = (time - state.localization_time_s).to(belief.state_mean)
        return replace(belief, state_covariance=belief.state_covariance +
            8. * elapsed[:, None, None, None, None] * torch.eye(4, device=q_actual.device))

    @torch.no_grad()
    def advance_readback(self, state, q_position, readback_time_s, geometry=None, readback_std=None):
        """Localization preview only: no image, transition, forecast or BC step.

        Runtime retains the original transaction state separately. Never sample
        old pixels at new coordinates and pretend that was a new observation.
        """
        if q_position.shape != state.q_position.shape or not torch.isfinite(q_position).all():
            raise ValueError("actual readback must be finite and match source joints")
        time = _time(readback_time_s, q_position)
        if (time <= state.localization_time_s).any():
            raise ValueError("actual readback timestamps must increase")
        prior = self._actual_prior(state, q_position, time, geometry, readback_std)
        valid = torch.zeros_like(state.view_valid)
        blank = replace(state.observation, belief=prior,
            component_features=torch.zeros_like(state.observation.component_features),
            scene_tokens=torch.zeros_like(state.observation.scene_tokens),
            pooling_map=torch.zeros_like(state.observation.pooling_map))
        grounding, innovation = self.control_encoding(blank, state.prior,
            state.measured_q_path, state.predicted_q_path, self.q_loss_scale, view_valid=valid)
        return replace(state, observation=blank, q_position=q_position.detach().clone(),
            grounding_tracks=grounding, innovation_tracks=innovation, view_valid=valid,
            localization_time_s=time, localization_prior=prior)

    def act(self, state, task_embedding, innovation_mode=None):
        if not torch.equal(state.localization_time_s, state.endpoint_time_s):
            raise ValueError("readback preview is not a synchronized action endpoint")
        return super().act(state, task_embedding, innovation_mode)

    def predict(self, state, commands_position, time_deltas, geometry=None, *, return_arm_hidden=False):
        if not torch.equal(state.localization_time_s, state.endpoint_time_s):
            raise ValueError("ARM forecast must start at a synchronized endpoint")
        hidden = None
        if return_arm_hidden:
            output, hidden = self.lewm.forward_with_hidden(state.q_position, commands_position,
                                                           state.history, time_deltas)
        else:
            output = self.lewm(state.q_position, commands_position, state.history, time_deltas)
        with torch.no_grad():
            future_time = state.endpoint_time_s + time_deltas.double().sum(-1)
            if geometry is not None and state.correction_memory is not None:
                prior = geometry.project_forecast(state.correction_memory,
                    output.predicted_q_path[:, -1].detach(),
                    future_time[:, None] - state.correction_time_s,
                    forecast_std=output.q_uncertainty[:, -1].detach())
            else:
                prior = self._actual_prior(state, output.predicted_q_path[:, -1].detach(),
                    future_time, geometry)
        return V203Prediction(commands_position, output.predicted_q_path, output.q_uncertainty,
            prior, state.q_position, state.endpoint_time_s, time_deltas, hidden)

    def complete_from_measurement(self, previous, prediction, measurement, measured_q_path,
            view_valid, endpoint_time_s, actual_time_deltas=None, *, geometry=None, readback_std=None):
        if not torch.allclose(prediction.source_q_position, previous.q_position, atol=1e-4, rtol=0) or not torch.allclose(prediction.source_time_s, previous.endpoint_time_s, atol=1e-6, rtol=0):
            raise ValueError("prediction belongs to a different actual endpoint")
        if measured_q_path.shape != (len(previous.q_position), 6, 6) or not torch.isfinite(measured_q_path).all() or not torch.allclose(measured_q_path[:, 0], previous.q_position, atol=1e-4, rtol=0):
            raise ValueError("actual path must be finite aligned q0..q5")
        time = _time(endpoint_time_s, previous.q_position)
        elapsed = prediction.time_deltas if actual_time_deltas is None else actual_time_deltas
        if elapsed.shape != prediction.time_deltas.shape or not torch.isfinite(elapsed).all() or not (elapsed > 0).all():
            raise ValueError("actual time intervals invalid")
        if not torch.allclose(time - previous.endpoint_time_s, elapsed.double().sum(-1), atol=1e-5, rtol=0):
            raise ValueError("actual endpoint clock differs from readback intervals")
        # CRITICAL: use actual q for localization. Forecast is retained solely
        # as the independent expected outcome for execution innovation.
        actual_q = measured_q_path[:, -1]
        with torch.no_grad():
            actual_prior = self._actual_prior(previous, actual_q, time, geometry, readback_std)
        observation = self.observer.assimilate(measurement, actual_prior, view_valid)
        grounding, innovation = self.control_encoding(observation, prediction.prior,
            measured_q_path, prediction.predicted_q_path, self.q_loss_scale, view_valid=view_valid)
        memory, memory_time = previous.correction_memory, previous.correction_time_s
        if geometry is not None:
            # No threshold: exactly zero spatial evidence is a missing update.
            observed = view_valid.bool() & (measurement.belief.visible_probability.amax((-1, -2)) > 0)
            # V20.3.1's Column observer exposes a bounded residual contract.
            # A saturated visual estimate is retained for diagnosis but must
            # not overwrite the episode correction memory with a large jump.
            limit = getattr(self.observer, "max_column_residual_px", None)
            if limit is not None:
                nominal = geometry.make_prior(actual_q)
                residual = (observation.belief.state_mean[:, 1] - nominal.state_mean[:, 1]).abs().amax((-1, -2))
                observed[:, 1] = observed[:, 1] & (residual <= float(limit) + 1e-6)
            memory = geometry.capture_correction(observation.belief.detach(), actual_q.detach(),
                view_valid=observed, previous_memory=memory)
            accepted = getattr(memory, "update_mask", observed)
            memory_time = torch.where(accepted, time[:, None], memory_time)
        history = append_completed(previous.history, prediction.commands_position, measured_q_path,
                                    previous.observation.scene_tokens.mean(-2), elapsed)
        return V203ActualState(observation, actual_q, history, grounding, innovation, time,
            view_valid.bool(), prediction.prior, measured_q_path, prediction.predicted_q_path,
            memory, memory_time, time, actual_prior)
