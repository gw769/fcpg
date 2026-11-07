"""Runtime bridge for V20.3.2's single-update dual-view observer."""
from __future__ import annotations

from dataclasses import replace

import torch

from .prod_v20_3_runtime import V203Runtime
from .prod_v20_3 import _time
from .prod_v20_3_2 import V2032Core
from .prod_v20_timing import FISH_COMMAND_PERIOD_S


class V2032Runtime(V203Runtime):
    def __init__(self, core, *args, **kwargs):
        if not isinstance(core, V2032Core):
            raise TypeError("V20.3.2 runtime requires V2032Core")
        super().__init__(core, *args, **kwargs)

    def plan(self, time_deltas):
        if not torch.allclose(time_deltas, torch.full_like(time_deltas, FISH_COMMAND_PERIOD_S), atol=1e-7, rtol=0):
            raise ValueError("runtime command period differs from the fish training clock")
        return super().plan(time_deltas)

    @torch.inference_mode()
    def refresh_endpoint(self, endpoint):
        """Refresh from a raw current image and the actual prior exactly once."""
        with self.lock:
            self.lease_check()
            if self.executing or self.pending is not None:
                raise RuntimeError("cannot refresh while a five-command transaction is pending")
            endpoint.validate(self.max_endpoint_skew_s)
            if self.state is None:
                return self.observe_initial(endpoint)
            current_time = endpoint.readback_time_s.reshape(-1)
            raw = self.core.observer.measurement(
                endpoint.rgb, endpoint.dino_tokens, endpoint.view_valid,
            )
            self.state = self.core.refresh_from_measurement(
                self.state, raw, endpoint.q_position, endpoint.view_valid,
                current_time, geometry=self.geometry,
            )
            self.stopped = False
            self.pending_at = None
            return self.status()

    def status(self):
        result = super().status()
        result["version"] = "20.3.2"
        result["contract"] = "proprioceptive-pair-observer-v3-dual-view-attention"
        result["policy_command_period_s"] = FISH_COMMAND_PERIOD_S
        if self.state is not None:
            feedback_time = self.state.feedback_endpoint_time_s
            result["feedback"] = dict(
                from_completed_transaction=feedback_time is not None,
                source_endpoint_time_s=None if feedback_time is None else feedback_time.detach().cpu().tolist(),
                current_observation_time_s=self.state.endpoint_time_s.detach().cpu().tolist(),
                age_at_observation_s=None if feedback_time is None else
                    (self.state.endpoint_time_s - feedback_time).detach().cpu().tolist())
        observation = None if self.state is None else self.state.observation
        if self.state is not None and self.geometry is not None:
            active = self.readback_state if self.readback_state is not None else self.state
            with torch.no_grad():
                nominal, valid = self.geometry.project(active.q_position)
            result["geometry_prior_fm_px"] = nominal.detach().cpu().tolist()
            result["geometry_prior_valid"] = valid.detach().cpu().tolist()
        if observation is not None:
            result.setdefault("observer", {})["update_applied"] = bool(
                getattr(observation, "update_applied", False)
            )
            result["observer"]["spatial_update_valid"] = getattr(
                observation, "spatial_update_valid", None
            )
            if isinstance(result["observer"]["spatial_update_valid"], torch.Tensor):
                result["observer"]["spatial_update_valid"] = result["observer"]["spatial_update_valid"].detach().cpu().tolist()
            result["observer"]["out_of_support"] = getattr(observation, "out_of_support", None)
            if isinstance(result["observer"]["out_of_support"], torch.Tensor):
                result["observer"]["out_of_support"] = result["observer"]["out_of_support"].detach().cpu().tolist()
            raw_center = getattr(observation, "raw_column_center_px", None)
            result["observer"]["raw_column_center_px"] = None if raw_center is None else raw_center.detach().cpu().tolist()
            raw_pair = getattr(observation, "raw_column_fm_px", None)
            result["observer"]["raw_column_fm_px"] = None if raw_pair is None else raw_pair.detach().cpu().tolist()
            result["observer"]["raw_visual_is_verified_localization"] = False
        return result


__all__ = ["V2032Runtime"]
