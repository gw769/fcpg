"""V20.3.2 dual-view proprioceptive feedback core.

Wrist and column observations have different contracts.  The wrist spatial
pair is an analytic readout of the current gripper state.  The column spatial
pair is one learned update conditioned on the current q/readback history and
the current image field.  ARM/LeWM and Actor retain the V20.3 architecture and
their source-local SIGReg training contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json

import torch

from .prod_v20_1_belief import JointPairBelief
from .prod_v20_policy import append_completed, empty_history
from .prod_v20_3 import V203ActualState, V203Config, V203Core, _time
from .prod_v20_3_2_observer import V2032ObserverOutput, V2032PairObserver


VERSION = "20.3.2"
CONTRACT = "proprioceptive-pair-observer-v3-dual-view-attention"
POSITION_UNIT = "lerobot_calibrated_position"


@dataclass(frozen=True)
class V2032ActualState(V203ActualState):
    feedback_endpoint_time_s: torch.Tensor | None = None
    feedback_observation: object = None
    feedback_view_valid: torch.Tensor | None = None


@dataclass(frozen=True)
class V2032Config(V203Config):
    seed: int = 20000
    observer_attention_dim: int = 192
    observer_attention_heads: int = 6
    observer_attention_layers: int = 2
    visibility_evidence_threshold: float = 0.5
    max_column_residual_px: float = 16.0

    def __post_init__(self):
        super().__post_init__()
        if (self.observer_attention_dim, self.observer_attention_heads,
                self.observer_attention_layers) != (192, 6, 2):
            raise ValueError("V20.3.2 observer attention is fixed at 192D/6 heads/2 layers")
        if not 0.0 < self.visibility_evidence_threshold < 1.0:
            raise ValueError("V20.3.2 visibility threshold must be in (0,1)")
        if self.max_column_residual_px != 16.0:
            raise ValueError("V20.3.2 fixes the Column visual center residual at 16 px")


class V2032Core(V203Core):
    CHECKPOINT_VERSION = VERSION
    CHECKPOINT_CONTRACT = CONTRACT
    CHECKPOINT_POSITION_UNIT = POSITION_UNIT
    INITIAL_CONSUMES_RAW_MEASUREMENT = True

    def __init__(self, config: V2032Config = V2032Config()):
        if not isinstance(config, V2032Config):
            raise TypeError("V20.3.2 requires V2032Config")
        super().__init__(config)
        self.observer = V2032PairObserver(
            attention_dim=config.observer_attention_dim,
            attention_heads=config.observer_attention_heads,
            attention_layers=config.observer_attention_layers,
            evidence_threshold=config.visibility_evidence_threshold,
            max_column_residual_px=config.max_column_residual_px,
        )

    def _evidence(self, observation, view_valid):
        valid = torch.as_tensor(view_valid, device=observation.belief.state_mean.device,
                                dtype=torch.bool)
        update_valid = getattr(observation, "spatial_update_valid", None)
        if update_valid is None:
            probability = observation.belief.visible_probability.mean((-1, -2))
            update_valid = probability > self.config.visibility_evidence_threshold
        else:
            update_valid = torch.as_tensor(update_valid, device=valid.device, dtype=torch.bool)
        out = getattr(observation, "out_of_support", None)
        if out is not None:
            update_valid = update_valid & ~torch.as_tensor(out, device=valid.device, dtype=torch.bool)
        # Wrist position is owned by the tool-mounted q6 geometry.  Its image
        # visibility can reach Actor diagnostics, but it must never become a
        # learned pixel correction in the persistent memory.
        update_valid = update_valid.clone()
        update_valid[:, 0] = False
        return valid & update_valid

    def _update_raw(self, measurement, prior, view_valid, q_position, history, q_available=None,
                    mechanical_prior=None):
        if isinstance(measurement, V2032ObserverOutput) and measurement.update_applied:
            raise ValueError("core needs raw image features; observation already updated")
        return self.observer.update(
            measurement, prior, view_valid, q_position=q_position,
            actual_history=history, q_available=q_available, mechanical_prior=mechanical_prior,
        )

    def initial(self, rgb, dino_tokens, q_position, view_valid, endpoint_time_s, geometry=None):
        raw = self.observer.measurement(rgb, dino_tokens, view_valid)
        return self.initial_from_measurement(raw, q_position, view_valid,
                                             endpoint_time_s, geometry=geometry)

    def initial_from_measurement(self, measurement, q_position, view_valid,
                                 endpoint_time_s, *, geometry=None):
        if q_position.shape != (len(q_position), 6) or not torch.isfinite(q_position).all():
            raise ValueError("current q must be finite calibrated positions [B,6]")
        time = _time(endpoint_time_s, q_position)
        valid = torch.as_tensor(view_valid, device=q_position.device, dtype=torch.bool)
        history = empty_history(len(q_position), device=q_position.device, dtype=q_position.dtype)
        prior = geometry.make_prior(q_position) if geometry is not None else None
        observation = self._update_raw(measurement, prior, valid, q_position, history,
                                       q_available=True)
        grounding, innovation = self.control_encoding(
            observation, None, None, None, self.q_loss_scale, view_valid=valid
        )
        evidence = self._evidence(observation, valid)
        memory = None
        if geometry is not None:
            memory = geometry.capture_correction(
                observation.belief.detach(), q_position.detach(), view_valid=evidence
            )
        return V2032ActualState(
            observation, q_position, history, grounding, innovation, time, valid,
            correction_memory=memory,
            correction_time_s=time[:, None].expand(-1, 2).clone(),
            localization_time_s=time,
            localization_prior=prior,
        )

    def refresh_from_measurement(self, previous, measurement, q_position, view_valid,
                                 endpoint_time_s, *, geometry=None, readback_std=None):
        """Use one fresh dual-view image at a measured endpoint."""
        time = _time(endpoint_time_s, q_position)
        if previous.localization_time_s is None or bool((time < previous.localization_time_s).any()):
            raise ValueError("refresh endpoint cannot travel backwards in time")
        valid = torch.as_tensor(view_valid, device=q_position.device, dtype=torch.bool)
        prior = self._actual_prior(previous, q_position, time, geometry, readback_std)
        observation = self._update_raw(
            measurement, prior, valid, q_position, previous.history,
            q_available=True, mechanical_prior=geometry.make_prior(q_position) if geometry is not None else None,
        )
        evidence = self._evidence(observation, valid)
        memory, memory_time = previous.correction_memory, previous.correction_time_s
        if geometry is not None:
            memory = geometry.capture_correction(
                observation.belief.detach(), q_position.detach(), view_valid=evidence,
                previous_memory=memory,
            )
            accepted = getattr(memory, "update_mask", evidence)
            memory_time = torch.where(accepted, time[:, None], memory_time)
        grounding, null_innovation = self.control_encoding(
            observation, None, None, None, self.q_loss_scale, view_valid=valid
        )
        # A new image is not a newly completed command. Preserve the last
        # transaction's realized error and forecast; only current G changes.
        innovation = (previous.innovation_tracks if previous.feedback_endpoint_time_s is not None
                      else null_innovation)
        return replace(
            previous,
            observation=observation,
            q_position=q_position.detach().clone(),
            grounding_tracks=grounding,
            innovation_tracks=innovation,
            endpoint_time_s=time.detach().clone(),
            view_valid=valid,
            correction_memory=memory,
            correction_time_s=memory_time,
            localization_time_s=time,
            localization_prior=prior,
        )

    def complete_from_measurement(
        self, previous, prediction, measurement, measured_q_path, view_valid,
        endpoint_time_s, actual_time_deltas=None, *, geometry=None, readback_std=None,
    ):
        if (not torch.allclose(prediction.source_q_position, previous.q_position,
                               atol=1e-4, rtol=0)
                or not torch.allclose(prediction.source_time_s, previous.endpoint_time_s,
                                      atol=1e-6, rtol=0)):
            raise ValueError("prediction belongs to a different actual endpoint")
        if (measured_q_path.shape != (len(previous.q_position), 6, 6)
                or not torch.isfinite(measured_q_path).all()
                or not torch.allclose(measured_q_path[:, 0], previous.q_position,
                                      atol=1e-4, rtol=0)):
            raise ValueError("actual path must be finite aligned q0..q5")
        time = _time(endpoint_time_s, previous.q_position)
        elapsed = prediction.time_deltas if actual_time_deltas is None else actual_time_deltas
        if (elapsed.shape != prediction.time_deltas.shape or not torch.isfinite(elapsed).all()
                or not bool((elapsed > 0).all())):
            raise ValueError("actual time intervals invalid")
        if not torch.allclose(time - previous.endpoint_time_s,
                              elapsed.double().sum(-1), atol=1e-5, rtol=0):
            raise ValueError("actual endpoint clock differs from readback intervals")

        actual_q = measured_q_path[:, -1]
        with torch.no_grad():
            actual_prior = self._actual_prior(previous, actual_q, time, geometry, readback_std)
        # The just-completed command/readback is already history at the new
        # camera endpoint and therefore belongs in the Column attention input.
        history = append_completed(
            previous.history, prediction.commands_position, measured_q_path,
            previous.observation.scene_tokens.mean(-2), elapsed,
        )
        valid = torch.as_tensor(view_valid, device=actual_q.device, dtype=torch.bool)
        observation = self._update_raw(
            measurement, actual_prior, valid, actual_q, history,
            q_available=True, mechanical_prior=geometry.make_prior(actual_q) if geometry is not None else None,
        )
        grounding, innovation = self.control_encoding(
            observation, prediction.prior, measured_q_path,
            prediction.predicted_q_path, self.q_loss_scale, view_valid=valid,
        )
        evidence = self._evidence(observation, valid)
        memory, memory_time = previous.correction_memory, previous.correction_time_s
        if geometry is not None:
            memory = geometry.capture_correction(
                observation.belief.detach(), actual_q.detach(), view_valid=evidence,
                previous_memory=memory,
            )
            accepted = getattr(memory, "update_mask", evidence)
            memory_time = torch.where(accepted, time[:, None], memory_time)
        return V2032ActualState(
            observation, actual_q, history, grounding, innovation, time, valid,
            prediction.prior, measured_q_path, prediction.predicted_q_path,
            memory, memory_time, time, actual_prior,
            feedback_endpoint_time_s=time.detach().clone(),
            feedback_observation=observation,
            feedback_view_valid=valid.detach().clone(),
        )

    def encode_innovation(self, state, mode="full"):
        # Interventions must use the SAME actual transaction as factual E,
        # including after current-image refreshes have changed G.
        if state.feedback_observation is None:
            return super().encode_innovation(state, mode)
        return self.control_encoding(state.feedback_observation, state.prior,
            state.measured_q_path, state.predicted_q_path, self.q_loss_scale,
            innovation_mode=mode, view_valid=state.feedback_view_valid)[1]


def validate_v2032_metadata(payload: dict, config: V2032Config, *, stage: str | None = None):
    expected = {
        "version": VERSION,
        "contract": CONTRACT,
        "position_unit": POSITION_UNIT,
        "config": asdict(config),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"V20.3.2 checkpoint {key} differs")
    if stage is not None and payload.get("stage") != stage:
        raise ValueError("V20.3.2 checkpoint stage differs")
    if not payload.get("metrics", {}).get("complete"):
        raise ValueError("V20.3.2 checkpoint is incomplete")
    return payload


def load_trained_runtime_v2032(output_directory, task_embedding, *, device,
                               physical_check, lease_check, calibration_path=None,
                               **kwargs):
    """Load only a complete V20.3.2 fish checkpoint.

    The loader constructs a fresh core and checks the complete metadata before
    opening the runtime.  A V20.3/V20.3.1 checkpoint therefore cannot silently
    enter the dual-view service.
    """
    from .prod_v20_3_2_pipeline import V2032Schedule
    from .prod_v20_3_training import load_checkpoint
    from .robot_kinematic_graph import DifferentiableRobotKinematics, so101_robot_spec
    from .prod_v20_3_geometry import V203FittedMechanicalProjection
    from .prod_v20_3_2_runtime import V2032Runtime

    root = Path(output_directory)
    config_data = json.loads((root / "config.json").read_text())
    if config_data.get("version") != VERSION or config_data.get("contract") != CONTRACT:
        raise ValueError("output is not a V20.3.2 run")
    from .prod_v20_timing import FISH_COMMAND_PERIOD_S
    for key, expected in (("smoke", False), ("runtime_feedback_contract", "last-completed-feedback-v1"),
                          ("refit_history_schedule", "independent-view-suffix-v2"),
                          ("policy_command_period_s", FISH_COMMAND_PERIOD_S)):
        if config_data.get(key) != expected:
            raise ValueError(f"V20.3.2 deployment {key} differs from the corrected contract")
    config = V2032Config(**config_data["model"])
    schedule = V2032Schedule(**config_data.get("schedule", {}))
    core = V2032Core(config).to(device)
    payload = load_checkpoint(root / "fish" / "final.pt", core, stage="fish", schedule=schedule)
    validate_v2032_metadata(payload, config, stage="fish")
    if payload.get("step") != schedule.fish_updates:
        raise ValueError("V20.3.2 fish checkpoint is incomplete")
    if calibration_path is None:
        calibration_path = config_data["sources"]["real_calibration"]
    calibration = json.loads(Path(calibration_path).read_text())
    if calibration.get("fits") != config_data.get("real_projection_fits"):
        raise ValueError("current camera projection differs from V20.3.2 training")
    kinematics = DifferentiableRobotKinematics(so101_robot_spec()).to(device).eval().requires_grad_(False)
    geometry = V203FittedMechanicalProjection(calibration_path, device=device, kinematics=kinematics)
    return V2032Runtime(core, task_embedding.to(device), geometry=geometry,
                        physical_check=physical_check, lease_check=lease_check, **kwargs)


__all__ = [
    "VERSION", "CONTRACT", "POSITION_UNIT", "V2032Config", "V2032Core", "V2032ActualState",
    "validate_v2032_metadata", "load_trained_runtime_v2032",
]
