"""V20.3 model-only runtime with transaction-isolated actual-readback previews.

The external hardware owner still owns serial writes, physical limits, leases
and motor cancellation. This module never opens hardware or sends commands.
Readback previews do not become new image endpoints or Actor planning states.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from .prod_v20_runtime import Endpoint, V20Runtime
from .prod_v20_3 import V203Core, V203Config, VERSION, CONTRACT, POSITION_UNIT


class V203Runtime(V20Runtime):
    def __init__(self, core, *args, **kwargs):
        if not isinstance(core, V203Core):
            raise TypeError("V20.3 runtime cannot reinterpret another model version")
        super().__init__(core, *args, **kwargs)
        self.readback_state = None
        self.transaction_state = None
        self._transaction_source_q = None
        self._transaction_source_time = None

    @torch.inference_mode()
    def observe_initial(self, endpoint: Endpoint):
        with self.lock:
            super().observe_initial(endpoint)
            self.readback_state = None
            self.transaction_state = None
            self._transaction_source_q = None
            self._transaction_source_time = None
            return self.status()

    @torch.inference_mode()
    def refresh_endpoint(self, endpoint: Endpoint):
        """Re-measure a current endpoint while carrying actual correction memory.

        A refresh is a real observation boundary, not an H0 reset.  The prior
        is propagated from the last accepted visual correction using the new
        measured q; the image then performs one measurement update.  This is
        the runtime path that makes the observer improve over successive
        execute/reobserve transactions.
        """
        with self.lock:
            self.lease_check()
            if self.executing or self.pending is not None:
                raise RuntimeError("cannot refresh while a five-command transaction is pending")
            endpoint.validate(self.max_endpoint_skew_s)
            if self.state is None:
                return self.observe_initial(endpoint)
            current_time = endpoint.readback_time_s.reshape(-1)
            if self.geometry is not None and self.state.correction_memory is not None:
                elapsed = current_time[:, None] - self.state.correction_time_s
                if (elapsed < 0).any():
                    raise ValueError("refresh endpoint precedes the last accepted correction")
                mechanical = self.geometry.project_correction(
                    self.state.correction_memory, endpoint.q_position, elapsed)
            elif self.geometry is not None:
                mechanical = self.geometry.make_prior(endpoint.q_position)
            else:
                mechanical = None
            observation = self.core.observer(endpoint.rgb, endpoint.dino_tokens,
                                             endpoint.view_valid, prior=mechanical)
            memory = self.state.correction_memory
            correction_time = self.state.correction_time_s
            if self.geometry is not None:
                observed = endpoint.view_valid.bool() & (
                    observation.belief.visible_probability.amax((-1, -2)) > 0)
                limit = getattr(self.core.observer, "max_column_residual_px", None)
                if limit is not None:
                    nominal = self.geometry.make_prior(endpoint.q_position)
                    residual = (observation.belief.state_mean[:, 1] - nominal.state_mean[:, 1]).abs().amax((-1, -2))
                    observed[:, 1] = observed[:, 1] & (residual <= float(limit) + 1e-6)
                memory = self.geometry.capture_correction(
                    observation.belief.detach(), endpoint.q_position.detach(),
                    view_valid=observed, previous_memory=memory)
                accepted = getattr(memory, "update_mask", observed)
                correction_time = torch.where(accepted, current_time[:, None], correction_time)
            grounding, innovation = self.core.control_encoding(
                observation, mechanical, None, None, self.core.q_loss_scale,
                view_valid=endpoint.view_valid)
            self.state = replace(self.state, observation=observation,
                                 q_position=endpoint.q_position.detach().clone(),
                                 grounding_tracks=grounding, innovation_tracks=innovation,
                                 endpoint_time_s=current_time.detach().clone(),
                                 view_valid=endpoint.view_valid.detach().clone(),
                                 prior=mechanical, correction_memory=memory,
                                 correction_time_s=correction_time,
                                 localization_time_s=current_time,
                                 localization_prior=mechanical)
            self.stopped = False
            self.pending_at = None
            return self.status()

    @torch.inference_mode()
    def discard_pending_for_replan(self):
        """Drop an unexecuted preview while preserving completed history."""
        with self.lock:
            if self.executing:
                raise RuntimeError("cannot discard a transaction that is executing")
            self.pending = None
            self.pending_at = None
            self.stopped = False

    def begin_execution(self):
        with self.lock:
            commands = super().begin_execution()
            self.transaction_state = self.state
            self._transaction_source_q = self.state.q_position.detach().clone()
            self._transaction_source_time = self.state.endpoint_time_s.detach().clone()
            self.readback_state = None
            try:
                self._validate_transaction_source()
            except Exception:
                self.stop()
                raise
            return commands

    @torch.inference_mode()
    def rebase_pending_execution_epoch(self, q_position, *, timestamp_s):
        """Bind a pending action to a fresh, unchanged actual q readback."""
        with self.lock:
            self.lease_check()
            if self.stopped or self.pending is None or self.state is None or self.executing:
                raise RuntimeError("cannot rebase without one pending V20.3 plan")
            q = torch.as_tensor(q_position, device=self.state.q_position.device,
                                dtype=self.state.q_position.dtype)
            timestamp = torch.as_tensor(timestamp_s, device=q.device, dtype=torch.float64).reshape(1)
            if q.shape != self.state.q_position.shape or not torch.isfinite(q).all():
                raise ValueError("rebase q must be one finite six-joint readback")
            if not torch.isfinite(timestamp).all() or timestamp.numel() != 1:
                raise ValueError("rebase timestamp must be one finite float64 second")
            if not torch.allclose(q, self.pending.source_q_position, atol=1.e-4, rtol=0):
                self.stop()
                raise RuntimeError("current q changed after planning; discard stale action")
            if timestamp.item() < self.pending.source_time_s.item():
                self.stop()
                raise RuntimeError("rebase timestamp moved backwards")
            if timestamp.item() - self.pending.source_time_s.item() > self.max_plan_age_s:
                self.stop()
                raise RuntimeError("pending visual endpoint exceeded its rebase age")
            self.state = replace(self.state, endpoint_time_s=timestamp,
                                 localization_time_s=timestamp)
            self.pending = self.core.predict(self.state, self.pending.commands_position,
                                             self.pending.time_deltas, self.geometry)
            self.pending_at = self.clock()
            return self.pending.commands_position.detach().clone()

    def _validate_transaction_source(self):
        if (self.stopped or not self.executing or self.pending is None or
                self.transaction_state is None):
            raise RuntimeError("readback/completion requires an executing transaction")
        if self.state is not self.transaction_state:
            raise RuntimeError("transaction endpoint replaced during execution")
        for actual, expected, tolerance in (
            (self.transaction_state.q_position, self._transaction_source_q, 1.e-4),
            (self.pending.source_q_position, self._transaction_source_q, 1.e-4),
            (self.transaction_state.endpoint_time_s, self._transaction_source_time, 1.e-6),
            (self.pending.source_time_s, self._transaction_source_time, 1.e-6),
        ):
            if not torch.allclose(actual, expected, atol=tolerance, rtol=0):
                raise RuntimeError("prediction/actual path source changed during execution")

    def _validate_clock(self, value, *, name):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(f"{name} must be floating seconds")
        if (not torch.isfinite(value).all() or value.numel() == 0 or
                torch.finfo(value.dtype).eps * float(value.abs().max()) > self.max_endpoint_skew_s / 10):
            raise ValueError(f"{name} requires finite float64 absolute seconds at this uptime")

    @torch.inference_mode()
    def update_readback(self, q_position, readback_time_s, *, source_time_s, readback_std=None):
        """Propagate a display-only localization from real, transaction-bound q.

        source_time_s is mandatory: a delayed sample from another plan must not
        attach itself to the current plan. No image measurement or transition
        is fabricated. Invalid data fail closed just like completion failures.
        """
        with self.lock:
            try:
                self.lease_check()
                self._validate_transaction_source()
                self._validate_clock(readback_time_s, name="readback clock")
                self._validate_clock(source_time_s, name="readback source clock")
                if readback_time_s.numel() != 1 or source_time_s.numel() != 1:
                    raise ValueError("readback clocks must identify one live sample")
                if q_position.shape != (1, 6) or not q_position.is_floating_point() or not torch.isfinite(q_position).all():
                    raise ValueError("readback q must be one finite six-joint calibrated position")
                if not torch.allclose(source_time_s.reshape(1).to(self._transaction_source_time),
                                      self._transaction_source_time, atol=1.e-6, rtol=0):
                    raise RuntimeError("readback belongs to a different transaction source")
                previous = self.readback_state if self.readback_state is not None else self.transaction_state
                timestamp = readback_time_s.reshape(1)
                if not bool((timestamp > previous.localization_time_s).all()):
                    raise RuntimeError("actual readback clock must be strictly monotonic")
                deadline = self.pending.source_time_s + self.pending.time_deltas.sum(-1)
                if bool((timestamp > deadline + self.max_endpoint_skew_s).any()):
                    raise RuntimeError("readback is outside the current five-command transaction")
                if timestamp.item() > self.clock() + self.max_endpoint_skew_s:
                    raise RuntimeError("future readback cannot be treated as an actual measurement")
                if readback_std is not None and (readback_std.shape != q_position.shape or
                        not torch.isfinite(readback_std).all() or bool((readback_std < 0).any())):
                    raise ValueError("readback uncertainty must be finite and nonnegative")
                self.readback_state = self.core.advance_readback(
                    previous, q_position.detach().clone(), timestamp.detach().clone(),
                    self.geometry, readback_std=readback_std)
                # A core implementation must not mutate the forecast origin.
                self._validate_transaction_source()
                return self.status()
            except Exception:
                self.stop()
                raise

    def _complete(self, endpoint: Endpoint, actual_commands_position, measured_q_path, readback_times_s):
        """The established five-command checks, with V20.3 actual-q repropagation."""
        with self.lock:
            self.lease_check()
            self._validate_transaction_source()
            endpoint.validate(self.max_endpoint_skew_s)
            prediction = self.pending
            self._validate_clock(readback_times_s, name="actual absolute readback clocks")
            if (actual_commands_position.shape != (1, 5, 6) or
                    not torch.isfinite(actual_commands_position).all() or
                    not torch.allclose(actual_commands_position, prediction.commands_position, atol=1.e-4, rtol=0)):
                raise RuntimeError("sent commands differ from predicted commands; do not fabricate innovation")
            if (measured_q_path.shape != (1, 6, 6) or not torch.isfinite(measured_q_path).all() or
                    not torch.allclose(measured_q_path[:, 0], self._transaction_source_q, atol=1.e-4, rtol=0)):
                raise RuntimeError("actual path must begin at the original transaction q0")
            if readback_times_s.shape != (1, 6) or not bool((readback_times_s[:, 1:] > readback_times_s[:, :-1]).all()):
                raise RuntimeError("readbacks must be ordered q0 through q5")
            if not torch.allclose(readback_times_s[:, 0], prediction.source_time_s, atol=1.e-6, rtol=0):
                raise RuntimeError("readback source timestamp differs from prediction endpoint")
            expected = prediction.source_time_s[:, None] + prediction.time_deltas.cumsum(-1)
            if not torch.allclose(readback_times_s[:, 1:], expected, atol=self.max_endpoint_skew_s, rtol=0):
                raise RuntimeError("execution/readback clock differs from the predicted clock")
            # Image acquisition has a separate skew tolerance; a q readback
            # timestamp is the actual q5 timestamp, not a later settled sample.
            if abs(endpoint.readback_time_s.item() - readback_times_s[0, -1].item()) > 1.e-6:
                raise RuntimeError("later settled image cannot be paired with the previous prediction")
            if not torch.allclose(endpoint.q_position, measured_q_path[:, -1], atol=1.e-4, rtol=0):
                raise RuntimeError("endpoint q differs from the last actual readback")
            if self.readback_state is not None:
                preview_time = self.readback_state.localization_time_s
                if bool((preview_time > endpoint.readback_time_s.reshape(1) + 1.e-6).any()):
                    raise RuntimeError("completed endpoint precedes an already received actual readback")
                if (torch.allclose(preview_time, endpoint.readback_time_s.reshape(1), atol=1.e-6, rtol=0) and
                        not torch.allclose(self.readback_state.q_position, endpoint.q_position, atol=1.e-4, rtol=0)):
                    raise RuntimeError("same-clock readback preview and endpoint disagree")
            current_mechanical = self.geometry.make_prior(endpoint.q_position) if self.geometry is not None else None
            measurement = self.core.observer.measurement(
                endpoint.rgb, endpoint.dino_tokens, endpoint.view_valid, mechanical_prior=current_mechanical)
            # Not readback_state: the raw path, prior and innovation retain q0.
            self.state = self.core.complete_from_measurement(
                self.transaction_state, prediction, measurement, measured_q_path, endpoint.view_valid,
                endpoint.readback_time_s.reshape(1), actual_time_deltas=readback_times_s.diff(dim=1),
                geometry=self.geometry)
            self.pending = None
            self.pending_at = None
            self.executing = False
            self.readback_state = None
            self.transaction_state = None
            self._transaction_source_q = None
            self._transaction_source_time = None
            return self.status()

    def stop(self):
        with self.lock:
            super().stop()
            self.pending_at = None
            self.readback_state = None
            self.transaction_state = None
            self._transaction_source_q = None
            self._transaction_source_time = None

    @staticmethod
    def _pair_status(state):
        belief = state.observation.belief
        return {
            "component_probability": belief.log_weights.exp().detach().cpu().tolist(),
            "component_fm_px": belief.means_px.detach().cpu().tolist(),
            "joint_state_covariance": belief.state_covariance.detach().cpu().tolist(),
            "visible_probability": belief.visible_probability.detach().cpu().tolist(),
            "identity_probability": belief.identity_probability.detach().cpu().tolist(),
        }

    def status(self):
        with self.lock:
            result = super().status()
            result.update(version=VERSION, contract=CONTRACT, position_unit=POSITION_UNIT,
                          action_source="v20_3_actor", geometry_available=self.geometry is not None,
                          metric_geometry_available=bool(getattr(self.geometry, "metric_3d", False)),
                          geometry_kind=getattr(self.geometry, "geometry_kind", "unavailable" if self.geometry is None else "unverified_projection"))
            if self.state is not None:
                result["pair_belief"] = self._pair_status(self.state)
                result["feedback_present"] = self.state.prior is not None
                result["endpoint_time_s"] = self.state.endpoint_time_s.detach().cpu().tolist()
                displayed = self.readback_state if self.readback_state is not None else self.state
                result["actual_localization"] = {
                    "source": "actual_readback_propagation" if self.readback_state is not None else "completed_observation_endpoint",
                    "localization_time_s": displayed.localization_time_s.detach().cpu().tolist(),
                    "observation_endpoint_time_s": displayed.endpoint_time_s.detach().cpu().tolist(),
                    "correction_time_s": displayed.correction_time_s.detach().cpu().tolist(),
                    "q_position": displayed.q_position.detach().cpu().tolist(),
                    "pair_belief": self._pair_status(displayed),
                    "new_image_used": self.readback_state is None and bool(displayed.view_valid.any()),
                    "new_image_used_by_view": (displayed.view_valid if self.readback_state is None else
                        torch.zeros_like(displayed.view_valid)).detach().cpu().tolist(),
                    "usable_for_actor_plan": (self.readback_state is None and not self.stopped and not self.executing
                        and self.pending is None and self.clock() - self.state.endpoint_time_s.item() <= self.max_plan_age_s),
                }
            result["forecast"] = None if self.pending is None else {
                "source": "lewm_prediction_not_actual_readback",
                # ``transaction_id`` is the source clock, not an action
                # index.  It lets the dashboard pair this forecast with the
                # actual q0..q5 path returned by the bridge without guessing
                # from wall-clock arrival order.
                "transaction_id": self.pending.source_time_s.detach().cpu().tolist(),
                "source_time_s": self.pending.source_time_s.detach().cpu().tolist(),
                "source_q_position": self.pending.source_q_position.detach().cpu().tolist(),
                "commands_position": self.pending.commands_position.detach().cpu().tolist(),
                "time_deltas_s": self.pending.time_deltas.detach().cpu().tolist(),
                "predicted_q_path": self.pending.predicted_q_path.detach().cpu().tolist(),
            }
            return result


def load_trained_runtime(output_directory, task_embedding, *, device, physical_check, lease_check,
                         calibration_path=None, **kwargs):
    """Strict full-trained V20.3 entry point; V20.1 weights are not reinterpreted."""
    from .prod_v20_3_training import load_checkpoint, V203Schedule
    from .prod_v20_3_geometry import V203FittedMechanicalProjection

    root = Path(output_directory)
    config = json.loads((root / "config.json").read_text())
    if config.get("smoke") is not False:
        raise ValueError("smoke/unidentified output is not a trained live controller")
    if config.get("version") != VERSION or config.get("contract") != CONTRACT:
        raise ValueError("runtime requires an explicit V20.3 version and contract")
    try:
        schedule = V203Schedule(**config.get("schedule", {}))
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime requires the full fixed V20.3 training schedule") from exc
    if config.get("schedule") != asdict(V203Schedule(sigreg_weight=schedule.sigreg_weight)):
        raise ValueError("runtime requires the full fixed V20.3 training schedule")
    model = config.get("model", {})
    if (model.get("state_representation") != "mechanical-reference-correction-and-joint-pair" or
            model.get("actor_mode") != "full" or model.get("seed") != V203Config().seed):
        raise ValueError("runtime requires the exact V20.3 full Actor model contract")
    calibration_path = calibration_path or config["sources"]["real_calibration"]
    current = json.loads(Path(calibration_path).read_text())
    if current.get("fits") is None or current.get("fits") != config.get("real_projection_fits"):
        raise ValueError("current camera projection differs from training")
    core = V203Core(V203Config(**model)).to(device)
    payload = load_checkpoint(root / "fish" / "final.pt", core, stage="fish", schedule=schedule)
    if not payload.get("metrics", {}).get("complete") or payload.get("step") != schedule.fish_updates:
        raise ValueError("full V20.3 Actor training has not completed")
    # The fitted pixel transport still needs the same SO-101 kinematic graph
    # used during training to lift a paired belief into tool/base references.
    # Omitting it leaves the checkpoint load apparently successful but makes
    # the first real H0 observation fail when ``make_prior`` accesses the
    # anchor indices.  This is a geometry dependency, not a new action solver.
    from .robot_kinematic_graph import DifferentiableRobotKinematics, so101_robot_spec
    kinematics = DifferentiableRobotKinematics(so101_robot_spec()).to(device).eval().requires_grad_(False)
    geometry = V203FittedMechanicalProjection(calibration_path, device=device, kinematics=kinematics)
    return V203Runtime(core, task_embedding.to(device), geometry=geometry,
                       physical_check=physical_check, lease_check=lease_check, **kwargs)


def load_runtime_for_task(output_directory, task_text, *, device, physical_check, lease_check, **kwargs):
    from .prod_v17_8_training import _encode_task_texts

    config = json.loads((Path(output_directory) / "config.json").read_text())
    if config.get("version") != VERSION or config.get("contract") != CONTRACT:
        raise ValueError("runtime requires an explicit V20.3 version and contract")
    task = " ".join(str(task_text).split()).casefold()
    if not task:
        raise ValueError("nonempty task text is required")
    embedding = _encode_task_texts([task], model_name=config["model"]["task_encoder"],
                                  device=torch.device(device), batch_size=1)
    return load_trained_runtime(output_directory, torch.as_tensor(embedding, device=device), device=device,
                                physical_check=physical_check, lease_check=lease_check, **kwargs)
