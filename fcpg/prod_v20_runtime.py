"""Model-only V20 five-command transaction; hardware remains an external owner.

The existing lease/stop/physical-limit layer must be used by the caller.  This
class deliberately has no serial, HOME, ranking or target-coordinate API.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
import json
from pathlib import Path
from typing import Callable

import torch

from .prod_v20 import V20Core, V20Config, V20Prediction, VERSION, CONTRACT, POSITION_UNIT


@dataclass(frozen=True)
class Endpoint:
    rgb: torch.Tensor
    dino_tokens: torch.Tensor
    q_position: torch.Tensor
    view_valid: torch.Tensor
    image_time_s: torch.Tensor
    readback_time_s: torch.Tensor

    def validate(self, maximum_skew_s: float):
        if self.q_position.shape != (1, 6) or self.view_valid.shape != (1, 2):
            raise ValueError("live endpoint must contain one six-joint dual-view observation")
        for value in (self.q_position, self.image_time_s, self.readback_time_s):
            if not torch.isfinite(value).all():
                raise ValueError("non-finite endpoint")
        for timestamp in (self.image_time_s, self.readback_time_s):
            if not timestamp.is_floating_point():
                raise ValueError("endpoint timestamps must be floating seconds")
            if torch.finfo(timestamp.dtype).eps * float(timestamp.abs().max()) > maximum_skew_s / 10:
                raise ValueError("absolute endpoint clocks require float64 seconds at this uptime")
        if self.image_time_s.numel() not in (1, 2) or self.readback_time_s.numel() != 1:
            raise ValueError("invalid endpoint clock shape")
        active = self.view_valid.flatten()
        skew = (self.image_time_s.flatten().expand(2) - self.readback_time_s.item()).abs()
        if bool((skew[active] > maximum_skew_s).any()):
            raise ValueError("RGB and q are not from the same endpoint clock")


class V20Runtime:
    def __init__(self, core: V20Core, task_embedding: torch.Tensor, *, geometry=None,
                 max_endpoint_skew_s: float = 0.05, max_plan_age_s: float = 1.0,
                 physical_check: Callable[[torch.Tensor, torch.Tensor], None],
                 lease_check: Callable[[], None], clock=time.monotonic):
        self.core = core.eval().requires_grad_(False)
        self.task_embedding = task_embedding
        self.geometry = geometry
        self.max_endpoint_skew_s = max_endpoint_skew_s
        self.max_plan_age_s = max_plan_age_s
        self.physical_check = physical_check
        self.lease_check = lease_check
        self.clock = clock
        self.state = None
        self.pending: V20Prediction | None = None
        self.pending_at = None
        self.executing = False
        self.stopped = True
        self.lock = threading.RLock()

    @torch.inference_mode()
    def observe_initial(self, endpoint: Endpoint):
        with self.lock:
            self.lease_check()
            if self.executing or self.pending is not None:
                raise RuntimeError("cannot reset while a five-command transaction is pending")
            endpoint.validate(self.max_endpoint_skew_s)
            self.state = self.core.initial(endpoint.rgb, endpoint.dino_tokens, endpoint.q_position,
                                          endpoint.view_valid, endpoint.readback_time_s.reshape(1), self.geometry)
            self.stopped = False
            return self.status()

    @torch.inference_mode()
    def plan(self, time_deltas: torch.Tensor):
        with self.lock:
            self.lease_check()
            if self.stopped or self.state is None or self.pending is not None:
                raise RuntimeError("observe H0 first; one outstanding transaction only")
            if self.clock() - self.state.endpoint_time_s.item() > self.max_plan_age_s:
                self.stop()
                raise RuntimeError("actual endpoint expired; reobserve before planning")
            action = self.core.act(self.state, self.task_embedding)
            # Physical legality is checked BEFORE LeWM sees the command.  A
            # rejection never mutates/rescales the Actor's five-command output.
            self.physical_check(self.state.q_position, action.commands_position)
            self.pending = self.core.predict(self.state, action.commands_position, time_deltas, self.geometry)
            self.pending_at = self.clock()
            return self.pending

    def begin_execution(self):
        with self.lock:
            self.lease_check()
            if self.stopped or self.pending is None or self.executing:
                raise RuntimeError("no executable V20 plan")
            if self.clock() - self.pending.source_time_s.item() > self.max_plan_age_s:
                self.stop()
                raise RuntimeError("plan expired; reobserve before executing")
            first_readback = self.pending.source_time_s.item() + self.pending.time_deltas[0, 0].item()
            if self.clock() >= first_readback:
                self.stop()
                raise RuntimeError("first readback deadline already passed; reobserve before execution")
            self.physical_check(self.state.q_position, self.pending.commands_position)
            self.executing = True
            return self.pending.commands_position.detach().clone()

    @torch.inference_mode()
    def complete(self, endpoint: Endpoint, actual_commands_position: torch.Tensor,
                 measured_q_path: torch.Tensor, readback_times_s: torch.Tensor):
        with self.lock:
            try:
                return self._complete(endpoint, actual_commands_position, measured_q_path, readback_times_s)
            except Exception:
                self.stop()
                raise

    def _complete(self, endpoint: Endpoint, actual_commands_position: torch.Tensor,
                  measured_q_path: torch.Tensor, readback_times_s: torch.Tensor):
        with self.lock:
            self.lease_check()
            if self.stopped or not self.executing or self.pending is None:
                raise RuntimeError("completion without an active transaction")
            endpoint.validate(self.max_endpoint_skew_s)
            prediction = self.pending
            if (not readback_times_s.is_floating_point() or
                    torch.finfo(readback_times_s.dtype).eps * float(readback_times_s.abs().max()) > self.max_endpoint_skew_s / 10):
                raise ValueError("actual absolute readback clocks require float64 seconds")
            if actual_commands_position.shape != (1, 5, 6) or not torch.allclose(actual_commands_position, prediction.commands_position, atol=1.e-4, rtol=0):
                self.stop()
                raise RuntimeError("sent commands differ from predicted commands; do not fabricate innovation")
            if readback_times_s.shape != (1, 6) or not bool((readback_times_s[:, 1:] > readback_times_s[:, :-1]).all()):
                self.stop()
                raise RuntimeError("readbacks must be ordered q0 through q5")
            if not torch.allclose(readback_times_s[:, 0], prediction.source_time_s, atol=1.e-6, rtol=0):
                raise RuntimeError("readback source timestamp differs from prediction endpoint")
            expected = prediction.source_time_s[:, None] + prediction.time_deltas.cumsum(-1)
            if not torch.allclose(readback_times_s[:, 1:], expected, atol=self.max_endpoint_skew_s, rtol=0):
                self.stop()
                raise RuntimeError("execution/readback clock differs from the predicted clock")
            if abs(endpoint.readback_time_s.item() - readback_times_s[0, -1].item()) > self.max_endpoint_skew_s:
                self.stop()
                raise RuntimeError("later settled image cannot be paired with the previous prediction")
            if not torch.allclose(endpoint.q_position, measured_q_path[:, -1], atol=1.e-4, rtol=0):
                self.stop()
                raise RuntimeError("endpoint q differs from the last actual readback")
            current_mechanical = self.geometry.make_prior(endpoint.q_position) if self.geometry is not None else None
            measurement = self.core.observer.measurement(endpoint.rgb, endpoint.dino_tokens, endpoint.view_valid,
                                                         mechanical_prior=current_mechanical)
            self.state = self.core.complete_from_measurement(self.state, prediction, measurement,
                          measured_q_path, endpoint.view_valid, endpoint.readback_time_s.reshape(1),
                          actual_time_deltas=readback_times_s.diff(dim=1))
            self.pending = None
            self.executing = False
            return self.status()

    def stop(self):
        # This is a model stop. The hardware owner must concurrently cancel its
        # write loop; no success claim here implies motors have stopped.
        with self.lock:
            self.stopped = True
            self.pending = None
            self.executing = False

    def status(self):
        result = dict(version=VERSION, contract=CONTRACT, position_unit=POSITION_UNIT,
                      action_source="v20_actor", stopped=self.stopped, pending=self.pending is not None,
                      executing=self.executing, pickup=None, drop=None)
        if self.state is not None:
            belief = self.state.observation.belief
            result["pair_belief"] = {
                "component_probability": belief.log_weights.exp().detach().cpu().tolist(),
                "component_fm_px": belief.means_px.detach().cpu().tolist(),
                "visible_probability": belief.visible_probability.detach().cpu().tolist(),
            }
            result["q_position"] = self.state.q_position.detach().cpu().tolist()
            result["view_valid"] = self.state.view_valid.detach().cpu().tolist()
            result["estimate_source"] = ["current_image_and_mechanical_prior" if valid else "propagated_prior_only"
                                         for valid in self.state.view_valid[0].detach().cpu().tolist()]
        return result


def load_trained_runtime(output_directory, task_embedding, *, device, physical_check, lease_check,
                         calibration_path=None, **kwargs):
    """Strict deploy entry point; never treat an initialized/smoke model as trained.

    Uses the exact ordinary projection coefficients recorded before training.
    The external hardware owner still provides the established lease/limits.
    """
    from .prod_v20_training import load_checkpoint
    from .prod_v20_geometry import V20FittedMechanicalProjection
    root = Path(output_directory)
    config = json.loads((root / "config.json").read_text())
    if config.get("smoke") is not False:
        raise ValueError("smoke/unidentified output is not a trained live controller")
    calibration_path = calibration_path or config["sources"]["real_calibration"]
    current = json.loads(Path(calibration_path).read_text())
    if current.get("fits") != config.get("real_projection_fits"):
        raise ValueError("current camera projection differs from the training configuration")
    core = V20Core(V20Config(**config["model"])).to(device)
    payload = load_checkpoint(root / "fish" / "final.pt", core, stage="fish")
    if not payload.get("metrics", {}).get("complete") or payload.get("step") != config["schedule"]["fish_updates"]:
        raise ValueError("full Actor training has not completed")
    geometry = V20FittedMechanicalProjection(calibration_path, device=device)
    return V20Runtime(core, task_embedding.to(device), geometry=geometry,
                      physical_check=physical_check, lease_check=lease_check, **kwargs)


def load_runtime_for_task(output_directory, task_text: str, *, device, physical_check, lease_check, **kwargs):
    """User-facing task input is text, never hand-specified goal coordinates."""
    from .prod_v17_8_training import _encode_task_texts
    config = json.loads((Path(output_directory) / "config.json").read_text())
    task = " ".join(str(task_text).split()).casefold()
    if not task:
        raise ValueError("a nonempty task text is required")
    embedding = _encode_task_texts([task], model_name=config["model"]["task_encoder"],
                                  device=torch.device(device), batch_size=1)
    return load_trained_runtime(output_directory, torch.as_tensor(embedding, device=device),
                                device=device, physical_check=physical_check, lease_check=lease_check, **kwargs)
