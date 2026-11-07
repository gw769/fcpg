"""V20.3 hardware owner: real q/cameras, strict five-step bridge.

The service is opt-in and owns the existing SO-101 lease.  It never calls the
legacy V17 clocked executor; all policy actions go through
``V203TransactionBridge`` and the V20.3 runtime completion contract.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import threading
import time
import json
from typing import Any

import cv2
import numpy as np
import torch

from fcpg.prod_v17_8_1_live import load_fish_task_embedding_cache
from fcpg.prod_v18_1_training import extract_dinov2_tokens, load_frozen_dinov2
from fcpg.prod_v20_3_hardware_bridge import V203TransactionBridge
from fcpg.prod_v20_3_runtime import Endpoint, load_trained_runtime
from fcpg.prod_v17_7_dino_targets import locate_dino_targets
from fcpg.so101_visual_targets import VisualTargetTracker
import scripts.serve_so101_prod_v17_7 as v177
from scripts.serve_so101_prod_v17_7 import V177ClearanceGate, _audit_v177_chunk



HOME_JOINT_STEP = 12.0  # per audited HOME chunk (five commands at 0.2 s), all moving joints together

class V203HardwareService:
    def __init__(self, *, model_output: str | Path, output: str | Path,
                 device: str = "cuda:0", robot_port: str = v177.DEFAULT_ROBOT_PORT,
                 wrist_device: str = v177.DEFAULT_WRIST_DEVICE,
                 column_device: str = v177.DEFAULT_COLUMN_DEVICE,
                 runtime_loader=None, policy_period_s: float = .2, task_cache_path=None):
        self.root = Path(model_output).expanduser().resolve()
        self.output = Path(output).expanduser().resolve()
        self.device = torch.device(device)
        self.policy_period_s = float(policy_period_s)
        if not np.isfinite(self.policy_period_s) or self.policy_period_s <= 0:
            raise ValueError("policy command period must be finite positive seconds")
        self.stop_event = threading.Event()
        self.bus_lock = threading.RLock()
        self.lease = None
        self.io = None
        self._frame_times = {}
        self._dino_target_profile = Path('/home/james/so101-vda-disk/feedback-corrected-grounding/outputs/lerobot/so101/v12_1_actor_v12_4_lewm20k_live_20260817/target_profile/class_1')
        self._dino_target_tracker = None
        self._registered_target_records = {}
        try:
            _profile_targets = json.loads((self._dino_target_profile / "targets.json").read_text())
            for _item in _profile_targets.get("targets", []):
                _key = (str(_item.get("role")), str(_item.get("view")))
                _centroid = _item.get("registered_centroid_px")
                if _key[0] in {"pickup", "drop_zone"} and _key[1] in {"wrist", "column"}:
                    if isinstance(_centroid, list) and len(_centroid) == 2 and all(np.isfinite(_centroid)):
                        self._registered_target_records[_key] = {
                            "centroid_px": _centroid,
                            "axis_px": [1.0, 0.0],
                            "width_px": float(_item.get("selection_radius_px", 18.0)),
                            "length_px": float(_item.get("selection_radius_px", 18.0)) * 2.0,
                        }
        except Exception:
            self._registered_target_records = {}
        self.dino_targets = {"source": "frozen_dinov2_profile_prototype", "display_only": True,
                             "control_input": False, "targets": {"pickup": {}, "drop_zone": {}}}
        # Last completed model observation used strictly for camera overlays.
        # It is never fed back into the controller and is cleared on startup.
        self.display_pair_belief = None
        try:
            cache_root = (Path(task_cache_path).expanduser().resolve() if task_cache_path is not None
                          else self.root / "task-cache" if (self.root / "task-cache").is_dir()
                          else self.root.parent / "task-cache")
            self.task = load_fish_task_embedding_cache(cache_root).to(self.device).unsqueeze(0)
            calibration = self.root / ".." / ".."  # replaced below by training metadata
            self.calibration = Path("/home/james/feedback-corrected-grounding/outputs/lerobot/so101/my_first_task_clean_v1_dual_camera_calibration/calibration_letterbox_v1.json")
            loader = runtime_loader or load_trained_runtime
            self.runtime = loader(self.root, self.task, device=self.device,
                physical_check=lambda q, a: self._physical_check(q, a), lease_check=lambda: self._lease_check(),
                # Keep a generous cache lifetime for the dashboard, but
                # execute() below always captures a fresh endpoint first.  A
                # stale visual plan therefore can never be sent to hardware.
                calibration_path=self.calibration, max_plan_age_s=300.0)
            self.dino = load_frozen_dinov2("/home/james/.cache/torch/hub/facebookresearch_dinov2_main",
                                           "/home/james/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth", device=self.device)
            self.kinematics = v177.SO101MuJoCoKinematics(v177.DEFAULT_KINEMATICS_MODEL)
            self.clearance = V177ClearanceGate(kinematics=self.kinematics,
                table_z_mm=v177.TABLE_Z_MM, planned_clearance_mm=v177.PLANNED_CLEARANCE_MM,
                readback_clearance_mm=v177.READBACK_CLEARANCE_MM)
            self.home_state = v177._load_v177_home_state(
                Path('/home/james/feedback-corrected-grounding-v203/configs/prod_v17_7_legacy_v17_4_home.json'),
                gripper_position=31.340341567993164,
            )
            self.lease = v177.acquire_so101_hardware_lease(lock_path=v177.DEFAULT_HARDWARE_LOCK,
                serial_device=robot_port, wrist_device=wrist_device, column_device=column_device)
            io_args = SimpleNamespace(wrist_device=wrist_device, column_device=column_device,
                output=self.output, robot_port=robot_port, robot_id="myfollower01", calibration_dir=None,
                max_command_jump=float(np.max(v177.SO101_POSITION_UPPER-v177.SO101_POSITION_LOWER)+1e-4),
                settle_timeout_s=20., settle_position_tolerance=3., settle_step_tolerance=.25,
                settle_samples=3, self_contact_calibration=False,
                i_accept_empty_jaw_self_contact=False, experiment_binding=None)
            self.io = v177.V177HardwareSO101IO(io_args, clearance_gate=self.clearance,
                gripper_calibration_closed=v177.DEFAULT_GRIPPER_CLOSED_POSITION,
                gripper_calibration_open=v177.DEFAULT_GRIPPER_OPEN_POSITION,
                endpoint_envelope=None)
            self.current_endpoint = self.capture_endpoint()
            self.bridge = V203TransactionBridge(self.runtime, self.io,
                capture_endpoint=lambda q, t: self.capture_endpoint(q_expected=q, timestamp=t),
                read_q=self._read_q, write_q=self._write_q, halt=self._halt,
                physical_check=self._physical_check, period_s=self.policy_period_s,
                cancelled=self.stop_event.is_set, wait=self.stop_event.wait)
            self.bridge.set_endpoint(self.current_endpoint)
            self.task_text = "夹起鱼并放入盒子"
        except BaseException:
            self.close()
            raise

    def _lease_check(self):
        if self.lease is None or not self.lease.acquired:
            raise RuntimeError("V20.3 hardware lease is not held")

    def _read_q(self):
        with self.bus_lock:
            return self.io._read_state_only()

    def _write_q(self, command, previous):
        with self.bus_lock:
            return self.io._write_position_command(command, previous)

    def _halt(self):
        # STOP and HTTP status share the same bus owner and transaction lock.
        # The transport's hold includes a read and write, so lock both together.
        with self.bus_lock:
            return self.io._hold_at_measured_pose(None, reason="v20_3_bridge_fault")

    def _physical_check(self, current, commands):
        current_np = current.detach().cpu().numpy()[0] if isinstance(current, torch.Tensor) and current.ndim == 2 else np.asarray(current)
        commands_np = commands.detach().cpu().numpy()[0] if isinstance(commands, torch.Tensor) and commands.ndim == 3 else np.asarray(commands)
        audit = _audit_v177_chunk(commands_np.astype(np.float32), current_np.astype(np.float32), clearance_gate=self.clearance)
        if not bool(audit.get("safe")):
            raise RuntimeError("V20.3 command rejected: " + ", ".join(audit.get("reasons", [])))

    def _endpoint_from_observation(self, observation, *, q_expected=None, timestamp=None):
        rgb = np.stack([np.asarray(observation.frames["wrist"]), np.asarray(observation.frames["column"])], 0)
        if rgb.shape != (2, 224, 224, 3) or rgb.dtype != np.uint8:
            raise RuntimeError(f"live camera frames must be RGB uint8 [2,224,224,3], got {rgb.shape}/{rgb.dtype}")
        dino = extract_dinov2_tokens(self.dino, rgb, device=self.device, batch_size=2)
        q = np.asarray(observation.state if q_expected is None else q_expected, dtype=np.float32)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise RuntimeError("live endpoint q is invalid")
        stamp = float(observation.timestamp_s if timestamp is None else timestamp)
        return Endpoint(torch.as_tensor(rgb, device=self.device).permute(0,3,1,2)[None],
                        torch.as_tensor(dino, device=self.device)[None], torch.as_tensor(q, device=self.device)[None],
                        torch.ones((1,2), device=self.device, dtype=torch.bool),
                        torch.tensor([stamp], device=self.device, dtype=torch.float64),
                        torch.tensor([stamp], device=self.device, dtype=torch.float64))

    def _capture_observation(self):
        # Capture both cameras first, then read q immediately after them. This
        # gives the endpoint one truthful bracket and avoids the legacy
        # executor's post-q5 settled sample.
        with self.io.camera_lock:
            frames = {view: self.io._read_camera(view) for view in ("wrist", "column")}
        state = self._read_q()
        observation = SimpleNamespace(frames=frames, state=state,
                                      timestamp_s=time.monotonic())
        return observation

    def capture_endpoint(self, *, q_expected=None, timestamp=None):
        self._lease_check()
        observation = self._capture_observation()
        frames = observation.frames
        try:
            if self._dino_target_tracker is None:
                self._dino_target_tracker = VisualTargetTracker(self._dino_target_profile)
            self.dino_targets = locate_dino_targets(frames, tracker=self._dino_target_tracker)
            self.dino_targets["source_timestamp_s"] = observation.timestamp_s
            self.dino_targets["profile"] = str(self._dino_target_profile)
        except Exception as exc:
            self.dino_targets = {"source": "frozen_dinov2_profile_prototype", "display_only": True,
                                 "control_input": False, "source_timestamp_s": observation.timestamp_s,
                                 "targets": {"pickup": {}, "drop_zone": {}},
                                 "error": f"{type(exc).__name__}: {exc}"}
        if q_expected is not None and np.max(np.abs(np.asarray(observation.state)-np.asarray(q_expected))) > 3.0:
            raise RuntimeError("final q5 changed while capturing the synchronous endpoint")
        endpoint = self._endpoint_from_observation(observation, q_expected=q_expected, timestamp=timestamp)
        self.current_endpoint = endpoint
        return endpoint

    def plan(self):
        self.current_endpoint = self.capture_endpoint()
        result = self.bridge.plan(self.current_endpoint)
        runtime_status = result.get("runtime", {}) if isinstance(result, dict) else {}
        self.display_pair_belief = (
            (result.get("pair_belief") if isinstance(result, dict) else None)
            or (runtime_status.get("pair_belief") if isinstance(runtime_status, dict) else None)
        )
        result["preview_only"] = False
        result["next_action"] = {"commands_position": self.bridge.last_prediction.commands_position.detach().cpu().tolist(),
                                  "preview_only": False}
        return result

    def execute(self):
        # Re-observe immediately before every physical transaction.  This
        # makes the five-minute dashboard cache a display convenience rather
        # than permission to execute stale camera state.
        if self.bridge.last_prediction is not None:
            # Discard only the unexecuted preview; completed actual history is
            # retained by refresh_endpoint on the subsequent fresh plan.
            self.runtime.discard_pending_for_replan()
            self.bridge.last_prediction = None
        self.plan()
        try:
            result = self.bridge.execute()
            self.current_endpoint = self.bridge.last_endpoint
            return result
        finally:
            if self.bridge.last_attempt is not None:
                path = self.output / "last_attempt.json"
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(self.bridge.last_attempt, ensure_ascii=False, allow_nan=False), encoding="utf-8")
                temporary.replace(path)

    def _home_step(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Choose one small, clearance-audited joint step toward fixed HOME."""
        candidates = []
        source = np.asarray(current, dtype=np.float32)
        goal = np.asarray(target, dtype=np.float32)
        tolerance = float(getattr(self.io, "settle_position_tolerance", 3.0))
        # The established HOME route first recovers a low-clearance pose by
        # lifting the shoulder (J2) outward.  Without this branch a partially
        # parked arm can leave every direct J3/J4 step below the table floor.
        if self.clearance.clearance_mm(source) < self.clearance.planned_clearance_mm - 1e-9:
            endpoint = source.copy()
            endpoint[1] = max(float(v177.SO101_POSITION_LOWER[1]), float(source[1]) - 10.0)
            chunk = np.repeat(source[None], 5, axis=0).astype(np.float32)
            chunk[:, 1] = np.linspace(source[1], endpoint[1], 6)[1:]
            audit = _audit_v177_chunk(chunk, source, clearance_gate=self.clearance)
            if not bool(audit.get("safe")):
                raise RuntimeError("HOME clearance recovery rejected: " + ", ".join(audit.get("reasons", [])))
            return chunk
        # Fast path: every joint that still has to move steps together (up to
        # HOME_JOINT_STEP each), audited as one chunk. One joint at a time was
        # safe but took a minute; the audit, not the sequencing, is the safety.
        moving = np.abs(goal - source) > tolerance
        if moving.any():
            joint_step=float(getattr(self,'home_joint_step',HOME_JOINT_STEP))
            if not np.isfinite(joint_step) or not 0<joint_step<=20.:
                raise ValueError('invalid bounded HOME joint step')
            endpoint = source.copy()
            endpoint[moving] = source[moving] + np.clip(goal[moving] - source[moving], -joint_step, joint_step)
            chunk = np.stack([source + (endpoint - source) * a for a in np.linspace(0.0, 1.0, 6)[1:]]).astype(np.float32)
            audit = _audit_v177_chunk(chunk, source, clearance_gate=self.clearance)
            if bool(audit.get("safe")):
                return chunk
        for joint in range(6):
            remaining = float(goal[joint] - source[joint])
            if abs(remaining) <= tolerance:
                continue
            endpoint = source.copy()
            endpoint[joint] = source[joint] + np.clip(remaining, -5.0, 5.0)
            chunk = np.repeat(source[None], 5, axis=0).astype(np.float32)
            chunk[:, joint] = np.linspace(source[joint], endpoint[joint], 6)[1:]
            audit = _audit_v177_chunk(chunk, source, clearance_gate=self.clearance)
            if bool(audit.get("safe")):
                score = (float(audit.get("minimum_planned_clearance_mm", -1e9)),
                         abs(remaining) - abs(float(goal[joint] - endpoint[joint])))
                candidates.append((score, chunk))
        if not candidates:
            raise RuntimeError("fixed HOME has no clearance-safe next joint step")
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def home(self):
        """Move only along the established fixed HOME route, then reobserve."""
        self._lease_check()
        self.runtime.stop()
        self.bridge.last_prediction = None
        target = np.asarray(self.home_state, dtype=np.float32)
        try:
            for _ in range(80):
                current = np.asarray(self._read_q(), dtype=np.float32)
                if current.shape != (6,) or not np.isfinite(current).all():
                    raise RuntimeError("HOME readback invalid")
                self.clearance.require_readback(current, label="HOME readback")
                if float(np.max(np.abs(current - target))) <= float(self.io.settle_position_tolerance):
                    break
                chunk = self._home_step(current, target)
                with self.bus_lock:
                    self.io.execute_home_chunk(chunk, period_s=.2, stop_event=self.stop_event)
            else:
                raise RuntimeError("fixed HOME did not converge inside its budget")
            final_q = np.asarray(self._read_q(), dtype=np.float32)
            self.current_endpoint = self.capture_endpoint(q_expected=final_q)
            self.bridge.set_endpoint(self.current_endpoint)
            self.runtime.observe_initial(self.current_endpoint)
            return self.status()
        except BaseException:
            self.runtime.stop()
            raise

    def status(self):
        status = self.bridge.status()
        status["runtime"]["task_text"] = self.task_text
        status["runtime"]["preview_only"] = False
        if self.bridge.last_prediction is not None:
            status["runtime"]["next_action"] = {"commands_position": self.bridge.last_prediction.commands_position.detach().cpu().tolist(),
                                                   "preview_only": False}
        # The controller owns the cameras, so there are no CameraWorker
        # snapshots for the dashboard's normal `cameras` field. Publish the
        # same freshness/shape contract here; JPEGs are refreshed separately.
        endpoint = self.current_endpoint
        status["cameras"] = {
            view: {
                "available": True,
                "width": 224,
                "height": 224,
                "age_s": (max(0.0, time.monotonic() - self._frame_times[view])
                          if view in self._frame_times else None),
                "captured_at": self._frame_times.get(view),
                "error": None,
            }
            for view in ("wrist", "column")
        }
        try:
            q_live = np.asarray(self._read_q(), dtype=np.float32)
            if q_live.shape == (6,) and np.isfinite(q_live).all():
                status["hardware_q_position"] = q_live.tolist()
        except Exception:
            status["hardware_q_position"] = None
        status["runtime"]["pickup"] = status["runtime"].get("pickup")
        status["runtime"]["drop"] = status["runtime"].get("drop")
        status["dino_targets"] = self.dino_targets
        return status

    def handle(self, name, payload=None):
        if name == "plan": return self.plan()
        if name == "execute": return self.execute()
        if name == "stop": self.stop(); return self.status()
        if name == "home": return self.home()
        raise RuntimeError(f"unknown V20.3 hardware operation: {name}")

    def frame(self, view):
        if view not in ("wrist", "column"):
            return None
        frame = self.io.preview_frame(view)
        self._frame_times[view] = time.monotonic()
        # Draw the model's paired fingertips on the raw camera frame.  This is
        # deliberately display-only: the controller consumes the latent
        # posterior, not these rendered coordinates.
        belief = self.display_pair_belief if isinstance(self.display_pair_belief, dict) else {}
        if not belief and hasattr(self, "bridge"):
            try:
                live_status = self.bridge.runtime.status()
                belief = live_status.get("pair_belief", {}) if isinstance(live_status, dict) else {}
            except Exception:
                belief = {}
        return self._render_frame(view, frame, belief, self.dino_targets)

    def _render_frame(self, view, frame, belief, dino_targets, *, title=None):
        """Render given pixels/belief only; never fetch a newer camera frame."""
        bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        try:
            probs = np.asarray(belief.get("component_probability"), dtype=np.float32)
            means = np.asarray(belief.get("component_fm_px"), dtype=np.float32)
            view_index = ("wrist", "column").index(view)
            if probs.ndim == 3 and probs.shape[0] == 1:
                probs = probs[0]
            if means.ndim == 5 and means.shape[0] == 1:
                means = means[0]
            if probs.ndim == 2 and means.ndim == 4 and probs.shape[0] == 2 and means.shape[0] == 2:
                probs_view, means_view = probs[view_index], means[view_index]
                order = np.argsort(-probs_view)
                # The camera overlay is for fast human debugging: show only
                # the MAP paired hypothesis (exactly one F/M pair).  The full
                # four-component belief remains available in the dashboard
                # table/SVG, so uncertainty is not discarded from the model.
                for rank, idx in enumerate(order[:1]):
                    if idx >= means_view.shape[0] or float(probs_view[idx]) < 0.01:
                        continue
                    pair = means_view[idx]
                    if pair.shape != (2, 2) or not np.isfinite(pair).all():
                        continue
                    alpha = 0.35 + 0.65 * float(np.clip(probs_view[idx], 0.0, 1.0))
                    # Low-probability hypotheses are intentionally subdued;
                    # the highest-probability pair remains unambiguous.
                    f_color = tuple(int(round(c * alpha)) for c in (70, 255, 70))
                    m_color = tuple(int(round(c * alpha)) for c in (255, 80, 230))
                    f, m = [tuple(np.round(p).astype(int).tolist()) for p in pair]
                    cv2.line(bgr, f, m, (180, 220, 255), 1 if rank else 2, cv2.LINE_AA)
                    cv2.circle(bgr, f, 4 if rank == 0 else 3, f_color, -1, cv2.LINE_AA)
                    cv2.circle(bgr, m, 4 if rank == 0 else 3, m_color, -1, cv2.LINE_AA)
                    if rank == 0:
                        cv2.putText(bgr, "F", (max(2, f[0] + 4), max(13, f[1] - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, f_color, 1, cv2.LINE_AA)
                        cv2.putText(bgr, "M", (max(2, m[0] + 4), max(13, m[1] + 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, m_color, 1, cv2.LINE_AA)
                # OpenCV's default Hershey font is ASCII-only; keep this
                # camera overlay legible in every deployment locale.
                logic = "APERTURE PRIOR" if view == "wrist" else "Q1-Q6 + HISTORY"
                cv2.putText(bgr, title or "FCPG F/M | " + logic, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            # Rendering must never make the hardware endpoint unavailable.
            pass
        # Display-only DINOv2 target overlay; never enters control inputs.
        targets = dino_targets.get("targets", {}) if isinstance(dino_targets, dict) else {}
        pickup = targets.get("pickup", {}) if isinstance(targets, dict) else {}
        drop = targets.get("drop_zone", {}) if isinstance(targets, dict) else {}
        records = [("TARGET", pickup.get(view), (40, 211, 255), "pickup") if isinstance(pickup, dict) else None,
                   ("DROP", drop.get(view), (72, 184, 255), "drop_zone") if isinstance(drop, dict) else None]
        for item in records:
            if item is None:
                continue
            label, record, color, role = item
            registered = False
            if isinstance(record, dict) and record.get("visible") is not True:
                fallback = self._registered_target_records.get((role, view))
                if fallback is not None:
                    record = dict(fallback)
                    registered = True
            if not isinstance(record, dict):
                continue
            center = np.asarray(record.get("centroid_px"), dtype=np.float32)
            axis = np.asarray(record.get("axis_px"), dtype=np.float32)
            if center.shape != (2,) or axis.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(axis).all():
                continue
            width = max(3.0, float(record.get("width_px", 8.0)))
            length = max(width, float(record.get("length_px", width)))
            c = tuple(np.round(center).astype(int).tolist())
            angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
            draw_color = tuple(int(round(v * 0.55)) for v in color) if registered else color
            cv2.ellipse(bgr, c, (int(length / 2), int(width / 2)), angle, 0, 360, draw_color, 1, cv2.LINE_AA)
            tip = center + axis / max(float(np.linalg.norm(axis)), 1e-6) * (length / 2)
            cv2.line(bgr, c, tuple(np.round(tip).astype(int).tolist()), draw_color, 2 if not registered else 1, cv2.LINE_AA)
            cv2.circle(bgr, c, 3, draw_color, -1, cv2.LINE_AA)
            # V21: no TARGET/DROP text on the JPEG (letters hide the fish / CoG).
            if label:
                shown_label = label + (" (registered)" if registered else "")
                cv2.putText(bgr, shown_label, (max(2, c[0] - 28), max(14, c[1] - int(width / 2) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, draw_color, 1, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return encoded.tobytes() if ok else None

    def stop(self):
        self.stop_event.set()
        if hasattr(self, "bridge"):
            self.bridge.stop()

    def close(self):
        try:
            if self.io is not None:
                self.io.close()
        finally:
            self.io = None
            if self.lease is not None:
                self.lease.close()
                self.lease = None
