#!/usr/bin/env python3
"""Supervised V21 research five-step trial. Default checks files only, owns no hardware."""
import argparse
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
from types import SimpleNamespace
from collections.abc import Mapping

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcpg.v21_core import VERSION
from fcpg.v21_runtime import load_runtime, EncodedEndpoint
from fcpg.v21_memory import VisualObservation
from fcpg.v21_saved_observation import load_saved_column
from fcpg.v21_hardware_bridge import V21TransactionBridge
from fcpg.v21_capture import capture_actual_views, CameraDeliveryUnavailable
from fcpg.v21_lens_cover import PaperCoverHold, frame_is_covered
from fcpg.v21_source_evidence import SourceEvidenceStore, object_token_provenance
from fcpg.v21_actuator_history import history_for
from fcpg.v21_training import write_json
from fcpg.prod_v18_1_training import load_frozen_dinov2, extract_dinov2_tokens
from fcpg.v21_encoder import DIM, DINOV2_OFF, ENCODER, PATCHES
from fcpg.v21_live_encoder import ModelTokens
from fcpg.v21_recording_look import RecordingLook, policy_pictures, recording_look_enabled
from fcpg.prod_v20_3_hardware_service import V203HardwareService
import scripts.serve_so101_prod_v17_7 as established


_CAMERA_VIEW_PATTERN = re.compile(r"\bcamera[ _-]+['\"]?(wrist|column)\b", re.IGNORECASE)


def _exception_chain(error: BaseException) -> list[dict[str, str]]:
    """Return the finite exception chain without serialising traceback objects."""
    rows: list[dict[str, str]] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        rows.append({"type": type(current).__name__, "message": str(current)})
        current = current.__cause__ or current.__context__
    return rows


def _camera_failure_reason(error_chain: list[dict[str, str]]) -> str:
    """Classify only evidence present in the transport exception text."""
    text = " ".join(row["message"] for row in error_chain).lower()
    if any(token in text for token in ("dqbuf", "vidioc_dqbuf", "v4l2_buf_flag_error")):
        return "v4l2_dqbuf_error"
    if any(token in text for token in ("timeout", "timed out", "did not recover")):
        return "camera_timeout"
    if any(token in text for token in ("disconnect", "not connected", "failed to open")):
        return "camera_disconnected"
    return "camera_startup_error"


def camera_startup_diagnostic(error: BaseException, *, io=None,
                              active_views=("wrist", "column"), stage="startup"):
    """Build JSON-safe camera failure evidence for the failed V21 startup.

    The helper only reads state already held by the owner. It does not probe a
    device, reconnect a camera, or inspect historical frames. A diagnostic is
    returned only when the exception chain is camera-related or the transport
    has recorded a per-camera failure.
    """
    chain = _exception_chain(error)
    text = " ".join(row["message"] for row in chain).lower()
    last_errors = getattr(io, "camera_last_errors", {}) if io is not None else {}
    reconnects = getattr(io, "camera_reconnect_counts", {}) if io is not None else {}
    if not isinstance(last_errors, Mapping):
        last_errors = {}
    if not isinstance(reconnects, Mapping):
        reconnects = {}
    active = tuple(str(view) for view in active_views)
    camera_related = "camera" in text or bool(last_errors)
    if not camera_related:
        return None
    expected = {
        "wrist": str(getattr(established, "DEFAULT_WRIST_DEVICE", "")),
        "column": str(getattr(established, "DEFAULT_COLUMN_DEVICE", "")),
    }
    mentioned = set(_CAMERA_VIEW_PATTERN.findall(" ".join(row["message"] for row in chain)))
    rows = []
    for view in ("wrist", "column"):
        value = reconnects.get(view)
        try:
            value = int(value) if value is not None else None
        except (TypeError, ValueError):
            value = None
        rows.append(dict(view=view, active=view in active, device=expected[view],
                         mentioned=view in mentioned, reconnect_attempts=value,
                         last_error=None if last_errors.get(view) is None else str(last_errors[view])))
    failed = [row["view"] for row in rows if row["last_error"] is not None or row["mentioned"]]
    if not failed and camera_related:
        failed = [row["view"] for row in rows if row["active"]]
    return dict(
        schema="v21-camera-startup-diagnostic-v1",
        source="v21_live_hardware_startup",
        stage=str(stage),
        failure_reason=_camera_failure_reason(chain),
        failed_views=failed,
        active_views=list(active),
        cameras=rows,
        read_timeout_ms=float(getattr(established, "CAMERA_READ_TIMEOUT_MS", 1000.0)),
        recovery_attempts=int(getattr(established, "CAMERA_RECOVERY_ATTEMPTS", 0)),
        exception_chain=chain,
        motion_authorized=False,
    )


def attach_camera_startup_diagnostic(error: BaseException, *, io=None,
                                     active_views=("wrist", "column"), stage="startup"):
    """Attach a diagnostic to an exception while preserving its original type/message."""
    diagnostic = camera_startup_diagnostic(error, io=io, active_views=active_views, stage=stage)
    if diagnostic is not None:
        try:
            setattr(error, "camera_startup_diagnostic", diagnostic)
        except Exception:
            # Some extension exceptions disallow attributes; the original
            # exception remains the authoritative failure in that case.
            pass
    return error


def clearance_gate(kinematics, floor_mm=None, plan_mm=None):
    """The V17.7 table gate of a trial. floor_mm=None: the established -6.0 mm for planned paths and read-backs. A
    number (at least -10.0 mm): planned paths, read-backs and the physical floor all at that value (James 10/16, -8.0:
    the jaws resting on the table read -6.4 to -6.7 mm; audit doc section 14.12). plan_mm (with floor_mm): planned paths
    at least plan_mm instead; read-backs and the physical floor stay at floor_mm, and the gate's own monotonic recovery
    governs a source between the two (training overhaul doc section 17.91, James 10/24: near the table the arm ends 5-8 mm
    under its commands, so commands at the floor sank through it)."""
    if floor_mm is None:
        if plan_mm is not None:
            raise ValueError('a planning clearance comes with an explicit floor')
        return established.V177ClearanceGate(kinematics=kinematics, table_z_mm=established.TABLE_Z_MM,
            planned_clearance_mm=established.PLANNED_CLEARANCE_MM, readback_clearance_mm=established.READBACK_CLEARANCE_MM)
    floor_mm = float(floor_mm)
    plan_mm = floor_mm if plan_mm is None else float(plan_mm)
    return established.V177ClearanceGate(kinematics=kinematics, table_z_mm=established.TABLE_Z_MM,
        planned_clearance_mm=plan_mm, readback_clearance_mm=floor_mm, physical_floor_mm=floor_mm)


class TrialOwner:
    _physical_check = V203HardwareService._physical_check
    _lease_check = V203HardwareService._lease_check
    close = V203HardwareService.close

    def _halt(self):
        from fcpg.v21_hardware_state import verified_hold
        return self._record_hold(lambda events:verified_hold(self,actuator_events=events))

    def _completion_hold(self, goal, follow, q5):
        from fcpg.v21_hardware_state import completed_command_hold
        return self._record_hold(lambda events:completed_command_hold(
            self,goal,follow,q5,actuator_events=events))

    def _record_hold(self, operation):
        events=[]
        failure=None
        try:
            with self.bus_lock:
                started=time.monotonic()
                try:
                    result=operation(events)
                except BaseException as error:
                    history_for(self).observe_operation('hold',started,time.monotonic(),
                        error=f'{type(error).__name__}: {error}')
                    raise
                history_for(self).observe_operation('hold',started,time.monotonic(),hold=result)
                return result
        except BaseException as exc:
            failure=exc
            raise
        finally:
            # STOP never waits for filesystem work before its physical freeze,
            # and failed/partially returned writes still have durable evidence.
            self.last_actuator_events=events
            try:
                output=getattr(self,'output',None)
                if output is not None:
                    for event in events:
                        self.io._append_command_journal(output/'actuator-events.jsonl',event)
            except Exception as exc:
                if failure is None:raise
                note=f'actuator event persistence failed: {type(exc).__name__}: {exc}'
                self.last_actuator_persistence_error=note
                add_note=getattr(failure,'add_note',None)
                if callable(add_note):add_note(note)  # Python 3.10 has no BaseException.add_note.

    def _read_q(self):
        requested=time.monotonic()
        with self.bus_lock:
            entered=time.monotonic()
            try:
                q=self.io._read_state_only()
                history_for(self).observe_read(q,entered,time.monotonic())
                return q
            except BaseException as error:
                history_for(self).observe_read_failure(f'{type(error).__name__}: {error}',time.monotonic())
                raise
            finally:
                self.last_motor_read_timing=dict(lock_wait_s=entered-requested,
                    transport_s=time.monotonic()-entered)

    def _read_pre_dispatch_state(self):
        from fcpg.v21_hardware_state import pre_dispatch_actuator_state
        self._lease_check()
        with self.bus_lock:
            state=pre_dispatch_actuator_state(self.io.follower.bus,cancelled=self.stop_event.is_set)
            state['owner_observation_history']=history_for(self).snapshot(state['finished_s'])
            return state

    def _verify_policy_goal(self, command):
        from fcpg.v21_hardware_state import verify_policy_goal_position
        if self.stop_event.is_set():
            raise RuntimeError('policy goal verification cancelled before bus lock')
        self._lease_check()
        with self.bus_lock:
            return verify_policy_goal_position(self.io.follower.bus,command,
                                               cancelled=self.stop_event.is_set)

    def _finish_position(self, execution):
        if getattr(self, 'arm_position_servo', False):
            from fcpg.v21_arm_position_servo import finish_original_arm_goal as finish
            enabled = self.arm_position_servo
        else:
            from fcpg.v21_j3_position_servo import finish_original_j3_goal as finish
            enabled = self.j3_position_servo
        self._lease_check()
        with self.bus_lock:
            return finish(self, execution, enabled=enabled,
                cancelled=self.stop_event.is_set, clock=time.monotonic, wait=time.sleep)

    def _write_q(self, command, previous):
        requested=time.monotonic()
        with self.bus_lock:
            # STOP sets this event before waiting for the same bus lock.
            # A queued command may not resume motion after the STOP hold.
            if self.stop_event.is_set():
                raise RuntimeError('operator STOP is latched before motor write')
            self._v21_hold_cache=None  # Invalidate BEFORE a write that might partially reach hardware.
            entered=time.monotonic()
            cpu_entered=time.thread_time()
            try:
                returned=self.io._write_position_command(command, previous)
                history_for(self).observe_operation('position_goal_write',entered,time.monotonic(),
                    command=command,previous=previous,returned=returned)
                return returned
            except BaseException as error:
                history_for(self).observe_operation('position_goal_write',entered,time.monotonic(),
                    command=command,previous=previous,error=f'{type(error).__name__}: {error}')
                raise
            finally:
                ended=time.monotonic()
                cpu_elapsed=time.thread_time()-cpu_entered
                self.last_motor_io_timing=dict(lock_wait_s=entered-requested,transport_s=ended-entered,
                    caller_thread_cpu_s=cpu_elapsed,
                    off_cpu_or_io_wait_s=max(0.,ended-entered-cpu_elapsed),
                    off_cpu_reason_identified=False)

    def __init__(self, root, output, task, device, *, open_hardware=False,
                 column_memory=None, column_scene_unchanged=False, wrist_only=False,normal_follow_budget_s=0.,
                 gripper_force='lerobot', camera_condition=None, max_plan_age_s=300.,
                 j3_position_servo=False, arm_position_servo=False,
                 maximum_lateness_s=established.MAX_MODEL_CLOCK_LATENESS_S, clearance_floor_mm=None,
                 clearance_plan_mm=None, arm_p_coefficient=None):
        if type(j3_position_servo) is not bool or type(arm_position_servo) is not bool:
            raise TypeError('position servo flags must be a bool')
        from fcpg.v21_arm_gain import check_arm_p_coefficient
        # training overhaul doc section 17.92: shoulder lift and elbow flex at this P after the verified connect
        self.arm_p_coefficient = check_arm_p_coefficient(arm_p_coefficient)
        self.arm_gain = None
        if self.arm_p_coefficient is not None and (j3_position_servo or arm_position_servo):
            raise ValueError('the arm gain is not combined with a position servo mode')
        if j3_position_servo and arm_position_servo:
            raise ValueError('only one position servo mode may be enabled')
        if (j3_position_servo or arm_position_servo) and normal_follow_budget_s != 2.:
            raise ValueError('position servo requires the original two-second follow')
        self.j3_position_servo = j3_position_servo
        self.arm_position_servo = arm_position_servo
        if not np.isfinite(max_plan_age_s) or max_plan_age_s <= 0.:
            raise ValueError('max_plan_age_s must be finite and positive')
        if (isinstance(maximum_lateness_s, bool)
                or not isinstance(maximum_lateness_s, (int, float))
                or not np.isfinite(maximum_lateness_s)
                or not 0. < float(maximum_lateness_s) <= .25):
            raise ValueError('maximum_lateness_s must be finite and at most 0.25s')
        maximum_lateness_s = float(maximum_lateness_s)
        self.root, self.output, self.device = root, output, torch.device(device)
        self.bus_lock = threading.RLock(); self.stop_event = threading.Event()
        self.io = self.lease = None; self.frames = {}; self.acquisitions = []
        self.lens_cover = PaperCoverHold()
        self.lens_cover_status = {}
        self.source_evidence_store = SourceEvidenceStore()
        self.active_views = ('wrist',) if wrist_only else ('wrist','column')
        self.saved_column = None
        if column_memory is not None:
            if ENCODER != 'dinov2_vits14':
                raise ValueError('the saved Column holds DINOv2 tokens; a DINOv3-L model needs the live Column')
            if not column_scene_unchanged:
                raise ValueError('cached Column use requires operator confirmation of unchanged camera/box placement')
            self.saved_column = load_saved_column(column_memory,
                now=torch.tensor([time.monotonic()],device=self.device,dtype=torch.float64),device=self.device)
            self.active_views = ('wrist',)
        self.kinematics = established.SO101MuJoCoKinematics(established.DEFAULT_KINEMATICS_MODEL)
        self.clearance = clearance_gate(self.kinematics, clearance_floor_mm, clearance_plan_mm)
        self.runtime = load_runtime(root, task, device=device, physical_check=self._physical_check,
                                    lease_check=self._lease_check, max_plan_age_s=float(max_plan_age_s),
                                    arm_p_coefficient=self.arm_p_coefficient)     # 17.153: the LeWM's setting
        self.object_relative = None
        if camera_condition is not None and getattr(camera_condition, 'h0_committed', False):
            raise ValueError("camera condition H0 is committed only after empty memory accepts a seed")
        self.camera_condition = camera_condition
        self.last_proposed_camera_condition = None
        # the object cue's encoder (a model with config.object_relative) is installed by the service
        # (V21LiveService._install_object_finder: the learned finder or SAM 3); the operator colour encoder that stood
        # here is gone from the code (10/31, James 「不要的废弃代码 这种颜色函数删掉」)
        if (self.runtime.text_ids == 1).any():
            raise ValueError("trial instruction has untrained vocabulary; use the recorded task wording")
        if not open_hardware:
            return
        # V21_DINOV2=off (section 17.57): a service that only computes the model's own DINOv3-L tokens
        self.dino = None if DINOV2_OFF else load_frozen_dinov2('/home/james/.cache/torch/hub/facebookresearch_dinov2_main',
            '/home/james/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth', device=self.device)
        # a DINOv3-L model reads its own encoder's tokens beside the DINOv2 ones (fcpg/v21_live_encoder.py, 17.55)
        self.model_tokens = None if ENCODER == 'dinov2_vits14' else ModelTokens(self.device, dinov2=self.dino)
        try:
            self.lease = established.acquire_so101_hardware_lease(lock_path=established.DEFAULT_HARDWARE_LOCK,
                serial_device=established.DEFAULT_ROBOT_PORT, wrist_device=established.DEFAULT_WRIST_DEVICE,
                column_device=established.DEFAULT_COLUMN_DEVICE)
            args = SimpleNamespace(output=output, robot_port=established.DEFAULT_ROBOT_PORT, robot_id='myfollower01',
                wrist_device=established.DEFAULT_WRIST_DEVICE, column_device=established.DEFAULT_COLUMN_DEVICE,
                calibration_dir=None, max_command_jump=float(np.max(established.SO101_POSITION_UPPER-
                    established.SO101_POSITION_LOWER)+1e-4), settle_timeout_s=20., settle_position_tolerance=3.,
                settle_step_tolerance=.25, settle_samples=3, self_contact_calibration=False,
                i_accept_empty_jaw_self_contact=False, experiment_binding=None, active_camera_views=self.active_views)
            from fcpg.v21_hardware_state import existing_owner_connection_initializer,ORDER
            args.connection_initializer=existing_owner_connection_initializer(self,gripper_force=gripper_force,
                arm_p_coefficient=self.arm_p_coefficient)
            try:
                self.io = established.V177HardwareSO101IO(args, clearance_gate=self.clearance,
                    gripper_calibration_closed=established.DEFAULT_GRIPPER_CLOSED_POSITION,
                    gripper_calibration_open=established.DEFAULT_GRIPPER_OPEN_POSITION, endpoint_envelope=None)
            except BaseException as error:
                # Preserve the original constructor error while making a
                # camera-open failure diagnosable by the loopback dashboard.
                attach_camera_startup_diagnostic(error, active_views=self.active_views,
                                                 stage="hardware_open")
                raise
            connected=self.io.safe_energize_preseed
            if connected.get('connection_mode')=='warm_preserved_goal' and not connected.get('arm_recovery'):
                self._v21_hold_cache=dict(goal_raw=connected['goal_raw'],
                    reference_present=[connected['observed_position'][k] for k in ORDER],
                    held_position=[connected['holding_position'][k] for k in ORDER])
            from fcpg.v21_gripper_force import apply_gripper_force_profile
            self.gripper_force = apply_gripper_force_profile(self.io.follower.bus, gripper_force)
            if self.arm_p_coefficient is not None:
                from fcpg.v21_arm_gain import apply_arm_p_coefficient
                with self.bus_lock:
                    self.arm_gain = apply_arm_p_coefficient(self.io.follower.bus, self.arm_p_coefficient)
                write_json(Path(output) / 'arm-gain.json', self.arm_gain)
            self.runtime.configure_source_update_tolerance(self.io.settle_step_tolerance)
            # Live: never abandon a block for pose drift or plan age. Recomputable
            # plans are re-derived from the actual q at dispatch instead.
            self.runtime.configure_pre_execution_recompute('always')
            self.bridge = V21TransactionBridge(self.runtime, read_q=self.read_checked, write_q=self._write_q,
                capture_endpoint=self.capture, physical_check=self._physical_check, halt=self._halt,
                completion_hold=self._completion_hold, request_stop=self.stop_event.set,
                cancelled=self.stop_event.is_set, wait=self.stop_event.wait,
                write_metrics=lambda:getattr(self,'last_motor_io_timing',None),
                read_metrics=lambda:getattr(self,'last_motor_read_timing',None),
                read_pre_dispatch_state=self._read_pre_dispatch_state,
                verify_policy_goal=self._verify_policy_goal,
                finish_position=self._finish_position if self.j3_position_servo or self.arm_position_servo else None,
                normal_follow_budget_s=normal_follow_budget_s,
                follow_position_tolerance=self.io.settle_position_tolerance,
                follow_step_tolerance=self.io.settle_step_tolerance,
                follow_quiet_samples=self.io.settle_samples,
                maximum_lateness_s=maximum_lateness_s)
            if self.saved_column is not None:
                self.restore_column_memory()
        except BaseException:
            self.close(); raise

    def read_checked(self):
        q = self._read_q()
        diagnostic=dict(q_position=np.asarray(q).tolist(),time_s=time.monotonic(),
                        clearance_validated=False)
        self.last_readback_diagnostic=diagnostic
        try:
            self.clearance.require_readback(q, label="V21 actual readback")
        except Exception as exc:
            self.last_readback_diagnostic=dict(diagnostic,error=f'{type(exc).__name__}: {exc}')
            raise
        self.last_readback_diagnostic=dict(diagnostic,clearance_validated=True)
        return q

    def restore_column_memory(self):
        if self.saved_column is not None:
            old = self.saved_column
            store = getattr(self, 'source_evidence_store', None)
            if store is None:
                store = self.source_evidence_store = SourceEvidenceStore()
            rgb = np.zeros((2, 224, 224, 3), np.uint8)
            rgb[1] = old.rgb
            store.remember(rgb, old.observation, acquisition=dict(
                source="validated-saved-actual-column", source_path=old.source_path,
                original_clock_preserved=True, clock="host-camera-read-finished-not-sensor-exposure"))
            self.runtime.observe(EncodedEndpoint(old.source_q,old.source_time,old.observation))

    def capture(self):
        self._lease_check()
        delivered = {}
        def camera_identity(view):
            expected = Path(established.DEFAULT_WRIST_DEVICE if view=='wrist' else established.DEFAULT_COLUMN_DEVICE)
            if not expected.exists():
                raise CameraDeliveryUnavailable('known_camera_device_missing',str(expected))
            try:
                resolved = expected.resolve(strict=True)
            except FileNotFoundError as exc:
                raise CameraDeliveryUnavailable('known_camera_device_missing',str(expected)) from exc
            return dict(schema='v21-camera-identity-v1',view=view,
                        stable_device=str(expected),resolved_device=str(resolved))
        def read_camera(view):
            expected = Path(established.DEFAULT_WRIST_DEVICE if view=='wrist' else established.DEFAULT_COLUMN_DEVICE)
            if not expected.exists():
                raise CameraDeliveryUnavailable('known_camera_device_missing',str(expected))
            try:
                frame = self.io._read_camera(view)
            except RuntimeError as exc:
                # Only a verified absence of the expected stable device may
                # become missing data. Wrong identity/geometry still fail.
                if not expected.exists():
                    raise CameraDeliveryUnavailable('known_camera_device_missing',str(expected)) from exc
                raise
            delivered[view] = frame
            return frame
        try:
            with self.io.camera_lock:
                acquired = capture_actual_views(read_camera,self.read_checked,active_views=self.active_views,
                    camera_identity_check=camera_identity,
                    read_raw_frame=lambda view:getattr(self.io,'_last_raw_frames',{}).get(view,delivered[view]))
        except BaseException as error:
            # The active transport records bounded reconnect details on its IO
            # object. Attach those details before V21LiveService closes it so
            # startup diagnostics retain the failed view and last read error.
            attach_camera_startup_diagnostic(error, io=self.io, active_views=self.active_views,
                                             stage="initial_observation")
            raise
        rgb = acquired.rgb.copy()
        q_images = acquired.q_at_image.copy()
        times = acquired.image_times_s.copy()
        valid = acquired.measurement_available.copy()
        cover_now = time.monotonic()
        cover_status = {}
        for index, view in enumerate(("wrist", "column")):
            if not bool(valid[index]):
                cover_status[view] = dict(covered=False, retained_real_frame=False,
                                          age_s=None, within_paper_window=False)
                continue
            covered = frame_is_covered(rgb[index])
            decision = self.lens_cover.update(view, rgb[index], covered, cover_now)
            cover_status[view] = {key: decision[key] for key in
                                  ("covered", "retained_real_frame", "age_s", "within_paper_window")}
            if not covered:
                continue
            valid[index] = False
            q_images[index] = 0
            times[index] = 0
            rgb[index] = 0
            if decision["within_paper_window"]:
                retained = self.lens_cover.retained_rgb(view)
                if retained is not None:
                    rgb[index] = retained
        self.lens_cover_status = cover_status
        self.frames = acquired.raw_frames
        dino = None
        if not DINOV2_OFF:
            dino = np.zeros((2,256,384),np.float16)
            if valid.any():
                dino[valid] = extract_dinov2_tokens(self.dino, rgb[valid], device=self.device, batch_size=2)
        # the learned object finder (training overhaul doc section 17.34) reads this exposure's own wrist tokens;
        # None with V21_DINOV2=off (section 17.57): every finder then reads last_model_tokens
        self.last_dino_tokens = dino
        # the model reads its own encoder's tokens: these DINOv2 ones, or DINOv3-L (fcpg/v21_live_encoder.py, 17.55)
        tokens = dino
        prefix = None
        policy_tokens, policy_look = None, None
        if ENCODER != 'dinov2_vits14':
            if getattr(self, 'model_tokens', None) is None:      # an owner without hardware (offline replays)
                self.model_tokens = ModelTokens(self.device, dinov2=self.dino)
            tokens = np.zeros((2, PATCHES, DIM), np.float16)
            # the same pictures' prefix tokens, for the task-text finder's vision head (fcpg/v21_text_finder.py)
            prefix = np.zeros((2, 5, DIM), np.float16)
            if valid.any():
                tokens[valid] = self.model_tokens.encode(rgb[valid])
                if getattr(self.model_tokens, 'last_prefix', None) is not None:
                    prefix[valid] = self.model_tokens.last_prefix
            if recording_look_enabled():
                # section 17.64: the policy (Actor, LeWM) was trained on recordings; it reads the recording look of the
                # same raw frames (fcpg/v21_recording_look.py); the finders and judges keep the live letterbox below
                if getattr(self, 'recording_look', None) is None:
                    self.recording_look = RecordingLook()
                pictures, policy_look = policy_pictures(self.recording_look, acquired.raw_frames, valid)
                policy_tokens = np.zeros((2, PATCHES, DIM), np.float16)
                if valid.any():
                    policy_tokens[valid] = self.model_tokens.encode(pictures[valid])
        # a learned finder or judge made on the model's encoder reads these (fcpg/v21_live_service.py, 17.56)
        self.last_model_tokens = tokens
        self.last_model_prefix = prefix
        relative = place = None
        if getattr(self, 'object_relative', None) is not None:
            wrist_held = bool((self.lens_cover_status.get("wrist") or {}).get("within_paper_window"))
            if valid[0]:
                vector = self.object_relative.encode(rgb[0], q_images[0])
                self.last_object_relative = vector.tolist()
            elif wrist_held:
                vector = np.asarray(getattr(self, "last_object_relative", np.zeros(8, np.float32)), np.float32)
            else:
                vector = np.zeros(8, np.float32)
                if hasattr(self.object_relative, "clear_observation"):
                    self.object_relative.clear_observation()
                self.last_object_relative = vector.tolist()
            diagnostic=(getattr(self.object_relative,'last_diagnostic_vector',vector)
                        if valid[0] or wrist_held else np.zeros(8,np.float32))
            self.last_object_diagnostic=np.asarray(diagnostic,np.float32).tolist()
            relative = torch.as_tensor(vector, device=self.device)[None]
            # Where the object goes (training overhaul doc section 17): the same encoder, the same wrist picture and
            # jaw pair, only for a model that reads it; a covered wrist keeps the last one like the object token.
            if getattr(getattr(self.runtime, 'core', None), 'config', None) is not None and \
                    getattr(self.runtime.core.config, 'place_relative', False):
                if valid[0]:
                    place_vector = self.object_relative.encode_place(rgb[0], q_images[0])
                elif wrist_held:
                    place_vector = np.asarray(getattr(self, 'last_place_relative', np.zeros(8, np.float32)), np.float32)
                else:
                    place_vector = np.zeros(8, np.float32)
                self.last_place_relative = np.asarray(place_vector, np.float32).tolist()
                place = torch.as_tensor(place_vector, device=self.device)[None]
        # The object encoder also runs a visual backbone. Bind current q/time
        # only after ALL image computation, without retimestamping either view.
        q = self.read_checked(); now = time.monotonic()
        goal = None
        if getattr(getattr(getattr(self, 'runtime', None), 'core', None), 'config', None) is not None and \
                getattr(self.runtime.core.config, 'place_goal', False):
            # the place goal (training overhaul doc section 17.7) against the jaws at this very readback; zeros (no goal)
            # until the service has located the box
            source = getattr(self, 'place_goal', None)
            vector = np.zeros(8, np.float32) if source is None else np.asarray(source.vector(q), np.float32)
            self.last_place_goal = vector.tolist()
            goal = torch.as_tensor(vector, device=self.device)[None]
        self.acquisitions.append(dict(clock="host-read-brackets-not-sensor-exposure", views=acquired.views,
            measurement_available=valid.tolist(), lens_cover=self.lens_cover_status,
            q_association="post actual readback accepted only inside fresh pre/post pose-stability envelope; no interpolation",
            acquisition_final_readback_sequence_id=acquired.q_sequence_id,
            current_q_time_s=now, current_q_after_all_visual_encoding=True,
            policy_look=policy_look if policy_tokens is not None else 'live letterbox'))
        del self.acquisitions[:-64]
        obs = VisualObservation(torch.as_tensor(tokens if policy_tokens is None else policy_tokens, device=self.device)[None],
            torch.as_tensor(q_images, device=self.device)[None],
            torch.as_tensor(times, device=self.device, dtype=torch.float64)[None],
            torch.as_tensor(valid, device=self.device)[None], object_relative=relative,
            **({} if place is None else dict(place_relative=place)),
            **({} if goal is None else dict(place_goal=goal)))
        # Optional filter is a proposal. Capture never calls runtime.observe
        # and never retains next_condition. Preview discards it; the caller
        # commits H0 only after that seed enters empty actual memory.
        obs = self._apply_camera_condition(obs)
        endpoint = EncodedEndpoint(torch.as_tensor(q, device=self.device)[None],
                                   torch.tensor([now], device=self.device, dtype=torch.float64), obs)
        # Preserve source pixels together with their original clocks/q/features.
        # A later endpoint capture may replace last_rgb before execution is saved.
        encoder = getattr(self, 'object_relative', None)
        current_wrist = bool(obs.measurement_available[0, 0])
        provenance = object_token_provenance(
            obs.object_relative, wrist_measurement_available=current_wrist,
            encoder_enabled=encoder is not None,
            policy_record=getattr(encoder, 'last_policy_record', None),
            diagnostic_record=getattr(encoder, 'last_record', None),
            image_time_s=float(obs.image_times[0, 0]) if current_wrist else None,
            q_at_image=obs.q_at_image[0, 0] if current_wrist else None)
        store = getattr(self, 'source_evidence_store', None)
        if store is None:  # Also supports model-only/injected owner fixtures.
            store = self.source_evidence_store = SourceEvidenceStore()
        store.remember(rgb, obs, acquisition=self.acquisitions[-1], object_provenance=provenance)
        self.last_object_token_provenance = provenance
        self.last_rgb, self.last_endpoint = rgb.copy(), endpoint
        return endpoint

    def actual_memory_is_empty(self):
        state = getattr(getattr(self, 'runtime', None), 'state', None)
        if state is None:
            return True
        memory = getattr(state, 'memory', None)
        if memory is None:
            return True
        valid = getattr(memory, 'valid', None)
        if not isinstance(valid, torch.Tensor):
            return False
        return not bool(valid.any())

    def _apply_camera_condition(self, observation):
        condition = getattr(self, 'camera_condition', None)
        if condition is None:
            self.last_proposed_camera_condition = None
            self._h0_seed_from_empty_memory = False
            return observation
        filtered, proposed = condition.filter(observation)
        self.last_proposed_camera_condition = proposed
        self._h0_seed_from_empty_memory = self.actual_memory_is_empty()
        return filtered

    def discard_proposed_camera_condition(self):
        """Preview path: do not retain next_condition."""
        self.last_proposed_camera_condition = None
        self._h0_seed_from_empty_memory = False

    def commit_camera_condition_h0(self):
        """Retain last filter proposal after the seed entered empty actual memory."""
        condition = getattr(self, 'camera_condition', None)
        if condition is None:
            self.discard_proposed_camera_condition()
            return None
        if condition.h0_committed:
            self.discard_proposed_camera_condition()
            return condition
        proposed = getattr(self, 'last_proposed_camera_condition', None)
        if proposed is None:
            raise ValueError("H0 commit requires a proposed camera condition")
        if not getattr(self, '_h0_seed_from_empty_memory', False) or self.actual_memory_is_empty():
            raise ValueError("H0 commit requires accepting the seed into empty actual memory")
        self.camera_condition = proposed
        self.discard_proposed_camera_condition()
        return proposed

    def images(self, name):
        for view, rgb in self.frames.items():
            cv2.imwrite(str(self.output/f'{name}-{view}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if self.saved_column is not None:
            cv2.imwrite(str(self.output/f'{name}-column-historical.png'),cv2.cvtColor(self.saved_column.rgb,cv2.COLOR_RGB2BGR))


def cpu_actual_state(value):
    from fcpg.v21_picture_memory import PictureMemory, snapshot_pictures
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,PictureMemory): return snapshot_pictures(value)    # 17.167: times, joints, validity; no tokens
    if is_dataclass(value): return type(value)(**{f.name:cpu_actual_state(getattr(value,f.name)) for f in fields(value)})
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', default='Pick up the red fish toy and place it in the cardboard box')
    p.add_argument('--device', default='cpu')
    p.add_argument('--execute-five', action='store_true')
    p.add_argument('--operator-present', action='store_true')
    p.add_argument('--workspace-clear', action='store_true')
    p.add_argument('--estop-ready', action='store_true')
    p.add_argument('--column-memory', type=Path)
    p.add_argument('--column-scene-unchanged', action='store_true')
    args = p.parse_args()
    if args.execute_five and not (args.operator_present and args.workspace_clear and args.estop_ready):
        raise ValueError('fresh on-site confirmation is required for the five-step trial')
    if args.device.startswith('cuda') and os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('use physical GPU1 only')
    if args.output.exists():
        raise ValueError('each trial needs a new output directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    owner = None
    result = dict(model_version=VERSION, policy_command_budget=5, hardware_opened=False,
                  hardware_opening_requested=args.execute_five, task=args.task, success_not_assessed=True)
    try:
        owner = TrialOwner(args.model_output, args.output, args.task, args.device, open_hardware=args.execute_five,
                           column_memory=args.column_memory,column_scene_unchanged=args.column_scene_unchanged)
        result.update(strict_load_passed=True, hardware_opened=owner.io is not None)
        if owner.saved_column is not None:
            result['column_memory'] = dict(path=owner.saved_column.source_path,
                image_time_s=float(owner.saved_column.observation.image_times[0,1]),
                initial_age_s=owner.saved_column.age_s,measurement_available_now=False,
                scene_unchanged_confirmed=args.column_scene_unchanged)
        if args.execute_five:
            def stop(_sig, _frame):
                owner.stop_event.set()
            for number in (signal.SIGINT, signal.SIGTERM): signal.signal(number, stop)
            with torch.inference_mode():
                # Warm up before binding any physical action clock. Planning
                # performs mechanical admission and is not an execution.
                owner.bridge.plan(); owner.images('before')
                owner.bridge.plan()  # fresh measured source after warm-up/file output
                result['plan'] = owner.runtime.pending.trace.positions.cpu().tolist()
                result['trial'] = owner.bridge.execute()
                owner.images('after')
                result['runtime'] = owner.runtime.status()
                # Inspect the next policy output using the just-completed REAL
                # history, without arming or issuing a second action block.
                offsets=torch.arange(5,device=owner.device,dtype=torch.float64)[None]*.1
                result['next_readonly_preview'] = owner.runtime.core.act(
                    owner.runtime.state,owner.runtime.text_ids,offsets).commands_position.cpu().tolist()
                torch.save(dict(version=owner.runtime.status()['version'],contract=owner.runtime.status()['contract'],
                    kind='actual-state-only',model_output=str(args.model_output.resolve()),
                    state=cpu_actual_state(owner.runtime.state),stopped=True,pending=False),args.output/'actual-state.pt')
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        if owner is None and args.execute_five:
            result['hardware_opened'] = None  # constructor may have failed after opening the bus
        raise
    finally:
        if owner is not None:
            if hasattr(owner, 'bridge'):
                result['attempt'] = owner.bridge.last_attempt
                owner.stop_event.set()
                try: owner.bridge.stop()
                except Exception as exc: result['stop_error'] = f'{type(exc).__name__}: {exc}'
            result['acquisitions'] = owner.acquisitions
            try: owner.close()
            except Exception as exc: result['close_error'] = f'{type(exc).__name__}: {exc}'
        write_json(args.output/'trial.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result.get('stop_error') or result.get('close_error'):
            raise RuntimeError('hardware cleanup did not complete; inspect trial.json')


if __name__ == '__main__':
    main()
