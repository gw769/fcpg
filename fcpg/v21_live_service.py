"""Resident V21.0.2 service. GETs never update model memory or send commands.

BIAS NOTICE: enabled assistance can override/supplement Actor commands using
task-specific targets and control rules. Assisted outcomes are NOT autonomous
FCPG/LeWM/Actor evidence. See docs/ASSISTED_PATH_BIAS.md; safety checks remain.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import torch

from scripts.canary_so101_v21_research import TrialOwner, established, cpu_actual_state
from .v21_camera_condition import ADDRESS_MODES, CONDITIONS, V21CameraCondition
from .v21_core import VERSION, CONTRACT, V21ResearchCore
from .v21_encoder import DINOV2_OFF, ENCODER
from .v21_training import write_json
from .v21_home import clear_pending_home_receipt, execute_home_readback_chunk, plan_home_chunk
from .prod_v20_3_hardware_service import V203HardwareService
from .prod_v17_7_dino_targets import locate_dino_targets
from .v21_target_tracker import V21_TARGET_SITES, V21TargetTracker, load_wrist_drop_zone
from .so101_camera_calibration import ProjectionFit, approximate_column_backprojection_on_plane, predict_projection
from .v21_visual_alignment import alignment_observation
from .v21_alignment_session import AlignmentSession,observation_payload
from .v21_lateral_alignment import LateralAlignmentSession
from .v21_auxiliary_budget import AuxiliaryMotionBudget
from .v21_actual_resume import load_actual_resume,resume_actual_state,save_actual_resume
from .v21_vertical_approach import VerticalApproachSolver,approach_geometry
from .v21_tip_display import (
    COLUMN_TIP_SOURCE, WRIST_TIP_SOURCE, TipDisplay, tip_source_matches_view,
)
from .v21_column_dino_transport import (
    ColumnDinoFGSample, bind_column_dino_fg, fit_column_dino_fg_transport,
    short_window_keeps_actual_response,
    require_actual_cem_response, require_actual_probe_response,
)
from .v21_gripper_check import empty_gripper_follow_check
from .v21_assistance_state import AssistanceStateCoordinator
from .v21_capped_trial import CappedTrial
from .v21_bus_faults import BusFaultLog
from .v21_runtime import EncodedEndpoint, PreExecutionStateChanged
from .v21_object_relative import assistance_cue
from .v21_centroid_tracking import error_px as centroid_error_px, gate_closure, progress as centroid_progress
from .v21_execution_geometry import (
    GRASP_LIFT_M, TABLE_Z_M, project_trial_geometry, project_vertical_pinch,
    project_fk_m_toward_xyz, project_wrist_m_toward_pixel, fk_midpoint_m)

TASK = 'Pick up the red fish toy and place it in the cardboard box'
WRIST_REFERENCE = Path('/home/james/so101-vda-disk/feedback-corrected-grounding/outputs/lerobot/so101/my_first_task_clean_v1_dual_camera_calibration/calibration_letterbox_v1.json')
WRIST_CAMERA_REFERENCE = Path('/home/james/so101-vda-disk/feedback-corrected-grounding/outputs/lerobot/so101/prod_v17_2_metric_calibration/metric-calibration.json')
FISH_RELEASE_REFERENCE = Path(__file__).resolve().parents[1] / 'configs/v21_fish_release_reference.json'
# The legacy V17.4 fixed fish-task HOME with the live open gripper. --home-state replaces it with a file whose
# current_state gives all six joints (docs/V21_SYSTEM_LEWM_CEM_AUDIT_20261013.md section 14.9).
LEGACY_HOME_STATE = Path('/home/james/feedback-corrected-grounding-v203/configs/prod_v17_7_legacy_v17_4_home.json')
LEGACY_HOME_GRIPPER = 31.340341567993164
_BLOCK_IO_OBJECT_RELATIVE_UNSET = object()
_OBJECT_RELATIVE_PROVENANCE_UNSET = object()


def exposure_tokens(owner, encoder):
    """This exposure's tokens [2, P, D] in the encoder a learned finder or judge was made on: the service's DINOv2
    (TrialOwner.last_dino_tokens, always computed) or the model's own (TrialOwner.last_model_tokens, DINOv3-L in a
    V21_ENCODER=dinov3_vitl16 service; training overhaul doc section 17.56)."""
    if encoder == 'dinov2_vits14':
        if DINOV2_OFF:
            raise ValueError('a finder made on DINOv2 in a V21_DINOV2=off service (section 17.57)')
        return owner.last_dino_tokens
    if encoder != ENCODER:
        raise ValueError(f'a finder made on {encoder} needs a V21_ENCODER={encoder} service, this one runs {ENCODER}')
    return owner.last_model_tokens


def _compact_position_servo_status(record):
    """Keep dashboard facts while leaving high-rate servo samples in the artifact."""
    if not isinstance(record, dict):
        return deepcopy(record)
    heavy = {'axes', 'readbacks', 'parameter_writes', 'original_goal_hold'}
    result = {key: deepcopy(value) for key, value in record.items() if key not in heavy}
    axes = record.get('axes')
    if isinstance(axes, dict):
        result['axes'] = {
            name: {key: deepcopy(axis[key]) for key in (
                'axis', 'identity', 'original_target', 'applied', 'completed',
                'corridor_precheck', 'integral_exposure', 'reason') if key in axis}
            for name, axis in axes.items() if isinstance(axis, dict)
        }
    hold = record.get('original_goal_hold')
    if isinstance(hold, dict):
        result['original_goal_hold'] = {
            key: deepcopy(value) for key, value in hold.items()
            if key not in {'readbacks', 'snapshot_attempts'}
        }
        result['original_goal_hold']['readback_count'] = len(hold.get('readbacks') or ())
        result['original_goal_hold']['snapshot_attempt_count'] = len(
            hold.get('snapshot_attempts') or ())
    result['telemetry_omitted_from_status'] = True
    result['telemetry_counts'] = dict(
        axes=len(axes) if isinstance(axes, dict) else 0,
        readbacks=len(record.get('readbacks') or ()),
        parameter_writes=len(record.get('parameter_writes') or ()),
    )
    return result


def _compact_execution_status(record):
    """Bound the 10 Hz ROS status payload without altering persisted evidence."""
    if not isinstance(record, dict):
        return deepcopy(record)
    return {
        key: (_compact_position_servo_status(value)
              if key == 'position_servo' else deepcopy(value))
        for key, value in record.items()
    }


class V21LiveService:
    def __init__(self, root, output, *, device='cuda:0', task=TASK, owner_factory=TrialOwner,
                 clock=time.monotonic, target_diagnostics=True, column_memory=None, column_scene_unchanged=False,
                 alignment_motion_enabled=False,wrist_reference=WRIST_REFERENCE,wrist_only=False,
                 gripper_check_enabled=False,normal_follow_budget_s=0.,gripper_force='lerobot',
                 grounding_correction=False,grounding_px_per_unit=None,head_grasp=False,grasp_target='body',
                 record_policy_inputs=False,alignment_scope='full',resume_auxiliary_state=None,
                 camera_condition='both_fresh',camera_address='factual',raw_actor=False,
                 lewm_cem=False,column_dino_fg_calibration=False,
                 lewm_cem_objective='column_visual',lewm_shadow_checkpoints=(),
                 lewm_shadow_parent_sha256=None,trial_max_blocks=16,
                 lewm_cem_candidate_parameterization=None, lewm_cem_action_scale=1.0,
                 lewm_control_checkpoint=None,lewm_control_parent_sha256=None,
                 lewm_control_candidate_sha256=None, max_plan_age_s=None,
                 hardware_maximum_lateness_s=None, replan_on_execute=True,
                 temporal_profile=None, j3_pid_diagnostic=False,
                 j3_pid_continuation_baseline=None, j3_pid_continuation_variant=None,
                 j3_range_diagnostic=False, j3_range_amplitude=5,
                 j3_compensation_diagnostic=False, j3_native_response_diagnostic=False,
                 same_goal_p32_source=None, same_goal_i1_source=None,
                 same_goal_integral_max_seconds=2., j3_position_servo=False, arm_position_servo=False,
                 wrist_grasp_calibration=None, c4_planner=None, c4_planner_factory=None,
                 alignment_cem=None, alignment_response=None, alignment_cem_options=None,
                 explore_actor=False, explore_seed=0, home_state=None, clearance_floor_mm=None,
                 wrist_drop_zone=None, place_goal_camera=None, object_finder=None, clearance_plan_mm=None,
                 home_lift_below_floor=False, arm_p_coefficient=None):
        if type(home_lift_below_floor) is not bool:
            raise TypeError('home_lift_below_floor is a bool')
        # training overhaul doc section 17.91: HOME first lifts one joint when the arm stopped just under the floor
        self.home_lift_below_floor = home_lift_below_floor
        self.home_state_path = None if home_state is None else Path(home_state)
        if self.home_state_path is not None:
            self._home_target()                     # a broken HOME file fails here, before any hardware opens
        # fcpg/v21_target_tracker.py: the drop zone's wrist-view registration (training overhaul doc section 17.6);
        # None keeps the column registration's role prototype in the wrist. A broken file fails here too.
        if object_finder is not None and object_finder.get('box_model') is not None and wrist_drop_zone is not None:
            raise ValueError('the learned box finder (section 17.57) replaces the wrist drop-zone registration')
        self.wrist_drop_zone = None if wrist_drop_zone is None else load_wrist_drop_zone(wrist_drop_zone)
        self.dinov2_forbidden = []
        if DINOV2_OFF:
            # section 17.57 (James 10/22 「都用阿」): only DINOv3-L in this service; every learned part reads it
            if (object_finder is None or object_finder.get('kind') != 'learned'
                    or object_finder.get('box_model') is None):
                raise ValueError('a V21_DINOV2=off service needs the learned finder and the learned box finder')
            from .v21_live_encoder import forbid_dinov2
            self.dinov2_forbidden = forbid_dinov2()
        if type(j3_position_servo) is not bool or type(arm_position_servo) is not bool:
            raise TypeError('position servo flags must be a bool')
        if c4_planner is not None:
            # docs/V21_LEWM_C4_CONTROL_PLAN_20261011.md: C4 plans around the unassisted native40
            # Actor; no old LeWM CEM, no hand-written assistance, no LeWM control/shadow candidates.
            if c4_planner not in ('shadow', 'control') or c4_planner_factory is None:
                raise ValueError('C4 planner mode must be shadow or control, with its planner factory')
            if (lewm_cem or raw_actor or grounding_correction or head_grasp or alignment_motion_enabled
                    or column_dino_fg_calibration or gripper_check_enabled or lewm_shadow_checkpoints
                    or lewm_control_checkpoint is not None or resume_auxiliary_state is not None
                    or column_memory is not None or wrist_only):
                raise ValueError('C4 planning runs on the unassisted native40 Actor only')
        elif c4_planner_factory is not None:
            raise ValueError('a C4 planner factory needs a C4 planner mode')
        if alignment_cem is not None:
            # fcpg/v21_alignment_cem.py: the paper's CEM on the system LeWM, around the unassisted Actor.
            if alignment_cem not in ('shadow', 'control') or alignment_response is None:
                raise ValueError('alignment CEM mode must be shadow or control, with a measured wrist response')
            if (c4_planner is not None or lewm_cem or raw_actor or grounding_correction or head_grasp
                    or alignment_motion_enabled or column_dino_fg_calibration or gripper_check_enabled
                    or lewm_shadow_checkpoints or lewm_control_checkpoint is not None
                    or resume_auxiliary_state is not None or column_memory is not None or wrist_only):
                raise ValueError('alignment CEM runs on the unassisted Actor only')
        elif alignment_response is not None or alignment_cem_options:
            raise ValueError('an alignment response or options need an alignment CEM mode')
        if explore_actor and (c4_planner is not None or alignment_cem is not None or lewm_cem or raw_actor
                              or grounding_correction or head_grasp or alignment_motion_enabled
                              or column_dino_fg_calibration or gripper_check_enabled or lewm_shadow_checkpoints
                              or lewm_control_checkpoint is not None or resume_auxiliary_state is not None
                              or column_memory is not None or wrist_only):
            # fcpg/v21_explore.py: random arm offsets around the unassisted Actor, for recording only.
            raise ValueError('exploration runs on the unassisted Actor only')
        if j3_position_servo and arm_position_servo:
            raise ValueError('only one position servo mode may be enabled')
        if (j3_position_servo or arm_position_servo) and normal_follow_budget_s != 2.:
            raise ValueError('position servo requires the original two-second follow')
        self.j3_position_servo = j3_position_servo
        self.arm_position_servo = arm_position_servo
        if same_goal_p32_source is not None and same_goal_i1_source is not None:
            raise ValueError('only one same-goal diagnostic mode may be enabled')
        if (isinstance(same_goal_integral_max_seconds, bool)
                or not isinstance(same_goal_integral_max_seconds, (int, float))
                or same_goal_integral_max_seconds not in (2., 10.)):
            raise ValueError('same-goal integral_max_seconds must be 2 or 10')
        if same_goal_integral_max_seconds != 2. and same_goal_i1_source is None:
            raise ValueError('extended same-goal integral_max_seconds requires an explicit I1 source')
        if (isinstance(lewm_cem_action_scale, bool)
                or not isinstance(lewm_cem_action_scale, (int, float))
                or lewm_cem_action_scale not in (1.0, 1.5, 3.0)):
            raise ValueError('lewm_cem_action_scale must be 1.0, 1.5, or 3.0')
        if lewm_cem_action_scale != 1.0 and (not lewm_cem or raw_actor
                or lewm_cem_objective != 'actor_response_tracking'
                or lewm_cem_candidate_parameterization not in ('amplified_actor_sequence', 'smooth_actor_endpoint')):
            raise ValueError('nondefault lewm_cem_action_scale requires LeWM CEM '
                             'actor_response_tracking with amplified_actor_sequence')
        if (lewm_cem_candidate_parameterization == 'amplified_actor_sequence'
                and lewm_cem_action_scale == 1.0):
            raise ValueError('amplified_actor_sequence requires nondefault lewm_cem_action_scale')
        from .v21_j3_pid_service import J3PIDServiceCoordinator, DEFAULT_SEAL_PATH
        self.j3_pid_diagnostic = J3PIDServiceCoordinator(self,
            enabled=j3_pid_diagnostic, seal_path=DEFAULT_SEAL_PATH,
            reports_directory=Path(output).resolve()/'j3-pid-diagnostic',
            continuation_baseline_report=j3_pid_continuation_baseline,
            continuation_variant_report=j3_pid_continuation_variant)
        # A prior gain-write intent must be reviewed before a new owner can
        # reconnect. This path is stable across output directories and models.
        self.j3_pid_diagnostic.require_motion_unsealed('startup')
        from .v21_j3_range_service import J3RangeDiagnostic
        self.j3_range_diagnostic = J3RangeDiagnostic(self, enabled=j3_range_diagnostic,
            output=output, amplitude=j3_range_amplitude)
        from .v21_j3_compensation_service import J3CompensationDiagnostic
        self.j3_compensation_diagnostic = J3CompensationDiagnostic(
            self, enabled=j3_compensation_diagnostic, output=output)
        from .v21_j3_native_service import J3NativeResponseDiagnostic
        self.j3_native_response_diagnostic = J3NativeResponseDiagnostic(
            self, enabled=j3_native_response_diagnostic, output=output)
        from .v21_same_goal_service import SameGoalServiceCoordinator
        same_goal_source = (same_goal_i1_source if same_goal_i1_source is not None
                            else same_goal_p32_source)
        self.same_goal_diagnostic = SameGoalServiceCoordinator(
            self, enabled=same_goal_source is not None,
            source_execution_path=same_goal_source,
            mode='i1' if same_goal_i1_source is not None else 'p32', output=output,
            integral_max_seconds=same_goal_integral_max_seconds)
        if type(replan_on_execute) is not bool:
            raise ValueError('replan_on_execute must be boolean')
        self.replan_on_execute = replan_on_execute
        if type(trial_max_blocks) is not int or not 1 <= trial_max_blocks <= 32:
            raise ValueError('trial_max_blocks must be an integer budget between 1 and 32')
        if lewm_cem_candidate_parameterization not in (
                None, 'bounded_endpoint_then_hold', 'five_command_sequence',
                'scaled_actor_sequence', 'scaled_actor_endpoint_hold',
                'amplified_actor_sequence', 'smooth_actor_endpoint'):
            raise ValueError('unknown LeWM CEM candidate parameterization')
        if lewm_cem_candidate_parameterization is not None and (not lewm_cem or raw_actor):
            raise ValueError('candidate parameterization requires LeWM CEM and cannot use raw Actor mode')
        if max_plan_age_s is not None and (not isinstance(max_plan_age_s, (int, float))
                or not np.isfinite(max_plan_age_s) or max_plan_age_s <= 0.):
            raise ValueError('max_plan_age_s must be a finite positive number')
        if hardware_maximum_lateness_s is not None and (
                isinstance(hardware_maximum_lateness_s, bool)
                or not isinstance(hardware_maximum_lateness_s, (int, float))
                or not np.isfinite(hardware_maximum_lateness_s)
                or not 0. < float(hardware_maximum_lateness_s) <= .25):
            raise ValueError('hardware_maximum_lateness_s must be finite and at most 0.25s')
        if temporal_profile is not None:
            if temporal_profile != 'actual-readback-progress-compatibility-v1':
                raise ValueError('service temporal profile must use actual readback progress')
            from .v21_temporal_contract import validate_temporal_checkpoint
            validate_temporal_checkpoint(Path(root)/'fish/final.pt', temporal_profile)
        self.temporal_contract = None
        control_options = (lewm_control_checkpoint, lewm_control_parent_sha256, lewm_control_candidate_sha256)
        control_enabled = any(value is not None for value in control_options)
        if control_enabled and not all(value is not None for value in control_options):
            raise ValueError('LeWM control checkpoint and both SHA256 values are required together')
        prepared_control = None
        if control_enabled:
            if (not lewm_cem or lewm_cem_objective != 'actor_response_tracking' or raw_actor
                    or grounding_correction or head_grasp or alignment_motion_enabled
                    or column_dino_fg_calibration or gripper_check_enabled or grasp_target != 'body'
                    or resume_auxiliary_state is not None or column_memory is not None or trial_max_blocks > 2):
                raise ValueError('LeWM control requires unassisted response CEM, no resume, and trial_max_blocks <= 2')
            if not lewm_shadow_checkpoints or lewm_shadow_parent_sha256 != lewm_control_parent_sha256:
                raise ValueError('LeWM control requires shadow candidates with the same original parent SHA256')
            from .v21_lewm_control import prepare_lewm_controller
            prepared_control = prepare_lewm_controller(
                root, lewm_control_checkpoint, lewm_control_parent_sha256, lewm_control_candidate_sha256)
            if not any(hashlib.sha256(Path(path).read_bytes()).hexdigest() == lewm_control_candidate_sha256
                       for path in lewm_shadow_checkpoints):
                raise ValueError('LeWM control candidate SHA256 must also be included among shadow candidates')
        self.root, self.output = Path(root), Path(output)
        self._bus_faults = BusFaultLog(self.output)    # 11/06: bus-faults.jsonl, the bus going silent and coming back
        self.lewm_controller = None if prepared_control is None else deepcopy(prepared_control.identity)
        self.lewm_parent_baseline = None
        self.lewm_shadow = None
        self.lewm_shadow_plan = None
        self.lewm_shadow_last = None
        if bool(lewm_shadow_checkpoints) != bool(lewm_shadow_parent_sha256):
            raise ValueError('shadow candidates require their explicit parent SHA256')
        if raw_actor and (grounding_correction or head_grasp or alignment_motion_enabled or lewm_cem):
            raise ValueError('raw Actor comparison cannot mix assisted command overrides')
        if lewm_cem and (grounding_correction or head_grasp or alignment_motion_enabled):
            raise ValueError('LeWM CEM cannot mix assisted command overrides')
        self.raw_actor = bool(raw_actor)
        self.lewm_cem = bool(lewm_cem)
        if lewm_cem_objective not in ('column_visual', 'actor_response_tracking', 'actor_command_tracking',
                                      'wrist_distance_alignment'):
            raise ValueError('unknown LeWM CEM objective')
        if lewm_cem_objective == 'wrist_distance_alignment' and not lewm_cem:
            raise ValueError('Wrist distance/alignment requires CEM')
        if wrist_grasp_calibration is not None and lewm_cem_objective not in ('wrist_distance_alignment', 'actor_response_tracking'):
            raise ValueError('Wrist calibration requires a visual LeWM response objective')
        self.wrist_grasp_response = None
        self.wrist_grasp_calibration = None
        self.wrist_scene_cost_status = dict(available=False, reason='moving Wrist response not calibrated')
        if wrist_grasp_calibration is not None:
            from .v21_grasp_cost import load_measured_wrist_response
            calibration_path = Path(wrist_grasp_calibration).resolve()
            self.wrist_grasp_response = load_measured_wrist_response(calibration_path)
            self.wrist_grasp_calibration = dict(path=str(calibration_path),
                sha256=hashlib.sha256(calibration_path.read_bytes()).hexdigest())
        self.lewm_cem_objective = lewm_cem_objective
        self.lewm_cem_candidate_parameterization = (
            'five_step_endpoint_then_hold_to_40' if lewm_cem_objective == 'wrist_distance_alignment'
            else lewm_cem_candidate_parameterization)
        self.lewm_cem_action_scale = float(lewm_cem_action_scale)
        self.lewm_cem_response_joint_count = (
            6 if lewm_cem_objective == 'wrist_distance_alignment' or
            lewm_cem_candidate_parameterization in ('amplified_actor_sequence', 'smooth_actor_endpoint') else 5)
        # This only exposes one explicitly-confirmed, physical-gate-covered
        # calibration chunk at a time. It never alters an Actor/CEM action.
        self.column_dino_fg_calibration_enabled = bool(column_dino_fg_calibration)
        self.record_policy_inputs = bool(record_policy_inputs)
        self.camera_condition = V21CameraCondition(camera_condition, camera_address)
        if ((self.camera_condition.name != 'both_fresh' or self.camera_condition.address != 'factual')
                and (wrist_only or column_memory is not None or resume_auxiliary_state is not None)):
            raise ValueError('camera-condition trials need empty dual-view memory and both actual cameras at H0')
        self._filter_actual_observations = True
        self._proposed_condition = None
        if alignment_scope not in ('full','lateral'):
            raise ValueError('unknown alignment scope')
        self.alignment_scope=alignment_scope
        self.auxiliary_budget=AuxiliaryMotionBudget.restore_latest(self.output)
        saved_actual=None if resume_auxiliary_state is None else load_actual_resume(resume_auxiliary_state,self.root)
        self.actual_resume=None
        if grasp_target not in ('body','head'):
            raise ValueError('grasp_target must be body or head')
        if head_grasp or grasp_target=='head':
            # 10/31: the head target was the operator colour model (the colour head detector), gone from the code
            # with every other colour function
            raise ValueError('head grasp is gone: its head target was the operator colour model, removed 10/31')
        self.head_grasp = False     # 10/31: gone with the colour head target (the constructor refuses it above)
        # The requested part is independent of the experimental vertical solver.
        # Neither changes the trained Actor input or its checkpoint contract.
        self.grasp_target = 'body'
        self.head_targets = {}
        # Grounding-corrected execution: the Actor still proposes every command; the
        # wrist object-vs-jaw grounding (the same eight numbers the Actor token receives)
        # recentres J1 while the jaws are open and vetoes a close while the object is
        # still small in the wrist view. Its own action source, never labelled as Actor output.
        self.grounding_correction=bool(grounding_correction)
        self.grounding_px_per_unit=None if grounding_px_per_unit is None else float(grounding_px_per_unit)
        self.grounding_record=None; self._grounding_vetoes=0
        self.execution_geometry_record=None
        self.lewm_cem_record=None
        self.lewm_cem_trace_record=None
        self._planned_source_evidence=None
        # Frozen at planning/dispatch for post-HOLD observability.  The owner
        # refreshes its display cue after execution, so last_object_relative
        # is not a reliable record of the vector the Actor actually saw.
        self._planned_object_relative_snapshot=None
        self._cem_source_evidence=None
        # Derived only from actual, exposure-paired Column DINO F/G readbacks.
        # This remains unavailable until the local response fit passes; no
        # Column FK projection or static capture pixel may replace it.
        self._column_dino_fg_samples=[]
        self.column_dino_fg_response=None
        self.column_dino_fg_transport_status=dict(available=False, sample_count=0,
                                                  reason='no_actual_column_dino_fg_response')
        self.centroid_tracking=None
        # Steady-state sag under gravity at full reach: a small J2/J3 step is not realised by the
        # position servo. Integrate (commanded end - readback) of the previous grounded step.
        self._grounding_bias=np.zeros(2); self._grounding_prev_end=None
        self._roll_gain=None; self._roll_prev=None   # (J5, theta_deg) of the previous grounded step
        self._search=dict(dir=1.,prev=None,votes=[],blocks=0)   # column-guided J1 search when the wrist cannot see the object
        # Carry after a grounded close: lift, align the COLUMN visual fingertips with the
        # registered drop zone (pixel loop, gains estimated online), open. The column FK
        # projection is not used (measured ~100 px off); the visual tips are.
        self._carry=None; self._home_after_release=False   # task done -> next plan request homes instead
        # Execution-geometry place FSM (not grounding _carry): close→lift→box→open.
        self._geom_place=None
        self._fish_task = dict(phase='approach')
        self._fish_task_proposal = None
        self._assistance = AssistanceStateCoordinator(self)
        self._assistance_proposal_id = None
        self._trial_admission_lock = threading.RLock()
        self._trial_stop_generation = 0
        self._trial_admission_in_progress = False
        self._trial_admission_cancelled = None
        # ROS requests carry a one-shot admission epoch.  STOP invalidates it
        # before asserting the physical stop event, so a worker that was
        # accepted but delayed cannot re-arm a stopped robot.
        self._ros_control_epoch_lock = threading.RLock()
        self._ros_control_epoch = 0
        self._ros_control_local = threading.local()
        self.output.mkdir(parents=True, exist_ok=True)
        self.trial = CappedTrial(lambda:self.handle('plan'),lambda:self.handle('execute'),
            lambda:self.stop(trial_completion=True),
            lambda data:write_json(self.output/('capped-trial-'+data['trial_id']+'.json'),data),
            last_attempt=lambda:getattr(self.owner.bridge,'last_attempt',None),
            max_blocks=trial_max_blocks)
        self.task, self.clock = task, clock
        self.operation_lock = threading.Lock()
        self.display_lock = threading.RLock()
        self.closing = threading.Event()
        self.owner = None; self.monitor = None; self.c4 = None
        self.sequence = 0; self.snapshot = None; self.live_frames = {}; self.live_times = {}
        self.phase = 'loading'; self.error = None; self.stop_error = None
        self.model_status = {}; self.current_q = None; self.current_q_time = None
        self.startup_observation = dict(attempts=0, bracket_timeouts=[])
        self.home_observation = None
        self.plan_record = None; self.preview_record = None; self.last_execution = None; self.last_attempt = None
        self.candidate_preview_record = None
        self.fixed_goal_compensation = None
        self.fixed_goal_compensation_path = None
        self.motion_validation_record = None
        self.execution_history = []  # compact records of the last completed Actor blocks, display only
        self.memory_reset_reason = 'service startup: no old predicted or actual state imported'
        self.target_diagnostics = target_diagnostics; self.target_tracker = None
        self.object_finder = None
        self.targets = dict(display_only=True, control_input=False, targets={}, available=False)
        self.tip_display = None; self.fingertips = {}
        self.alignment_motion_enabled=bool(alignment_motion_enabled)
        self.wrist_reference=Path(wrist_reference);self._wrist_fit=None
        self.alignment=None;self._alignment_session=None
        self.gripper_check_enabled=bool(gripper_check_enabled);self.gripper_check=None
        try:
            extra = {} if column_memory is None else dict(column_memory=column_memory,
                                                         column_scene_unchanged=column_scene_unchanged)
            if wrist_only:
                extra['wrist_only'] = True
            if normal_follow_budget_s:
                extra['normal_follow_budget_s']=normal_follow_budget_s
            if j3_position_servo:
                extra['j3_position_servo'] = True
            if arm_position_servo:
                extra['arm_position_servo'] = True
            if gripper_force!='lerobot':
                extra['gripper_force']=gripper_force
            if max_plan_age_s is not None:
                extra['max_plan_age_s']=float(max_plan_age_s)
            if hardware_maximum_lateness_s is not None:
                extra['maximum_lateness_s']=float(hardware_maximum_lateness_s)
            if clearance_floor_mm is not None:
                extra['clearance_floor_mm'] = float(clearance_floor_mm)   # audit doc 14.12: the table gate's floor
            if clearance_plan_mm is not None:
                extra['clearance_plan_mm'] = float(clearance_plan_mm)     # 17.91: planned paths above the floor
            if arm_p_coefficient is not None:
                extra['arm_p_coefficient'] = int(arm_p_coefficient)       # 17.92: shoulder lift + elbow flex gain
            self.owner = owner_factory(self.root, self.output, task, device, open_hardware=True, **extra)
            if self.home_state_path is not None and hasattr(self.owner, 'home_state'):
                self.owner.home_state = self._home_target()
            self.owner.bridge.cem_response_joint_count = self.lewm_cem_response_joint_count
            # training overhaul doc sections 17.32 / 17.34: the live object finder (the learned finder on the exposure's
            # own DINO tokens, or SAM 3 named by the task's object word) gives the owner its object encoder
            self.object_finder = None if object_finder is None else self._install_object_finder(object_finder)
            if (self.object_finder is None and getattr(self.owner.runtime.core.config, 'object_relative', False)
                    and hasattr(self.owner, 'object_relative') and self.owner.object_relative is None):
                # 10/31: the owner no longer makes an operator colour encoder; its object cue needs a finder (an owner
                # without the encoder slot makes the cue itself: the tests' stand-ins)
                raise ValueError('this model reads the wrist object cue: start the service with an object finder '
                                 '(--object-finder learned with --object-finder-model, or sam3)')
            # training overhaul doc section 17.7: a model that reads the place goal gets it from the box the column
            # camera sees (fcpg/v21_place_goal.LivePlaceGoal); TrialOwner.capture computes it at every readback.
            # Section 17.12: only while the jaws hold the fish, when the model's goal tables were labelled that way;
            # section 17.14: before the hold the fish the same camera sees, for tables labelled 'target'
            if getattr(self.owner.runtime.core.config, 'place_goal', False):
                if place_goal_camera is None:
                    raise ValueError('a model that reads the place goal needs its column camera (--place-goal-column-camera)')
                from .v21_place_goal import LivePlaceGoal, goal_availability
                self.owner.place_goal = LivePlaceGoal.from_availability(place_goal_camera, goal_availability(self.root))
                if self.object_finder is not None:
                    from .v21_object_finder import COLUMN_SOURCE
                    from .v21_sam3_finder import SOURCE as SAM3_SOURCE
                    # the column fish from SAM 3 (17.32) or from the learned column finder (17.51), never colour
                    self.owner.place_goal.fish_source = (COLUMN_SOURCE if self.object_finder['kind'] == 'learned'
                                                         else SAM3_SOURCE)
            elif place_goal_camera is not None:
                raise ValueError('--place-goal-column-camera is for a model that reads the place goal')
            if c4_planner is not None:
                from .v21_c4_service import C4ServiceAdapter
                self.c4 = C4ServiceAdapter(c4_planner, c4_planner_factory)
                self.c4.attach(self)
            if alignment_cem is not None:
                from .v21_alignment_cem import AlignmentCEMAdapter
                self.c4 = AlignmentCEMAdapter(alignment_cem, alignment_response, **(alignment_cem_options or {}))
                self.c4.attach(self)
            if explore_actor:
                from .v21_explore import ExploreAdapter
                self.c4 = ExploreAdapter(seed=explore_seed)
                self.c4.attach(self)
            if prepared_control is not None:
                from .v21_lewm_control import install_prepared_lewm_controller
                self.lewm_parent_baseline = install_prepared_lewm_controller(self.owner.runtime.core, prepared_control)
            if temporal_profile is not None:
                from .v21_temporal_contract import install_temporal_contract
                self.temporal_contract = install_temporal_contract(
                    self.owner.runtime.core, self.root/'fish/final.pt', profile=temporal_profile)
            if lewm_shadow_checkpoints:
                from .v21_lewm_adaptation import load_lewm_shadow_candidate
                from .v21_lewm_shadow import LeWMShadowEvaluator
                parent = (self.owner.runtime.core.lewm if prepared_control is None else self.lewm_parent_baseline)
                candidates = [load_lewm_shadow_candidate(parent,
                    parent_checkpoint=self.root/'fish/final.pt', candidate_path=path,
                    expected_parent_sha256=lewm_shadow_parent_sha256) for path in lewm_shadow_checkpoints]
                shadow_kwargs = ({} if prepared_control is None else
                    dict(controller_candidate_sha256=lewm_control_candidate_sha256))
                self.lewm_shadow = LeWMShadowEvaluator(parent,candidates,**shadow_kwargs)
            self._restore_incomplete_execution()
            if column_memory is not None:
                self.memory_reset_reason = 'restored one real historical Column frame at original q/time; no execution history invented'
            self._install_camera_condition_runtime()
            self.owner.stop_event.set()
            self.phase = 'observing'
            if saved_actual is None:
                self._prepare_condition_reader()
                self._observe_startup()
            else:
                endpoint=self.owner.capture()
                self._restore_or_observe_stopped(saved_actual,endpoint)
                if self.actual_resume and self.actual_resume.get('restored') and not self.camera_condition.h0_committed:
                    self._force_commit_condition()
                self._refresh_status();self._publish()
            self.phase = 'idle'
            self.monitor = threading.Thread(target=self._monitor_q, name='v21-live-readback', daemon=True)
            self.monitor.start()
        except BaseException:
            self.close(); raise

    def _restore_incomplete_execution(self):
        path = self.output/'incomplete-execution.json'
        if path.exists():
            record = json.loads(path.read_text())
            if type(record.get('requires_home')) is not bool:
                raise ValueError('invalid incomplete-execution recovery record')
            self.owner.runtime.requires_home_after_incomplete_execution = record['requires_home']

    def ros_control_admission_epoch(self):
        """Return the current ROS admission epoch without changing hardware."""
        with self._ros_control_epoch_lock:
            return self._ros_control_epoch

    def _require_ros_control_admission(self):
        epoch = getattr(self._ros_control_local, 'admission_epoch', None)
        if epoch is None:
            return
        with self._ros_control_epoch_lock:
            if epoch != self._ros_control_epoch:
                raise RuntimeError('ROS request was cancelled by STOP before motion admission')

    def _clear_stop_for_current_operation(self):
        """Atomically arm an admitted ROS operation, or reject its stale token."""
        epoch = getattr(self._ros_control_local, 'admission_epoch', None)
        if epoch is None:
            self.owner.stop_event.clear()
            return
        with self._ros_control_epoch_lock:
            if epoch != self._ros_control_epoch:
                raise RuntimeError('ROS request was cancelled by STOP before motion admission')
            self.owner.stop_event.clear()

    def handle_ros_control(self, action, payload, *, admission_epoch):
        """Ingress used only by the ROS owner after transport validation."""
        if action == 'stop':
            raise ValueError('ROS STOP must use handle_ros_stop')
        with self._ros_control_epoch_lock:
            if admission_epoch != self._ros_control_epoch:
                raise RuntimeError('ROS request was cancelled by STOP before service entry')
        self._ros_control_local.admission_epoch = admission_epoch
        try:
            self._require_ros_control_admission()
            return self.handle(action, payload)
        finally:
            try:
                del self._ros_control_local.admission_epoch
            except AttributeError:
                pass

    def handle_ros_stop(self, payload=None):
        if payload:
            raise ValueError('ROS STOP accepts an empty payload only')
        self.stop()
        return self.transport_status()

    def _restore_or_observe_stopped(self,saved_actual,endpoint):
        """A rejected old state cannot authorize motion, but need not hide video."""
        try:
            self.actual_resume=resume_actual_state(saved_actual,self.owner.runtime,self.auxiliary_budget,
                endpoint,pose_tolerance=self.owner.io.settle_position_tolerance)
        except (ValueError,RuntimeError) as exc:
            self.owner.stop_event.set();self.owner.runtime.stop()
            if not self.auxiliary_budget.requires_home:
                raise  # Never mask a failure after the restart interlock was cleared.
            self.actual_resume=dict(restored=False,reason=str(exc),motion_authorized=False,
                                    requires_home=True,previous_history_loaded=False)
            self.owner.runtime.state=None
            self.owner.runtime.observe(endpoint)
            self.memory_reset_reason='old-state restore rejected; fresh observation only; unchanged budget still requires HOME'
            self.error='actual state not restored: '+str(exc)
            self.last_execution=None
            return
        self.last_execution=deepcopy(saved_actual.get('last_execution'))
        pair=(self.last_execution or {}).get('wrist_alignment')
        if pair is not None and self.alignment_scope=='lateral':
            from .v21_lateral_alignment import recorded_alignment_pair
            self._alignment_controller().pairs=[recorded_alignment_pair(pair)]
        self.memory_reset_reason='verified actual state resumed; fresh dual views/readback; no pending motion or old fit restored'

    def _refresh_status(self):
        value = self.owner.runtime.status()
        state = self.owner.runtime.state
        if state is not None:
            value.update(q_error=state.q_error.detach().cpu().tolist(),
                         grounding_norm=state.grounding.detach().float().norm(dim=-1).mean(-1).cpu().tolist(),
                         visual_error_norm=state.visual_error.detach().float().norm(dim=-1).mean(-1).cpu().tolist())
        with self.display_lock:
            self.model_status = value

    def _measure_head_targets(self,targets=None):
        """The fish head was found by the operator colour model (the colour head detector), which is gone from the
        code (10/31, James 「不要的废弃代码 这种颜色函数删掉」): no head target any more."""
        self.head_targets = {}
        return {}

    def _install_object_finder(self, config):
        """The live object finder (training overhaul doc section 17.32, fcpg/v21_sam3_finder.py; section 17.34) that
        gives the owner its object encoder; the operator colour model is gone from the code (10/31, James 「不要的废弃代码
        这种颜色函数删掉」). config: kind ('sam3', or 'learned': the learned finder of section 17.34 on the exposure's own DINO tokens, no
        SAM at inference; then model = its final.pt), socket, word (the task text's object word SAM 3 is asked for),
        anchor_score, hold_reference (the demonstrations' fish cue report, for a model that learned the outline cue of
        section 17.30); with 'learned', column_model (section 17.51: the learned column finder, a final.pt trained with
        --column-pairs) gives the column fish of the grasp target, which a model that reads the place goal needs;
        text (dict phrase, weights, bpe, repo; James 10/28) makes the wrist and column finders' reference from the task
        text with DINOv3's text alignment (fcpg/v21_text_finder.py) instead of the fish reference in their files."""
        from .v21_sam3_finder import FinderClient, Sam3ObjectRelativeEncoder, demo_hold_reference, object_prompts
        kind = str(config.get('kind', 'sam3'))
        if kind not in ('sam3', 'learned'):
            raise ValueError(f'unknown object finder {kind!r}')
        model = self.owner.runtime.core.config
        protocol = json.loads((self.root / 'config.json').read_text(encoding='utf-8')).get('protocol') or {}
        # section 17.117: a model that learned the place cue on the Archive gets it from the learned box finder
        reads_place = bool(getattr(model, 'place_relative', False))
        place_cues = bool((protocol.get('object_cues') or {}).get('place'))
        if not getattr(model, 'object_relative', False) or (
                reads_place and not (kind == 'learned' and place_cues and config.get('box_model') is not None)):
            raise ValueError('the finder gives the wrist object token; a model that also reads a place token is run '
                             'only with the learned box finder and the place cues of section 17.117 in its protocol')
        if kind == 'learned' and getattr(model, 'place_goal', False) and config.get('column_model') is None:
            raise ValueError('a model that reads the place goal needs the learned column finder (column_model, '
                             'section 17.51)')
        if config.get('column_model') is not None and kind != 'learned':
            raise ValueError('the learned column finder comes with the learned finder')
        if config.get('box_model') is not None and kind != 'learned':
            raise ValueError('the learned box finder comes with the learned finder')
        convention = 'hold' if protocol.get('object_cues') else 'jaw'   # the outline cue (17.30) or the colour layout
        hold = demo_hold_reference(config['hold_reference']) if convention == 'hold' else None
        owner = self.owner
        # a model trained on the fish token table of section 17.57 (scripts/build_v21_fish_token_table.py) learned the
        # 'jaw' layout with the score field its table states ('one'): no DINO tracker is asked for it
        fish_token = protocol.get('fish_token') or {}
        score_source = fish_token.get('score_source')
        if fish_token and (fish_token.get('convention') != 'jaw' or score_source != 'one'):
            raise ValueError(f'unknown fish token table convention {fish_token.get("convention")!r} / score '
                             f'{score_source!r}')
        if score_source is not None and kind != 'learned':
            raise ValueError('a model with the 17.57 fish token is run with the learned finder')
        if DINOV2_OFF and convention == 'jaw' and score_source is None:
            raise ValueError("the colour-era token's score field is the frozen DINOv2 tracker's: not in a "
                             'V21_DINOV2=off service (a model trained on the 17.57 fish token table is)')
        # the token's score field as the colour-era token had it ('jaw'): the frozen DINO confirmation of the pickup,
        # from a frozen-feature tracker the owner offers, else one made here (only its frozen-feature locate is used)
        dino = getattr(getattr(owner, 'object_relative', None), 'tracker', None)
        if convention == 'jaw' and dino is None and score_source is None:
            from .so101_visual_targets import VisualTargetTracker
            from .v21_object_relative import TRACKER_MINIMUM_SCORE
            dino = VisualTargetTracker(Path(established.DEFAULT_DINO_TARGET_PROFILE).expanduser().resolve())
            dino.minimum_score = float(TRACKER_MINIMUM_SCORE)
        column = box = None
        if kind == 'learned':
            from .v21_object_finder import LearnedBoxFinder, LearnedObjectRelativeEncoder
            client, word = None, None
            # James 10/28: the task text names the thing; DINOv3's text alignment makes the finders' reference from
            # every exposure (fcpg/v21_text_finder.py) instead of the fish reference in their model files
            scorer, text = None, config.get('text')
            if text is not None:
                from .v21_text_finder import TextScorer, TextWristEncoder, object_phrase
                if ENCODER != 'dinov3_vitl16':
                    raise ValueError('the task-text finder reads DINOv3-L tokens (V21_ENCODER=dinov3_vitl16)')
                phrase = str(text.get('phrase') or object_phrase(self.task))
                if phrase not in str(self.task).lower():
                    raise ValueError(f'the text finder phrase {phrase!r} is not in the task text {self.task!r}')
                scorer = TextScorer(phrase, weights=text['weights'], bpe=text['bpe'], repo=text['repo'],
                                    device=getattr(owner, 'device', 'cpu'))
                word = scorer.phrase
            if config.get('box_model') is not None:
                box = LearnedBoxFinder(config['box_model'], device=getattr(owner, 'device', 'cpu'))
            # 11/03: the wrist finder's reference from the column picture of the same exposure (TextWristEncoder)
            extra = {} if scorer is None else dict(
                scorer=scorer, prefix_of=lambda: owner.last_model_prefix[0],
                reference_of=lambda: (owner.last_model_prefix[1], exposure_tokens(owner, encoder.encoder)[1]))
            encoder = (LearnedObjectRelativeEncoder if scorer is None else TextWristEncoder)(
                config['model'], self.wrist_reference, owner.kinematics, convention=convention, hold_px=hold,
                tokens_of=lambda: exposure_tokens(owner, encoder.encoder)[0], score_source=score_source,
                dino_tracker=dino if convention == 'jaw' and score_source is None else None,
                device=getattr(owner, 'device', 'cpu'), place_finder=box if reads_place else None,
                holding=self._holding_now if reads_place else None, **extra)
            if encoder.encoder not in ('dinov2_vits14', ENCODER) or (DINOV2_OFF and encoder.encoder != ENCODER):
                raise ValueError(f'the learned finder reads {encoder.encoder}; this service runs {ENCODER}'
                                 + (' only' if DINOV2_OFF else ''))
            described = dict(finder='learned' if scorer is None else 'learned+dinotxt', model=str(config['model']),
                             sam_used=False)
            if scorer is not None:
                described.update(text=scorer.describe())
            if config.get('column_model') is not None:
                from .v21_object_finder import LearnedColumnFinder
                if scorer is None:
                    column = LearnedColumnFinder(config['column_model'], device=getattr(owner, 'device', 'cpu'))
                else:
                    from .v21_text_finder import TextColumnFinder
                    column = TextColumnFinder(config['column_model'], scorer=scorer,
                                              prefix_of=lambda: owner.last_model_prefix[1],
                                              device=getattr(owner, 'device', 'cpu'))
                if column.encoder not in ('dinov2_vits14', ENCODER) or (DINOV2_OFF and column.encoder != ENCODER):
                    raise ValueError(f'the column finder reads {column.encoder}; this service runs {ENCODER}'
                                     + (' only' if DINOV2_OFF else ''))
                described.update(column=column.describe())
            if box is not None:
                if box.encoder not in ('dinov2_vits14', ENCODER) or (DINOV2_OFF and box.encoder != ENCODER):
                    raise ValueError(f'the box finder reads {box.encoder}; this service runs {ENCODER}'
                                     + (' only' if DINOV2_OFF else ''))
                described.update(box=box.describe())
        else:
            word = str(config['word'])
            if word not in object_prompts(self.task):
                raise ValueError(f'{word!r} is not an object word of the task text {self.task!r}')
            client = FinderClient(config['socket'])
            described = client.describe()          # fails here, before any trial, when the finder is not running
            encoder = Sam3ObjectRelativeEncoder(
                client, self.wrist_reference, owner.kinematics, convention=convention, hold_px=hold, prompts=[word],
                mode='track', anchor_word=word, anchor_score=float(config['anchor_score']),
                raw_frame=lambda: (getattr(owner, 'frames', None) or {}).get('wrist'),
                dino_tracker=dino if convention == 'jaw' else None)
        owner.object_relative = encoder
        record = dict(schema='v21-object-finder-v1', section='17.34' if kind == 'learned' else '17.32', kind=kind,
                      finder=described, encoder=encoder.describe(),
                      model_convention_source='config.json protocol.object_cues' if convention == 'hold' else
                      'config.json protocol.fish_token (section 17.57)' if score_source is not None else
                      'no object_cues in the model protocol: the colour token layout',
                      hold_reference=None if hold is None else str(config['hold_reference']),
                      colour_functions='removed from the code (10/31)', dinov2_off=DINOV2_OFF,
                      dinov2_functions_forbidden=list(getattr(self, 'dinov2_forbidden', [])))
        write_json(self.output / 'object-finder.json', record)
        print(json.dumps(dict(event='v21_object_finder', kind=kind, convention=convention, word=word)), flush=True)
        return dict(client=client, encoder=encoder, word=word, record=record, kind=kind, column=column, box=box)

    def _holding_now(self):
        """Section 17.117: the gripper holds the thing, so the Actor gets the place cue (the Archive taught it during
        the carry only): the alignment CEM's grip latch when it runs one, else its planner's grasped flag."""
        c4 = getattr(self, 'c4', None)
        latch = getattr(c4, 'latch', None)
        if latch is not None:
            return bool(latch.holding)
        return bool(getattr(getattr(c4, 'planner', None), 'grasped', False))

    def _learned_targets(self, view_sources):
        """The service's targets from the learned finders alone (training overhaul doc section 17.57): the box (drop zone)
        in the wrist and the column from the learned box finder, the fish in the wrist from the object token's own
        record of this exposure (the learned finder), the fish in the column from the learned column finder when the
        service has one; all on this exposure's own tokens of the finders' encoder. No frozen-feature prototype, no
        registration crop, no hand-set similarity bar, no jaw-pixel exclusion. The schema of
        fcpg/prod_v17_7_dino_targets.locate_dino_targets (the old frozen-DINOv2 tracker), so every reader is unchanged."""
        from .v21_object_finder import BOX_SOURCE, BOX_VIEWS, COLUMN_SOURCE, SOURCE as LEARNED_WRIST_SOURCE
        finder = self.object_finder
        owner = self.owner

        def not_visible(role, view, source, reason):
            return dict(role=role, view=view, visible=False, centroid_px=None, axis_px=None, width_px=None,
                        length_px=None, score=-1., separation=0., source=source, reason=str(reason))
        out = dict(pickup={}, drop_zone={})
        box = finder['box']
        tokens = exposure_tokens(owner, box.encoder)
        for index, view in enumerate(BOX_VIEWS):
            if view_sources.get(view) != 'current_observation' or tokens is None:
                getattr(box, 'last_prob', {}).pop(view, None)          # section 17.118: no stale map for the box solid
                out['drop_zone'][view] = not_visible('drop_zone', view, BOX_SOURCE, 'no current picture of this view')
            else:
                out['drop_zone'][view] = box.record(np.asarray(tokens)[index], view)
        encoder = getattr(owner, 'object_relative', None)
        record = getattr(encoder, 'last_record', None)
        source = LEARNED_WRIST_SOURCE
        if view_sources.get('wrist') != 'current_observation':
            out['pickup']['wrist'] = not_visible('pickup', 'wrist', source, 'no current wrist picture')
        elif not isinstance(record, dict) or record.get('cog_px') is None:
            out['pickup']['wrist'] = not_visible('pickup', 'wrist', source, 'the learned finder found nothing')
        else:
            centre = [float(v) for v in record['cog_px']]
            theta = float(record.get('theta_image') or 0.)
            out['pickup']['wrist'] = dict(role='pickup', view='wrist', visible=True, cog_px=centre,
                                          centroid_px=list(centre), axis_px=[float(np.cos(theta)), float(np.sin(theta))],
                                          length_px=float(record.get('length_px') or 0.),
                                          width_px=float(record.get('width_px') or 0.),
                                          score=float(record.get('finder_score', record.get('score')) or 0.),
                                          separation=0., source=record.get('source', source),
                                          cells=record.get('cells'))
        column = finder.get('column')
        if column is None:
            out['pickup']['column'] = not_visible('pickup', 'column', COLUMN_SOURCE, 'no learned column finder')
        elif view_sources.get('column') != 'current_observation':
            out['pickup']['column'] = not_visible('pickup', 'column', COLUMN_SOURCE, 'no current column picture')
        else:
            out['pickup']['column'] = column.record(np.asarray(exposure_tokens(owner, column.encoder))[1])
        return dict(version='17.57', source='learned_finders', display_only=True, control_input=False, targets=out)

    def _sam3_column_pickup(self, targets, column_letterbox):
        """The column pick-up record while the live place goal still collects the fish for its grasp target
        (fcpg/v21_place_goal.LivePlaceGoal.observe_fish): from SAM 3 (fcpg/v21_sam3_finder.column_pickup_record), or
        with the learned finder from the learned column finder on this exposure's own column DINO tokens (section
        17.51, fcpg/v21_object_finder.LearnedColumnFinder); otherwise the DINO record stays, display only."""
        goal = getattr(self.owner, 'place_goal', None)
        if goal is None or not goal.grasp_target or goal.fish_xy is not None or goal.hold.held:
            return
        if self.object_finder['kind'] == 'learned':
            column = self.object_finder.get('column')
            if column is None:
                return
            record = column.record(np.asarray(exposure_tokens(self.owner, column.encoder))[1])
        else:
            from .v21_sam3_finder import column_pickup_record
            record = column_pickup_record(self.object_finder['client'], column_letterbox,
                                          (getattr(self.owner, 'frames', None) or {}).get('column'),
                                          prompts=[self.object_finder['word']])
        targets.setdefault('targets', {}).setdefault('pickup', {})['column'] = record

    def _target_consumer(self):
        """Who plans with this observation's DINO targets; None when they are display only."""
        if self.phase in ('alignment-probe','alignment-return','alignment-step','alignment-reach'):
            return 'visual_alignment'
        c4=getattr(self,'c4',None)
        if (getattr(c4,'record_key',None)=='alignment_cem' and getattr(c4,'mode',None)=='control'
                and getattr(c4,'place_source','service_targets')=='service_targets'):
            # a model with the place token gives the CEM its own state's token; the targets stay display-only then
            return 'alignment_cem_wrist_drop_zone'   # fcpg/v21_alignment_cem.py aligns shut jaws to the box
        return None

    def _publish(self):
        owner = self.owner
        endpoint = owner.last_endpoint
        object_overlay=None
        enc=getattr(owner,'object_relative',None)
        # Display the same colour-support target used by the current Actor cue.
        # Partial visibility does not establish the complete fish's centre.
        rec=(getattr(enc,'last_record',None) if enc is not None else None)
        if not isinstance(rec,dict) or (rec.get('cog_px') is None and rec.get('centroid_px') is None):
            rec=getattr(enc,'last_policy_record',None) if enc is not None else None
        cue=getattr(owner,'last_object_relative',None)
        # Draw the detected support / C whenever the wrist frame is current, even
        # when the policy 8-vector is gated to zero (clipped / not aimable).
        wrist_retained = (getattr(owner, 'lens_cover_status', {}) or {}).get('wrist') or {}
        wrist_visible = bool(endpoint.observation.measurement_available[0,0]) or (
            wrist_retained.get('within_paper_window') and wrist_retained.get('retained_real_frame'))
        if (wrist_visible and isinstance(rec,dict)
                and (rec.get('cog_px') is not None or rec.get('centroid_px') is not None)):
            if cue is None:
                cue=np.zeros(8)
            try:
                from .so101_frozen_vit_grounding import support_contour_px
                g=np.asarray(rec['cog_px'] if rec.get('cog_px') is not None else rec['centroid_px'],np.float64)
                ax=np.asarray(rec.get('axis_px',(1.,0.)),np.float64)
                pair=np.asarray(enc.jaw_pair(endpoint.observation.q_at_image[0,0].cpu().numpy()),np.float64)
                size=[float(rec.get('length_px') or 0.),float(rec.get('width_px') or 0.)]
                cue=np.asarray(cue,np.float64).reshape(-1)
                if g.shape==(2,) and ax.shape==(2,) and pair.shape==(2,2) and np.isfinite(np.r_[g,ax,pair.ravel(),cue,size]).all():
                    contour=rec.get('contour_px') or []
                    if not contour:
                        mask=rec.get('support_mask')
                        contour=support_contour_px(mask) if mask is not None else []
                    body=rec.get('source')=='operator_colour_body'
                    # Display contract: blue grasp line = ⊥ long axis ([-ay,ax]).
                    # Always recompute from axis_px so a stale/swapped grasp_axis
                    # cannot paint the long axis in blue.
                    ax_u=ax/max(float(np.linalg.norm(ax)),1e-12)
                    grasp_disp=[float(-ax_u[1]), float(ax_u[0])]
                    object_overlay=dict(centroid_px=g.tolist(),cog_px=g.tolist(),
                        view='wrist',
                        grasp_kind=('colour_body_width_run' if body
                                    else 'shape_prior_uniform_density_cog'),
                        axis_px=ax.tolist(),
                        grasp_axis_px=grasp_disp,
                        jaw_pair_px=pair.tolist(),
                        jaw_pair_source='fk_projection_display_only',
                        length_px=size[0],width_px=size[1],
                        contour_px=contour,
                        source=(rec.get('source') if rec.get('source') in ('estimated_fg', 'dino_fg_near_prior',
                                                                            'sam3_task_word')
                                else 'operator_colour_body' if body else 'frozen_symmetric_support'),
                        symmetry_iou=float(rec.get('symmetry_iou') or 0.),
                        support_clipped=(bool(rec.get('body_truncated')) if body
                                         else bool(rec.get('support_clipped'))),
                        body_truncated=(bool(rec.get('body_truncated')) if body else None),
                        # Colour segmentation estimates no physical aimability.
                        # Token availability and full-object visibility differ.
                        aimable=(None if body else bool(rec.get('aimable', True))),
                        policy_token_available=bool(cue.size == 8 and cue[0] == 1),
                        target_geometry=('visible_colour_support_width_run' if body
                                         else 'frozen_feature_support'),
                        vector=cue.tolist(),
                        image_time_s=float(endpoint.observation.image_times[0,0]))
            except (AttributeError,ValueError,TypeError):
                pass  # Optional display diagnostics cannot invalidate a real observation.
        saved = getattr(owner,'saved_column',None)
        display_rgb = {v:owner.last_rgb[i].copy() for i,v in enumerate(('wrist','column'))}
        display_raw = {v:owner.frames[v].copy() for v in owner.frames}
        image_times = endpoint.observation.image_times.cpu().tolist()
        view_sources = dict(wrist='current_observation',column='current_observation')
        cover = getattr(owner, 'lens_cover_status', {}) or {}
        for i, view in enumerate(('wrist','column')):
            if bool(endpoint.observation.measurement_available[0,i]):
                continue
            info = cover.get(view) or {}
            if info.get('within_paper_window') and info.get('retained_real_frame'):
                view_sources[view] = 'retained_last_real_observation'
            else:
                view_sources[view] = 'unavailable'
        if saved is not None:
            display_rgb['column'] = saved.rgb.copy(); display_raw['column'] = saved.rgb.copy()
            image_times[0][1] = float(saved.observation.image_times[0,1])
            view_sources['column'] = 'historical_observation_not_live'
        # Tips before targets: jaw pixels must be known so claw occlusion
        # does not pull support / 重心 when the gripper covers the object.
        if self.target_diagnostics:
            try:
                if self.tip_display is None:
                    self.tip_display = TipDisplay(self.wrist_reference, owner.kinematics, device=owner.device)
                from .v21_column_temporal_prior import ColumnTemporalTracker
                if not hasattr(self, '_column_temporal_tracker'):
                    self._column_temporal_tracker = ColumnTemporalTracker()
                temporal = self._column_temporal_tracker.for_current(
                    owner.runtime.state, getattr(owner, 'bridge', None),
                    getattr(self.tip_display, 'wrist_kinematics', None),
                    getattr(self.tip_display, 'column_fit', None))
                tip_kwargs = {} if temporal is None else dict(column_temporal_prior=temporal)
                if DINOV2_OFF:      # section 17.57: the column jaw-tip observer reads DINOv2; geometry draws instead
                    tip_kwargs['column_observer'] = False
                elif ENCODER != 'dinov2_vits14':    # the model reads DINOv3-L; the column observer reads DINOv2
                    tip_kwargs['dino_tokens'] = owner.last_dino_tokens
                tips = self.tip_display.observe(owner.last_rgb, endpoint, owner.frames,
                                                owner.runtime.state.history, **tip_kwargs)
                if isinstance(tips.get('column'), dict):
                    tips['column']['temporal_prior_status'] = deepcopy(
                        self._column_temporal_tracker.diagnostic)
                if (view_sources['column'] == 'current_observation'
                        and not getattr(owner.runtime, 'requires_home_after_incomplete_execution', False)):
                    self._column_temporal_tracker.record_observation(
                        owner.runtime.state, tips.get('column'))
            except Exception as exc:
                tips = {v:dict(available=False, reason=f'{type(exc).__name__}: {exc}') for v in ('wrist','column')}
        else:
            tips = {}
        jaw_pairs = {}
        for view, tip in tips.items():
            if (tip_source_matches_view(view, tip) and tip.get('available')
                    and tip.get('fm_px') is not None):
                pair = np.asarray(tip['fm_px'], np.float64)
                if pair.shape == (2, 2) and np.isfinite(pair).all():
                    jaw_pairs[view] = pair
        if 'wrist' not in jaw_pairs and enc is not None:
            try:
                jaw_pairs['wrist'] = np.asarray(
                    enc.jaw_pair(endpoint.observation.q_at_image[0,0].cpu().numpy()), np.float64)
            except Exception:
                pass
        targets = dict(display_only=True, control_input=False, targets={}, available=False)
        target_tip_inputs = set()
        if self.target_diagnostics:
            try:
                if (getattr(self, 'object_finder', None) or {}).get('box') is not None:
                    # section 17.57: every target from the learned finders on this exposure's own tokens
                    targets = self._learned_targets(view_sources)
                else:
                    if self.target_tracker is None:
                        self.target_tracker = V21TargetTracker(
                            established.DEFAULT_DINO_TARGET_PROFILE,
                            wrist_drop_zone=None if self.wrist_drop_zone is None else self.wrist_drop_zone['path'])
                    target_tip_inputs.update(jaw_pairs)
                    targets = locate_dino_targets(
                        {v:rgb for v,rgb in display_rgb.items() if view_sources[v] != 'unavailable'},
                        tracker=self.target_tracker,
                        jaw_pairs_px=jaw_pairs or None,
                        sites=V21_TARGET_SITES,
                    )
                # the column fish from the object finder (SAM 3, section 17.32, or the learned column finder); the
                # operator colour body that stood in without a finder is gone from the code (10/31)
                col_rgb = display_rgb.get('column')
                if (col_rgb is not None and view_sources.get('column') != 'unavailable'
                        and getattr(self, 'object_finder', None) is not None):
                    self._sam3_column_pickup(targets, col_rgb)
                consumer=self._target_consumer()
                targets.update(display_only=consumer is None,control_input=consumer is not None,available=True,
                               control_consumer=consumer)
                for views in targets.get('targets',{}).values():
                    for view, record in views.items():
                        record.update(image_time_s=image_times[0][('wrist','column').index(view)],
                                      view_source=view_sources[view])
            except Exception as exc:
                targets['error'] = f'{type(exc).__name__}: {exc}'
        column_tip = tips.get('column')
        if isinstance(column_tip, dict):
            control_consumers = []
            source_valid = tip_source_matches_view('column', column_tip)
            if column_tip.get('available') and source_valid:
                if self.grounding_correction:
                    control_consumers.append('grounding_assistance')
                if self.lewm_cem and self.lewm_cem_objective == 'column_visual':
                    control_consumers.append('lewm_cem_visual_ranking')
                if 'column' in target_tip_inputs and self._uses_vertical_pinch_geometry():
                    control_consumers.append('execution_geometry_target_localization')
            # The RGB pair is not a learned-policy input. Some explicitly
            # enabled engineering paths can consume its midpoint or the target
            # localizer's symmetric jaw mask; display flags are not IO gates.
            column_tip.update(learned_policy_input=False,
                source_contract_valid=source_valid,
                target_localizer_input=source_valid and 'column' in target_tip_inputs,
                enabled_control_consumers=control_consumers,
                control_input=bool(control_consumers), display_only=not control_consumers,
                usage_semantics='enabled consumer paths; not evidence of motion or dispatch')
        # Identity support and shape candidate must come from this exposure,
        # not from self.targets belonging to the previous snapshot.
        self._update_fish_drop_goal(targets)
        if getattr(owner, 'place_goal', None) is not None:
            owner.place_goal.observe(self._fish_task.get('drop_goal'))
            owner.place_goal.observe_fish(((targets.get('targets') or {}).get('pickup') or {}).get('column'))
        heads = self._measure_head_targets(targets)
        with self.display_lock:
            self.sequence += 1
            self.targets = targets
            self.fingertips = tips
            self.snapshot = dict(sequence=self.sequence, q=endpoint.q.cpu().tolist(),
                time=endpoint.time.item(), image_times=image_times, view_sources=view_sources,
                lens_cover=deepcopy(getattr(owner, 'lens_cover_status', {}) or {}),
                measurement_available=endpoint.observation.measurement_available.cpu().tolist(),
                rgb=display_rgb,raw=display_raw,targets=deepcopy(targets),fingertips=deepcopy(tips))
            self.snapshot['head_targets'] = deepcopy(heads)
            self.snapshot['object_overlay'] = object_overlay
            self.current_q, self.current_q_time = endpoint.q[0].cpu().tolist(), endpoint.time.item()
        if view_sources.get('column') == 'current_observation':
            self._record_column_dino_fg_sample(tips)
        # Current diagnostic snapshot is not an execution record or a policy input.
        path = self.output/'current-observation.npz'
        temporary = self.output/'current-observation.tmp.npz'
        # dino_tokens: DINOv2 as before; a V21_DINOV2=off service (section 17.57) has none and keeps model_tokens only
        arrays = dict(rgb=owner.last_rgb,
            measurement_available=endpoint.observation.measurement_available.cpu().numpy(),
            **(dict(dino_tokens=endpoint.observation.features.cpu().numpy()) if ENCODER == 'dinov2_vits14' else
               {} if DINOV2_OFF else dict(dino_tokens=np.asarray(owner.last_dino_tokens)[None])),
            **({} if ENCODER == 'dinov2_vits14' else dict(model_tokens=endpoint.observation.features.cpu().numpy())),
            q_position=endpoint.q.cpu().numpy(), q_at_image=endpoint.observation.q_at_image.cpu().numpy(),
            image_times_s=endpoint.observation.image_times.cpu().numpy(), q_time_s=endpoint.time.cpu().numpy())
        # Retain the original pixels paired with each current exposure for
        # visual debugging. These are not extra captures or policy features;
        # missing/historical views must not acquire a current raw-image label.
        for index, view in enumerate(('wrist', 'column')):
            if (bool(endpoint.observation.measurement_available[0, index])
                    and view_sources[view] == 'current_observation'):
                arrays[view + '_raw_rgb'] = display_raw[view]
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)

    def _observe(self):
        self.owner.runtime.observe(self.owner.capture())
        self._refresh_status(); self._publish()

    def _observe_bracket_retry(self, receipt):
        """Retry only discarded captures whose 50 ms camera/readback bracket expired."""
        for attempt in range(1, 4):
            receipt['attempts'] = attempt
            try:
                self._observe()
                return
            except RuntimeError as error:
                if str(error) != 'camera/readback acquisition bracket exceeds 50ms':
                    raise
                receipt['bracket_timeouts'].append(str(error))
                if attempt == 3:
                    raise

    def _observe_startup(self):
        """Retry only a discarded startup capture whose 50 ms bracket expired."""
        self._observe_bracket_retry(self.startup_observation)

    def _install_camera_condition_runtime(self):
        runtime = self.owner.runtime
        if getattr(runtime, '_v21_condition_filter', False):
            return
        self._runtime_observe = runtime.observe
        self._runtime_preview = runtime.preview
        self._runtime_plan = runtime.plan
        self._runtime_complete = runtime.complete
        runtime.observe = self._observe_conditioned
        runtime.preview = self._preview_conditioned
        runtime.plan = self._plan_conditioned
        runtime.complete = self._complete_conditioned
        runtime._v21_condition_filter = True

    def _fresh_eval_core(self):
        old = self.owner.runtime.core
        core = V21ResearchCore(old.config)
        core.load_state_dict(old.state_dict())
        core.to(next(old.parameters()).device)
        core.eval().requires_grad_(False)
        if self.temporal_contract is not None:
            from .v21_temporal_contract import install_temporal_contract
            restored = install_temporal_contract(core, self.root/'fish/final.pt',
                                                  profile=self.temporal_contract['profile'])
            if restored != self.temporal_contract:
                raise RuntimeError('cloned core temporal feature identity differs')
        return core

    def _prepare_condition_reader(self):
        """Set addressing on an unused eval core; never rewrite a populated reader."""
        core = self.owner.runtime.core
        if self.camera_condition.address == 'factual' and core.reader.mode == 'readback':
            return
        if (core.reader._projection_cache is not None or self.camera_condition.h0_committed
                or any(module.training for module in core.modules())):
            core = self._fresh_eval_core()
            self.owner.runtime.core = core
        self.camera_condition.configure_reader(core)

    def _reset_camera_condition(self, name=None, address=None):
        name = self.camera_condition.name if name is None else name
        address = self.camera_condition.address if address is None else address
        self._reject_imported_memory_for_condition(name, address)
        self.camera_condition = V21CameraCondition(name, address)
        self._proposed_condition = None

    def _reject_imported_memory_for_condition(self, name, address):
        if name == 'both_fresh' and address == 'factual':
            return
        if getattr(self.owner, 'saved_column', None) is not None:
            raise RuntimeError('camera-condition trials cannot import historical column memory')
        if getattr(self.owner, 'active_views', None) == ('wrist',):
            raise RuntimeError('camera-condition H0 requires both actual cameras')

    def _force_commit_condition(self):
        following = replace(self.camera_condition)
        object.__setattr__(following, '_h0_committed', True)
        self.camera_condition = following

    def _conditioned_endpoint(self, endpoint):
        self._proposed_condition = None
        if getattr(self.owner, 'camera_condition', None) is not None:
            # Capture already proposed a filter. A second pass would fail the
            # two-delivery H0 seed after wrist/external was already zeroed.
            return endpoint
        if (not self._filter_actual_observations or endpoint is None
                or not isinstance(endpoint, EncodedEndpoint) or endpoint.observation is None):
            return endpoint
        filtered, proposed = self.camera_condition.filter(endpoint.observation)
        result = replace(endpoint, observation=filtered)
        if getattr(self.owner, 'last_endpoint', None) is endpoint:
            self.owner.last_endpoint = result
        self._proposed_condition = proposed
        return result

    def _commit_h0_if_empty_seed(self, previous):
        proposed = self._proposed_condition
        self._proposed_condition = None
        if previous is None and proposed is not None and self.owner.runtime.state is not None:
            self.camera_condition = proposed

    def _observe_conditioned(self, endpoint):
        previous = self.owner.runtime.state
        endpoint = self._conditioned_endpoint(endpoint)
        try:
            result = self._runtime_observe(endpoint)
        except BaseException:
            self._proposed_condition = None
            raise
        self._commit_h0_if_empty_seed(previous)
        return result

    def _preview_conditioned(self, endpoint, *args, **kwargs):
        endpoint = self._conditioned_endpoint(endpoint)
        self._proposed_condition = None
        return self._runtime_preview(endpoint, *args, **kwargs)

    def _plan_conditioned(self, *args, endpoint=None, **kwargs):
        if endpoint is not None:
            endpoint = self._conditioned_endpoint(endpoint)
        self._proposed_condition = None
        return self._runtime_plan(*args, endpoint=endpoint, **kwargs)

    def _complete_conditioned(self, actual_trace, measured_path, readback_times, endpoint, **kwargs):
        endpoint = self._conditioned_endpoint(endpoint)
        self._proposed_condition = None
        return self._runtime_complete(actual_trace, measured_path, readback_times, endpoint, **kwargs)

    def _parse_camera_condition_payload(self, payload, *, required=False):
        payload = {} if payload is None else payload
        if not isinstance(payload, dict):
            raise ValueError('camera condition payload must be a JSON object')
        extra = set(payload) - {'camera_condition', 'camera_address', 'name'}
        if extra:
            raise ValueError('camera condition accepts only camera_condition and camera_address')
        if required and not payload:
            raise ValueError('camera condition requires camera_condition or camera_address')
        name, address = payload.get('camera_condition', payload.get('name')), payload.get('camera_address')
        if payload.get('camera_condition') is not None and payload.get('name') not in (None, payload.get('camera_condition')):
            raise ValueError('camera_condition and name disagree')
        if name is not None and name not in CONDITIONS:
            raise ValueError('unknown camera condition')
        if address is not None and address not in ADDRESS_MODES:
            raise ValueError('camera address must be factual or zero')
        return name, address

    def _require_idle_home_empty_history(self):
        if self.trial.running or self._trial_admission_in_progress:
            raise RuntimeError('cannot change camera condition during a trial')
        rt = self.owner.runtime
        if rt.pending is not None or rt.executing:
            raise RuntimeError('cannot apply a camera condition onto an active live runtime')
        if rt.state is None:
            raise RuntimeError('explicit HOME is required before setting a camera condition')
        if bool(rt.state.history.valid.any()):
            raise RuntimeError('cannot apply a camera condition onto a populated live runtime')
        target = self._home_target()
        q = self.owner.read_checked()
        if np.max(np.abs(q-target)) > self.owner.io.settle_position_tolerance:
            raise RuntimeError('camera condition can only be set idle at HOME')

    def _set_camera_condition_at_home(self, payload):
        name, address = self._parse_camera_condition_payload(payload, required=True)
        self._require_idle_home_empty_history()
        self.owner.runtime.stop()
        self.owner.runtime.state = None
        self._reset_camera_condition(name, address)
        self._prepare_condition_reader()
        self.memory_reset_reason = 'camera condition set idle at HOME; empty memory reseeds H0'
        self.plan_record = None; self.preview_record = None
        with torch.inference_mode():
            self._observe()

    def _monitor_q(self):
        while not self.closing.wait(.5):
            if not self.operation_lock.acquire(blocking=False): continue
            try:
                self._monitor_readback_once()
            except Exception as exc:
                text=str(exc)
                packet='no status packet' in text.lower()
                misses=0
                if packet:
                    misses=getattr(self,'_status_packet_misses',0)+1
                    self._status_packet_misses=misses
                log=getattr(self,'_bus_faults',None)
                if log is not None and (packet or isinstance(exc,ConnectionError)):
                    log.failed(exc,'idle readback',getattr(self.owner,'last_readback_diagnostic',None))
                # #region agent log
                try:
                    import json as _json, time as _time
                    with open('/home/james/feedback-corrected-grounding/.cursor/debug-55ff20.log','a') as _f:
                        _f.write(_json.dumps({'sessionId':'55ff20','runId':'post-fix','hypothesisId':'A','location':'v21_live_service.py:_monitor_q','message':'idle joint read failed','data':{'packet':packet,'consecutive':misses,'stopped':not packet or misses>=3,'error':f'{type(exc).__name__}: {text}'[:180]},'timestamp':int(_time.time()*1000)})+'\n')
                except Exception:
                    pass
                # #endregion
                with self.display_lock: self.error = f'readback: {exc}'
                if packet and misses<3:
                    pass
                elif not getattr(self,'_idle_guard_tripped',False):
                    self._idle_guard_tripped=True
                    try:self.stop()
                    except Exception as stop_exc:
                        with self.display_lock:self.stop_error=str(stop_exc)
            finally:
                self.operation_lock.release()

    def _monitor_readback_once(self):
        """The idle readback display must enforce the same physical guard."""
        q=np.asarray(self.owner.read_checked())
        if q.shape!=(6,) or not np.isfinite(q).all():
            raise ValueError('idle physical readback must be finite [6]')
        log=getattr(self,'_bus_faults',None)
        if log is not None:
            log.good(q)
        self.owner._physical_check(q,np.repeat(q[None],5,axis=0))
        with self.display_lock:
            self.current_q,self.current_q_time=q.tolist(),self.clock()
            if 'no status packet' in str(getattr(self,'error','') or '').lower():
                self.error=None
        self._status_packet_misses=0
        self._idle_guard_tripped=False

    def status(self, *, compact=False):
        # No serial reads, camera reads, GPU calls or hidden observe/plan here.
        with self.display_lock:
            runtime = deepcopy(self.model_status)
            plan_requires_replan = False
            if self.owner is not None:
                rt = self.owner.runtime
                stopped = rt.stopped or self.owner.stop_event.is_set()
                ready = rt.pending is not None and not stopped and self.phase == 'idle'
                runtime.update(stopped=stopped, pending=rt.pending is not None, executing=rt.executing,
                    requires_home_after_incomplete_execution=bool(getattr(rt,'requires_home_after_incomplete_execution',False)))
                if rt.pending is not None and self.plan_record is not None and self.lewm_cem:
                    source = self.plan_record.get('source_time_s')
                    source_age = None if source is None else self.clock()-source
                    plan_requires_replan = (source_age is None or not 0 <= source_age <= rt.max_plan_age_s
                                            or not rt.pending_is_recently_ready())
            else:
                ready = False
            snapshot = None if self.snapshot is None else {k:deepcopy(v) for k,v in self.snapshot.items() if k not in ('rgb','raw')}
            compensation = getattr(self, 'fixed_goal_compensation', None) or {}
            compensation_source = compensation.get('source_time_s')
            # A restored process may still have no in-memory last_attempt: the
            # coordinator has already checked its same-output source file.
            # A newer execution must not inherit that old receipt's UI state.
            compensation_current = (type(compensation_source) in (int, float)
                and np.isfinite(compensation_source)
                and (self.last_attempt is None
                     or self.last_attempt.get('source_time_s') == compensation_source))
            return dict(model=dict(version=VERSION, contract=CONTRACT, weights_version=VERSION,
                            checkpoint=str(self.root/'fish'/'final.pt'), task=self.task,
                            target_token_repair=deepcopy(getattr(getattr(self.owner,'runtime',None),'target_token_repair',None)),
                            deployment_revision_only=False, pixel_keypoint_head=False,
                            runtime_revision='21.0.2-fish-body-target',
                            head_grasp_enabled=self.head_grasp,
                            grasp_target=self.grasp_target,
                            normal_follow_budget_s=getattr(getattr(self.owner,'bridge',None),'normal_follow_budget_s',0.),
                            hardware_maximum_lateness_s=getattr(getattr(self.owner,'bridge',None),'maximum_lateness_s',None),
                            normal_follow_observation_mode=('full_budget_same_goal'
                                if getattr(getattr(self.owner,'bridge',None),'normal_follow_budget_s',0.) == 2.
                                else 'bounded_until_hold_quiet'
                                if getattr(getattr(self.owner,'bridge',None),'normal_follow_budget_s',0.)
                                else 'disabled'),
                            normal_follow_arrival_criterion=dict(position_tolerance=.2,
                                step_tolerance=.1, quiet_samples=3,
                                position_unit='lerobot_calibrated_position'),
                            max_plan_age_s=getattr(getattr(self.owner,'runtime',None),'max_plan_age_s',None),
                            max_plan_dwell_s=getattr(getattr(self.owner,'runtime',None),'max_plan_dwell_s',None),
                            execution_planning='automatic_refresh' if self.replan_on_execute else 'explicit_plan_only',
                            gripper_follow_check_enabled=self.gripper_check_enabled,
                            tip_display_mode='same-exposure-wrist-fk-column-rgb',
                            target_reference_policy='same-view-registration-only',
                            planning_observations='candidate-until-execution',
                            assistance_state_updates='completed-transactions-only',
                            alignment_motion_enabled=self.alignment_motion_enabled,
                            alignment_scope=self.alignment_scope,
                            raw_actor=self.raw_actor,
                            place_goal=(None if getattr(self.owner, 'place_goal', None) is None
                                        else self.owner.place_goal.describe()),
                            lewm_cem=self.lewm_cem,
                            lewm_cem_objective=self.lewm_cem_objective,
                            lewm_cem_candidate_parameterization=self.lewm_cem_candidate_parameterization,
                            lewm_cem_action_scale=self.lewm_cem_action_scale,
                            lewm_cem_actual_response_joint_count=self.lewm_cem_response_joint_count,
                            lewm_controller=deepcopy(self.lewm_controller),
                            temporal_contract=deepcopy(self.temporal_contract),
                            temporal_provenance=self._temporal_provenance(),
                            lewm_cem_last=deepcopy(self.lewm_cem_record),
                            fish_task=deepcopy(self._fish_task),
                            wrist_scene_cost=deepcopy(self.wrist_scene_cost_status),
                            lewm_cem_trace=deepcopy(self.lewm_cem_trace_record),
                            lewm_shadow=dict(enabled=getattr(self,'lewm_shadow',None) is not None,
                                control_influence=False,last=deepcopy(getattr(self,'lewm_shadow_last',None))),
                            complete_rejected_response_recording=True,
                            policy_goal_readback_enabled=(getattr(getattr(self.owner,'bridge',None),
                                                                 'verify_policy_goal',None) is not None),
                            j3_position_servo_enabled=(bool(getattr(self.owner,'j3_position_servo',False))
                                and getattr(getattr(self.owner,'bridge',None),'finish_position',None) is not None),
                            arm_position_servo_enabled=(bool(getattr(self.owner,'arm_position_servo',False))
                                and getattr(getattr(self.owner,'bridge',None),'finish_position',None) is not None),
                            column_dino_fg_calibration_enabled=self.column_dino_fg_calibration_enabled,
                            column_dino_fg_transport=deepcopy(self.column_dino_fg_transport_status),
                            gripper_force=deepcopy(getattr(self.owner,'gripper_force',None)),
                            object_relative=deepcopy(getattr(self.owner,'last_object_relative',None)),
                            object_relative_source=(getattr(self.owner,'last_object_token_provenance',None) or {}).get('source'),
                            object_token_provenance=deepcopy(getattr(self.owner,'last_object_token_provenance',None)),
                            object_target_matches_overlay=bool(
                                getattr(self.owner,'last_object_relative',None) is not None
                                and self.owner.last_object_relative[0] > 0
                                and np.array_equal(self.owner.last_object_relative,
                                                   getattr(self.owner,'last_object_diagnostic',None))),
                            object_diagnostic=deepcopy(getattr(self.owner,'last_object_diagnostic',None)),
                            grounding_correction=dict(enabled=self.grounding_correction,px_per_unit=self.grounding_px_per_unit,
                                                      last=deepcopy(self.grounding_record)),
                            execution_geometry=dict(kind='vertical_pinch_at_close',
                                                    enabled=self._uses_vertical_pinch_geometry(),
                                                    learned_policy_success=False,
                                                    last=deepcopy(self.execution_geometry_record))),
                runtime=runtime, phase='trial' if self.trial.running and self.phase=='idle' else self.phase,
                ready=ready, plan_requires_replan=plan_requires_replan,
                connected=self.owner is not None, capped_trial=self.trial.status(),
                auxiliary_trial=self.auxiliary_budget.status(),
                actual_resume=deepcopy(self.actual_resume),
                actuator_state=deepcopy(getattr(self,'actuator_state',None)),
                j3_pid_diagnostic=(self.j3_pid_diagnostic.status()
                    if getattr(self, 'j3_pid_diagnostic', None) is not None else None),
                j3_range_diagnostic=(self.j3_range_diagnostic.status()
                    if getattr(self, 'j3_range_diagnostic', None) is not None else None),
                j3_compensation_diagnostic=(self.j3_compensation_diagnostic.status()
                    if getattr(self, 'j3_compensation_diagnostic', None) is not None else None),
                j3_native_response_diagnostic=(self.j3_native_response_diagnostic.status()
                    if getattr(self, 'j3_native_response_diagnostic', None) is not None else None),
                same_goal_diagnostic=(self.same_goal_diagnostic.status()
                    if getattr(self, 'same_goal_diagnostic', None) is not None else None),
                fixed_goal_compensation=dict(
                    enabled=bool(getattr(self,'arm_position_servo',False)),
                    source_time_s=compensation_source,
                    source_matches_current_execution=bool(compensation_current),
                    completed=compensation.get('completed'),
                    configuration_restored=compensation.get('configuration_restored'),
                    error=compensation.get('error'),
                    result_path=getattr(self,'fixed_goal_compensation_path',None)),
                last_hold={key:deepcopy(getattr(self.owner,'last_hold_record',{}).get(key)) for key in
                           ('event','reason','reused_existing_hold','motor_register_writes','monotonic_s')},
                hardware_connection={key:deepcopy(getattr(getattr(self.owner,'io',None),'safe_energize_preseed',{}).get(key))
                    for key in ('connection_mode','motor_register_writes','torque_changed','position_delta')},
                error=self.error, stop_error=self.stop_error,
                q_position=deepcopy(self.current_q),
                q_source_monotonic_s=deepcopy(self.current_q_time),
                q_age_s=None if self.current_q_time is None else self.clock()-self.current_q_time,
                camera_condition=self.camera_condition.name,
                camera_address=self.camera_condition.address,
                camera_h0_committed=self.camera_condition.h0_committed,
                memory_reset_reason=self.memory_reset_reason, snapshot=snapshot,
                input_mode=('live-wrist-plus-historical-column' if getattr(self.owner,'saved_column',None) is not None
                            else 'live-wrist-only' if getattr(self.owner,'active_views',None)==('wrist',)
                            else 'live-dual-view'),
                view_age_s=None if self.snapshot is None else [[self.clock()-t for t in self.snapshot['image_times'][0]]],
                snapshot_age_s=None if self.snapshot is None else self.clock()-self.snapshot['time'],
                preview=deepcopy(self.preview_record), plan=deepcopy(self.plan_record),
                last_execution=(_compact_execution_status(self.last_execution)
                                if compact else deepcopy(self.last_execution)),
                centroid_tracking=deepcopy(self.centroid_tracking),
                candidate_preview=deepcopy(self.candidate_preview_record),
                motion_validation=deepcopy(self.motion_validation_record),
                last_attempt=(_compact_execution_status(self.last_attempt)
                              if compact else deepcopy(self.last_attempt)),
                last_readback_diagnostic=deepcopy(getattr(self.owner,'last_readback_diagnostic',None)),
                execution_history=deepcopy(self.execution_history),
                fingertips=self._fingertips_with_step5(deepcopy(self.fingertips)),
                gripper_follow_check=deepcopy(self.gripper_check),
                alignment=deepcopy(self.alignment),
                camera_age_s={v:None if v not in self.live_times else self.clock()-self.live_times[v] for v in ('wrist','column')})

    def transport_status(self):
        """Compact status for the high-rate ROS/HTTP dashboard transport."""
        return self.status(compact=True)

    def _project_fg(self, view, q):
        """Same letterbox F/G projection already used to draw the current tips."""
        display = getattr(self, 'tip_display', None)
        kinematics = getattr(display, 'wrist_kinematics', None) or getattr(getattr(self, 'owner', None), 'kinematics', None)
        fit = None if display is None else (display.wrist_fit if view == 'wrist' else getattr(display, 'column_fit', None))
        if kinematics is None or fit is None:
            return None
        q = np.asarray(q, np.float64).reshape(6)
        world, local = kinematics.forward_anchor_frames(q[None])
        pair = predict_projection(fit, world, local, np.asarray([q[-1]]))[0, 1:3]
        pair = np.asarray(pair, np.float64)
        if pair.shape != (2, 2) or not np.isfinite(pair).all():
            return None
        return pair

    def _image_anchored_lewm_cost(self):
        """Distance plus angle on every view that actually sees the fish."""
        from .v21_visual_task_cost import BothViewsVisualCost, ImageAnchoredVisualCost, MeasuredWristVisualCost, MovingWristPlaneCost, signed_blue_white_angle_deg
        targets = ((getattr(self, 'targets', None) or {}).get('targets') or {})
        pickup = targets.get('pickup') or {}
        tips = getattr(self, 'fingertips', None) or {}
        enc = getattr(getattr(self, 'owner', None), 'object_relative', None)
        wrist_record = getattr(enc, 'last_policy_record', None) if enc is not None else None
        parts = []
        used = []
        self.wrist_scene_cost_status = dict(available=False, reason='no current measured Wrist scene response')

        def add(view, cog, image, q0, fish_axis):
            cog = np.asarray(cog, np.float64).reshape(2)
            image = np.asarray(image, np.float64)
            q0 = np.asarray(q0, np.float64).reshape(6)
            if (cog.shape != (2,) or image.shape != (2, 2) or q0.shape != (6,)
                    or not np.isfinite(np.r_[cog, image.ravel(), q0]).all()):
                return
            blue = None
            axis = np.asarray([] if fish_axis is None else fish_axis, np.float64).reshape(-1)
            if axis.shape == (2,) and np.isfinite(axis).all() and float(np.linalg.norm(axis)) > 1e-8:
                axis = axis / float(np.linalg.norm(axis))
                blue = np.array((-axis[1], axis[0]), np.float64)
            parts.append(ImageAnchoredVisualCost(
                lambda q, view=view: self._project_fg(view, q),
                image, q0, cog, view=view, blue_axis_px=blue))
            used.append(dict(view=view, target=np.round(cog, 1).tolist(),
                             jaw=np.round(image.mean(0), 1).tolist()))

        def mass_px(record):
            # The blue line is drawn through this point. White FG coincides
            # with it when jaw midpoint M sits here and the jaw is parallel.
            return (record.get('cog_px') or record.get('centroid_px')
                    or record.get('blob_centroid_px'))

        column_fish = pickup.get('column') or {}
        column_mass = mass_px(column_fish) if isinstance(column_fish, dict) else None
        if (isinstance(column_fish, dict) and column_fish.get('source') != 'estimated_fg'
                and column_fish.get('visible') is not False and column_mass is not None):
            tip = tips.get('column') or {}
            image = tip.get('fm_px') if tip.get('available') is True else tip.get('display_fm_px')
            q0 = tip.get('q_at_image') or getattr(self, 'current_q', None)
            if image is not None and q0 is not None:
                add('column', column_mass, image, q0, column_fish.get('axis_px'))
        if isinstance(wrist_record, dict) and wrist_record.get('source') not in ('estimated_fg', 'dino_fg_near_prior'):
            cog = mass_px(wrist_record)
            tip = tips.get('wrist') or {}
            image = tip.get('fm_px') if tip.get('available') is True else tip.get('display_fm_px')
            q0 = tip.get('q_at_image') or getattr(self, 'current_q', None)
            if image is None and q0 is not None:
                image = self._project_fg('wrist', q0)
            if cog is not None and image is not None and q0 is not None:
                # Jaw-local FK says almost nothing about a static fish moving
                # through an eye-in-hand image. Use actual measured scene
                # response; never silently fall back to fixed target pixels.
                response = getattr(self, 'wrist_grasp_response', None)
                self.wrist_scene_cost_status = dict(available=False,
                    reason='moving Wrist response not calibrated; stationary target scoring disabled')
                if response is not None:
                    try:
                        # Existing consumer pairs the Actor target to its
                        # camera-acquisition readback, never commanded q.
                        part = MeasuredWristVisualCost(self._alignment_measure(), response)
                        parts.append(part)
                        self.wrist_scene_cost_status = part.receipt()
                    except (KeyError, TypeError, ValueError) as error:
                        self.wrist_scene_cost_status = dict(available=False, reason=str(error))
                if not any(getattr(part, 'view', None) == 'wrist' for part in parts):
                    try:
                        from .v21_fish_task_cost import FishTaskCost
                        current = self.owner.last_endpoint.observation
                        capture = current.q_at_image[0, 0].detach().cpu().numpy()
                        stamp = float(current.image_times[0, 0])
                        low = ('low_z_m' in self._fish_task or self.owner.clearance.clearance_mm(capture)
                               <= FishTaskCost.WORKING_CLEARANCE_MM + FishTaskCost.FLOOR_FINISH_MARGIN_MM)
                        if low and self._fish_wrist_geometry(capture) is not None:
                            if not hasattr(self, '_wrist_camera_model'):
                                self._wrist_camera_model = json.loads(WRIST_CAMERA_REFERENCE.read_text())['wrist']
                            pair = self._project_fg('wrist', capture)
                            part = MovingWristPlaneCost(self.owner.kinematics, self._wrist_camera_model,
                                capture, pair, cog, wrist_record['axis_px'],
                                plane_z_m=self.owner.clearance.table_z_mm / 1000., image_time_s=stamp)
                            parts.append(part)
                            self.wrist_scene_cost_status = part.receipt()
                    except (OSError, KeyError, TypeError, ValueError) as error:
                        self.wrist_scene_cost_status = dict(available=False, reason=str(error))
                used.append(dict(view='wrist', target=np.asarray(cog).tolist(),
                                 jaw=np.asarray(image).mean(0).tolist()))
        wrist_dx = None
        wrist_dy = None
        wrist_angle = None
        if isinstance(wrist_record, dict) and wrist_record.get('source') != 'estimated_fg':
            tip = tips.get('wrist') or {}
            jaw = tip.get('fm_px') if tip.get('available') is True else tip.get('display_fm_px')
            if jaw is None:
                q_jaw = tip.get('q_at_image') or getattr(self, 'current_q', None)
                jaw = None if q_jaw is None else self._project_fg('wrist', q_jaw)
            wrist_angle = signed_blue_white_angle_deg(jaw, wrist_record.get('axis_px'))
        for item in used:
            if item['view'] == 'wrist':
                wrist_dx = float(item['target'][0] - item['jaw'][0])
                wrist_dy = float(item['target'][1] - item['jaw'][1])
        # #region agent log
        try:
            import json as _json, time as _time
            with open('/home/james/feedback-corrected-grounding/.cursor/debug-55ff20.log', 'a') as _f:
                _f.write(_json.dumps({'sessionId': '55ff20', 'hypothesisId': 'j1-side', 'runId': 'post-fix', 'location': 'v21_live_service.py:_image_anchored_lewm_cost', 'message': 'j1 follows wrist side', 'data': {'wrist_dx': None if wrist_dx is None else round(float(wrist_dx), 1), 'views': [item['view'] for item in used]}, 'timestamp': int(_time.time() * 1000)}) + '\n')
        except Exception:
            pass
        # #endregion
        if not parts:
            return None, 'current image F/G and fish are required to score LeWM'
        try:
            wrist_dist = None
            if wrist_dx is not None and wrist_dy is not None:
                wrist_dist = float(np.hypot(wrist_dx, wrist_dy))
            return BothViewsVisualCost(
                parts, wrist_dx_px=wrist_dx, wrist_angle_deg=wrist_angle,
                wrist_dist_px=wrist_dist), None
        except (TypeError, ValueError) as error:
            return None, str(error)

    def _fingertips_with_step5(self, tips):
        """Current column F/G stays on screen. Predicted F/G stays until the next plan."""
        if not isinstance(tips, dict):
            return tips
        self._ensure_column_display_fg(tips)
        plan = getattr(self, 'plan_record', None)
        if not isinstance(plan, dict):
            self._keep_column_prediction(tips)
            return tips
        try:
            predicted = np.asarray(plan.get('predicted_q_path'), np.float64)
            commands = np.asarray(plan.get('commands_position'), np.float64)
            if predicted.ndim == 3:
                predicted = predicted[0]
            if commands.ndim == 3:
                commands = commands[0]
            if predicted.shape != (6, 6) or commands.shape[0] < 1 or commands.shape[1:] != (6,):
                self._keep_column_prediction(tips)
                return tips
            targets = (('f5_px', 'g5_px', predicted[-1]), ('raw_f5_px', 'raw_g5_px', commands[-1]))
            for tip in tips.values():
                if not isinstance(tip, dict):
                    continue
                raw_image = tip.get('fm_px') if tip.get('available') else tip.get('display_fm_px')
                image = np.asarray(raw_image, np.float64)
                q0 = np.asarray(tip.get('q_at_image'), np.float64).reshape(6)
                if q0.shape != (6,) or not np.isfinite(q0).all():
                    continue
                view = 'wrist' if tip.get('source') == 'wrist-fk-projection' else 'column'
                start = self._project_fg(view, q0)
                if image.shape != (2, 2) or not np.isfinite(image).all():
                    if start is None:
                        continue
                    image = start
                if start is None:
                    continue
                shift = image - start
                for f_key, g_key, target_q in targets:
                    future = self._project_fg(view, target_q)
                    if future is None:
                        continue
                    shown = future + shift
                    tip[f_key] = shown[0].tolist()
                    tip[g_key] = shown[1].tolist()
        except Exception:
            self._keep_column_prediction(tips)
            return tips
        column = tips.get('column')
        if isinstance(column, dict) and column.get('f5_px') and column.get('g5_px'):
            self._column_fg_prediction = dict(f5_px=column['f5_px'], g5_px=column['g5_px'])
        else:
            self._keep_column_prediction(tips)
        return tips

    def _ensure_column_display_fg(self, tips):
        column = tips.get('column')
        if not isinstance(column, dict):
            return
        if column.get('available') and column.get('fm_px') is not None:
            column.setdefault('display_fm_px', column.get('fm_px'))
            return
        return

    def _keep_column_prediction(self, tips):
        saved = getattr(self, '_column_fg_prediction', None)
        column = tips.get('column') if isinstance(tips, dict) else None
        if isinstance(saved, dict) and isinstance(column, dict) and not column.get('f5_px'):
            column.update(saved)

    @staticmethod
    def _forecast_record(commands, forecast):
        return dict(source_time_s=forecast.source_time.item(),
            commands_position=commands.detach().cpu().tolist(), source_q_position=forecast.source_q.cpu().tolist(),
            command_offsets_s=forecast.trace.offsets.cpu().tolist(),
            predicted_q_path=forecast.response.predicted_q_path.cpu().tolist(),
            readback_offsets_s=forecast.query_offsets.cpu().tolist())

    def _record_plan(self, commands):
        self.lewm_shadow_plan = None
        self.plan_record = self._forecast_record(commands, self.owner.runtime.pending)
        if (self.lewm_cem_record or {}).get('command_origin') == 'python_near_blue_close':
            # Retain the real model forecast for display/error measurement, but
            # it neither chooses nor changes this J6-only assisted command.
            self.lewm_cem_record['selected_predicted_q_path'] = deepcopy(
                self.plan_record['predicted_q_path'][0])
        self.plan_record['plan_ready_time_s'] = self.owner.runtime.pending_ready_time_s
        self.plan_record['action_source']=self.owner.runtime.pending_action_source
        if self.lewm_controller is not None:
            self.plan_record['lewm_controller'] = deepcopy(self.lewm_controller)
        if self.temporal_contract is not None:
            self.plan_record['temporal_contract'] = deepcopy(self.temporal_contract)
        if self.grounding_correction: self.plan_record['grounding_correction']=deepcopy(self.grounding_record)
        if self.lewm_cem: self.plan_record['lewm_cem']=deepcopy(self.lewm_cem_record)
        if self.c4 is not None: self.plan_record[getattr(self.c4,'record_key','c4_planner')]=deepcopy(self.c4.record)
        source = self.owner.runtime._pending_state
        previous = self.owner.runtime.state
        self._planned_source_evidence = None
        self._planned_object_relative_snapshot = self._object_relative_snapshot(source)
        if source is not None:
            if self.owner.runtime.pending_action_source == 'v21_lewm_cem' and self.lewm_cem_trace_record is not None:
                trace = self.lewm_cem_trace_record
                if (not trace.get('complete') or trace.get('source_time_s') != float(source.time[0])
                        or trace.get('source_q_position') != source.q[0].detach().cpu().tolist()
                        or trace.get('selected_commands_position') != commands.detach().cpu().tolist()):
                    raise RuntimeError('CEM trace is not bound to this pending source and selected commands')
                self.plan_record['cem_trace'] = deepcopy(trace)
            if self.record_policy_inputs:
                from .v21_source_evidence import snapshot_source_evidence, save_source_evidence
                frozen = self._cem_source_evidence if self.owner.runtime.pending_action_source == 'v21_lewm_cem' else None
                if (frozen is not None and frozen[0].manifest['source_time_s'] == float(source.time[0])
                        and frozen[0].manifest['source_q_position'] == source.q[0].detach().cpu().tolist()):
                    bundle, reference = frozen
                else:
                    bundle = snapshot_source_evidence(source, getattr(self.owner,'source_evidence_store',None))
                    reference = save_source_evidence(self.output, f'source-plan-{time.time_ns()}', bundle)
                self._planned_source_evidence = (bundle, reference)
                self.plan_record['source_visual_evidence'] = deepcopy(reference)
            self.plan_record['visual_feedback'] = dict(
                available=source.visual_feedback_available.cpu().tolist(),
                event_time_s=source.visual_feedback_time.cpu().tolist(),
                norm=source.visual_error.norm(dim=(-1, -2)).cpu().tolist(),
                equal_to_committed_event=None if previous is None else
                    (source.visual_error == previous.visual_error).all(-1).all(-1).cpu().tolist(),
                semantics='last unconsumed completed-execution event; current grounding is refreshed')
        if getattr(self,'lewm_shadow',None) is not None:
            record=self.lewm_shadow.plan(source,self.owner.runtime.pending)
            path=self.output/f'lewm-shadow-plan-{time.time_ns()}.json'
            write_json(path,record)
            self.lewm_shadow_plan=dict(record=record,path=str(path.resolve()),
                persisted_at_s=float(self.clock()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            self.plan_record['lewm_shadow_plan']=path.name

    @staticmethod
    def _coerce_object_relative(value):
        """Normalize one Actor object cue without changing the control path."""
        result = dict(vector=None, shape=None, finite=False, available=False,
                      valid=False, reason='missing')
        if value is None:
            return result
        try:
            raw = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            result['shape'] = list(raw.shape)
            # Runtime state is [B,8]; a single-sample source may also be [8].
            if raw.shape not in ((8,), (1, 8)):
                result['reason'] = 'invalid_shape'
                return result
            vector = np.asarray(raw, dtype=np.float64).reshape(8)
        except (TypeError, ValueError, RuntimeError):
            result['reason'] = 'invalid_value'
            return result
        result['finite'] = bool(np.isfinite(vector).all())
        if not result['finite']:
            result['reason'] = 'nonfinite'
            return result
        result['vector'] = [float(item) for item in vector]
        result['available'] = bool(vector[0] > 0.)
        result['valid'] = True
        result['reason'] = 'available' if result['available'] else 'unavailable'
        return result

    def _object_relative_snapshot(self, source, *, provenance=_OBJECT_RELATIVE_PROVENANCE_UNSET):
        """Freeze the exact source cue and identity used by a planned block."""
        normalized = self._coerce_object_relative(
            None if source is None else getattr(source, 'object_relative', None))
        if provenance is _OBJECT_RELATIVE_PROVENANCE_UNSET:
            provenance = getattr(self.owner, 'last_object_token_provenance', None)
        snapshot = dict(
            object_relative=normalized['vector'],
            object_relative_available=normalized['available'],
            object_relative_reason=('source_missing' if source is None else normalized['reason']),
            object_relative_finite=normalized['finite'],
            object_relative_shape=normalized['shape'],
            object_relative_snapshot_valid=normalized['valid'],
            object_relative_source='dispatch_actor_input',
            object_token_provenance=deepcopy(provenance),
            object_relative_source_time_s=None,
            object_relative_source_q_position=None,
        )
        if source is not None:
            # Internal identity only; it is intentionally omitted from the
            # JSON-facing block record.
            snapshot['_source_object_id'] = id(source)
        if source is None:
            return snapshot
        try:
            source_time = source.time.detach().cpu().numpy().reshape(-1)
            if source_time.shape == (1,) and np.isfinite(source_time).all():
                snapshot['object_relative_source_time_s'] = float(source_time[0])
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass
        try:
            source_q = source.q.detach().cpu().numpy()
            if source_q.shape == (1, 6) and np.isfinite(source_q).all():
                snapshot['object_relative_source_q_position'] = source_q[0].astype(float).tolist()
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass
        return snapshot

    def _save_lewm_shadow_attempt(self):
        """Evaluate frozen dispatch inputs only after bridge completion or HOLD."""
        if getattr(self,'lewm_shadow',None) is None:return
        bridge=self.owner.bridge
        attempt=bridge.last_attempt
        source=getattr(bridge,'last_source_state',None)
        forecast=getattr(bridge,'last_forecast',None)
        if attempt is None or source is None or forecast is None:return
        if 'lewm_shadow_evaluation' in attempt:return
        try:
            record=self.lewm_shadow.complete(source,forecast,attempt,self.lewm_shadow_plan)
            path=self.output/f'lewm-shadow-execution-{time.time_ns()}.json'
            write_json(path,record)
            summary=dict(path=path.name,control_influence=False,
                         physical_acceptance_validated=False)
        except Exception as exc:
            # A logging/model failure must not replace the physical failure or
            # make a partially executed block look like a valid comparison.
            summary=dict(error=f'{type(exc).__name__}: {exc}',control_influence=False,
                         physical_acceptance_validated=False)
        attempt['lewm_shadow_evaluation']=summary
        self.lewm_shadow_last=deepcopy(summary)

    def _record_column_dino_fg_sample(self, tips):
        """Accumulate actual Column F/G observations; this never plans or moves."""
        if getattr(self, '_column_dino_fg_calibration_frozen', False):
            return
        tip = (tips or {}).get('column') or {}
        if not tip_source_matches_view('column', tip) or not tip.get('available'):
            return
        try:
            sample = ColumnDinoFGSample(
                tip.get('q_at_image'), tip.get('m_px'), float(tip.get('image_time_s')))
        except (TypeError, ValueError):
            return
        samples = self._column_dino_fg_samples
        if samples and sample.image_time_s <= samples[-1].image_time_s:
            return
        samples.append(sample)
        del samples[:-12]
        self._refit_column_dino_fg_transport()

    def _refit_column_dino_fg_transport(self):
        """Publish only a fit backed by the retained actual exposure samples."""
        samples = self._column_dino_fg_samples
        try:
            response = fit_column_dino_fg_transport(samples)
        except (TypeError, ValueError) as exc:
            self.column_dino_fg_response = None
            self.column_dino_fg_transport_status = dict(
                available=False, sample_count=len(samples), reason=str(exc))
            return
        self.column_dino_fg_response = response
        self.column_dino_fg_transport_status = dict(
            available=True, sample_count=int(response['sample_count']),
            residual_rms_px=float(response['residual_rms_px']),
            design_condition=float(response['design_condition']),
            source='actual_exposure_paired_column_dino_fg',
            response_joint_count=int(np.asarray(response['jacobian_px_per_joint']).shape[1]),
            gripper_response_measured=np.asarray(response['jacobian_px_per_joint']).shape[1] == 6,
            lower_q=np.asarray(response['lower_q']).tolist(), upper_q=np.asarray(response['upper_q']).tolist(),
            lower_q6=response.get('lower_q6'), upper_q6=response.get('upper_q6'),
        )

    def _column_dino_fg_probe(self, payload):
        """One confirmed, open-jaw calibration chunk for the Column DINO F/G fit."""
        if not self.column_dino_fg_calibration_enabled:
            raise RuntimeError('Column DINO F/G calibration was not enabled at startup')
        if (not isinstance(payload, dict) or set(payload) != {'joint', 'direction', 'confirmed'}
                or type(payload['joint']) is not int or not 0 <= payload['joint'] < 6
                or type(payload['direction']) is not int or payload['direction'] not in (-1, 1)
                or payload['confirmed'] is not True):
            raise ValueError('probe requires joint 0..5, direction -1/+1, and confirmed=true')
        if self.owner.runtime.pending is not None or self.owner.runtime.executing:
            raise RuntimeError('stop the pending transaction before a Column DINO F/G probe')
        tip = (self.fingertips or {}).get('column') or {}
        if not tip_source_matches_view('column', tip) or not tip.get('available'):
            raise RuntimeError('a fresh column-rgb-jaw-contour observation is required')
        joint, direction = payload['joint'], payload['direction']
        response = self.column_dino_fg_response
        upgrading_gripper = (joint == 5 and isinstance(response, dict)
            and np.asarray(response.get('jacobian_px_per_joint')).shape == (2, 5))
        if self.column_dino_fg_transport_status.get('available') and not upgrading_gripper:
            raise RuntimeError('Column DINO F/G transport is already fitted; do not extend the bounded calibration')
        if self.auxiliary_budget.active and self.auxiliary_budget.block_attempts >= self.auxiliary_budget.maximum_blocks-1:
            raise RuntimeError('Column DINO calibration must retain one auxiliary block for HOME')
        if not self.auxiliary_budget.active:
            # A new open-jaw calibration may not silently mix an earlier task's
            # gripper pose, target layout, or post-restart state with its local
            # arm response.  The current exposure becomes the only baseline.
            self._column_dino_fg_samples=[]
            self._column_dino_fg_calibration_frozen=False
            self.column_dino_fg_response=None
            self.column_dino_fg_transport_status=dict(
                available=False, sample_count=0,
                reason='starting_fresh_open_jaw_column_dino_fg_calibration')
            self._record_column_dino_fg_sample(self.fingertips)
        joint, direction = payload['joint'], payload['direction']

        def build(state):
            q = state.q[0]
            ramp = torch.linspace(.2, 1., 5, device=q.device, dtype=q.dtype)
            action = q[None, None, :].repeat(1, 5, 1).clone()
            action[0, :, joint] += float(direction) * 2. * ramp
            if joint != 5:
                action[0, :, 5] = q[5]
            else:
                # J6 calibration is a small empty-jaw perturbation near HOME,
                # never a grasp attempt or a task completion label.
                home = self._home_target()
                if (np.max(np.abs(q[:5].detach().cpu().numpy()-home[:5])) > 3.
                        or not 28. <= float(action[0, :, 5].min())
                        or not float(action[0, :, 5].max()) <= 36.):
                    raise RuntimeError('J6 Column calibration requires empty open jaws near HOME')
            return action

        samples_before = tuple(self._column_dino_fg_samples)
        try:
            result = self._run_alignment_motion(build, 'column_dino_fg_probe')
            result['actual_probe_delta'] = require_actual_probe_response(
                result, joint, direction,
                minimum_motion=float(self.owner.io.settle_step_tolerance)).tolist()
        except BaseException:
            # _run_alignment_motion may already have published a post-command
            # camera frame.  It is not calibration evidence when the matching
            # measured q did not respond; remove only that newly appended frame.
            self._column_dino_fg_samples[:] = samples_before
            self._refit_column_dino_fg_transport()
            raise
        # Keep one factual endpoint per successful probe; intermediate captures
        # must not evict independent early axes from the bounded fit window.
        self._column_dino_fg_samples[:] = samples_before
        self._refit_column_dino_fg_transport()
        self._record_column_dino_fg_sample(self.fingertips)
        if (not self._column_dino_fg_samples or not samples_before
                or self._column_dino_fg_samples[-1].image_time_s <= samples_before[-1].image_time_s):
            self._column_dino_fg_samples[:] = samples_before
            self._refit_column_dino_fg_transport()
            raise RuntimeError('verified probe lacks a new same-exposure Column midpoint sample')
        fitted_six = (self.column_dino_fg_transport_status.get('available')
            and isinstance(self.column_dino_fg_response, dict)
            and np.asarray(self.column_dino_fg_response.get('jacobian_px_per_joint')).shape == (2, 6))
        if fitted_six:
            # Preserve the independent calibration design across HOME and
            # later policy captures. Binding still uses each fresh exposure.
            self._column_dino_fg_calibration_frozen = True
            self.column_dino_fg_transport_status['calibration_frozen'] = True
        result['column_dino_fg_probe'] = dict(
            joint=joint, direction=direction, requested_delta=2.,
            confirmed=True, gripper_command_held=joint != 5,
            calibration_scope='empty_jaw_six_joint_local_image_response',
            actual_joint_response_verified=True, response_sample_count=len(self._column_dino_fg_samples),
            physical_response_verified=False, task_success_verified=False,
            calibration_samples=[dict(q_at_image=row.q_at_image.tolist(),
                midpoint_px=row.midpoint_px.tolist(), image_time_s=row.image_time_s)
                for row in self._column_dino_fg_samples] if fitted_six else None,
            transport=deepcopy(self.column_dino_fg_transport_status),
        )
        write_json(self.output/f'column-dino-fg-probe-{time.time_ns()}.json', result)
        # The motion runner saved its snapshot before the probe's readback and
        # fit checks. Publish the validated receipt for HTTP/ROS consumers too.
        self.last_execution = deepcopy(result)
        return result

    # Post-HOLD logging only. execution_geometry still runs core.act(); remaps commands after.
    _RECORDABLE_POLICY_INPUTS = ('v21_actor', 'v21_actor_execution_geometry', 'v21_lewm_cem',
                                 'v21_actor_centroid_gated', 'visual_alignment_servo',
                                 'v21_actor_explore', 'v21_actor_cue_shadow', 'v21_lewm_cue_cem')

    def _block_io_snapshot(self, object_relative=_BLOCK_IO_OBJECT_RELATIVE_UNSET, *,
                           object_relative_snapshot=None):
        """Compact in/out for one executed block — answers 'what did we see / send?'."""
        if object_relative_snapshot is not None:
            # A captured source, including an all-zero valid cue, is explicit.
            # Never replace it with the owner's post-HOLD display state.
            cue = object_relative_snapshot.get('object_relative')
            normalized = self._coerce_object_relative(cue)
            cue_source = object_relative_snapshot.get('object_relative_source',
                                                       'dispatch_actor_input')
            cue_metadata = deepcopy(object_relative_snapshot)
        elif object_relative is _BLOCK_IO_OBJECT_RELATIVE_UNSET:
            cue = getattr(self.owner, 'last_object_relative', None)
            normalized = self._coerce_object_relative(cue)
            cue_source = 'owner_last_object_relative'
            cue_metadata = dict(
                object_relative_source=cue_source,
                object_relative_snapshot_valid=False,
                object_relative_finite=normalized['finite'],
                object_relative_available=normalized['available'],
                object_relative_reason='owner_fallback_' + normalized['reason'],
                object_relative_shape=normalized['shape'],
                object_token_provenance=deepcopy(
                    getattr(self.owner, 'last_object_token_provenance', None)),
                object_relative_source_time_s=None,
                object_relative_source_q_position=None,
            )
        else:
            normalized = self._coerce_object_relative(object_relative)
            cue_source = 'dispatch_actor_input'
            cue_metadata = dict(
                object_relative_source=cue_source,
                object_relative_snapshot_valid=normalized['valid'],
                object_relative_finite=normalized['finite'],
                object_relative_available=normalized['available'],
                object_relative_reason=normalized['reason'],
                object_relative_shape=normalized['shape'],
                object_token_provenance=None,
                object_relative_source_time_s=None,
                object_relative_source_q_position=None,
            )
        or8 = normalized['vector']
        enc = getattr(self.owner, 'object_relative', None)
        policy = getattr(enc, 'last_policy_record', None) or {}
        display = getattr(enc, 'last_record', None) or {}
        pickup = ((self.targets or {}).get('targets') or {}).get('pickup') or {}
        column = pickup.get('column') or {}
        wrist = pickup.get('wrist') or {}
        geom = deepcopy(self.execution_geometry_record) if self.execution_geometry_record else None
        wrist_r = (geom or {}).get('wrist_r') if isinstance(geom, dict) else None
        aim = (geom or {}).get('aim') if isinstance(geom, dict) else None
        column_px = None
        try:
            column_px = self._column_grasp_pixel(column)
            if column_px is not None:
                column_px = [float(column_px[0]), float(column_px[1])]
        except (TypeError, ValueError, AttributeError):
            column_px = None
        target_xyz = None
        try:
            xyz = self._column_grasp_target_xyz()
            if xyz is not None:
                target_xyz = [float(x) for x in np.asarray(xyz, np.float64).reshape(-1)[:3]]
        except (TypeError, ValueError, AttributeError):
            target_xyz = None

        def _px(rec):
            if not isinstance(rec, dict):
                return None
            for key in ('cog_px', 'centroid_px'):
                value = rec.get(key)
                if value is None:
                    continue
                try:
                    p = np.asarray(value, np.float64).reshape(-1)
                    if p.shape == (2,) and np.isfinite(p).all():
                        return [float(p[0]), float(p[1])]
                except (TypeError, ValueError):
                    pass
            return None

        r = None
        if or8 is not None and len(or8) >= 4:
            r = float(np.hypot(or8[2], or8[3]))
        tip = (self.fingertips or {}).get('column') or {}
        cm = None
        try:
            c, m = np.asarray(_px(column), dtype=float), np.asarray(tip.get('m_px'), dtype=float)
            if (tip_source_matches_view('column', tip) and tip.get('available')
                    and c.shape == m.shape == (2,)
                    and np.isfinite(c).all() and np.isfinite(m).all()):
                cm = float(np.linalg.norm(c-m))
        except (TypeError, ValueError):
            pass
        return dict(
            event='v21_block_io',
            object_relative=or8,
            object_relative_source=cue_source,
            object_relative_snapshot_valid=bool(
                cue_metadata.get('object_relative_snapshot_valid', False) and normalized['valid']),
            object_relative_finite=bool(
                cue_metadata.get('object_relative_finite', normalized['finite']) and normalized['finite']),
            object_relative_available=bool(
                cue_metadata.get('object_relative_available', normalized['available']) and normalized['available']),
            object_relative_reason=cue_metadata.get('object_relative_reason'),
            object_relative_shape=deepcopy(cue_metadata.get('object_relative_shape', normalized['shape'])),
            object_relative_source_time_s=cue_metadata.get('object_relative_source_time_s'),
            object_relative_source_q_position=deepcopy(cue_metadata.get('object_relative_source_q_position')),
            object_token_provenance=deepcopy(cue_metadata.get('object_token_provenance')),
            object_token_provenance_matches_dispatch_vector=bool(
                cue_metadata.get('object_token_provenance_matches_dispatch_vector', False)),
            wrist_r=r,
            wrist_policy_cog_px=_px(policy),
            wrist_display_cog_px=_px(display),
            wrist_policy_source=policy.get('source'),
            wrist_display_source=display.get('source'),
            column_pickup_cog_px=_px(column),
            column_pickup_source=column.get('source'),
            column_cm_px=cm,
            column_cm_diagnostic_only=True,
            column_m_source=tip.get('source'),
            column_m_accuracy_verified=bool(tip.get('accuracy_verified')),
            column_grasp_pixel_px=column_px,
            column_target_xyz_m=target_xyz,
            column_support_clipped=column.get('support_clipped'),
            wrist_dino_clipped=wrist.get('support_clipped'),
            approach_override=(geom or {}).get('approach_override') if isinstance(geom, dict) else None,
            close_veto=(wrist_r or {}).get('close_veto') if isinstance(wrist_r, dict) else None,
            wrist_cue=(wrist_r or {}).get('cue') if isinstance(wrist_r, dict) else None,
            wrist_servo=(None if not isinstance(wrist_r, dict) else
                         {k: (wrist_r.get('servo') or {}).get(k)
                          for k in ('applied', 'reason', 'error_px', 'error_px_after_step',
                                    'joint_delta', 'solver', 'err_x_px', 'j1_pan_deg',
                                    'mid_px', 'target_px', 'actor_reach_preserved')}),
            aim=None if not isinstance(aim, dict) else
                {k: aim.get(k) for k in ('applied', 'reason', 'error_m', 'error_m_after_step',
                                         'target_xyz', 'm_xyz')},
            geom_kind=(geom or {}).get('kind') if isinstance(geom, dict) else None,
            force_close=(geom or {}).get('force_close') if isinstance(geom, dict) else None,
            force_close_deferred=(geom or {}).get('force_close_deferred') if isinstance(geom, dict) else None,
        )

    def _temporal_provenance(self):
        """Expose the loaded consumer separately from the optional deployment patch."""
        from .v21_temporal_contract import _consumer_contract
        core = getattr(getattr(self.owner, 'runtime', None), 'core', None)
        training = getattr(core, 'temporal_training_contract', None)
        identity = getattr(core, 'temporal_contract_identity', None)
        effective = _consumer_contract(core)
        return dict(
            deployment_compatibility_identity=deepcopy(self.temporal_contract),
            checkpoint_training_contract=deepcopy(training),
            effective_consumer_contract=deepcopy(effective),
            effective_source=('deployment_compatibility_identity' if identity is not None
                else 'checkpoint_training_contract' if effective is not None
                else 'legacy_wall_clock'),
            real_timestamps_preserved=True)

    def _save_failed_dispatch_evidence(self, attempt):
        """Retain replay inputs after HOLD without creating a complete training sample.

        The normal execution writer deliberately requires completion. Failed
        dispatches have their own artifact kind and filename so existing dataset
        discovery and validators cannot mistake a partial path for five labels.
        """
        if (attempt is None
                or (attempt.get('completed') is True and attempt.get('protective_hold_after_q5') is True)
                or (attempt.get('failure_evidence_record') or {}).get('saved') is True
                or attempt.get('action_source') not in self._RECORDABLE_POLICY_INPUTS):
            return
        bridge = self.owner.bridge
        source = getattr(bridge, 'last_actor_input', None)
        trace = self.lewm_cem_trace_record if attempt.get('action_source') == 'v21_lewm_cem' else None
        if source is None or (attempt.get('predicted_q_path') is None and trace is None):
            return
        stamp = time.time_ns()
        receipt = dict(path=f'failed-execution-{stamp}.json', saved=False,
            executable=False, training_sample=False, model_feedback_label=False,
            task_success_inferred=False)
        attempt['failure_evidence_record'] = receipt
        try:
            dispatch = getattr(bridge, 'last_source_state', None)
            if (dispatch is None
                    or float(dispatch.time[0]) != float(attempt['source_time_s'])
                    or not np.array_equal(dispatch.q[0].detach().cpu().numpy(), attempt['source_q_position'])
                    or float(source.time[0]) > float(dispatch.time[0])):
                raise ValueError('failed replay input differs from this dispatch source')
            if attempt['action_source'] == 'v21_lewm_cem':
                attempt['lewm_cem'] = deepcopy(self.lewm_cem_record)
                if trace is not None:
                    attempt['cem_trace'] = deepcopy(trace)
            from .v21_source_evidence import snapshot_source_evidence, save_source_evidence
            evidence = self._planned_source_evidence
            if (evidence is None or evidence[0].manifest['source_time_s'] != float(source.time[0])
                    or evidence[0].manifest['source_q_position'] != source.q[0].detach().cpu().tolist()):
                bundle = snapshot_source_evidence(source, getattr(self.owner, 'source_evidence_store', None))
                reference = save_source_evidence(self.output, f'source-failed-policy-{stamp}', bundle)
            else:
                bundle, reference = evidence
            attempt['source_visual_evidence'] = deepcopy(reference)
            forecast = getattr(bridge, 'last_forecast', None)
            commands = (None if forecast is None else
                forecast.trace.positions[0].detach().cpu().tolist())
            path = self.output/f'failed-policy-input-{stamp}.pt'
            possibly_written = (None if commands is None else
                deepcopy(commands[:attempt['policy_commands_attempted']]))
            returned = deepcopy(attempt.get('commands_position'))
            actual_readbacks = deepcopy(attempt.get('measured_q_path'))
            torch.save(dict(kind='failed-dispatch-input', schema=2, version=VERSION, contract=CONTRACT,
                model_output=str(self.root.resolve()), state=cpu_actual_state(source),
                text_ids=self.owner.runtime.text_ids.detach().cpu().clone(),
                object_target_source=bundle.manifest['object_token']['source'],
                object_token_provenance=deepcopy(bundle.manifest['object_token']),
                source_visual_evidence=deepcopy(reference),
                dispatch_source_time_s=attempt['source_time_s'],
                dispatch_source_q_position=deepcopy(attempt['source_q_position']),
                commands_position=commands,
                commands_scope='original_complete_plan_not_complete_execution',
                requested_commands_position=deepcopy(commands),
                requested_commands_scope='full_model_plan_requested_before_dispatch',
                possibly_written_commands_position=possibly_written,
                possibly_written_commands_scope='write_q_input_prefix_may_have_reached_hardware_no_position_proof',
                # Kept as a compatibility alias for existing offline readers.
                attempted_commands_position=deepcopy(possibly_written),
                attempted_commands_scope='requested_targets_not_proof_of_applied_motion',
                returned_commands_position=returned,
                returned_commands_scope='write_q_returned_targets_not_actual_position_readbacks',
                actual_readback_q_path=actual_readbacks,
                actual_readback_q_path_scope='real_position_readbacks_only_no_interpolation_or_goal_substitution',
                actual_readback_times_s=deepcopy(attempt.get('readback_times_s')),
                actual_readback_scope='read_q_samples_only; does_not_fill_unobserved_post_write_state',
                policy_commands_attempted=attempt['policy_commands_attempted'],
                policy_commands_returned=attempt['policy_commands_returned'],
                command_offsets_s=deepcopy(attempt.get('commanded_offsets_s')),
                action_source=attempt['action_source'],
                lewm_cem=deepcopy(attempt.get('lewm_cem')), cem_trace=deepcopy(attempt.get('cem_trace')),
                lewm_controller=deepcopy(self.lewm_controller), temporal_contract=deepcopy(self.temporal_contract),
                temporal_provenance=self._temporal_provenance(),
                executable=False, training_sample=False, model_feedback_label=False,
                transaction_completed=attempt.get('completed') is True,
                task_success_inferred=False), path)
            attempt['policy_input_snapshot'] = path.name
            attempt['temporal_provenance'] = self._temporal_provenance()
            receipt['saved'] = True
            try:
                write_json(self.output/receipt['path'], attempt)
            except BaseException:
                receipt['saved'] = False
                raise
        except Exception as exc:
            # Keep the physical failure primary, even when evidence storage fails.
            attempt['failure_evidence_recording_error'] = f'{type(exc).__name__}: {exc}'

    def _save_execution(self,result,*,require_policy_input=False):
        if (result.get('execution_record') or {}).get('saved') is True:
            return
        self._save_lewm_shadow_attempt()
        stamp=time.time_ns()
        record=dict(path=f'execution-{stamp}.json',saved=False,
                    completed_means='five_command_transaction_not_task_success',
                    task_success_inferred=False)
        result['execution_record']=record
        controller_record = ({} if self.lewm_controller is None else
            dict(lewm_controller=deepcopy(self.lewm_controller)))
        controller_record['temporal_provenance'] = self._temporal_provenance()
        if self.temporal_contract is not None:
            controller_record['temporal_contract'] = deepcopy(self.temporal_contract)
        result.update(controller_record)
        # Keep the exact state that generated the dispatched Actor command.
        # The owner may recapture after HOLD and overwrite last_object_relative
        # with a zero/unavailable display cue before this record is written.
        dispatch_source = getattr(self.owner.bridge, 'last_actor_input', None)
        planned_object_relative = self._planned_object_relative_snapshot
        planned_vector = (None if planned_object_relative is None
                          else planned_object_relative.get('object_relative'))
        source_normalized = self._coerce_object_relative(
            None if dispatch_source is None else getattr(dispatch_source, 'object_relative', None))
        same_source_object = bool(
            dispatch_source is not None and planned_object_relative is not None
            and planned_object_relative.get('_source_object_id') == id(dispatch_source))
        if planned_object_relative is not None and (
                same_source_object or planned_vector is None or source_normalized['vector'] is None
                or np.array_equal(planned_vector, source_normalized['vector'])):
            # Keep the detached plan copy even if a later state refresh mutates
            # the source object in place. Refresh identity clocks/q when the
            # bridge has a distinct dispatch source (for a legal rebase).
            dispatch_object_relative = deepcopy(planned_object_relative)
            if dispatch_source is not None:
                dispatched_identity = self._object_relative_snapshot(
                    dispatch_source,
                    provenance=planned_object_relative.get('object_token_provenance'))
                for key in ('object_relative_source_time_s', 'object_relative_source_q_position'):
                    dispatch_object_relative[key] = dispatched_identity[key]
                dispatch_object_relative['object_token_provenance_matches_dispatch_vector'] = True
        else:
            dispatch_object_relative = self._object_relative_snapshot(
                dispatch_source,
                provenance=None)
            dispatch_object_relative['object_token_provenance_matches_dispatch_vector'] = bool(
                planned_object_relative is None)
        if result.get('action_source') == 'v21_lewm_cem':
            result['lewm_cem'] = deepcopy(self.lewm_cem_record)
            if self.lewm_cem_trace_record is not None:
                result['cem_trace'] = deepcopy(self.lewm_cem_trace_record)
        if self.c4 is not None and result.get('action_source') == self.c4.source:
            result[getattr(self.c4, 'record_key', 'c4_planner')] = deepcopy(self.c4.record)
        if (self.record_policy_inputs or require_policy_input) and result.get('action_source') in self._RECORDABLE_POLICY_INPUTS:
            source=dispatch_source
            if source is None or not result.get('completed') or not result.get('protective_hold_after_q5'):
                raise RuntimeError('policy input recording requires a completed and held transaction')
            from .v21_source_evidence import snapshot_source_evidence, save_source_evidence
            evidence = self._planned_source_evidence
            if (evidence is None or evidence[0].manifest['source_time_s'] != float(source.time[0])
                    or evidence[0].manifest['source_q_position'] != source.q[0].detach().cpu().tolist()):
                bundle = snapshot_source_evidence(source, getattr(self.owner,'source_evidence_store',None))
                reference = save_source_evidence(self.output, f'source-policy-{stamp}', bundle)
            else:
                bundle, reference = evidence
            result['source_visual_evidence'] = deepcopy(reference)
            path=self.output/f'policy-input-{stamp}.pt'
            torch.save(dict(kind='actual-actor-input',schema=2,version=VERSION,contract=CONTRACT,
                model_output=str(self.root.resolve()),state=cpu_actual_state(source),
                object_target_source=bundle.manifest['object_token']['source'],
                object_token_provenance=deepcopy(bundle.manifest['object_token']),
                source_visual_evidence=deepcopy(reference),
                dispatch_source_time_s=result['source_time_s'],
                dispatch_source_q_position=result['source_q_position'],
                text_ids=self.owner.runtime.text_ids.detach().cpu().clone(),
                commands_position=result['commands_position'],command_offsets_s=result['commanded_offsets_s'],
                action_source=result['action_source'],
                lewm_cem=deepcopy(result.get('lewm_cem')),cem_trace=deepcopy(result.get('cem_trace')),
                executable=False,**controller_record),path)
            result['policy_input_snapshot']=path.name
        state=self.owner.runtime.state
        current_index = getattr(state.memory, 'current_readback_index', None)
        if current_index is not None:
            result['readback_progress'] = dict(unit='completed_policy_readback_sample_index',
                completed_index=current_index.detach().cpu().tolist(),
                session_id=getattr(state,'progress_session_id',None),
                origin=getattr(state,'progress_origin',None),
                real_timestamps_preserved=True)
        residual=state.q_error.detach().cpu().numpy()[0]
        path=np.asarray(result['measured_q_path'])
        if 'feedback_q_path' in result:
            feedback_path=np.asarray(result['feedback_q_path'])
            result['feedback_predicted_q_path']=np.concatenate((feedback_path[:1],feedback_path[1:]-residual)).tolist()
            result['feedback_q_path_residual']=residual.tolist()
        else:
            result['retimed_predicted_q_path']=np.concatenate((path[:1],path[1:]-residual)).tolist()
            result['q_path_residual']=residual.tolist()
        # Structured I/O for post-trial dump (observability): one queryable object per block.
        try:
            result['block_io']=self._block_io_snapshot(
                object_relative_snapshot=dispatch_object_relative)
        except Exception as exc:
            result['block_io']=dict(event='v21_block_io', error=f'{type(exc).__name__}: {exc}')
        # Commit the evidence before optional dashboard/image publication. A
        # later presentation error must not cause the same block to be saved twice.
        self.owner.stop_event.set()
        self._commit_fish_stage(result)
        record['saved']=True
        try:
            write_json(self.output/record['path'],result)
        except BaseException:
            record['saved']=False
            raise
        self.last_execution=deepcopy(result)
        keep=('source_time_s','readback_times_s','command_times_s','measured_q_path','retimed_predicted_q_path',
              'commands_position','action_source','completed','response_verified','response_error',
              'original_target_arrival','original_target_reached','normal_follow',
              'actual_cem_response_diagnostic',
              'execution_record','block_io','lewm_cem','lewm_controller')
        self.execution_history=(self.execution_history+[{k:deepcopy(result[k]) for k in keep if k in result}])[-12:]
        self._refresh_status();self._publish()
        self.owner.images(f'execution-{stamp}')

    def _save_rejected_complete_response(self,attempt):
        """After rejection HOLD, retain a full factual block even when response failed."""
        if (attempt is None or attempt.get('completed') is not True
                or attempt.get('policy_commands_attempted') != 5
                or attempt.get('policy_commands_returned') != 5
                or attempt.get('protective_hold_after_q5') is not True
                or attempt.get('response_verified') is not False
                or not attempt.get('response_error')
                or (attempt.get('execution_record') or {}).get('saved') is True):
            return
        try:
            for key,shape in (('commands_position',(5,6)),('measured_q_path',(6,6)),
                              ('command_times_s',(5,)),('readback_times_s',(6,))):
                values=np.asarray(attempt[key],dtype=np.float64)
                if values.shape!=shape or not np.isfinite(values).all():
                    raise ValueError('rejected response lacks a complete actual '+key)
            source=self.owner.bridge.last_source_state
            if (source is None
                    or not np.array_equal(source.q[0].detach().cpu().numpy(),attempt['source_q_position'])
                    or not np.array_equal(attempt['source_q_position'],attempt['measured_q_path'][0])
                    or float(source.time.item())!=float(attempt['source_time_s'])):
                raise ValueError('rejected response differs from this dispatch source')
            self._save_execution(attempt,require_policy_input=True)
        except Exception as exc:
            # Preserve the physical/service rejection as the exception being raised.
            attempt['execution_recording_error']=f'{type(exc).__name__}: {exc}'

    def _alignment_measure(self):
        endpoint=self.owner.last_endpoint
        if self.snapshot is None or self.snapshot['time']!=endpoint.time.item():self._publish()
        if self._wrist_fit is None:
            data=json.loads(self.wrist_reference.read_text(encoding='utf-8'))
            raw=self.owner.frames['wrist']
            if (data.get('geometry_protocol')!='letterbox_v1'
                    or data.get('content_rect')!=[0,28,224,196]
                    or (data.get('source_height'),data.get('source_width'))!=raw.shape[:2]):
                raise ValueError('Wrist reference and actual camera geometry do not match')
            row=data['fits']['wrist']
            self._wrist_fit=ProjectionFit('wrist',tuple(row['feature_names']),
                np.asarray(row['coefficients'],np.float64),float(row['ridge']))
        if not bool(endpoint.observation.measurement_available[0,0]):
            raise ValueError('a new Wrist frame is required for alignment')
        capture=next(v for v in self.owner.acquisitions[-1]['views'] if v['view']=='wrist')
        q=endpoint.observation.q_at_image[0,0].detach().cpu().numpy()
        world,local=self.owner.kinematics.forward_anchor_frames(q[None])
        fm=predict_projection(self._wrist_fit,world,local,np.asarray([q[-1]]))[0,1:3]
        # Collect teacher response against the SAME target fed to the Actor,
        # not a second independently localized DINO peak.
        enc=getattr(self.owner,'object_relative',None)
        record=getattr(enc,'last_policy_record',None)
        cue=getattr(self.owner,'last_object_relative',None)
        if not isinstance(record,dict) or cue is None or cue[0] <= 0:
            raise ValueError('current Actor fish-body target is required for alignment')
        record=dict(record,role='pickup',view='wrist',visible=True,
                    view_source='current_observation',
                    image_time_s=float(endpoint.observation.image_times[0,0]))
        return alignment_observation(record,q,fm,
            image_time_s=float(endpoint.observation.image_times[0,0]),q_time_s=capture['q_sample_s'],
            now_s=self.clock(),max_age_s=.5)

    GROUNDING_DEADBAND_PX=12.; GROUNDING_MAX_J1=3.; GROUNDING_STEP_FRACTION=.6
    GROUNDING_MAX_VETOES=8; GROUNDING_OPEN_Q6=20.
    # Close decision from grounding, not from the object's apparent size: once the jaws
    # straddle the object the tracker sees less of it, so a length threshold blocks exactly
    # the right moment. Centred object + jaw tips at table height => close in place.
    GROUNDING_CENTRED_PX=28.; GROUNDING_OFF_CENTRE_PX=40.; GROUNDING_CLOSE_CLEARANCE_MM=15.; GROUNDING_CLOSED_Q6=5.
    GROUNDING_CLOSE_MIN_CLEARANCE_MM=3.   # the moving jaw tip sweeps an arc while closing; below this it dips under the floor
    # Grounded descent: with the object centred and the jaws open but still above the table,
    # lower the shoulder (J2 only, slope from the FK clearance) toward the target clearance.
    # A close proposed while the tips are far above the table is a close on air: vetoed.
    GROUNDING_TARGET_CLEARANCE_MM=8.; GROUNDING_DESCEND_MAX_J2=4.; GROUNDING_AIR_CLOSE_MM=25.
    # Depth (along the reach) from dy: the object must sit between the tips, not ahead of
    # them. One 2x2 numerical Jacobian of the tip centre's (radial distance, height) w.r.t.
    # J2/J3 turns the wanted (reach, descent) into a joint step; per-transaction clipped.
    GROUNDING_DEPTH_PX=20.; GROUNDING_REACH_MM_PER_PX=.3; GROUNDING_REACH_MAX_MM=15.; GROUNDING_MAX_J23=4.
    GROUNDING_APPROACH_BELOW_MM=60.   # the Actor owns the descent from home; grounding takes the last centimetres
    # Reach envelope of the demonstrated grasps: beyond it the arm is stretching along the
    # table toward an object it cannot reach; stop extending and say so.
    GROUNDING_REACH_J2_MAX=45.; GROUNDING_REACH_J3_MIN=0.
    SEARCH_MAX_BLOCKS=12; GROUNDING_CENTRE_FIRST_PX=40.; GROUNDING_MAX_J1_FAR=6.
    # Jaw line perpendicular to the object's long axis before closing: wrist roll (J5) from the
    # axis angle in the eight-vector; degrees per J5 unit estimated online from the first step.
    GROUNDING_ROLL_TOL_DEG=20.; GROUNDING_ROLL_MAX_STEP=6.; GROUNDING_ROLL_PROBE=4.
    # This rig was replugged and the operator explicitly authorized the
    # horizontal-fish orientation needed here.  The target needs about -29;
    # keep a finite margin from the earlier cable-disconnect pose at -11.
    GROUNDING_ROLL_J5_MIN=-95.; GROUNDING_ROLL_J5_MAX=-25.
    GROUNDING_ROLL_ENABLED=False   # operator-requested direct depth experiment; hardware limits remain authoritative
    GROUNDING_UNFOLDED_J2=-60.        # the folded rest pose also has the tips near the table: never 'grasp' there
    CARRY_LIFT_CLEARANCE_MM=90.; CARRY_LIFT_J2_STEP=8.; CARRY_X_TOL_PX=30.; CARRY_Y_TOL_PX=25.
    CARRY_MAX_STEP=5.; CARRY_PROBE_J1=3.; CARRY_PROBE_J3=-3.; CARRY_OPEN_Q6=32.; CARRY_MAX_BLOCKS=40
    # Carry pose (J2,J3,J4) from the demonstrations' mid-carry frames: arm up, forearm level.
    CARRY_POSE_J234=(-45.,60.,45.); CARRY_POSE_STEP=10.; CARRY_POSE_TOL=6.; CARRY_LIFT_MAX_BLOCKS=30
    CARRY_TARGET_ABOVE_PX=20.; CARRY_PLACE_TOL_PX=30.; CARRY_JOINT_STEP=8.
    LEWM_RELEASE_ARM_RMS_MAX=3.; LEWM_RELEASE_ARM_MAX=5.
    # Reach the drop zone before letting go: aligned within tolerance (measured, not solved),
    # then lower to just above the box rim, then open. Occlusion alone never releases.
    CARRY_LOWER_CLEARANCE_MM=75.; CARRY_LOWER_STEP_MM=15.; CARRY_REALIGN_PX=45.; CARRY_OCCLUDED_OK_PX=40.

    def _grounding_builder(self,state,command_offsets):
        rt=self.owner.runtime
        action=rt.core.act(state,rt.text_ids,command_offsets).commands_position.clone()
        # The bridge captured a new endpoint. Engineering image diagnostics must
        # refer to that endpoint, not the preceding UI refresh or transaction.
        self._publish()
        v=assistance_cue(self.owner)
        q_now=state.q[0]
        record=dict(available=bool(v is not None and v[0]>0),dx_px=None,len_px=None,j1_delta=0.,gripper_veto=False,
                    px_per_unit=self.grounding_px_per_unit,actor_gripper_end=float(action[0,-1,5]))
        carried=self._carry_builder(action,q_now,record)
        if carried is not None:
            record['commands_position']=carried[0].detach().cpu().tolist(); self.grounding_record=record
            return carried
        if not record['available'] and float(q_now[5])>self.GROUNDING_OPEN_Q6 and float(q_now[1])>self.GROUNDING_UNFOLDED_J2:
            # Wrist cannot see the object: steer J1 from the column view (object centroid vs the
            # visual fingertips). The fingertip estimate jitters by several px, so the direction
            # flips only on a majority of three growing-error observations (15 px hysteresis).
            # Only J1 is overridden: the Actor keeps its own descent. Bounded search budget.
            targets=(self.targets.get('targets') or {}); obj=(targets.get('pickup') or {}).get('column') or {}
            column_tip=(self.fingertips.get('column') or {})
            tips=(column_tip.get('fm_px')
                  if tip_source_matches_view('column', column_tip) else None); sr=self._search
            if obj.get('visible') and obj.get('centroid_px') is not None and tips is not None and sr['blocks']<self.SEARCH_MAX_BLOCKS:
                ex=float(np.asarray(obj['centroid_px'],np.float64)[0]-np.asarray(tips,np.float64).reshape(-1,2).mean(0)[0])
                if sr['prev'] is not None:
                    sr['votes'].append(abs(ex)>abs(sr['prev'])+15.); sr['votes']=sr['votes'][-3:]
                    if len(sr['votes'])==3 and sum(sr['votes'])>=2: sr['dir']=-sr['dir']; sr['votes']=[]
                sr['prev']=ex; sr['blocks']+=1
                if abs(ex)>15.:
                    ramp=torch.linspace(.2,1.,5,device=action.device,dtype=action.dtype)
                    action[0,:,0]=q_now[0]+sr['dir']*5.*ramp          # J1 only; J2..J6 stay the Actor's plan
                    record['column_search']=dict(ex_px=ex,dir=sr['dir'],blocks=sr['blocks'])
                    record['commands_position']=action[0].detach().cpu().tolist(); self.grounding_record=record
                    return action
            elif sr['blocks']>=self.SEARCH_MAX_BLOCKS: record['column_search']=dict(exhausted=True,blocks=sr['blocks'])
            else: sr['prev']=None
        elif record['available']: self._search.update(prev=None,votes=[],blocks=0)
        if record['available']:
            dx_px=float(v[2])*224.; len_px=float(v[4])*224.
            record.update(dx_px=dx_px,len_px=len_px,score=float(v[1]))
            if (self.grounding_px_per_unit and float(q_now[5])>self.GROUNDING_OPEN_Q6 and abs(dx_px)>self.GROUNDING_DEADBAND_PX):
                delta=-dx_px/self.grounding_px_per_unit*self.GROUNDING_STEP_FRACTION
                cap=self.GROUNDING_MAX_J1_FAR if abs(dx_px)>self.GROUNDING_CENTRE_FIRST_PX else self.GROUNDING_MAX_J1
                delta=max(-cap,min(cap,delta))
                ramp=torch.linspace(.2,1.,5,device=action.device,dtype=action.dtype)
                action[0,:,0]=action[0,:,0]+delta*ramp; record['j1_delta']=float(delta)
            closing=bool((action[0,:,5]<float(q_now[5])-3.).any())
            clearance=None
            try:
                clearance=float(self.owner.clearance.clearance_mm(q_now.detach().cpu().numpy()))
                if not np.isfinite(clearance):raise ValueError('nonfinite clearance')
            except Exception as exc:
                record['clearance_error']=f'{type(exc).__name__}: {exc}'
                self.grounding_record=record
                raise RuntimeError('assisted approach requires finite clearance; no close or movement authorized') from exc
            record.update(clearance_mm=clearance,grounding_close=False)
            centred=abs(dx_px)<=self.GROUNDING_CENTRED_PX
            in_air=clearance is not None and clearance>self.GROUNDING_AIR_CLOSE_MM
            if closing and (abs(dx_px)>self.GROUNDING_OFF_CENTRE_PX or in_air):
                if self._grounding_vetoes>=self.GROUNDING_MAX_VETOES:
                    record.update(gripper_veto=True,veto_reason='alignment_budget_exhausted')
                    self.grounding_record=record
                    raise RuntimeError('assisted alignment budget exhausted; still misaligned, closure is not authorized')
                action[0,:,5]=q_now[5]; record['gripper_veto']=True; record['veto_reason']='in_air' if in_air else 'off_centre'
                self._grounding_vetoes+=1; closing=False
            elif not closing: self._grounding_vetoes=0
            record['grounding_descend_j2']=0.
            if not (float(q_now[5])>self.GROUNDING_OPEN_Q6 and clearance is not None and clearance<=self.GROUNDING_APPROACH_BELOW_MM):
                self._grounding_prev_end=None; self._grounding_bias=np.zeros(2)
            dy_px=float(v[3])*224.; record['dy_px']=dy_px
            centred=centred and abs(dy_px)<=self.GROUNDING_DEPTH_PX
            # axis angle vs the jaw line, in (-90, 90]; the closest perpendicular is +-90
            theta=float(np.degrees(np.arctan2(float(v[7]),float(v[6])))/2.) if (abs(float(v[6]))+abs(float(v[7])))>1e-6 else None
            roll_err=None
            # A change caused by J1/J2/J3 must not be learned as a J5 response.
            # Only the preceding, completed rotation-in-place is identifiable.
            if theta is not None and self._roll_prev is not None and len(self._roll_prev)==3:
                prior_q=np.asarray(self._roll_prev[2],np.float64)
                actual_q=q_now.detach().cpu().numpy()
                fixed=np.array([0,1,2,3,5])
                dj5=float(q_now[4])-self._roll_prev[0]
                dth=(theta-self._roll_prev[1]+90.)%180.-90.
                if np.max(np.abs(actual_q[fixed]-prior_q[fixed]))<=.25 and abs(dj5)>1. and abs(dth)<80.:
                    g=dth/dj5
                    self._roll_gain=g if self._roll_gain is None else .5*self._roll_gain+.5*g
            self._roll_prev=None
            if theta is not None:
                # two equivalent perpendicular targets; keep J5 inside the demonstrated range
                gain=self._roll_gain if (self._roll_gain is not None and abs(self._roll_gain)>.05) else 1.7
                best=None
                for e in (90.-theta,-90.-theta):
                    j5=max(self.GROUNDING_ROLL_J5_MIN,min(self.GROUNDING_ROLL_J5_MAX,float(q_now[4])+e/gain))
                    achievable=(j5-float(q_now[4]))*gain; residual=e-achievable
                    if best is None or abs(residual)<abs(best[1]): best=(achievable,residual)
                roll_err=best[0]; record.update(roll_residual_deg=best[1],roll_blocked_by_range=abs(best[1])>1.)
            record.update(axis_theta_deg=theta,roll_err_deg=roll_err)
            record['roll_gain_deg_per_unit']=self._roll_gain
            # aligned enough when the remaining rotation is small AND the unreachable residual is modest
            roll_ok=(not self.GROUNDING_ROLL_ENABLED) or roll_err is None or (abs(roll_err)<=self.GROUNDING_ROLL_TOL_DEG and abs(record.get('roll_residual_deg',0.))<=30.)
            def roll_step():
                if (not self.GROUNDING_ROLL_ENABLED) or roll_err is None or abs(roll_err)<=self.GROUNDING_ROLL_TOL_DEG*.5: return 0.
                step=self.GROUNDING_ROLL_PROBE if (self._roll_gain is None or abs(self._roll_gain)<.05) else roll_err/self._roll_gain
                return float(max(-self.GROUNDING_ROLL_MAX_STEP,min(self.GROUNDING_ROLL_MAX_STEP,step)))
            unfolded=float(q_now[1])>self.GROUNDING_UNFOLDED_J2; record['unfolded']=unfolded
            if (unfolded and centred and roll_ok and float(q_now[5])>self.GROUNDING_OPEN_Q6 and clearance is not None
                    and self.GROUNDING_CLOSE_MIN_CLEARANCE_MM<=clearance<=self.GROUNDING_CLOSE_CLEARANCE_MM):
                # Jaws around the centred object at table height: shut them in place, whatever the
                # Actor proposes (its simultaneous wrist motion would push the tips below the table).
                ramp=torch.linspace(.2,1.,5,device=action.device,dtype=action.dtype)
                action[0,:,:5]=q_now[:5]
                action[0,:,5]=q_now[5]+(self.GROUNDING_CLOSED_Q6-float(q_now[5]))*ramp
                record['grounding_close']=True
                self._grounding_prev_end=None; self._grounding_bias=np.zeros(2)
                self._carry=dict(phase='armed',blocks=0,lifted=0,k1=None,k3=None,prev=None)
            elif (unfolded and float(q_now[5])>self.GROUNDING_OPEN_Q6 and clearance is not None
                    and clearance<=self.GROUNDING_APPROACH_BELOW_MM):
                # Grounded approach: descend toward the target clearance and move along the reach
                # until the object sits between the tips; J1 recentring from above is kept.
                # Regulate height toward the target in both directions: reaching along the table
                # drifts the tips down, and a close needs a few millimetres of clearance.
                dz=max(-12.,min(12.,self.GROUNDING_TARGET_CLEARANCE_MM-clearance))
                if abs(dz)<2.: dz=0.
                if abs(dx_px)>self.GROUNDING_CENTRE_FIRST_PX and dz<0.: dz=0.; record['centre_first']=True   # centre before descending
                # In the live wrist convention, negative C-minus-M depth means
                # the fish is forward of the jaw midpoint.  The measured
                # approach response shows it needs the opposite radial sign.
                dr=0. if abs(dy_px)<=self.GROUNDING_DEPTH_PX else max(-self.GROUNDING_REACH_MAX_MM,min(self.GROUNDING_REACH_MAX_MM,dy_px*self.GROUNDING_REACH_MM_PER_PX))
                q_np=q_now.detach().cpu().numpy()
                if self._grounding_prev_end is not None:
                    self._grounding_bias=np.clip(self._grounding_bias+(self._grounding_prev_end-q_np[1:3]),-8.,8.)
                rs=roll_step()
                if abs(rs)>0. and clearance>self.GROUNDING_CLOSE_MIN_CLEARANCE_MM:
                    j5=float(max(self.GROUNDING_ROLL_J5_MIN,min(self.GROUNDING_ROLL_J5_MAX,float(q_now[4])+rs)))
                    if abs(j5-float(q_now[4]))>.05:
                        # The J23 solver assumes fixed J5. Rotate, reobserve,
                        # then solve translation at the realized roll angle.
                        # Never apply sag/J1/descent to this isolated rotation.
                        held=q_now[None,:].repeat(5,1).clone()
                        ramp=torch.linspace(.2,1.,5,device=action.device,dtype=action.dtype)
                        held[:,4]=q_now[4]+(j5-float(q_now[4]))*ramp
                        self._roll_prev=(float(q_now[4]),theta,q_np.copy())
                        self._grounding_prev_end=None
                        action[0]=held
                        record.update(stage='roll_then_reobserve',roll_step=j5-float(q_now[4]),
                                      approach_deferred=True,commands_position=held.detach().cpu().tolist())
                        self.grounding_record=record
                        return action  # unchanged bridge audits the entire physical path
                step=self._approach_step(q_np,dz,dr)
                record.update(approach_dz_mm=dz,approach_dr_mm=dr,approach_step=None if step is None else [float(x) for x in step])
                if step is not None:
                    j2_next=float(q_now[1])+float(step[0])+float(self._grounding_bias[0])
                    j3_next=float(q_now[2])+float(step[1])+float(self._grounding_bias[1])
                    extending=(j2_next>self.GROUNDING_REACH_J2_MAX and j2_next>float(q_now[1])) or (j3_next<self.GROUNDING_REACH_J3_MIN and j3_next<float(q_now[2]))
                    if extending:
                        # A zero step was not a hold: sag bias and J1/J5 were
                        # still appended below. Reject before any goal write.
                        record.update(unreachable=True,stage='approach_boundary',
                                      reason='outside_existing_assisted_reach_envelope')
                        self.grounding_record=record
                        raise RuntimeError('assisted reach boundary: no command sent; reset the approach')
                if step is not None:
                    ramp=torch.linspace(.2,1.,5,device=action.device,dtype=action.dtype)
                    held=q_now[None,:].repeat(5,1).clone(); held[:,0]=action[0,:,0]
                    held[:,1]=q_now[1]+float(step[0])*ramp+float(self._grounding_bias[0])
                    held[:,2]=q_now[2]+float(step[1])*ramp+float(self._grounding_bias[1])
                    action[0]=held; record['grounding_descend_j2']=float(step[0]); record['sag_bias']=[float(x) for x in self._grounding_bias]
                    self._grounding_prev_end=np.array([float(q_now[1])+float(step[0]),float(q_now[2])+float(step[1])])
                else: self._grounding_prev_end=None
        import scripts.serve_so101_prod_v17_7 as established
        lower=torch.as_tensor(established.SO101_POSITION_LOWER,device=action.device,dtype=action.dtype)
        upper=torch.as_tensor(established.SO101_POSITION_UPPER,device=action.device,dtype=action.dtype)
        action=torch.maximum(torch.minimum(action,upper),lower)
        record['commands_position']=action[0].detach().cpu().tolist()
        self.grounding_record=record
        return action

    _grounding_builder.wants_command_offsets=True

    def _carry_builder(self,action,q_now,record):
        """Carry after a grounded close: lift a fixed amount, then align the held object
        (its own COLUMN centroid -- it sits in the jaws, so it marks the gripper) with the
        registered drop zone by a pixel loop whose gains are estimated online, then open.
        Owns the arm while the object is held; hands back to the Actor after the release."""
        q6=float(q_now[5]); v=assistance_cue(self.owner)
        # Grasp check from the wrist view: a held object stays large and between the jaws.
        holding_view=bool(v is not None and v[0]>0 and float(v[4])*224.>80. and abs(float(v[2]))*224.<45. and abs(float(v[3]))*224.<60.)
        c=self._carry
        if c is None:
            over=abs(float(q_now[0])+65.)<=12.
            carry_pose=np.asarray(self.CARRY_POSE_J234,np.float64)
            already_lifted=bool(np.max(np.abs(q_now[1:4].detach().cpu().numpy()-carry_pose))<=self.CARRY_POSE_TOL)
            # Resume carry after a restart while the object is still in the wrist jaws,
            # even if the last release opened q6 (fish can sit on a serrated jaw).
            if holding_view and (q6<15. or over):
                # A restart may happen while a held object is already at the carry
                # pose.  Resume at image alignment, never at lower/release: an
                # unverified pose above the box is not release authorization.
                phase='align' if (already_lifted or over) else 'lift'
                c=self._carry=dict(phase=phase,blocks=0,lifted=0,k1=None,k3=None,prev=None,record_only=False,
                                   resumed_while_held=True)
            else: return None
        blocks=c.get('blocks')
        if type(blocks) is not int or not 0<=blocks<self.CARRY_MAX_BLOCKS:
            record['carry']=dict(phase=c.get('phase'),blocks=blocks,reason='carry_budget_exhausted_or_invalid')
            raise RuntimeError('assisted carry budget exhausted or invalid; no further movement or release authorized')
        c['blocks']+=1; record['carry']=dict(phase=c['phase'],blocks=c['blocks'])
        if c['phase']=='armed':
            if q6>self.GROUNDING_OPEN_Q6: self._carry=None; return None
            c['phase']='lift'
        if c['phase'] in ('lift','align') and q6>self.GROUNDING_OPEN_Q6+2:
            self._carry=None; record['carry']['dropped']=True; return None
        if c['phase'] in ('lift','align'):
            c['miss']=0 if holding_view else c.get('miss',0)+1; c['seen']=c.get('seen',0)+int(holding_view)
            record['carry'].update(holding_view=holding_view,miss=c['miss'])
            # Closed on air: within the first blocks after the close the object must show as held.
            if (c['miss']>=2 and (c['phase']=='lift' or c['seen']<3)) or c['miss']>=3: c['phase']='failed'   # lost at any point
        if c['phase']=='failed':
            record['carry']['grasp_failed']=True
            if q6>=self.GROUNDING_OPEN_Q6+5.: self._carry=None; return None       # open again: approach resumes
            held=q_now[None,:].repeat(5,1).clone(); ramp=torch.linspace(.2,1.,5,device=action.device,dtype=action.dtype)
            held[:,5]=q6+(self.CARRY_OPEN_Q6-q6)*ramp; action[0]=held; record['carry']['phase_now']='failed'; return action
        held=q_now[None,:].repeat(5,1).clone(); ramp=torch.linspace(.2,1.,5,device=action.device,dtype=action.dtype)
        # Keep squeezing while carrying: commanding the readback zeroes the servo error and
        # the grip force with it (the object then slips as soon as the arm swings).
        held[:,5]=min(q6,self.GROUNDING_CLOSED_Q6)
        if c['phase']=='lift':
            # Go to the carry pose with large-error steps: at full reach a small position
            # error gives the shoulder servo too little torque to lift arm plus object.
            target=np.asarray(self.CARRY_POSE_J234); cur=q_now[1:4].detach().cpu().numpy()
            delta=np.clip(target-cur,-self.CARRY_POSE_STEP,self.CARRY_POSE_STEP)
            if np.abs(target-cur).max()<=self.CARRY_POSE_TOL or c['lifted']>=self.CARRY_LIFT_MAX_BLOCKS: c['phase']='align'
            else:
                for j in range(3): held[:,1+j]=q_now[1+j]+float(delta[j])*ramp
                c['lifted']+=1; record['carry']['to_carry_pose']=[float(x) for x in delta]
        if c['phase']=='align':
            # Place with the column camera: the held object marks the tips, so (object px - FK
            # projected tips) is the current projection bias; with that bias removed, solve the
            # J1/J3 that put the tips just above the drop zone centroid at the carry height.
            targets=(self.targets.get('targets') or {})
            box=(targets.get('drop_zone') or {}).get('column') or {}; obj=(targets.get('pickup') or {}).get('column') or {}
            q_np=q_now.detach().cpu().numpy().astype(np.float64)
            proj_now=self._column_tip_px(q_np)
            if obj.get('visible') and obj.get('centroid_px') is not None and proj_now is not None:
                bias=np.asarray(obj['centroid_px'],np.float64)-proj_now
                c['bias']=bias if c.get('bias') is None else .5*c['bias']+.5*bias
            record['carry']['projection_bias_px']=None if c.get('bias') is None else [float(x) for x in c['bias']]
            if not box.get('visible') or box.get('centroid_px') is None:
                # The arm over the box hides it from the column camera: keep going to the last
                # solution found while it was visible, and release on arrival.
                # Hidden by the arm: hold still (the measured climb owns the alignment; the biased
                # projection solution must not drag the arm back). If the last measured error was
                # already small, being hidden means we are above the box: lower.
                if (c.get('last_err') is not None and c['last_err']<=self.CARRY_OCCLUDED_OK_PX) or (
                        abs(float(q_np[0])+65.)<=12. and not obj.get('visible')):
                    c['phase']='lower'
                else: record['carry']['waiting']='drop zone hidden; holding until it is visible again'
            elif c.get('bias') is None:
                if not obj.get('visible') and abs(float(q_np[0])+65.)<=12.:
                    c['phase']='lower'
                elif not obj.get('visible'):
                    d1=max(-self.CARRY_JOINT_STEP,min(self.CARRY_JOINT_STEP,-65.-float(q_np[0])))
                    held[:,0]=q_now[0]+float(d1)*ramp
                    record['carry']['waiting']='occluded; stepping J1 toward drop zone'
                else:
                    record['carry']['waiting']='held object not yet seen in the column view (no projection bias)'
            else:
                goal=np.asarray(box['centroid_px'],np.float64)+np.array([0.,-self.CARRY_TARGET_ABOVE_PX])
                servoed=False
                if obj.get('visible') and obj.get('centroid_px') is not None:
                    err=goal-np.asarray(obj['centroid_px'],np.float64); record['carry']['measured_err_px']=[float(x) for x in err]
                    c['last_err']=float(np.hypot(*err))
                    if abs(float(err[0]))<=self.CARRY_X_TOL_PX and abs(float(err[1]))<=self.CARRY_Y_TOL_PX+10.: c['phase']='lower'
                    elif c.get('resumed_while_held'):
                        # After a restart with a verified held object, use the
                        # current C-to-box error itself as the coordinate-descent
                        # objective.  Do not assume J1 is the only place axis:
                        # the camera/arm geometry changes across the carry pose.
                        axes=(0,2,1)  # pan, reach, height; gripper remains closed
                        c.setdefault('place_axis_index',0)
                        c.setdefault('place_dirs',{0:-1.,2:-1.,1:-1.})
                        c.setdefault('place_reversed',False)
                        current=float(np.hypot(*err)); previous=c.get('place_previous_error')
                        axis=axes[c['place_axis_index']]
                        if previous is not None and current>=previous-2.:
                            if not c['place_reversed']:
                                c['place_dirs'][axis]=-c['place_dirs'][axis]
                                c['place_reversed']=True
                            else:
                                c['place_axis_index']=(c['place_axis_index']+1)%len(axes)
                                axis=axes[c['place_axis_index']]
                                c['place_reversed']=False
                        step=3.
                        held[:,axis]=q_now[axis]+float(c['place_dirs'][axis])*step*ramp
                        c['place_previous_error']=current
                        record['carry']['climb']=dict(axis=('J1' if axis==0 else 'J3' if axis==2 else 'J2'),
                                                     dir=float(c['place_dirs'][axis]),size=step,
                                                     objective='column_fish_C_to_box_target',
                                                     error_px=[float(err[0]),float(err[1])],
                                                     resumed_coordinate_descent=True)
                        servoed=True
                    else:
                        # Measured error visible: fixed-step hill climb per axis (J1 for x, then J3
                        # for y). Robust to the biased column projection: a step that made the
                        # error grow flips the direction; no Jacobian, no gain estimate.
                        ex,ey=float(err[0]),float(err[1])
                        c.setdefault('dir1',1.); c.setdefault('dir3',1.)
                        if abs(ex)>self.CARRY_X_TOL_PX:
                            best=c.get('best_abs_ex')
                            c['x_stall']=0 if best is None or abs(ex)<best-2. else c.get('x_stall',0)+1
                            c['best_abs_ex']=min(abs(ex),best) if best is not None else abs(ex)
                            if c.get('prev_ex') is not None and abs(ex)>abs(c['prev_ex'])+2.:
                                c['dir1']=-c['dir1']
                            # After two non-improving horizontal steps, take one
                            # bounded depth-axis sample.  The next actual image
                            # decides its direction; no release can occur here.
                            if c['x_stall']>=2:
                                if c.get('prev_ey') is not None and abs(ey)>abs(c['prev_ey'])+2.:
                                    c['dir3']=-c['dir3']
                                c.setdefault('dir3',1.)
                                size=3.
                                held[:,2]=q_now[2]+c['dir3']*size*ramp
                                c['prev_ey']=ey; c['x_stall']=0
                                record['carry']['climb']=dict(axis='J3',dir=c['dir3'],size=size,
                                                             reason='horizontal_stall_depth_probe')
                            else:
                                size=3. if abs(ex)>30. else 1.
                                held[:,0]=q_now[0]+c['dir1']*size*ramp; c['prev_ex']=ex; c['prev_ey']=None
                                record['carry']['climb']=dict(axis='J1',dir=c['dir1'],size=size,
                                                             horizontal_stall=c['x_stall'])
                        else:
                            if c.get('prev_ey') is not None and abs(ey)>abs(c['prev_ey'])+6.: c['dir3']=-c['dir3']
                            size=5. if abs(ey)>40. else 3.
                            held[:,2]=q_now[2]+c['dir3']*size*ramp; c['prev_ey']=ey; c['prev_ex']=None
                            record['carry']['climb']=dict(axis='J3',dir=c['dir3'],size=size)
                        servoed=True
                if c['phase']=='align' and not servoed and not (obj.get('visible') and obj.get('centroid_px') is not None):
                    best=None
                    for j1 in np.linspace(-80.,80.,33):
                        for j3 in np.linspace(q_np[2]-40.,q_np[2]+40.,9):
                            cand=q_np.copy(); cand[0]=j1; cand[2]=j3
                            pr=self._column_tip_px(cand)
                            if pr is None: continue
                            e=float(np.hypot(*(pr+c['bias']-goal)))
                            if best is None or e<best[0]: best=(e,j1,j3)
                    if best is not None:
                        e,j1,j3=best; record['carry'].update(solved_j1=float(j1),solved_j3=float(j3),solved_residual_px=float(e)); c['solution']=(float(j1),float(j3))
                        d1=max(-self.CARRY_JOINT_STEP,min(self.CARRY_JOINT_STEP,j1-q_np[0])); d3=max(-self.CARRY_JOINT_STEP,min(self.CARRY_JOINT_STEP,j3-q_np[2]))
                        held[:,0]=q_now[0]+float(d1)*ramp; held[:,2]=q_now[2]+float(d3)*ramp
                        if abs(j1-q_np[0])<=2. and abs(j3-q_np[2])<=2. and not obj.get('visible'):
                            # last_err is only written when the column still sees the object.
                            # The solver's best joint is the place pose; once we have arrived,
                            # lower even if the biased residual is a few pixels over 40.
                            c['phase']='lower'
                    else: record['carry']['waiting']='no projection solution'
        if c['phase']=='lower':
            targets=(self.targets.get('targets') or {})
            box=(targets.get('drop_zone') or {}).get('column') or {}; obj=(targets.get('pickup') or {}).get('column') or {}
            if box.get('visible') and box.get('centroid_px') is not None and obj.get('visible') and obj.get('centroid_px') is not None:
                goal=np.asarray(box['centroid_px'],np.float64)+np.array([0.,-self.CARRY_TARGET_ABOVE_PX])
                e=float(np.hypot(*(goal-np.asarray(obj['centroid_px'],np.float64)))); record['carry']['lower_err_px']=e
                if e>self.CARRY_REALIGN_PX: c['phase']='align'; record['carry']['realign']=True
            if c['phase']=='lower':
                clearance=None
                try: clearance=float(self.owner.clearance.clearance_mm(q_now.detach().cpu().numpy()))
                except Exception: pass
                record['carry']['lower_clearance_mm']=clearance
                if clearance is None:
                    raise RuntimeError('carry clearance unavailable; release is not authorized')
                if clearance<=self.CARRY_LOWER_CLEARANCE_MM: c['phase']='release'
                else:
                    dz=-min(self.CARRY_LOWER_STEP_MM,clearance-self.CARRY_LOWER_CLEARANCE_MM)
                    step=self._approach_step(q_now.detach().cpu().numpy(),dz,0.)
                    if step is None:
                        raise RuntimeError('carry descent has no valid solution; release is not authorized')
                    else:
                        held[:,1]=q_now[1]+float(step[0])*ramp; held[:,2]=q_now[2]+float(step[1])*ramp; record['carry']['lower_step']=[float(x) for x in step]
        if c['phase']=='release':
            # Releasing is not an Actor timeout.  FCPG supplies the current
            # Column fish/box relation; LeWM must also have a fresh, bounded
            # actual joint-response residual for the preceding carried block.
            state=self.owner.runtime.state
            residual=state.q_error.detach().cpu().numpy()[0,:,:5]
            feedback=bool(state.feedback_available[0])
            rms=float(np.sqrt(np.mean(residual*residual)))
            maximum=float(np.max(np.abs(residual)))
            targets=(self.targets.get('targets') or {})
            box=(targets.get('drop_zone') or {}).get('column') or {}
            obj=(targets.get('pickup') or {}).get('column') or {}
            visual_error=None
            # Display targets may come from a restored historical Column image.
            # Visibility alone must never authorize opening over that old box.
            column_current=False
            try:
                observation=self.owner.last_endpoint.observation
                stamp=float(observation.image_times[0,1])
                age=float(self.clock())-stamp
                column_current=bool(observation.measurement_available[0,1]) and (
                    np.isfinite([stamp,age]).all()
                    and 0. <= age <= self.owner.runtime.max_plan_age_s
                    and all(target.get('view_source')=='current_observation'
                        and target.get('image_time_s')==stamp for target in (box,obj)))
            except (AttributeError,TypeError,ValueError,IndexError):
                pass
            if column_current and box.get('visible') and obj.get('visible') and box.get('centroid_px') is not None and obj.get('centroid_px') is not None:
                goal=np.asarray(box['centroid_px'],np.float64)+np.array([0.,-self.CARRY_TARGET_ABOVE_PX])
                visual_error=float(np.hypot(*(goal-np.asarray(obj['centroid_px'],np.float64))))
            release_ok=bool(feedback and rms<=self.LEWM_RELEASE_ARM_RMS_MAX and maximum<=self.LEWM_RELEASE_ARM_MAX
                            and visual_error is not None and visual_error<=self.CARRY_PLACE_TOL_PX)
            record['carry']['lewm_release_gate']=dict(feedback_available=feedback,arm_residual_rms=rms,
                arm_residual_max=maximum,visual_error_px=visual_error,accepted=release_ok,
                current_column_observation=bool(column_current),
                thresholds=dict(rms=self.LEWM_RELEASE_ARM_RMS_MAX,maximum=self.LEWM_RELEASE_ARM_MAX,
                                visual=self.CARRY_PLACE_TOL_PX))
            if not release_ok:
                # Keep squeezing.  A subsequent held block supplies the missing
                # actual LeWM feedback; a bad visual relation returns to align.
                if not column_current:
                    record['carry']['release_blocked']='column_observation_untrusted'
                elif visual_error is not None and visual_error>self.CARRY_PLACE_TOL_PX:
                    c['phase']='align'; record['carry']['release_blocked']='column_not_aligned'
                else:
                    record['carry']['release_blocked']='lewm_feedback_untrusted'
                action[0]=held; record['carry']['phase_now']=c['phase']; return action
            # Keep commanding the open until the readback shows it (an aborted transaction may
            # have opened the jaws only part way); only then hold still and hand back.
            if q6>=self.GROUNDING_OPEN_Q6+5.: c['phase']='done'; c['settle']=0
            else: held[:,5]=q6+(self.CARRY_OPEN_Q6-q6)*ramp; record['carry']['release']=True
        if c['phase']=='done':
            c['settle']=c.get('settle',0)+1; record['carry']['release']=True
            if c['settle']>2:
                self._carry=None; self._home_after_release=True; record['carry']['home_next']=True; return None
        action[0]=held; record['carry']['phase_now']=c['phase']
        return action

    def _column_tip_px(self,q_np):
        """FK tip centre projected into the COLUMN letterbox frame (biased; corrected online)."""
        try:
            if getattr(self,'_column_fit',None) is None:
                row=json.loads(WRIST_REFERENCE.read_text(encoding='utf-8'))['fits']['column']
                self._column_fit=ProjectionFit('column',tuple(row['feature_names']),np.asarray(row['coefficients'],np.float64),float(row['ridge']))
            world,local=self.owner.kinematics.forward_anchor_frames(np.asarray(q_np,np.float32)[None])
            return np.asarray(predict_projection(self._column_fit,world,local,np.asarray([q_np[-1]]))[0,0],np.float64)
        except Exception: return None

    def _approach_step(self,q_np,dz_mm,dr_mm):
        """(dJ2,dJ3) moving the tip centre by (dr radial, dz vertical) mm; numerical Jacobian, clipped."""
        if abs(dz_mm)<1e-6 and abs(dr_mm)<1e-6: return None
        kin=self.owner.kinematics
        def rz(q):
            c=np.asarray(kin.forward_anchors(np.asarray(q,np.float64)))[0]
            return np.array([1000.*float(np.hypot(c[0],c[1])),1000.*float(c[2])])
        base=rz(q_np); J=np.zeros((2,2))
        for i,j in enumerate((1,2)):
            probe=np.asarray(q_np,np.float64).copy(); probe[j]+=.5; J[:,i]=(rz(probe)-base)/.5
        if abs(np.linalg.det(J))<1e-6: return None
        d=np.linalg.solve(J,np.array([dr_mm,dz_mm]))
        return d*min(1.,self.GROUNDING_MAX_J23/max(float(np.abs(d).max()),1e-9))

    def _wrist_forward_step(self, q_np, dy_px):
        """Reach so wrist M closes on a centroid that is still in front of it.

        Negative dy is the fish forward of M. The measured approach uses that
        sign directly as the radial step. Height is unchanged here.
        """
        dy = float(dy_px)
        if abs(dy) <= self.GROUNDING_DEPTH_PX:
            return None
        dr = max(-self.GROUNDING_REACH_MAX_MM, min(
            self.GROUNDING_REACH_MAX_MM, dy * self.GROUNDING_REACH_MM_PER_PX))
        try:
            return self._approach_step(q_np, 0., dr)
        except Exception:
            return None

    def _jaw_white_angle_deg(self, q_np):
        pair = self._project_fg('wrist', q_np)
        if pair is None:
            return None
        jaw = np.asarray(pair[1] - pair[0], np.float64)
        if float(np.linalg.norm(jaw)) < 1e-6:
            return None
        return float(np.degrees(np.arctan2(jaw[1], jaw[0])))

    def _keep_above_floor(self, q_np, end, floor):
        """Scale the J2/J3 step so the endpoint stays at or above the table floor."""
        try:
            if float(self.owner.clearance.clearance_mm(end)) >= floor - 1e-6:
                return end
        except Exception:
            return end
        lo, hi, best = 0., 1., np.asarray(q_np, np.float64).copy()
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            trial = np.asarray(q_np, np.float64).copy()
            trial[1:3] = q_np[1:3] + mid * (end[1:3] - q_np[1:3])
            try:
                clearance = float(self.owner.clearance.clearance_mm(trial))
            except Exception:
                return end
            if clearance >= floor:
                best, lo = trial, mid
            else:
                hi = mid
        kept = np.asarray(end, np.float64).copy()
        kept[1:3] = best[1:3]
        return kept

    def _fish_wrist_geometry(self, q):
        enc = getattr(self.owner, 'object_relative', None)
        rec = getattr(enc, 'last_policy_record', None)
        if not isinstance(rec, dict) or rec.get('source') in ('estimated_fg', 'dino_fg_near_prior'):
            return None
        observation = self.owner.last_endpoint.observation
        stamp = float(observation.image_times[0, 0])
        cue = getattr(self.owner, 'last_object_relative', None)
        if (not bool(observation.measurement_available[0, 0])
                or not 0. <= float(self.clock()) - stamp <= self.owner.runtime.max_plan_age_s
                or cue is None or cue[0] <= 0):
            return None
        cog = np.asarray(rec.get('cog_px') or rec.get('centroid_px'), np.float64).reshape(-1)
        axis = np.asarray(rec.get('axis_px'), np.float64).reshape(-1)
        capture_q = observation.q_at_image[0, 0].detach().cpu().numpy()
        pair = self._project_fg('wrist', capture_q)
        if (cog.shape != (2,) or axis.shape != (2,) or pair is None
                or not np.isfinite(np.r_[cog, axis]).all() or np.linalg.norm(axis) < 1e-8):
            return None
        pair = np.asarray(pair, np.float64)
        jaw = pair[1] - pair[0]
        if not np.isfinite(pair).all() or np.linalg.norm(jaw) < 1e-8:
            return None
        blue = np.array([-axis[1], axis[0]]) / np.linalg.norm(axis)
        width = float(rec.get('width_px') or 16.)
        if not np.isfinite(width) or width <= 0:
            return None
        # Exactly the finite blue segment drawn by _draw_object_geometry.
        # Clamp to its endpoints: being near the infinite extension is not
        # being near the fish. Partial current colour support remains usable.
        half = max(14., .55 * max(width, 16.))
        mid = pair.mean(0)
        nearest = cog + np.clip(np.dot(mid - cog, blue), -half, half) * blue
        angle = float(np.degrees(np.arctan2(jaw[0]*blue[1]-jaw[1]*blue[0], np.dot(jaw, blue))))
        angle = (angle + 90.) % 180. - 90.
        delta = cog - pair.mean(0)
        return dict(dist_px=float(np.linalg.norm(delta)), angle_deg=angle,
                    dx_px=float(delta[0]), dy_px=float(delta[1]),
                    # Reuse the established carry-view criterion. Closing a
                    # soft jaw moves F/G: do not reapply the OPEN-jaw pinch
                    # alignment tolerance to an already held fish.
                    held_object_visible=bool(float(rec.get('length_px') or 0.) > 80.
                        and abs(float(delta[0])) < 45. and abs(float(delta[1])) < 60.),
                    blue_segment_distance_px=float(np.linalg.norm(mid - nearest)),
                    blue_segment_px=[(cog-half*blue).tolist(), (cog+half*blue).tolist()],
                    grasp_width_px=float(2.*half),
                    image_time_s=stamp)

    def _update_fish_drop_goal(self, targets):
        # The fixed Column camera sees a stationary box. Accumulate distinct
        # exposures before carrying; retain the majority location when the
        # arm/fish starts occluding the box instead of chasing its visible edge.
        history = self._fish_task.setdefault('drop_zone_observations', [])
        frozen = bool(self._fish_task.get('close_goal_issued') and history)
        box = ((targets.get('targets') or {}).get('drop_zone') or {}).get('column') or {}
        if not frozen and box.get('visible') is True and box.get('role') == 'drop_zone' and box.get('view_source') == 'current_observation':
            px = np.asarray(box.get('cog_px') or box.get('centroid_px'), np.float64).reshape(-1)
            stamp = box.get('image_time_s')
            if px.shape == (2,) and np.isfinite(px).all() and stamp is not None and np.isfinite(stamp):
                if not any(item['image_time_s'] == stamp for item in history):
                    history.append(dict(target_px=px.tolist(), image_time_s=float(stamp)))
                    del history[:-21]
        if history:
            self._fish_task['drop_goal'] = dict(
                source='dino_column_drop_zone_temporal_median',
                target_px=np.median([item['target_px'] for item in history], axis=0).tolist(),
                sample_count=len(history), frozen_for_carry=frozen,
                observation_times_s=[item['image_time_s'] for item in history],
                assumes_stationary_box_and_column=True)

    def _fish_stage_cost(self, state, visual):
        from .v21_fish_task_cost import FishTaskCost
        from .v21_visual_task_cost import ColumnDropVisualCost
        q = state.q[0].detach().cpu().numpy()
        gate, kin = self.owner.clearance, self.owner.kinematics
        release_mm = drop_visual = None
        if FISH_RELEASE_REFERENCE.exists():
            reference = json.loads(FISH_RELEASE_REFERENCE.read_text(encoding='utf-8'))
            if reference.get('source') == 'operator_confirmed_release_height':
                release_mm = float(gate.clearance_mm(np.asarray(reference['q_position'], np.float64)))
        goal = self._fish_task.get('drop_goal') or {}
        tip = (self.fingertips or {}).get('column') or {}
        observation = self.owner.last_endpoint.observation
        stamp = float(observation.image_times[0, 1])
        cog = np.asarray(goal.get('target_px'), np.float64).reshape(-1)
        fit = getattr(self.tip_display, 'column_fit', None)
        if (goal.get('sample_count', 0) > 0
                and bool(observation.measurement_available[0, 1])
                and 0 <= float(self.clock()) - stamp <= self.owner.runtime.max_plan_age_s
                and tip.get('available') is True and tip.get('image_time_s') == stamp
                and cog.shape == (2,) and np.isfinite(cog).all() and fit is not None):
            drop_visual = ColumnDropVisualCost(lambda value: self._project_fg('column', value),
                tip['fm_px'], observation.q_at_image[0, 1].detach().cpu().numpy(), cog,
                midpoint=lambda value: fk_midpoint_m(kin, value),
                z_pixel_per_m=fit.coefficients[fit.feature_names.index('z_base')],
                opening_z_m=gate.table_z_mm / 1000. + .10, image_time_s=stamp,
                target_memory=goal)
        return FishTaskCost(q, self._fish_task, visual=visual, wrist=self._fish_wrist_geometry(q),
            clearance=gate.clearance_mm, midpoint=lambda value: fk_midpoint_m(kin, value),
            floor_mm=gate.planned_clearance_mm,
            closed=established.DEFAULT_GRIPPER_CLOSED_POSITION, opened=established.DEFAULT_GRIPPER_OPEN_POSITION,
            release_mm=release_mm, drop_visual=drop_visual)

    def _record_fish_stage(self, state, selected, task):
        next_state = deepcopy(task.state)
        if task.phase == 'close':
            next_state['close_goal_issued'] = float(selected[0, -1, 5]) <= task.closed + 3.
        self._fish_task_proposal = dict(source_time_s=float(state.time[0]),
            source_q_position=state.q[0].detach().cpu().tolist(),
            commands_position=selected[0].detach().cpu().tolist(), next_state=next_state)

    def _commit_fish_stage(self, result):
        proposal = self._fish_task_proposal
        # Dispatch retimes a matching source to a fresh actual readback.
        # Compare the exact commands and source pose, not identical clocks.
        source_matches = False
        if proposal is not None:
            try:
                actual_q = np.asarray(result['measured_q_path'][0], np.float64)
                planned_q = np.asarray(proposal['source_q_position'], np.float64)
                age = float(result['source_time_s']) - proposal['source_time_s']
                source_matches = (actual_q.shape == planned_q.shape == (6,)
                    and np.isfinite(actual_q).all() and np.isfinite(planned_q).all()
                    and 0 <= age <= self.owner.runtime.max_plan_age_s
                    and np.max(np.abs(actual_q-planned_q)) <= self.owner.runtime.source_update_tolerance)
            except (KeyError, TypeError, ValueError, IndexError):
                pass
        if (proposal is not None and result.get('action_source') == 'v21_lewm_cem'
                and result.get('completed') is True
                and (result.get('response_verified') is True or proposal['next_state']['phase'] == 'wrist_xy')
                and result.get('policy_commands_returned') == 5
                and source_matches
                and result.get('commands_position') == proposal['commands_position']):
            self._fish_task = deepcopy(proposal['next_state'])
            if self._fish_task['phase'] == 'wrist_xy':
                actual_delta = np.asarray(result['endpoint_q_position']) - np.asarray(proposal['source_q_position'])
                command_delta = np.asarray(proposal['commands_position'][-1]) - np.asarray(proposal['source_q_position'])
                # Remember the observed height / realized roll, not "every
                # joint matched the forecast". Preserve response_verified as
                # reported. Closure uses the NEXT fresh image, not this flag.
                if abs(command_delta[4]) > .25 and actual_delta[4]*np.sign(command_delta[4]) > .25:
                    self._fish_task['low_roll_done'] = True
            if (self._fish_task['phase'] == 'release'
                    and float(result['endpoint_q_position'][5]) >= established.DEFAULT_GRIPPER_OPEN_POSITION - 3.):
                self._fish_task['phase'] = 'done'
            self._fish_task_proposal = None
            result['fish_task'] = deepcopy(self._fish_task)
            result['four_stage_sequence_complete'] = self._fish_task['phase'] == 'done'

    def _aligned_python_lower(self, state, task, dt):
        # Authorized short vertical lowering after current visual alignment.
        q = state.q[0]
        measured = q.detach().cpu().numpy()
        dz = max(-12., min(0., task.working_clearance_mm - task.actual_clearance_mm))
        step = self._approach_step(measured, dz, 0.)
        if step is None:
            task.python_lower_rejection = 'vertical lowering has no local solution'
            return None
        end = q.clone(); end[1:3] += q.new_tensor(step)
        alpha = torch.linspace(.2, 1., 5, device=q.device, dtype=q.dtype)
        selected = (q + alpha[:, None] * (end - q))[None]
        rt = self.owner.runtime
        try:
            rt._physical(state.q, selected)
        except RuntimeError as error:
            if not str(error).startswith('V20.3 command rejected:'):
                raise
            task.python_lower_rejection = str(error)
            return None
        from .v21_models import CommandTrace
        prediction = rt.core.predict(state, CommandTrace.policy(selected, dt), dt.double().cumsum(-1)).response
        try:
            rt._physical(state.q, prediction.predicted_q_path[:, 1:])
        except RuntimeError as error:
            if not str(error).startswith('V20.3 command rejected:'):
                raise
            task.python_lower_rejection = 'predicted lower path: ' + str(error)
            return None
        self.lewm_cem_trace_record = None
        self.lewm_cem_record = dict(available=True, objective=self.lewm_cem_objective,
            fish_task=task.receipt(), cem_bypassed=True, command_origin='python_aligned_vertical_lower',
            selected_predicted_q_path=prediction.predicted_q_path[0].detach().cpu().tolist(),
            ranking_state_source=None, learned_policy_success=False, uniform_chunk=True)
        self._record_fish_stage(state, selected, task)
        return selected

    def _near_blue_python_close(self, state, task, dt):
        """Automatically pinch on current low/coarse alignment; hold the arm."""
        if task.phase != 'close' or not task.close_ready:
            raise RuntimeError('Python close requires current low Wrist M/C and angle alignment')
        q = state.q[0]
        end = q.clone(); end[5] = task.closed
        alpha = torch.linspace(.2, 1., 5, device=q.device, dtype=q.dtype)
        selected = (q + alpha[:, None] * (end - q))[None]
        task.require_commands(state.q, selected)
        rt = self.owner.runtime
        rt._physical(state.q, selected)
        # This is a J6-only Python intervention, not a LeWM-selected arm move.
        # Runtime still forecasts these exact commands for the dashboard and
        # actual-response history. Predicted arm motion must not choose or veto
        # this fixed-arm close; command/path and hardware readback checks remain.
        self.lewm_cem_trace_record = None
        self.lewm_cem_record = dict(available=True, objective=self.lewm_cem_objective,
            fish_task=task.receipt(), cem_bypassed=True, command_origin='python_near_blue_close',
            selected_predicted_q_path=None, forecast_is_diagnostic=True,
            ranking_state_source=None, learned_policy_success=False, uniform_chunk=True,
            post_cem_command_override=False)
        self._record_fish_stage(state, selected, task)
        return selected

    def _uses_vertical_pinch_geometry(self):
        """Align FK M to the grasp point; close-time J4/J5. Off when assist flags are on."""
        if self.raw_actor or self.lewm_cem:
            return False
        if self.grounding_correction or self.head_grasp or self.alignment_motion_enabled:
            return False
        if self.grasp_target == 'head':
            return False
        return True

    def _column_body_station_px(self, mask):
        """Longest mid-width station on the support principal axis.

        Caudal fin is a short width spike, so argmax(width) and r² CoG land
        on the tail. Body is the longest run of mid-range slice widths.
        """
        array = np.asarray(mask, dtype=bool)
        if array.ndim != 2 or not bool(array.any()):
            return None
        ys, xs = np.nonzero(array)
        if xs.size < 24:
            return None
        pts = np.stack((xs.astype(np.float64), ys.astype(np.float64)), -1)
        origin = pts.mean(0)
        delta = pts - origin
        cov = delta.T @ delta / float(len(pts))
        _vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, int(np.argmax(_vals))]
        if float(np.linalg.norm(axis)) < 1e-12:
            return None
        unit = axis / (float(np.linalg.norm(axis)) + 1e-12)
        normal = np.array((-unit[1], unit[0]), np.float64)
        along = delta @ unit
        across = delta @ normal
        lo, hi = float(along.min()), float(along.max())
        span = hi - lo
        if span < 1e-6:
            return None
        bins = max(12, min(48, int(np.ceil(span))))
        edges = np.linspace(lo, hi + 1e-6, bins + 1)
        which = np.clip(np.digitize(along, edges) - 1, 0, bins - 1)
        widths = np.zeros(bins, np.float64)
        stations = np.full(bins, np.nan, np.float64)
        for k in range(bins):
            sel = which == k
            if int(sel.sum()) < 4:
                continue
            widths[k] = 2.0 * float(np.percentile(np.abs(across[sel]), 95))
            stations[k] = float(along[sel].mean())
        positive = widths[widths > 0.0]
        if positive.size < 3:
            return None
        mid = float(np.median(positive))
        if mid <= 0.0:
            return None
        mid_bins = (widths >= 0.55 * mid) & (widths <= 1.45 * mid)
        best_i = best_n = 0
        i = 0
        while i < bins:
            if mid_bins[i]:
                j = i + 1
                while j < bins and mid_bins[j]:
                    j += 1
                if j - i > best_n:
                    best_i, best_n = i, j - i
                i = j
            else:
                i += 1
        if best_n < 2:
            return None
        run = stations[best_i:best_i + best_n]
        run = run[np.isfinite(run)]
        if run.size == 0:
            return None
        grasp = origin + float(run.mean()) * unit
        return np.array((float(grasp[0]), float(grasp[1])), np.float64)

    def _column_grasp_pixel(self, column):
        """Column grasp pixel in letterbox224 for FK-M.

        Colour-body records already put C on the fish torso — use that cog
        directly. Never rebuild a station from DINO ``support_mask``: that
        locked M onto the blue tube (trial 1789359718702675047).
        """
        if column.get('source') == 'operator_colour_body':
            for key in ('cog_px', 'centroid_px'):
                value = column.get(key)
                if value is None:
                    continue
                pixel = np.asarray(value, np.float64).reshape(-1)
                if pixel.shape == (2,) and np.isfinite(pixel).all():
                    return pixel
            return None
        # Non-colour / DINO: use reported CoG only — never support_mask station.
        for key in ('cog_px', 'centroid_px'):
            value = column.get(key)
            if value is None:
                continue
            pixel = np.asarray(value, np.float64).reshape(-1)
            if pixel.shape == (2,) and np.isfinite(pixel).all():
                return pixel
        return None

    def _column_grasp_target_xyz(self):
        """Column-view 重心 back-projected onto the table + lift; target for FK M.

        Column tips are visual only — never FK. Wrist alone uses FK for F/G/M.
        """
        pickup = ((self.targets or {}).get('targets') or {}).get('pickup') or {}
        column = pickup.get('column') or {}
        pixel = self._column_grasp_pixel(column)
        if not column.get('visible') or pixel is None:
            return None
        try:
            if getattr(self, '_column_fit', None) is None:
                row = json.loads(WRIST_REFERENCE.read_text(encoding='utf-8'))['fits']['column']
                self._column_fit = ProjectionFit(
                    'column', tuple(row['feature_names']),
                    np.asarray(row['coefficients'], np.float64), float(row['ridge']))
            return approximate_column_backprojection_on_plane(
                self._column_fit, pixel,
                z_base=TABLE_Z_M + GRASP_LIFT_M)
        except (ValueError, KeyError, TypeError, OSError):
            return None

    def _column_drop_target_xyz(self):
        """Column drop-zone 重心 on the table + lift; target after a closed grasp."""
        drop = ((self.targets or {}).get('targets') or {}).get('drop_zone') or {}
        column = drop.get('column') or {}
        pixel = self._column_grasp_pixel(column)
        if not column.get('visible') or pixel is None:
            return None
        try:
            if getattr(self, '_column_fit', None) is None:
                row = json.loads(WRIST_REFERENCE.read_text(encoding='utf-8'))['fits']['column']
                self._column_fit = ProjectionFit(
                    'column', tuple(row['feature_names']),
                    np.asarray(row['coefficients'], np.float64), float(row['ridge']))
            return approximate_column_backprojection_on_plane(
                self._column_fit, pixel,
                z_base=TABLE_Z_M + GRASP_LIFT_M + self.GEOMETRY_PLACE_LIFT_M)
        except (ValueError, KeyError, TypeError, OSError):
            return None

    # Geometry place: shut jaws near the fish, lift, servo to box, open.
    # Independent of grounding `_carry` (wrong source / visual-pixel assist).
    # Actor-verify default: force_close OFF so J4–J6 stay on Actor until it closes.
    # Re-enable with GEOMETRY_FORCE_CLOSE_ENABLED = True (threshold still GEOMETRY_FORCE_CLOSE_M).
    GEOMETRY_FORCE_CLOSE_ENABLED = False
    GEOMETRY_FORCE_CLOSE_M = 0.040
    GEOMETRY_PLACE_LIFT_M = 0.040
    GEOMETRY_PLACE_XY_M = 0.040
    GEOMETRY_PLACE_MAX_BLOCKS = 8
    # Wrist r = |object CoG − jaw midpoint| / 224 from the same 8-D the Actor
    # sees. Above WRIST_R_CLOSE the jaws may not close and J1–J3 servo the
    # wrist projection toward the CoG (no roll, no FG⊥). Column M is an
    # auxiliary check only — trial 1789358513759997162 had M≈1 cm with r≈0.5.
    WRIST_R_CLOSE = 0.12
    WRIST_R_MIN_SCORE = 0.35
    WRIST_R_BORDER_PX = 12.0
    # Keep width gate strict: trial 1789359718702675047 B08/B16 wrist CoG was
    # already the blue tube; lowering this would IBVS onto the tube. Fix column
    # fish lock first; wrist cue must reject non-fish before relaxing shape.
    WRIST_R_MIN_WIDTH = 0.05
    WRIST_R_MIN_ASPECT = 1.2

    def _wrist_r_cue(self):
        """Validated wrist proximity cue for the approach/close gate, or None.

        When a colour-body fish record exists, gate and servo from that body's
        8-D (diagnostic), not the thin DINO Actor strip that often
        ``shape_rejected`` and hands XY to column M (tube).
        """
        enc = getattr(self.owner, 'object_relative', None)
        record = getattr(enc, 'last_record', None) or {}
        vector = getattr(self.owner, 'last_object_relative', None)
        if record.get('source') == 'operator_colour_body':
            for candidate in (
                    getattr(self.owner, 'last_object_diagnostic', None),
                    getattr(enc, 'last_diagnostic_vector', None) if enc is not None else None):
                if candidate is None:
                    continue
                cand = np.asarray(candidate, np.float64).reshape(-1)
                if cand.shape == (8,) and np.isfinite(cand).all() and float(cand[0]) > 0:
                    vector = cand
                    break
        if vector is None:
            return None
        v = np.asarray(vector, np.float64).reshape(-1)
        if v.shape != (8,) or not np.isfinite(v).all():
            return None
        cue = dict(available=bool(v[0] > 0), score=float(v[1]), dx=float(v[2]), dy=float(v[3]),
                   r=float(np.hypot(v[2], v[3])), cog_px=None, valid=False, reason=None)
        if not cue['available']:
            cue['reason'] = 'unavailable'
            return cue
        if cue['score'] < self.WRIST_R_MIN_SCORE:
            cue['reason'] = 'low_score'
            return cue
        width, length = float(v[5]), float(v[4])
        if width < self.WRIST_R_MIN_WIDTH or length < self.WRIST_R_MIN_ASPECT * max(width, 1e-6):
            cue['reason'] = 'shape_rejected'
            return cue
        # Prefer colour-body CoG for the servo target; DINO policy CoG can sit
        # on the caudal fin or latch near the tube after approach.
        if record.get('source') != 'operator_colour_body':
            record = getattr(enc, 'last_policy_record', None) or {}
        cog = record.get('cog_px') if record.get('cog_px') is not None else record.get('centroid_px')
        if cog is None:
            cue['reason'] = 'no_cog'
            return cue
        pixel = np.asarray(cog, np.float64).reshape(-1)
        if pixel.shape != (2,) or not np.isfinite(pixel).all():
            cue['reason'] = 'no_cog'
            return cue
        b = self.WRIST_R_BORDER_PX
        if not (b <= pixel[0] <= 224. - b and b <= pixel[1] <= 224. - b):
            cue['reason'] = 'cog_on_border'
            cue['cog_px'] = pixel.tolist()
            return cue
        # If the colour body is missing and DINO support is clipped / non-aimable,
        # do not servo — column M must own approach (tube vs fish).
        policy = getattr(enc, 'last_policy_record', None) or {}
        if (record.get('source') != 'operator_colour_body'
                and (policy.get('support_clipped') or policy.get('aimable') is False)):
            cue['reason'] = 'dino_clipped_no_colour'
            cue['cog_px'] = pixel.tolist()
            return cue
        cue.update(cog_px=pixel.tolist(), valid=True, source=record.get('source'),
                   body_complete=not bool(record.get('body_truncated')))
        return cue

    def _geom_place_commands(self, q, cmds, action):
        """Force-close / lift / box / open. Returns (commands[5,6], diagnostic) or None."""
        place = self._geom_place
        if place is None:
            return None
        kin = getattr(self.owner, 'kinematics', None)
        lower = established.SO101_POSITION_LOWER
        upper = established.SO101_POSITION_UPPER
        closed = float(established.DEFAULT_GRIPPER_CLOSED_POSITION)
        opened = float(established.DEFAULT_GRIPPER_OPEN_POSITION)
        ramp = np.linspace(0.2, 1.0, 5)
        place['blocks'] = int(place.get('blocks', 0)) + 1
        if place['blocks'] > self.GEOMETRY_PLACE_MAX_BLOCKS:
            self._geom_place = None
            return cmds, dict(kind='execution_geometry_place', phase='budget_exhausted',
                              learned_policy_success=False, assistance=False)
        phase = place.get('phase')
        out = np.asarray(cmds, np.float64).copy()
        diag = dict(kind='execution_geometry_place', phase=phase,
                    learned_policy_success=False, assistance=False, blocks=place['blocks'])
        if phase == 'close':
            # Hold pose, ramp J6 to the calibration closed position.
            for i in range(5):
                out[i, :5] = q[:5]
                out[i, 5] = q[5] + (closed - q[5]) * ramp[i]
            place['phase'] = 'lift'
            diag['phase'] = 'close'
            return out, diag
        if phase == 'lift':
            lift_xyz = place.get('lift_xyz')
            if lift_xyz is None or kin is None:
                self._geom_place = None
                return None
            out, aim = project_fk_m_toward_xyz(q, out, kin, lift_xyz, lower, upper)
            out[:, 5] = min(float(q[5]), closed)
            diag['aim'] = aim
            try:
                mid = fk_midpoint_m(kin, q)
                if float(mid[2]) >= float(lift_xyz[2]) - 0.015:
                    place['phase'] = 'carry'
            except (ValueError, TypeError, RuntimeError, AttributeError):
                pass
            # Also advance after one lift block so a soft Z miss cannot stall.
            if place['blocks'] >= 2 and place.get('phase') == 'lift':
                place['phase'] = 'carry'
            return out, diag
        if phase == 'carry':
            drop = self._column_drop_target_xyz()
            if drop is None:
                # Keep last commanded height; hold closed until the box is visible.
                out[:, :] = q[None, :]
                out[:, 5] = min(float(q[5]), closed)
                diag['waiting'] = 'drop_zone_hidden'
                return out, diag
            out, aim = project_fk_m_toward_xyz(q, out, kin, drop, lower, upper)
            out[:, 5] = min(float(q[5]), closed)
            diag['aim'] = aim
            try:
                mid = fk_midpoint_m(kin, q)
                if (float(np.linalg.norm(mid[:2] - np.asarray(drop, np.float64)[:2]))
                        <= self.GEOMETRY_PLACE_XY_M):
                    place['phase'] = 'open'
            except (ValueError, TypeError, RuntimeError, AttributeError):
                pass
            if place['blocks'] >= 6:
                place['phase'] = 'open'
            return out, diag
        if phase == 'open':
            for i in range(5):
                out[i, :5] = q[:5]
                out[i, 5] = q[5] + (opened - q[5]) * ramp[i]
            self._geom_place = None
            self._home_after_release = True
            diag['phase'] = 'open'
            diag['home_next'] = True
            return out, diag
        self._geom_place = None
        return None

    def _vertical_pinch_builder(self, state, command_offsets):
        rt = self.owner.runtime
        action = rt.core.act(state, rt.text_ids, command_offsets).commands_position.clone()
        if not getattr(rt, 'recomputing_at_dispatch', False):
            self._publish()  # display only; skipped on the dispatch-deadline path
        q_t = state.q[0]
        q = q_t.detach().cpu().numpy()
        cmds = action[0].detach().cpu().numpy()
        kin = getattr(self.owner, 'kinematics', None)
        lower = established.SO101_POSITION_LOWER
        upper = established.SO101_POSITION_UPPER
        # Place FSM owns the arm after a geometry force-close.
        placed = self._geom_place_commands(q, cmds, action)
        if placed is not None:
            projected, diagnostic = placed
            action[0] = torch.as_tensor(projected, device=action.device, dtype=action.dtype)
            self.execution_geometry_record = diagnostic
            return action
        target = self._column_grasp_target_xyz()
        unfolded = float(q[1]) > self.GROUNDING_UNFOLDED_J2
        jaws_open = float(q[5]) > float(established.DEFAULT_GRIPPER_CLOSED_POSITION) + 10.
        closing = bool((np.diff(np.concatenate(([float(q[5])], cmds[:, 5]))) < -1e-5).any())
        cue = self._wrist_r_cue()
        r_far = cue is not None and cue['valid'] and cue['r'] > self.WRIST_R_CLOSE
        # Close gate: the Actor may shut the jaws only when the wrist sees the
        # object inside WRIST_R_CLOSE. Far, unavailable or rejected → hold J6.
        close_veto = None
        if closing and unfolded and jaws_open:
            if cue is None or not cue['valid']:
                close_veto = 'wrist_cue_' + str((cue or {}).get('reason') or 'missing')
            elif r_far:
                close_veto = 'wrist_r_far'
        if close_veto is not None:
            cmds = cmds.copy()
            cmds[:, 5] = float(q[5])
            closing = False
        diagnostic = None
        projected = cmds
        wrist_diag = None
        enc = getattr(self.owner, 'object_relative', None)
        jaw_pair_fn = getattr(enc, 'jaw_pair', None)
        if r_far and unfolded and jaws_open and jaw_pair_fn is not None:
            # Approach phase: J1–J3 shrink the wrist r. Column M-align is not
            # run here — it saturates on a table-plane proxy and steals J1.
            try:
                projected, wrist_diag = project_wrist_m_toward_pixel(
                    q, cmds, jaw_pair_fn, cue['cog_px'], lower, upper, kinematics=kin)
            except (ValueError, RuntimeError, TypeError) as exc:
                wrist_diag = dict(kind='execution_geometry_wrist_r', applied=False,
                                  reason=f'{type(exc).__name__}')
                projected = cmds
            if wrist_diag.get('applied'):
                diagnostic = dict(kind='execution_geometry_trial', learned_policy_success=False,
                                  assistance=False, applied=True, closing=False,
                                  aim=dict(kind='execution_geometry_align_M', applied=False,
                                           reason='wrist_r_owns_approach'),
                                  pinch=dict(kind='execution_geometry_vertical_pinch',
                                             applied=False, reason='not_closing'),
                                  approach_override='wrist_r')
            else:
                # Do not let Actor dive into the table while cue says fish is still far.
                projected = cmds.copy()
                projected[:, :3] = q[:3]
                override, aim_reason = 'wrist_r_hold', 'wrist_r_hold_no_safe_step'
                column = ((getattr(self, 'targets', None) or {}).get('targets') or {}).get('pickup', {}).get('column') or {}
                if ((wrist_diag or {}).get('reason') == 'no_progress_within_limits'
                        and column.get('source') == 'operator_colour_body'
                        and target is not None and kin is not None):
                    try:
                        pulled, aim2 = project_fk_m_toward_xyz(
                            q, projected, kin, target, lower, upper)
                        if aim2.get('applied'):
                            projected = pulled
                            override = 'column_m_after_wrist_stall'
                            aim_reason = 'column_m_after_wrist_stall'
                            wrist_diag = dict(wrist_diag or {}, column_fallback=True)
                            diagnostic = dict(kind='execution_geometry_trial',
                                              learned_policy_success=False, assistance=False,
                                              applied=True, closing=False, aim=aim2,
                                              pinch=dict(kind='execution_geometry_vertical_pinch',
                                                         applied=False, reason='not_closing'),
                                              approach_override=override)
                    except (ValueError, RuntimeError, TypeError):
                        pass
                if diagnostic is None:
                    diagnostic = dict(kind='execution_geometry_trial', learned_policy_success=False,
                                      assistance=False, applied=True, closing=False,
                                      aim=dict(kind='execution_geometry_align_M', applied=False,
                                               reason=aim_reason),
                                      pinch=dict(kind='execution_geometry_vertical_pinch',
                                                 applied=False, reason='not_closing'),
                                      approach_override=override)
                    wrist_diag = dict(wrist_diag or {}, held_xyz=True)
        if diagnostic is None:
            try:
                projected, diagnostic = project_trial_geometry(
                    q, cmds, kin, lower, upper, target_xyz_m=target)
            except (ValueError, RuntimeError, TypeError) as exc:
                diagnostic = dict(kind='execution_geometry_trial', applied=False,
                                  reason=f'{type(exc).__name__}', learned_policy_success=False)
                try:
                    projected, pinch = project_vertical_pinch(q, cmds, kin, lower, upper)
                    diagnostic['pinch'] = pinch
                except (ValueError, RuntimeError, TypeError):
                    projected = cmds
            # Legacy column pull: only when the wrist cue cannot own the approach
            # (no valid cue) or the wrist already reports near. Never from the
            # folded home pose — that Jacobians J2 into the limit. Never while a
            # wrist colour-body fish exists: column M can still sit on the tube.
            aim = (diagnostic or {}).get('aim') or {}
            wrist_colour = (getattr(enc, 'last_record', None) or {}).get('source') == 'operator_colour_body'
            if (target is not None and kin is not None and unfolded and not aim.get('applied')
                    and not wrist_colour
                    and float(q[5]) > float(established.DEFAULT_GRIPPER_CLOSED_POSITION) + 5.):
                try:
                    pulled, aim2 = project_fk_m_toward_xyz(q, projected, kin, target, lower, upper)
                    if aim2.get('applied'):
                        projected = pulled
                        diagnostic['aim'] = aim2
                        diagnostic['approach_override'] = True
                except (ValueError, RuntimeError, TypeError):
                    pass
        aim = (diagnostic or {}).get('aim') or {}
        diagnostic['wrist_r'] = dict(cue=cue, close_veto=close_veto, servo=wrist_diag)
        action[0] = torch.as_tensor(projected, device=action.device, dtype=action.dtype)
        err = aim.get('error_m_after_step')
        if err is None:
            err = aim.get('error_m')
        near = bool(aim.get('applied')) and err is not None and float(err) <= self.GEOMETRY_FORCE_CLOSE_M
        near = near and cue is not None and cue['valid'] and not r_far
        closed = float(established.DEFAULT_GRIPPER_CLOSED_POSITION)
        column_src = (((getattr(self, 'targets', None) or {}).get('targets') or {}).get('pickup', {}).get('column') or {}).get('source')
        # Over the fish but still high: go down to the table plane, then close.
        if (cue is not None and cue.get('valid')
                and cue.get('source') == 'operator_colour_body'
                and float(cue['r']) <= 0.20
                and not (wrist_diag or {}).get('j1_pan_deg')
                and column_src == 'operator_colour_body'
                and unfolded and kin is not None
                and self._geom_place is None
                and float(q[5]) > closed + 2.):
            try:
                mid = np.asarray(fk_midpoint_m(kin, q), np.float64).reshape(3)
                if np.isfinite(mid).all() and float(mid[2]) > TABLE_Z_M + 0.022:
                    down = mid.copy()
                    down[2] = TABLE_Z_M + 0.016
                    # Wrist owns lateral centering. Descend at the current
                    # XY, never overwrite it with the biased Column plane proxy.
                    pulled, aim2 = project_fk_m_toward_xyz(
                        q, projected, kin, down, lower, upper, lock_j1=True)
                    if aim2.get('applied'):
                        projected = pulled
                        projected[:, 5] = float(q[5])
                        action[0] = torch.as_tensor(
                            projected, device=action.device, dtype=action.dtype)
                        diagnostic = dict(diagnostic or {}, approach_override='descend_onto_fish',
                                          aim=aim2)
            except (ValueError, TypeError, RuntimeError, AttributeError):
                pass
        wrist_place = (
            cue is not None and cue.get('valid')
            and cue.get('source') == 'operator_colour_body'
            and float(cue['r']) <= 0.18
            and column_src == 'operator_colour_body'
            and unfolded and kin is not None
            and float(q[5]) > closed + 2.
            and self._geom_place is None
        )
        if wrist_place:
            try:
                z_m = float(fk_midpoint_m(kin, q)[2])
                # Hovering over the fish with r≤0.18 is not a pinch: trial
                # 1789364859151839820 closed in air then homed. Require M
                # near the table before shutting the jaws.
                wrist_place = z_m <= TABLE_Z_M + 0.022
            except (ValueError, TypeError, RuntimeError, AttributeError):
                wrist_place = False
        if (self.GEOMETRY_FORCE_CLOSE_ENABLED and near and jaws_open and unfolded) or wrist_place:
            # Freeze the pickup XYZ so lift/carry does not chase a held-fish CoG.
            try:
                freeze = np.asarray(fk_midpoint_m(kin, q), np.float64).reshape(3)
                if not np.isfinite(freeze).all():
                    raise ValueError('fk mid')
            except (ValueError, TypeError, RuntimeError, AttributeError):
                freeze = None if target is None else np.asarray(target, np.float64).reshape(3)
            if freeze is not None and np.isfinite(freeze).all():
                lift_xyz = freeze.copy()
                lift_xyz[2] = float(freeze[2]) + self.GEOMETRY_PLACE_LIFT_M
                self._geom_place = dict(phase='close', blocks=0,
                                        pickup_xyz=freeze.tolist(), lift_xyz=lift_xyz.tolist())
                projected2, diag2 = self._geom_place_commands(q, projected, action)
                if projected2 is not None:
                    action[0] = torch.as_tensor(projected2, device=action.device, dtype=action.dtype)
                    diagnostic = dict(diagnostic or {}, force_close=True,
                                      force_close_error_m=None if err is None else float(err),
                                      wrist_place=bool(wrist_place), place=diag2)
        elif near and jaws_open and unfolded and not self.GEOMETRY_FORCE_CLOSE_ENABLED:
            diagnostic = dict(diagnostic or {}, force_close=False,
                              force_close_deferred=True,
                              force_close_error_m=float(err))
        self.execution_geometry_record = diagnostic
        return action
    _vertical_pinch_builder.wants_command_offsets=True

    def _lewm_cem_builder(self, state, command_offsets):
        from .v21_lewm_cem import plan_commands
        rt = self.owner.runtime
        # bridge.plan has just acquired a NEW endpoint. Bind target/F/G pixels
        # to it before scoring, rather than using the previous dashboard frame.
        self._publish()
        self._cem_source_evidence = None
        self._fish_task_proposal = None
        planning_input = None
        planning_input_error = None
        if self.record_policy_inputs:
            from .v21_source_evidence import snapshot_source_evidence, save_source_evidence
            stamp = time.time_ns()
            bundle = snapshot_source_evidence(state, getattr(self.owner,'source_evidence_store',None))
            input_path = self.output/f'planning-input-{stamp}.pt'
            try:
                reference = save_source_evidence(self.output, f'source-plan-{stamp}', bundle)
                self._cem_source_evidence = (bundle, reference)
                with input_path.open('xb') as stream:
                    torch.save(dict(kind='nonexecutable-planning-input', schema=1, executable=False,
                        actual_future_labels_available=False, state=cpu_actual_state(state),
                        model_output=str(self.root.resolve()), text_ids=rt.text_ids.detach().cpu().clone(),
                        command_offsets_s=command_offsets.detach().cpu().clone(),
                        source_visual_evidence=deepcopy(reference),
                        lewm_controller=deepcopy(self.lewm_controller),
                        temporal_contract=deepcopy(self.temporal_contract)), stream)
                planning_input = dict(path=input_path.name, executable=False,
                    sha256=hashlib.sha256(input_path.read_bytes()).hexdigest())
            except Exception as capture_error:
                planning_input_error = capture_error
                input_path.unlink(missing_ok=True)
        self._horizon40_extension = None
        self._horizon40_note = None
        if rt.core.actor.horizon == 40:
            dt40 = torch.full((1, 40), .1, device=state.q.device, dtype=torch.float64)
            offsets40 = torch.cat((dt40.new_zeros(1, 1), dt40.cumsum(-1)[:, :-1]), -1)
            full = rt.core.act(state, rt.text_ids, offsets40).commands_position
            if self.lewm_cem_objective == 'wrist_distance_alignment':
                # This objective forecasts each sampled candidate for 40 steps.
                # It must never fall through to Actor tracking when vision/fit
                # is missing, or to the old early-return CEM bypass.
                proposal = full
            else:
                from .v21_approach_horizon import select_approach_commands
                straight, self._horizon40_note = select_approach_commands(
                    state.q, full, getattr(state, 'object_relative', None))
                proposal = straight[None]
                self._horizon40_extension = None
                self._horizon40_note = dict(self._horizon40_note or {}, inferred_steps=40,
                    execute_steps=5, lewm_steps=5, actor_execute_steps=5)
                # Close is decided after the chunk is built: the jaw has to sit
                # on the blue grasp line (distance 0 is allowed) and at the floor.
                self._grasp_hold_q6 = None
                joined = torch.cat((state.q[0].to(full)[None], proposal[0]), 0)
                j5_step = float((joined[1:, 4] - joined[:-1, 4]).abs().max())
                # #region agent log
                try:
                    import json as _json, time as _time
                    _rel = getattr(state, 'object_relative', None)
                    _rel_l = None if _rel is None else [round(float(x), 4) for x in _rel.detach().float().cpu().reshape(-1)[:8]]
                    _cmd = full[0, :15].detach().float().cpu()
                    _q = state.q[0].detach().float().cpu()
                    _prev = torch.cat((_q[None], _cmd[:-1]), 0)
                    _d = (_cmd - _prev).mean(0)
                    with open('/home/james/feedback-corrected-grounding/.cursor/debug-666150.log', 'a') as _f:
                        _f.write(_json.dumps({'sessionId': '666150', 'hypothesisId': 'A', 'location': 'v21_live_service.py:actor40', 'message': 'actor 15-step vs fish dx', 'data': {'dx': None if _rel_l is None else _rel_l[2], 'dy': None if _rel_l is None else _rel_l[3], 'available': None if _rel_l is None else _rel_l[0], 'mean_delta': [round(float(x), 3) for x in _d], 'j5_step_max': j5_step, 'execute_steps': None if self._horizon40_note is None else self._horizon40_note.get('execute_steps'), 'q': [round(float(x), 2) for x in _q]}, 'timestamp': int(_time.time() * 1000)}) + '\n')
                except Exception:
                    pass
                # #endregion
        else:
            if self.lewm_cem_objective == 'wrist_distance_alignment':
                raise ValueError('distance/alignment CEM requires the native40 checkpoint')
            proposal = rt.core.act(state, rt.text_ids, command_offsets).commands_position
        dt = torch.full((1, 5), .1, device=state.q.device, dtype=state.q.dtype)
        cem_kwargs = ({} if self.lewm_cem_candidate_parameterization is None else
            dict(candidate_parameterization=self.lewm_cem_candidate_parameterization))
        cem_kwargs['action_scale'] = self.lewm_cem_action_scale
        self.lewm_cem_trace_record = None
        if self.record_policy_inputs:
            from .v21_cem_trace import CEMTrace
            cem_kwargs['trace'] = CEMTrace()

        def recorded_plan(*args, planner=plan_commands, **kwargs):
            failure = None
            planned_result = None
            try:
                planned_result = planner(*args, **kwargs)
                return planned_result
            except BaseException as error:
                failure = error
                raise
            finally:
                trace = kwargs.get('trace')
                if trace is not None:
                    trace_path = None
                    trace_record = None
                    trace_error = None
                    try:
                        payload = trace.payload()
                        payload.update(source_time_s=float(state.time[0]),
                            source_q_position=state.q[0].detach().cpu().tolist(),
                            model_output=str(self.root.resolve()),
                            lewm_controller=deepcopy(self.lewm_controller),
                            temporal_contract=deepcopy(self.temporal_contract),
                            planning_input=deepcopy(planning_input),
                            source_visual_evidence=(None if self._cem_source_evidence is None else
                                deepcopy(self._cem_source_evidence[1])))
                        trace_path = self.output/f'cem-trace-{time.time_ns()}.pt'
                        with trace_path.open('xb') as stream:
                            torch.save(payload, stream)
                        trace_record = dict(path=trace_path.name, saved=True,
                            sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                            complete=payload['complete'], executable=False,
                            source_time_s=float(state.time[0]),
                            source_q_position=state.q[0].detach().cpu().tolist(),
                            planning_input=deepcopy(planning_input),
                            source_visual_evidence=(None if self._cem_source_evidence is None else
                                deepcopy(self._cem_source_evidence[1])),
                            selected_commands_position=(None if planned_result is None else
                                planned_result[0].detach().cpu().tolist()))
                    except Exception as capture_error:
                        trace_error = capture_error
                    evidence_errors = [error for error in (planning_input_error, trace_error)
                                       if error is not None]
                    if evidence_errors:
                        if trace_path is not None:
                            trace_path.unlink(missing_ok=True)
                        self.lewm_cem_trace_record = dict(saved=False, executable=False,
                            error='; '.join(f'{type(error).__name__}: {error}'
                                           for error in evidence_errors))
                        if failure is None:
                            raise RuntimeError('CEM source trace could not be persisted') from evidence_errors[0]
                    else:
                        self.lewm_cem_trace_record = trace_record
        if self.lewm_cem_objective == 'wrist_distance_alignment':
            from .v21_grasp_cem import plan_grasp_commands
            from .v21_grasp_cost import WristGraspCost
            self.lewm_cem_record = dict(objective=self.lewm_cem_objective, available=False,
                cem_bypassed=False, inferred_steps=40, execute_steps=5,
                small_step_extension_enabled=False, physical_alignment_verified=False)
            try:
                response = getattr(self, 'wrist_grasp_response', None)
                if response is None:
                    raise ValueError('measured Wrist distance/angle response unavailable; calibration required')
                observation = self._alignment_measure()
                memory = state.memory
                if (not bool(state.measurement_available[0, 0])
                        or float(memory.image_times[0, 0, -1]) != observation.image_time_s
                        or not np.allclose(memory.q_at_image[0, 0, -1].detach().cpu().numpy(),
                                           observation.q_position, rtol=0, atol=1e-5)):
                    raise ValueError('Wrist scoring observation differs from current policy exposure')
                record = self.owner.object_relative.last_policy_record
                if record.get('body_truncated'):
                    raise ValueError('complete current fish axis required for CEM alignment')
                task_cost = WristGraspCost(observation, response)
                self.lewm_cem_record['visual_task_cost'] = task_cost.receipt()
                selected, record = recorded_plan(rt.core.lewm, state, proposal, dt,
                    rt.physical_check, planner=plan_grasp_commands, task=task_cost,
                    lower=rt.core.actor.position_lower.detach().cpu().numpy(),
                    upper=rt.core.actor.position_upper.detach().cpu().numpy(),
                    clearance=self.owner.clearance.clearance_mm, trace=cem_kwargs.get('trace'))
                self.lewm_cem_record = dict(record, available=True,
                    calibration=getattr(self, 'wrist_grasp_calibration', None))
                return selected
            except (ValueError, RuntimeError) as error:
                self.lewm_cem_record.update(available=False, reason=str(error))
                raise
        if self.lewm_cem_objective in ('actor_response_tracking', 'actor_command_tracking'):
            visual_task = None
            fish_task = None
            visual_reason = 'actor_command_tracking_has_no_visual_response_cost'
            if self.lewm_cem_objective == 'actor_response_tracking':
                from .v21_visual_task_cost import bind_visual_task_cost, visual_task_precondition
                response = getattr(self, 'column_dino_fg_response', None)
                visual_reason = visual_task_precondition(response, state)
                if visual_reason is None:
                    # Tracking must not require, inspect, or truth-test visual
                    # state before a genuine 6D fit and model input are present.
                    tips = getattr(self, 'fingertips', None)
                    targets = getattr(self, 'targets', None)
                    if type(tips) is not dict or type(targets) is not dict:
                        visual_reason = 'measured_visual_state_unavailable'
                    else:
                        tip = tips.get('column')
                        groups = targets.get('targets')
                        pickup = groups.get('pickup') if type(groups) is dict else None
                        target = pickup.get('column') if type(pickup) is dict else None
                        if type(tip) is not dict or type(target) is not dict:
                            visual_reason = 'measured_visual_state_unavailable'
                        else:
                            try:
                                visual_task = bind_visual_task_cost(response, tip, target, state,
                                                                   actor_commands=proposal)
                            except (ValueError, TypeError, KeyError, AttributeError) as error:
                                visual_reason = str(error)
                if visual_task is None:
                    visual_task, image_reason = self._image_anchored_lewm_cost()
                    if visual_task is not None:
                        visual_reason = None
                    elif image_reason:
                        visual_reason = image_reason
                # Both views supply the phase-specific visual cost. World-model
                # predictions, not a post-CEM servo, decide the free joints.
                image_task, _ = self._image_anchored_lewm_cost()
                fish_task = self._fish_stage_cost(state, image_task or visual_task)
                if fish_task.phase == 'close':
                    return self._near_blue_python_close(state, fish_task, dt)
                if fish_task.phase == 'lower':
                    lowered = self._aligned_python_lower(state, fish_task, dt)
                    if lowered is not None:
                        return lowered
                    # A singular / out-of-bounds vertical helper is not a
                    # command. Let CEM search legal J2/J3 motion at the SAME
                    # height goal and with J1 still held. All physical and
                    # LeWM candidate checks remain in place.
                    fish_task.lowering_source = 'cem_selected_unchanged'
                visual_task = fish_task
            selected, record = recorded_plan(
                rt.core.lewm, state, proposal, dt, rt.physical_check,
                objective=self.lewm_cem_objective, visual_task_cost=visual_task,
                single_axis_stall=False, limit_step=False, **cem_kwargs)
            # #region agent log
            try:
                import json as _json, time as _time
                _gap = (selected - proposal).detach().float().abs().mean().item()
                with open('/home/james/feedback-corrected-grounding/.cursor/debug-666150.log', 'a') as _f:
                    _f.write(_json.dumps({'sessionId': '666150', 'hypothesisId': 'C', 'location': 'v21_live_service.py:cem', 'message': 'cem rewrote the actor prefix', 'data': {'mean_abs_gap': round(float(_gap), 4), 'selected0': [round(float(x), 2) for x in selected[0, 0].detach().float().cpu()], 'proposal0': [round(float(x), 2) for x in proposal[0, 0].detach().float().cpu()]}, 'timestamp': int(_time.time() * 1000)}) + '\n')
            except Exception:
                pass
            # #endregion
            self.lewm_cem_record = dict(record, visual_goal=visual_task is not None,
                column_dino_fg_transport_installed=visual_task is not None)
            if self._horizon40_note is not None:
                self.lewm_cem_record.update(self._horizon40_note,
                    cem_bypassed=False, cem_forecast_steps=5,
                    path='native40_actor_reference_then_five_step_cem')
            if visual_task is None:
                self.lewm_cem_record['visual_task_cost'] = dict(available=False, reason=visual_reason)
            if fish_task is not None:
                self.lewm_cem_record.update(fish_task=fish_task.receipt(), uniform_chunk=True,
                    command_origin='cem_selected_unchanged', post_cem_command_override=False)
                self._record_fish_stage(state, selected, fish_task)
            # NO rewriting selected or its trace here. Uniform timing and joint
            # permissions were applied to candidates BEFORE LeWM evaluated them.
            return selected
        tip = (self.fingertips or {}).get('column') or {}
        target = ((self.targets.get('targets') or {}).get('pickup') or {}).get('column') or {}
        project = target_px = observed = capture_q = None
        if (tip_source_matches_view('column', tip) and tip.get('available')
                and target.get('visible')):
            raw = tip.get('m_px')
            mid = None if raw is None else np.asarray(raw, np.float64).reshape(-1)
            if mid is not None and mid.shape == (2,) and np.isfinite(mid).all():
                observed = mid
                raw_q = tip.get('q_at_image')
                try:
                    candidate_q = np.asarray(raw_q, np.float64).reshape(-1)
                    if candidate_q.shape == (6,) and np.isfinite(candidate_q).all():
                        capture_q = candidate_q
                except (TypeError, ValueError):
                    pass
                # Bind only a measured response to this exact DINO F/G
                # exposure.  A static capture M is not a candidate prediction.
                try:
                    capture = ColumnDinoFGSample(capture_q, observed,
                                                  float(tip.get('image_time_s')))
                    transport = bind_column_dino_fg(self.column_dino_fg_response, capture)
                    project = transport.project
                except (TypeError, ValueError):
                    pass
            target_px = target.get('cog_px')
        selected, record = recorded_plan(
            rt.core.lewm, state, proposal, dt, rt.physical_check,
            project=project, target_px=target_px,
            observed_midpoint_px=observed, capture_q=capture_q,
            long_axis_px=target.get('axis_px'),
            grasp_axis_px=None,
            length_px=target.get('length_px'),
            width_px=target.get('width_px'),
            clearance_mm=getattr(getattr(self.owner, 'clearance', None),
                                 'minimum_planned_clearance_mm', None),
            require_candidate_sensitive_visual=True, **cem_kwargs)
        self.lewm_cem_record = dict(
            record,
            visual_goal=bool(observed is not None and target_px is not None and capture_q is not None),
            column_dino_fg_transport_installed=bool(project is not None),
            column_dino_fg_transport=deepcopy(self.column_dino_fg_transport_status),
        )
        return selected
    _lewm_cem_builder.wants_command_offsets = True

    def _plan_with_c4(self, reuse_completed):
        plan_kw = dict(command_builder=self.c4.builder, action_source=self.c4.source)
        if reuse_completed and self.trial.owns_current_thread() and self.trial.status()['blocks_completed'] > 0:
            planner = getattr(self.owner.bridge, 'plan_from_completed_observation', None)
            commands = planner(**plan_kw) if planner is not None else self.owner.bridge.plan(**plan_kw)
        else:
            commands = self.owner.bridge.plan(**plan_kw)
        if getattr(self.owner.runtime, 'pending_action_source', None) != self.c4.source:
            raise RuntimeError('C4 planning did not produce its own plan')
        return commands

    def _plan_with_policy(self, *, reuse_completed=False):
        if self.c4 is not None:
            return self._plan_with_c4(reuse_completed)
        if self.grounding_correction:
            if not getattr(self.owner.runtime.core.config,'object_relative',False):
                raise RuntimeError('grounding correction needs an object-relative Actor')
            proposal = self._assistance.propose(lambda: self.owner.bridge.plan(
                command_builder=self._grounding_builder,action_source='v21_actor_grounding_corrected'))
            self._assistance_proposal_id = proposal.proposal_id
            return proposal.result
        if self.lewm_cem:
            plan_kw = dict(command_builder=self._lewm_cem_builder, action_source='v21_lewm_cem')
            if (reuse_completed and self.trial.owns_current_thread()
                    and self.trial.status()['blocks_completed'] > 0):
                planner = getattr(self.owner.bridge, 'plan_from_completed_observation', None)
                commands = planner(**plan_kw) if planner is not None else self.owner.bridge.plan(**plan_kw)
            else:
                commands = self.owner.bridge.plan(**plan_kw)
            if getattr(self.owner.runtime, 'pending_action_source', None) != 'v21_lewm_cem':
                raise RuntimeError('LeWM CEM cannot fall back to Actor')
            return commands
        if reuse_completed and self.trial.owns_current_thread():
            # A capped non-CEM trial is the unmodified Actor experiment. The
            # first block captures H0 once; later blocks consume the actual
            # endpoint already committed by the preceding transaction.
            if self.trial.status()['blocks_completed'] > 0:
                planner = getattr(self.owner.bridge, 'plan_from_completed_observation', None)
                if planner is not None:
                    return planner()
            return self.owner.bridge.plan()
        if self._uses_vertical_pinch_geometry():
            return self.owner.bridge.plan(command_builder=self._vertical_pinch_builder,
                                          action_source='v21_actor_execution_geometry')
        return self.owner.bridge.plan()

    def _centroid_tracking_block(self):
        """Execute one Actor block only when C-minus-M is available.

        The Actor still produces all arm motion.  This wrapper prevents an air
        close while the same target it received is far from the jaws, then uses
        the next actual image to decide whether another block is permitted.
        """
        previous = self.centroid_tracking or {}
        if previous and previous.get('accepted') is False:
            raise RuntimeError('previous centroid-tracking block did not reduce error; HOME and inspect before retrying')
        tracking = {}

        def build(state, offsets):
            before = centroid_error_px(state.object_relative[0].detach().cpu().numpy())
            raw = self.owner.runtime.core.act(state, self.owner.runtime.text_ids, offsets).commands_position
            gated, close = gate_closure(state.q[0].detach().cpu().numpy(), raw[0].detach().cpu().numpy(), before)
            tracking.update(before_error_px=before, close=close)
            return torch.as_tensor(gated, device=raw.device, dtype=raw.dtype)[None]

        build.wants_command_offsets = True
        self._clear_stop_for_current_operation()
        commands = self.owner.bridge.plan(command_builder=build, action_source='v21_actor_centroid_gated')
        self._record_plan(commands)
        result = self.owner.bridge.execute()
        after = centroid_error_px(self.owner.runtime.state.object_relative[0].detach().cpu().numpy())
        record = centroid_progress(tracking['before_error_px'], after)
        record.update(action_source='v21_actor_centroid_gated', close=tracking['close'],
                      physical_grasp_verified=False, physical_place_verified=False)
        result['centroid_tracking'] = record
        self.centroid_tracking = record
        self._save_execution(result)
        return result

    def _carry_visual_probe(self, joint, direction):
        if type(joint) is not int or joint not in (0,1,2,3) or type(direction) is not int or direction not in (-1,1):
            raise ValueError('carry visual probe needs joint 0/1/2/3 and direction +/-1')
        record={}
        def current_target():
            targets=(self.targets.get('targets') or {})
            fish=(targets.get('pickup') or {}).get('column') or {}; box=(targets.get('drop_zone') or {}).get('column') or {}
            if fish.get('source')!='operator_colour_body' or fish.get('cog_px') is None or not box.get('visible') or box.get('cog_px') is None:
                raise RuntimeError('current Column fish body and box target are required')
            c=np.asarray(fish['cog_px'],np.float64); goal=np.asarray(box['cog_px'],np.float64)+np.array([0.,-self.CARRY_TARGET_ABOVE_PX])
            if c.shape!=(2,) or goal.shape!=(2,) or not np.isfinite([c,goal]).all():
                raise RuntimeError('invalid Column carry target')
            return c,goal
        def build(state, offsets):
            self._publish(); q=state.q[0].detach().cpu().numpy();v=assistance_cue(self.owner)
            holding=bool(v is not None and v[0]>0 and q[5]<15. and float(v[4])*224.>80. and abs(float(v[2]))*224.<45. and abs(float(v[3]))*224.<60.)
            high=bool(-55.<=q[1]<=-35. and q[2]>=55. and 40.<=q[3]<=50.)
            if not (holding and high):
                raise RuntimeError('carry visual probe requires held fish at carry height')
            before,goal=current_target();action=state.q[:,None].repeat(1,5,1).clone()
            ramp=torch.linspace(.2,1.,5,device=action.device,dtype=action.dtype)
            action[0,:,joint]=state.q[0,joint]+float(direction)*3.*ramp;action[0,:,5]=min(float(q[5]),self.GROUNDING_CLOSED_Q6)
            record.update(joint=joint,direction=direction,before_c_px=before.tolist(),goal_c_px=goal.tolist(),
                          before_error_px=float(np.linalg.norm(goal-before)),physical_grasp_verified=True,physical_place_verified=False)
            return action
        build.wants_command_offsets=True;self._clear_stop_for_current_operation()
        commands=self.owner.bridge.plan(command_builder=build,action_source='fcpg_carry_visual_probe')
        self._record_plan(commands);result=self.owner.bridge.execute();self._publish()
        after,goal=current_target();record.update(after_c_px=after.tolist(),after_error_px=float(np.linalg.norm(goal-after)),
            improvement_px=float(record['before_error_px']-np.linalg.norm(goal-after)))
        result['fcpg_carry_probe']=record;self._save_execution(result);return result

    def _grounding_calibrate(self,delta_j1=2.):
        """Pair fresh exposure-time q and image changes; never substitute requested motion."""
        if not np.isfinite(delta_j1) or not 0 < abs(delta_j1) <= 2.:
            raise ValueError('calibration probe must be nonzero and at most two position units')
        samples=[]
        def sample():
            obs=self.owner.last_endpoint.observation
            value=assistance_cue(self.owner)
            if not bool(obs.measurement_available[0,0]) or value is None or value[0]<=0:
                raise RuntimeError('current Wrist object observation is required')
            return (obs.q_at_image[0,0].detach().cpu().numpy().copy(),
                    float(obs.image_times[0,0]),float(value[2])*224.)
        def builder(state):
            samples.append(sample())
            q=state.q[0]; ramp=torch.linspace(.2,1.,5,device=q.device,dtype=q.dtype)
            action=q[None,None,:].repeat(1,5,1).clone(); action[0,:,0]=action[0,:,0]+float(delta_j1)*ramp
            return action
        self._run_alignment_motion(builder,'visual_alignment_probe')
        before=samples[-1];after=sample()
        dq=after[0]-before[0];tolerance=self.owner.io.settle_step_tolerance
        if after[1] <= before[1] or abs(dq[0]) <= tolerance or np.max(np.abs(dq[1:])) > tolerance:
            raise RuntimeError('insufficient or mixed actual probe motion; gain is not updated')
        moved=float(dq[0]);dpx=after[2]-before[2];px_per_unit=dpx/moved
        if not np.isfinite(px_per_unit) or abs(px_per_unit) < 1e-6:
            raise RuntimeError('no finite measured image response; gain is not updated')
        self.grounding_px_per_unit=float(px_per_unit)
        self.grounding_record=dict(calibration=dict(delta_j1=float(delta_j1),j1_moved=moved,
            image_time_before_s=before[1],image_time_after_s=after[1],dx_before_px=before[2],
            dx_after_px=after[2],px_per_unit=self.grounding_px_per_unit,
            physical_response_verified=False,requested_motion_used_as_readback=False))
        return self.grounding_record

    def _run_alignment_motion(self,builder,source):
        from .v21_wrist_guard import require_wrist_path, require_wrist_readback
        column_dino_probe = source == 'column_dino_fg_probe'
        if not self.alignment_motion_enabled and not (
                column_dino_probe and self.column_dino_fg_calibration_enabled):
            raise RuntimeError('explicit auxiliary motion mode is not enabled')
        if source not in ('visual_alignment_probe','visual_alignment_return','visual_alignment_servo',
                          'visual_alignment_open_departure','visual_alignment_view_step','visual_alignment_reach',
                          'column_dino_fg_probe'):
            raise ValueError('auxiliary commands cannot be labelled as raw Actor output')
        self.owner.bridge.last_attempt=None
        # A newly requested operation may clear the old STOP, but a STOP arriving
        # during ANY preparation/read/persistence must never be cleared again.
        self._clear_stop_for_current_operation()
        reserved=False
        previous_guard=self.owner.bridge.extra_readback_check
        def guarded_readback(q):
            require_wrist_readback(q)
            if previous_guard is not None:previous_guard(q)
        self.owner.bridge.extra_readback_check=guarded_readback
        try:
            if not self.auxiliary_budget.active:
                target=self._home_target()
                q=self.owner.read_checked()
                if np.max(np.abs(q-target)) > self.owner.io.settle_position_tolerance:
                    raise RuntimeError('independent auxiliary experiment must start at HOME')
            if self.owner.stop_event.is_set():
                raise RuntimeError('auxiliary motion cancelled during preparation')
            self.auxiliary_budget.reserve()
            reserved=True
            # Persist the reservation before any motor dispatch. A restart cannot
            # silently erase an in-flight block or resume without its actual state.
            self._persist_auxiliary_budget()
            before=[];replans=[]
            wants_offsets=bool(getattr(builder,'wants_command_offsets',False))
            def bind(state,command_offsets=None):
                commands=builder(state,command_offsets) if wants_offsets else builder(state)
                require_wrist_path(state.q[0].detach().cpu().numpy(),
                                   commands[0].detach().cpu().numpy())
                before.append(self._external_alignment_observation())
                return commands
            bind.wants_command_offsets=wants_offsets
            for attempt in range(3):
                if self.owner.stop_event.is_set():
                    raise RuntimeError('auxiliary motion cancelled before dispatch')
                self.owner.bridge.last_attempt=None
                commands=self.owner.bridge.plan(command_builder=bind,action_source=source)
                self._record_plan(commands)
                try:
                    result=self.owner.bridge.execute()  # Same clocks and physical guards.
                    break
                except PreExecutionStateChanged as exc:
                    info=self.owner.bridge.last_attempt or {}
                    if (attempt==2 or self.owner.stop_event.is_set()
                            or info.get('policy_commands_attempted')!=0
                            or info.get('policy_commands_returned')!=0):
                        raise
                    replans.append(dict(exc.diagnostic,no_policy_commands_sent=True))
            result['pre_dispatch_replans']=replans
            result['external_task_before']=before[-1]
            self._save_execution(result)
            result['external_task_after']=self._external_alignment_observation()
            self.last_execution=deepcopy(result)
            return result
        except BaseException:
            self.owner.stop_event.set()
            raise
        finally:
            self.owner.bridge.extra_readback_check=previous_guard
            if reserved:
                self.auxiliary_budget.finish(self.owner.bridge.last_attempt)
                self._persist_auxiliary_budget()

    def _persist_auxiliary_budget(self):
        write_json(self.output/f'auxiliary-budget-{self.auxiliary_budget.trial_id}.json',
                   self.auxiliary_budget.status())

    def _alignment_departure(self):
        """At most two metered open-jaw Actor setup blocks, each reobserved and stopped."""
        if not self.alignment_motion_enabled:
            raise RuntimeError('visual alignment motion is not enabled')
        budget=self.auxiliary_budget.status()
        if budget['active'] and (len(budget['records'])>=2 or any(
                row.get('action_source')!='visual_alignment_open_departure' for row in budget['records'])):
            raise RuntimeError('view-acquisition departure is limited to the first two blocks after HOME')
        if self.grounding_correction or self.head_grasp:
            raise RuntimeError('open-jaw departure cannot run the legacy grasp/carry controller')
        self._alignment_session=None
        proposals=[]
        def build(state,offsets):
            q=state.q
            if float(q[0,1])>self.GROUNDING_UNFOLDED_J2 or float(q[0,5])<=self.GROUNDING_OPEN_Q6:
                raise RuntimeError('view acquisition requires folded HOME and already open jaws')
            core=self.owner.runtime.core
            if getattr(core.actor,'horizon',5)==40:
                offsets40=torch.arange(40,device=q.device,dtype=torch.float64)[None]*.1
                full=core.act(state,self.owner.runtime.text_ids,offsets40).commands_position
                if full.shape!=(1,40,6) or not torch.isfinite(full).all():
                    raise RuntimeError('nonfinite or malformed native40 Actor departure')
                original=full[:,:5].contiguous()
            else:
                full=None
                original=core.act(state,self.owner.runtime.text_ids,offsets).commands_position
            if original.shape!=(1,5,6) or not torch.isfinite(original).all():
                raise RuntimeError('nonfinite or malformed Actor departure')
            if float(original[0,-1,1])<=float(q[0,1])+.25:
                raise RuntimeError('Actor departure does not unfold from HOME')
            commands=original.detach().clone()
            commands[:,:,5]=q[:,None,5]
            proposals.append(dict(original_actor_commands=original.detach().cpu().tolist(),
                inferred_steps=40 if full is not None else 5,execute_steps=5,
                actual_source_q_position=q.detach().cpu().tolist(),
                intervention='hold_existing_gripper_open_only',
                arm_commands_unchanged=True,physical_grasp_verified=False))
            return commands
        build.wants_command_offsets=True
        result=self._run_alignment_motion(build,'visual_alignment_open_departure')
        result['view_acquisition']=proposals[-1]
        result['view_acquisition']['completed']=bool(result.get('completed'))
        self.last_execution=deepcopy(result)
        write_json(self.output/f'view-acquisition-{time.time_ns()}.json',result)
        return result

    def _close_auxiliary_after_home(self):
        identity_floor=self.auxiliary_budget.identity_floor
        maximum_blocks=self.auxiliary_budget.maximum_blocks
        if self.auxiliary_budget.active:
            write_json(self.output/f'auxiliary-budget-{self.auxiliary_budget.trial_id}.json',
                       self.auxiliary_budget.closed_after_home_status())
        self.auxiliary_budget=AuxiliaryMotionBudget(identity_floor=identity_floor,maximum_blocks=maximum_blocks)

    def _alignment_view_step(self):
        """One high-clearance orientation/forward-view probe, not grasp or carry."""
        from dataclasses import replace
        from .v21_view_feedback import tracking_evidence,compensate
        if not self.alignment_motion_enabled or self.grounding_correction or self.head_grasp:
            raise RuntimeError('view step requires independent visual alignment mode')
        owner=self.owner
        high_gate=replace(owner.clearance,planned_clearance_mm=max(20.,owner.clearance.planned_clearance_mm),
                          readback_clearance_mm=max(20.,owner.clearance.readback_clearance_mm))
        previous_guard=owner.bridge.extra_readback_check
        self._alignment_session=None  # J234 changes invalidate the previous lateral fit.
        plans=[]
        previous_execution=deepcopy(self.last_execution)
        def build(state):
            q=state.q[0].cpu().numpy()
            if q[5]<=self.GROUNDING_OPEN_Q6 or q[1]<=self.GROUNDING_UNFOLDED_J2:
                raise RuntimeError('view step needs open jaws and an unfolded arm')
            high_gate.require_readback(q,label='high view source')
            solver=VerticalApproachSolver(owner.kinematics,owner.clearance,
                owner.runtime.core.actor.position_lower.cpu().numpy(),
                owner.runtime.core.actor.position_upper.cpu().numpy(),vertical_tolerance_deg=10.)
            follow=(previous_execution or {}).get('normal_follow') or {}
            old_follow=np.asarray(follow.get('positions',[]),np.float64)
            source_changed=(old_follow.ndim==2 and old_follow.shape[1:]==(6,) and len(old_follow)
                            and np.isfinite(old_follow[-1]).all() and np.abs(q-old_follow[-1]).max()>.25)
            # The optional bias is only valid at a matched pose. Fresh high-view
            # planning does not require a stale bias to remain usable.
            evidence=(dict(usable_offset=False,recovering=False,reason='old_shortfall_pose_changed')
                      if source_changed else tracking_evidence(q,previous_execution))
            if evidence is not None and evidence['recovering']:
                target=q.copy();target[1:4]=evidence['nominal_target'][1:4]
                nominal=q+np.linspace(.2,1.,5)[:,None]*(target-q)
                diagnostic=dict(stage='tracking_recovery',source=approach_geometry(owner.kinematics,q),
                    nominal_commands_position=nominal.tolist(),endpoint=approach_geometry(owner.kinematics,target))
            else:
                # One bounded high-view acquisition, not many tiny stop/start
                # probes. All five waypoints retain the same 6-unit step and
                # high-clearance constraints, including after compensation.
                proposal=solver.propose_view_chunk(q,radial_probe_mm=30.)
                if not proposal.accepted:raise RuntimeError('view step rejected: '+proposal.diagnostic['reason'])
                nominal=proposal.commands_position;diagnostic=deepcopy(proposal.diagnostic)
            commands=nominal
            if evidence is not None and evidence.get('usable_offset',True):
                commands,correction=compensate(q,nominal,evidence,solver.lower,solver.upper)
                diagnostic.update(tracking_correction=correction,nominal_commands_position=nominal.tolist())
            elif evidence is not None:
                diagnostic['discarded_tracking_correction']=deepcopy(evidence)
            high_gate.require_policy_path(q,commands)
            diagnostic['issued_commands_position']=commands.tolist()
            plans.append(diagnostic)
            return torch.as_tensor(commands,device=state.q.device,dtype=state.q.dtype)[None]
        owner.bridge.extra_readback_check=lambda q:high_gate.require_readback(q,label='high view actual readback')
        try:
            result=self._run_alignment_motion(build,'visual_alignment_view_step')
            result['view_step']=dict(plan=plans[-1],actual_endpoint_geometry=approach_geometry(
                owner.kinematics,owner.last_endpoint.q[0].cpu().numpy()),
                actual_endpoint_clearance_mm=float(owner.clearance.clearance_mm(owner.last_endpoint.q[0].cpu().numpy())),
                actual_metric_geometry_independently_verified=False,physical_grasp_verified=False)
            self.last_execution=deepcopy(result)
            write_json(self.output/f'view-step-{time.time_ns()}.json',result)
            return result
        finally:
            owner.bridge.extra_readback_check=previous_guard

    def _external_alignment_observation(self):
        endpoint=self.owner.last_endpoint
        snap=self.snapshot
        if (snap is None or snap['time']!=endpoint.time.item()
                or not bool(endpoint.observation.measurement_available[0,1])):
            return dict(available=False,reason='no matched current External observation')
        targets=self.targets.get('targets') or {}
        return dict(available=True,image_time_s=float(endpoint.observation.image_times[0,1]),
            q_at_image=endpoint.observation.q_at_image[0,1].cpu().tolist(),
            pickup=deepcopy((targets.get('pickup') or {}).get('column')),
            drop=deepcopy((targets.get('drop_zone') or {}).get('column')),
            head_candidate=deepcopy(self.head_targets.get('column')),
            control_use=False,semantic_head_identity_verified=False)

    def _alignment_reach(self):
        from dataclasses import replace
        if not self.alignment_motion_enabled or self.alignment_scope!='lateral':
            raise RuntimeError('high-view reach needs the independent lateral controller')
        gate=replace(self.owner.clearance,planned_clearance_mm=max(20.,self.owner.clearance.planned_clearance_mm),
                     readback_clearance_mm=max(20.,self.owner.clearance.readback_clearance_mm))
        old=self.owner.bridge.extra_readback_check
        original=self._alignment_controller().run_motion
        def run(builder,source):
            def checked(state):
                commands=builder(state)
                gate.require_policy_path(state.q[0].cpu().numpy(),commands[0].cpu().numpy())
                return commands
            return self._run_alignment_motion(checked,source)
        self.owner.bridge.extra_readback_check=lambda q:gate.require_readback(q,label='high lateral actual')
        self._alignment_controller().run_motion=run
        try:return self._alignment_controller().advance()
        finally:
            self.owner.bridge.extra_readback_check=old
            self._alignment_controller().run_motion=original

    def _alignment_controller(self):
        if self._alignment_session is None:
            cls=LateralAlignmentSession if self.alignment_scope=='lateral' else AlignmentSession
            self._alignment_session=cls(self._alignment_measure,self._run_alignment_motion,
                lower=established.SO101_POSITION_LOWER,upper=established.SO101_POSITION_UPPER,
                gripper_tolerance=self.owner.io.settle_step_tolerance)
        return self._alignment_session

    def handle(self, action, payload=None):
        if action == 'trial':
            with self._trial_admission_lock:
                admission_generation = self._trial_stop_generation
        if action == 'stop':
            self.stop()
            return self.transport_status()
        diagnostic = getattr(self, 'j3_pid_diagnostic', None)
        if diagnostic is not None:
            diagnostic.require_motion_unsealed(action)
        same_goal = getattr(self, 'same_goal_diagnostic', None)
        if same_goal is not None:
            same_goal.require_motion_unsealed(action)
        if self.trial.running and not self.trial.owns_current_thread():
            raise RuntimeError('capped trial owns motion; STOP remains available')
        if action in ('execute','trial','target-track','carry-visual-probe') and self.auxiliary_budget.active:
            raise RuntimeError('auxiliary experiment is active; HOME before switching back to Actor')
        if action == 'trial':
            admission_cancelled = None
            operation_acquired = False
            try:
                if not self.operation_lock.acquire(blocking=False):
                    raise RuntimeError('another operation is active; STOP remains available')
                operation_acquired = True
                if diagnostic is not None:
                    diagnostic.require_motion_unsealed(action)
                if same_goal is not None:
                    same_goal.require_motion_unsealed(action)
                # Another request may have started after the outer running check.
                # A rejected request must never replace that trial's STOP token.
                if self.trial.running or (self.trial.thread is not None and self.trial.thread.is_alive()):
                    raise RuntimeError('capped trial owns motion; STOP remains available')
                with self._trial_admission_lock:
                    if self._trial_admission_in_progress:
                        raise RuntimeError('another trial admission is active; STOP remains available')
                    admission_cancelled = threading.Event()
                    self._trial_admission_in_progress = True
                    self._trial_admission_cancelled = admission_cancelled
                    # Cover STOP before publication, including during lock acquisition.
                    if admission_generation != self._trial_stop_generation:
                        admission_cancelled.set()
                if admission_cancelled.is_set():
                    raise RuntimeError('capped trial admission cancelled by STOP')
                if self.grounding_correction or self.head_grasp or self.alignment_motion_enabled:
                    raise RuntimeError('capped trial uses the unmodified Actor only')
                if self.owner.runtime.pending is not None or self.owner.runtime.executing:
                    raise RuntimeError('finish or stop the pending transaction before a trial')
                if self.owner.runtime.state is None or bool(self.owner.runtime.state.history.valid.any()):
                    raise RuntimeError('explicit HOME and object reset required before each trial')
                target=self._home_target()
                q=self.owner.read_checked()
                if np.max(np.abs(q-target))>self.owner.io.settle_position_tolerance:
                    raise RuntimeError('capped trial must start at the existing HOME pose')
                if admission_cancelled.is_set():
                    raise RuntimeError('capped trial admission cancelled by STOP')
                if self.camera_condition.address != 'factual':
                    raise RuntimeError('scored trials use factual address only')
                trial_source = (self.c4.source if self.c4 is not None
                                else 'v21_lewm_cem' if self.lewm_cem else 'v21_actor')
                self.trial.start(paused=True,
                    action_source=trial_source,
                    camera_condition=self.camera_condition.name,
                    address_mode=self.camera_condition.address,
                    reset_pose=np.asarray(q, dtype=float).tolist())
                self.operation_lock.release()
                operation_acquired = False
                # No slow precheck or persistence holds this short lock. STOP
                # and gate release have one ordering, without waiting on IO.
                with self._trial_admission_lock:
                    if admission_cancelled.is_set():
                        self.trial.cancel()
                        raise RuntimeError('capped trial admission cancelled by STOP')
                    self.trial.release_start()
                return self.transport_status()
            finally:
                if operation_acquired:
                    self.operation_lock.release()
                if admission_cancelled is not None:
                    with self._trial_admission_lock:
                        self._trial_admission_in_progress = False
        alignment_actions=('alignment-observe','alignment-probe','alignment-return','alignment-fit',
                           'alignment-step','alignment-reset','alignment-departure','alignment-view-step','alignment-reach')
        if action not in ('observe', 'preview', 'candidate-preview','motion-validation-probe','motion-validation-return', 'plan', 'execute', 'target-track','carry-visual-probe', 'home','camera-condition','actuator-observe','auxiliary-budget','gripper-follow-check','grounding-calibrate','column-dino-fg-probe','j3-pid-baseline','j3-pid-variant','j3-range-diagnostic','j3-compensation-diagnostic','j3-native-response-diagnostic','same-goal-p32','same-goal-i1','fixed-goal-compensate',*alignment_actions):
            raise ValueError('unknown controller operation')
        # The idle readback monitor briefly owns this same lock. A following
        # plan/execute/home waits that read out instead of aborting the chunk.
        if action in ('plan', 'execute', 'home'):
            acquired = self.operation_lock.acquire(timeout=2.)
        elif action in ('actuator-observe', 'observe'):
            acquired = self.operation_lock.acquire(timeout=.25)
        else:
            acquired = self.operation_lock.acquire(blocking=False)
        if not acquired:
            raise RuntimeError('another operation is active; STOP remains available')
        try:
            # Recheck under operation ownership. Rejected admission has not
            # moved hardware and must not enter motion-failure HOLD cleanup.
            if diagnostic is not None:
                diagnostic.require_motion_unsealed(action)
            if same_goal is not None:
                same_goal.require_motion_unsealed(action)
        except BaseException:
            self.operation_lock.release()
            raise
        execute_attempt=None
        operation_error=None
        try:
            self.error = None; self.phase = action
            if action == 'fixed-goal-compensate':
                if payload:raise ValueError('fixed-goal compensation accepts no parameters')
                from .v21_fixed_goal_service import handle_fixed_goal_compensation
                handle_fixed_goal_compensation(self)
            elif action in ('same-goal-p32', 'same-goal-i1'):
                if payload:raise ValueError('same-goal diagnostic accepts no parameters')
                if action != 'same-goal-' + self.same_goal_diagnostic.mode:
                    raise ValueError('same-goal request does not match the enabled diagnostic mode')
                self.same_goal_diagnostic.handle()
            elif action == 'j3-native-response-diagnostic':
                if payload:raise ValueError('native response diagnostic accepts no parameters')
                self.j3_native_response_diagnostic.handle()
            elif action == 'j3-compensation-diagnostic':
                if payload:
                    raise ValueError('fixed J3 compensation diagnostic accepts no parameters')
                self.j3_compensation_diagnostic.handle()
            elif action == 'j3-range-diagnostic':
                if payload:
                    raise ValueError('fixed J3 range diagnostic accepts no parameters')
                self.j3_range_diagnostic.handle()
            elif action in ('j3-pid-baseline', 'j3-pid-variant'):
                if payload:
                    raise ValueError('fixed J3 diagnostic accepts no parameters')
                if diagnostic is None:
                    raise RuntimeError('J3 diagnostic is disabled')
                diagnostic.handle('baseline' if action == 'j3-pid-baseline' else 'variant')
            elif action in ('motion-validation-probe','motion-validation-return'):
                from .v21_motion_validation import run_probe
                rt=self.owner.runtime
                if not rt.stopped or rt.pending is not None or rt.executing:
                    raise RuntimeError('motion validation requires STOP and no pending command')
                if not self.alignment_motion_enabled:
                    raise RuntimeError('explicit diagnostic motion mode required')
                if action=='motion-validation-probe':
                    if not isinstance(payload,dict) or not {'joint','direction'} <= set(payload) or set(payload)-{'joint','direction','amplitude'}:
                        raise ValueError('diagnostic accepts joint/direction only')
                    if (self.motion_validation_record or {}).get('return_pending'):
                        raise RuntimeError('return from the previous diagnostic before another probe')
                    with torch.inference_mode(): run_probe(self,payload['joint'],payload['direction'],amplitude=payload.get('amplitude',2.))
                else:
                    if payload: raise ValueError('return accepts no custom command')
                    previous=self.motion_validation_record or {}
                    with torch.inference_mode(): run_probe(self,previous.get('joint'),previous.get('direction'),returning=True)
                write_json(self.output/f'motion-validation-{time.time_ns()}.json',self.motion_validation_record)
            elif action=='candidate-preview':
                from .v21_candidate_preview import preview_actor_candidates
                self.candidate_preview_record = None
                self.candidate_preview_record = preview_actor_candidates(self)
            elif action=='actuator-observe':
                if payload:raise ValueError('actuator observation takes no parameters')
                rt=self.owner.runtime
                if not rt.stopped or rt.executing or rt.pending is not None:
                    raise RuntimeError('actuator diagnostic requires STOP and no pending motion')
                from .v21_hardware_state import pre_dispatch_actuator_state, actuator_diagnostics
                from .v21_actuator_history import history_for
                from .servo_load_telemetry import make_servo_load_telemetry, read_servo_load_registers
                self.owner._lease_check()
                with self.owner.bus_lock:
                    bus=self.owner.io.follower.bus
                    self.actuator_state=pre_dispatch_actuator_state(bus,observation_phase='stopped_observation')
                    self.actuator_state['owner_observation_history']=history_for(self.owner).snapshot(
                        self.actuator_state['finished_s'])
                    # An explicitly requested diagnostic may take the two
                    # additional group reads.  This is observability only:
                    # it never changes a command, a gate, or motor settings.
                    raw_load=read_servo_load_registers(bus)
                    self.actuator_state['diagnostics']=actuator_diagnostics(bus)
                self.actuator_state['servo_load_telemetry']=make_servo_load_telemetry(
                    present_load=raw_load.get('present_load'),
                    present_current=raw_load.get('present_current'),
                    timestamp_s=float(raw_load['timestamp_s']),
                    wall_time_unix_s=float(raw_load['wall_time_unix_s']),
                    read_duration_ms=float(raw_load['read_duration_ms']),
                    round_id='actuator-observe',phase='diagnostic',
                    load_error=raw_load.get('load_error'),current_error=raw_load.get('current_error'),
                    extra_sync_reads=int(raw_load['extra_sync_reads']),
                    provenance='explicit_read_only_actuator_observe',
                ).to_dict()
                write_json(self.output/f'actuator-observe-{time.time_ns()}.json',self.actuator_state)
            elif action=='auxiliary-budget':
                if payload!={'maximum_blocks':30} or type(payload.get('maximum_blocks')) is not int:
                    raise ValueError('explicit budget request must be maximum_blocks=30')
                rt=self.owner.runtime
                if not rt.stopped or rt.executing or rt.pending is not None:
                    raise RuntimeError('budget change requires STOP with no pending motion')
                if not self.auxiliary_budget.active:
                    raise RuntimeError('extend the existing trial; do not invent or reset its counters')
                before=self.auxiliary_budget.status()
                previous_maximum=self.auxiliary_budget.maximum_blocks
                self.auxiliary_budget.extend_to_thirty()
                try:
                    self._persist_auxiliary_budget()
                except BaseException:
                    self.auxiliary_budget.maximum_blocks=previous_maximum
                    raise
                write_json(self.output/f'budget-extension-{time.time_ns()}.json',dict(
                    kind='operator_budget_extension',before=before,after=self.auxiliary_budget.status(),
                    motor_commands_sent=0,task_success_verified=False))
            elif action in ('observe', 'preview'):
                if self.owner.runtime.pending is not None:
                    raise RuntimeError('stop or execute the pending plan before a new observation')
                with torch.inference_mode():
                    if action == 'observe':
                        self._observe()
                    else:
                        endpoint = self.owner.capture()
                        self.preview_record = None
                        try:
                            discard = getattr(self.owner, 'discard_proposed_camera_condition', None)
                            if discard is not None:
                                discard()
                            if self.grounding_correction:
                                proposal = self._assistance.propose(lambda: self.owner.runtime.preview(
                                    endpoint, command_builder=self._grounding_builder))
                                commands, forecast, admission = proposal.result
                                self._assistance.abort(proposal.proposal_id)
                                self._assistance_proposal_id = None
                                preview_source='v21_actor_grounding_corrected'
                            elif self.lewm_cem:
                                commands, forecast, admission = self.owner.runtime.preview(
                                    endpoint, command_builder=self._lewm_cem_builder)
                                preview_source='v21_lewm_cem'
                            elif self._uses_vertical_pinch_geometry():
                                commands, forecast, admission = self.owner.runtime.preview(
                                    endpoint, command_builder=self._vertical_pinch_builder)
                                preview_source='v21_actor_execution_geometry'
                            else:
                                commands, forecast, admission = self.owner.runtime.preview(endpoint)
                                preview_source='v21_actor'
                        finally:
                            self._refresh_status(); self._publish()
                        self.preview_record = dict(self._forecast_record(commands, forecast),
                            executable=False, memory_committed=False, assistance_state_committed=False,
                            action_source=preview_source,
                            mechanical_admission=admission, observation_sequence=self.sequence)
                        if self.lewm_cem:
                            self.preview_record['lewm_cem'] = deepcopy(self.lewm_cem_record)
                        if self.lewm_controller is not None:
                            self.preview_record['lewm_controller'] = deepcopy(self.lewm_controller)
                        # The images are real, but a UI preview does not evict
                        # committed frames or overwrite execution feedback.
            elif action == 'plan' and self._home_after_release:
                # The object was released over the drop zone: the task is complete. Go HOME
                # directly instead of letting the policy wander with an empty gripper.
                self._home_after_release=False
                with torch.inference_mode(): self._home()
                self.task_complete_record=dict(homed_after_release=True, sequence=self.sequence)
                self._refresh_status(); self._publish()
            elif action == 'plan':
                self._clear_stop_for_current_operation()
                with torch.inference_mode():
                    commands = self._plan_with_policy(reuse_completed=self.trial.owns_current_thread())
                    if self.owner.stop_event.is_set():
                        self.owner.runtime.stop()
                        self._assistance.abort()
                        self._assistance_proposal_id = None
                        self.plan_record = None
                    else:
                        self._record_plan(commands); self._refresh_status()
                        if getattr(self.owner.bridge, 'last_plan_observation_source', None) != 'completed_transaction_endpoint':
                            self._publish()
                    # The interactive plan becomes available only after its
                    # audit/diagnostic publication. This is a readiness clock,
                    # never a replacement for the real source/image clocks.
                    rt = self.owner.runtime
                    with rt.lock:
                        if rt.pending is not None and not rt.stopped:
                            rt.pending_ready_time_s = float(self.clock())
                            self.plan_record['plan_ready_time_s'] = rt.pending_ready_time_s
                    self._refresh_status()
            elif action == 'execute':
                if self.owner.runtime.pending is None or self.owner.runtime.stopped or self.owner.stop_event.is_set():
                    raise RuntimeError('explicit plan is required before execution')
                with torch.inference_mode():
                    self._alignment_session=None
                    self.owner.bridge.last_attempt=None
                    # Interactive execution still gets fresh images. The
                    # in-process continuous worker has JUST prepared a plan;
                    # do not replace its genuine completion observation again.
                    replans=[]
                    for attempt in range(3):
                        if self.owner.stop_event.is_set():
                            raise RuntimeError('execution cancelled before fresh planning')
                        pending = self.owner.runtime.pending
                        age = None if pending is None else self.clock()-pending.source_time.item()
                        just_prepared = (self.trial.owns_current_thread() and age is not None and 0 <= age <= .5)
                        if self.lewm_cem:
                            if getattr(self.owner.runtime, 'pending_action_source', None) != 'v21_lewm_cem':
                                raise RuntimeError('LeWM CEM cannot fall back to Actor')
                            if (attempt == 0 and pending is not None
                                    and age is not None
                                    and 0 <= age <= self.owner.runtime.max_plan_age_s
                                    and self.owner.runtime.pending_is_recently_ready()):
                                just_prepared = True
                        if self.replan_on_execute and (attempt or not just_prepared):
                            commands = self._plan_with_policy()
                            self._record_plan(commands)
                        assistance_proposal_id = self._assistance_proposal_id
                        try:
                            previous_attempt=self.owner.bridge.last_attempt
                            _elbow_p_restore=False
                            _wrist_p_restore=False
                            try:
                                if self.lewm_cem:
                                    # region agent log
                                    _bus=self.owner.io.follower.bus
                                    with self.owner.bus_lock:
                                        _p_before=int(_bus.read('P_Coefficient','elbow_flex',normalize=False,num_retry=4))
                                        _bus.write('P_Coefficient','elbow_flex',32,normalize=False,num_retry=0)
                                        _p_set=int(_bus.read('P_Coefficient','elbow_flex',normalize=False,num_retry=4))
                                    _elbow_p_restore=True
                                    try:
                                        import json as _json, time as _time
                                        with open('/home/james/feedback-corrected-grounding/.cursor/debug-75d952.log','a') as _f:
                                            _f.write(_json.dumps({'sessionId':'75d952','runId':'post-fix','hypothesisId':'K','location':'v21_live_service.py:execute','message':'elbow P32 for this chunk','data':{'before':_p_before,'written':_p_set},'timestamp':int(_time.time()*1000)})+'\n')
                                    except Exception:
                                        pass
                                    # endregion
                                    if _p_set!=32:
                                        raise RuntimeError('elbow P32 write did not read back')
                                    _wrist_names=('wrist_flex','wrist_roll')
                                    _wrist_before={}; _wrist_set={}
                                    with self.owner.bus_lock:
                                        for _name in _wrist_names:
                                            _wrist_p_restore=True
                                            _wrist_before[_name]=int(_bus.read('P_Coefficient',_name,normalize=False,num_retry=4))
                                            _bus.write('P_Coefficient',_name,32,normalize=False,num_retry=0)
                                            _wrist_set[_name]=int(_bus.read('P_Coefficient',_name,normalize=False,num_retry=4))
                                    # region agent log
                                    try:
                                        import json as _json, time as _time
                                        with open('/home/james/feedback-corrected-grounding/.cursor/debug-75d952.log','a') as _f:
                                            _f.write(_json.dumps({'sessionId':'75d952','runId':'post-fix','hypothesisId':'M','location':'v21_live_service.py:execute','message':'wrist P32 for this chunk','data':{'before':_wrist_before,'written':_wrist_set},'timestamp':int(_time.time()*1000)})+'\n')
                                    except Exception:
                                        pass
                                    # endregion
                                    if any(value!=32 for value in _wrist_set.values()):
                                        raise RuntimeError('wrist P32 write did not read back')
                                self.owner.bridge.extra_position_commands = getattr(self, '_horizon40_extension', None)
                                self.owner.bridge.extra_position_note = getattr(self, '_horizon40_note', None)
                                self._horizon40_extension = None
                                result = self.owner.bridge.execute()
                            finally:
                                if _elbow_p_restore or _wrist_p_restore:
                                    _bus=self.owner.io.follower.bus
                                    _restored={}
                                    with self.owner.bus_lock:
                                        if _wrist_p_restore:
                                            for _name in ('wrist_flex','wrist_roll'):
                                                _bus.write('P_Coefficient',_name,16,normalize=False,num_retry=0)
                                                _restored[_name]=int(_bus.read('P_Coefficient',_name,normalize=False,num_retry=4))
                                        if _elbow_p_restore:
                                            _bus.write('P_Coefficient','elbow_flex',16,normalize=False,num_retry=0)
                                            _p_restored=int(_bus.read('P_Coefficient','elbow_flex',normalize=False,num_retry=4))
                                            _restored['elbow_flex']=_p_restored
                                    # region agent log
                                    try:
                                        import json as _json, time as _time
                                        with open('/home/james/feedback-corrected-grounding/.cursor/debug-75d952.log','a') as _f:
                                            _f.write(_json.dumps({'sessionId':'75d952','runId':'post-fix','hypothesisId':'K','location':'v21_live_service.py:execute','message':'elbow P restored','data':{'restored':_p_restored},'timestamp':int(_time.time()*1000)})+'\n')
                                    except Exception:
                                        pass
                                    # endregion
                                    # region agent log
                                    try:
                                        import json as _json, time as _time
                                        with open('/home/james/feedback-corrected-grounding/.cursor/debug-75d952.log','a') as _f:
                                            _f.write(_json.dumps({'sessionId':'75d952','runId':'post-fix','hypothesisId':'M','location':'v21_live_service.py:execute','message':'gain restored','data':{'restored':_restored},'timestamp':int(_time.time()*1000)})+'\n')
                                    except Exception:
                                        pass
                                    # endregion
                                    if _elbow_p_restore and _p_restored!=16:
                                        raise RuntimeError('elbow P16 restoration did not read back')
                                    if _wrist_p_restore and any(_restored.get(_name)!=16 for _name in ('wrist_flex','wrist_roll')):
                                        raise RuntimeError('wrist P16 restoration did not read back')
                                current_attempt=self.owner.bridge.last_attempt
                                if current_attempt is not previous_attempt:
                                    execute_attempt=current_attempt
                            execute_attempt=result
                            if self.lewm_cem:
                                # A normal follow receipt may tolerate a loaded
                                # endpoint that has not moved yet.  CEM must not
                                # call that a completed feedback transition:
                                # its next LeWM state needs an actual response
                                # for every materially commanded arm axis.
                                try:
                                    result['actual_cem_delta'] = require_actual_cem_response(
                                        result,
                                        minimum_motion=float(self.owner.io.settle_step_tolerance),
                                        checked_joint_count=self.lewm_cem_response_joint_count,
                                    ).tolist()
                                except ValueError as exc:
                                    result['response_verified'] = False
                                    result['response_error'] = str(exc)
                                    diagnostic = getattr(exc, 'diagnostic', None)
                                    if diagnostic is not None:
                                        result['actual_cem_response_diagnostic'] = deepcopy(diagnostic)
                                    result['lewm_cem'] = deepcopy(self.lewm_cem_record)
                                    _keep = short_window_keeps_actual_response(result, exc)
                                    # region agent log
                                    try:
                                        import json as _json, time as _time
                                        with open('/home/james/feedback-corrected-grounding/.cursor/debug-75d952.log','a') as _f:
                                            _f.write(_json.dumps({'sessionId':'75d952','hypothesisId':'D','location':'v21_live_service.py:execute','message':'response gate','data':{'keep_short_window':_keep,'budget_s':(result.get('normal_follow') or {}).get('budget_s'),'completed':result.get('completed'),'returned':result.get('policy_commands_returned'),'blocked':(diagnostic or {}).get('blocked_joint_indices'),'commanded_peak':(diagnostic or {}).get('commanded_peak_delta'),'achieved':(diagnostic or {}).get('achieved_directional_motion'),'minimum_motion':(diagnostic or {}).get('minimum_motion')},'timestamp':int(_time.time()*1000)})+'\n')
                                    except Exception:
                                        pass
                                    # endregion
                                    # region agent log
                                    try:
                                        import json as _json, time as _time
                                        from .v21_hardware_state import read_map as _read_map
                                        with self.owner.bus_lock:
                                            _goal = _read_map(self.owner.io.follower.bus, 'Goal_Position')
                                            _present = _read_map(self.owner.io.follower.bus, 'Present_Position')
                                            _moving = _read_map(self.owner.io.follower.bus, 'Moving')
                                        with open('/home/james/feedback-corrected-grounding/.cursor/debug-75d952.log','a') as _f:
                                            _f.write(_json.dumps({'sessionId':'75d952','hypothesisId':'H','location':'v21_live_service.py:execute','message':'elbow registers after follow','data':{'elbow_goal_raw':_goal.get('elbow_flex'),'elbow_present_raw':_present.get('elbow_flex'),'elbow_moving':_moving.get('elbow_flex'),'raw_error':None if _goal.get('elbow_flex') is None or _present.get('elbow_flex') is None else int(_goal['elbow_flex'])-int(_present['elbow_flex'])},'timestamp':int(_time.time()*1000)})+'\n')
                                    except Exception as _exc:
                                        try:
                                            import json as _json, time as _time
                                            with open('/home/james/feedback-corrected-grounding/.cursor/debug-75d952.log','a') as _f:
                                                _f.write(_json.dumps({'sessionId':'75d952','hypothesisId':'H','location':'v21_live_service.py:execute','message':'elbow register read failed','data':{'error':f'{type(_exc).__name__}: {_exc}'},'timestamp':int(_time.time()*1000)})+'\n')
                                        except Exception:
                                            pass
                                    # endregion
                                    if not _keep:
                                        raise
                                    result['short_window_actual_response_retained'] = True
                                else:
                                    result['response_verified'] = True
                            break
                        except PreExecutionStateChanged as exc:
                            info=self.owner.bridge.last_attempt or {}
                            if (not self.replan_on_execute or attempt==2 or self.owner.stop_event.is_set()
                                    or self.trial.cancelled.is_set() and self.trial.running
                                    or info.get('policy_commands_attempted')!=0
                                    or info.get('policy_commands_returned')!=0):
                                raise
                            # Keep the exact-pose/clock checks unchanged. A
                            # different pose gets a NEW observation, Actor
                            # plan, physical audit and LeWM forecast. Never
                            # resend a block after even one attempted write.
                            replans.append(dict(exc.diagnostic,no_policy_commands_sent=True))
                            self._assistance.abort()
                            self._assistance_proposal_id=None
                    result['pre_dispatch_replans']=replans
                    if self.grounding_correction:
                        result['assistance_state_committed'] = self._assistance.commit_completed(
                            assistance_proposal_id, completed=result.get('completed') is True)
                        self._assistance_proposal_id = None
                    self._save_execution(result)
            elif action=='target-track':
                if self.lewm_cem:
                    raise RuntimeError('LeWM CEM cannot fall back to Actor')
                if self.owner.runtime.pending is not None or self.owner.runtime.executing:
                    raise RuntimeError('target tracking requires STOP with no pending command')
                with torch.inference_mode():
                    self._alignment_session=None
                    self._centroid_tracking_block()
            elif action=='carry-visual-probe':
                if self.lewm_cem:
                    raise RuntimeError('LeWM CEM cannot fall back to Actor')
                if self.owner.runtime.pending is not None or self.owner.runtime.executing:
                    raise RuntimeError('carry visual probe requires STOP with no pending command')
                if not isinstance(payload,dict) or set(payload)!={'joint','direction'}:
                    raise ValueError('carry visual probe accepts joint and direction only')
                with torch.inference_mode():
                    self._carry_visual_probe(payload['joint'],payload['direction'])
            elif action=='grounding-calibrate':
                if self.owner.runtime.pending is not None or self.owner.runtime.executing:
                    raise RuntimeError('stop the pending transaction before calibration')
                with torch.inference_mode(): self._grounding_calibrate()
            elif action=='column-dino-fg-probe':
                self._alignment_session=None
                with torch.inference_mode(): self._column_dino_fg_probe(payload)
            elif action in alignment_actions:
                if self.owner.runtime.pending is not None or self.owner.runtime.executing:
                    raise RuntimeError('stop the pending transaction before alignment operations')
                with torch.inference_mode():
                    if action=='alignment-departure':
                        self.alignment=self._alignment_departure()
                    elif action=='alignment-view-step':
                        self.alignment=self._alignment_view_step()
                    elif action=='alignment-reach':
                        self.alignment=self._alignment_reach()
                    elif action=='alignment-reset':
                        self._alignment_session=None;self.alignment=dict(reset=True,physical_grasp_verified=False)
                    elif action=='alignment-observe':
                        self.owner.capture()
                        self.alignment=observation_payload(self._alignment_measure())
                    elif action=='alignment-fit':
                        self.alignment=self._alignment_controller().fit()
                    else:
                        if not self.alignment_motion_enabled:raise RuntimeError('visual alignment motion is not enabled')
                        controller=self._alignment_controller()
                        if action=='alignment-probe':
                            if not isinstance(payload,dict) or set(payload)!={'joint','direction'}:
                                raise ValueError('probe accepts only joint and direction')
                            self.alignment=controller.probe(payload['joint'],payload['direction'])
                        elif action=='alignment-return':self.alignment=controller.return_from_probe()
                        else:self.alignment=controller.step()
                    if action=='alignment-step' and self.last_execution is not None:
                        self.alignment['policy_input_snapshot']=self.last_execution.get('policy_input_snapshot')
                    write_json(self.output/f'{action}-{time.time_ns()}.json',self.alignment)
                    if ('before' in self.alignment and 'after' in self.alignment
                            and self.last_execution is not None and self.last_execution.get('completed')):
                        self.last_execution['wrist_alignment']={key:deepcopy(self.alignment[key]) for key in ('before','after')}
            elif action=='gripper-follow-check':
                if not self.gripper_check_enabled:
                    raise RuntimeError('one-shot gripper diagnostic is not enabled')
                if self.owner.runtime.pending is not None or self.owner.runtime.executing:
                    raise RuntimeError('stop the pending model transaction first')
                self.gripper_check_enabled=False
                self._clear_stop_for_current_operation()
                target=self._home_target()
                try:
                    empty_gripper_follow_check(self.owner,target)
                finally:
                    self.gripper_check=deepcopy(getattr(self.owner,'gripper_check_record',None))
                    if self.gripper_check is not None:
                        write_json(self.output/f'gripper-follow-check-{time.time_ns()}.json',self.gripper_check)
                    self.plan_record=None;self.preview_record=None
                    self.memory_reset_reason='manual empty-gripper diagnostic; no model history fabricated'
                with torch.inference_mode():self._observe()
            elif action=='camera-condition':
                self._set_camera_condition_at_home(payload)
            else:
                self._alignment_session=None
                self._home(payload)
            self.phase = 'idle'
            return self.transport_status()
        except BaseException as exc:
            operation_error=exc
            log=getattr(self,'_bus_faults',None)
            if log is not None:
                log.operation_failed(action,exc)
            if action == 'execute':
                self._assistance.reset()  # partial physical motion is not a valid sag/carry update
            else:
                self._assistance.abort()
            self._assistance_proposal_id = None
            self.error = f'{type(exc).__name__}: {exc}'
            if action == 'plan':
                # A plan whose evidence could not be committed must never stay
                # armed after its request failed. Planning performed no motion.
                self.owner.stop_event.set()
                self.owner.runtime.stop()
            if action in ('execute','target-track','carry-visual-probe','home','column-dino-fg-probe','alignment-probe','alignment-return','alignment-step',
                          'alignment-departure','alignment-view-step','alignment-reach','gripper-follow-check'):
                self.owner.stop_event.set(); self.owner.runtime.stop()
                try: self.owner._halt()
                except Exception as stop_exc: self.stop_error = str(stop_exc)
            if action == 'execute':
                self._save_rejected_complete_response(execute_attempt)
                self._save_failed_dispatch_evidence(execute_attempt)
            self.phase = 'idle'
            raise
        finally:
            diagnostic_attempt = None
            protocol = {}
            try:
                if action in ('same-goal-p32', 'same-goal-i1') and same_goal is not None:
                    protocol = same_goal.last_result or {}
                if action in ('j3-pid-baseline', 'j3-pid-variant', 'j3-range-diagnostic', 'j3-compensation-diagnostic', 'j3-native-response-diagnostic') and getattr(self, 'owner', None) is not None:
                    if action == 'j3-native-response-diagnostic':
                        protocol = getattr(self.j3_native_response_diagnostic, 'last_result', None) or {}
                    elif action == 'j3-compensation-diagnostic':
                        protocol = getattr(self.j3_compensation_diagnostic, 'last_result', None) or {}
                    elif action == 'j3-range-diagnostic':
                        protocol = getattr(self.j3_range_diagnostic, 'last_result', None) or {}
                    else:
                        diagnostic_result = getattr(diagnostic, 'last_result', None) or {}
                        protocol = diagnostic_result.get('protocol_report') or {}
                    diagnostic_attempt = protocol.get('drive_result') or protocol.get('failed_drive_attempt')
                    if diagnostic_attempt is not None:
                        self.last_attempt = deepcopy(diagnostic_attempt)
                    recording_errors = []
                    if self.owner.runtime.requires_home_after_incomplete_execution:
                        try:
                            write_json(self.output/'incomplete-execution.json', dict(
                                requires_home=True, reason='diagnostic motion requires HOME before policy resume'))
                        except Exception as error:
                            recording_errors.append(error)
                    if diagnostic_attempt is not None:
                        try:
                            write_json(self.output/'last-attempt.json', self.last_attempt)
                        except Exception as error:
                            recording_errors.append(error)
                    if recording_errors:
                        raise recording_errors[0]
                if (action in ('execute','target-track','carry-visual-probe','column-dino-fg-probe','alignment-probe','alignment-return','alignment-step','alignment-departure','alignment-view-step','alignment-reach')
                        and (action!='execute' or execute_attempt is not None)):
                    if action == 'execute':
                        self._save_lewm_shadow_attempt()
                    self.last_attempt = deepcopy(execute_attempt if action=='execute' else self.owner.bridge.last_attempt)
                    if self.owner.runtime.requires_home_after_incomplete_execution:
                        write_json(self.output/'incomplete-execution.json',dict(
                            requires_home=True,reason='attempted motion without a complete actual transaction'))
                    if self.last_attempt is not None: write_json(self.output/'last-attempt.json', self.last_attempt)
                    if (self.auxiliary_budget.active and self.last_attempt is not None
                            and self.last_attempt.get('completed') is True
                            and self.auxiliary_budget.status()['command_counters_complete']):
                        save_actual_resume(self.output/'auxiliary-actual-latest.pt',self.owner.runtime,
                                           self.auxiliary_budget,self.root,self.last_execution)
                self._refresh_status()
            except Exception as recording_exc:
                diagnostic_action = action in ('j3-pid-baseline', 'j3-pid-variant',
                                                'j3-range-diagnostic', 'j3-compensation-diagnostic', 'j3-native-response-diagnostic',
                                                'same-goal-p32', 'same-goal-i1')
                if operation_error is None or (action != 'execute' and not diagnostic_action):
                    raise
                if action == 'execute' and execute_attempt is not None:
                    execute_attempt['final_recording_error']=f'{type(recording_exc).__name__}: {recording_exc}'
                    self.last_attempt=deepcopy(execute_attempt)
                elif diagnostic_action:
                    message = f'{type(recording_exc).__name__}: {recording_exc}'
                    protocol['final_recording_error'] = message
                    if diagnostic_attempt is not None:
                        self.last_attempt['final_recording_error'] = message
            finally:
                self.operation_lock.release()

    def _clear_recent_readback(self):
        """Drop the 0–1.0 s readback counter as soon as HOME is requested."""
        self.last_execution = None
        self.last_attempt = None
        self.execution_history = []
        # #region agent log
        try:
            import json as _json, time as _time
            with open('/home/james/feedback-corrected-grounding/.cursor/debug-55ff20.log', 'a') as _f:
                _f.write(_json.dumps({'sessionId': '55ff20', 'hypothesisId': 'home-clear', 'location': 'v21_live_service.py:_clear_recent_readback', 'message': 'cleared recent readback on home', 'data': {'last_execution': self.last_execution is None, 'history': len(self.execution_history)}, 'timestamp': int(_time.time() * 1000)}) + '\n')
        except Exception:
            pass
        # #endregion
        self._refresh_status()

    def _home_target(self):
        """The HOME pose [6]: the legacy V17.4 fixed fish-task HOME with the live open gripper, or the --home-state
        file's current_state (all six joints). Objects built without __init__ (older tests) keep the legacy HOME."""
        path = getattr(self, 'home_state_path', None)
        if path is None:
            return established._load_v177_home_state(LEGACY_HOME_STATE, gripper_position=LEGACY_HOME_GRIPPER)
        state = json.loads(path.read_text(encoding='utf-8'))['current_state']
        return established._load_v177_home_state(path, gripper_position=float(state[5]))

    def _home(self, payload=None):
        self._clear_recent_readback()
        name, address = self._parse_camera_condition_payload(payload)
        self._reject_imported_memory_for_condition(
            self.camera_condition.name if name is None else name,
            self.camera_condition.address if address is None else address)
        owner = self.owner
        # HOME changes both wrist viewpoint and apparent fish scale. Retaining
        # the close-up area/anchor rejects the smaller fish on the next trial.
        # Keep the registered appearance; discard only observation continuity.
        encoder = getattr(owner, 'object_relative', None)
        if encoder is not None:
            encoder.clear_observation()
            owner.last_object_relative = np.zeros(8, np.float32).tolist()
            owner.last_object_diagnostic = np.zeros(8, np.float32).tolist()
        if getattr(owner.runtime, 'requires_home_after_incomplete_execution', False):
            from .v21_position_servo_recovery import restore_failed_position_servo_baseline
            from .v21_fixed_goal_service import verify_fixed_goal_compensation_before_home
            # Restore only the recorded failed profile while STOP is still set.
            # The existing HOME below remains responsible for actual recovery.
            recovery = verify_fixed_goal_compensation_before_home(self)
            if recovery is None:
                recovery = restore_failed_position_servo_baseline(self)
            if recovery is not None:
                self.position_servo_recovery = deepcopy(recovery)
        self._assistance.reset()  # manual HOME invalidates old grasp-phase state, even on failure
        self._assistance_proposal_id = None
        self._geom_place = None
        self._carry = None
        self._fish_task = dict(phase='approach')
        if getattr(owner, 'place_goal', None) is not None:
            owner.place_goal.reset()      # a new trial locates its box again (section 17.7)
        self._fish_task_proposal = None
        self._clear_stop_for_current_operation(); owner.runtime.stop()
        below_floor_lift = None
        if getattr(self, 'home_lift_below_floor', False):
            from .v21_home import lift_below_floor      # section 17.91: one joint up first when stopped under the floor
            below_floor_lift = lift_below_floor(owner, self.output)
        home_records=[]
        target = self._home_target()
        for _ in range(80):
            if owner.stop_event.is_set(): raise RuntimeError('HOME cancelled')
            q = owner.read_checked()
            if np.max(np.abs(q-target)) <= owner.io.settle_position_tolerance: break
            # Fold J2/J3/J4 first when the base is still over the table; otherwise the
            # joint-wise interpolant sweeps the gripper through the box.
            if abs(float(q[0])-float(target[0]))>15. and (
                    abs(float(q[1])-float(target[1]))>8. or abs(float(q[2])-float(target[2]))>8.):
                fold=np.asarray(q,dtype=np.float32).copy()
                fold[1:4]=target[1:4]; fold[5]=target[5]
                chunk = plan_home_chunk(owner, q, fold)
            else:
                chunk = plan_home_chunk(owner, q, target)
            if self.auxiliary_budget.active:
                owner.last_home_attempt=None
                self.auxiliary_budget.reserve_home()
                try:
                    self._persist_auxiliary_budget()  # Before any manual HOME command.
                    home_records.append(execute_home_readback_chunk(owner,chunk,period_s=.2))
                finally:
                    self.auxiliary_budget.finish(owner.last_home_attempt)
                    self._persist_auxiliary_budget()
            else:
                home_records.append(execute_home_readback_chunk(owner,chunk,period_s=.2))
        else:
            raise RuntimeError('fixed HOME did not converge inside its existing budget')
        # HOME is not a learned five-step transition; don't manufacture its history.
        owner.runtime.state = None
        self._last_head_observations={}  # A manual trial reset is not object continuity.
        self._assistance.reset()
        self._assistance_proposal_id = None
        self._filter_actual_observations = False
        try:
            if getattr(owner,'saved_column',None) is not None: owner.restore_column_memory()
        finally:
            self._filter_actual_observations = True
        self._reset_camera_condition(name, address)
        self._prepare_condition_reader()
        self.memory_reset_reason = 'explicit HOME: reset unmodelled home trajectory history'
        self.plan_record = None; self.preview_record = None; owner.stop_event.set()
        self.home_observation = dict(attempts=0, bracket_timeouts=[])
        with torch.inference_mode(): self._observe_bracket_retry(self.home_observation)
        if not self.camera_condition.h0_committed and self.camera_condition.name == 'both_fresh':
            self._force_commit_condition()
        write_json(self.output/f'home-{time.time_ns()}.json',dict(
            kind='fixed-home',model_transition=False,target_position=target.tolist(),chunks=home_records,
            below_floor_lift=None if below_floor_lift is None else below_floor_lift.get('path'),
            camera_condition=self.camera_condition.name,camera_address=self.camera_condition.address,
            observation=deepcopy(self.home_observation)))
        self._close_auxiliary_after_home()
        write_json(self.output/'incomplete-execution.json',dict(
            requires_home=False,reason='completed explicit HOME and fresh observation'))
        # The false marker is durable before removing the endpoint receipt. If
        # the process dies in this tiny window, restart ignores the stale
        # pending file because HOME is already recorded complete.
        clear_pending_home_receipt(self.output)
        owner.runtime.requires_home_after_incomplete_execution = False
        self._clear_recent_readback()

    def stop(self, *, trial_completion=False):
        # This short lock is paired with _clear_stop_for_current_operation.
        # A STOP either arrives before a ROS request arms (then it rejects), or
        # after it arms (then this set wins and the established bridge aborts).
        with self._ros_control_epoch_lock:
            self._ros_control_epoch += 1
            if self.owner is not None:
                self.owner.stop_event.set()
        with self._trial_admission_lock:
            self._trial_stop_generation += 1
            admission_cancelled = self._trial_admission_cancelled
            if admission_cancelled is not None:
                admission_cancelled.set()
            if not trial_completion:
                self.trial.cancel()
            if self.owner is not None:
                self.owner.stop_event.set()
        if self.owner is None: return
        self._assistance.abort()
        self._assistance_proposal_id = None
        try:
            self.owner.bridge.stop()
        except Exception as exc:
            self.stop_error = f'{type(exc).__name__}: {exc}'
            raise
        if not self.operation_lock.locked(): self._refresh_status()

    @staticmethod
    def _draw_cog_caption(bgr, org, dx, dy, scale=1.):
        """Wrist JPEG caption: 重心 / CoG. Hershey cannot draw CJK; try a system CJK font."""
        caption='重心 CoG dx %+.0f dy %+.0f'%(dx,dy)
        ascii_caption='CoG dx %+.0f dy %+.0f'%(dx,dy)
        try:
            from PIL import Image, ImageDraw, ImageFont
            for font_path in (
                Path(r'C:\Windows\Fonts\msyh.ttc'),
                Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
                Path('/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'),
                Path('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'),
            ):
                if not font_path.is_file():
                    continue
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                image=Image.fromarray(rgb)
                font=ImageFont.truetype(str(font_path),max(11,int(13*max(1.,scale*.8))))
                ImageDraw.Draw(image).text(org,caption,font=font,fill=(255,230,0))
                return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        except Exception:
            pass
        cv2.putText(bgr,ascii_caption,org,cv2.FONT_HERSHEY_SIMPLEX,.33*max(1.,scale*.8),(0,230,255),1,cv2.LINE_AA)
        return bgr

    def _draw_object_geometry(self, bgr, scale=1., offset=(0., 0.), *, overlay=None):
        """Draw an available Wrist body target; FK tips are rendered separately."""
        if self.head_grasp or self.grasp_target=='head':
            return bgr  # live video must not reuse a stale head/body overlay
        rec=overlay
        v=None if overlay is None else overlay.get('vector')
        if not isinstance(rec,dict): return bgr
        body=rec.get('source')=='operator_colour_body'
        if v is not None:
            try:
                v=np.asarray(v,np.float64)
                available=v.shape==(8,) and np.isfinite(v).all() and v[0]>0
            except (ValueError,TypeError):
                available=False
        else:
            # Older display records may lack the cue vector. Keep their
            # declared body/support contract; a partial colour body can be
            # valid, whereas clipped frozen support is only a diagnostic.
            available=rec.get('policy_token_available', body or rec.get('source') in
                ('frozen_symmetric_support','frozen_vit_features'))
        if (not available or rec.get('policy_token_available') is False
                or rec.get('visible') is False or rec.get('aimable') is False
                or (not body and rec.get('support_clipped'))):
            return bgr
        mark=rec.get('cog_px') if rec.get('cog_px') is not None else rec.get('centroid_px')
        if mark is None: return bgr
        def P(xy): return (int(round((float(xy[0])-offset[0])*scale)),int(round((float(xy[1])-offset[1])*scale)))
        try:
            g=np.asarray(mark,np.float64)
            contour=rec.get('contour_px') or []
            if not contour:
                mask=rec.get('support_mask')
                if mask is not None:
                    from .so101_frozen_vit_grounding import support_contour_px
                    contour=support_contour_px(mask)
            if len(contour)>=3:
                pts=np.array([P(p) for p in contour],np.int32)
                cv2.polylines(bgr,[pts],True,(0,220,255),1,cv2.LINE_AA)
            axis=np.asarray(rec.get('axis_px',(1.,0.)),np.float64).reshape(2)
            length=float(rec.get('length_px') or 40.)
            if np.isfinite(axis).all() and float(np.linalg.norm(axis))>1e-6:
                unit=axis/np.linalg.norm(axis)
                half=max(12.,0.45*length)*unit
                cv2.line(bgr,P(g-half),P(g+half),(80,220,80),2,cv2.LINE_AA)  # green long / symmetry axis
                # Blue = grasp direction, always ⊥ unit (ignore stored grasp_axis).
                gu=np.array([-unit[1], unit[0]], np.float64)
                width=float(rec.get('width_px') or 16.)
                gh=max(14.,0.55*max(width,16.))*gu
                cv2.line(bgr,P(g-gh),P(g+gh),(255,140,40),2,cv2.LINE_AA)  # blue grasp ⊥ axis
            pair=np.asarray(rec.get('jaw_pair_px'),np.float64) if rec.get('jaw_pair_px') is not None else None
            if pair is not None and pair.shape==(2,2) and np.isfinite(pair).all():
                f,g_tip=P(pair[0]),P(pair[1])
                cv2.line(bgr,f,g_tip,(255,255,255),1,cv2.LINE_AA)  # FG white (live jaws; not forced ⊥)
                mid=pair.mean(0)
                cv2.line(bgr,P(mid),P(g),(0,200,255),1,cv2.LINE_AA)  # CM orange (not blue)
                cv2.circle(bgr,P(mid),1,(0,220,255),-1,cv2.LINE_AA)  # M dot
            cv2.circle(bgr,P(g),2,(0,230,255),-1,cv2.LINE_AA)
            gp=P(g)
            cv2.putText(bgr,'C',(max(1,min(210,gp[0]+3)),max(10,min(210,gp[1]-3))),
                        cv2.FONT_HERSHEY_SIMPLEX,.36,(0,230,255),1,cv2.LINE_AA)
            if v is not None:
                org=(P((4,196))[0],P((4,196))[1]-4)
                dx,dy=v[2]*224,v[3]*224
                bgr=self._draw_cog_caption(bgr,org,dx,dy,scale)
        except Exception: pass
        return bgr

    def frame(self, view, source='live', sequence=None):
        if self.closing.is_set(): raise RuntimeError('service is closing')
        if view not in ('wrist','column') or source not in ('live','endpoint'):
            raise ValueError('unknown image source')
        if source == 'endpoint':
            with self.display_lock:
                snap = self.snapshot
                if snap is None or (sequence is not None and int(sequence) != snap['sequence']):
                    raise ValueError('observation snapshot changed; refresh status')
                if snap.get('view_sources',{}).get(view) == 'unavailable':
                    raise ValueError('no current or restored historical frame for this view')
                rgb = snap['rgb'][view].copy(); targets = deepcopy(snap['targets'])
                tips = deepcopy(snap.get('fingertips',{}).get(view))
                if view == 'column' and isinstance(tips, dict):
                    tips = self._fingertips_with_step5({'column': tips}).get('column')
                object_overlay=deepcopy(snap.get('object_overlay'))
                head = deepcopy(snap.get('head_targets',{}).get(view,{}))
                historical = snap.get('view_sources',{}).get(view)=='historical_observation_not_live'
                head_time = head.get('image_time_s')
                frame_time = snap['image_times'][0][('wrist','column').index(view)]
                same_exposure = (isinstance(head_time,(float,int)) and np.isfinite(head_time)
                    and abs(head_time-frame_time)<=1e-6
                    and snap.get('view_sources',{}).get(view)=='current_observation'
                    and bool(snap['measurement_available'][0][('wrist','column').index(view)]))
                if not same_exposure:
                    head = dict(available=False, reason='head_frame_mismatch')
            if self.head_grasp or self.grasp_target=='head':
                targets.get('targets',{}).pop('pickup',None)  # do not label old body target as head
            renderer = SimpleTargetRenderer()
            jpeg=renderer.render(view,rgb,targets,tips)
            if self.head_grasp or self.grasp_target=='head':
                picture=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
                if head.get('available'):
                    point=tuple(np.rint(head['centroid_px']).astype(int))
                    cv2.circle(picture,point,4,(220,80,240),-1,cv2.LINE_AA)
                    cv2.putText(picture,'HEAD CANDIDATE',(5,210),cv2.FONT_HERSHEY_SIMPLEX,.34,(220,80,240),1,cv2.LINE_AA)
                else:
                    cv2.putText(picture,'HEAD NOT OBSERVED',(5,210),cv2.FONT_HERSHEY_SIMPLEX,.34,(0,180,255),1,cv2.LINE_AA)
                ok,encoded=cv2.imencode('.jpg',picture,[cv2.IMWRITE_JPEG_QUALITY,92])
                if ok:jpeg=encoded.tobytes()
            if view=='wrist':
                picture=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
                picture=self._draw_object_geometry(picture,1.,(0.,0.),overlay=object_overlay)
                ok,encoded=cv2.imencode('.jpg',picture,[cv2.IMWRITE_JPEG_QUALITY,92])
                if ok: jpeg=encoded.tobytes()
            if historical:
                picture=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
                cv2.putText(picture,'HISTORICAL COLUMN - NOT LIVE',(5,15),cv2.FONT_HERSHEY_SIMPLEX,.31,(0,200,255),1,cv2.LINE_AA)
                ok,encoded=cv2.imencode('.jpg',picture)
                if not ok:raise ValueError('historical frame JPEG failed')
                jpeg=encoded.tobytes()
            return jpeg
        # Live video does not run the policy or write visual memory.
        if view not in getattr(self.owner,'active_views',('wrist','column')):
            raise ValueError('Column has no current frames; historical snapshot is available only if explicitly restored')
        io = self.owner.io
        if io.camera_lock.acquire(blocking=False):
            try:
                small = io._read_camera(view)
                rgb = getattr(io, '_last_raw_frames', {}).get(view, small).copy()
                with self.display_lock:
                    self.live_frames[view] = rgb; self.live_times[view] = self.clock()
            finally: io.camera_lock.release()
        else:
            with self.display_lock:
                rgb = self.live_frames.get(view)
                snap = self.snapshot
                live_time = self.live_times.get(view)
                current_snapshot = (snap is not None and
                    snap.get('view_sources',{}).get(view)=='current_observation')
                # A missing camera must not silently become an old "live" JPEG
                # while another acquisition owns the camera lock.
                if (snap is not None and not current_snapshot and
                        (live_time is None or live_time<=snap['time'])):
                    raise ValueError('no current live frame: view unavailable')
                if rgb is None and current_snapshot:
                    rgb = snap.get('raw',{}).get(view)
                    live_time = snap['time']
                if rgb is None or live_time is None or self.clock()-live_time>1.:
                    raise ValueError('camera capture is busy; no fresh live frame')
        bgr=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # Live pixels never borrow object geometry from an older endpoint.
        ok, result = cv2.imencode('.jpg', bgr)
        if not ok: raise ValueError('JPEG encode failed')
        return result.tobytes()

    def close(self):
        self.closing.set()
        if self.owner is None: return
        try:
            self.stop()
            self.trial.join(timeout=5.)
        finally:
            if self.monitor is not None: self.monitor.join(timeout=2.)
            if self.c4 is not None: self.c4.close()
            acquired = self.operation_lock.acquire(timeout=5.)
            try:
                if not acquired: raise RuntimeError('hardware operation did not stop before close')
                if self.owner.io is not None:
                    with self.owner.io.camera_lock: self.owner.close()
                else:
                    self.owner.close()
            finally:
                if acquired: self.operation_lock.release()


class SimpleTargetRenderer:
    """Per-view object contour + axis + C; drop ellipse; F/G/M. No score text.

    Wrist and column each use that camera's own support. Never paste wrist
    mask/contour onto the column JPEG, and never draw the object as an ellipse.
    """
    _registered_target_records = {}

    @staticmethod
    def _draw_object_shape(bgr, view, targets):
        """Free-form support contour + long-axis from the *same* view record.

        Wrist skips here: endpoint path draws one C+contour from the policy
        ``object_overlay`` (wrist 8-D source) so the JPEG does not get two CoGs.
        """
        if view == "wrist":
            return None
        from .v21_overlay_geometry import same_view_object_contour
        pickup = ((targets or {}).get("targets") or {}).get("pickup") or {}
        rec = pickup.get(view) or {}
        if not isinstance(rec, dict) or not rec.get("visible"):
            return None
        # Refuse cross-camera paste before any ink hits the JPEG.
        if rec.get("view") not in (None, view):
            return None
        cog = rec.get("cog_px") if rec.get("cog_px") is not None else rec.get("centroid_px")
        if cog is None:
            return None
        cog = np.asarray(cog, np.float64).reshape(2)
        if not np.isfinite(cog).all():
            return None
        # Same-camera contour only — never wrist mask pasted onto column.
        contour = same_view_object_contour(rec, view, bgr.shape[:2])
        if contour and len(contour) >= 3:
            pts = np.array([(int(round(p[0])), int(round(p[1]))) for p in contour], np.int32)
            # Outline only — never a filled puddle, never an ellipse.
            cv2.polylines(bgr, [pts], True, (0, 220, 255), 1, cv2.LINE_AA)
        # Green = long / symmetry axis through C. Blue = grasp direction (⊥ axis).
        axis = np.asarray(rec.get("axis_px", (1.0, 0.0)), np.float64).reshape(2)
        length = float(rec.get("length_px") or 40.0)
        if np.isfinite(axis).all() and float(np.linalg.norm(axis)) > 1e-6:
            unit = axis / np.linalg.norm(axis)
            half = max(12.0, 0.45 * length) * unit
            a = tuple(np.round(cog - half).astype(int))
            b = tuple(np.round(cog + half).astype(int))
            cv2.line(bgr, a, b, (80, 220, 80), 2, cv2.LINE_AA)  # green long axis
            gu = np.array([-unit[1], unit[0]], np.float64)
            width = float(rec.get("width_px") or 16.0)
            gh = max(14.0, 0.55 * max(width, 16.0)) * gu
            ga = tuple(np.round(cog - gh).astype(int))
            gb = tuple(np.round(cog + gh).astype(int))
            cv2.line(bgr, ga, gb, (255, 140, 40), 2, cv2.LINE_AA)  # blue grasp ⊥ axis
        c = tuple(np.round(cog).astype(int))
        cv2.circle(bgr, c, 2, (0, 230, 255), -1, cv2.LINE_AA)
        cv2.putText(bgr, "C", (max(1, min(210, c[0] + 3)), max(10, min(210, c[1] - 3))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 230, 255), 1, cv2.LINE_AA)
        return cog

    def render(self, view, rgb, targets, tips=None):
        # Empty DINO overlay dict: parent ellipse/TARGET path must not run for pickup.
        image = V203HardwareService._render_frame(self, view, rgb, {}, {"targets": {}})
        bgr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        cog = self._draw_object_shape(bgr, view, targets)
        drop = ((targets or {}).get("targets") or {}).get("drop_zone") or {}
        drop_rec = drop.get(view) or {}
        if isinstance(drop_rec, dict) and drop_rec.get("visible") and drop_rec.get("centroid_px") is not None:
            center = np.asarray(drop_rec["centroid_px"], np.float64).reshape(2)
            if np.isfinite(center).all():
                axis = np.asarray(drop_rec.get("axis_px", (1.0, 0.0)), np.float64).reshape(2)
                if not np.isfinite(axis).all() or float(np.linalg.norm(axis)) < 1e-6:
                    axis = np.array([1.0, 0.0])
                width = max(8.0, float(drop_rec.get("width_px") or 16.0))
                length = max(width, float(drop_rec.get("length_px") or width * 1.5))
                angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
                c = tuple(np.round(center).astype(int))
                cv2.ellipse(bgr, c, (int(length / 2), int(width / 2)), angle, 0, 360,
                            (72, 184, 255), 1, cv2.LINE_AA)
                cv2.circle(bgr, c, 2, (72, 184, 255), -1, cv2.LINE_AA)
                # Drop stays an ellipse; no DROP/TARGET text.
        draw = tips
        if (view in ("column", "wrist") and isinstance(tips, dict) and not tips.get("available")
                and tips.get("display_fm_px") is not None):
            draw = dict(tips, available=True, fm_px=tips.get("display_fm_px"),
                        unordered=False, ordered_identity_available=True)
        if (isinstance(draw, dict) and tip_source_matches_view(view, draw)
                and draw.get("available")):
            pair = draw.get("fm_px")
            centre = None
            # On wrist, CoG/shape come from object_overlay; do not draw a second C here.
            if pair is not None:
                arr = np.asarray(pair, dtype=np.float64)
                if arr.shape == (2, 2) and np.isfinite(arr).all():
                    def shown(point):
                        # Ink stays on the picture when the tip itself is outside.
                        return (int(np.clip(np.rint(point[0]), 2, bgr.shape[1] - 3)),
                                int(np.clip(np.rint(point[1]), 2, bgr.shape[0] - 3)))
                    f, g = shown(arr[0]), shown(arr[1])
                    if view == "column":
                        cv2.line(bgr, f, g, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.line(bgr, f, g, (255, 255, 255), 1, cv2.LINE_AA)  # FG white
                    identity_unavailable = (
                        draw.get("unordered") is True
                        or draw.get("ordered_identity_available") is False
                        or (view == "column" and draw.get("ordered_identity_available") is not True))
                    labels = ("P1", "P2") if identity_unavailable else ("F", "G")
                    for label, point, color in ((labels[0], f, (60, 255, 60)), (labels[1], g, (255, 70, 230))):
                        if view == "column":
                            # Keep the RGB endpoint at the ring's centre and
                            # retain contrast on both pale jaws and the table.
                            cv2.circle(bgr, point, 4, (0, 0, 0), 3, cv2.LINE_AA)
                            cv2.circle(bgr, point, 4, color, 1, cv2.LINE_AA)
                            cv2.circle(bgr, point, 1, color, -1, cv2.LINE_AA)
                            (width, height), baseline = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, .4, 1)
                            origin = (max(1, min(bgr.shape[1]-width-2, point[0]+6)),
                                      max(height+2, min(bgr.shape[0]-baseline-2, point[1]-7)))
                            cv2.putText(bgr, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                                        .4, (0, 0, 0), 3, cv2.LINE_AA)
                            cv2.putText(bgr, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                                        .4, color, 1, cv2.LINE_AA)
                            continue
                        cv2.circle(bgr, point, 2, color, -1, cv2.LINE_AA)
                        cv2.putText(
                            bgr, label,
                            (max(1, min(210, point[0] + 3)), max(10, min(210, point[1] - 4))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
                    centre = arr.mean(axis=0)
            if centre is None:
                raw = tips.get("m_px")
                if raw is not None:
                    centre = np.asarray(raw, np.float64)
            if centre is not None and np.isfinite(centre).all():
                m = tuple(np.round(np.asarray(centre).reshape(2)).astype(int))
                cv2.circle(bgr, m, 1, (0, 220, 255), -1, cv2.LINE_AA)
                if cog is not None:
                    cv2.line(bgr, m, tuple(np.round(cog).astype(int)), (0, 200, 255), 1, cv2.LINE_AA)  # CM orange
        if view == "column" and isinstance(tips, dict):
            predicted = []
            for key in ("f5_px", "g5_px"):
                point = np.asarray(tips.get(key), np.float64).reshape(-1)
                if point.shape == (2,) and np.isfinite(point).all():
                    predicted.append(tuple(np.rint(point).astype(int)))
            if len(predicted) == 2:
                cv2.line(bgr, predicted[0], predicted[1], (255, 210, 80), 1, cv2.LINE_AA)
                for point in predicted:
                    cv2.circle(bgr, point, 5, (255, 210, 80), 1, cv2.LINE_AA)
        suggestion = tips.get("prediction_suggestion") if isinstance(tips, dict) else None
        if (view == "column" and isinstance(suggestion, dict)
                and suggestion.get("current_frame_available") is True
                and suggestion.get("available") is False
                and suggestion.get("observed") is False):
            pair = np.asarray(suggestion.get("suggested_fm_px"), dtype=np.float64)
            if (pair.shape == (2, 2) and np.isfinite(pair).all()
                    and not (pair < [0., 28.]).any() and not (pair >= [224., 196.]).any()):
                points = np.rint(pair).astype(int)
                colour = (255, 190, 80)
                delta = points[1] - points[0]
                # Predicted suggestions use a dashed segment and open markers;
                # solid white F/G remains reserved for current RGB evidence.
                for start in np.linspace(0., .8, 5):
                    a = tuple(np.rint(points[0] + start * delta).astype(int))
                    b = tuple(np.rint(points[0] + min(1., start + .1) * delta).astype(int))
                    cv2.line(bgr, a, b, colour, 1, cv2.LINE_AA)
                for label, point in zip(("PF", "PG"), points):
                    p = tuple(point)
                    cv2.circle(bgr, p, 4, colour, 1, cv2.LINE_AA)
                    cv2.putText(bgr, label,
                        (max(1, min(204, p[0] + 6)), max(10, min(210, p[1] - 7))),
                        cv2.FONT_HERSHEY_SIMPLEX, .34, colour, 1, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return encoded.tobytes() if ok else image
