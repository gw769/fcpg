"""Prepared FCPG + LeWM planner for interactive SO-101 control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
import uuid
from typing import Any

import numpy as np
import torch

from fcpg.coarse_fine_planner import (
    apply_coarse_fine_observe_points,
    assert_coarse_fine_planner_policy,
    coarse_fine_observe_diagnostics,
    disabled_global_tip_diagnostics,
    extract_coarse_fine_checkpoint_metadata,
    grounding_arm_name,
    planner_grounding_ranking_contract,
)
from fcpg.grounding_confidence_fusion import (
    COARSE_FINE_VARIANT,
    GROUNDING_GATED_FCPG,
    GROUNDING_NOMINAL,
    RANKING_OFF,
    RANKING_ON,
    GroundingFusionState,
    planner_envelope_should_block,
)
from fcpg.so101_fcpg_lewm_bridge import (
    SO101FCPGConditionedLeWM,
    build_so101_fcpg_corrector,
    infer_so101_fcpg_corrector_variant,
)
from fcpg.modules import sample_feature_map_at_pixels
from fcpg.so101_active_gripper import (
    DEFAULT_MAXIMUM_LIVE_ROTATION_DEG,
    apply_live_center_translation,
    apply_live_similarity_transform,
    invert_live_similarity_transform,
    select_live_moving_tip_similarity_transform,
    select_live_moving_tip_translation,
)
from fcpg.so101_camera_calibration import (
    ProjectionFit,
    approximate_column_backprojection_on_plane,
)
from fcpg.so101_column_extrinsic_observer import (
    ColumnExtrinsicObserver,
    apply_column_frame_offset_px,
    measure_column_fk_translation,
    refine_column_live_nominal,
)

from fcpg.observer_runtime import (
    close_sigma_limit,
    maybe_excitation,
    maybe_register_or_reproject_drop,
    shadow_enabled,
    shadow_step,
)
from fcpg.so101_object_belief import (
    ObjectBelief,
    search_anchor_px,
    seed_object_belief,
    update_object_belief,
)
from fcpg.so101_goal_directed_candidates import (
    DEFAULT_APPROACH_VARIANTS,
    DEFAULT_TRANSPORT_VARIANTS,
    generate_fcpg_goal_prefixes,
    generate_gripper_phase_prefixes,
    scale_image_servo_variants,
)
from fcpg.so101_global_tip_locator import (
    SO101GlobalTipHeatmapHead,
    decode_structured_tip_heatmaps,
    global_tip_sequence_diagnostics,
    load_global_tip_heatmap_head,
)
from fcpg.so101_ik_candidates import (
    generate_ik_candidate_prefixes,
    realtime_path_specs_for_phase,
    track_anchor_trajectory,
)
from fcpg.letterbox_geometry import (
    LETTERBOX_V1,
    PLANNING_IMAGE_SIZE,
    SQUASH_V0,
    PlanningGeometry,
    planning_geometry_from_protocol,
)
from fcpg.so101_kinematics import SO101MuJoCoKinematics
from fcpg.so101_multiscale_memory import (
    MultiScaleHistoryLeWM,
    rollout_multiscale_candidates,
)
from fcpg.so101_realtime_mpc import (
    ManipulationPhase,
    PhaseEvidence,
    PhaseThresholds,
    PhaseTracker,
    PlanningRoundGuard,
    candidate_phase_mask,
    close_allowed_for_pickup,
    detected_tip_separation_px,
    empty_bite_reapproach_open_targets,
    exclusive_phase_gripper_open,
    grasp_aperture_candidates,
    grasp_closed_floor,
    jaws_in_holding_aperture,
    aperture_hold_cost,
    predicted_closure_cost,
    half_extent_along,
    judge_three_point_closure,
    JAW_COLLAPSED_TIP_SEPARATION_PX,
    keep_v11_candidates_phase_safe,
    phase_prediction_bounds_mask,
    target_role_for_phase,
    validate_command_chunk,
)
from fcpg.so101_retrieval_mpc import (
    SO101_POSITION_LOWER,
    SO101_POSITION_UPPER,
    candidate_safety_checks,
    candidate_start_rows,
    chunk_and_normalize_actions,
    raw_action_prefixes,
    rebase_action_prefixes,
    retrieve_diverse_candidates,
    validate_paired_camera_data,
)
from fcpg.so101_single_view_planning import (
    annotate_drop_phase_gate,
    compute_pickup_visible_with_fallback,
    gripper_inside_bbox,
)
from fcpg.so101_state_residual_lewm import StateResidualLeWM
from fcpg.prod_v16_5_runtime import (
    ANCHOR_AUTHORITIES as V165_ANCHOR_AUTHORITIES,
    V165RuntimeAdapter,
    conservative_joint_sweeps,
    no_contact_approach_mask,
    v165_command_reference_states,
    v165_rgb_view_nonblack,
)
from fcpg.prod_v16_8_live import (
    ACTOR_SOURCE as V168_ACTOR_SOURCE,
    V168FoundationRuntimeAdapter,
    V168LiveActorOverlay,
)
from fcpg.prod_v17_0_runtime import (
    V170LiveStack,
    assert_v170_shadow_bootstrap,
)
from fcpg.so101_v9 import (
    V6_PROTOCOL,
    V9_PROTOCOL,
    assert_v9_backbone_matches,
    should_use_memory,
)
from fcpg.so101_calibration_continuity import calibration_identity
from fcpg.so101_visual_targets import (
    TARGET_ROLES,
    TARGET_VIEWS,
    VisualTargetTracker,
)
from fcpg.so101_single_view_target_gate import (
    select_primary_target_view,
    should_block_on_target_visibility,
)
from fcpg.so101_lewm_rank_views import (
    RANK_VIEW,
    pixel_rank_view_for_rollout,
    rank_target_views,
    rank_view_contract_fields,
    ranking_pixel_goal_weight,
    wrist_target_px,
)
from fcpg.so101_lewm_prediction_residual import (
    RESIDUAL_VIEWS,
    context_residual,
    observed_change,
    predict_future_context,
)
from fcpg.so101_v11_6_lewm_rank import (
    V11_6_ROLLOUT_VARIANT,
    require_v11_6_rank_policy,
    rollout_v11_6_candidates,
)
from scripts.evaluate_so101_real_shadow_fcpg_lewm import (
    VIEWS,
    current_pixel_goal_cost,
    encode_fcpg_view,
    load_projection_fits,
    project_future_anchor_pixels,
    project_history,
    red_object_phase_distances as demonstration_pickup_phase_distances,
)
from scripts.evaluate_so101_retrieval_mpc import _rollout_candidates, _training_episodes
from scripts.train_so101_lewm import episode_splits, parse_exclusions
from scripts.train_so101_synthetic_fcpg import normalize_rgb
from scripts.visualize_so101_lewm import (
    B2HEAD_RUNTIME_PROTOCOL,
    CacheView,
    build_checkpoint_model,
    checkpoint_action_normalizer,
    checkpoint_contract,
    encode_rows,
    load_caches,
)


def _template_tracking_removed(role: str, view: str) -> dict[str, Any]:
    """The colour template tracker (VisualTargetTracker.track) is gone from the code (10/31, James 「不要的废弃代码
    这种颜色函数删掉」): an unconfigured, not-visible descriptor stands in its place."""
    return {"role": role, "view": view, "configured": False, "visible": False, "centroid_px": None,
            "source": "template_tracking_removed", "reason": "template_tracking_removed"}


class ContextRegistrationError(RuntimeError):
    """A recoverable unmet contract for registering or using live context."""


#: Executed scope of the runtime LeWM decision-independence claim.
#:
#: This string stays APPROACH-ONLY on purpose, and it is deliberately narrower
#: than the test coverage described below.  It names the scope that is cleared
#: to RUN, and the pilot it was written for is approach-only.
#:
#: Two different claims must not be collapsed into one here.
#:
#: 1. DECISION INDEPENDENCE (what the tests measure).  Tier 2 drives real
#:    ``plan()`` calls in ``control_mode="phase_target"`` at ``APPROACH``,
#:    ``GRASP`` and ``TRANSPORT``, and at ``APPROACH`` with a CONFIGURED peg so
#:    the real clearance path is exercised.  Each case is asserted non-vacuous
#:    (candidates produced, >= 2 eligible, a candidate selected, plan
#:    executable) before its identity result is allowed to count.  So
#:    independence in those phases is now a measured result, not an argument
#:    about shared code.  ``release`` is still only the shared-code argument:
#:    no test drives a plan through it.
#:
#: 2. EXECUTION VERIFICATION (what would clear a phase to run).  Tier 2 runs on
#:    a CPU-only synthetic scene and replaces the checkpointed model rollout
#:    with a deterministic stand-in; the frozen B2-head is never loaded, and no
#:    hardware is involved.  It therefore says nothing about whether ``grasp``,
#:    ``transport`` or peg-clearance BEHAVE correctly.  Those remain NOT
#:    execution-verified and are not cleared to run.
#:
#: ``control_mode="anchor_trajectory"`` is refused outright under
#: ``mpc_only_collection`` (see ``PreparedPlannerConfig``), because its
#: workspace veto is prediction-only and has no commanded-trajectory conjunct
#: to fall back on.
MPC_ONLY_INDEPENDENCE_SCOPE = "phase_target_approach"
V17_PIXEL_FM_SOURCES = {
    "v17.1": "v17_1_factorized_fused_fm",
    "v17.4": "v17_4_residual_belief_fm",
}

#: The tests that were executed to earn the scoped claim above.
#:
#: The first two measure decision independence (Tier 1: the pure-function
#: composition gate; Tier 2: real ``plan()`` calls with predictions injected at
#: the ``_predict_candidates`` seam).  The third gates this provenance block
#: itself: it is what makes a scopeless ``verified=True`` unrepresentable rather
#: than merely discouraged, so a reader who trusts the emitted scope is relying
#: on it.
MPC_ONLY_INDEPENDENCE_EVIDENCE: tuple[str, ...] = (
    "tests/test_so101_mpc_only_decision_independence.py",
    "tests/test_so101_tier2_mpc_only_replay_independence.py",
    "tests/test_so101_mpc_only_scope_provenance.py",
)

MPC_ONLY_INDEPENDENCE_INFLUENCE = "zero_shadow_logging_only"
MPC_ONLY_UNVERIFIED_INFLUENCE = (
    "lewm_excluded_from_cost_eligibility_and_workspace_veto"
)
LEWM_RANKING_INFLUENCE = "lewm_cost_in_candidate_ranking"

#: Command-side execute-horizon gates.  The 1 mm floor is the already-computed
#: ``ik_receding_horizon_qualified`` predicate.  The gap limit is the lewmpred
#: on-disk cutoff (selected-arc gaps 19–30 mm; successful servo/direct 8.5–13.4 mm).
IK_FIRST_CHUNK_PROGRESS_MIN_M = 1.0e-3
IK_RECEDING_MISMATCH_MIN_M = 1.0e-3
MODEL_COMMAND_XYZ_GAP_LIMIT_M = 0.012

#: GRASP ranking weight for the tip-corridor closure term.  Inside grasp range
#: the arm is already on target, so the cartesian spread across candidates is
#: both tiny and meaningless and closure has to outweigh it.  The term spans
#: [0, 6] and live rounds show a within-round closure spread near 2.9 against a
#: cartesian spread near 0.003, so 0.05 leaves closure decisive without letting
#: it override the safety masks.  This is the only tuned constant the closure
#: objective introduces; it replaces the retired aperture-ramp thresholds.
GRASP_CLOSURE_COST_WEIGHT = 0.05
#: Saturated worst score for a candidate whose jaws failed to project.
CLOSURE_COST_CEILING = 6.0
#: Jaw rates (deg/step) crossed with every actuator arm trajectory in GRASP.
#: A sampling grid, not a threshold: close faster / close / close slow / reopen
#: so the MPC can back out of a bad bite without a scripted recovery branch.
#: ``-2.0`` is the live open-jaw hole: ``-0.6`` from 31.4° is only −3°/chunk.
GRASP_JAW_RATES: tuple[float, ...] = (-2.0, -0.6, -0.25, 0.6)


def lewm_decision_independence_fields(
    *, mpc_only_collection: bool
) -> dict[str, Any]:
    """Build the LeWM decision-influence provenance block as one unit.

    One construction site for four fields that must agree with each other.
    ``lewm_decision_independence_verified`` is DERIVED from the scope instead
    of being written independently, which is what makes "verified with no
    scope" unrepresentable rather than merely discouraged: with no scope
    string there is no ``True`` to emit, and the influence label falls back to
    the weaker pre-verification wording that is still accurate.
    """

    if not mpc_only_collection:
        scope: str | None = None
        influence = LEWM_RANKING_INFLUENCE
        evidence: tuple[str, ...] = ()
    else:
        scope = MPC_ONLY_INDEPENDENCE_SCOPE or None
        influence = (
            MPC_ONLY_INDEPENDENCE_INFLUENCE if scope else MPC_ONLY_UNVERIFIED_INFLUENCE
        )
        evidence = MPC_ONLY_INDEPENDENCE_EVIDENCE if scope else ()
    verified = bool(scope)
    return {
        "lewm_decision_influence": influence,
        "lewm_decision_independence_verified": verified,
        # Always present, including when it is None: a reader of an archived
        # plan.json must never have to infer an absent scope from a missing key.
        "lewm_decision_independence_scope": scope,
        "lewm_decision_independence_evidence": list(evidence),
    }


@dataclass(frozen=True)
class PreparedPlannerConfig:
    bridge_checkpoint: Path
    calibration: Path
    secondary_cache: Path
    model_checkpoint: Path | None = None
    global_tip_checkpoint: Path | None = None
    global_tip_views: tuple[str, ...] = ("column",)
    device: str = "cuda:0"
    batch_size: int = 64
    candidates_per_episode: int = 8
    candidate_min_gap: int = 20
    state_retrieval_weight: float = 1.0
    retrieval_cost_weight: float = 0.05
    object_phase_weight: float = 4.0
    max_object_phase_distance: float = 0.08
    goal_final_steps: int = 3
    cartesian_goal_scale_m: float = 0.20
    pixel_goal_weight: float = 0.10
    max_first_command_jump: float = 8.0
    max_prefix_step_jump: float = 8.0
    prediction_bound_margin: float = 3.0
    fcpg_max_correction_px: float = 18.0
    # Optional V16.5 deployment bundle.  The authority and all five frozen
    # artifacts are an all-or-none startup contract; this frozen config is
    # the session authority and cannot be switched by a request.
    anchor_authority: str | None = None
    v165_original_fcpg_checkpoint: Path | None = None
    v165_source_checkpoint: Path | None = None
    v165_tuned_fcpg_checkpoint: Path | None = None
    v165_lewm_checkpoint: Path | None = None
    v165_actor_checkpoint: Path | None = None
    v165_no_contact_approach: bool = False
    v165_contact_authorization_receipt: Path | None = None
    # Optional V16.8 overlay. It deliberately does not consume a tuned V16.5
    # FCPG or legacy V16.5 Actor: original FCPG and the formally-GO State/FK
    # LeWM form the frozen foundation, and pass8 is the sole learned proposal.
    v168_mode: str | None = None
    v168_original_fcpg_checkpoint: Path | None = None
    v168_source_checkpoint: Path | None = None
    v168_lewm_checkpoint: Path | None = None
    v168_lewm_receipt: Path | None = None
    v168_training_receipt: Path | None = None
    v168_actor_checkpoint: Path | None = None
    v168_live_execute: bool = False
    # Explicit operator-supervised debug authorization.  This removes the
    # legacy V16.5 one-chunk/no-contact wrapper from a V16.8 candidate session
    # so the existing calibration, home, and bounded closed-loop entrypoints
    # can be exercised.  It does not bypass candidate legality, fresh-plan,
    # joint-bound, step-cap, projector, transport-readback, or watchdog checks.
    v168_supervised_debug: bool = False
    # Optional plain-checkpoint V17.0 stack.  It reuses the common planner,
    # ranker, projector and hardware supervisor; the model object itself has
    # no actuator surface.  No V17 checkpoint is content-addressed or bound to
    # a receipt.
    v170_mode: str | None = None
    v170_fcpg_checkpoint: Path | None = None
    v170_lewm_checkpoint: Path | None = None
    v170_actor_checkpoint: Path | None = None
    v170_live_execute: bool = False
    # Column camera-extrinsic observer.  A ~46 px Jieli shift sits outside
    # the 18 px FCPG envelope, so this translation is applied to the column
    # nominal *before* local FCPG.  Measurement is metal-tip appearance now;
    # the estimator is a 2-D complementary filter that tightens over time.
    column_extrinsic_observer: bool = True
    column_extrinsic_max_offset_px: float = 60.0
    column_extrinsic_max_residual_px: float = 8.0
    # 3-D recursive observer.  All three flags default off so the live
    # server's control loop is bit-identical until a caller opts in.
    recursive_observer_shadow: bool = False
    observer_active_excitation: bool = False
    observer_close_sigma_m: float | None = None
    observer_bundle_path: Path | None = None
    observer_drop_3d_path: Path | None = None
    # Persist Present_Load / Present_Current once per planning round.
    # Observability only: the planner, gates, and servo never read it.
    observer_servo_load: bool = False
    # Hard reject of fingertip detections that have locked onto a knuckle.
    # 10 px is the integer in the same-config gap on snapshot:metal
    # (good max 9.22 px vs mode-switch p10 11.34 px after transporting
    # the last accept by FK).  See fcpg/observer_tip_gate.py.
    fingertip_reject_radius_px: float = 10.0
    fingertip_stale_age_frames: int = 30
    fingertip_stale_dq_deg: float = 3.0
    global_tip_min_confidence: float = 0.08
    global_tip_edge_margin_px: float = 2.0
    global_tip_max_motion_disagreement_px: float = 30.0
    # How far the markerless locator may place the jaws from the FK projection
    # it replaces.  It is a coarse seed for local FCPG, not an independent
    # localizer: its own training error is 11.45 px at p90 and 99.4% recall
    # inside 24 px.  Measured over 212 accepted column rounds in the archived
    # 2026-08-15 session the disagreement is bimodal -- 95 rounds at or below
    # 38.5 px with the argmax on the jaws, 117 at or above 45.1 px with the
    # argmax on background -- and no round at all falls between.
    global_tip_max_nominal_disagreement_px: float = 40.0
    # How far a correction seed may sit outside the content rect before its
    # anchor stops being gradeable.  Half the 14 px frozen-ViT patch pitch, so
    # a tolerated seed still samples a feature cell whose centre is on real
    # camera pixels.  Measured need: at grasp range on 2026-08-15 the FK
    # projection of the fixed tip landed 1.14 px below the content bottom on
    # all 28 refused rounds while all three corrected anchors stayed inside by
    # at least 1.99 px.
    # A committed pickup identity may only be handed to a far-away candidate on
    # sustained evidence.  Measured 2026-08-15 rounds 23-35: the frozen-feature
    # fallback moved the pickup centroid 114-126 px between adjacent rounds and
    # back again, swinging the reported range 90mm/246mm, while genuine
    # round-to-round motion of a held identity stayed under 17 px.
    pickup_identity_hop_px: float = 60.0
    pickup_identity_confirmations: int = 2
    # How far the pickup pixel may drift before the committed servo goal is
    # re-derived rather than re-used.  Detector noise on a stationary object was
    # 0 px across the 20-round wobble window; a genuine re-detection moved it
    # 12 px or more.
    servo_goal_hold_tolerance_px: float = 4.0
    # A hold that never expires freezes an overshot goal.  Progress releases it
    # first; this ceiling is only the permanent-freeze backstop.
    servo_goal_hold_max_rounds: int = 12
    # Enter the live-anchor transform at radius; leave only past
    # radius*(1+band).  Measured 2026-08-16: the working point rode the radius
    # boundary (inside -1.5..-3.7mm, outside +4.8..+11.1mm) and flipped the
    # gate every round; 0.35 (~16mm at the measured 46mm radius) covers that
    # span so the hard step is no longer inside the closed loop.
    live_anchor_transform_hysteresis_band: float = 0.35
    # How much further from the jaws the wrist may place the pickup than the
    # column does, measured in each view's own FK tip separation, before the
    # two views are treated as contradicting each other.
    #
    # The column projection can go wrong without anything in the column view
    # being able to notice: the only FK-independent look at the jaws there is
    # the coarse locator, which is too weak to trust and is therefore gated
    # against the very projection it would correct.  When it does go wrong the
    # same bad transform places both the jaws and the back-projected pickup, so
    # the error cancels in their difference and the column reports contact.
    # 2026-08-17 it reported 7mm while the wrist showed the fish at the frame
    # edge, and the jaws closed on a bag.  The wrist cannot drift against the
    # gripper it is bolted to, so it is the second opinion.
    #
    # Measured on rounds where both views see the pickup: 586 archived
    # 2026-08-15 rounds against the 40 broken 2026-08-17 ones.  Restricted to
    # the held/in_corridor rounds the phase machine acts on, `wrist - column`
    # above 0.4 covers 12 of 357 archived rounds and 6 of 9 broken ones, and 9
    # of those 12 archived trips are themselves rounds where the column live
    # anchor was not applied -- the same failure, not a false alarm.  Below 0.4
    # the archived rate climbs (0.3 -> 21 rounds, 0.2 -> 26) without separating
    # the broken session any better.
    cross_view_max_offset_disagreement: float = 0.4
    # Below this the FK projection itself reports the jaws closed to within a
    # patch, so the ratio's denominator is noise and the check stands down.
    cross_view_separation_floor_px: float = 8.0
    # Planning step gain only.  Measured effective loop gain was 2.0-2.4 with
    # the frozen target at (88.9, 165.3); 0.25 is a conservative initial value
    # that puts effective gain in roughly 0.5-1.0, not a calibration result.
    column_servo_gain: float = 0.25
    # Multiplies FCPG image-servo gain/step, column_servo_gain, and GRASP
    # jaw rates.  1.0 keeps the conservative 0.65–0.95°/step caps
    # (3.25–4.75° per executed chunk) and the −0.6°/step jaw grid.
    # Live launch uses 3.0 so a winning servo_slow chunk is ~10° instead of
    # 3.25° and a −2.0 jaw row is −6°/step (one chunk from 31° open).
    # Capped at 8 so a scaled step cannot exceed max_prefix_step_jump.
    live_action_scale: float = 1.0
    workspace_margin_m: float = 0.040
    # Extra height above the demonstration transport level while carrying to
    # the drop zone: the carry must clear container walls the planner cannot
    # see (no obstacle model exists), and the release then drops the object
    # in from above.
    # 0.09 rather than 0.07: FK-vs-target tracking shows a persistent ~2cm
    # droop during the carry, so the commanded plane must overshoot for the
    # achieved height to clear container walls.
    release_clearance_m: float = 0.09
    # Eye-in-hand wrist correction (active only when a wrist pickup template
    # is registered): conservative gain + clamp + deadband, iterating with
    # re-measurement each round instead of one-shot correction.
    wrist_correction_gain: float = 0.6
    max_wrist_correction_m: float = 0.012
    wrist_correction_deadband_px: float = 12.0
    wrist_correction_min_margin: float = 0.015
    # Overwriting the description the wrist inherits is a stronger claim than
    # reporting a detection, so it takes a stronger margin. See
    # _hand_pickup_description_to_wrist for the replayed measurement.
    prototype_refresh_min_margin: float = 0.030
    # Single-view planning: minimum margin for target reliability
    single_view_min_margin: float = 0.020
    single_view_allow_held: bool = True
    max_column_servo_step_m: float = 0.040
    # Per-round vertical travel cap. The planar hop has always been bounded;
    # without this, height changes (descent, the carry lift to release
    # height) command their full extent in a single round.
    max_column_z_step_m: float = 0.030
    column_descent_radius_m: float = 0.050
    # Enter contact inside the radius; leave only past radius*(1+band).
    # A hard 50 mm step flipped standoff↔contact every round when planar
    # error rode the boundary (~21 mm Z chatter on a --chunks 2 replan).
    column_descent_hysteresis_band: float = 0.20
    command_chunk: int = 5
    ik_arrival_raw_steps: int = 20
    target_profile: Path | None = None
    task_class: int | None = None
    mpc_only_collection: bool = False  # Phase 2.5: LeWM shadow logging only, zero decision influence
    peg_clearance_m: float = 0.018
    recursive_grounding: bool = False
    # 2x2 planner arm. Defaults match the historical FCPG+LeWM path.
    grounding_mode: str = GROUNDING_GATED_FCPG
    ranking_mode: str = RANKING_ON
    # Historical live path may apply eye-in-hand correction. CoarseFine
    # forces this off so low-confidence / conflict cannot fall back to wrist.
    wrist_correction_enabled: bool = True
    allow_uncalibrated_coarse_fine_shadow: bool = False
    request_live_execute: bool = False
    memory_policy: str = "auto"
    require_live_column_calibration: bool = False
    # H2 changes only which candidates the assigned method may select. The
    # execution-time supervisor always receives the registered context.
    experiment_method: str = "full"
    # Target-conditioned control is the deployable path. The demonstration
    # trajectory mode remains available for replay and temporal-contract tests.
    control_mode: str = "phase_target"
    # B1 pixel-geometry contract.  The defaults reproduce the historical
    # squash_v0 semantics exactly, so every legacy / offline / replay caller
    # keeps bit-identical behaviour; only the live hardware path opts into
    # letterbox_v1.  Geometry is never inferred from a 224x224 frame shape:
    # both protocols produce that same canvas, so the shape carries no
    # information about which one preprocessed the pixels.
    geometry_protocol: str = SQUASH_V0
    # Required for letterbox_v1 (the pad band cannot be derived without it)
    # and unused by squash_v0.
    source_image_size: tuple[int, int] | None = None
    planning_image_size: int = PLANNING_IMAGE_SIZE

    def __post_init__(self) -> None:
        if self.device.startswith("cuda:1"):
            raise ValueError("CUDA1 is forbidden for this project")
        v165_paths = (
            self.v165_original_fcpg_checkpoint,
            self.v165_source_checkpoint,
            self.v165_tuned_fcpg_checkpoint,
            self.v165_lewm_checkpoint,
            self.v165_actor_checkpoint,
        )
        v168_paths = (
            self.v168_original_fcpg_checkpoint,
            self.v168_source_checkpoint,
            self.v168_lewm_checkpoint,
            self.v168_lewm_receipt,
            self.v168_training_receipt,
            self.v168_actor_checkpoint,
        )
        v170_paths = (
            self.v170_fcpg_checkpoint,
            self.v170_lewm_checkpoint,
            self.v170_actor_checkpoint,
        )
        if self.v170_mode is None:
            if any(path is not None for path in v170_paths) or self.v170_live_execute:
                raise ValueError("V17.0 checkpoints/live flag require v170_mode")
        else:
            if self.v170_mode not in {"shadow", "candidate"}:
                raise ValueError("v170_mode must be shadow or candidate")
            if any(path is None for path in v170_paths):
                raise ValueError("V17.0 requires final FCPG, LeWM, and Actor checkpoints")
            if any(path is not None for path in (*v165_paths, *v168_paths)):
                raise ValueError("V17.0 cannot coexist with a V16.5/V16.8 runtime bundle")
            if self.anchor_authority is not None:
                raise ValueError("V17.0 owns F/M/C and must not select a legacy anchor authority")
            if self.control_mode != "phase_target":
                raise ValueError("V17.0 live integration requires control_mode=phase_target")
            if self.ranking_mode != RANKING_ON or self.mpc_only_collection:
                raise ValueError("V17.0 requires active common candidate ranking")
            if self.v170_mode == "shadow":
                assert_v170_shadow_bootstrap(
                    execute_enabled=self.request_live_execute,
                    mode=self.v170_mode,
                )
                if self.v170_live_execute:
                    raise ValueError("V17.0 shadow cannot enable live execution")
            elif self.v170_live_execute != bool(self.request_live_execute):
                raise ValueError(
                    "V17.0 candidate execution requires both request_live_execute "
                    "and v170_live_execute"
                )
        if self.v168_mode is not None:
            if self.v168_mode not in {"shadow", "candidate"}:
                raise ValueError("v168_mode must be shadow or candidate")
            if any(path is None for path in v168_paths):
                raise ValueError("V16.8 requires all original-FCPG/LeWM/Actor artifacts")
            if any(path is not None for path in v165_paths):
                raise ValueError("V16.8 must not load a tuned or legacy V16.5 Actor bundle")
            if self.v165_contact_authorization_receipt is not None:
                raise ValueError("V16.8 has no contact-hardware authorization receipt")
            if self.anchor_authority != "original_fcpg":
                raise ValueError("V16.8 requires anchor_authority=original_fcpg")
            if self.control_mode != "phase_target":
                raise ValueError("V16.8 requires control_mode=phase_target")
            if self.grounding_mode != GROUNDING_GATED_FCPG:
                raise ValueError("V16.8 requires grounding_mode=gated_fcpg")
            if self.ranking_mode != RANKING_ON or self.mpc_only_collection:
                raise ValueError("V16.8 requires active frozen State/FK LeWM ranking")
            if self.observer_active_excitation or self.observer_close_sigma_m is not None:
                raise ValueError("V16.8 keeps legacy observer controls decision-off")
            if self.v168_mode == "shadow" and self.request_live_execute:
                raise ValueError("V16.8 shadow requires commands_sent=0")
            if self.v168_mode == "shadow" and self.v168_supervised_debug:
                raise ValueError("V16.8 shadow cannot enable supervised debug motion")
            if self.v168_live_execute and (
                self.v168_mode != "candidate" or not self.request_live_execute
            ):
                raise ValueError(
                    "v168_live_execute requires candidate mode and live execute"
                )
            if self.request_live_execute and not self.v168_live_execute:
                raise ValueError("V16.8 live execute needs a separate explicit flag")
            if self.v168_supervised_debug and (
                self.v168_mode != "candidate"
                or not self.request_live_execute
                or not self.v168_live_execute
            ):
                raise ValueError(
                    "V16.8 supervised debug requires candidate live execute"
                )
            if self.v168_supervised_debug and self.v165_no_contact_approach:
                raise ValueError(
                    "V16.8 supervised debug and legacy no-contact mode conflict"
                )
            if (
                self.request_live_execute
                and not self.v165_no_contact_approach
                and not self.v168_supervised_debug
            ):
                raise ValueError(
                    "V16.8 live execute requires no-contact or supervised debug"
                )
        elif self.anchor_authority is None:
            if any(path is not None for path in (*v165_paths, *v168_paths)):
                raise ValueError(
                    "runtime checkpoint paths require a session anchor_authority"
                )
            if self.v165_no_contact_approach:
                raise ValueError(
                    "v165_no_contact_approach requires a complete runtime bundle"
                )
            if self.v165_contact_authorization_receipt is not None:
                raise ValueError(
                    "V16.5 contact authorization requires a complete runtime bundle"
                )
            if self.v168_live_execute:
                raise ValueError("v168_live_execute requires a V16.8 overlay")
            if self.v168_supervised_debug:
                raise ValueError("v168_supervised_debug requires a V16.8 overlay")
        else:
            if (
                any(path is not None for path in v168_paths)
                or self.v168_live_execute
                or self.v168_supervised_debug
            ):
                raise ValueError("V16.8 artifacts/flags require v168_mode")
            if self.anchor_authority not in V165_ANCHOR_AUTHORITIES:
                raise ValueError(
                    "anchor_authority must be nominal, original_fcpg, or tuned_fcpg"
                )
            if any(path is None for path in v165_paths):
                raise ValueError(
                    "anchor_authority requires original/tuned FCPG, V12 source, LeWM, and Actor checkpoints"
                )
            if self.control_mode != "phase_target":
                raise ValueError(
                    "V16.5 runtime requires control_mode=phase_target"
                )
            if self.grounding_mode != GROUNDING_GATED_FCPG:
                raise ValueError(
                    "V16.5 runtime uses anchor_authority and requires grounding_mode=gated_fcpg"
                )
            if self.ranking_mode != RANKING_ON or self.mpc_only_collection:
                raise ValueError(
                    "V16.5 runtime requires active LeWM ranking"
                )
            if self.observer_active_excitation or self.observer_close_sigma_m is not None:
                raise ValueError(
                    "V16.5 keeps legacy observer excitation/sigma close control decision-off"
                )
            if self.v165_no_contact_approach and self.v165_contact_authorization_receipt is not None:
                raise ValueError(
                    "V16.5 no-contact and contact-authorized modes are mutually exclusive"
                )
            if self.v165_contact_authorization_receipt is not None:
                raise ValueError(
                    "V16.5 contact hardware remains sealed until post-training integration review"
                )
            if (
                self.request_live_execute
                and not self.v165_no_contact_approach
            ):
                raise ValueError(
                    "V16.5 live execute is currently restricted to the no-contact approach"
                )
        positive = (
            self.batch_size,
            self.candidates_per_episode,
            self.goal_final_steps,
            self.cartesian_goal_scale_m,
            self.max_first_command_jump,
            self.max_prefix_step_jump,
            self.fcpg_max_correction_px,
            self.column_extrinsic_max_offset_px,
            self.column_extrinsic_max_residual_px,
            self.global_tip_max_motion_disagreement_px,
            self.global_tip_max_nominal_disagreement_px,
            self.workspace_margin_m,
            self.max_column_servo_step_m,
            self.max_column_z_step_m,
            self.column_descent_radius_m,
            self.command_chunk,
            self.ik_arrival_raw_steps,
            self.peg_clearance_m,
            self.wrist_correction_gain,
            self.max_wrist_correction_m,
            self.wrist_correction_deadband_px,
            self.wrist_correction_min_margin,
            self.prototype_refresh_min_margin,
            self.single_view_min_margin,
            self.cross_view_max_offset_disagreement,
            self.cross_view_separation_floor_px,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("planner configuration values must be positive")
        if (
            self.column_extrinsic_max_offset_px > 80.0
            or self.column_extrinsic_max_residual_px > 16.0
        ):
            raise ValueError(
                "column calibration bounds are capped at 80 px offset and "
                "16 px RMS residual"
            )
        if (
            self.column_extrinsic_max_offset_px > 60.0
            or self.column_extrinsic_max_residual_px > 8.0
        ) and not (
            self.v168_mode == "candidate" and self.v168_supervised_debug
        ):
            raise ValueError(
                "widened column calibration bounds require V16.8 "
                "candidate supervised debug"
            )
        if not np.isfinite(self.pixel_goal_weight) or self.pixel_goal_weight < 0.0:
            raise ValueError("pixel_goal_weight must be finite and non-negative")
        if (
            not np.isfinite(self.pickup_identity_hop_px)
            or self.pickup_identity_hop_px <= 0.0
            or not np.isfinite(self.servo_goal_hold_tolerance_px)
            or self.servo_goal_hold_tolerance_px < 0.0
            or self.pickup_identity_confirmations < 1
            or self.servo_goal_hold_max_rounds < 1
        ):
            raise ValueError("pickup identity hold values must be positive")
        if (
            not np.isfinite(self.column_servo_gain)
            or self.column_servo_gain <= 0.0
            or self.column_servo_gain > 1.0
        ):
            raise ValueError("column_servo_gain must be in (0, 1]")
        if (
            not np.isfinite(self.live_action_scale)
            or self.live_action_scale <= 0.0
            or self.live_action_scale > 8.0
        ):
            raise ValueError("live_action_scale must be in (0, 8]")
        if (
            not np.isfinite(self.live_anchor_transform_hysteresis_band)
            or self.live_anchor_transform_hysteresis_band < 0.0
        ):
            raise ValueError(
                "live_anchor_transform_hysteresis_band must be finite and "
                "non-negative"
            )
        if (
            not np.isfinite(self.column_descent_hysteresis_band)
            or self.column_descent_hysteresis_band < 0.0
        ):
            raise ValueError(
                "column_descent_hysteresis_band must be finite and non-negative"
            )
        if (
            not np.isfinite(self.global_tip_min_confidence)
            or self.global_tip_min_confidence < 0.0
            or not np.isfinite(self.global_tip_edge_margin_px)
            or self.global_tip_edge_margin_px < 0.0
        ):
            raise ValueError("global tip gate values must be finite and non-negative")
        if self.grounding_mode not in {GROUNDING_NOMINAL, GROUNDING_GATED_FCPG}:
            raise ValueError(
                "grounding_mode must be "
                f"{GROUNDING_NOMINAL!r} or {GROUNDING_GATED_FCPG!r}"
            )
        if self.ranking_mode not in {RANKING_OFF, RANKING_ON}:
            raise ValueError(
                f"ranking_mode must be {RANKING_OFF!r} or {RANKING_ON!r}"
            )
        if self.memory_policy not in {"auto", "off", "always"}:
            raise ValueError("memory_policy must be auto, off, or always")
        if self.control_mode not in {"anchor_trajectory", "phase_target"}:
            raise ValueError("control_mode must be anchor_trajectory or phase_target")
        if self.mpc_only_collection and self.control_mode == "anchor_trajectory":
            # anchor_trajectory's workspace veto is prediction-only: it has no
            # commanded-trajectory conjunct to fall back on, so LeWM influence
            # cannot be removed there by dropping a term.  Synthesising a new
            # commanded check would be an unvalidated behavioural change, and
            # silently proceeding would break the guarantee that
            # mpc_only_collection means zero LeWM decision influence.  Refuse
            # the combination instead of weakening either property.
            raise ValueError(
                "mpc_only_collection is not supported with "
                "control_mode=anchor_trajectory: that branch's workspace safety "
                "is prediction-only, so LeWM decision influence cannot be "
                "removed without adding an unvalidated commanded-safety path. "
                "Use control_mode=phase_target for MPC-only collection."
            )
        if self.experiment_method not in {"full", "no_context"}:
            raise ValueError("experiment_method must be full or no_context")
        if (
            not self.global_tip_views
            or len(set(self.global_tip_views)) != len(self.global_tip_views)
            or any(view not in VIEWS for view in self.global_tip_views)
        ):
            raise ValueError("global_tip_views must be unique SO-101 camera views")
        if self.ik_arrival_raw_steps % self.command_chunk:
            raise ValueError("ik_arrival_raw_steps must align to command_chunk")
        if self.geometry_protocol not in {SQUASH_V0, LETTERBOX_V1}:
            raise ValueError(
                f"geometry_protocol must be {SQUASH_V0!r} or {LETTERBOX_V1!r}, "
                f"got {self.geometry_protocol!r}"
            )
        if self.planning_image_size <= 0:
            raise ValueError("planning_image_size must be positive")
        if self.geometry_protocol == LETTERBOX_V1:
            if self.source_image_size is None:
                raise ValueError(
                    f"geometry_protocol={LETTERBOX_V1!r} requires "
                    "source_image_size: the padding band is a function of the "
                    "raw camera aspect ratio and cannot be recovered from the "
                    f"{self.planning_image_size}x{self.planning_image_size} "
                    "canvas alone"
                )
            width, height = self.source_image_size
            if int(width) <= 0 or int(height) <= 0:
                raise ValueError("source_image_size must be positive")
        # Build once so an invalid combination fails at construction time
        # rather than on the first live frame.
        self.planning_geometry

    @property
    def planning_geometry(self) -> PlanningGeometry:
        """Canonical pixel geometry for this configuration.

        Never inferred: the protocol is declared, and ``letterbox_v1``
        additionally carries the raw source size that fixes the pad band.
        """

        source = self.source_image_size
        return planning_geometry_from_protocol(
            self.geometry_protocol,
            source_width=None if source is None else int(source[0]),
            source_height=None if source is None else int(source[1]),
            image_size=int(self.planning_image_size),
        )

    def require_source_image_size(
        self, observed: Mapping[str, tuple[int, int]]
    ) -> None:
        """Fail closed unless every live view matches the declared raw size.

        One ``source_image_size`` serves both views, which is only sound while
        the cameras share a raw resolution.  That assumption is made explicit
        here rather than left implicit: the pad band is a function of the raw
        aspect ratio, so a view at a different resolution would be scored with
        the wrong content rectangle -- silently, because both protocols yield
        the same 224x224 canvas.  Under ``squash_v0`` there is no pad band and
        no source dependence, so an undeclared size is not an error there.
        """

        declared = self.source_image_size
        if declared is None:
            if self.geometry_protocol != SQUASH_V0:
                raise ValueError(
                    f"geometry_protocol={self.geometry_protocol!r} without a "
                    "declared source_image_size cannot be checked against live "
                    "cameras"
                )
            return
        expected = (int(declared[0]), int(declared[1]))
        missing = [view for view in VIEWS if view not in observed]
        if missing:
            raise ValueError(
                "live source resolution check requires every view; missing: "
                + ", ".join(sorted(missing))
            )
        mismatched = {
            view: (int(observed[view][0]), int(observed[view][1]))
            for view in VIEWS
            if (int(observed[view][0]), int(observed[view][1])) != expected
        }
        if mismatched:
            detail = "; ".join(
                f"{view}={size[0]}x{size[1]}"
                for view, size in sorted(mismatched.items())
            )
            raise ValueError(
                f"live camera geometry mismatch: declared source_image_size="
                f"{expected[0]}x{expected[1]} for geometry_protocol="
                f"{self.geometry_protocol!r}, but {detail}. A single planning "
                "geometry serves both views, so differing raw resolutions "
                "would be scored with the wrong content rectangle. Re-declare "
                "the geometry or restore the expected camera mode; the padding "
                "is never recomputed per view here."
            )


def planning_geometry_contract_fields(
    config: PreparedPlannerConfig,
) -> dict[str, Any]:
    """Build the pixel-geometry provenance block for ``model_contract``.

    Recorded because a ``plan.json`` is otherwise indistinguishable between the
    two protocols: every frame, heatmap and pixel cost in it is ``224x224``
    under both, so an archived session cannot be told apart after the fact
    without this.  ``content_rect`` is derived here rather than left to the
    reader -- it is the half-open ``(left, top, right, bottom)`` box of real
    camera content, and the pad band outside it carries no camera data however
    finite its coordinates look.

    One construction site, mirroring
    :func:`lewm_decision_independence_fields`, so the four fields cannot drift
    apart or be emitted with a geometry the config does not actually hold.
    """

    source = config.source_image_size
    return {
        "geometry_protocol": config.geometry_protocol,
        "source_image_size": (
            None if source is None else [int(source[0]), int(source[1])]
        ),
        "planning_image_size": int(config.planning_image_size),
        "content_rect": list(config.planning_geometry.content_rect),
    }


#: ``model_contract`` fields that are published by value inside ``__init__`` but
#: whose backing attribute can be rebound afterwards -- notably
#: ``--trial-contact-height-m``, which the server applies at
#: ``serve_so101_realtime_fcpg_lewm.py:3741-3748``, i.e. AFTER the planner (and
#: therefore the contract dict) already exists.  Every planning site reads the
#: live attribute, so publishing the frozen value makes ``plan.json`` report a
#: height the planner never used.  Republished from the live attribute at
#: plan-publication time by :meth:`PreparedSO101RealtimePlanner.live_contract_fields`.
LIVE_CONTRACT_HEIGHT_FIELDS: tuple[str, ...] = (
    "contact_height_m",
    "transport_height_m",
    "release_height_m",
)


@dataclass(frozen=True)
class CausalPlanningMemory:
    """Older decision-boundary observations aligned with executed action chunks."""

    frames: dict[str, np.ndarray]
    states: np.ndarray
    raw_actions: np.ndarray
    offsets_raw: np.ndarray

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=np.float32)
        raw_actions = np.asarray(self.raw_actions, dtype=np.float32)
        offsets = np.asarray(self.offsets_raw, dtype=np.int64)
        if states.ndim != 2 or states.shape[1:] != (6,) or len(states) == 0:
            raise ValueError("memory states must have shape [M,6]")
        if raw_actions.shape != (len(states), 5, 6):
            raise ValueError("memory raw_actions must have shape [M,5,6]")
        if offsets.shape != (len(states),):
            raise ValueError("memory offsets_raw must have shape [M]")
        if np.any(offsets <= 0) or np.any(np.diff(offsets) <= 0):
            raise ValueError("memory offsets_raw must be positive and increasing")
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(raw_actions)):
            raise ValueError("memory states and actions must be finite")
        if set(self.frames) != set(VIEWS):
            raise ValueError("memory frames must contain wrist and column views")
        for view in VIEWS:
            values = np.asarray(self.frames[view], dtype=np.uint8)
            if values.shape != (len(states), 224, 224, 3):
                raise ValueError(f"{view} memory frames have an incompatible shape")


@dataclass(frozen=True)
class DemonstrationProgress:
    """Next action-prefix location after a completed real command chunk."""

    episode: int
    next_frame: int
    completed_chunks: int


@dataclass
class RealtimePlan:
    guard: PlanningRoundGuard
    phase: ManipulationPhase
    phase_status: dict[str, Any]
    current_state: np.ndarray
    frames: dict[str, np.ndarray]
    nominal: dict[str, np.ndarray]
    correction_seed: dict[str, np.ndarray]
    corrected: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]
    grounding_diagnostics: dict[str, dict[str, Any]]
    target_descriptors: dict[str, dict[str, Any]]
    target_profile: dict[str, Any]
    target: dict[str, Any]
    selected_index: int | None
    selected_chunk: np.ndarray | None
    selected_prefix: np.ndarray | None
    predicted_states: np.ndarray | None
    commanded_states: np.ndarray | None
    predicted_gripper_px: np.ndarray | None
    commanded_gripper_px: np.ndarray | None
    predicted_anchor_px: np.ndarray | None
    commanded_anchor_px: np.ndarray | None
    # view -> {"predicted"|"commanded": [H, 3, 2]}. The two fields above are the
    # column entry of this map, kept because the JSON round record publishes
    # them under those names.
    anchor_paths_px: dict[str, dict[str, np.ndarray]]
    candidate_rows: list[dict[str, Any]]
    executable: bool
    blocked_reasons: tuple[str, ...]
    model_contract: dict[str, Any]
    #: How far the previous round's predicted scene was from the one that
    #: arrived, per view. ``None`` before any expectation has been settled.
    world_model_residual: dict[str, Any] | None = None
    #: Why the eye-in-hand correction stage did or did not fire this round, and
    #: how much of its output the cap removed. Recorded, never acted on.
    wrist_correction: dict[str, Any] | None = None
    #: Raw automatic-H0/CoTracker/seed13161 posterior history. Once automatic
    #: identity is established, its current row is also the control-time tip
    #: measurement; the history itself remains display/artifact data.
    prod_v13_tip_trajectory: dict[str, Any] | None = None
    #: The exact safe-check input bank, retained only for default-off V14
    #: shadow ranking. It is never serialized and never changes selected_chunk.
    candidate_chunks: np.ndarray | None = None
    #: Post-plan V14 counterfactual. It cannot alter this plan's command.
    prod_v14_shadow: dict[str, Any] | None = None
    #: Post-plan V16.1 causal secant proposal. It is diagnostic-only and is
    #: never inserted into the executable candidate bank.
    prod_v16_secant_shadow: dict[str, Any] | None = None

    def lewm_selection_effect(self) -> dict[str, Any] | None:
        """Whether the world model changed which candidate was executed.

        Derived from what was already recorded, so it costs the round nothing
        and cannot disagree with the ranking the planner used. ``None`` when the
        world model was not consulted, since then there is nothing to compare.
        """
        rows = self.candidate_rows
        if not rows or any(row.get("mpc_only_cost") is None for row in rows):
            return None
        costs = np.asarray(
            [float(row["mpc_only_cost"]) for row in rows], dtype=np.float64
        )
        without = counterfactual_selection_without_lewm(
            mpc_only_costs=costs,
            decision_reasons=[tuple(row.get("reasons") or ()) for row in rows],
        )
        return {
            "selected_index": self.selected_index,
            "selected_index_without_lewm": without,
            "changed": None
            if (without is None or self.selected_index is None)
            else bool(without != self.selected_index),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "round_id": self.guard.round_id,
            "phase": self.phase.value,
            "phase_status": self.phase_status,
            "current_state": self.current_state,
            "target": self.target,
            "selected_index": self.selected_index,
            "selected_chunk": self.selected_chunk,
            "selected_prefix": self.selected_prefix,
            "predicted_states": self.predicted_states,
            "commanded_states": self.commanded_states,
            "predicted_gripper_px": self.predicted_gripper_px,
            "commanded_gripper_px": self.commanded_gripper_px,
            "predicted_anchor_px": self.predicted_anchor_px,
            "commanded_anchor_px": self.commanded_anchor_px,
            "candidates": self.candidate_rows,
            "lewm_selection_effect": self.lewm_selection_effect(),
            "world_model_residual": self.world_model_residual,
            "prod_v14_shadow": self.prod_v14_shadow,
            "prod_v16_secant_shadow": self.prod_v16_secant_shadow,
            "executable": self.executable,
            "blocked_reasons": self.blocked_reasons,
            "model_contract": self.model_contract,
            "grounding": {
                view: {
                    "nominal": (
                        None
                        if self.grounding_diagnostics[view].get(
                            "legacy_v12_pixel_head_executed"
                        )
                        is False
                        else self.nominal[view][-1]
                    ),
                    "correction_seed": self.correction_seed[view][-1],
                    "corrected": self.corrected[view][-1],
                    "valid": self.valid[view][-1],
                    "prior_source": self.grounding_diagnostics[view][
                        "prior_source"
                    ],
                    "mean_local_correction_px": float(
                        np.mean(
                            np.linalg.norm(
                                self.corrected[view][-1]
                                - self.correction_seed[view][-1],
                                axis=-1,
                            )
                        )
                    ),
                    "mean_total_correction_px": (
                        None
                        if self.grounding_diagnostics[view].get(
                            "legacy_v12_pixel_head_executed"
                        )
                        is False
                        else float(
                            np.mean(
                                np.linalg.norm(
                                    self.corrected[view][-1]
                                    - self.nominal[view][-1],
                                    axis=-1,
                                )
                            )
                        )
                    ),
                    "mean_correction_px": (
                        None
                        if self.grounding_diagnostics[view].get(
                            "legacy_v12_pixel_head_executed"
                        )
                        is False
                        else float(
                            np.mean(
                                np.linalg.norm(
                                    self.corrected[view][-1]
                                    - self.nominal[view][-1],
                                    axis=-1,
                                )
                            )
                        )
                    ),
                    "fk_delta_px": self.grounding_diagnostics[view].get(
                        "fk_delta_px"
                    ),
                    "posterior_offset_px": self.grounding_diagnostics[view].get(
                        "posterior_offset_px"
                    ),
                    "global_locator": {
                        "enabled": self.grounding_diagnostics[view].get(
                            "global_locator_enabled", False
                        ),
                        "accepted": self.grounding_diagnostics[view].get(
                            "global_locator_accepted", False
                        ),
                        "rejection_reasons": self.grounding_diagnostics[view].get(
                            "global_locator_rejection_reasons", ()
                        ),
                        "confidence": self.grounding_diagnostics[view].get(
                            "global_locator_confidence"
                        ),
                        "quality": self.grounding_diagnostics[view].get(
                            "global_locator_quality"
                        ),
                        # Without the decoded pixels a rejected locator leaves
                        # no record of WHERE it wanted to put the tips, so a
                        # disagreement with the projection cannot be measured
                        # after the fact.
                        "points_px": self.grounding_diagnostics[view].get(
                            "global_locator_points_px"
                        ),
                    },
                    "live_anchor_transform": self.grounding_diagnostics[view].get(
                        "live_anchor_transform", {"present": False, "applied": False}
                    ),
                    "column_extrinsic": self.grounding_diagnostics[view].get(
                        "column_extrinsic"
                    ),
                    "recursive_observer": self.grounding_diagnostics[view].get(
                        "recursive_observer"
                    ),
                    "coarse_fine": self.grounding_diagnostics[view].get(
                        "coarse_fine"
                    ),
                }
                for view in VIEWS
            },
            "targets": self.target_descriptors,
            "target_profile": self.target_profile,
            "wrist_correction": self.wrist_correction,
            "prod_v13_tip_trajectory": self.prod_v13_tip_trajectory,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [_jsonable(item) for item in value.tolist()]
        if value.shape == ():
            item = value.item()
            return None if item is None else _jsonable(item)
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ManipulationPhase):
        return value.value
    return value


def plan_json(plan: RealtimePlan) -> dict[str, Any]:
    return _jsonable(plan.summary())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def command_prefix_context_clearance(
    kinematics: SO101MuJoCoKinematics,
    current_state: np.ndarray,
    command_prefixes: np.ndarray,
    context_xy_m: np.ndarray,
    *,
    required_clearance_m: float,
) -> dict[str, np.ndarray]:
    """Audit nominal three-anchor command polylines against one context point."""

    current = np.asarray(current_state, dtype=np.float64)
    prefixes = np.asarray(command_prefixes, dtype=np.float64)
    context = np.asarray(context_xy_m, dtype=np.float64)
    if current.shape != (6,) or not np.all(np.isfinite(current)):
        raise ValueError("current_state must be one finite six-joint vector")
    if (
        prefixes.ndim != 3
        or prefixes.shape[0] < 1
        or prefixes.shape[1] < 1
        or prefixes.shape[2] != 6
        or not np.all(np.isfinite(prefixes))
    ):
        raise ValueError("command_prefixes must be finite [N,T,6]")
    if context.shape != (2,) or not np.all(np.isfinite(context)):
        raise ValueError("context_xy_m must be one finite XY point")
    if not np.isfinite(required_clearance_m) or required_clearance_m <= 0.0:
        raise ValueError("required_clearance_m must be finite and positive")

    candidates, steps = prefixes.shape[:2]
    initial = np.broadcast_to(current[None, None], (candidates, 1, 6))
    path = np.concatenate((initial, prefixes), axis=1)
    anchors = np.asarray(
        kinematics.forward_anchors(path.reshape(-1, 6)), dtype=np.float64
    ).reshape(candidates, steps + 1, 3, 3)
    start = anchors[:, :-1, :, :2]
    end = anchors[:, 1:, :, :2]
    segment = end - start
    point = context[None, None, None, :]
    squared_length = np.square(segment).sum(axis=-1)
    fraction = np.divide(
        ((point - start) * segment).sum(axis=-1),
        squared_length,
        out=np.zeros_like(squared_length),
        where=squared_length > 1.0e-16,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = start + fraction[..., None] * segment
    distance = np.linalg.norm(point - closest, axis=-1)
    per_anchor = distance.min(axis=1)
    minimum = per_anchor.min(axis=1)
    return {
        "minimum_distance_m": minimum.astype(np.float32),
        "per_anchor_minimum_distance_m": per_anchor.astype(np.float32),
        "safe": (minimum >= float(required_clearance_m)),
    }


def merge_extra_actor_prefixes(
    prefixes: np.ndarray,
    candidate_ik_safe: np.ndarray,
    retrieval_cost: np.ndarray,
    selected_phase_distance: np.ndarray,
    candidate_metadata: list[dict[str, Any]],
    extra_prefixes: np.ndarray,
    extra_metadata: list[dict[str, Any]],
    *,
    prepend: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    """Merge actor prefixes into the existing common LeWM ranking batch."""

    extra = np.asarray(extra_prefixes, dtype=np.float32)
    existing = np.asarray(prefixes, dtype=np.float32)
    if (
        extra.ndim != 3
        or extra.shape[1] != existing.shape[1]
        or extra.shape[2] != 6
        or not np.all(np.isfinite(extra))
    ):
        raise ValueError(
            "extra actor prefixes must be finite "
            f"[N,{existing.shape[1]},6], got {extra.shape}"
        )
    if len(extra_metadata) != len(extra):
        raise ValueError("extra actor metadata count must match prefixes")
    actor_safe = np.ones(len(extra), dtype=bool)
    actor_retrieval = np.zeros(len(extra), dtype=np.float32)
    actor_distance = np.zeros(len(extra), dtype=np.float32)
    if prepend:
        return (
            np.concatenate((extra, existing), axis=0),
            np.concatenate((actor_safe, candidate_ik_safe)),
            np.concatenate((actor_retrieval, retrieval_cost)),
            np.concatenate((actor_distance, selected_phase_distance)),
            [*list(extra_metadata), *candidate_metadata],
        )
    return (
        np.concatenate((existing, extra), axis=0),
        np.concatenate((candidate_ik_safe, actor_safe)),
        np.concatenate((retrieval_cost, actor_retrieval)),
        np.concatenate((selected_phase_distance, actor_distance)),
        [*candidate_metadata, *list(extra_metadata)],
    )


def experiment_candidate_selection_mask(
    base_safe: np.ndarray,
    peg_safe: np.ndarray,
    *,
    experiment_method: str,
) -> np.ndarray:
    """Apply context only at the assigned method-selection boundary."""

    base = np.asarray(base_safe)
    peg = np.asarray(peg_safe)
    if (
        base.ndim != 1
        or peg.shape != base.shape
        or base.dtype != np.bool_
        or peg.dtype != np.bool_
    ):
        raise ValueError("candidate safety masks must be matching boolean vectors")
    if experiment_method == "full":
        return base & peg
    if experiment_method == "no_context":
        return base.copy()
    raise ValueError("experiment_method must be full or no_context")


@dataclass(frozen=True)
class CandidateDecision:
    """The decision-driving outputs of candidate composition.

    Every field here either gates execution or ranks candidates.  Shadow
    diagnostics are deliberately absent: this object is what the planner acts
    on, so an independence test that compares two of these compares exactly the
    quantities that can change robot behaviour.
    """

    base_safe: np.ndarray
    eligible: np.ndarray
    costs: np.ndarray
    decision_reasons: tuple[tuple[str, ...], ...]
    #: ``base_safe & peg_safe``, the supervisor's own precheck.  It is a
    #: decision-bearing field rather than a diagnostic, so it belongs here and
    #: must be compared by the independence test.
    supervisor_precheck_safe: np.ndarray
    #: The goal-cost variants that actually entered ``costs``.  Reporting reads
    #: these rather than re-selecting, so an artifact can never disagree with
    #: the cost the planner ranked on.
    cartesian_goal_cost: np.ndarray
    pixel_goal_cost: np.ndarray


def first_chunk_failclosed_cartesian_goal_cost(
    cartesian_goal_cost: np.ndarray,
    *,
    first_chunk_target_progress_m: np.ndarray | None = None,
    terminal_target_progress_m: np.ndarray | None = None,
    command_first_chunk_cartesian_goal_cost: np.ndarray | None = None,
    model_command_max_xyz_gap_m: np.ndarray | None = None,
    model_command_xyz_gap_limit_m: float = MODEL_COMMAND_XYZ_GAP_LIMIT_M,
) -> np.ndarray:
    """Rewrite terminal cartesian cost so an unexecuted tail cannot win.

    Execute sends ``selected_prefix[:5]`` and replans.  LeWM ranking that
    scores predicted-terminal cartesian therefore buys progress that never
    runs.  This leaves the ``0.05 * model_command_cost`` term untouched and
    only rewrites the cartesian vector:

    * where command-side first-chunk cartesian is finite, take
      ``max(terminal, first_chunk)``;
    * where first-chunk and terminal progress are both finite, scale by
      ``terminal / first`` so a 2.4 mm / 31 mm prefix cannot keep a 0.021
      terminal score.  Rows with missing progress (servo_*) stay unchanged.
    """

    ranked = np.asarray(cartesian_goal_cost, dtype=np.float64).copy()
    if ranked.ndim != 1:
        raise ValueError("cartesian_goal_cost must be a 1-D vector")
    count = int(ranked.shape[0])

    if command_first_chunk_cartesian_goal_cost is not None:
        chunk = np.asarray(command_first_chunk_cartesian_goal_cost, dtype=np.float64)
        if chunk.shape != (count,):
            raise ValueError(
                "command_first_chunk_cartesian_goal_cost must match candidates"
            )
        take = np.isfinite(chunk)
        ranked = np.where(take, np.maximum(ranked, chunk), ranked)

    rewrite = np.zeros(count, dtype=bool)
    if model_command_max_xyz_gap_m is not None:
        gap = np.asarray(model_command_max_xyz_gap_m, dtype=np.float64)
        if gap.shape != (count,):
            raise ValueError("model_command_max_xyz_gap_m must match candidates")
        rewrite |= np.isfinite(gap) & (gap >= float(model_command_xyz_gap_limit_m))

    if first_chunk_target_progress_m is None and terminal_target_progress_m is None:
        return ranked
    if first_chunk_target_progress_m is None or terminal_target_progress_m is None:
        raise ValueError("first-chunk and terminal progress must be passed together")

    first = np.asarray(first_chunk_target_progress_m, dtype=np.float64)
    term = np.asarray(terminal_target_progress_m, dtype=np.float64)
    if first.shape != (count,) or term.shape != (count,):
        raise ValueError("progress vectors must match candidates")

    known = np.isfinite(first) & np.isfinite(term)
    rewrite |= known & ((term - first) >= IK_RECEDING_MISMATCH_MIN_M)
    untrusted = known & ((term <= 0.0) | (first <= 0.0))
    ranked = np.where(untrusted, np.inf, ranked)
    usable = rewrite & known & (term > 0.0) & (first > 0.0)
    if np.any(usable):
        ratio = first[usable] / term[usable]
        scale = 1.0 / np.minimum(np.maximum(ratio, 1.0e-6), 1.0)
        ranked[usable] = ranked[usable] * scale
    return ranked


def compose_candidate_decision(
    *,
    hard_command_safe: np.ndarray,
    candidate_ik_safe: np.ndarray,
    phase_safe: np.ndarray,
    phase_distance_safe: np.ndarray,
    command_workspace_safe: np.ndarray,
    tool_travel_safe: np.ndarray,
    chunk_safe: np.ndarray,
    chunk_reasons: tuple[tuple[str, ...], ...],
    peg_safe: np.ndarray,
    peg_reason: str,
    lewm_predicted_workspace_safe: np.ndarray,
    phase_prediction_safe: np.ndarray,
    command_cartesian_goal_cost: np.ndarray,
    lewm_cartesian_goal_cost: np.ndarray,
    command_pixel_goal_cost: np.ndarray,
    lewm_pixel_goal_cost: np.ndarray,
    retrieval_cost: np.ndarray,
    model_command_cost: np.ndarray,
    pixel_goal_weight: float,
    retrieval_cost_weight: float,
    experiment_method: str,
    mpc_only_collection: bool,
    lewm_ranking: bool | None = None,
    closure_cost: np.ndarray | None = None,
    closure_cost_weight: float = 0.0,
    first_chunk_target_progress_m: np.ndarray | None = None,
    terminal_target_progress_m: np.ndarray | None = None,
    command_first_chunk_cartesian_goal_cost: np.ndarray | None = None,
    model_command_max_xyz_gap_m: np.ndarray | None = None,
) -> CandidateDecision:
    """Compose candidate eligibility, ranking cost and decision reasons.

    Pure: it consumes already-computed safety masks and cost vectors and never
    computes or reads a model prediction itself.  That is what makes it
    testable -- an independence test can hold every command-side input fixed,
    vary every LeWM-side input arbitrarily, and check that nothing observable
    moves under ``mpc_only_collection``.

    LeWM-side inputs are exactly ``lewm_predicted_workspace_safe``,
    ``phase_prediction_safe``, ``lewm_cartesian_goal_cost``,
    ``lewm_pixel_goal_cost`` and ``model_command_cost``.  Under
    ``mpc_only_collection`` none of them may reach any returned field.

    ``lewm_ranking`` is ranking-only.  ``None`` keeps the historical default
    (``not mpc_only_collection``).  ``False`` switches costs to the
    command-side ranking without dropping LeWM eligibility gates.

    ``closure_cost`` is command-side: it is the grasp objective measured on the
    exact forward kinematics of the command, not on a rollout.  Measured live,
    the frozen dynamics head moves this quantity by 4.8% of its true value over
    a 26.8 deg gripper sweep, so scoring closure on the rollout would rank a
    jaw slam above a partial close that keeps the pickup in the corridor.

    ``first_chunk_target_progress_m``, ``terminal_target_progress_m``,
    ``command_first_chunk_cartesian_goal_cost`` and
    ``model_command_max_xyz_gap_m`` are command-side.  Finite first-chunk
    progress below ``IK_FIRST_CHUNK_PROGRESS_MIN_M`` is ineligible (the
    already-computed 1 mm flag).  When ``mpc_only_collection`` is false the
    cartesian term is rewritten so an unexecuted tail cannot beat a servo.
    """

    command_cartesian = np.asarray(command_cartesian_goal_cost)
    count = command_cartesian.shape[0]
    if len(chunk_reasons) != count:
        raise ValueError("chunk_reasons must carry one entry per candidate")
    base_safe = (
        np.asarray(hard_command_safe, dtype=bool)
        & np.asarray(candidate_ik_safe, dtype=bool)
        & np.asarray(phase_safe, dtype=bool)
        & np.asarray(phase_distance_safe, dtype=bool)
        & np.asarray(command_workspace_safe, dtype=bool)
        & np.asarray(tool_travel_safe, dtype=bool)
        & np.asarray(chunk_safe, dtype=bool)
    )
    if first_chunk_target_progress_m is not None:
        first_progress = np.asarray(first_chunk_target_progress_m, dtype=np.float64)
        if first_progress.shape != (count,):
            raise ValueError("first_chunk_target_progress_m must match candidates")
        unqualified = np.isfinite(first_progress) & (
            first_progress < IK_FIRST_CHUNK_PROGRESS_MIN_M
        )
        base_safe = base_safe & ~unqualified
    if lewm_ranking is None:
        lewm_ranking = not mpc_only_collection
    if not mpc_only_collection:
        base_safe = (
            base_safe
            & np.asarray(phase_prediction_safe, dtype=bool)
            & np.asarray(lewm_predicted_workspace_safe, dtype=bool)
        )
    eligible = experiment_candidate_selection_mask(
        base_safe,
        np.asarray(peg_safe, dtype=bool),
        experiment_method=experiment_method,
    )
    if not lewm_ranking:
        cartesian_goal_cost = command_cartesian
        pixel_goal_cost = np.asarray(command_pixel_goal_cost)
        lewm_cost_contribution: np.ndarray | float = 0.0
    else:
        cartesian_goal_cost = np.asarray(lewm_cartesian_goal_cost)
        pixel_goal_cost = np.asarray(lewm_pixel_goal_cost)
        lewm_cost_contribution = 0.05 * np.asarray(model_command_cost)
    closure_contribution: np.ndarray | float = 0.0
    if closure_cost is not None and float(closure_cost_weight) != 0.0:
        closure = np.asarray(closure_cost, dtype=np.float64)
        if closure.shape != (count,):
            raise ValueError("closure_cost must match candidates")
        closure_contribution = float(closure_cost_weight) * closure
    if not mpc_only_collection and (
        first_chunk_target_progress_m is not None
        or terminal_target_progress_m is not None
        or command_first_chunk_cartesian_goal_cost is not None
        or model_command_max_xyz_gap_m is not None
    ):
        cartesian_goal_cost = first_chunk_failclosed_cartesian_goal_cost(
            cartesian_goal_cost,
            first_chunk_target_progress_m=first_chunk_target_progress_m,
            terminal_target_progress_m=terminal_target_progress_m,
            command_first_chunk_cartesian_goal_cost=(
                command_first_chunk_cartesian_goal_cost
            ),
            model_command_max_xyz_gap_m=model_command_max_xyz_gap_m,
        )
    costs = (
        cartesian_goal_cost
        + pixel_goal_weight * pixel_goal_cost
        + retrieval_cost_weight * np.asarray(retrieval_cost)
        + lewm_cost_contribution
        + closure_contribution
    )
    decision_reasons: list[tuple[str, ...]] = []
    for index in range(count):
        reasons = list(chunk_reasons[index])
        if not hard_command_safe[index]:
            reasons.append("raw_prefix_safety")
        if not candidate_ik_safe[index]:
            reasons.append("ik_not_converged")
        if not phase_safe[index]:
            reasons.append("gripper_phase")
        if not phase_distance_safe[index]:
            reasons.append("object_phase_distance")
        if not command_workspace_safe[index]:
            reasons.append("command_workspace")
        if not tool_travel_safe[index]:
            reasons.append("tool_travel_exceeded")
        if first_chunk_target_progress_m is not None:
            first_value = float(
                np.asarray(first_chunk_target_progress_m, dtype=np.float64)[index]
            )
            if np.isfinite(first_value) and first_value < IK_FIRST_CHUNK_PROGRESS_MIN_M:
                reasons.append("ik_receding_horizon_unqualified")
        if not mpc_only_collection:
            if not phase_prediction_safe[index]:
                reasons.append("lewm_prediction_bounds")
            if not lewm_predicted_workspace_safe[index]:
                reasons.append("lewm_prediction_workspace")
        if not peg_safe[index] and experiment_method == "full":
            reasons.append(peg_reason)
        decision_reasons.append(tuple(sorted(set(reasons))))
    return CandidateDecision(
        base_safe=base_safe,
        eligible=eligible,
        costs=costs,
        decision_reasons=tuple(decision_reasons),
        supervisor_precheck_safe=base_safe & np.asarray(peg_safe),
        cartesian_goal_cost=cartesian_goal_cost,
        pixel_goal_cost=pixel_goal_cost,
    )


LEWM_ONLY_REJECTION_REASONS = frozenset(
    {"lewm_prediction_bounds", "lewm_prediction_workspace"}
)
"""The reasons a candidate is refused that only exist when the world model runs.

Every other reason is command-side and would have refused the candidate anyway,
so lifting exactly these recovers the field the planner would have chosen from
without the world model.
"""


def executed_prefix_ranking_cost(
    *,
    prefix_cartesian_goal_cost: np.ndarray,
    command_pixel_goal_cost: np.ndarray,
    retrieval_cost: np.ndarray,
    pixel_goal_weight: float,
    retrieval_cost_weight: float,
    closure_cost: np.ndarray | None = None,
    closure_cost_weight: float = 0.0,
) -> np.ndarray:
    """Rank the same candidates on the part of them that actually gets sent.

    Purely a record; nothing here reaches the robot. The acted-on cartesian
    term is terminal over the whole horizon, but only the first
    ``command_chunk`` steps are ever transmitted before the next round replans.
    Proposers that fill the horizon are therefore scored on where they would
    have arrived, while a proposer that offers a short chunk and holds is
    scored on where it stops -- measured live, the learned actor's chunks are
    five steps of a horizon of eight and it commands about a third of the
    per-round joint travel of the analytic IK candidates.

    This says nothing about whether the difference is unfair, and it is not a
    proposal to rank this way. It only lets the question be settled afterwards
    from the round artifact instead of argued.

    The pixel term is left over the full commanded horizon: the cartesian term
    carries three quarters of the measured gap between proposers, so truncating
    it alone isolates the effect without inventing a second quantity that
    production never computes.
    """
    costs = (
        np.asarray(prefix_cartesian_goal_cost, dtype=np.float64)
        + float(pixel_goal_weight)
        * np.asarray(command_pixel_goal_cost, dtype=np.float64)
        + float(retrieval_cost_weight) * np.asarray(retrieval_cost, dtype=np.float64)
    )
    if closure_cost is not None and float(closure_cost_weight) != 0.0:
        costs = costs + float(closure_cost_weight) * np.asarray(
            closure_cost, dtype=np.float64
        )
    return costs


def mpc_only_ranking_cost(
    *,
    command_cartesian_goal_cost: np.ndarray,
    command_pixel_goal_cost: np.ndarray,
    retrieval_cost: np.ndarray,
    pixel_goal_weight: float,
    retrieval_cost_weight: float,
    closure_cost: np.ndarray | None = None,
    closure_cost_weight: float = 0.0,
) -> np.ndarray:
    """Rank the same candidates as if the world model had not been consulted.

    Purely a record. Nothing here reaches the robot: the planner ranks on
    :func:`compose_candidate_decision`, and this only rides along in the round
    artifact so that afterwards one can ask whether the world model changed the
    choice, without needing a second run of the arm to compare against.

    It uses the commanded trajectory where the acted-on cost uses the predicted
    one, and drops the model-command agreement term, which is what the planner
    itself does under ``mpc_only_collection``. The closure objective is measured
    on the commanded jaws either way, so it carries over unchanged.
    """
    costs = (
        np.asarray(command_cartesian_goal_cost, dtype=np.float64)
        + float(pixel_goal_weight)
        * np.asarray(command_pixel_goal_cost, dtype=np.float64)
        + float(retrieval_cost_weight) * np.asarray(retrieval_cost, dtype=np.float64)
    )
    if closure_cost is not None and float(closure_cost_weight) != 0.0:
        costs = costs + float(closure_cost_weight) * np.asarray(
            closure_cost, dtype=np.float64
        )
    return costs


def counterfactual_selection_without_lewm(
    *,
    mpc_only_costs: np.ndarray,
    decision_reasons: Sequence[Sequence[str]],
) -> int | None:
    """Which candidate the planner would have executed with the model switched off.

    A candidate is back in the running when the only thing said against it came
    from the world model. Returns ``None`` when that leaves nothing to choose
    from, which is the honest answer for a round where every candidate was
    refused on command-side grounds.
    """
    costs = np.asarray(mpc_only_costs, dtype=np.float64)
    eligible = [
        index
        for index, reasons in enumerate(decision_reasons)
        if set(reasons) <= LEWM_ONLY_REJECTION_REASONS and np.isfinite(costs[index])
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda index: float(costs[index]))


V11_PRED_SOURCE = "v11_dual_view_policy"
CALIBRATED_IK_SOURCE = "calibrated_cartesian_ik"
COLUMN_ARC_IK_COST_PENALTY = 0.05
COLUMN_PRED_SANITY_MARGIN_PX = 6.0


def is_column_arc_ik(metadata: Mapping[str, Any]) -> bool:
    """Scripted ``arc_*`` IK midpoints (+20–25 mm) that look drunk on column."""

    source = str(metadata.get("source") or "")
    variant = str(metadata.get("variant") or "")
    return source == CALIBRATED_IK_SOURCE and variant.startswith("arc_")


def _column_rank_points(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Terminal C and start slice for column ranking.

    Accepts C-only ``[K,H,2]`` or three-anchor ``[K,H,3,2]``. Terminal
    score is always the C tip (or the only point) versus the current
    column target. Start score keeps L/R when present so a tilted metal
    leftover can mark commanded IK insane.
    """

    values = np.asarray(paths, dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 2:
        return values[:, -1], values[:, 0]
    if values.ndim == 4 and values.shape[-2:] == (3, 2):
        return values[:, -1, 0], values[:, 0]
    raise ValueError("column paths must have shape [K,H,2] or [K,H,3,2]")


def prefer_sane_column_pred_costs(
    costs: np.ndarray,
    *,
    safe: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    predicted_column_px: np.ndarray,
    commanded_column_px: np.ndarray,
    target_px: np.ndarray | None,
    current_px: np.ndarray | None = None,
) -> np.ndarray:
    """Prefer PRED on the column view when it is geometrically saner than IK.

    Ranking cartesian cost can pick a scripted ``arc_*`` or a servo whose
    commanded FK projects as a nonsense curve, while the LeWM/V11 PRED path
    stays on the jaws and the current column target. A safe
    ``v11_dual_view_policy`` candidate that finishes closer to that target
    keeps rank; ``arc_*`` is always downweighted against it. A commanded
    path that starts far from the live jaws (raw FK in the uncorrected
    column frame) is treated as insane even if its terminal pixel error is
    slightly better. Does not touch eligibility or LeWM-independence
    inputs — only the acted-on cost vector after ``compose_candidate_decision``.
    """

    adjusted = np.asarray(costs, dtype=np.float64).copy()
    safe_mask = np.asarray(safe, dtype=bool)
    if adjusted.shape != safe_mask.shape:
        raise ValueError("costs and safe must match")
    if len(metadata) != int(adjusted.shape[0]):
        raise ValueError("metadata must match candidates")
    pred_indices = [
        index
        for index, row in enumerate(metadata)
        if str(row.get("source") or "") == V11_PRED_SOURCE and bool(safe_mask[index])
    ]
    has_pred = bool(pred_indices)
    for index, row in enumerate(metadata):
        if is_column_arc_ik(row) and has_pred:
            adjusted[index] += COLUMN_ARC_IK_COST_PENALTY
    if not has_pred or target_px is None:
        return adjusted
    target = np.asarray(target_px, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(target)):
        return adjusted
    predicted = np.asarray(predicted_column_px, dtype=np.float64)
    commanded = np.asarray(commanded_column_px, dtype=np.float64)
    if predicted.shape[0] != adjusted.shape[0] or commanded.shape[0] != adjusted.shape[0]:
        raise ValueError("column paths must match candidates")
    pred_term, pred_start_pts = _column_rank_points(predicted)
    cmd_term, cmd_start_pts = _column_rank_points(commanded)
    pred_err = np.linalg.norm(pred_term - target[None], axis=-1)
    cmd_err = np.linalg.norm(cmd_term - target[None], axis=-1)
    pred_start_err = None
    cmd_start_err = None
    if current_px is not None:
        current = np.asarray(current_px, dtype=np.float64)
        if current.shape == (3, 2) and pred_start_pts.ndim == 3:
            if np.all(np.isfinite(current)):
                pred_start_err = np.linalg.norm(
                    pred_start_pts - current[None], axis=-1
                ).max(axis=-1)
                cmd_start_err = np.linalg.norm(
                    cmd_start_pts - current[None], axis=-1
                ).max(axis=-1)
        else:
            current = current.reshape(2)
            if np.all(np.isfinite(current)):
                pred_start_err = np.linalg.norm(
                    pred_start_pts.reshape(pred_start_pts.shape[0], -1, 2)[:, 0]
                    - current[None],
                    axis=-1,
                )
                cmd_start_err = np.linalg.norm(
                    cmd_start_pts.reshape(cmd_start_pts.shape[0], -1, 2)[:, 0]
                    - current[None],
                    axis=-1,
                )
    best_pred = min(pred_indices, key=lambda index: (float(adjusted[index]), float(pred_err[index])))
    pred_geo = float(pred_err[best_pred])
    pred_start = (
        None if pred_start_err is None else float(pred_start_err[best_pred])
    )
    for index, row in enumerate(metadata):
        if index == best_pred or not bool(safe_mask[index]):
            continue
        source = str(row.get("source") or "")
        if source not in {CALIBRATED_IK_SOURCE, "fcpg_goal_servo"}:
            continue
        cmd_geo = float(cmd_err[index])
        # Sibling: PRED wins when it finishes closer to the current column
        # target. Keep that. Do not also flatten a commanded path whose
        # terminal is closer — start-offset alone is how a receding actor
        # beat an approaching servo at 67 mm.
        if cmd_geo + COLUMN_PRED_SANITY_MARGIN_PX < pred_geo:
            continue
        terminal_insane = pred_geo + COLUMN_PRED_SANITY_MARGIN_PX < cmd_geo
        start_insane = (
            pred_start is not None
            and cmd_start_err is not None
            and pred_start + COLUMN_PRED_SANITY_MARGIN_PX
            < float(cmd_start_err[index])
        )
        if not terminal_insane and not start_insane:
            continue
        gap = float(adjusted[best_pred] - adjusted[index])
        if gap >= 0.0:
            adjusted[index] = float(adjusted[best_pred]) + 1.0e-3
            if is_column_arc_ik(row):
                adjusted[index] += COLUMN_ARC_IK_COST_PENALTY
    return adjusted


def column_live_pixel_to_base_xyz(
    fit: ProjectionFit,
    target_live_px: np.ndarray,
    *,
    z_base: float,
    live_transform_px: np.ndarray | None = None,
) -> np.ndarray:
    """Backproject one live column pixel after undoing live image calibration."""

    live = np.asarray(target_live_px, dtype=np.float32)
    if live.shape != (2,) or not np.all(np.isfinite(live)):
        raise ValueError("target_live_px must be one finite pixel")
    calibration_px = (
        live
        if live_transform_px is None
        else invert_live_similarity_transform(live, live_transform_px)
    )
    return np.asarray(
        approximate_column_backprojection_on_plane(
            fit, calibration_px, z_base=float(z_base)
        ),
        dtype=np.float32,
    )


def wrist_tip_similarity_correction(
    wrist_anchor_px: np.ndarray,
    fk_anchors_xyz: np.ndarray,
    relative_centroid_px: np.ndarray,
    *,
    gain: float,
    deadband_px: float,
    max_correction_m: float,
    diagnostics: dict[str, Any] | None = None,
) -> np.ndarray | None:
    """Map a wrist-view object offset into a bounded base-frame XY correction.

    The wrist camera rides on the gripper, so the pixel vector between the
    FCPG-corrected fingertip anchors and the FK vector between the physical
    tips give a per-round similarity (scale + rotation) from wrist pixels to
    base XY — self-calibrating for camera height and wrist roll. Validated
    offline on the 2026-07-26 session: holdout on the missed close recovered
    the true miss vector with ~4 deg direction error; the calibrated wrist
    projection fit's Jacobian is degenerate off the anchor line and must not
    be used instead.

    ``diagnostics`` is written to and never read: it exposes why a round did
    or did not correct, and how much of the correction the cap removed, so the
    stage can be observed on the live robot without changing what it does.
    """

    if diagnostics is not None:
        diagnostics.update(
            {
                "reason": None,
                "offset_norm_px": None,
                "scale_m_per_px": None,
                "raw_magnitude_m": None,
                "applied_magnitude_m": None,
                "capped": False,
            }
        )

    def _refuse(reason: str) -> None:
        if diagnostics is not None:
            diagnostics["reason"] = reason
        return None

    tips_px = np.asarray(wrist_anchor_px, dtype=np.float64)
    tips_xyz = np.asarray(fk_anchors_xyz, dtype=np.float64)
    v_px = tips_px[2] - tips_px[1]
    u_base = tips_xyz[2, :2] - tips_xyz[1, :2]
    norm_px = float(np.linalg.norm(v_px))
    norm_base = float(np.linalg.norm(u_base))
    if norm_px < 8.0 or norm_base < 1.0e-4:
        # Closed or near-closed jaws give a degenerate similarity.
        return _refuse("degenerate_jaw_span")
    offset = np.asarray(relative_centroid_px, dtype=np.float64)
    if not np.all(np.isfinite(offset)) or offset.shape != (2,):
        return _refuse("offset_not_finite")
    offset_norm_px = float(np.linalg.norm(offset))
    if diagnostics is not None:
        diagnostics["offset_norm_px"] = offset_norm_px
    if offset_norm_px <= deadband_px:
        return _refuse("under_deadband")
    scale = norm_base / norm_px
    cos_t = float(np.dot(v_px, u_base)) / (norm_px * norm_base)
    sin_t = float(v_px[0] * u_base[1] - v_px[1] * u_base[0]) / (
        norm_px * norm_base
    )
    rotation = np.asarray([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    delta = gain * scale * (rotation @ offset)
    magnitude = float(np.linalg.norm(delta))
    capped = magnitude > max_correction_m
    if capped:
        delta *= max_correction_m / magnitude
    if diagnostics is not None:
        diagnostics.update(
            {
                "scale_m_per_px": scale,
                "raw_magnitude_m": magnitude,
                "applied_magnitude_m": float(np.linalg.norm(delta)),
                "capped": bool(capped),
            }
        )
    return delta.astype(np.float32)


def live_anchor_transform_applied(
    distance_m: float,
    radius_m: float,
    *,
    previously_applied: bool,
    hysteresis_band: float,
) -> bool:
    """Whether the live-anchor transform stays on for this support distance.

    Enter when ``distance <= radius``; leave only when
    ``distance > radius * (1 + band)``.  The band does not change the intent to
    refuse extrapolation of a two-point fit outside the measured neighbourhood;
    it only stops the closed loop from riding the hard threshold.
    """

    if (
        not np.isfinite(distance_m)
        or not np.isfinite(radius_m)
        or radius_m <= 0.0
        or not np.isfinite(hysteresis_band)
        or hysteresis_band < 0.0
    ):
        raise ValueError("live-anchor transform gate distances must be finite")
    if not previously_applied:
        return float(distance_m) <= float(radius_m)
    return float(distance_m) <= float(radius_m) * (1.0 + float(hysteresis_band))


def descent_uses_standoff(
    planar_error_m: float,
    descent_radius_m: float,
    *,
    previously_standoff: bool | None = None,
    hysteresis_band: float = 0.0,
) -> bool:
    """Whether this round commands the standoff height instead of contact.

    First look uses the hard radius. After that, stay in standoff until the
    error is inside ``radius * (1 - band)`` and stay in contact until it is
    outside ``radius * (1 + band)``. A zero band restores the hard step.
    """

    if (
        not np.isfinite(planar_error_m)
        or planar_error_m < 0.0
        or not np.isfinite(descent_radius_m)
        or descent_radius_m <= 0.0
        or not np.isfinite(hysteresis_band)
        or hysteresis_band < 0.0
    ):
        raise ValueError("descent-radius gate distances must be finite")
    if previously_standoff is None or hysteresis_band == 0.0:
        return float(planar_error_m) > float(descent_radius_m)
    if previously_standoff:
        return float(planar_error_m) > float(descent_radius_m) * (
            1.0 - float(hysteresis_band)
        )
    return float(planar_error_m) > float(descent_radius_m) * (
        1.0 + float(hysteresis_band)
    )


def bounded_column_servo_target_xyz(
    fit: ProjectionFit,
    current_live_px: np.ndarray,
    target_live_px: np.ndarray,
    current_base_xyz: np.ndarray,
    *,
    target_z_base: float,
    max_planar_step_m: float,
    descent_radius_m: float,
    standoff_z_base: float | None,
    live_transform_px: np.ndarray | None = None,
    max_z_step_m: float | None = None,
    held_relative_target_xyz: np.ndarray | None = None,
    servo_gain: float = 1.0,
    previously_standoff: bool | None = None,
    hysteresis_band: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert a column-image error into one bounded Cartesian servo target.

    The absolute affine backprojection is poorly conditioned far from the
    locally calibrated gripper. Subtracting backprojections of the target and
    current gripper cancels the affine offset; bounding that relative motion
    keeps each closed-loop IK request inside the locally observed region.

    That subtraction also puts the gripper anchor's own error on the goal with
    unity gain, and the anchor error moves with arm state, so re-deriving the
    goal every round closes a loop that oscillates instead of converging.
    Measured 2026-08-15 with the pickup pixel frozen at (88.9, 165.3) for 20
    consecutive rounds: the goal alternated by 21mm and the arm tracked it,
    shoulder_pan chattering 5.1deg/8.8deg every round. Passing
    ``held_relative_target_xyz`` re-uses an already committed absolute goal and
    only re-bounds the step toward it, which is what breaks that loop; the
    caller owns the decision of when a fresh derivation is warranted.

    ``servo_gain`` scales only the planning step (before the planar clamp).
    ``relative_target`` keeps the full planar delta so distance metrics and
    verdicts stay honest.  Measured effective loop gain was ~2; a gain below 1
    is what makes the commanded step smaller than the observed error.

    ``previously_standoff`` / ``hysteresis_band`` stop the 50 mm descent-radius
    Z flip from toggling every round when planar error chatters around the
    threshold.  A zero band keeps the historical hard step.
    """

    current_xyz = np.asarray(current_base_xyz, dtype=np.float32)
    if current_xyz.shape != (3,) or not np.all(np.isfinite(current_xyz)):
        raise ValueError("current_base_xyz must be one finite XYZ vector")
    if (
        not np.isfinite(target_z_base)
        or not np.isfinite(max_planar_step_m)
        or max_planar_step_m <= 0.0
        or not np.isfinite(descent_radius_m)
        or descent_radius_m <= 0.0
    ):
        raise ValueError("column servo distances must be finite and positive")
    if standoff_z_base is not None and not np.isfinite(standoff_z_base):
        raise ValueError("standoff_z_base must be finite when provided")
    if (
        not np.isfinite(servo_gain)
        or servo_gain <= 0.0
        or servo_gain > 1.0
    ):
        raise ValueError("servo_gain must be in (0, 1]")

    if held_relative_target_xyz is None:
        current_on_plane = column_live_pixel_to_base_xyz(
            fit,
            current_live_px,
            z_base=float(target_z_base),
            live_transform_px=live_transform_px,
        )
        target_on_plane = column_live_pixel_to_base_xyz(
            fit,
            target_live_px,
            z_base=float(target_z_base),
            live_transform_px=live_transform_px,
        )
        planar_delta = np.asarray(
            target_on_plane[:2] - current_on_plane[:2], dtype=np.float32
        )
    else:
        held = np.asarray(held_relative_target_xyz, dtype=np.float32)
        if held.shape != (3,) or not np.all(np.isfinite(held)):
            raise ValueError(
                "held_relative_target_xyz must be one finite XYZ vector"
            )
        planar_delta = np.asarray(held[:2] - current_xyz[:2], dtype=np.float32)
    planar_error_m = float(np.linalg.norm(planar_delta))
    # relative_target stays at full delta; only the commanded step is gained.
    gained_delta = np.asarray(planar_delta * float(servo_gain), dtype=np.float32)
    gained_error_m = float(np.linalg.norm(gained_delta))
    step_delta = gained_delta.copy()
    if gained_error_m > max_planar_step_m:
        step_delta *= float(max_planar_step_m / gained_error_m)

    relative_target = current_xyz.copy()
    relative_target[:2] += planar_delta
    relative_target[2] = float(target_z_base)
    planning_target = current_xyz.copy()
    planning_target[:2] += step_delta
    use_standoff = standoff_z_base is not None and descent_uses_standoff(
        planar_error_m,
        descent_radius_m,
        previously_standoff=previously_standoff,
        hysteresis_band=hysteresis_band,
    )
    if use_standoff:
        planning_target[2] = max(
            float(current_xyz[2]), float(standoff_z_base)
        )
    else:
        planning_target[2] = float(target_z_base)
    if max_z_step_m is not None:
        if not np.isfinite(max_z_step_m) or max_z_step_m <= 0.0:
            raise ValueError("max_z_step_m must be finite and positive")
        z_delta = float(
            np.clip(
                planning_target[2] - current_xyz[2],
                -max_z_step_m,
                max_z_step_m,
            )
        )
        planning_target[2] = float(current_xyz[2]) + z_delta
    return planning_target, relative_target, planar_error_m


def cartesian_terminal_goal_cost(
    predicted_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    final_steps: int,
    scale_m: float,
) -> np.ndarray:
    """Score the LeWM-predicted terminal tool position in base-frame metres."""

    predicted = np.asarray(predicted_xyz, dtype=np.float32)
    target = np.asarray(target_xyz, dtype=np.float32)
    if predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError("predicted_xyz must have shape [candidates,horizon,3]")
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_xyz must be one finite XYZ vector")
    if final_steps <= 0 or final_steps > predicted.shape[1]:
        raise ValueError("final_steps is outside the prediction horizon")
    if not np.isfinite(scale_m) or scale_m <= 0.0:
        raise ValueError("scale_m must be finite and positive")
    error = (predicted[:, -final_steps:] - target[None, None]) / float(scale_m)
    return np.mean(np.sum(np.square(error), axis=-1), axis=-1).astype(np.float32)


@dataclass
class _AnchorPosterior:
    state_latest: np.ndarray
    frame_latest: np.ndarray
    nominal_latest_px: np.ndarray
    seed_history_px: np.ndarray
    corrected_latest_px: np.ndarray


@dataclass
class GroundingObservation:
    anchors_xyz: np.ndarray
    nominal: dict[str, np.ndarray]
    correction_seed: dict[str, np.ndarray]
    corrected: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]
    contexts: dict[str, np.ndarray]
    #: Raw projected-CLS context, i.e. ``base.projector(encoder(pixels)[:, 0])``
    #: with no FCPG correction added.  The dynamics head is trained and
    #: evaluated offline on exactly this representation, so the runtime must
    #: feed it this one rather than ``contexts`` (which carries the additive
    #: FCPG bridge residual the head never saw during training).
    raw_contexts: dict[str, np.ndarray]
    diagnostics: dict[str, dict[str, Any]]
    blocked_reasons: tuple[str, ...]
    #: Anchor tokens produced from the same ViT forward as ``contexts``.
    #: ``None`` means this observation predates the grounded-rank contract.
    anchor_tokens: dict[str, np.ndarray] | None = None
    #: Current endpoint's native 16x16 ViT patch grid for both views. V16.5
    #: Actor inference consumes this exact shared forward; it is never encoded
    #: a second time and never becomes an anchor authority by itself.
    full256_tokens: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class LiveFingertipMeasurement:
    """Current fixed/moving fingertip pixels supplied by Prod-V13.

    The two slots are physical observations, ordered as fixed then moving.
    They replace V12's legacy pixel anchors before any anchor-conditioned
    computation. The frozen V12 ViT/global context and LeWM carrier remain,
    while the legacy pixel corrector is not executed.
    """

    tip_px: Mapping[str, np.ndarray]
    visible: Mapping[str, np.ndarray]
    source: str
    generation: int = 0
    horizon: int = 0

    def __post_init__(self) -> None:
        if set(self.tip_px) != set(VIEWS) or set(self.visible) != set(VIEWS):
            raise ValueError("live fingertip measurement requires wrist and column")
        points: dict[str, np.ndarray] = {}
        visibility: dict[str, np.ndarray] = {}
        for view in VIEWS:
            view_points = np.asarray(self.tip_px[view], dtype=np.float32)
            view_visible = np.asarray(self.visible[view], dtype=bool)
            if (
                view_points.shape != (2, 2)
                or not np.all(np.isfinite(view_points))
                or np.any(view_points < 0.0)
                or np.any(view_points > 223.0)
            ):
                raise ValueError(f"{view} live fingertip pixels must be finite [2,2]")
            if view_visible.shape != (2,):
                raise ValueError(f"{view} live fingertip visibility must be [2]")
            points[view] = view_points.copy()
            visibility[view] = view_visible.copy()
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("live fingertip measurement source must be non-empty")
        if self.generation < 0 or self.horizon < 0:
            raise ValueError("live fingertip generation/horizon must be non-negative")
        object.__setattr__(self, "tip_px", points)
        object.__setattr__(self, "visible", visibility)


def live_fingertip_control_anchors(
    measurement: LiveFingertipMeasurement,
    geometry: PlanningGeometry,
) -> dict[str, np.ndarray]:
    """Build ``[midpoint, fixed, moving]`` from the current measurement.

    This is the only pixel triple admitted after automatic H0. V12's old
    pixel prediction is not a fallback once Prod-V13 supplies this value.
    """

    result: dict[str, np.ndarray] = {}
    for view in VIEWS:
        tips = np.asarray(measurement.tip_px[view], dtype=np.float32)
        seen = np.asarray(measurement.visible[view], dtype=bool)
        anchors = np.concatenate((tips.mean(axis=0, keepdims=True), tips), axis=0)
        anchor_valid = np.asarray(
            [bool(np.all(seen)), bool(seen[0]), bool(seen[1])], dtype=bool
        ) & geometry.content_mask(anchors)
        if not bool(np.all(anchor_valid)):
            raise ValueError(
                f"{view} Prod-V13 fingertip measurement is not fully visible"
            )
        result[view] = anchors
    return result


def recursive_anchor_seed(
    nominal_history_px: np.ndarray,
    current_state: np.ndarray,
    current_frame: np.ndarray,
    posterior: _AnchorPosterior | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Propagate one calibrated anchor posterior with measured FK displacement."""

    nominal = np.asarray(nominal_history_px, dtype=np.float32)
    state = np.asarray(current_state, dtype=np.float32)
    frame = np.asarray(current_frame, dtype=np.uint8)
    if nominal.ndim != 3 or nominal.shape[-2:] != (3, 2):
        raise ValueError("nominal_history_px must have shape [T,3,2]")
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("current_state must be a finite six-vector")
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("current_frame must be an RGB image")
    if not np.all(np.isfinite(nominal)):
        raise ValueError("nominal_history_px must be finite")

    if posterior is None:
        seed = nominal.copy()
        source = "calibrated_nominal"
        fk_delta = np.zeros((3, 2), dtype=np.float32)
        posterior_offset = np.zeros((3, 2), dtype=np.float32)
    elif np.array_equal(state, posterior.state_latest) and np.array_equal(
        frame, posterior.frame_latest
    ):
        seed = posterior.seed_history_px.copy()
        source = "same_observation_replay"
        fk_delta = nominal[-1] - posterior.nominal_latest_px
        posterior_offset = seed[-1] - nominal[-1]
    else:
        fk_delta = nominal[-1] - posterior.nominal_latest_px
        posterior_offset = (
            posterior.corrected_latest_px - posterior.nominal_latest_px
        )
        seed = nominal + posterior_offset[None]
        source = "posterior_plus_fk_delta"

    return seed.astype(np.float32), {
        "prior_source": source,
        "fk_delta_px": fk_delta.astype(np.float32),
        "posterior_offset_px": posterior_offset.astype(np.float32),
    }


#: Gripper reading that is unambiguously open. Live empty-close is ~1.8°,
#: the fish stalls near 7°, and approach open is ~31°. Used only to decide
#: whether a tip span may be recorded as that view's open-jaw observation.
_OPEN_JAW_GRIPPER_DEG = 20.0


def native_progress_held(verdict: Mapping[str, Any] | None) -> bool:
    """True only for a same-view ``held`` that used closing progress.

    ``held`` from collapsed tips without open/empty spans is not native
    progress: dummy or uncalibrated anchors can collapse and sit on a
    centroid by construction. Live promotion needs ``target_between`` and
    ``closing_progress >= 0.5`` from that view's own spans.
    """

    if not isinstance(verdict, Mapping):
        return False
    progress = verdict.get("closing_progress")
    if progress is None or not np.isfinite(progress):
        return False
    return (
        verdict.get("verdict") == "held"
        and bool(verdict.get("target_between_tips"))
        and float(progress) >= 0.5
    )


def native_spans_usable(
    open_span: float | None, empty_span: float | None
) -> bool:
    """True when this view can score closing progress from its own spans."""

    if open_span is None or empty_span is None:
        return False
    if not (np.isfinite(open_span) and np.isfinite(empty_span)):
        return False
    return (float(open_span) - float(empty_span)) > 1.0


def grasp_should_close_jaws(
    three_point: Mapping[str, Any] | None,
    closure: Mapping[str, Any] | None = None,
    *,
    gripper_to_pickup_px: float | None = None,
    gripper_to_pickup_m: float | None = None,
    pickup_near_px: float = 28.0,
    pickup_near_m: float = 0.018,
) -> bool:
    """GRASP may close only on a hold or a metrically near pickup.

    Pixel corridor / between-tips / closure_reachable are projective and
    do not authorize a close by themselves.
    """

    del closure
    verdict = None
    if isinstance(three_point, Mapping):
        verdict = three_point.get("verdict")
    return close_allowed_for_pickup(
        three_point_verdict=None if verdict is None else str(verdict),
        gripper_to_pickup_px=gripper_to_pickup_px,
        gripper_to_pickup_m=gripper_to_pickup_m,
        pickup_near_px=pickup_near_px,
        pickup_near_m=pickup_near_m,
    )


def prefer_closing_jaw_costs(
    costs: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    *,
    hold_penalty: float = 1.0,
) -> np.ndarray:
    """Rank a closing jaw prefix above a frozen-open original."""

    ranked = np.asarray(costs, dtype=np.float64).copy()
    if ranked.shape != (len(metadata),):
        raise ValueError("closure costs must align with candidate metadata")
    for index, row in enumerate(metadata):
        rate = row.get("jaw_rate")
        if rate is None or not np.isfinite(float(rate)) or float(rate) >= 0.0:
            if np.isfinite(ranked[index]):
                ranked[index] += float(hold_penalty)
    return ranked


def measured_object_shape(descriptor: Mapping[str, Any] | None) -> dict[str, Any]:
    """The pickup's line and size, when the detector measured them.

    Absent for a detector that only reports a point, and every caller then
    falls back to treating the object as one.
    """

    if not isinstance(descriptor, Mapping):
        return {}
    axis = descriptor.get("axis_px")
    if axis is None:
        return {}
    return {
        "target_axis_px": np.asarray(axis, dtype=np.float64),
        "target_length_px": descriptor.get("length_px"),
        "target_width_px": descriptor.get("width_px"),
    }


class PreparedSO101RealtimePlanner:
    """Load model/data once and plan repeatedly from fresh real observations."""

    def __init__(self, config: PreparedPlannerConfig) -> None:
        self.config = config
        # Single source of truth for canvas-vs-content bounds.  Resolved once
        # here so no call site re-derives pad arithmetic; under the historical
        # `squash_v0` default the content rect IS the full canvas, so every
        # check below reduces to its historical expression exactly.
        self.planning_geometry = config.planning_geometry
        self.device = torch.device(config.device)
        bridge_path = config.bridge_checkpoint.expanduser().resolve()
        bridge_payload = torch.load(bridge_path, map_location="cpu", weights_only=False)
        source_checkpoint_path = Path(
            bridge_payload["args"]["checkpoint"]
        ).resolve()
        (
            source_checkpoint,
            source_embedded_args,
            cache_paths,
            source_frameskip,
            source_history,
            source_checkpoint_sha,
        ) = checkpoint_contract(source_checkpoint_path, None)
        checkpoint_path = (
            config.model_checkpoint.expanduser().resolve()
            if config.model_checkpoint is not None
            else source_checkpoint_path
        )
        (
            checkpoint,
            embedded_args,
            _,
            frameskip,
            history,
            checkpoint_sha,
        ) = checkpoint_contract(checkpoint_path, None)
        if (source_frameskip, source_history) != (frameskip, history):
            raise ValueError(
                "bridge source and dynamics checkpoint sequence contracts differ"
            )
        if frameskip != 5 or history != 3:
            raise ValueError("real-time planner requires frameskip=5 and history=3")
        if config.command_chunk != frameskip:
            raise ValueError("command_chunk must match the checkpoint frameskip")
        grounding_model = build_checkpoint_model(
            source_checkpoint, source_frameskip, source_history, self.device
        )
        if not isinstance(grounding_model, StateResidualLeWM):
            raise TypeError("FCPG source checkpoint must be a state-residual LeWM")
        loaded_model = build_checkpoint_model(
            checkpoint, frameskip, history, self.device
        )
        self.memory_model: MultiScaleHistoryLeWM | None = None
        self.memory_offsets_raw: tuple[int, ...] = ()
        selected_protocol = checkpoint["run_config"]["contract"].get("protocol")
        if isinstance(loaded_model, MultiScaleHistoryLeWM):
            if selected_protocol not in {V6_PROTOCOL, V9_PROTOCOL}:
                raise ValueError("real-time planner supports V6 or V9 memory models")
            if selected_protocol == V9_PROTOCOL:
                assert_v9_backbone_matches(
                    checkpoint["model"], source_checkpoint["model"]
                )
                # V9's embedded backbone is exactly the task-local model used
                # by the FCPG bridge. Use that one object for grounding, bank
                # encoding, recent rollout, and memory-augmented rollout.
                grounding_model = loaded_model.backbone
            self.memory_model = loaded_model
            self.memory_offsets_raw = tuple(
                int(value)
                for value in checkpoint["run_config"]["contract"]["memory_offsets"]
            )
            model = loaded_model.backbone
        else:
            model = loaded_model
        # Frozen B2 deployment contract, asserted rather than assumed.
        #
        # Today the frozen B2 head reaches this point with `memory_model=None`
        # only because its protocol is absent from `MEMORY_PROTOCOLS`, so
        # `build_checkpoint_model` returns a non-multiscale model and the branch
        # above is never taken.  That is an accident of registry membership, not
        # a stated contract: adding the B2 protocol to `MEMORY_PROTOCOLS`, or
        # teaching the builder to wrap it, would silently activate the causal
        # memory path -- and that path composes its memory embedding from
        # element 1 of `encode_fcpg_view` (the corrected/bridged context) rather
        # than element 0, so the frozen B2 dynamics head would begin reading a
        # representation it was never trained on, mid-pilot, with no error.
        # Pinning it here makes that a startup failure instead of a quiet change
        # in what the published comparison measures.
        if selected_protocol == B2HEAD_RUNTIME_PROTOCOL and (
            self.memory_model is not None or self.memory_offsets_raw
        ):
            raise ValueError(
                f"dynamics protocol {B2HEAD_RUNTIME_PROTOCOL!r} is the frozen B2 "
                "shadow head and must run without causal memory, but this "
                "checkpoint resolved to memory_model="
                f"{type(self.memory_model).__name__} with offsets "
                f"{tuple(self.memory_offsets_raw)!r}. The frozen head takes the "
                "raw projected visual context; the memory path would feed it the "
                "corrected context instead. Refusing to launch rather than "
                "changing the frozen comparison silently"
            )
        if not isinstance(model, StateResidualLeWM):
            raise TypeError("real-time planner requires state-residual LeWM dynamics")
        self._corrector_variant = infer_so101_fcpg_corrector_variant(bridge_payload)
        self._coarse_fine_metadata = extract_coarse_fine_checkpoint_metadata(
            bridge_payload
        )
        self._coarse_fine_policy = assert_coarse_fine_planner_policy(
            variant=self._corrector_variant,
            metadata=bridge_payload,
            recursive_grounding=bool(config.recursive_grounding),
            global_tip_checkpoint=config.global_tip_checkpoint,
            request_live_execute=bool(config.request_live_execute),
            allow_uncalibrated_coarse_fine_shadow=bool(
                config.allow_uncalibrated_coarse_fine_shadow
            ),
        )
        self._fusion_state = GroundingFusionState()
        corrector = build_so101_fcpg_corrector(
            grounding_model,
            camera_context_dim=5,
            variant=self._corrector_variant,
        )
        self.system = SO101FCPGConditionedLeWM(
            grounding_model,
            corrector=corrector,
            camera_context_dim=5,
            freeze_model=True,
        ).to(self.device)
        self.system.corrector.load_state_dict(bridge_payload["corrector"], strict=True)
        self.system.bridge.load_state_dict(bridge_payload["bridge"], strict=True)
        self.system.eval()
        self._v165_runtime_adapter: V165RuntimeAdapter | None = None
        self._v168_runtime_adapter: V168LiveActorOverlay | None = None
        self._v170_runtime_adapter: V170LiveStack | None = None
        self._last_v170_shadow: dict[str, Any] | None = None
        self._last_v170_shadow_error: str | None = None
        if config.v168_mode is not None:
            assert config.v168_original_fcpg_checkpoint is not None
            assert config.v168_source_checkpoint is not None
            assert config.v168_lewm_checkpoint is not None
            assert config.v168_lewm_receipt is not None
            assert config.v168_training_receipt is not None
            assert config.v168_actor_checkpoint is not None
            foundation = V168FoundationRuntimeAdapter.from_registered_components(
                context_bridge_checkpoint=bridge_path,
                original_fcpg_checkpoint=config.v168_original_fcpg_checkpoint,
                v165_source_checkpoint=config.v168_source_checkpoint,
                lewm_checkpoint=config.v168_lewm_checkpoint,
                lewm_receipt=config.v168_lewm_receipt,
                grounding_model=grounding_model,
                runtime_action_lower_deg=SO101_POSITION_LOWER,
                runtime_action_upper_deg=SO101_POSITION_UPPER,
                runtime_max_step_deg=np.full(
                    6,
                    min(
                        float(config.max_first_command_jump),
                        float(config.max_prefix_step_jump),
                    ),
                    dtype=np.float32,
                ),
                original_max_residual_px=float(config.fcpg_max_correction_px),
                device=self.device,
            )
            self._v165_runtime_adapter = foundation
            self._v168_runtime_adapter = V168LiveActorOverlay.from_registered_pass8(
                receipt_path=config.v168_training_receipt,
                checkpoint_path=config.v168_actor_checkpoint,
                foundation_receipt=foundation.receipt,
                mode=config.v168_mode,
                device=self.device,
            )
        elif config.anchor_authority is not None:
            assert config.v165_original_fcpg_checkpoint is not None
            assert config.v165_source_checkpoint is not None
            assert config.v165_tuned_fcpg_checkpoint is not None
            assert config.v165_lewm_checkpoint is not None
            assert config.v165_actor_checkpoint is not None
            self._v165_runtime_adapter = V165RuntimeAdapter.from_checkpoints(
                anchor_authority=config.anchor_authority,
                context_bridge_checkpoint=bridge_path,
                original_fcpg_checkpoint=config.v165_original_fcpg_checkpoint,
                v165_source_checkpoint=config.v165_source_checkpoint,
                tuned_fcpg_checkpoint=config.v165_tuned_fcpg_checkpoint,
                lewm_checkpoint=config.v165_lewm_checkpoint,
                actor_checkpoint=config.v165_actor_checkpoint,
                grounding_model=grounding_model,
                runtime_action_lower_deg=SO101_POSITION_LOWER,
                runtime_action_upper_deg=SO101_POSITION_UPPER,
                runtime_max_step_deg=np.full(
                    6,
                    min(
                        float(config.max_first_command_jump),
                        float(config.max_prefix_step_jump),
                    ),
                    dtype=np.float32,
                ),
                original_max_residual_px=float(config.fcpg_max_correction_px),
                device=self.device,
                contact_authorization_receipt=config.v165_contact_authorization_receipt,
            )
        if config.v170_mode is not None:
            assert config.v170_fcpg_checkpoint is not None
            assert config.v170_lewm_checkpoint is not None
            assert config.v170_actor_checkpoint is not None
            self._v170_runtime_adapter = V170LiveStack.from_final_checkpoints(
                fcpg_checkpoint=config.v170_fcpg_checkpoint,
                lewm_checkpoint=config.v170_lewm_checkpoint,
                actor_checkpoint=config.v170_actor_checkpoint,
                mode=config.v170_mode,
                device=self.device,
            )
        self.global_tip_head: SO101GlobalTipHeatmapHead | None = None
        self.global_tip_contract: dict[str, Any] | None = None
        if config.global_tip_checkpoint is not None:
            self.global_tip_head, self.global_tip_contract = (
                load_global_tip_heatmap_head(
                    config.global_tip_checkpoint,
                    self.device,
                )
            )
            if self.global_tip_head.visual_dim != grounding_model.latent_dim:
                raise ValueError("global tip head visual dimension mismatch")
            if self.global_tip_head.camera_context_dim != 5:
                raise ValueError("global tip head camera context mismatch")
            if self.global_tip_head.num_anchors != 3:
                raise ValueError("global tip head must predict three anchors")
        self.grounding_model = grounding_model
        self.model = model
        self.v11_6_lewm = None
        self._pending_expectation = None
        self._pending_expectation_grounded = None
        self._pending_expectation_seen = None
        self._last_grounding = None
        self._executed_round = None
        self.last_world_model_residual: dict[str, Any] | None = None
        self.frameskip = frameskip
        self.history = history
        self.horizon = 8
        self._rank_prediction_steps = int(self.horizon)
        self.horizon_raw = frameskip * self.horizon
        if config.ik_arrival_raw_steps > self.horizon_raw:
            raise ValueError("ik_arrival_raw_steps exceeds the prediction horizon")

        caches = load_caches(cache_paths)
        if len(caches) != 1:
            raise ValueError("planner checkpoint must bind exactly one primary cache")
        self.primary: CacheView = caches[0]
        self.secondary: CacheView = load_caches(
            [config.secondary_cache.expanduser().resolve()]
        )[0]
        validate_paired_camera_data(self.primary.data, self.secondary.data)

        contract = checkpoint["run_config"]["contract"]
        self.state_center = np.asarray(
            contract["normalization_center"], dtype=np.float32
        )
        self.state_scale = np.asarray(
            contract["normalization_scale"], dtype=np.float32
        )
        (
            self.action_center,
            self.action_scale,
            self.action_normalization,
        ) = checkpoint_action_normalizer(checkpoint, embedded_args)
        self.kinematics = SO101MuJoCoKinematics(
            Path(__file__).resolve().parents[1]
            / "third_party/so_arm100/Simulation/SO101/so101_new_calib.xml"
        )
        # The projection fit maps anchors to PIXELS, so it is only valid in the
        # geometry it was fitted in.  The contract is driven by the declared
        # `geometry_protocol` rather than hard-wired: under the historical
        # `squash_v0` default this stays permissive, so every legacy, offline
        # and replay caller (including the frozen B0 Tier-2 replay, which loads
        # the unversioned rig calibration) keeps bit-identical behaviour.  The
        # moment a caller declares `letterbox_v1` the calibration must declare
        # it too, which is what stops a letterbox runtime from silently reading
        # a fit of unknown geometry.
        self.fits = load_projection_fits(
            config.calibration,
            expected_geometry=(
                LETTERBOX_V1 if config.geometry_protocol == LETTERBOX_V1 else None
            ),
        )

        assignments = episode_splits(
            [self.primary.data],
            parse_exclusions(source_embedded_args.get("exclude_episode", [])),
            split_seed=int(source_embedded_args["split_seed"]),
            val_fraction=float(source_embedded_args["val_fraction"]),
        )
        self.train_episodes = _training_episodes(self.primary, assignments)
        self.bank_rows = candidate_start_rows(
            self.primary.data.episodes,
            self.primary.data.frame_indices,
            self.train_episodes,
            horizon_raw=self.horizon_raw,
            stride=self.frameskip,
        )
        self.bank_states = np.asarray(
            self.primary.data.states[self.bank_rows], dtype=np.float32
        )
        self.bank_episodes = np.asarray(
            self.primary.data.episodes[self.bank_rows], dtype=np.int64
        )
        self.bank_frames = np.asarray(
            self.primary.data.frame_indices[self.bank_rows], dtype=np.int64
        )
        self._bank_index_by_episode_frame: dict[tuple[int, int], int] = {}
        for index, (episode, frame) in enumerate(
            zip(self.bank_episodes, self.bank_frames, strict=True)
        ):
            key = (int(episode), int(frame))
            if key in self._bank_index_by_episode_frame:
                raise ValueError(
                    "demonstration bank contains duplicate episode/frame candidates"
                )
            self._bank_index_by_episode_frame[key] = index
        self.bank_prefixes = raw_action_prefixes(
            self.primary.data.actions,
            self.bank_rows,
            horizon_raw=self.horizon_raw,
        )
        future_offsets = np.arange(
            self.frameskip,
            self.horizon_raw + 1,
            self.frameskip,
            dtype=np.int64,
        )
        self.bank_future_rows = self.bank_rows[:, None] + future_offsets[None]
        if not np.all(
            np.asarray(self.primary.data.episodes)[self.bank_future_rows]
            == self.bank_episodes[:, None]
        ):
            raise ValueError("demonstration future trajectory crosses an episode")
        frame_delta = (
            np.asarray(self.primary.data.frame_indices)[self.bank_future_rows]
            - np.asarray(self.primary.data.frame_indices)[self.bank_rows, None]
        )
        if not np.array_equal(
            frame_delta,
            np.broadcast_to(future_offsets, frame_delta.shape),
        ):
            raise ValueError("demonstration future trajectory is not frame-aligned")
        self.bank_future_states = np.asarray(
            self.primary.data.states[self.bank_future_rows], dtype=np.float32
        )
        self.primary_bank = encode_rows(
            self.model, self.primary, self.bank_rows, config.batch_size
        )
        self.secondary_bank = encode_rows(
            self.model, self.secondary, self.bank_rows, config.batch_size
        )
        _, bank_nominal, _ = project_history(
            self.bank_states,
            self.fits,
            self.kinematics,
            image_size=(224, 224),
        )
        self.bank_wrist_gripper_px = np.asarray(
            bank_nominal["wrist"][:, 0], dtype=np.float32
        )

        train_rows = np.concatenate(
            [
                np.flatnonzero(
                    np.asarray(self.primary.data.episodes) == episode
                )
                for episode in sorted(self.train_episodes)
            ]
        )
        train_states = np.asarray(
            self.primary.data.states[train_rows], dtype=np.float32
        )
        workspace = self.kinematics.forward(train_states)
        train_anchor_centers = self.kinematics.forward_anchors(train_states)[:, 0]
        self.contact_height_m = float(np.percentile(train_anchor_centers[:, 2], 10))
        self.transport_height_m = float(np.percentile(train_anchor_centers[:, 2], 65))
        if (
            not np.isfinite(config.release_clearance_m)
            or config.release_clearance_m < 0
        ):
            raise ValueError("release_clearance_m must be finite and >= 0")
        self.release_height_m = (
            self.transport_height_m + config.release_clearance_m
        )
        # Snapshot the as-derived heights before anything can rebind them, so a
        # published plan can say BOTH what training data implied and what the
        # session actually planned with, instead of silently reporting one and
        # obeying the other.
        self.bridge_derived_heights_m = {
            field: float(getattr(self, field))
            for field in LIVE_CONTRACT_HEIGHT_FIELDS
        }
        self.workspace_lower = np.min(workspace, axis=0) - config.workspace_margin_m
        self.workspace_upper = np.max(workspace, axis=0) + config.workspace_margin_m
        gripper = train_states[:, -1]
        self.gripper_calibration_closed = float(np.percentile(gripper, 1))
        self.gripper_calibration_open = float(np.percentile(gripper, 99))
        if self.gripper_calibration_open - self.gripper_calibration_closed < 20.0:
            raise ValueError(
                "training data does not contain a safe, observable gripper pulse"
            )
        self.thresholds = PhaseThresholds(
            pickup_near_px=28.0,
            # 25px ~ 3.8cm at column scale: releases must land inside the
            # container mouth's half-width, not merely near the rim (a 35px
            # gate accepted a release 5.5cm short of the mouth).
            drop_near_px=25.0,
            pickup_near_m=0.018,
            drop_near_m=0.025,
            gripper_closed=float(np.percentile(gripper, 25)),
            gripper_open=float(np.percentile(gripper, 80)),
            grasp_roi_px=30.0,
            confirmations=2,
            gripper_empty_closed=self.gripper_calibration_closed + 2.0,
        )
        self.phase_tracker = PhaseTracker(self.thresholds)
        self._transport_squeeze_chunks = 0
        self._grasp_empty_close_events = 0
        self._grasp_recovery_engaged = False
        # Persisted appearance templates and centroids live in whatever
        # planning geometry cut them, and a 224x224 crop is consistent with
        # both protocols, so a squash-registered profile would otherwise load
        # silently under a letterbox runtime and bias `target_px` by the pad
        # affine (worst at the top and bottom of frame, zero at the centre --
        # i.e. it reads as intermittent tracking noise, not as a geometry
        # fault).  Pin the tracker only once the caller has actually opted into
        # a new geometry: under the historical `squash_v0` default every
        # existing caller, including the recorded Tier 2 sessions whose
        # profiles predate the geometry field, must keep loading exactly as
        # before.  Requiring a declaration there would reject legacy profiles
        # that are in fact correct for their own runtime.
        # The calibration identity is pinned on the same condition, and for the
        # step the geometry stamp cannot take: geometry says WHICH pixel grid a
        # click lives on, not WHICH projection fit turns that pixel into a base
        # coordinate.  Two letterbox_v1 calibrations send one pixel to two
        # different places, so a profile registered under one and collected
        # under the other is invalid with nothing about it looking wrong.
        self.target_tracker = VisualTargetTracker(
            config.target_profile,
            required_geometry=(
                None
                if config.geometry_protocol == SQUASH_V0
                else config.planning_geometry
            ),
            registration_calibration=(
                None
                if config.geometry_protocol == SQUASH_V0
                else calibration_identity(
                    config.calibration, geometry_protocol=config.geometry_protocol
                )
            ),
            task_class=config.task_class,
        )
        # Peg/context registration is deliberately session-only.  The legacy
        # target profile has no camera-UUID or calibration binding, so writing
        # an obstacle reference into that v1 profile would make a stale click
        # look valid after a camera or calibration change.
        self.context_tracker = VisualTargetTracker(None)
        self.context_registration: dict[str, Any] | None = None
        self.live_anchor_offsets_px: dict[str, np.ndarray] = {}
        self.live_anchor_transforms_px: dict[str, np.ndarray] = {}
        # Rebuilt every round from what `nominal` was actually given. Empty here
        # so a projection taken before the first frame is prepared falls back to
        # no transform rather than to one that was never applied.
        self.applied_anchor_transforms_px: dict[str, np.ndarray | None] = {
            view: None for view in VIEWS
        }
        # 3-D neighbourhood each live transform was actually measured in; see
        # `set_live_moving_tip_track` and the scope check in `observe_grounding`.
        self.live_anchor_support: dict[str, dict[str, Any]] = {}
        # Same-view open-jaw tip spans observed while the gripper is clearly
        # open. Used only when that view has no pulse support of its own —
        # never as a stand-in for the other camera.
        self._open_jaw_tip_span_px: dict[str, float] = {}
        # Previous-round applied flags for the live-anchor transform hysteresis
        # gate.  Without these the radius threshold is a hard step inside the
        # closed loop.
        self._live_anchor_transform_applied: dict[str, bool] = {}
        self._column_extrinsic_observer = ColumnExtrinsicObserver(
            max_offset_px=float(config.column_extrinsic_max_offset_px),
        )
        self._recursive_observer = None
        self._observer_shadow = None
        self._observer_drop_zone = None
        self._observer_prev_joints = None
        self._observer_motion_records = []
        self._observer_last_diagnostics: dict[str, Any] = {"enabled": False}
        self._fingertip_gate = None
        self._fingertip_last_result = None
        self._anchor_posteriors: dict[str, _AnchorPosterior] = {}
        self._pickup_identity: dict[str, Any] | None = None
        self._pickup_goal_hold: dict[str, np.ndarray] | None = None
        self._pickup_goal_reused = False
        self._drop_goal_hold: dict[str, np.ndarray] | None = None
        self._drop_goal_reused = False
        self._descent_standoff: dict[str, bool] = {}
        self._fcpg_transit: dict[str, bool] = {}
        # Prior-round servo command vs gripper-anchor pixel, so true mm/px is
        # measurable from the archive after the transform gate stops flipping.
        self._servo_scale_prev: dict[str, Any] | None = None
        self._demonstration_progress: DemonstrationProgress | None = None
        self.model_contract = {
            "bridge_checkpoint": bridge_path,
            "bridge_source_checkpoint": source_checkpoint_path,
            "bridge_source_checkpoint_sha256": source_checkpoint_sha,
            "bridge_source_endpoint": int(source_checkpoint.get("step", -1)),
            "dynamics_checkpoint": checkpoint_path,
            "dynamics_checkpoint_sha256": checkpoint_sha,
            "dynamics_endpoint": int(checkpoint.get("step", -1)),
            "dynamics_protocol": selected_protocol,
            "dynamics_variant": (
                "v9_local_multiscale"
                if selected_protocol == V9_PROTOCOL
                else "v6_multiscale_memory"
                if selected_protocol == V6_PROTOCOL
                else "state_residual_backbone"
            ),
            "grounding_checkpoint": source_checkpoint_path,
            "grounding_checkpoint_sha256": source_checkpoint_sha,
            "grounding_contract": "legacy_fcpq_visual_semantics_for_current_anchor_measurement",
            "dynamics_context": (
                "shared_task_local_fcpg_backbone"
                if selected_protocol == V9_PROTOCOL
                else "separate_v6_backbone_with_fcpg_bridge_context"
            ),
            # B1: which pixel geometry this session actually ran in.
            **planning_geometry_contract_fields(config),
            "memory_policy": config.memory_policy,
            "memory_offsets_raw": list(self.memory_offsets_raw),
            "control_mode": config.control_mode,
            # Phase 2.5 provenance: a fresh MPC-only session must be
            # identifiable from its own plan.json, not from shell history.
            "mpc_only_collection": bool(config.mpc_only_collection),
            # Phase B0 item 5: the stronger "zero_shadow_logging_only" label is
            # now earned, but only for ONE scope --
            # MPC_ONLY_INDEPENDENCE_SCOPE, i.e. control_mode=phase_target at
            # the APPROACH phase with an unconfigured peg.  grasp, transport,
            # release and the peg-clearance branch are NOT execution-verified;
            # they share compose_candidate_decision but no test has driven a
            # plan through them.  The scope and the verified flag are built
            # together by one helper so a scopeless "verified" cannot be
            # emitted.
            **lewm_decision_independence_fields(
                mpc_only_collection=bool(config.mpc_only_collection)
            ),
            "experiment_method": config.experiment_method,
            "peg_visible_to_candidate_selection": (
                config.experiment_method == "full"
            ),
            "common_supervisor_peg_check": "always_when_registered",
            "frameskip": frameskip,
            "history": history,
            "horizon": self.horizon,
            "action_normalization": self.action_normalization,
            **self.live_contract_fields(),
            "gripper_calibration_positions": {
                "closed": self.gripper_calibration_closed,
                "open": self.gripper_calibration_open,
                "source": "train-only gripper position percentiles 1/99",
                "semantics": (
                    "commandable physical jaw stops; NOT the phase-classification "
                    "thresholds in phase_thresholds"
                ),
            },
            "workspace_lower_m": self.workspace_lower,
            "workspace_upper_m": self.workspace_upper,
            "workspace_margin_m": config.workspace_margin_m,
            "max_column_servo_step_m": config.max_column_servo_step_m,
            "column_descent_radius_m": config.column_descent_radius_m,
            "column_descent_hysteresis_band": config.column_descent_hysteresis_band,
            "column_servo_gain": config.column_servo_gain,
            "live_action_scale": config.live_action_scale,
            "effective_column_servo_gain": min(
                1.0, config.column_servo_gain * config.live_action_scale
            ),
            "grasp_jaw_rates": tuple(
                float(
                    np.clip(
                        rate * config.live_action_scale,
                        -float(config.max_prefix_step_jump),
                        float(config.max_prefix_step_jump),
                    )
                )
                for rate in GRASP_JAW_RATES
            ),
            "live_anchor_transform_hysteresis_band": (
                config.live_anchor_transform_hysteresis_band
            ),
            "servo_goal_hold_max_rounds": config.servo_goal_hold_max_rounds,
            "approach_candidates": (
                "bounded relative column-image servo, MuJoCo IK, and retrieved "
                "demonstrations"
            ),
            "cartesian_goal_scale_m": config.cartesian_goal_scale_m,
            "pixel_goal_weight": config.pixel_goal_weight,
            "target_contract": (
                "operator-configured pickup/drop_zone references plus a "
                "session-only round-bound column peg reference"
            ),
            "peg_context_contract": {
                "source": "operator column click on a frozen planning frame",
                "geometry": "affine task-plane backprojection; not metric depth",
                "tracking": "fresh appearance template; no occlusion hold",
                "clearance_m": config.peg_clearance_m,
                "clearance_semantics": (
                    "operator-declared composite peg-center-to-nominal-command-FK-"
                    "anchor XY envelope"
                ),
                "clearance_geometry": (
                    "minimum base-XY distance from the three nominal command-FK "
                    "anchor polylines to the registered peg center"
                ),
                "clearance_limitations": (
                    "not a link/mesh swept-volume collision guarantee; no Z, "
                    "volume, contact, or compliance model"
                ),
                "unconfigured_behavior": "no peg gate",
                "configured_failure_behavior": "fail closed",
            },
            "projection_calibration": config.calibration.expanduser().resolve(),
            "projection_calibration_sha256": _sha256_file(
                config.calibration.expanduser().resolve()
            ),
            "live_anchor_offsets_px": {},
            "live_anchor_transforms_px": {},
            "require_live_column_calibration": bool(
                config.require_live_column_calibration
            ),
            "recursive_grounding": {
                "enabled": bool(config.recursive_grounding),
                "propagation": (
                    "previous corrected anchor plus measured calibrated-FK "
                    "pixel displacement"
                ),
                "measurement_update": "current-frame local FCPG",
                "manual_label_supervision": False,
            },
            "coarse_fine_planner": {
                "variant": getattr(self, "_corrector_variant", "independent3"),
                "policy": dict(getattr(self, "_coarse_fine_policy", {})),
                "grounding_mode": config.grounding_mode,
                "ranking_mode": config.ranking_mode,
                "grounding_arm": grounding_arm_name(
                    config.grounding_mode, config.ranking_mode
                ),
                "wrist_correction_enabled": self._wrist_correction_allowed(),
                "allow_uncalibrated_coarse_fine_shadow": bool(
                    config.allow_uncalibrated_coarse_fine_shadow
                ),
                "request_live_execute": bool(config.request_live_execute),
                "next_seed_policy": "nominal",
            },
            "recursive_observer": {
                "shadow": bool(config.recursive_observer_shadow),
                "active_excitation": bool(
                    config.observer_active_excitation
                    and self._v165_runtime_adapter is None
                ),
                "close_sigma_m": (
                    config.observer_close_sigma_m
                    if self._v165_runtime_adapter is None
                    else None
                ),
                "control_influence": bool(
                    self._v165_runtime_adapter is None
                    and (
                        config.observer_active_excitation
                        or config.observer_close_sigma_m is not None
                    )
                ),
                "fingertip_gate": {
                    "reject_radius_px": float(config.fingertip_reject_radius_px),
                    "stale_age_frames": int(config.fingertip_stale_age_frames),
                    "stale_dq_deg": float(config.fingertip_stale_dq_deg),
                },
            },
            "servo_load_telemetry": {
                "enabled": bool(config.observer_servo_load),
                "registers": ["Present_Load", "Present_Current"],
                "frequency": "once_per_planning_round",
                "extra_sync_reads": 2,
                "control_influence": False,
            },
            "column_extrinsic_observer": {
                "enabled": bool(config.column_extrinsic_observer),
                "measurement": "column metal jaw tips versus FK",
                "estimator": "2d complementary / information filter",
                "max_offset_px": float(config.column_extrinsic_max_offset_px),
                "max_residual_px": float(config.column_extrinsic_max_residual_px),
                "applies_to": "column nominal before local FCPG",
            },
            "global_tip_locator": {
                "enabled": self.global_tip_head is not None,
                "views": config.global_tip_views,
                "checkpoint": self.global_tip_contract,
                "role": "markerless full-image coarse seed for local FCPG",
                "minimum_tip_confidence": config.global_tip_min_confidence,
                "edge_margin_px": config.global_tip_edge_margin_px,
                "max_motion_disagreement_px": (
                    config.global_tip_max_motion_disagreement_px
                ),
                "max_nominal_disagreement_px": (
                    config.global_tip_max_nominal_disagreement_px
                ),
                "center_rule": "exact midpoint of the two predicted fingertips",
            },
        }
        if self._v168_runtime_adapter is not None:
            self.model_contract.update(
                {
                    "v168_runtime": self._v168_runtime_adapter.receipt,
                    "v168_foundation": self._v165_runtime_adapter.receipt,
                    "anchor_authority": "original_fcpg",
                    "tuned_fcpg_present": False,
                    "legacy_v165_actor_present": False,
                    "legacy_v165_actor_candidate_enabled": False,
                    "prod_v13_control_influence": False,
                    "global_tip_control_influence": False,
                    "legacy_observer_sigma_controls_close": False,
                    "legacy_observer_excitation_control_influence": False,
                    "v165_no_contact_approach": bool(
                        config.v165_no_contact_approach
                    ),
                }
            )
        elif self._v165_runtime_adapter is not None:
            self.model_contract.update(
                {
                    "v165_runtime": self._v165_runtime_adapter.receipt,
                    "anchor_authority": self._v165_runtime_adapter.anchor_authority,
                    "prod_v13_control_influence": False,
                    "global_tip_control_influence": False,
                    "legacy_observer_sigma_controls_close": False,
                    "legacy_observer_excitation_control_influence": False,
                    "v165_no_contact_approach": bool(
                        config.v165_no_contact_approach
                    ),
                }
            )

    @property
    def v165_runtime_adapter(self) -> V165RuntimeAdapter | None:
        """Startup-created adapter; there is intentionally no mutation API."""

        return getattr(self, "_v165_runtime_adapter", None)

    @property
    def v168_runtime_adapter(self) -> V168LiveActorOverlay | None:
        """Startup-frozen overlay; there is intentionally no mutation API."""

        return getattr(self, "_v168_runtime_adapter", None)

    @property
    def v170_runtime_adapter(self) -> V170LiveStack | None:
        """Startup-created plain V17 stack; it deliberately has no actuator API."""

        return getattr(self, "_v170_runtime_adapter", None)

    @property
    def v170_shadow_adapter(self) -> V170LiveStack | None:
        """Compatibility name used by status/tests for the optional V17 stack."""

        return self.v170_runtime_adapter

    def v170_shadow_status(self) -> dict[str, Any] | None:
        adapter = self.v170_runtime_adapter
        if adapter is None:
            return None
        return {
            **adapter.status(),
            "last_measurement": self._last_v170_shadow,
            "last_error": self._last_v170_shadow_error,
        }

    def commit_v170_completed_chunk(
        self,
        *,
        requested_commands_deg: np.ndarray,
        sent_commands_deg: np.ndarray,
        measured_joint_path_deg: np.ndarray,
        round_id: str,
    ) -> dict[str, Any] | None:
        adapter = self.v170_runtime_adapter
        if adapter is None:
            return None
        return adapter.commit_completed_chunk(
            requested_commands_deg=requested_commands_deg,
            sent_commands_deg=sent_commands_deg,
            measured_joint_path_deg=measured_joint_path_deg,
            round_id=round_id,
        )

    def reset_v170_window(self, reason: str) -> dict[str, Any] | None:
        adapter = self.v170_runtime_adapter
        return None if adapter is None else adapter.reset_window(reason)

    def reset_v168_window(self, reason: str) -> dict[str, Any] | None:
        adapter = self.v168_runtime_adapter
        return None if adapter is None else adapter.reset_window(reason)

    def commit_v168_completed_chunk(
        self,
        *,
        sent_chunk: np.ndarray,
        round_id: str,
        selected_source: str | None,
    ) -> dict[str, Any] | None:
        adapter = self.v168_runtime_adapter
        if adapter is None:
            return None
        return adapter.commit_completed_chunk(
            sent_chunk,
            round_id=round_id,
            selected_source=selected_source,
        )

    def v165_contact_runtime_config(self) -> dict[str, Any]:
        """Stable planner/safety inputs bound by a contact authorization."""

        source_size = self.config.source_image_size
        return {
            "projection_calibration_sha256": str(
                self.model_contract["projection_calibration_sha256"]
            ),
            "live_anchor_calibration_sha256": self.model_contract.get(
                "live_anchor_calibration_sha256"
            ),
            "geometry_protocol": str(self.config.geometry_protocol),
            "source_image_size": (
                None
                if source_size is None
                else [int(source_size[0]), int(source_size[1])]
            ),
            "planning_image_size": int(self.config.planning_image_size),
            "contact_height_m": float(self.contact_height_m),
            "peg_clearance_m": float(self.config.peg_clearance_m),
            "max_first_command_jump_deg": float(
                self.config.max_first_command_jump
            ),
            "max_prefix_step_jump_deg": float(
                self.config.max_prefix_step_jump
            ),
            "live_action_scale": float(self.config.live_action_scale),
            "control_mode": str(self.config.control_mode),
            "experiment_method": str(self.config.experiment_method),
            "task_class": (
                None if self.config.task_class is None else int(self.config.task_class)
            ),
            "phase_contract": "phase_target_state_machine",
            "require_live_column_calibration": bool(
                self.config.require_live_column_calibration
            ),
        }

    def assert_v165_contact_runtime_binding(self) -> None:
        adapter = self.v165_runtime_adapter
        if adapter is None or not adapter.contact_hardware_authorized:
            return
        adapter.assert_contact_runtime_config(self.v165_contact_runtime_config())

    def _planner_corrector_variant(self) -> str:
        return str(getattr(self, "_corrector_variant", None) or "independent3")

    def _uses_coarse_fine_fusion(self) -> bool:
        return self._planner_corrector_variant() == COARSE_FINE_VARIANT

    def _grounding_ranking_contract(self):
        return planner_grounding_ranking_contract(
            grounding_mode=getattr(
                self.config, "grounding_mode", GROUNDING_GATED_FCPG
            ),
            ranking_mode=getattr(self.config, "ranking_mode", RANKING_ON),
        )

    def _lewm_ranking_disabled(self) -> bool:
        """True when LeWM must not change ranking costs.

        ``ranking_mode=off`` is ranking-only. ``mpc_only_collection`` is the
        stronger historical flag and also removes LeWM from eligibility.
        """

        if getattr(self.config, "ranking_mode", RANKING_ON) == RANKING_OFF:
            return True
        return bool(self.config.mpc_only_collection)

    def _lewm_eligibility_disabled(self) -> bool:
        return bool(self.config.mpc_only_collection)

    def _wrist_correction_allowed(self) -> bool:
        if self._uses_coarse_fine_fusion():
            return False
        return bool(getattr(self.config, "wrist_correction_enabled", True))

    def live_contract_fields(self) -> dict[str, Any]:
        """Contract fields read from live attributes, not from the frozen dict.

        ``model_contract`` is materialised once in ``__init__``.  Anything that
        rebinds a planner attribute afterwards -- the server's
        ``--trial-contact-height-m`` override is the load-bearing case -- changes
        what every planning site uses without changing what the contract
        publishes.  A published height that is not the planned height means the
        acceptance gates read from ``plan.json`` judge the wrong number.

        Every field here is therefore recomputed from the live attribute at
        plan-publication time, and the as-derived value is published alongside
        it so an override is visible rather than merely absent.
        """

        heights = {
            field: float(getattr(self, field))
            for field in LIVE_CONTRACT_HEIGHT_FIELDS
        }
        derived = getattr(self, "bridge_derived_heights_m", None) or dict(heights)
        return {
            **heights,
            "planning_height_provenance": {
                "in_use_m": dict(heights),
                "bridge_derived_m": {
                    field: float(derived.get(field, heights[field]))
                    for field in LIVE_CONTRACT_HEIGHT_FIELDS
                },
                "overridden_fields": tuple(
                    field
                    for field in LIVE_CONTRACT_HEIGHT_FIELDS
                    if field in derived
                    and float(derived[field]) != heights[field]
                ),
                # The override is a bare assignment with no frame conversion, so
                # spell the frame out: these are absolute base-datum heights, and
                # a table-relative figure is NOT interchangeable with them.
                "frame": "robot base datum (absolute); not table-relative",
                "bridge_derived_source": (
                    "train-only anchor-center Z percentiles 10/65, plus "
                    "release_clearance_m"
                ),
                "in_use_source": "live planner attribute at plan publication",
            },
            "phase_thresholds": {
                "pickup_near_px": self.thresholds.pickup_near_px,
                "drop_near_px": self.thresholds.drop_near_px,
                "pickup_near_m": self.thresholds.pickup_near_m,
                "drop_near_m": self.thresholds.drop_near_m,
                "gripper_closed": self.thresholds.gripper_closed,
                "gripper_open": self.thresholds.gripper_open,
                "gripper_empty_closed": self.thresholds.gripper_empty_closed,
                "grasp_roi_px": self.thresholds.grasp_roi_px,
                # Without this, `gripper_open` reads like the open STOP and gets
                # compared against gripper_calibration_positions.open (p99).  It
                # is a phase-classification percentile of the same training
                # column, ~12 counts lower, and the planner never commands it.
                "gripper_source": (
                    "train-only gripper position percentiles 25/80 "
                    "(phase classification only, not commandable stops)"
                ),
            },
        }

    @torch.inference_mode()
    def _global_tip_seed(
        self,
        view: str,
        feature_map: torch.Tensor,
        camera_context: np.ndarray,
        nominal: np.ndarray,
        fallback_seed: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.global_tip_head is None or view not in self.config.global_tip_views:
            reason = (
                "disabled"
                if self.global_tip_head is None
                else "disabled_for_view"
            )
            return fallback_seed, {
                "global_locator_enabled": False,
                "global_locator_accepted": False,
                "global_locator_rejection_reasons": (reason,),
            }
        maximum_separation = 126.0 if view == "wrist" else 56.0
        context = torch.from_numpy(camera_context).to(self.device)
        logits = self.global_tip_head(feature_map, context)
        # FK plus the projection fit already know how far apart, and along which
        # axis, the two jaws must appear for this joint state.  Without that the
        # pair scoring has a degenerate optimum at fixed==moving, which wins
        # every heatmap term at once and silently replaces a healthy nominal.
        expected_tip_offset = torch.from_numpy(
            np.ascontiguousarray(nominal[:, 2] - nominal[:, 1], dtype=np.float32)
        ).to(self.device)
        points, confidence = decode_structured_tip_heatmaps(
            logits,
            image_size=(224, 224),
            max_tip_separation_px=maximum_separation,
            expected_tip_offset_px=expected_tip_offset,
        )
        points_np = points.cpu().numpy().astype(np.float32)
        confidence_np = confidence.cpu().numpy().astype(np.float32)
        quality = global_tip_sequence_diagnostics(
            points_np,
            confidence_np,
            nominal,
            # `image_size` stays the CANVAS: it validates the heatmap decode
            # grid above, which really is 224x224 under both protocols.  Only
            # the visibility gate becomes content-aware, because "about to
            # leave the camera frame" is a statement about camera pixels.
            image_size=(224, 224),
            content_rect=self.planning_geometry.content_rect,
            max_tip_separation_px=maximum_separation,
            minimum_tip_confidence=self.config.global_tip_min_confidence,
            edge_margin_px=self.config.global_tip_edge_margin_px,
            max_motion_disagreement_px=(
                self.config.global_tip_max_motion_disagreement_px
            ),
            max_nominal_disagreement_px=(
                self.config.global_tip_max_nominal_disagreement_px
            ),
        )
        accepted = bool(quality["accepted"])
        return (points_np if accepted else fallback_seed), {
            "global_locator_enabled": True,
            "global_locator_accepted": accepted,
            "global_locator_rejection_reasons": quality["rejection_reasons"],
            "global_locator_points_px": points_np,
            "global_locator_confidence": confidence_np,
            "global_locator_quality": quality,
        }

    def reset_grounding_posterior(self) -> None:
        """Forget recursive image anchors after a discontinuous robot reset."""

        self._anchor_posteriors = {}

    def _column_applied_transform_px(self) -> np.ndarray | None:
        """Column image transform available for backprojection this round.

        Prefer the matrix actually written onto ``nominal`` this frame.  If
        that slot is empty, keep a stored pulse or the observer estimate —
        both are camera-global translations, and fail-closing on
        ``applied is None`` while a pulse file is loaded freezes execute.
        """

        applied = getattr(self, "applied_anchor_transforms_px", None)
        if applied is not None:
            column = applied.get("column")
            if column is not None:
                return column
        observer = getattr(self, "_column_extrinsic_observer", None)
        if observer is not None:
            getter = getattr(observer, "transform_px", None)
            if callable(getter):
                observed = getter()
                if observed is not None:
                    return observed
        live = getattr(self, "live_anchor_transforms_px", None) or {}
        return live.get("column")

    def _apply_column_hard_match_residual(
        self, paths: np.ndarray
    ) -> np.ndarray:
        """Shift projected column paths from pulse/FK onto this frame's metal."""

        return apply_column_frame_offset_px(
            paths, getattr(self, "_column_hard_match_residual_px", None)
        )

    def reset_task_progress(self) -> None:
        """Forget demonstration alignment after a discontinuous task reset."""

        self._demonstration_progress = None
        self.reset_pickup_identity()

    def commit_executed_plan(
        self,
        plan: RealtimePlan,
        *,
        command_count: int,
    ) -> dict[str, int] | None:
        """Advance demonstration time only after commands completed on the robot."""

        # Noted before the bookkeeping below can refuse the plan, because the
        # chunk reached the arm whatever this method then decides about
        # demonstration time. Read defensively because the only consumer is the
        # residual instrument: an unreadable round id costs it one record,
        # where raising here would abort a commit for a motion already made.
        guard = getattr(plan, "guard", None)
        round_id = getattr(guard, "round_id", None)
        if round_id is not None:
            self.note_round_executed(round_id)
        if plan.phase == ManipulationPhase.TRANSPORT:
            self._transport_squeeze_chunks += 1
        if plan.phase != ManipulationPhase.TRAJECTORY:
            return None
        if command_count != self.config.command_chunk:
            raise ValueError(
                "trajectory progress requires one complete planner command chunk"
            )
        if plan.selected_index is None:
            raise ValueError("cannot advance an unselected trajectory plan")
        if not 0 <= plan.selected_index < len(plan.candidate_rows):
            raise ValueError("selected trajectory candidate index is invalid")
        selected = plan.candidate_rows[plan.selected_index]
        if selected.get("source") not in {
            "implicit_successful_demonstration",
            "locked_demonstration_continuation",
        }:
            raise ValueError("selected trajectory candidate has no demonstration source")
        episode = int(selected["episode"])
        frame = int(selected["frame"])
        if self._demonstration_progress is not None and (
            episode != self._demonstration_progress.episode
            or frame != self._demonstration_progress.next_frame
        ):
            raise ValueError(
                "executed trajectory candidate does not match committed progress"
            )
        next_frame = frame + command_count
        completed_chunks = (
            1
            if self._demonstration_progress is None
            else self._demonstration_progress.completed_chunks + 1
        )
        self._demonstration_progress = DemonstrationProgress(
            episode=episode,
            next_frame=next_frame,
            completed_chunks=completed_chunks,
        )
        return {
            "episode": episode,
            "executed_source_frame": frame,
            "next_source_frame": next_frame,
            "completed_chunks": completed_chunks,
        }

    def set_live_anchor_center(
        self,
        view: str,
        center_px: np.ndarray,
        state: np.ndarray,
        *,
        anchor_index: int = 0,
    ) -> np.ndarray:
        """Bind a local projection offset from one observed physical anchor."""

        _, offset = self.set_live_anchor_candidates(
            view,
            np.asarray(center_px, dtype=np.float32)[None],
            state,
            anchor_index=anchor_index,
        )
        return offset

    def set_live_anchor_candidates(
        self,
        view: str,
        candidates_px: np.ndarray,
        state: np.ndarray,
        *,
        anchor_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select the observed envelope endpoint nearest one nominal anchor."""

        if view not in VIEWS:
            raise ValueError(f"unsupported live-anchor view: {view}")
        candidates = np.asarray(candidates_px, dtype=np.float32)
        current_state = np.asarray(state, dtype=np.float32)
        if (
            candidates.ndim != 2
            or candidates.shape[1:] != (2,)
            or len(candidates) == 0
            or not np.all(np.isfinite(candidates))
        ):
            raise ValueError("live anchor candidates must be finite [N,2]")
        if current_state.shape != (6,) or not np.all(np.isfinite(current_state)):
            raise ValueError("live anchor calibration state must be a finite six-vector")
        _, nominal, _ = project_history(
            current_state[None],
            self.fits,
            self.kinematics,
            image_size=(224, 224),
        )
        anchor = np.asarray(nominal[view][-1, anchor_index], dtype=np.float32)
        offsets = candidates - anchor[None]
        selected_index = int(np.argmin(np.linalg.norm(offsets, axis=1)))
        live_center = candidates[selected_index]
        _, offset = apply_live_center_translation(
            nominal[view], live_center, anchor_index=anchor_index
        )
        if float(np.linalg.norm(offset)) > 35.0:
            raise ValueError(
                "live anchor offset exceeds the 35 px local calibration envelope: "
                f"view={view}, anchor_index={anchor_index}, offset={offset.tolist()}"
            )
        self.live_anchor_offsets_px[view] = offset.copy()
        transform = np.asarray(
            [[1.0, 0.0, offset[0]], [0.0, 1.0, offset[1]]],
            dtype=np.float32,
        )
        self.live_anchor_transforms_px[view] = transform
        if view == "column" and hasattr(self, "context_tracker"):
            self.clear_context_reference()
        self.model_contract.pop("live_anchor_calibration_source", None)
        self.model_contract.pop("live_anchor_calibration_sha256", None)
        self.reset_grounding_posterior()
        self.model_contract["live_anchor_offsets_px"] = {
            name: values.tolist()
            for name, values in sorted(self.live_anchor_offsets_px.items())
        }
        self.model_contract["live_anchor_transforms_px"] = {
            name: values.tolist()
            for name, values in sorted(self.live_anchor_transforms_px.items())
        }
        return live_center.copy(), offset.copy()

    def set_live_moving_tip_track(
        self,
        view: str,
        moving_tip_track_candidates_px: np.ndarray,
        open_state: np.ndarray,
        closed_state: np.ndarray,
        *,
        center_anchor_index: int = 0,
        fixed_tip_anchor_index: int = 1,
        moving_tip_anchor_index: int = 2,
        minimum_scale: float = 0.5,
        maximum_scale: float = 1.5,
        maximum_rotation_deg: float = DEFAULT_MAXIMUM_LIVE_ROTATION_DEG,
        # 35 px was the old local envelope. Clean Jieli pulses now report
        # 46-56 px across the free-space poses. 60 admits that measured range
        # and still rejects the ~91 px shadow-contaminated candidate.
        maximum_center_offset_px: float | None = None,
        maximum_residual_px: float | None = None,
        exclude_bboxes_xyxy: Sequence[Sequence[float]] = (),
        prefer_bboxes_xyxy: Sequence[Sequence[float]] = (),
        metal_tip_hint_px: Sequence[float] | None = None,
        metal_tip_hint_radius_px: float | None = None,
    ) -> dict[str, Any]:
        """Fit image geometry from one moving fingertip observed open and closed."""

        if view not in VIEWS:
            raise ValueError(f"unsupported live-anchor view: {view}")
        candidates = np.asarray(moving_tip_track_candidates_px, dtype=np.float32)
        open_joint_state = np.asarray(open_state, dtype=np.float32)
        closed_joint_state = np.asarray(closed_state, dtype=np.float32)
        if (
            open_joint_state.shape != (6,)
            or closed_joint_state.shape != (6,)
            or not np.all(np.isfinite(open_joint_state))
            or not np.all(np.isfinite(closed_joint_state))
        ):
            raise ValueError(
                "open and closed calibration states must be finite six-vectors"
            )
        calibration_states = np.stack((open_joint_state, closed_joint_state))
        calibration_xyz, nominal, _ = project_history(
            calibration_states,
            self.fits,
            self.kinematics,
            image_size=(224, 224),
        )
        nominal_anchors = np.asarray(nominal[view], dtype=np.float32)
        # A 4-DOF similarity fitted to a 2-point track is exactly determined: its
        # residual is identically zero, so the fit can never reveal that it is
        # wrong.  A translation consumes the same four observations with two
        # degrees of freedom, so the leftover redundancy becomes a measured
        # disagreement between the observed jaw sweep and the FK prediction of
        # it -- which is what ranks the candidates below and what the runtime
        # gates on.  Measured on the archived pulses: a correctly extracted
        # track residual is 1.7-2.6 px, a shadow-contaminated one 21-25 px.
        config = getattr(self, "config", None)
        column_config = config if view == "column" else None
        resolved_maximum_center_offset_px = float(
            60.0
            if maximum_center_offset_px is None and column_config is None
            else (
                getattr(column_config, "column_extrinsic_max_offset_px", 60.0)
                if maximum_center_offset_px is None
                else maximum_center_offset_px
            )
        )
        resolved_maximum_residual_px = (
            None
            if maximum_residual_px is None and column_config is None
            else float(
                getattr(column_config, "column_extrinsic_max_residual_px", 8.0)
                if maximum_residual_px is None
                else maximum_residual_px
            )
        )
        selected = select_live_moving_tip_translation(
            nominal_anchors,
            candidates,
            center_anchor_index=center_anchor_index,
            fixed_tip_anchor_index=fixed_tip_anchor_index,
            moving_tip_anchor_index=moving_tip_anchor_index,
            maximum_center_offset_px=resolved_maximum_center_offset_px,
            maximum_residual_px=resolved_maximum_residual_px,
            exclude_bboxes_xyxy=exclude_bboxes_xyxy,
            prefer_bboxes_xyxy=prefer_bboxes_xyxy,
            metal_tip_hint_px=metal_tip_hint_px,
            metal_tip_hint_radius_px=metal_tip_hint_radius_px,
        )
        transform = np.asarray(selected["transform_px"], dtype=np.float32)
        # The residual above only proves the pulse and FK agree AT THE POSE THE
        # PULSE WAS TAKEN.  It says nothing about validity elsewhere: measured
        # across the two archived poses the true correction drifts by 0.271
        # px/mm, so a self-consistent fit (residual 1.8 px) is still 27 px wrong
        # 99 mm away.  Record the support so the transform can be withheld where
        # nothing was measured.  The two guards are complementary and both are
        # required; neither subsumes the other.
        support_center = np.asarray(
            calibration_xyz[:, center_anchor_index].mean(axis=0), dtype=np.float32
        )
        support_radius = float(
            np.linalg.norm(
                calibration_xyz[0, moving_tip_anchor_index]
                - calibration_xyz[1, moving_tip_anchor_index]
            )
        )
        self.live_anchor_support[view] = {
            "center_xyz": support_center,
            "radius_m": support_radius,
            "residual_rms_px": float(selected["residual_rms_px"]),
            "maximum_residual_px": float(selected["maximum_residual_px"]),
            "open_tip_separation_px": float(
                np.linalg.norm(
                    nominal_anchors[0, moving_tip_anchor_index]
                    - nominal_anchors[0, fixed_tip_anchor_index]
                )
            ),
            "empty_close_tip_separation_px": float(
                np.linalg.norm(
                    nominal_anchors[1, moving_tip_anchor_index]
                    - nominal_anchors[1, fixed_tip_anchor_index]
                )
            ),
            # Joint-space companion to the span above, and the only one of the
            # two that is measured rather than projected: reaching it means the
            # jaws closed on each other, so nothing is held.
            "empty_closed_gripper_position": float(closed_joint_state[-1]),
        }
        if selected.get("selection_reason") is not None:
            self.live_anchor_support[view]["selection_reason"] = str(
                selected["selection_reason"]
            )
        if selected.get("metal_tip_hint_px") is not None:
            self.live_anchor_support[view]["metal_tip_hint_px"] = [
                float(selected["metal_tip_hint_px"][0]),
                float(selected["metal_tip_hint_px"][1]),
            ]
            self.live_anchor_support[view]["metal_tip_hint_radius_px"] = float(
                selected["metal_tip_hint_radius_px"]
            )
            self.live_anchor_support[view]["hint_distance_px"] = float(
                selected["hint_distance_px"]
            )
            self.live_anchor_support[view]["candidate_hint_distances_px"] = [
                float(value) for value in selected.get("candidate_hint_distances_px", [])
            ]
        self.live_anchor_transforms_px[view] = transform.copy()
        if view == "column" and hasattr(self, "context_tracker"):
            self.clear_context_reference()
        self.model_contract.pop("live_anchor_calibration_source", None)
        self.model_contract.pop("live_anchor_calibration_sha256", None)
        self.reset_grounding_posterior()
        self.live_anchor_offsets_px.pop(view, None)
        self.model_contract["live_anchor_offsets_px"] = {
            name: values.tolist()
            for name, values in sorted(self.live_anchor_offsets_px.items())
        }
        self.model_contract["live_anchor_transforms_px"] = {
            name: values.tolist()
            for name, values in sorted(self.live_anchor_transforms_px.items())
        }
        self.model_contract["live_anchor_support"] = {
            name: {
                "center_xyz": values["center_xyz"].tolist(),
                "radius_m": values["radius_m"],
                **{
                    key: values[key]
                    for key in (
                        "residual_rms_px",
                        "maximum_residual_px",
                        "open_tip_separation_px",
                        "empty_close_tip_separation_px",
                        "empty_closed_gripper_position",
                        "selection_reason",
                        "metal_tip_hint_px",
                        "metal_tip_hint_radius_px",
                        "hint_distance_px",
                        "candidate_hint_distances_px",
                    )
                    if key in values
                },
            }
            for name, values in sorted(self.live_anchor_support.items())
        }
        return {
            **selected,
            "open_state": open_joint_state.copy(),
            "closed_state": closed_joint_state.copy(),
            "support_center_xyz": support_center.copy(),
            "support_radius_m": support_radius,
        }

    def _anchor_paths_px(
        self,
        *,
        predicted: np.ndarray | None,
        commanded: np.ndarray | None,
        nominal: Mapping[str, np.ndarray],
        corrected: Mapping[str, np.ndarray],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Project the selected rollout into every view, not just the column.

        Only the column projection used to reach the plan, so only the column
        frame could be drawn with a forecast on it, and the wrist -- the view
        that actually resolves the jaws at grasp range -- showed bare anchors.
        """

        paths: dict[str, dict[str, np.ndarray]] = {}
        for view in VIEWS:
            correction = corrected[view][-1] - nominal[view][-1]
            transform = (getattr(self, "applied_anchor_transforms_px", None) or {}).get(view)
            per_view: dict[str, np.ndarray] = {}
            for name, states in (("predicted", predicted), ("commanded", commanded)):
                if states is None:
                    continue
                projected = project_future_anchor_pixels(
                    np.asarray(states)[None],
                    self.fits[view],
                    self.kinematics,
                    current_correction_px=correction,
                    live_transform_px=transform,
                )[0]
                if view == "column":
                    projected = self._apply_column_hard_match_residual(projected)
                per_view[name] = projected
            if per_view:
                paths[view] = per_view
        return paths

    def _three_point_closure(
        self,
        corrected: Mapping[str, np.ndarray],
        valid: Mapping[str, np.ndarray],
        column_pickup: Mapping[str, Any],
        wrist_pickup: Mapping[str, Any],
        gripper_position: float | None = None,
    ) -> dict[str, Any]:
        """Score held / empty-bite from live FCPG three-anchor geometry."""

        empty_closed_gripper = self._empty_closed_for_confirmation()

        def _last_anchors(view: str) -> tuple[np.ndarray | None, np.ndarray | None]:
            series = corrected.get(view)
            flags = valid.get(view)
            if series is None or flags is None:
                return None, None
            anchors = np.asarray(series, dtype=np.float64)
            flag = np.asarray(flags)
            if anchors.ndim != 3 or anchors.shape[-2:] != (3, 2) or len(anchors) == 0:
                return None, None
            return anchors[-1], flag[-1] if len(flag) else None

        wrist_anchors, wrist_valid = _last_anchors("wrist")
        column_anchors, _column_valid = _last_anchors("column")
        self._note_open_jaw_span("wrist", wrist_anchors, gripper_position)
        self._note_open_jaw_span("column", column_anchors, gripper_position)
        wrist_ok = (
            wrist_anchors is not None
            and wrist_valid is not None
            and np.all(np.isfinite(wrist_anchors))
            and bool(np.all(wrist_valid))
            and native_spans_usable(*self._native_support_spans("wrist"))
        )
        if wrist_ok:
            target = None
            relative = wrist_pickup.get("relative_centroid_px")
            if relative is not None:
                offset = np.asarray(relative, dtype=np.float64)
                if offset.shape == (2,) and np.all(np.isfinite(offset)):
                    target = wrist_anchors[0] + offset
            elif wrist_pickup.get("visible") and wrist_pickup.get("centroid_px") is not None:
                candidate = np.asarray(wrist_pickup["centroid_px"], dtype=np.float64)
                if candidate.shape == (2,) and np.all(np.isfinite(candidate)):
                    target = candidate
            # Wrist anchors are scored only with wrist spans. Missing wrist
            # pulse support must not borrow the column pulse: that mix is the
            # live 0.354 in_corridor stall (wrist 28.7 px vs column 40.9/6.4).
            open_span, empty_span = self._native_support_spans("wrist")
            near = judge_three_point_closure(
                wrist_anchors,
                target,
                open_tip_separation_px=open_span,
                empty_close_tip_separation_px=empty_span,
                gripper_position=gripper_position,
                empty_closed_gripper_position=empty_closed_gripper,
                **measured_object_shape(wrist_pickup),
            )
            return self._confirm_across_views(
                near,
                column_anchors,
                column_pickup,
                gripper_position=gripper_position,
                near_anchors=wrist_anchors,
                near_pickup=wrist_pickup,
            )

        if column_anchors is None:
            column_anchors = np.full((3, 2), np.nan, dtype=np.float64)
        target = None
        if column_pickup.get("visible") and column_pickup.get("centroid_px") is not None:
            candidate = np.asarray(column_pickup["centroid_px"], dtype=np.float64)
            if candidate.shape == (2,) and np.all(np.isfinite(candidate)):
                target = candidate
        open_span, empty_span = self._native_support_spans("column")
        return judge_three_point_closure(
            column_anchors,
            target,
            open_tip_separation_px=open_span,
            empty_close_tip_separation_px=empty_span,
            gripper_position=gripper_position,
            empty_closed_gripper_position=empty_closed_gripper,
            **measured_object_shape(column_pickup),
        )

    def _normalized_pickup_offset(
        self,
        anchors: np.ndarray | None,
        pickup: Mapping[str, Any],
    ) -> float | None:
        """How far the pickup sits from the jaws, in that view's jaw widths.

        The unit is the view's own FK-projected tip separation, so the number is
        produced entirely inside one view and is still comparable with the other
        view's without either having been calibrated against it. That is the
        whole point: the two views share no frame, but they do share the
        physical gap they are both looking at.
        """

        if anchors is None or not pickup.get("visible", False):
            return None
        offset = pickup.get("relative_centroid_px")
        if offset is None:
            return None
        anchors_px = np.asarray(anchors, dtype=np.float64)
        offset_px = np.asarray(offset, dtype=np.float64)
        if anchors_px.shape != (3, 2) or offset_px.shape != (2,):
            return None
        if not (np.all(np.isfinite(anchors_px)) and np.all(np.isfinite(offset_px))):
            return None
        separation = float(np.linalg.norm(anchors_px[1] - anchors_px[2]))
        if separation < self.config.cross_view_separation_floor_px:
            return None
        return float(np.linalg.norm(offset_px)) / separation

    def _withhold_on_cross_view_disagreement(
        self,
        confirmed: dict[str, Any],
        near_anchors: np.ndarray | None,
        near_pickup: Mapping[str, Any] | None,
        far_anchors: np.ndarray | None,
        column_pickup: Mapping[str, Any],
        *,
        gripper_position: float | None = None,
    ) -> dict[str, Any]:
        """Refuse a closure the two views cannot both account for.

        Unlike the betweenness confirmation above, this compares *distance* and
        so it still applies when the far view returned ``insufficient``. It is
        one-sided on purpose: a column that places the pickup further away than
        the wrist only ever delays a grasp, while the reverse is the shape that
        closes empty jaws.
        """

        if near_pickup is None:
            return confirmed
        near_ratio = self._normalized_pickup_offset(near_anchors, near_pickup)
        far_ratio = self._normalized_pickup_offset(far_anchors, column_pickup)
        if near_ratio is None or far_ratio is None:
            return confirmed
        disagreement = near_ratio - far_ratio
        confirmed["near_view_pickup_offset_ratio"] = near_ratio
        confirmed["far_view_pickup_offset_ratio"] = far_ratio
        confirmed["cross_view_offset_disagreement"] = disagreement
        if (
            disagreement > self.config.cross_view_max_offset_disagreement
            and confirmed["verdict"] in {"held", "in_corridor"}
            and not (
                confirmed["verdict"] == "held"
                and jaws_in_holding_aperture(
                    gripper_position,
                    self._empty_closed_for_confirmation(),
                )
            )
        ):
            confirmed["withheld_by_cross_view_disagreement"] = confirmed["verdict"]
            confirmed["verdict"] = "approach_more"
        return confirmed

    def _confirm_across_views(
        self,
        near: dict[str, Any],
        far_anchors: np.ndarray | None,
        column_pickup: Mapping[str, Any],
        *,
        gripper_position: float | None,
        near_anchors: np.ndarray | None = None,
        near_pickup: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Let the far view veto a closure the near view cannot actually see.

        "Between the tips" is a question about where the object is along the
        direction the jaws close, and the wrist cannot answer it. Measured on
        385 archived rounds, that direction is the one its projection does not
        observe: the jaw anchors span 0.1mm across it, so a pixel there is worth
        7.59mm and an object well beyond the tips lands between them on the
        sensor. Live, that read a fish 123mm away as ``held``, the phase machine
        moved to transport, and the jaws had closed on nothing.

        The far view is not a better camera; it is a second direction. Where it
        can see the object it is allowed to withhold the promotion, and where it
        cannot -- occluded by the gripper it is watching, which is the normal
        case at contact -- it returns ``insufficient`` and says nothing. An
        empty bite from either view stands on its own: those readings are
        physical rather than projective.

        Betweenness is not the only thing the pair can disagree about, and the
        column is not always the sound one. ``_withhold_on_cross_view_
        disagreement`` runs on every path out of here and compares how far each
        view puts the pickup from its own jaws, which catches a column whose
        projection has moved out from under it.
        """

        def _with_offset_check(verdict: dict[str, Any]) -> dict[str, Any]:
            return self._withhold_on_cross_view_disagreement(
                dict(verdict),
                near_anchors,
                near_pickup,
                far_anchors,
                column_pickup,
                gripper_position=gripper_position,
            )

        if far_anchors is None or not np.all(np.isfinite(far_anchors)):
            return _with_offset_check(near)
        target = None
        if column_pickup.get("visible") and column_pickup.get("centroid_px") is not None:
            candidate = np.asarray(column_pickup["centroid_px"], dtype=np.float64)
            if candidate.shape == (2,) and np.all(np.isfinite(candidate)):
                target = candidate
        if target is None:
            return _with_offset_check(near)
        open_span, empty_span = self._native_support_spans("column")
        if not native_spans_usable(open_span, empty_span):
            return _with_offset_check(near)
        far = judge_three_point_closure(
            far_anchors,
            target,
            open_tip_separation_px=open_span,
            empty_close_tip_separation_px=empty_span,
            gripper_position=gripper_position,
            empty_closed_gripper_position=self._empty_closed_for_confirmation(),
            **measured_object_shape(column_pickup),
        )
        confirmed = dict(near)
        confirmed["far_view_verdict"] = far["verdict"]
        confirmed["far_view_target_between_tips"] = far["target_between_tips"]
        confirmed["far_view_closing_progress"] = far.get("closing_progress")
        if far["verdict"] == "insufficient":
            return _with_offset_check(confirmed)
        hold_protected = near["verdict"] == "held" and jaws_in_holding_aperture(
            gripper_position,
            self._empty_closed_for_confirmation(),
        )
        if (
            near["verdict"] in {"held", "in_corridor"}
            and not far["target_between_tips"]
            and not hold_protected
        ):
            # Downgraded to the gap, never to a bite. The far view's own bite
            # readings are not its to give: the physical one is the gripper
            # angle, which the near view already has, and the projective one
            # compares a pixel tip separation against a fixed threshold, which
            # shrinks with distance from the camera. Live that called an open
            # 10.51 degree jaw an empty bite from 210mm away. A pinched hold
            # is not un-held just because the column still sees pickup.
            confirmed["verdict"] = "approach_more"
            confirmed["withheld_by_far_view"] = near["verdict"]
        elif (
            near["verdict"] != "empty_bite"
            and native_progress_held(far)
            and not native_progress_held(near)
        ):
            # A column-native hold must not be vetoed by a wrist in_corridor
            # that had no wrist spans (or was scored on borrowed column
            # spans). Live that mix sat at progress 0.354 forever.
            confirmed["verdict"] = "held"
            confirmed["promoted_by_native_view"] = "column"
            confirmed["target_between_tips"] = True
            confirmed["closing_progress"] = far.get("closing_progress")
        return _with_offset_check(confirmed)

    def _hold_world_model_expectation(
        self,
        raw_contexts: Mapping[str, np.ndarray],
        normalized_states: np.ndarray,
        past_chunks: np.ndarray,
        executed_chunk: np.ndarray,
        *,
        round_id: str,
    ) -> None:
        """Record what the world model expects to see after the chunk it chose.

        The rollout that ranked the candidates produced this already and threw
        it away; this repeats it for the one candidate that will actually run,
        so the expectation on file belongs to an action the world answered.
        """

        policy = getattr(self, "v11_6_lewm", None)
        if policy is None:
            return
        try:
            common = dict(
                state_center=self.state_center,
                state_scale=self.state_scale,
                action_center=self.action_center,
                action_scale=self.action_scale,
                device=self.device,
                round_id=round_id,
            )
            blind = predict_future_context(
                policy,
                raw_contexts["wrist"],
                raw_contexts["column"],
                normalized_states,
                past_chunks,
                executed_chunk,
                **common,
            )
            grounded = None
            anchors = self._fcpg_anchor_inputs()
            if anchors is not None:
                grounded = predict_future_context(
                    policy,
                    raw_contexts["wrist"],
                    raw_contexts["column"],
                    normalized_states,
                    past_chunks,
                    executed_chunk,
                    **common,
                    **anchors,
                )
            self._pending_expectation = blind
            self._pending_expectation_grounded = grounded
            self._pending_expectation_seen = {
                view: np.asarray(raw_contexts[view], dtype=np.float32)[-1].copy()
                for view in RESIDUAL_VIEWS
            }
        except Exception as failure:
            # An instrument may not take the loop down with it, and a named set
            # of exception types was not that promise: ranking reaches the
            # frozen head through ``v11_lewm_ranker`` and returns before
            # validating ``v11_6_lewm``, so this is the first caller to type
            # check it, and that check raises TypeError. The failure is not
            # swallowed -- it is published in the round's residual record.
            self._pending_expectation = None
            self._pending_expectation_seen = None
            self.last_world_model_residual = {"error": f"{type(failure).__name__}: {failure}"}

    def note_round_executed(self, round_id: str) -> None:
        """Told by the runtime that this round's chunk reached the arm."""

        self._executed_round = str(round_id)

    def _fcpg_anchor_inputs(self) -> dict[str, np.ndarray] | None:
        """FCPG's measured anchors, shaped the way the dynamics head expects.

        ``None`` when the last grounding pass did not produce a usable set, in
        which case the blind assembly is the only question that can be asked.
        """

        grounding = getattr(self, "_last_grounding", None)
        if grounding is None:
            return None
        try:
            corrected = np.stack(
                [np.asarray(grounding.corrected[view], dtype=np.float32)
                 for view in RESIDUAL_VIEWS],
                axis=1,
            )
            seen = np.stack(
                [np.asarray(grounding.valid[view], dtype=np.float32)
                 for view in RESIDUAL_VIEWS],
                axis=1,
            )
            xyz = np.asarray(grounding.anchors_xyz, dtype=np.float32)
        except (AttributeError, KeyError, ValueError):
            return None
        if corrected.ndim != 4 or seen.ndim != 3 or xyz.ndim != 3:
            return None
        if not (np.all(np.isfinite(corrected)) and np.all(np.isfinite(xyz))):
            return None
        return {"anchor_px": corrected, "anchor_xyz": xyz, "anchor_seen": seen}

    def _settle_world_model_expectation(
        self, raw_contexts: Mapping[str, np.ndarray]
    ) -> None:
        """Compare the held expectation against the scene that actually arrived."""

        expectation = getattr(self, "_pending_expectation", None)
        if expectation is None:
            return
        # Only an expectation the arm actually acted on is worth settling. A
        # replan that supersedes an unexecuted chunk leaves a prediction of a
        # motion that never happened next to a scene that only moved by camera
        # noise, and differencing those measures nothing: measured that way the
        # residual read 6.6 against an observed change of 0.18, which is the
        # arithmetic of a stationary arm rather than a fact about the model.
        if getattr(self, "_executed_round", None) != expectation.round_id:
            self._pending_expectation = None
            self._pending_expectation_grounded = None
            self._pending_expectation_seen = None
            return
        before = getattr(self, "_pending_expectation_seen", None)
        grounded = getattr(self, "_pending_expectation_grounded", None)
        self._pending_expectation = None
        self._pending_expectation_grounded = None
        self._pending_expectation_seen = None
        try:
            now = {
                view: np.asarray(raw_contexts[view], dtype=np.float32)[-1]
                for view in RESIDUAL_VIEWS
            }
            # Every horizon step, not just the last. The model was trained to
            # predict 5, 10, 15 and 20 archive frames ahead, and one replan
            # here advances the scene by about a third of what the furthest of
            # those covers, so which step to compare against is a question for
            # the data rather than for a default.
            record: dict[str, Any] = {}
            for step in range(int(expectation.context.shape[0])):
                for name, value in context_residual(
                    expectation, now["wrist"], now["column"], horizon_index=step
                ).items():
                    record[f"h{step}_{name}"] = value
                if grounded is not None:
                    for name, value in context_residual(
                        grounded, now["wrist"], now["column"], horizon_index=step
                    ).items():
                        record[f"h{step}_grounded_{name}"] = value
            if before is not None:
                record.update(
                    observed_change(
                        before["wrist"], before["column"], now["wrist"], now["column"]
                    )
                )
            record["predicted_for_round"] = expectation.round_id
            self.last_world_model_residual = record
        except Exception as failure:
            # Same contract as holding the expectation: reported in the record,
            # never raised into the round that was only being measured.
            self.last_world_model_residual = {
                "error": f"{type(failure).__name__}: {failure}"
            }

    def _pulse_support_spans(
        self, view: str
    ) -> tuple[float | None, float | None]:
        support = getattr(self, "live_anchor_support", None) or {}
        if not isinstance(support, Mapping):
            return None, None
        entry = support.get(view)
        if not isinstance(entry, Mapping):
            return None, None
        open_span = entry.get("open_tip_separation_px")
        empty_span = entry.get("empty_close_tip_separation_px")
        return (
            None if open_span is None else float(open_span),
            None if empty_span is None else float(empty_span),
        )

    def _note_open_jaw_span(
        self,
        view: str,
        anchors: np.ndarray | None,
        gripper_position: float | None,
    ) -> None:
        """Record a same-view open-jaw tip span while the gripper is open."""

        if anchors is None or gripper_position is None:
            return
        if not np.all(np.isfinite(anchors)) or not np.isfinite(gripper_position):
            return
        if float(gripper_position) < _OPEN_JAW_GRIPPER_DEG:
            return
        if anchors.shape != (3, 2):
            return
        span = float(np.linalg.norm(anchors[1] - anchors[2]))
        if not np.isfinite(span) or span < 20.0:
            return
        recorded = getattr(self, "_open_jaw_tip_span_px", None)
        if recorded is None:
            self._open_jaw_tip_span_px = {}
            recorded = self._open_jaw_tip_span_px
        previous = recorded.get(view)
        if previous is None or span > float(previous):
            recorded[view] = span

    def _native_support_spans(
        self, view: str
    ) -> tuple[float | None, float | None]:
        """Open/empty spans from this view only. Never the other camera.

        Pulse support is preferred. If this view has no pulse open span, a
        recent same-view open-jaw observation may supply one. A missing empty
        span then uses the collapsed-tip floor — a view-independent pixel
        constant, not the other camera's pulse. If this view still has no
        open span, progress is skipped for it.
        """

        open_span, empty_span = self._pulse_support_spans(view)
        if open_span is None:
            observed = getattr(self, "_open_jaw_tip_span_px", None) or {}
            derived = observed.get(view)
            if derived is not None and np.isfinite(derived):
                open_span = float(derived)
        if open_span is not None and empty_span is None:
            empty_span = float(JAW_COLLAPSED_TIP_SEPARATION_PX)
        return open_span, empty_span

    def _support_spans_for_confirmation(
        self, view: str
    ) -> tuple[float | None, float | None]:
        return self._native_support_spans(view)

    def _empty_closed_for_confirmation(self) -> float | None:
        support = getattr(self, "live_anchor_support", None) or {}
        if not isinstance(support, Mapping):
            return None
        for view in ("column", "wrist"):
            entry = support.get(view)
            if isinstance(entry, Mapping):
                value = entry.get("empty_closed_gripper_position")
                if value is not None:
                    return float(value)
        return None

    def set_target_reference(
        self,
        role: str,
        view: str,
        frame: np.ndarray,
        centroid_px: np.ndarray,
        *,
        radius_px: float,
    ) -> dict[str, Any]:
        if role not in {"pickup", "drop_zone"} or view not in TARGET_VIEWS:
            raise ValueError("unsupported target role or view")
        result = self.target_tracker.register(
            role,
            view,
            frame,
            centroid_px,
            radius_px=radius_px,
        )
        beliefs = self._object_beliefs_map()
        beliefs[(role, view)] = seed_object_belief(role, view, centroid_px)
        if role == "pickup" and view == "column":
            beliefs.pop(("pickup", "wrist"), None)
        if role == "drop_zone":
            self._observer_drop_zone = None
        return result

    def clear_target_reference(self, role: str, view: str | None = None) -> None:
        self.target_tracker.clear(role, view)
        beliefs = self._object_beliefs_map()
        views = (view,) if view is not None else ("column", "wrist")
        for candidate in views:
            beliefs.pop((role, candidate), None)
        if role == "drop_zone":
            self._observer_drop_zone = None

    def _object_beliefs_map(self) -> dict[tuple[str, str], ObjectBelief]:
        beliefs = getattr(self, "_object_beliefs", None)
        if beliefs is None:
            self._object_beliefs = {}
            beliefs = self._object_beliefs
        return beliefs

    def _ensure_object_belief(self, role: str, view: str) -> ObjectBelief | None:
        return self._object_beliefs_map().get((role, view))

    def _visual_query(self, role: str) -> str | None:
        phrases = self._task_visual_queries().get(role) or ()
        return phrases[0] if phrases else None

    def _task_visual_queries(self) -> dict[str, tuple[str, ...]]:
        task_class = getattr(self.config, "task_class", None)
        if task_class is None:
            return {}
        from v10.task_class import task_visual_queries

        return {
            str(role): tuple(phrases)
            for role, phrases in task_visual_queries(task_class).items()
            if phrases
        }

    def _planning_content_rect(self) -> tuple[int, int, int, int] | None:
        geometry = getattr(self, "planning_geometry", None)
        if geometry is None:
            return None
        rect = getattr(geometry, "content_rect", None)
        if rect is None:
            return None
        return tuple(int(value) for value in rect)

    def _session_appearance_ready(self, role: str) -> bool:
        tracker = self.target_tracker
        ready = getattr(tracker, "session_appearance_ready", None)
        if callable(ready):
            return bool(ready(role))
        source = getattr(tracker, "frozen_prototype_source", {}).get(role)
        return bool(source) and str(source).endswith("_live")

    def _lock_column_roles(
        self,
        column_frame: np.ndarray,
        *,
        gripper_center_px: np.ndarray | None = None,
    ) -> dict[str, dict[str, Any]]:
        """One open-vocab forward on the far view for every task role."""

        self._column_role_measurements = {}
        queries = self._task_visual_queries()
        locate_roles = getattr(
            getattr(self, "target_tracker", None),
            "locate_roles_by_open_vocab",
            None,
        )
        if not queries or not callable(locate_roles):
            return {}
        hits = locate_roles(
            "column",
            column_frame,
            queries,
            content_rect=self._planning_content_rect(),
            gripper_center_px=gripper_center_px,
        )
        self._column_role_measurements = {
            str(role): dict(measurement) for role, measurement in hits.items()
        }
        return self._column_role_measurements

    def _appearance_radius_px(self, measurement: dict[str, Any]) -> float:
        bbox = measurement.get("bbox_xyxy")
        if bbox is not None:
            box = np.asarray(bbox, dtype=np.float32).reshape(-1)
            if box.shape[0] == 4:
                return max(8.0, 0.5 * float(max(box[2] - box[0], box[3] - box[1])))
        width = measurement.get("width_px")
        if width is not None and np.isfinite(float(width)) and float(width) > 0.0:
            return max(8.0, 0.5 * float(width))
        return 16.0

    def _bind_session_appearance(
        self,
        role: str,
        view: str,
        frame: np.ndarray,
        live: dict[str, Any],
    ) -> None:
        """Bind this episode's DINOv2 description from a just-accepted lock."""

        if str(live.get("source")) not in {"open_vocab", "frozen_vit"}:
            return
        if bool(live.get("identity_held", False)):
            return
        if not bool(live.get("visible", False)):
            return
        centroid = live.get("centroid_px")
        if centroid is None:
            return
        refresh = getattr(self.target_tracker, "refresh_frozen_prototype", None)
        if not callable(refresh):
            return
        refresh(
            role,
            view,
            frame,
            np.asarray(centroid, dtype=np.float32),
            self._appearance_radius_px(live),
        )

    def _measure_object(
        self,
        role: str,
        view: str,
        frame: np.ndarray,
        *,
        gripper_center_px: np.ndarray | None,
    ) -> dict[str, Any]:
        """Column: exclusive open-vocab lock. Other views: this episode's appearance."""

        if view != "column" and self._session_appearance_ready(role):
            belief = self._ensure_object_belief(role, view)
            hop = float(self.config.pickup_identity_hop_px)
            anchor = search_anchor_px(belief)
            located = self.target_tracker.locate_by_frozen_features(
                role,
                view,
                frame,
                gripper_center_px=gripper_center_px,
                search_centre_px=anchor,
                search_radius_px=None if anchor is None else hop,
            )
            # First commit in a near view needs a clear match. A 0.35 peak on
            # table grain is not the object that was locked in the far view.
            if belief is None:
                peak = located.get("score")
                try:
                    peak_value = float(peak)
                except (TypeError, ValueError):
                    peak_value = -1.0
                if not bool(located.get("visible", False)) or peak_value < 0.48:
                    return {
                        **located,
                        "visible": False,
                        "centroid_px": None,
                        "source": located.get("source") or "frozen_vit_features",
                        "reason": "peak_below_seed_threshold",
                    }
            return located

        if view == "column":
            cached = getattr(self, "_column_role_measurements", {}).get(role)
            if cached is not None:
                measurement = dict(cached)
                if (
                    gripper_center_px is not None
                    and measurement.get("centroid_px") is not None
                    and measurement.get("relative_centroid_px") is None
                ):
                    measurement["gripper_center_px"] = np.asarray(
                        gripper_center_px, dtype=np.float32
                    )
                    measurement["relative_centroid_px"] = (
                        np.asarray(measurement["centroid_px"], dtype=np.float32)
                        - measurement["gripper_center_px"]
                    )
                return measurement

        query = self._visual_query(role)
        if query is None:
            return {
                "visible": False,
                "configured": False,
                "role": role,
                "view": view,
                "centroid_px": None,
                "source": "open_vocab",
                "reason": "target_query_not_configured",
            }
        belief = self._ensure_object_belief(role, view)
        hop = float(self.config.pickup_identity_hop_px)
        anchor = search_anchor_px(belief)
        return self.target_tracker.locate_by_open_vocab(
            role,
            view,
            frame,
            query=query,
            gripper_center_px=gripper_center_px,
            search_centre_px=anchor,
            search_radius_px=None if anchor is None else hop,
        )

    def _recover_far_column_measurement(
        self,
        role: str,
        view: str,
        frame: np.ndarray,
        measurement: dict[str, Any],
        *,
        gripper_center_px: np.ndarray | None,
    ) -> dict[str, Any]:
        """Re-measure a rejected column peak on the current frame.

        Open-vocabulary role locking is useful for finding an object globally,
        but a new global peak can jump from the registered cardboard box to the
        white robot body.  A persistent belief used to reject that jump and
        silently keep the old pixel.  That preserved identity, but it was not a
        fresh observation after motion.

        Once a role has a live frozen-feature prototype, use that prototype to
        search the *current* frame near the committed identity whenever the
        global measurement is missing or too far away.  The old belief supplies
        only the search centre; the returned pixel still comes from current RGB.
        """

        if view != "column":
            return measurement
        belief = self._ensure_object_belief(role, view)
        if belief is None:
            return measurement
        measured = measurement.get("centroid_px")
        measured_px: np.ndarray | None = None
        if bool(measurement.get("visible", False)) and measured is not None:
            candidate = np.asarray(measured, dtype=np.float32).reshape(-1)
            if candidate.shape == (2,) and np.all(np.isfinite(candidate)):
                measured_px = candidate
        anchor = np.asarray(belief.centroid_px, dtype=np.float32)
        hop = float(self.config.pickup_identity_hop_px)
        if (
            measured_px is not None
            and float(np.linalg.norm(measured_px - anchor)) <= hop
        ):
            return measurement
        ready = getattr(self.target_tracker, "session_appearance_ready", None)
        if not callable(ready) or not bool(ready(role)):
            return measurement
        locate = getattr(self.target_tracker, "locate_by_frozen_features", None)
        if not callable(locate):
            return measurement
        recovered = locate(
            role,
            view,
            frame,
            gripper_center_px=gripper_center_px,
            search_centre_px=anchor,
            search_radius_px=hop,
        )
        recovered_px = recovered.get("centroid_px")
        if not bool(recovered.get("visible", False)) or recovered_px is None:
            return measurement
        point = np.asarray(recovered_px, dtype=np.float32).reshape(-1)
        if (
            point.shape != (2,)
            or not np.all(np.isfinite(point))
            or float(np.linalg.norm(point - anchor)) > hop
        ):
            return measurement
        return {
            **recovered,
            "fresh_measurement": True,
            "recovered_from_source": measurement.get("source"),
            "rejected_global_measurement_px": measured_px,
        }

    def ground_object(
        self,
        role: str,
        view: str,
        frame: np.ndarray,
        *,
        gripper_center_px: np.ndarray | None = None,
        update_track: bool = False,
    ) -> dict[str, Any]:
        """Live object state: FrozenViT measurement folded into persistent identity.

        ``track()`` is shadow-only. A new peak cannot re-register the object.
        """

        measurement = self._measure_object(
            role,
            view,
            frame,
            gripper_center_px=gripper_center_px,
        )
        measurement = self._recover_far_column_measurement(
            role,
            view,
            frame,
            measurement,
            gripper_center_px=gripper_center_px,
        )
        shadow = _template_tracking_removed(role, view)
        if view == "column":
            prototype_bound = self._visual_query(role) is not None
        else:
            prototype_bound = self._session_appearance_ready(role)
        belief, live = update_object_belief(
            self._ensure_object_belief(role, view),
            measurement,
            hop_px=float(self.config.pickup_identity_hop_px),
            prototype_bound=prototype_bound,
        )
        if belief is not None:
            self._object_beliefs_map()[(role, view)] = belief
        if view == "column":
            self._bind_session_appearance(role, view, frame, live)
        if (
            live.get("centroid_px") is not None
            and gripper_center_px is not None
            and live.get("relative_centroid_px") is None
        ):
            live["gripper_center_px"] = np.asarray(gripper_center_px, dtype=np.float32)
            live["relative_centroid_px"] = (
                np.asarray(live["centroid_px"], dtype=np.float32)
                - live["gripper_center_px"]
            )
        live["shadow_appearance"] = {
            "visible": bool(shadow.get("visible", False)),
            "centroid_px": shadow.get("centroid_px"),
            "score": shadow.get("score"),
            "margin": shadow.get("margin"),
            "source": shadow.get("source"),
        }
        return live

    def _ground_column_on_features_if_lost(
        self,
        column_pickup: dict[str, Any],
        column_frame: np.ndarray,
        *,
        gripper_center_px: np.ndarray | None,
    ) -> dict[str, Any]:
        """Fall back to frozen features when correlation loses the object.

        The template matcher searches a small radius around where it last saw
        the object and scores by correlation against a fixed-size crop, so it
        gives up on two ordinary things: the object moving further than the
        radius between looks, and the arm passing over it. On the frame that
        stalled this grasp it reported the pickup not visible in either view
        while the object sat in plain sight, and the frozen matcher put it at
        7.2 px with the peak nearly six background deviations clear.
        """

        if bool(column_pickup.get("visible", False)):
            return column_pickup
        located = self.target_tracker.locate_by_frozen_features(
            "pickup",
            "column",
            column_frame,
            gripper_center_px=gripper_center_px,
        )
        return located if located.get("visible") else column_pickup

    def reset_pickup_identity(self) -> None:
        """Forget which blob is the pickup after a discontinuous task reset."""

        self._pickup_identity = None
        self._pickup_goal_hold = None
        self._pickup_goal_reused = False
        self._drop_goal_hold = None
        self._drop_goal_reused = False
        self._descent_standoff = {}
        self._fcpg_transit = {}
        self._servo_scale_prev = None
        self._object_beliefs = {}

    def _hold_pickup_identity(self, column_pickup: dict[str, Any]) -> dict[str, Any]:
        """Keep one committed pickup blob unless a rival is seen more than once.

        ``locate_by_frozen_features`` searches the whole frame, so a distractor
        that correlates as well as the object wins outright on any single round
        and the next round hands it straight back. Measured 2026-08-15 rounds
        23-35: the centroid alternated (21.6, 131.6) / (137.5, 172.2) /
        (20.8, 126.6) / (130.0, 164.7), 114-126 px apart, at scores 0.524-0.574
        with no second candidate reported to separate them. Both readings cannot
        be the fish, and one round of correlation cannot say which is, so the
        committed one stays and the rival has to survive a second look. The
        initial commit takes the same two looks: a single first-arriver must
        not seize the identity forever while alternating distractors never
        accumulate a corroboration.
        """

        identity = getattr(self, "_pickup_identity", None)
        if not bool(column_pickup.get("visible", False)):
            return column_pickup
        centroid = column_pickup.get("centroid_px")
        if centroid is None:
            return column_pickup
        observed = np.asarray(centroid, dtype=np.float32)
        if observed.shape != (2,) or not np.all(np.isfinite(observed)):
            return column_pickup
        needed = int(self.config.pickup_identity_confirmations)

        if identity is None:
            self._pickup_identity = {
                "centroid_px": None,
                "challenger_px": observed.copy(),
                "challenges": 1,
            }
            identity = self._pickup_identity
            if needed <= 1:
                identity["centroid_px"] = observed.copy()
                identity["challenger_px"] = None
                identity["challenges"] = 0
            return column_pickup

        committed = identity.get("centroid_px")
        if committed is None:
            challenger = identity.get("challenger_px")
            if challenger is not None and (
                float(
                    np.linalg.norm(
                        observed - np.asarray(challenger, dtype=np.float32)
                    )
                )
                <= self.config.pickup_identity_hop_px
            ):
                identity["challenges"] = int(identity["challenges"]) + 1
            else:
                identity["challenges"] = 1
            identity["challenger_px"] = observed.copy()
            if int(identity["challenges"]) >= needed:
                identity["centroid_px"] = observed.copy()
                identity["challenger_px"] = None
                identity["challenges"] = 0
            return column_pickup

        committed_px = np.asarray(committed, dtype=np.float32)
        hop_px = float(np.linalg.norm(observed - committed_px))
        if hop_px <= self.config.pickup_identity_hop_px:
            identity["centroid_px"] = observed.copy()
            identity["challenger_px"] = None
            identity["challenges"] = 0
            return column_pickup

        challenger = identity.get("challenger_px")
        if challenger is not None and (
            float(np.linalg.norm(observed - np.asarray(challenger, dtype=np.float32)))
            <= self.config.pickup_identity_hop_px
        ):
            identity["challenges"] = int(identity["challenges"]) + 1
        else:
            identity["challenges"] = 1
        identity["challenger_px"] = observed.copy()
        if identity["challenges"] >= needed:
            identity["centroid_px"] = observed.copy()
            identity["challenger_px"] = None
            identity["challenges"] = 0
            return {
                **column_pickup,
                "identity_hop_px": hop_px,
                "identity_held": False,
            }
        # Fail closed on the committed blob: the servo keeps aiming where it
        # already was rather than taking a one-round hop it cannot corroborate.
        return {
            **column_pickup,
            "centroid_px": (float(committed_px[0]), float(committed_px[1])),
            "relative_centroid_px": None,
            "identity_hop_px": hop_px,
            "identity_held": True,
            "identity_challenges": int(identity["challenges"]),
        }

    def _held_pickup_goal_xyz(
        self,
        target_px: np.ndarray,
        current_gripper_xyz: np.ndarray,
    ) -> np.ndarray | None:
        """The committed absolute servo goal, while re-deriving would only chatter.

        A fresh derivation is warranted when the pixel moved, or when the arm
        has closed enough of the gap that was present at commit.  The period-2
        wobble does neither: progress stays near zero, so it never earns a
        re-derivation.  The previous release
        ``remaining <= derived_at - max_column_servo_step_m`` was unsatisfiable
        once ``derived_at`` fell below the step (terminal approach is 18-24mm
        against a 25-40mm step), which froze the overshot goal permanently.
        """

        hold = getattr(self, "_pickup_goal_hold", None)
        if hold is None:
            return None
        pixel_drift = float(
            np.linalg.norm(
                np.asarray(target_px, dtype=np.float32)
                - np.asarray(hold["target_px"], dtype=np.float32)
            )
        )
        if pixel_drift > self.config.servo_goal_hold_tolerance_px:
            return None
        if int(hold.get("rounds_held", 0)) >= int(
            self.config.servo_goal_hold_max_rounds
        ):
            return None
        goal = np.asarray(hold["relative_target_xyz"], dtype=np.float32)
        remaining = float(np.linalg.norm(current_gripper_xyz[:2] - goal[:2]))
        derived_at = float(
            np.linalg.norm(
                np.asarray(hold["derived_gripper_xyz"], dtype=np.float32)[:2]
                - goal[:2]
            )
        )
        progress = derived_at - remaining
        release_after = min(
            float(self.config.max_column_servo_step_m),
            0.5 * derived_at if derived_at > 0.0 else 0.0,
        )
        if progress >= release_after:
            return None
        return goal

    def _commit_pickup_goal(
        self,
        target_px: np.ndarray,
        relative_target_xyz: np.ndarray,
        current_gripper_xyz: np.ndarray,
        *,
        held: bool,
    ) -> None:
        self._pickup_goal_reused = bool(held)
        if held:
            hold = getattr(self, "_pickup_goal_hold", None)
            if hold is not None:
                hold["rounds_held"] = int(hold.get("rounds_held", 0)) + 1
            return
        self._pickup_goal_hold = {
            "target_px": np.asarray(target_px, dtype=np.float32).copy(),
            "relative_target_xyz": np.asarray(
                relative_target_xyz, dtype=np.float32
            ).copy(),
            "derived_gripper_xyz": np.asarray(
                current_gripper_xyz, dtype=np.float32
            ).copy(),
            "rounds_held": 0,
        }

    def _held_drop_goal_xyz(
        self,
        target_px: np.ndarray,
        current_gripper_xyz: np.ndarray,
    ) -> np.ndarray | None:
        """Reuse the committed drop goal so TRANSPORT does not re-derive XYZ."""

        hold = getattr(self, "_drop_goal_hold", None)
        if hold is None:
            return None
        pixel_drift = float(
            np.linalg.norm(
                np.asarray(target_px, dtype=np.float32)
                - np.asarray(hold["target_px"], dtype=np.float32)
            )
        )
        if pixel_drift > self.config.servo_goal_hold_tolerance_px:
            return None
        if int(hold.get("rounds_held", 0)) >= int(
            self.config.servo_goal_hold_max_rounds
        ):
            return None
        goal = np.asarray(hold["relative_target_xyz"], dtype=np.float32)
        remaining = float(np.linalg.norm(current_gripper_xyz[:2] - goal[:2]))
        derived_at = float(
            np.linalg.norm(
                np.asarray(hold["derived_gripper_xyz"], dtype=np.float32)[:2]
                - goal[:2]
            )
        )
        progress = derived_at - remaining
        release_after = min(
            float(self.config.max_column_servo_step_m),
            0.5 * derived_at if derived_at > 0.0 else 0.0,
        )
        if progress >= release_after:
            return None
        return goal

    def _commit_drop_goal(
        self,
        target_px: np.ndarray,
        relative_target_xyz: np.ndarray,
        current_gripper_xyz: np.ndarray,
        *,
        held: bool,
    ) -> None:
        self._drop_goal_reused = bool(held)
        if held:
            hold = getattr(self, "_drop_goal_hold", None)
            if hold is not None:
                hold["rounds_held"] = int(hold.get("rounds_held", 0)) + 1
            return
        self._drop_goal_hold = {
            "target_px": np.asarray(target_px, dtype=np.float32).copy(),
            "relative_target_xyz": np.asarray(
                relative_target_xyz, dtype=np.float32
            ).copy(),
            "derived_gripper_xyz": np.asarray(
                current_gripper_xyz, dtype=np.float32
            ).copy(),
            "rounds_held": 0,
        }

    def _live_action_scale(self) -> float:
        return float(getattr(self.config, "live_action_scale", 1.0))

    def _live_column_servo_gain(self) -> float:
        scale = float(getattr(self.config, "live_action_scale", 1.0))
        return min(
            1.0,
            float(getattr(self.config, "column_servo_gain", 0.25)) * scale,
        )

    def _live_image_servo_variants(self, variants):
        scale = float(getattr(self.config, "live_action_scale", 1.0))
        return scale_image_servo_variants(
            variants,
            scale,
            max_joint_step=float(self.config.max_prefix_step_jump),
        )

    def _live_grasp_jaw_rates(self) -> tuple[float, ...]:
        scale = float(getattr(self.config, "live_action_scale", 1.0))
        cap = float(self.config.max_prefix_step_jump)
        return tuple(
            float(np.clip(rate * scale, -cap, cap)) for rate in GRASP_JAW_RATES
        )

    def _measure_servo_scale(
        self,
        *,
        gripper_anchor_px: np.ndarray,
        current_gripper_xyz: np.ndarray,
        planning_target_xyz: np.ndarray | None,
        live_transform_applied: bool | None,
    ) -> dict[str, Any]:
        """Archive commanded Cartesian motion vs realized gripper-anchor pixels.

        True mm/px is polluted while the live-anchor transform flips; after
        hysteresis the same fields become a usable on-robot scale measurement.
        """

        prev = getattr(self, "_servo_scale_prev", None)
        realized_px: float | None = None
        prior_commanded_m: float | None = None
        mm_per_px: float | None = None
        if prev is not None:
            realized_px = float(
                np.linalg.norm(
                    np.asarray(gripper_anchor_px, dtype=np.float32)
                    - np.asarray(prev["gripper_anchor_px"], dtype=np.float32)
                )
            )
            prior_commanded_m = float(prev["commanded_displacement_m"])
            if realized_px > 1.0e-3:
                mm_per_px = 1000.0 * prior_commanded_m / realized_px
        commanded_m: float | None = None
        if planning_target_xyz is not None:
            commanded_m = float(
                np.linalg.norm(
                    np.asarray(planning_target_xyz, dtype=np.float32)[:2]
                    - np.asarray(current_gripper_xyz, dtype=np.float32)[:2]
                )
            )
            self._servo_scale_prev = {
                "gripper_anchor_px": np.asarray(
                    gripper_anchor_px, dtype=np.float32
                ).copy(),
                "commanded_displacement_m": commanded_m,
                "live_anchor_transform_applied": live_transform_applied,
            }
        else:
            self._servo_scale_prev = None
        return {
            "prior_commanded_cartesian_displacement_m": prior_commanded_m,
            "realized_gripper_anchor_displacement_px": realized_px,
            "mm_per_px": mm_per_px,
            "commanded_cartesian_displacement_m": commanded_m,
            "prior_live_anchor_transform_applied": (
                None if prev is None else prev.get("live_anchor_transform_applied")
            ),
            "live_anchor_transform_applied": live_transform_applied,
        }

    def _hand_pickup_description_to_wrist(
        self, column_pickup: dict[str, Any], column_frame: np.ndarray
    ) -> None:
        """Re-describe the pickup from the column while the column can still see it.

        The approach runs the object from far and small to near and large, and
        ends with the arm over it and the column blind. The wrist inherits
        whatever description was last taken, so it is taken here, while the
        column can still see the object. By the time the column loses it the
        wrist is holding a description recorded from a few centimetres away
        rather than the one the operator drew at the start of the session.

        The bar is deliberately higher than the tracker's own margin. Reporting
        a detection and overwriting the identity that will be carried through
        the occlusion are different decisions: refreshing at the tracker's
        0.015 put the wrist within 20 px of the object in 27% of the 22
        column-lost rounds of one replayed session, because the last refresh
        before the occlusion is taken while the arm is already entering the
        column view. Requiring 0.03 left 91% of those rounds on the object.
        This is an operating criterion read off one session, not a calibrated
        threshold; 22 rounds cannot support a tighter claim.
        """

        self._bind_session_appearance(
            "pickup", "column", column_frame, column_pickup
        )

    def _session_ground_wrist_jaw(
        self,
        wrist_pickup: dict[str, Any],
        column_pickup: dict[str, Any],
        *,
        wrist_frame: np.ndarray,
        gripper_center_px: np.ndarray,
        wrist_valid: bool,
        planar_error_m: float | None,
    ) -> dict[str, Any]:
        """Ground the wrist on the object itself, never on the jaws' own pose.

        This used to crop a session template at the jaw centre as soon as the
        column put the pickup within 0.13 m, on the assumption that being that
        close means the object is between the jaws. The assumption was never
        checked, and while the contact height was pinned 3.1 cm high it was
        false: the crop taken was the table under the fish. The fallback skipped
        the crop entirely and asserted the pickup sat at the jaw centre with
        score 1.0 and zero offset, which the closure objective then read as a
        perfectly centred grasp -- centred by construction, not by measurement.

        The wrist is worth grounding properly because it is the view that
        survives grasp range: the arm ends up directly over the object and
        occludes it in the column, exactly when the grasp needs to be judged.
        What crosses between the cameras is neither the template nor the
        colour. Not the template, because correlation is fixed-scale and the
        object spans ~38 px in the column against ~150 px in the wrist. Not the
        colour, because a chroma model is a description of the sensor as much
        as of the object: handed a greyscale copy of the same frame it does not
        degrade, it refuses, so the whole grasp rests on the camera staying
        colour and the scene staying neutral around it.

        What crosses is a vector of frozen ViT features. Measured on a live
        pair with the fish half-buried under the jaws: 6.2 px handing the
        description from the wrist to the occluded column, 23.1 px doing the
        same on greyscale frames, where the colour detector returns
        `colour_not_found` instead of a position.
        """

        del column_pickup, wrist_valid, planar_error_m
        return self.ground_object(
            "pickup",
            "wrist",
            wrist_frame,
            gripper_center_px=gripper_center_px,
        )

    def target_status(self) -> dict[str, Any]:
        return self._target_profile_status()

    def _target_profile_status(self) -> dict[str, Any]:
        status = self.target_tracker.status()
        context = self.context_tracker.status()
        registration_current: bool | None = None
        registration_error: str | None = None
        if bool(context.get("peg_ready", False)):
            try:
                self._validate_context_registration()
                registration_current = True
            except ContextRegistrationError as exc:
                registration_current = False
                registration_error = str(exc)
        status.update(
            {
                "peg_ready": bool(context.get("peg_ready", False)),
                "peg_session_only": True,
                "peg_clearance_m": self.config.peg_clearance_m,
                "peg_registration": self.context_registration,
                "peg_registration_current": registration_current,
                "peg_registration_error": registration_error,
            }
        )
        return status

    def _validate_context_registration(self) -> None:
        """Reject a peg reference whose local calibration contract changed."""

        registration = self.context_registration
        if not isinstance(registration, Mapping):
            raise ContextRegistrationError("peg registration metadata is missing")
        if self.config.require_live_column_calibration:
            if self.live_anchor_transforms_px.get("column") is None:
                raise ContextRegistrationError(
                    "peg registration requires column live calibration"
                )
            current_live_checksum = self.model_contract.get(
                "live_anchor_calibration_sha256"
            )
            if not isinstance(current_live_checksum, str) or not current_live_checksum:
                raise ContextRegistrationError(
                    "peg registration live_anchor_calibration_sha256 is missing"
                )
        for key in (
            "projection_calibration_sha256",
            "live_anchor_calibration_sha256",
        ):
            registered = registration.get(key)
            current = self.model_contract.get(key)
            if registered != current:
                raise ContextRegistrationError(
                    f"peg registration {key} no longer matches the current calibration"
                )
        registered_clearance = registration.get("clearance_m")
        if (
            not isinstance(registered_clearance, (int, float))
            or not np.isfinite(registered_clearance)
            or float(registered_clearance) != float(self.config.peg_clearance_m)
        ):
            raise ContextRegistrationError(
                "peg registration clearance_m no longer matches the planner contract"
            )

    def set_context_reference(
        self,
        frame: np.ndarray,
        centroid_px: np.ndarray,
        *,
        radius_px: float,
        registration: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Register a session-only peg from one frozen column planning frame."""

        if not isinstance(registration, Mapping):
            raise TypeError("context registration metadata must be a mapping")
        required = {"round_id", "camera_binding"}
        if not required.issubset(registration):
            raise ValueError("context registration lacks round or camera binding")
        if self.config.require_live_column_calibration:
            if self.live_anchor_transforms_px.get("column") is None:
                raise ContextRegistrationError(
                    "column live calibration is required for peg context"
                )
            live_checksum = self.model_contract.get(
                "live_anchor_calibration_sha256"
            )
            if not isinstance(live_checksum, str) or not live_checksum:
                raise ContextRegistrationError(
                    "peg context requires a recorded live calibration checksum"
                )
        result = self.context_tracker.register(
            "peg",
            "column",
            frame,
            centroid_px,
            radius_px=radius_px,
        )
        self.context_registration = {
            **dict(registration),
            "projection_calibration": str(
                self.config.calibration.expanduser().resolve()
            ),
            "projection_calibration_sha256": self.model_contract[
                "projection_calibration_sha256"
            ],
            "live_anchor_calibration_source": self.model_contract.get(
                "live_anchor_calibration_source"
            ),
            "live_anchor_calibration_sha256": self.model_contract.get(
                "live_anchor_calibration_sha256"
            ),
            "clearance_m": self.config.peg_clearance_m,
        }
        self._validate_context_registration()
        return {
            **result,
            "registration": self.context_registration,
        }

    def clear_context_reference(self) -> None:
        self.context_tracker.clear("peg", "column")
        self.context_registration = None

    def detect_live_context(
        self,
        column_frame: np.ndarray,
        *,
        update_track: bool = False,
    ) -> dict[str, Any]:
        """Track and backproject the configured peg on a fresh column frame."""

        descriptor = _template_tracking_removed("peg", "column")
        if not descriptor.get("configured", False):
            return {
                **descriptor,
                "registration": self.context_registration,
                "clearance_m": self.config.peg_clearance_m,
                "registration_current": None,
                "registration_error": None,
            }
        self._validate_context_registration()
        if not descriptor.get("visible", False):
            reason = descriptor.get("reason", "not reliably visible")
            raise RuntimeError(f"current peg is not reliably visible: {reason}")
        column_transform = self.live_anchor_transforms_px.get("column")
        if self.config.require_live_column_calibration and column_transform is None:
            raise RuntimeError("column live calibration is required for peg context")
        base_xyz = column_live_pixel_to_base_xyz(
            self.fits["column"],
            np.asarray(descriptor["centroid_px"], dtype=np.float32),
            z_base=self.contact_height_m,
            live_transform_px=column_transform,
        )
        return {
            **descriptor,
            "base_xyz": base_xyz,
            "task_plane_z_m": self.contact_height_m,
            "registration": self.context_registration,
            "clearance_m": self.config.peg_clearance_m,
            "registration_current": True,
            "registration_error": None,
        }

    def context_clearance(
        self,
        current_state: np.ndarray,
        command_prefixes: np.ndarray,
        context_xy_m: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return command_prefix_context_clearance(
            self.kinematics,
            current_state,
            command_prefixes,
            context_xy_m,
            required_clearance_m=self.config.peg_clearance_m,
        )

    @torch.inference_mode()
    def observe_grounding(
        self,
        frames: dict[str, np.ndarray],
        states: np.ndarray,
        *,
        commit: bool = True,
        current_fingertips: LiveFingertipMeasurement | None = None,
    ) -> GroundingObservation:
        """Encode one dual-view observation with recursive local FCPG updates."""

        state_history = np.asarray(states, dtype=np.float32)
        if state_history.shape != (self.history, 6):
            raise ValueError("states must have shape [3,6]")
        prepared_frames: dict[str, np.ndarray] = {}
        for view in VIEWS:
            values = np.asarray(frames[view], dtype=np.uint8)
            if values.shape != (self.history, 224, 224, 3):
                raise ValueError(f"{view} frames must have shape [3,224,224,3]")
            prepared_frames[view] = values
        v165 = self.v165_runtime_adapter
        v170 = self.v170_runtime_adapter
        v17_pixel_source = (
            None
            if v170 is None
            else V17_PIXEL_FM_SOURCES.get(v170.fcpg.model_family)
        )
        v17_pixel_candidate = bool(
            v170 is not None
            and v170.mode == "candidate"
            and v17_pixel_source is not None
        )
        v174_runtime = bool(
            v170 is not None and v170.fcpg.model_family == "v17.4"
        )
        # ProdV13 remains observable in V16.5 sessions, but it is never handed
        # to the anchor-control branch. Its trajectory is attached later by the
        # runtime as a shadow receipt only.
        control_anchors = (
            None
            if current_fingertips is None or v165 is not None or v170 is not None
            else live_fingertip_control_anchors(
                current_fingertips, self.planning_geometry
            )
        )

        current_state = state_history[-1]
        normalized_states = (state_history - self.state_center) / self.state_scale
        anchors_xyz, nominal, nominal_valid = project_history(
            state_history,
            self.fits,
            self.kinematics,
            image_size=(224, 224),
        )
        live_transform_scope: dict[str, dict[str, Any]] = {}
        # Which transform each view's `nominal` ACTUALLY carries this round.
        # Anything that later builds a correction as `corrected - nominal` has to
        # project through this and not through the raw transform, or the two
        # disagree by the whole transform.
        self.applied_anchor_transforms_px = {view: None for view in VIEWS}
        self._column_hard_match_residual_px = None
        self._column_tips_are_metal = False
        self._column_gripper_lock: dict[str, Any] | None = None
        self._lock_column_roles(prepared_frames["column"][-1])
        column_raw_nominal = nominal["column"]
        for view, transform in self.live_anchor_transforms_px.items():
            raw_nominal = nominal[view]
            support = self.live_anchor_support.get(view)
            applied = True
            distance = 0.0
            if support is not None:
                distance = float(
                    np.linalg.norm(anchors_xyz[-1, 0] - support["center_xyz"])
                )
                applied = live_anchor_transform_applied(
                    distance,
                    float(support["radius_m"]),
                    previously_applied=bool(
                        getattr(self, "_live_anchor_transform_applied", {}).get(
                            view, False
                        )
                    ),
                    hysteresis_band=float(
                        getattr(
                            self.config,
                            "live_anchor_transform_hysteresis_band",
                            0.35,
                        )
                    ),
                )
            if not hasattr(self, "_live_anchor_transform_applied"):
                self._live_anchor_transform_applied = {}
            self._live_anchor_transform_applied[view] = bool(applied)
            live_transform_scope[view] = {
                "applied": applied,
                "distance_m": distance,
                "radius_m": None if support is None else support["radius_m"],
                "residual_rms_px": (
                    None if support is None else support.get("residual_rms_px")
                ),
                "hysteresis_band": float(
                    getattr(
                        self.config,
                        "live_anchor_transform_hysteresis_band",
                        0.35,
                    )
                ),
            }
            if not applied:
                # Outside the neighbourhood the pulse observed, this correction
                # is pure extrapolation from a zero-residual two-point fit, and
                # it degrades the projection instead of repairing it.  Leaving
                # both `nominal` and its validity untouched is exactly the
                # never-calibrated path, which is the intended fallback.
                continue
            nominal[view] = apply_live_similarity_transform(
                raw_nominal, transform
            )
            self.applied_anchor_transforms_px[view] = transform
            # The projection fit's image bounds belong to the ORIGINAL
            # calibration session's camera aim. Once a live transform exists
            # the camera has been re-aimed, and a fingertip a few pixels past
            # the stale frame edge can be fully visible live (the affine fit
            # extrapolates linearly). Validity is therefore finiteness plus
            # the LIVE-frame bounds; the seed and corrected checks downstream
            # already operate in the live frame.
            # What is unique to this branch is the `raw_nominal` finiteness
            # semantics, and that is all it asserts.  Content narrowing is
            # deliberately NOT repeated here: it is the same pixel and the same
            # predicate as the unconditional pass below, so a second copy would
            # be redundant -- and redundancy is actively harmful here, because
            # it would mask a regression in the one pass that every view,
            # transformed or not, depends on.
            nominal_valid[view] = np.all(np.isfinite(raw_nominal), axis=-1)
        column_extrinsic_diagnostics: dict[str, Any] = {}
        block_column_out_of_support = False
        column_scope = live_transform_scope.get("column")
        pulse_applied = bool(column_scope and column_scope.get("applied"))
        pulse_transform = self.live_anchor_transforms_px.get("column")
        # Pulse is the one-shot rigid restore of FCPG support. Out of that
        # ball the corrector must degrade, not borrow metal / OWL / SE(3).
        if pulse_transform is not None and not pulse_applied:
            block_column_out_of_support = True
        observer = getattr(self, "_column_extrinsic_observer", None)
        observer_enabled = bool(
            getattr(self.config, "column_extrinsic_observer", True)
            and observer is not None
        )
        shadow_measured_px: np.ndarray | None = None
        if observer_enabled:
            refinement = refine_column_live_nominal(
                column_raw_nominal,
                prepared_frames["column"][-1],
                observer,
                pulse_applied=pulse_applied,
                pulse_transform_px=pulse_transform,
                pulse_residual_rms_px=(
                    None if column_scope is None else column_scope.get("residual_rms_px")
                ),
                max_offset_px=float(
                    getattr(self.config, "column_extrinsic_max_offset_px", 60.0)
                ),
                max_residual_px=float(
                    getattr(self.config, "column_extrinsic_max_residual_px", 8.0)
                ),
            )
            column_extrinsic_diagnostics = {
                **refinement.diagnostics,
                "control_authority": "fcpg",
            }
            measured = (refinement.diagnostics or {}).get("measurement") or {}
            metal_points = measured.get("metal_points_px")
            if metal_points is not None:
                shadow_measured_px = np.asarray(metal_points, dtype=np.float32)
            elif pulse_applied:
                late = measure_column_fk_translation(
                    prepared_frames["column"][-1],
                    nominal["column"][-1],
                    max_offset_px=float(
                        getattr(
                            self.config,
                            "column_extrinsic_max_offset_px",
                            60.0,
                        )
                    ),
                    max_residual_px=float(
                        getattr(
                            self.config,
                            "column_extrinsic_max_residual_px",
                            8.0,
                        )
                    ),
                )
                if late is not None:
                    shadow_measured_px = late.metal_points_px
        grip = (getattr(self, "_column_role_measurements", {}) or {}).get("gripper") or {}
        if bool(grip.get("visible")):
            # Exclusive object role only. Not tip authority.
            self._column_gripper_lock = {
                "visible": True,
                "source": "open_vocab",
                "query": grip.get("query"),
                "centroid_px": grip.get("centroid_px"),
                "bbox_xyxy": grip.get("bbox_xyxy"),
                "score": grip.get("score"),
            }
        if shadow_enabled(self):
            from fcpg.observer_bundle import ARM_JOINT_COUNT

            anchors = np.asarray(anchors_xyz[-1], dtype=np.float64)
            jacobian = np.zeros((len(anchors), 3, ARM_JOINT_COUNT), dtype=np.float64)
            step = 0.5
            for joint in range(ARM_JOINT_COUNT):
                delta = np.zeros(6, dtype=np.float64)
                delta[joint] = step
                plus = self.kinematics.forward_anchors(
                    np.clip(current_state + delta, -100.0, 100.0)
                )
                minus = self.kinematics.forward_anchors(
                    np.clip(current_state - delta, -100.0, 100.0)
                )
                jacobian[:, :, joint] = (plus - minus) / (2.0 * step)
            measured = shadow_measured_px
            if pulse_applied and pulse_transform is not None:
                # Cold-start association needs the exact pulse translation.
                # The 2-D observer deliberately Kalman-averages camera jumps;
                # feeding that lagged value into the 10 px hard gate can leave
                # a freshly measured tip 15-25 px outside the search window.
                # This offset is association-only in ``shadow_step``; the 3-D
                # observer still updates against raw pixels and its chi-square
                # gate remains authoritative.
                kf_offset = np.asarray(pulse_transform, dtype=np.float64)[:, 2]
            else:
                kf_offset = (
                    None
                    if observer is None
                    else getattr(observer, "translation_px", None)
                )
            phase_obj = getattr(getattr(self, "phase_tracker", None), "phase", None)
            phase_name = (
                None
                if phase_obj is None
                else getattr(phase_obj, "value", str(phase_obj))
            )
            self._observer_last_diagnostics = shadow_step(
                self,
                round_id=str(getattr(self, "_last_round_id", "")),
                phase=phase_name,
                joints=current_state,
                anchors_base_xyz=anchors,
                joint_jacobian=jacobian,
                measured_px=measured,
                measured_valid=None if measured is None else np.ones(3, dtype=bool),
                affine_tips_px=column_raw_nominal,
                kf2d_offset_px=kf_offset,
                frame_rgb=prepared_frames["column"][-1],
            )
            self._observer_last_diagnostics["excitation"] = maybe_excitation(
                self,
                phase=phase_name,
                jaws_holding=str(phase_name or "").lower()
                in {"grasp", "transport", "place", "release"},
                anchors_base_xyz=anchors,
                joint_jacobian=jacobian,
            )
        # THE single content gate for the nominal projection, covering every
        # view -- transformed or not.  The loop above only revisits views that
        # HAVE a live transform; an untransformed view would otherwise keep
        # `project_history`'s full-canvas verdict and report letterbox pad rows
        # as visible.  The fail-open did not live only on the transformed
        # branch.  A no-op under `squash_v0`, where the content rect is the
        # whole canvas.
        for view in VIEWS:
            nominal_valid[view] = np.asarray(
                nominal_valid[view], dtype=bool
            ) & self.planning_geometry.content_mask(nominal[view])

        contexts: dict[str, np.ndarray] = {}
        raw_contexts: dict[str, np.ndarray] = {}
        correction_seed: dict[str, np.ndarray] = {}
        corrected: dict[str, np.ndarray] = {}
        valid: dict[str, np.ndarray] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        anchor_tokens: dict[str, np.ndarray] = {}
        full256_tokens: dict[str, np.ndarray] = {}
        blocked: list[str] = []
        pending: dict[str, _AnchorPosterior] = {}
        safe_to_commit: dict[str, bool] = {}
        posteriors = (
            getattr(self, "_anchor_posteriors", {})
            if self.config.recursive_grounding
            else {}
        )

        for view_index, view in enumerate(VIEWS):
            current_media_nonblack = v165_rgb_view_nonblack(
                prepared_frames[view][-1]
            )
            if control_anchors is not None:
                # Prod-V13 owns current pixel geometry after automatic H0.
                # Do not execute recursive_anchor_seed, the global V12 point
                # locator, or the legacy V12/FCPG pixel corrector.  Frozen V12
                # remains useful as the visual carrier: use its global CLS for
                # every history row and sample anchor tokens only at the
                # current Prod-V13 [midpoint, fixed, moving] measurement.
                camera_context = np.zeros((self.history, 5), dtype=np.float32)
                camera_context[:, view_index] = 1.0
                pixels = (
                    normalize_rgb(prepared_frames[view])
                    .to(self.device)
                    .unsqueeze(0)
                )
                vision = self.system._encode_shared_vision(pixels)
                if self.v170_runtime_adapter is not None:
                    current_full256 = (
                        vision[1][0, -1]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    if current_full256.shape != (256, 192):
                        raise RuntimeError(
                            f"V12.4 native patch grid drifted: {current_full256.shape}"
                        )
                    if not current_media_nonblack:
                        current_full256 = np.zeros_like(current_full256)
                        blocked.append(f"{view}_current_rgb_black")
                    full256_tokens[view] = current_full256
                # A prior legacy call may have populated this mutable sidecar.
                # The Prod-V13 path never consumes a legacy native output.
                self.system.last_fcpg_native = None
                points = np.repeat(
                    control_anchors[view][None], self.history, axis=0
                ).astype(np.float32)
                current_valid = self.planning_geometry.content_mask(
                    control_anchors[view]
                )
                current_valid[0] = current_valid[1] & current_valid[2]
                corrected_valid = np.zeros(
                    (self.history, 3), dtype=bool
                )
                corrected_valid[-1] = current_valid
                bridge = self.system.bridge(
                    vision[0],
                    vision[1],
                    torch.from_numpy(points).to(self.device).unsqueeze(0),
                    torch.from_numpy(points).to(self.device).unsqueeze(0),
                    torch.from_numpy(anchors_xyz).to(self.device).unsqueeze(0),
                    torch.from_numpy(corrected_valid)
                    .to(self.device)
                    .unsqueeze(0),
                    image_size=(
                        int(prepared_frames[view].shape[2]),
                        int(prepared_frames[view].shape[1]),
                    ),
                )
                raw_context = (
                    vision[0][0].detach().cpu().numpy().astype(np.float32)
                )
                bridged = (
                    bridge.context_embedding[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                row_diagnostics: dict[str, Any] = {
                    "prior_source": current_fingertips.source,
                    "recursive_prior_source": "disabled_by_prod_v13",
                    "global_locator_enabled": False,
                    "global_locator_accepted": False,
                    "global_locator_rejection_reasons": (
                        "not_executed_prod_v13_pixel_authority",
                    ),
                    "pixel_control_authority": current_fingertips.source,
                    "pixel_control_generation": int(
                        current_fingertips.generation
                    ),
                    "pixel_control_horizon": int(current_fingertips.horizon),
                    "v12_pixel_anchor_role": (
                        "not_executed_current_geometry_owned_by_prod_v13"
                    ),
                    "legacy_v12_pixel_head_executed": False,
                    "fk_delta_px": None,
                    "posterior_offset_px": None,
                    "historical_anchor_rows_valid": 0,
                    "prod_v13_control_anchors_px": control_anchors[view],
                }
                scope = live_transform_scope.get(view)
                row_diagnostics["live_anchor_transform"] = (
                    {"present": False, "applied": False, "control_authority": False}
                    if scope is None
                    else {
                        "present": True,
                        **scope,
                        "control_authority": False,
                    }
                )
                if view == "column" and column_extrinsic_diagnostics:
                    row_diagnostics["column_extrinsic"] = {
                        **column_extrinsic_diagnostics,
                        "control_authority": "none",
                    }
                if not bool(np.all(corrected_valid[-1])):
                    blocked.append(f"{view}_anchor_out_of_bounds")
                contexts[view] = bridged
                raw_contexts[view] = raw_context
                correction_seed[view] = points.copy()
                corrected[view] = points
                valid[view] = corrected_valid
                diagnostics[view] = row_diagnostics
                anchor_tokens[view] = (
                    bridge.anchor_tokens[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                safe_to_commit[view] = False
                continue

            if v165 is None and v170 is None:
                seed, row_diagnostics = recursive_anchor_seed(
                    nominal[view],
                    current_state,
                    prepared_frames[view][-1],
                    posteriors.get(view),
                )
            else:
                seed = np.asarray(nominal[view], dtype=np.float32).copy()
                row_diagnostics = {
                    "prior_source": "nominal",
                    "recursive_prior_source": (
                        "disabled_by_v17"
                        if v170 is not None
                        else "shadow_disabled_by_v16_5"
                    ),
                    "v165_anchor_authority": (
                        None if v165 is None else v165.anchor_authority
                    ),
                    "prod_v13_shadow_only": current_fingertips is not None,
                }
            # `seed` is its OWN pixel, not `nominal[view]`: recursive_anchor_seed
            # may move it off the nominal projection, so narrowing
            # `nominal_valid` upstream does not cover this position.  It needs
            # its own content test.
            seed_valid = np.asarray(
                nominal_valid[view], dtype=bool
            ) & self.planning_geometry.content_mask(seed)
            camera_context = np.zeros((self.history, 5), dtype=np.float32)
            camera_context[:, view_index] = 1.0
            pixels = normalize_rgb(prepared_frames[view]).to(self.device).unsqueeze(0)
            vision = self.system._encode_shared_vision(pixels)
            needs_native_grid = (
                v165 is not None or self.v170_runtime_adapter is not None
            )
            if needs_native_grid:
                current_full256 = (
                    vision[1][0, -1]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                if current_full256.shape != (256, 192):
                    raise RuntimeError(
                        f"V12.4 native patch grid drifted: {current_full256.shape}"
                    )
                if not current_media_nonblack:
                    current_full256 = np.zeros_like(current_full256)
                    seed_valid = np.zeros_like(seed_valid, dtype=bool)
                    blocked.append(f"{view}_current_rgb_black")
                full256_tokens[view] = current_full256
                row_diagnostics["current_media_nonblack"] = bool(
                    current_media_nonblack
                )
            if v165 is not None:
                # Run the optional global locator on the same feature map for
                # diagnostics, but discard both its candidate seed and validity.
                try:
                    global_shadow_seed, global_shadow = self._global_tip_seed(
                        view,
                        vision[2],
                        camera_context,
                        nominal[view],
                        seed,
                    )
                except Exception as exc:
                    global_shadow_seed = seed.copy()
                    global_shadow = {
                        "global_locator_enabled": self.global_tip_head is not None,
                        "global_locator_accepted": False,
                        "global_locator_rejection_reasons": ("shadow_error",),
                        "shadow_error": f"{type(exc).__name__}: {exc}",
                    }
                global_diagnostics = disabled_global_tip_diagnostics()
                global_diagnostics["global_locator_shadow"] = {
                    **global_shadow,
                    "candidate_seed_px": np.asarray(
                        global_shadow_seed, dtype=np.float32
                    ),
                    "control_influence": False,
                }
            elif v170 is not None:
                if v174_runtime:
                    # The V17.4 observer is local around its nominal.  The live
                    # column FK projection can be outside its one-pulse support,
                    # so use the existing full-image measurement only as the
                    # nominal query.  V17.4 remains the sole final F/M output;
                    # this seed is neither returned directly nor a fallback
                    # replacement for its posterior.
                    seed, global_diagnostics = self._global_tip_seed(
                        view,
                        vision[2],
                        camera_context,
                        nominal[view],
                        seed,
                    )
                    global_diagnostics["v174_nominal_source"] = (
                        "global_tip_heatmap"
                        if global_diagnostics["global_locator_accepted"]
                        else "analytic_nominal"
                    )
                    global_diagnostics["v174_final_pixel_authority"] = (
                        "v17_4_residual_belief_fm"
                    )
                else:
                    # V17.0/V17.1 retain their frozen deployment contract.
                    global_diagnostics = disabled_global_tip_diagnostics()
                    seed = np.asarray(nominal[view], dtype=np.float32).copy()
                seed_valid = np.asarray(
                    nominal_valid[view], dtype=bool
                ) & self.planning_geometry.content_mask(seed)
            elif self._uses_coarse_fine_fusion():
                global_diagnostics = disabled_global_tip_diagnostics()
                # Next-round seed is always the current nominal. Recursive
                # posteriors and global-tip offsets must not enter the corrector.
                seed = np.asarray(nominal[view], dtype=np.float32).copy()
                seed_valid = np.asarray(
                    nominal_valid[view], dtype=bool
                ) & self.planning_geometry.content_mask(seed)
            else:
                seed, global_diagnostics = self._global_tip_seed(
                    view,
                    vision[2],
                    camera_context,
                    nominal[view],
                    seed,
                )
            row_diagnostics["recursive_prior_source"] = row_diagnostics[
                "prior_source"
            ]
            row_diagnostics.update(global_diagnostics)
            scope = live_transform_scope.get(view)
            row_diagnostics["live_anchor_transform"] = (
                {"present": False, "applied": False}
                if scope is None
                else {"present": True, **scope}
            )
            if view == "column" and column_extrinsic_diagnostics:
                row_diagnostics["column_extrinsic"] = column_extrinsic_diagnostics
            if view == "column":
                row_diagnostics["recursive_observer"] = getattr(
                    self, "_observer_last_diagnostics", {"enabled": False}
                )
                lock = getattr(self, "_column_gripper_lock", None)
                if lock is not None:
                    row_diagnostics["gripper_lock"] = {
                        key: (
                            lock[key].tolist()
                            if hasattr(lock[key], "tolist")
                            else lock[key]
                        )
                        for key in lock
                    }
            if v170 is not None:
                # Produce the same frozen V12 carrier tensors the Actor and
                # FCPG were trained on, but do not execute either legacy pixel
                # head. V17.0 remains a diagnostic over calibrated geometry;
                # V17.1 candidate mode replaces the current endpoint below,
                # after its full-image measurement has actually been decoded.
                points = (
                    np.asarray(seed, dtype=np.float32).copy()
                    if v174_runtime
                    else np.asarray(nominal[view], dtype=np.float32).copy()
                )
                corrected_valid = np.asarray(seed_valid, dtype=bool).copy()
                corrected_valid[:, 0] = (
                    corrected_valid[:, 1] & corrected_valid[:, 2]
                )
                bridge = self.system.bridge(
                    vision[0],
                    vision[1],
                    torch.from_numpy(nominal[view]).to(self.device).unsqueeze(0),
                    torch.from_numpy(points).to(self.device).unsqueeze(0),
                    torch.from_numpy(anchors_xyz).to(self.device).unsqueeze(0),
                    torch.from_numpy(corrected_valid).to(self.device).unsqueeze(0),
                    image_size=(
                        int(prepared_frames[view].shape[2]),
                        int(prepared_frames[view].shape[1]),
                    ),
                )
                contexts[view] = (
                    bridge.context_embedding[0]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                raw_contexts[view] = (
                    vision[0][0].detach().float().cpu().numpy().astype(np.float32)
                )
                anchor_tokens[view] = (
                    bridge.anchor_tokens[0]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                correction_seed[view] = points.copy()
                corrected[view] = points.copy()
                valid[view] = corrected_valid
                row_diagnostics.update(
                    {
                        "pixel_head": "v17_full_image_query_transformer_after_carrier",
                        "legacy_v12_pixel_head_executed": False,
                        "global_tip_control_influence": False,
                        "current_tip_authority": (
                            f"{v17_pixel_source}_pending"
                            if v17_pixel_candidate
                            else "calibrated_fk_projection"
                        ),
                        "v17_visual_control_influence": v17_pixel_candidate,
                    }
                )
                if (
                    view == "column"
                    and block_column_out_of_support
                    and not v17_pixel_candidate
                ):
                    blocked.append("column_live_anchor_out_of_support")
                diagnostics[view] = row_diagnostics
                safe_to_commit[view] = False
                continue
            # A two-point pulse fit is only a measurement at the pose it was
            # taken. Leaving that neighbourhood is out of FCPG support: block
            # rather than let metal / OWL / SE(3) invent a new tip triple.
            if view == "column" and block_column_out_of_support:
                blocked.append("column_live_anchor_out_of_support")
            if global_diagnostics["global_locator_accepted"]:
                row_diagnostics["prior_source"] = "global_tip_heatmap"
                # This branch REASSIGNS rather than intersects: when the global
                # locator has accepted, the stale nominal projection is
                # deliberately not allowed to gate the seed.  That override is
                # preserved exactly -- only its bound becomes content-aware.
                # Without this the accepted-locator path would discard every
                # content narrowing applied upstream, and it is enabled by
                # default for the column view on the live path.
                seed_valid = np.all(
                    np.isfinite(anchors_xyz), axis=-1
                ) & self.planning_geometry.content_mask(seed)
            if v165 is not None:
                observations = np.concatenate(
                    (normalized_states, camera_context), axis=-1
                ).astype(np.float32)
                routing = v165.route_anchors(
                    feature_map=v165.project_fcpg_feature_map(vision[1]),
                    nominal_px=nominal[view],
                    anchors_xyz=anchors_xyz,
                    observations=observations,
                    image_size=(
                        int(prepared_frames[view].shape[2]),
                        int(prepared_frames[view].shape[1]),
                    ),
                )
                points = routing.authoritative
                corrected_valid = seed_valid & self.planning_geometry.content_mask(
                    points
                )
                bridge = self.system.bridge(
                    vision[0],
                    vision[1],
                    torch.from_numpy(nominal[view]).to(self.device).unsqueeze(0),
                    torch.from_numpy(points).to(self.device).unsqueeze(0),
                    torch.from_numpy(anchors_xyz).to(self.device).unsqueeze(0),
                    torch.from_numpy(corrected_valid).to(self.device).unsqueeze(0),
                    image_size=(
                        int(prepared_frames[view].shape[2]),
                        int(prepared_frames[view].shape[1]),
                    ),
                )
                if not bool(np.all(corrected_valid[-1])):
                    blocked.append(f"{view}_anchor_out_of_bounds")
                if not routing.authority_within_envelope:
                    blocked.append(f"{view}_fcpg_outside_training_envelope")
                row_diagnostics["v165"] = routing.diagnostics()
                contexts[view] = (
                    bridge.context_embedding[0]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                raw_contexts[view] = (
                    vision[0][0].detach().float().cpu().numpy().astype(np.float32)
                )
                anchor_tokens[view] = (
                    bridge.anchor_tokens[0]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                correction_seed[view] = seed.copy()
                corrected[view] = points.copy()
                valid[view] = corrected_valid
                diagnostics[view] = row_diagnostics
                safe_to_commit[view] = False
                continue
            if self._uses_coarse_fine_fusion():
                try:
                    encoded = encode_fcpg_view(
                        self.system,
                        prepared_frames[view],
                        normalized_states,
                        anchors_xyz,
                        nominal[view],
                        seed_valid,
                        camera_context,
                        self.device,
                        correction_seed_px=seed,
                        derive_center_from_tips=True,
                        precomputed_vision=vision,
                        return_native=True,
                        return_bridge=True,
                    )
                    if len(encoded) == 5:
                        raw_context, bridged, points, native, bridge = encoded
                    elif len(encoded) == 4:
                        raw_context, bridged, points, native = encoded
                        bridge = None
                    else:
                        raw_context, bridged, points = encoded
                        native = getattr(self.system, "last_fcpg_native", None)
                        bridge = None
                    if native is None:
                        raise RuntimeError(
                            "coarse_fine observe_grounding missing model sidecar"
                        )
                    if getattr(self, "_fusion_state", None) is None:
                        self._fusion_state = GroundingFusionState()
                    contract = self._grounding_ranking_contract()
                    points, fusion = apply_coarse_fine_observe_points(
                        native,
                        seed_px=seed,
                        nominal_px=nominal[view],
                        view=view,
                        view_valid=seed_valid,
                        contract=contract,
                        fusion_state=self._fusion_state,
                    )
                    row_diagnostics["coarse_fine"] = coarse_fine_observe_diagnostics(
                        native,
                        fusion,
                        view=view,
                        variant=self._planner_corrector_variant(),
                        contract=contract,
                        global_tip_disabled=True,
                        wrist_correction_enabled=self._wrist_correction_allowed(),
                    )
                finally:
                    self.system.last_fcpg_native = None
            else:
                # Legacy spelling retained in this source for the raw/bridged
                # split contract; the explicit bridge sidecar is unpacked
                # below so grounded rank can consume it when available.
                # raw_context, bridged, points = encode_fcpg_view(
                encoded = encode_fcpg_view(
                    self.system,
                    prepared_frames[view],
                    normalized_states,
                    anchors_xyz,
                    nominal[view],
                    seed_valid,
                    camera_context,
                    self.device,
                    correction_seed_px=seed,
                    derive_center_from_tips=True,
                    precomputed_vision=vision,
                    return_bridge=True,
                )
                if len(encoded) == 4:
                    raw_context, bridged, points, bridge = encoded
                else:
                    raw_context, bridged, points = encoded
                    bridge = None
                self.system.last_fcpg_native = None
                if (
                    getattr(self.config, "grounding_mode", GROUNDING_GATED_FCPG)
                    == GROUNDING_NOMINAL
                ):
                    points = np.asarray(nominal[view], dtype=np.float32).copy()
            # `points` is a third distinct pixel: FCPG applies a learned
            # correction to `seed` bounded by fcpg_max_correction_px (18.0),
            # which exceeds the 28-row pad band, so a seed on real content can
            # be corrected into padding.  This gates `blocked` directly.
            corrected_valid = seed_valid & self.planning_geometry.content_mask(
                points
            )
            corrected_valid[:, 0] = (
                corrected_valid[:, 1] & corrected_valid[:, 2]
            )
            if bridge is not None and hasattr(bridge, "anchor_tokens"):
                # CoarseFine may replace the point after the first bridge call.
                # Re-encode at the final point so ranker's context and tokens
                # describe the same grounded observation.
                bridge = self.system.bridge(
                    vision[0],
                    vision[1],
                    torch.from_numpy(nominal[view]).to(self.device).unsqueeze(0),
                    torch.from_numpy(points).to(self.device).unsqueeze(0),
                    torch.from_numpy(anchors_xyz).to(self.device).unsqueeze(0),
                    torch.from_numpy(corrected_valid).to(self.device).unsqueeze(0),
                    image_size=(
                        int(prepared_frames[view].shape[2]),
                        int(prepared_frames[view].shape[1]),
                    ),
                )
                bridged = (
                    bridge.context_embedding[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            local_correction = np.linalg.norm(points[-1] - seed[-1], axis=-1)
            view_safe = True
            if not np.all(corrected_valid[-1]):
                blocked.append(f"{view}_anchor_out_of_bounds")
                view_safe = False
            if self._uses_coarse_fine_fusion():
                if planner_envelope_should_block(
                    self._planner_corrector_variant(),
                    float(np.nanmax(local_correction)),
                    float(self.config.fcpg_max_correction_px),
                ):
                    blocked.append(f"{view}_fcpg_outside_training_envelope")
                    view_safe = False
            elif np.any(local_correction > self.config.fcpg_max_correction_px):
                blocked.append(f"{view}_fcpg_outside_training_envelope")
                view_safe = False

            contexts[view] = bridged
            raw_contexts[view] = raw_context
            if bridge is not None and hasattr(bridge, "anchor_tokens"):
                anchor_tokens[view] = (
                    bridge.anchor_tokens[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            correction_seed[view] = seed
            corrected[view] = points
            valid[view] = corrected_valid
            diagnostics[view] = row_diagnostics
            safe_to_commit[view] = view_safe
            pending[view] = _AnchorPosterior(
                state_latest=current_state.copy(),
                frame_latest=prepared_frames[view][-1].copy(),
                nominal_latest_px=nominal[view][-1].copy(),
                seed_history_px=seed.copy(),
                corrected_latest_px=points[-1].copy(),
            )

        if (
            commit
            and self.config.recursive_grounding
            and not self._uses_coarse_fine_fusion()
        ):
            for view, posterior in pending.items():
                if safe_to_commit[view]:
                    self._anchor_posteriors[view] = posterior

        return GroundingObservation(
            anchors_xyz=anchors_xyz,
            nominal=nominal,
            correction_seed=correction_seed,
            corrected=corrected,
            valid=valid,
            contexts=contexts,
            raw_contexts=raw_contexts,
            diagnostics=diagnostics,
            blocked_reasons=tuple(sorted(set(blocked))),
            anchor_tokens=anchor_tokens or None,
            full256_tokens=full256_tokens or None,
        )

    def _past_action_chunks(
        self, current_state: np.ndarray, past_raw_actions: np.ndarray | None
    ) -> np.ndarray:
        hold = np.repeat(
            np.asarray(current_state, dtype=np.float32)[None],
            self.frameskip,
            axis=0,
        )
        if past_raw_actions is None:
            raw = np.stack((hold, hold))
        else:
            values = np.asarray(past_raw_actions, dtype=np.float32)
            if values.ndim != 3 or values.shape[1:] != (self.frameskip, 6):
                raise ValueError("past_raw_actions must have shape [N,5,6]")
            if len(values) > 2 or not np.all(np.isfinite(values)):
                raise ValueError("at most two finite past action chunks are allowed")
            padding = [hold] * (2 - len(values))
            raw = np.stack([*padding, *list(values)])
        normalized = (
            raw - self.action_center[None, None]
        ) / self.action_scale[None, None]
        return normalized.reshape(2, self.frameskip * 6).astype(np.float32)

    def _prepare_causal_memory(
        self, memory: CausalPlanningMemory
    ) -> dict[str, np.ndarray]:
        """Compose the memory rows a multiscale rollout conditions on.

        B0 GENERIC CAUSAL-MEMORY LATENT ROUTING DEFECT = DEFERRED.

        The second returned element of ``encode_fcpg_view`` -- the FCPG-corrected
        context -- is what this method hands back as ``"context"``, and therefore
        what reaches ``memory_embedding``.  For any dynamics head trained on the
        first element that is the wrong representation, silently: shapes agree,
        nothing raises, and the rollout simply conditions on features the head
        never saw.  This is a real defect, not a stylistic one.

        It is DEFERRED rather than fixed, deliberately.  Changing what the memory
        path feeds would alter V6/V9 behaviour, and those checkpoints' published
        numbers were produced by this code as written -- so a fix here invalidates
        the comparisons it was supposed to protect, and belongs with a retrain
        rather than on the way to a pilot.  The frozen B2 head is protected
        instead by refusing to enter this path at all; see the guard in
        ``_predict_candidates`` and ``tests/test_b1_frozen_b2_memory_guard.py``.

        Anyone lifting that deferral owns re-deriving the V6/V9 results.
        """

        if self.memory_model is None:
            raise RuntimeError("causal memory requires a v6 memory checkpoint")
        states = np.asarray(memory.states, dtype=np.float32)
        offsets = np.asarray(memory.offsets_raw, dtype=np.int64)
        allowed_offsets = set(self.memory_offsets_raw)
        if any(int(value) not in allowed_offsets for value in offsets):
            raise ValueError("runtime memory offsets do not match the v6 contract")
        if np.any(offsets % self.frameskip):
            raise ValueError("runtime memory offsets must align to frameskip")

        normalized_states = (states - self.state_center) / self.state_scale
        anchors_xyz, nominal, valid = project_history(
            states,
            self.fits,
            self.kinematics,
            image_size=(224, 224),
        )
        transform = self.live_anchor_transforms_px.get("wrist")
        if transform is not None:
            nominal["wrist"] = apply_live_similarity_transform(
                nominal["wrist"], transform
            )
        # `project_history` reports CANVAS validity; camera-content validity is
        # refined here, where observability actually matters.  A historical
        # anchor projected into the letterbox pad band corresponds to no camera
        # pixel, so it must not seed a memory row.  Applied unconditionally
        # rather than inside the `transform is not None` branch: project_history
        # reports full-canvas bounds either way, so the untransformed path was
        # the second, quieter half of the same fail-open.  Under `squash_v0` the
        # content rect is the whole canvas and project_history's margin=1.0
        # result is already a strict subset, so this is exactly a no-op there.
        valid["wrist"] = np.asarray(
            valid["wrist"], dtype=bool
        ) & self.planning_geometry.content_mask(nominal["wrist"])
        camera_context = np.zeros((len(states), 5), dtype=np.float32)
        camera_context[:, 0] = 1.0
        try:
            _, context, corrected = encode_fcpg_view(
                self.system,
                np.asarray(memory.frames["wrist"], dtype=np.uint8),
                normalized_states,
                anchors_xyz,
                nominal["wrist"],
                valid["wrist"],
                camera_context,
                self.device,
                derive_center_from_tips=True,
            )
        finally:
            self.system.last_fcpg_native = None
        if self._uses_coarse_fine_fusion():
            corrected = np.asarray(nominal["wrist"], dtype=np.float32).copy()
        normalized_actions = (
            np.asarray(memory.raw_actions, dtype=np.float32)
            - self.action_center[None, None]
        ) / self.action_scale[None, None]
        return {
            "context": context.astype(np.float32),
            "states": normalized_states.astype(np.float32),
            "actions": normalized_actions.reshape(len(states), -1).astype(np.float32),
            "ages": (offsets.astype(np.float32) / float(self.frameskip)),
            "corrected": corrected.astype(np.float32),
        }

    def attach_v11_6_lewm(self, model: Any) -> None:
        """Use V11.6 dual-view LeWM as the live rank head. B2 stays backup."""

        self.v11_6_lewm = require_v11_6_rank_policy(model)
        self._publish_rank_horizon()

    def _publish_rank_horizon(self) -> None:
        """Put the attached rank model's real reach on the record.

        A world model whose action horizon is shorter than the planner's gets
        its rollout padded out by repeating the last predicted state, so what
        reaches the cost function has the right shape while its tail carries no
        new information -- and a repeated state is indistinguishable from a
        genuinely stationary one, so nothing downstream can notice. That is how
        a ``goal_final_steps=3`` terminal cost came to average a single 0.67s
        prediction three times on V12.1, which reads as a full-horizon score.

        Both numbers go into the contract, which every plan artifact carries, so
        a horizon mismatch is recorded at the point of attach rather than left
        for a later reader to rederive from checkpoint metadata.
        """

        horizon = int(self.horizon)
        declared = getattr(
            getattr(self, "v11_lewm_ranker", None), "rank_prediction_steps", None
        )
        steps = horizon if declared is None else max(1, min(int(declared), horizon))
        self.model_contract.update(
            {
                "rank_prediction_steps": steps,
                "rank_planner_horizon": horizon,
                "rank_prediction_padded_tail": horizon - steps,
                "rank_goal_final_steps": max(
                    1, min(int(self.config.goal_final_steps), steps)
                ),
            }
        )
        if horizon > steps:
            print(
                f"[V11] rank horizon mismatch: model predicts {steps} chunk "
                f"states, planner asks for {horizon}; terminal cost will read "
                f"the first {steps} only (goal_final_steps clamped to "
                f"{self.model_contract['rank_goal_final_steps']})",
                flush=True,
            )

    def _lewm_goal_horizon(self, length: int) -> tuple[int, int]:
        """Genuine prediction prefix and the terminal-cost window on it.

        Rollouts pad a short world model out to the planner horizon by
        repeating the last predicted state. Scoring that padded tail with
        ``goal_final_steps=3`` averages the same 0.67s state three times.
        Commanded-trajectory costs must keep the full horizon -- only the
        LeWM-derived predicted terms use this prefix.
        """

        genuine = max(
            1,
            min(int(getattr(self, "_rank_prediction_steps", length)), int(length)),
        )
        final_steps = max(1, min(int(self.config.goal_final_steps), genuine))
        return genuine, final_steps

    def _closure_geometry(
        self,
        view: str,
        *,
        current_state: np.ndarray,
        nominal: Mapping[str, np.ndarray],
        corrected: Mapping[str, np.ndarray],
        valid: Mapping[str, np.ndarray],
        pickup: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Everything the closure objective needs, all read in one view.

        The jaw span and the pickup offset compared against it have to come
        from the same camera. The detected span is still reported next to the
        projected one: within a view they agree by construction, so a live gap
        between them means the correction or the transform has drifted.
        """

        centroid = pickup.get("centroid_px")
        if not pickup.get("visible") or centroid is None:
            return None
        correction = corrected[view][-1] - nominal[view][-1]
        # The transform `nominal` carries, not the raw one. Out of the pulse's
        # neighbourhood `nominal` is left untransformed on purpose, and pairing
        # its correction with the transform anyway put the projected jaws ~27px
        # (2.5cm) from where the detector had them -- spans still matched, since
        # a translation cannot change a separation, so it read as a real gap
        # between the pickup and the jaws and the grasp was ruled unreachable
        # while the arm was already on the fish.
        transform = (getattr(self, "applied_anchor_transforms_px", None) or {}).get(view)
        current_anchor_px = project_future_anchor_pixels(
            current_state[None, None],
            self.fits[view],
            self.kinematics,
            current_correction_px=correction,
            live_transform_px=transform,
        )
        projected_span = float(
            np.linalg.norm(current_anchor_px[0, 0, 2] - current_anchor_px[0, 0, 1])
        )
        detected_span = detected_tip_separation_px(corrected[view][-1], valid[view][-1])
        if not np.isfinite(projected_span) or projected_span <= 1.0e-6:
            return None
        # Published because "the jaws will not close" has two causes that look
        # identical in a total offset, and they need opposite responses: too far
        # along the jaw axis means centre the arm sideways, too far across means
        # the pickup is still in front of the jaws and the arm has to come in.
        tips = current_anchor_px[0, 0]
        axis = (tips[2] - tips[1]) / projected_span
        delta = np.asarray(centroid, dtype=np.float64) - 0.5 * (tips[1] + tips[2])
        return {
            "view": view,
            "correction": correction,
            "transform": transform,
            "projected_span_px": projected_span,
            "detected_span_px": detected_span,
            "target_along_px": abs(float(delta @ axis)),
            "closing_axis_px": np.asarray(axis, dtype=np.float64),
            "target_across_px": abs(
                float(delta[0] * axis[1] - delta[1] * axis[0])
            ),
            "target_px": np.asarray(centroid, dtype=np.float64),
        }

    def _closure_objective(
        self,
        phase: ManipulationPhase,
        *,
        predicted_states: np.ndarray,
        commanded_states: np.ndarray,
        current_state: np.ndarray,
        nominal: Mapping[str, np.ndarray],
        corrected: Mapping[str, np.ndarray],
        valid: Mapping[str, np.ndarray],
        pickups: Mapping[str, Mapping[str, Any]],
        column_predicted_anchor_px: np.ndarray,
        column_commanded_anchor_px: np.ndarray,
        three_point: Mapping[str, Any] | None = None,
        gripper_to_pickup_px: float | None = None,
        gripper_to_pickup_m: float | None = None,
    ) -> dict[str, Any]:
        """Score the grasp in the best view that can still see the pickup.

        Two costs come back for the same objective: the commanded one is exact
        FK of the action and is what ranks, the predicted one is the same
        objective read off the LeWM rollout. Their gap is the measurement of
        how much of the grasp the world model can actually account for.
        """

        candidates = int(commanded_states.shape[0])
        zeros = np.zeros(candidates, dtype=np.float64)
        empty = {
            "closure_cost": zeros,
            "commanded_closure_cost": zeros.copy(),
            "closure_weight": 0.0,
            "closure_view": None,
            "closure_reachable": False,
            "projected_span_px": float("nan"),
            "detected_span_px": float("nan"),
            "target_along_px": float("nan"),
            "target_across_px": float("nan"),
        }
        if phase is not ManipulationPhase.GRASP:
            return empty
        geometry = None
        for view in ("wrist", "column"):
            geometry = self._closure_geometry(
                view,
                current_state=current_state,
                nominal=nominal,
                corrected=corrected,
                valid=valid,
                pickup=pickups.get(view) or {},
            )
            if geometry is not None:
                break
        if geometry is None:
            return empty

        view = geometry["view"]
        if view == "column":
            predicted_px = column_predicted_anchor_px
            commanded_px = column_commanded_anchor_px
        else:
            predicted_px = project_future_anchor_pixels(
                predicted_states,
                self.fits[view],
                self.kinematics,
                current_correction_px=geometry["correction"],
                live_transform_px=geometry["transform"],
            )
            commanded_px = project_future_anchor_pixels(
                commanded_states,
                self.fits[view],
                self.kinematics,
                current_correction_px=geometry["correction"],
                live_transform_px=geometry["transform"],
            )
        shape = measured_object_shape(pickups.get(geometry["view"]))
        shared = {
            "current_tip_separation_px": geometry["projected_span_px"],
            # The chunk that will actually be executed before the next replan.
            "horizon_index": 0,
            **shape,
        }
        # Pixel containment / corridor is not a close license. Score the
        # jaw only when the configured pickup is metrically near or held.
        thresholds = getattr(self, "thresholds", None)
        reachable = close_allowed_for_pickup(
            three_point_verdict=(
                None
                if not isinstance(three_point, Mapping)
                else three_point.get("verdict")
            ),
            gripper_to_pickup_px=gripper_to_pickup_px,
            gripper_to_pickup_m=gripper_to_pickup_m,
            pickup_near_px=getattr(thresholds, "pickup_near_px", 28.0),
            pickup_near_m=getattr(thresholds, "pickup_near_m", 0.018),
        )
        score = (
            (lambda anchors: predicted_closure_cost(anchors, geometry["target_px"], **shared))
            if reachable
            # Holding the jaw still is not a claim about the object, so the
            # shape half of ``shared`` is not its business and it does not
            # accept it.
            else (
                lambda anchors: aperture_hold_cost(
                    anchors,
                    current_tip_separation_px=shared["current_tip_separation_px"],
                    horizon_index=shared["horizon_index"],
                )
            )
        )
        raw_commanded = score(commanded_px)
        # A candidate whose jaws did not project is not preferred: it takes the
        # saturated worst score, never a free zero.
        saturate = {"nan": CLOSURE_COST_CEILING, "posinf": CLOSURE_COST_CEILING}
        return {
            "closure_cost": np.nan_to_num(score(predicted_px), **saturate),
            "commanded_closure_cost": np.nan_to_num(raw_commanded, **saturate),
            "closure_weight": (
                GRASP_CLOSURE_COST_WEIGHT
                if np.any(np.isfinite(raw_commanded))
                else 0.0
            ),
            "closure_view": view,
            "closure_reachable": bool(reachable),
            "projected_span_px": geometry["projected_span_px"],
            "detected_span_px": geometry["detected_span_px"],
            "target_along_px": geometry["target_along_px"],
            "target_across_px": geometry["target_across_px"],
        }

    def _predict_candidates(
        self,
        context: np.ndarray,
        normalized_states: np.ndarray,
        past_chunks: np.ndarray,
        candidate_chunks: np.ndarray,
        causal_memory: CausalPlanningMemory | None,
        *,
        measurement_reliable: bool,
        column_context: np.ndarray | None = None,
        grounded_rank_inputs: Mapping[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, str, int]:
        # How much of the rollout below is a prediction rather than a padded
        # repeat of its last state. Written before any rollout runs and on every
        # path, so a short-horizon ranker cannot leave a stale count behind for
        # the next plan to score against. Full horizon is the default because
        # every rollout other than the two-view ranker fills it genuinely.
        self._rank_prediction_steps = int(candidate_chunks.shape[1])
        ranker = getattr(self, "v11_lewm_ranker", None)
        observation_fn = getattr(self, "v11_lewm_observation_fn", None)
        raw_prefixes = getattr(self, "_pending_v11_rank_prefixes", None)
        if (
            ranker is not None
            and callable(observation_fn)
            and raw_prefixes is not None
        ):
            prefixes = np.asarray(raw_prefixes, dtype=np.float32).copy()
            observations = observation_fn()
            raw_states = ranker.rollout_candidate_raw_states(
                observations,
                prefixes,
                horizon=int(candidate_chunks.shape[1]),
            )
            if raw_states is not None:
                declared = getattr(ranker, "rank_prediction_steps", None)
                if declared is None:
                    declared = raw_states.shape[1]
                self._rank_prediction_steps = max(
                    1, min(int(declared), int(candidate_chunks.shape[1]))
                )
                normalized = (
                    raw_states - self.state_center[None, None]
                ) / self.state_scale[None, None]
                latent = np.zeros(
                    (len(raw_states), int(candidate_chunks.shape[1]), 1),
                    dtype=np.float32,
                )
                return (
                    latent,
                    normalized.astype(np.float32),
                    V11_6_ROLLOUT_VARIANT,
                    0,
                )
        rank_policy = getattr(self, "v11_6_lewm", None)
        if rank_policy is not None and column_context is not None:
            if grounded_rank_inputs is None:
                grounded_rank_inputs = getattr(
                    self, "_grounded_rank_inputs", None
                )
            grounded = (
                {} if grounded_rank_inputs is None else dict(grounded_rank_inputs)
            )
            latent, state = rollout_v11_6_candidates(
                rank_policy,
                grounded.get("wrist_context", context),
                grounded.get("column_context", column_context),
                normalized_states,
                past_chunks,
                candidate_chunks,
                state_center=self.state_center,
                state_scale=self.state_scale,
                action_center=self.action_center,
                action_scale=self.action_scale,
                device=self.device,
                anchor_px=grounded.get("anchor_px"),
                anchor_xyz=grounded.get("anchor_xyz"),
                anchor_seen=grounded.get("anchor_seen"),
                anchor_tokens=grounded.get("anchor_tokens"),
            )
            rank_config = getattr(rank_policy, "config", None)
            offsets = (
                getattr(rank_config, "future_offsets", None)
                if rank_config is not None
                else None
            )
            if offsets is None:
                declared = int(getattr(rank_policy, "future_horizon", 0) or 0)
            else:
                declared = len(tuple(offsets))
            if declared:
                self._rank_prediction_steps = max(
                    1, min(declared, int(candidate_chunks.shape[1]))
                )
            return latent, state, V11_6_ROLLOUT_VARIANT, 0
        use_memory = should_use_memory(
            self.config.memory_policy,
            model_available=self.memory_model is not None,
            memory_available=causal_memory is not None,
            measurement_reliable=measurement_reliable,
        )
        if use_memory and self.model_contract["dynamics_protocol"] == (
            B2HEAD_RUNTIME_PROTOCOL
        ):
            raise RuntimeError(
                f"dynamics protocol {B2HEAD_RUNTIME_PROTOCOL!r} reached the "
                "causal-memory rollout, which feeds the corrected context to a "
                "head frozen on the raw context. Failing closed"
            )
        if not use_memory:
            latent, state = _rollout_candidates(
                self.model,
                context,
                normalized_states,
                past_chunks,
                candidate_chunks,
            )
            variant = (
                "v9_local_backbone"
                if self.model_contract["dynamics_protocol"] == V9_PROTOCOL
                else "v6_backbone"
                if self.model_contract["dynamics_protocol"] == V6_PROTOCOL
                else "state_residual"
            )
            return latent, state, variant, 0

        memory = self._prepare_causal_memory(causal_memory)

        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(value, device=self.device, dtype=torch.float32)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            latent_tensor, state_tensor = rollout_multiscale_candidates(
                self.memory_model,
                tensor(context),
                tensor(normalized_states),
                tensor(past_chunks),
                tensor(candidate_chunks),
                memory_embedding=tensor(memory["context"]),
                memory_actions=tensor(memory["actions"]),
                memory_state=tensor(memory["states"]),
                memory_age=tensor(memory["ages"]),
                memory_valid=torch.ones(
                    len(memory["ages"]), dtype=torch.bool, device=self.device
                ),
            )
        return (
            latent_tensor.float().cpu().numpy(),
            state_tensor.float().cpu().numpy(),
            (
                "v9_memory_fallback"
                if self.model_contract["dynamics_protocol"] == V9_PROTOCOL
                else "v6_multiscale_memory"
            ),
            len(memory["ages"]),
        )

    def _collect_extra_actor_candidates(
        self,
        current_state: np.ndarray,
        *,
        relative_goal_xyz: np.ndarray | None,
        phase: ManipulationPhase,
    ) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
        """Ask the runtime for extra actor prefixes. Provider may see the goal; LeWM does not."""

        provider = getattr(self, "extra_actor_prefix_provider", None)
        if not callable(provider):
            return None
        provided = provider(
            np.asarray(current_state, dtype=np.float32),
            relative_goal_xyz=relative_goal_xyz,
            phase=phase,
        )
        if provided is None:
            return None
        prefixes = np.asarray(provided["prefixes"], dtype=np.float32)
        metadata = list(provided["metadata"])
        if prefixes.ndim != 3 or prefixes.shape[1] != self.horizon_raw or prefixes.shape[2] != 6:
            raise ValueError(
                "extra actor prefixes must have shape "
                f"[N,{self.horizon_raw},6], got {prefixes.shape}"
            )
        if len(metadata) != len(prefixes):
            raise ValueError("extra actor metadata count must match prefixes")
        return prefixes, metadata

    def _attach_extra_actor_candidates(
        self,
        prefixes: np.ndarray,
        candidate_ik_safe: np.ndarray,
        retrieval_cost: np.ndarray,
        selected_phase_distance: np.ndarray,
        candidate_metadata: list[dict[str, Any]],
        *,
        current_state: np.ndarray,
        relative_goal_xyz: np.ndarray | None,
        phase: ManipulationPhase,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        v165 = self.v165_runtime_adapter
        v168 = self.v168_runtime_adapter
        v170 = self.v170_runtime_adapter
        if v170 is not None:
            provided = v170.propose_actor_prefix(
                current_state=np.asarray(current_state, dtype=np.float32),
                baseline_prefixes=prefixes,
                candidate_ik_safe=candidate_ik_safe,
                selected_phase_distance=selected_phase_distance,
                horizon_raw=self.horizon_raw,
                phase=phase,
            )
            extra_prefixes = np.asarray(provided["prefixes"], dtype=np.float32)
            extra_metadata = list(provided["metadata"])
            if extra_prefixes.shape != (len(extra_metadata), self.horizon_raw, 6):
                raise ValueError(
                    "V17.0 Actor prefixes must align with metadata and horizon"
                )
            if v170.mode == "shadow":
                if len(extra_prefixes):
                    chunks = extra_prefixes[:, : self.config.command_chunk]
                    fixed = v170.action_contract_fixed_point_mask(
                        current_state=np.asarray(current_state, dtype=np.float32),
                        five_command_candidates=chunks,
                    )
                    if not bool(fixed.all()):
                        raise RuntimeError(
                            "V17.0 shadow Actor output is not a projector fixed point"
                        )
                    v170.predict_candidate_states(
                        current_state=np.asarray(current_state, dtype=np.float32),
                        five_command_candidates=chunks,
                    )
                return (
                    prefixes,
                    candidate_ik_safe,
                    retrieval_cost,
                    selected_phase_distance,
                    candidate_metadata,
                )
            extra = (extra_prefixes, extra_metadata)
        elif v168 is not None:
            provided = v168.propose_actor_prefix(
                current_state=np.asarray(current_state, dtype=np.float32),
                baseline_prefixes=prefixes,
                candidate_ik_safe=candidate_ik_safe,
                selected_phase_distance=selected_phase_distance,
                horizon_raw=self.horizon_raw,
                phase=phase,
            )
            extra_prefixes = np.asarray(provided["prefixes"], dtype=np.float32)
            extra_metadata = list(provided["metadata"])
            if extra_prefixes.shape != (len(extra_metadata), self.horizon_raw, 6):
                raise ValueError(
                    "V16.8 Actor prefixes must align with metadata and horizon"
                )
            if v168.mode == "shadow":
                if len(extra_prefixes):
                    if v165 is None:
                        raise RuntimeError("V16.8 shadow has no frozen State/FK LeWM")
                    chunks = extra_prefixes[:, : self.config.command_chunk]
                    fixed = v165.action_contract_fixed_point_mask(
                        current_state=np.asarray(current_state, dtype=np.float32),
                        five_command_candidates=chunks,
                    )
                    if not bool(fixed.all()):
                        raise RuntimeError(
                            "V16.8 shadow Actor output is not a projector fixed point"
                        )
                    v168.record_shadow_lewm(
                        v165.predict_candidate_states(
                            current_state=np.asarray(current_state, dtype=np.float32),
                            five_command_candidates=chunks,
                        )
                    )
                # Shadow inference is receipt-only: it never enters any array
                # consumed by common candidate safety, cost normalization, or
                # selected_index.
                return (
                    prefixes,
                    candidate_ik_safe,
                    retrieval_cost,
                    selected_phase_distance,
                    candidate_metadata,
                )
            extra = (extra_prefixes, extra_metadata)
        elif v165 is None:
            extra = self._collect_extra_actor_candidates(
                current_state,
                relative_goal_xyz=relative_goal_xyz,
                phase=phase,
            )
        else:
            provided = v165.propose_actor_prefix(
                current_state=np.asarray(current_state, dtype=np.float32),
                baseline_prefixes=prefixes,
                candidate_ik_safe=candidate_ik_safe,
                selected_phase_distance=selected_phase_distance,
                horizon_raw=self.horizon_raw,
                phase=phase,
            )
            extra_prefixes = np.asarray(provided["prefixes"], dtype=np.float32)
            extra_metadata = list(provided["metadata"])
            if extra_prefixes.shape != (len(extra_metadata), self.horizon_raw, 6):
                raise ValueError(
                    "V16.5 Actor prefixes must align with metadata and horizon"
                )
            extra = (extra_prefixes, extra_metadata)
        if extra is None:
            return (
                prefixes,
                candidate_ik_safe,
                retrieval_cost,
                selected_phase_distance,
                candidate_metadata,
            )
        extra_prefixes, extra_metadata = extra
        return merge_extra_actor_prefixes(
            prefixes,
            candidate_ik_safe,
            retrieval_cost,
            selected_phase_distance,
            candidate_metadata,
            extra_prefixes,
            extra_metadata,
            prepend=v168 is None and v170 is None,
        )

    def _plan_anchor_trajectory(
        self,
        prepared_frames: dict[str, np.ndarray],
        state_history: np.ndarray,
        grounding: GroundingObservation,
        *,
        past_raw_actions: np.ndarray | None,
        causal_memory: CausalPlanningMemory | None,
        measurement_reliable: bool,
    ) -> RealtimePlan:
        """Follow an implicit successful demonstration via LeWM anchor forecasts."""

        current_state = np.asarray(state_history[-1], dtype=np.float32)
        normalized_states = (
            np.asarray(state_history, dtype=np.float32) - self.state_center
        ) / self.state_scale
        contexts = grounding.contexts
        raw_contexts = grounding.raw_contexts
        blocked = list(grounding.blocked_reasons)
        peg_descriptor = _template_tracking_removed("peg", "column")
        peg_registration_current: bool | None = None
        peg_registration_error: str | None = None
        if peg_descriptor.get("configured", False):
            try:
                self._validate_context_registration()
                peg_registration_current = True
            except ContextRegistrationError as exc:
                peg_registration_current = False
                peg_registration_error = str(exc)
                if self.config.experiment_method == "full":
                    blocked.append("peg_registration_contract_stale")
        peg_descriptor = {
            **peg_descriptor,
            "registration": self.context_registration,
            "clearance_m": self.config.peg_clearance_m,
            "registration_current": peg_registration_current,
            "registration_error": peg_registration_error,
        }
        if peg_descriptor.get("configured", False):
            blocked.append("peg_context_requires_phase_target_control")
        query_embedding = np.concatenate(
            (contexts["wrist"][-1], contexts["column"][-1])
        )
        bank_embeddings = np.concatenate(
            (self.primary_bank, self.secondary_bank), axis=1
        )
        continuation = self._demonstration_progress
        continuation_missing = False
        continuation_bank_index: int | None = None
        if continuation is not None:
            continuation_bank_index = self._bank_index_by_episode_frame.get(
                (continuation.episode, continuation.next_frame)
            )
            continuation_missing = continuation_bank_index is None
            if continuation_missing:
                blocked.append("demonstration_trajectory_complete")

        if continuation_bank_index is None:
            retrieval = retrieve_diverse_candidates(
                query_embedding,
                current_state,
                bank_embeddings,
                self.bank_states,
                self.bank_rows,
                self.bank_episodes,
                state_scale=self.state_scale,
                per_episode=self.config.candidates_per_episode,
                minimum_row_gap=self.config.candidate_min_gap,
                state_weight=self.config.state_retrieval_weight,
            )
        else:
            candidate_slice = np.asarray([continuation_bank_index], dtype=np.int64)
            retrieval = retrieve_diverse_candidates(
                query_embedding,
                current_state,
                bank_embeddings[candidate_slice],
                self.bank_states[candidate_slice],
                self.bank_rows[candidate_slice],
                self.bank_episodes[candidate_slice],
                state_scale=self.state_scale,
                per_episode=1,
                minimum_row_gap=0,
                state_weight=self.config.state_retrieval_weight,
            )
        bank_index = {int(row): index for index, row in enumerate(self.bank_rows)}
        source_indices = np.asarray(
            [bank_index[int(row)] for row in retrieval.rows], dtype=np.int64
        )
        prefixes = rebase_action_prefixes(
            self.bank_prefixes[source_indices],
            self.bank_states[source_indices],
            current_state,
        )
        candidate_chunks = chunk_and_normalize_actions(
            prefixes,
            frameskip=self.frameskip,
            center=self.action_center,
            scale=self.action_scale,
        )
        past_chunks = self._past_action_chunks(current_state, past_raw_actions)
        self._pending_v11_rank_prefixes = prefixes
        rank_grounding = (
            {
                "wrist_context": contexts["wrist"],
                "column_context": contexts["column"],
                "anchor_px": np.stack(
                    [corrected[view] for view in RESIDUAL_VIEWS], axis=1
                ).astype(np.float32),
                "anchor_xyz": np.asarray(grounding.anchors_xyz, dtype=np.float32),
                "anchor_seen": np.stack(
                    [valid[view] for view in RESIDUAL_VIEWS], axis=1
                ).astype(np.float32),
                "anchor_tokens": np.stack(
                    [grounding.anchor_tokens[view] for view in RESIDUAL_VIEWS],
                    axis=1,
                ).astype(np.float32),
            }
            if grounding.anchor_tokens is not None
            and all(view in grounding.anchor_tokens for view in RESIDUAL_VIEWS)
            else None
        )
        previous_rank_grounding = getattr(self, "_grounded_rank_inputs", None)
        self._grounded_rank_inputs = rank_grounding
        try:
            (
                _latent_prediction,
                state_prediction,
                rollout_variant,
                memory_token_count,
            ) = self._predict_candidates(
                # Raw projected CLS remains the legacy fallback. When the
                # complete grounded payload exists, the base implementation
                # replaces it with the corrected context/tokens explicitly.
                raw_contexts["wrist"],
                normalized_states,
                past_chunks,
                candidate_chunks,
                causal_memory,
                measurement_reliable=measurement_reliable,
                column_context=raw_contexts["column"],
            )
        finally:
            self._grounded_rank_inputs = previous_rank_grounding
        predicted_states = (
            state_prediction * self.state_scale[None, None]
            + self.state_center[None, None]
        ).astype(np.float32)
        predicted_anchors = np.asarray(
            self.kinematics.forward_anchors(predicted_states), dtype=np.float64
        )
        commanded_states = prefixes[:, self.frameskip - 1 :: self.frameskip]
        rebase_delta = (
            current_state[None] - self.bank_states[source_indices]
        )
        demonstrated_future = (
            self.bank_future_states[source_indices] + rebase_delta[:, None]
        )
        demonstration_cost = np.mean(
            np.square(
                (predicted_states - demonstrated_future)
                / self.state_scale[None, None]
            ),
            axis=(1, 2),
        )
        model_command_cost = np.mean(
            np.square(
                (predicted_states - commanded_states)
                / self.state_scale[None, None]
            ),
            axis=(1, 2),
        )

        def unit_cost(values: np.ndarray) -> np.ndarray:
            vector = np.asarray(values, dtype=np.float64)
            minimum = float(np.min(vector))
            spread = float(np.max(vector) - minimum)
            if spread <= 1e-12:
                return np.zeros_like(vector)
            return (vector - minimum) / spread

        costs = (
            unit_cost(retrieval.combined_distance)
            + unit_cost(demonstration_cost)
            + 0.25 * unit_cost(model_command_cost)
        )
        input_safety = candidate_safety_checks(
            prefixes[:, 0],
            current_state,
            max_joint_jump=self.config.max_first_command_jump,
            raw_prefixes=prefixes,
            predicted_states=predicted_states,
            position_lower=SO101_POSITION_LOWER,
            position_upper=SO101_POSITION_UPPER,
            max_prefix_step_jump=self.config.max_prefix_step_jump,
            prediction_bound_margin=self.config.prediction_bound_margin,
        )
        predicted_xyz = np.asarray(
            self.kinematics.forward(predicted_states), dtype=np.float64
        )
        workspace_safe = np.all(
            (predicted_xyz >= self.workspace_lower[None, None])
            & (predicted_xyz <= self.workspace_upper[None, None]),
            axis=(1, 2),
        )
        pre_safe = input_safety.safe & workspace_safe
        if continuation_missing:
            pre_safe[:] = False
        candidate_rows: list[dict[str, Any]] = []
        for index, row in enumerate(retrieval.rows):
            reasons: list[str] = []
            if not input_safety.hard_command_safe[index]:
                reasons.append("demonstration_prefix_safety")
            if not input_safety.prediction_plausible[index]:
                reasons.append("lewm_prediction_bounds")
            if not workspace_safe[index]:
                reasons.append("lewm_prediction_workspace")
            candidate_rows.append(
                {
                    "source": (
                        "locked_demonstration_continuation"
                        if continuation_bank_index is not None
                        else "implicit_successful_demonstration"
                    ),
                    "index": index,
                    "row": int(row),
                    "episode": int(retrieval.episodes[index]),
                    "frame": int(self.primary.data.frame_indices[int(row)]),
                    "cost": float(costs[index]),
                    "retrieval_cost": float(retrieval.combined_distance[index]),
                    "demonstration_future_cost": float(demonstration_cost[index]),
                    "model_command_cost": float(model_command_cost[index]),
                    "safe": False,
                    "reasons": reasons,
                    "anchor_ik_attempted": False,
                    "continuation_required": continuation is not None,
                    "continuation_match": bool(
                        continuation is not None
                        and int(retrieval.episodes[index]) == continuation.episode
                        and int(self.primary.data.frame_indices[int(row)])
                        == continuation.next_frame
                    ),
                }
            )

        selected_index: int | None = None
        selected_prefix: np.ndarray | None = None
        selected_prediction: np.ndarray | None = None
        selected_commanded: np.ndarray | None = None
        selected_ik_error: float | None = None
        for index in np.argsort(costs, kind="stable"):
            index = int(index)
            if not pre_safe[index]:
                continue
            trajectory = track_anchor_trajectory(
                self.kinematics,
                current_state,
                predicted_anchors[index],
                frameskip=self.frameskip,
                tolerance_m=3e-3,
                max_step_normalized=self.config.max_prefix_step_jump,
            )
            candidate_rows[index]["anchor_ik_attempted"] = True
            candidate_rows[index]["anchor_ik_max_error_m"] = float(
                np.max(trajectory.per_horizon_max_error_m)
            )
            if not trajectory.converged:
                candidate_rows[index]["reasons"].append("anchor_ik_not_converged")
                continue
            command_safety = candidate_safety_checks(
                trajectory.raw_prefix[None, 0],
                current_state,
                max_joint_jump=self.config.max_first_command_jump,
                raw_prefixes=trajectory.raw_prefix[None],
                predicted_states=trajectory.horizon_states[None],
                position_lower=SO101_POSITION_LOWER,
                position_upper=SO101_POSITION_UPPER,
                max_prefix_step_jump=self.config.max_prefix_step_jump,
                prediction_bound_margin=0.0,
            )
            chunk_check = validate_command_chunk(
                trajectory.raw_prefix[: self.config.command_chunk],
                current_state,
                expected_commands=self.config.command_chunk,
                max_first_jump=self.config.max_first_command_jump,
                max_step_jump=self.config.max_prefix_step_jump,
            )
            command_xyz = np.asarray(
                self.kinematics.forward(trajectory.horizon_states), dtype=np.float64
            )
            command_workspace_safe = bool(
                np.all(command_xyz >= self.workspace_lower[None])
                and np.all(command_xyz <= self.workspace_upper[None])
            )
            if not bool(command_safety.safe[0]):
                candidate_rows[index]["reasons"].append("anchor_ik_command_safety")
                continue
            if not chunk_check.safe:
                candidate_rows[index]["reasons"].extend(chunk_check.reasons)
                continue
            if not command_workspace_safe:
                candidate_rows[index]["reasons"].append("anchor_ik_workspace")
                continue
            candidate_rows[index]["safe"] = True
            selected_index = index
            selected_prefix = trajectory.raw_prefix.copy()
            selected_prediction = predicted_states[index].copy()
            selected_commanded = trajectory.horizon_states.copy()
            selected_ik_error = float(np.max(trajectory.per_horizon_max_error_m))
            break

        if selected_index is None:
            blocked.append("no_safe_anchor_trajectory_candidate")

        selected_chunk = (
            None
            if selected_prefix is None
            else selected_prefix[: self.config.command_chunk].copy()
        )
        prediction_anchor_px: np.ndarray | None = None
        command_anchor_px: np.ndarray | None = None
        anchor_paths_px: dict[str, dict[str, np.ndarray]] = {}
        if selected_prediction is not None and selected_commanded is not None:
            anchor_paths_px = self._anchor_paths_px(
                predicted=selected_prediction,
                commanded=selected_commanded,
                nominal=grounding.nominal,
                corrected=grounding.corrected,
            )
            prediction_anchor_px = anchor_paths_px["column"]["predicted"]
            command_anchor_px = anchor_paths_px["column"]["commanded"]

        round_id = f"r-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        guard_point = np.asarray(
            grounding.corrected["column"][-1, 0], dtype=np.float32
        )
        guard = PlanningRoundGuard(
            round_id=round_id,
            created_monotonic_s=time.monotonic(),
            state=current_state.copy(),
            target_px=guard_point.copy(),
            target_kind="grounded_trajectory",
        )
        model_contract = {
            **self.model_contract,
            **self.live_contract_fields(),
            "control_contract": (
                "FCPG current three anchors -> action-conditioned LeWM three-anchor "
                "forecast -> three-anchor IK -> execute one chunk -> reobserve"
            ),
            "task_progress_source": (
                "initial observation alignment, then committed same-episode "
                "five-frame demonstration continuation"
            ),
            "demonstration_progress": (
                None
                if continuation is None
                else {
                    "episode": continuation.episode,
                    "required_source_frame": continuation.next_frame,
                    "completed_chunks": continuation.completed_chunks,
                    "continuation_available": not continuation_missing,
                }
            ),
            "handwritten_grasp_release": False,
            "rollout_variant": rollout_variant,
            "active_memory_tokens": int(memory_token_count),
            "measurement_reliable": bool(measurement_reliable),
            "v168_runtime_status": (
                None
                if self.v168_runtime_adapter is None
                else self.v168_runtime_adapter.status()
            ),
            "selected_anchor_ik_max_error_m": selected_ik_error,
        }
        empty_target = {"visible": False, "reason": "not_used_for_control"}
        return RealtimePlan(
            guard=guard,
            phase=ManipulationPhase.TRAJECTORY,
            phase_status={
                "phase": ManipulationPhase.TRAJECTORY.value,
                "pending_phase": None,
                "pending_confirmations": 0,
                "required_confirmations": 0,
            },
            current_state=current_state.copy(),
            frames={view: prepared_frames[view].copy() for view in VIEWS},
            nominal=grounding.nominal,
            correction_seed=grounding.correction_seed,
            corrected=grounding.corrected,
            valid=grounding.valid,
            grounding_diagnostics=grounding.diagnostics,
            target_descriptors={
                "pickup_column": dict(empty_target),
                "pickup_wrist": dict(empty_target),
                "drop_zone_column": dict(empty_target),
                "peg_column": peg_descriptor,
            },
            target_profile=self._target_profile_status(),
            target={
                "kind": "implicit_demonstration_trajectory",
                "visible": False,
                "centroid_px": guard_point,
                "base_xyz": None,
                "gripper_distance_m": None,
            },
            selected_index=selected_index,
            selected_chunk=selected_chunk,
            selected_prefix=selected_prefix,
            predicted_states=selected_prediction,
            commanded_states=selected_commanded,
            predicted_gripper_px=(
                None if prediction_anchor_px is None else prediction_anchor_px[:, 0]
            ),
            commanded_gripper_px=(
                None if command_anchor_px is None else command_anchor_px[:, 0]
            ),
            predicted_anchor_px=prediction_anchor_px,
            commanded_anchor_px=command_anchor_px,
            anchor_paths_px=anchor_paths_px,
            candidate_rows=candidate_rows,
            executable=selected_index is not None and not blocked,
            blocked_reasons=tuple(sorted(set(blocked))),
            model_contract=model_contract,
        )

    def detect_live_target(
        self,
        column_frame: np.ndarray,
        phase: ManipulationPhase,
        *,
        update_track: bool = False,
        gripper_center_px: np.ndarray | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Re-find the phase's target on a frame taken after the plan was made.

        Composed exactly like the planning-time search, frozen-feature recovery
        included. A re-check built on the correlation matcher alone is blind on
        precisely the frames the recovery path exists for, so every plan the
        planner grounded on frozen features would be rejected here rather than
        re-validated: live, a fish 144mm away passed the plan gate and failed
        this one ten times out of ten, on the same round the planner had already
        needed the fallback to see it at all.
        """

        role = target_role_for_phase(phase)
        if role is None:
            raise RuntimeError(f"phase {phase.value} has no executable target")
        self._lock_column_roles(
            column_frame, gripper_center_px=gripper_center_px
        )
        descriptor = self.ground_object(
            role,
            "column",
            column_frame,
            gripper_center_px=gripper_center_px,
            update_track=update_track,
        )
        if not descriptor.get("visible", False):
            reason = descriptor.get("reason", "not reliably visible")
            raise RuntimeError(
                f"current {role} is not reliably visible: {reason}"
            )
        return role, descriptor

    @torch.inference_mode()
    def plan(
        self,
        frames: dict[str, np.ndarray],
        states: np.ndarray,
        *,
        past_raw_actions: np.ndarray | None = None,
        causal_memory: CausalPlanningMemory | None = None,
        current_fingertips: LiveFingertipMeasurement | None = None,
    ) -> RealtimePlan:
        state_history = np.asarray(states, dtype=np.float32)
        if state_history.shape != (self.history, 6):
            raise ValueError("states must have shape [3,6]")
        prepared_frames = {
            view: np.asarray(frames[view], dtype=np.uint8) for view in VIEWS
        }
        current_state = state_history[-1]
        # Bind the settled endpoint, selected command and transport readback to
        # one ordinary in-memory id.  This is sequencing state, not a
        # checkpoint hash or cryptographic identity.
        round_id = f"r-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        normalized_states = (state_history - self.state_center) / self.state_scale
        # V12.4 still supplies the frozen ViT/bridge/LeWM carrier. Once
        # Prod-V13 has a physical current-tip measurement, observe_grounding
        # does not execute the old V12 pixel predictor at all.
        if current_fingertips is None:
            # Keep existing planner subclasses and offline replay fixtures
            # source-compatible.  The new keyword is only meaningful when a
            # live Prod-V13 measurement is actually present.
            grounding = self.observe_grounding(prepared_frames, state_history)
        else:
            grounding = self.observe_grounding(
                prepared_frames,
                state_history,
                current_fingertips=current_fingertips,
            )
        nominal = grounding.nominal
        correction_seed = grounding.correction_seed
        corrected = grounding.corrected
        valid = grounding.valid
        contexts = grounding.contexts
        raw_contexts = grounding.raw_contexts
        self._last_grounding = grounding
        v165 = self.v165_runtime_adapter
        if v165 is not None:
            if grounding.full256_tokens is None:
                raise RuntimeError("V16.5 grounding did not expose shared full256 tokens")
            v165.prepare_round(
                full256_by_view=grounding.full256_tokens,
                view_valid={
                    view: bool(np.all(valid[view][-1])) for view in VIEWS
                },
                current_state=current_state,
                past_raw_actions=past_raw_actions,
            )
        v168 = self.v168_runtime_adapter
        if v168 is not None:
            if grounding.full256_tokens is None:
                raise RuntimeError("V16.8 grounding did not expose shared full256 tokens")
            v168.prepare_round(
                full256_by_view=grounding.full256_tokens,
                view_valid={
                    view: bool(np.all(valid[view][-1])) for view in VIEWS
                },
                current_state_deg=current_state,
            )
        v170 = self.v170_runtime_adapter
        if v170 is not None:
            try:
                v17_pixel_source = V17_PIXEL_FM_SOURCES.get(
                    v170.fcpg.model_family
                )
                v17_pixel_candidate = bool(
                    v170.mode == "candidate"
                    and v17_pixel_source is not None
                )
                v174_runtime = v170.fcpg.model_family == "v17.4"
                if grounding.full256_tokens is None:
                    raise RuntimeError(
                        "V17.0 grounding did not expose shared full256 tokens"
                    )
                v170_measurement = v170.prepare_endpoint(
                    rgb_by_view={
                        view: prepared_frames[view][-1] for view in VIEWS
                    },
                    frozen_v12_tokens_by_view=grounding.full256_tokens,
                    nominal_fm_px_by_view={
                        view: np.clip(
                            np.asarray(
                                (
                                    correction_seed[view][-1, 1:3]
                                    if v174_runtime
                                    else nominal[view][-1, 1:3]
                                ),
                                dtype=np.float32,
                            ),
                            0.0,
                            223.0,
                        )
                        for view in VIEWS
                    },
                    view_valid={
                        view: bool(
                            v165_rgb_view_nonblack(prepared_frames[view][-1])
                            and np.all(
                                self.planning_geometry.content_mask(
                                    np.asarray(
                                        (
                                            correction_seed[view][-1, 1:3]
                                            if v174_runtime
                                            else nominal[view][-1, 1:3]
                                        ),
                                        dtype=np.float32,
                                    )
                                )
                            )
                        )
                        for view in VIEWS
                    },
                    current_state_deg=current_state,
                    round_id=round_id,
                )
                for view_index, view in enumerate(VIEWS):
                    model_anchors = np.stack(
                        (
                            v170_measurement.midpoint_px[view_index],
                            v170_measurement.fm_px[view_index, 0],
                            v170_measurement.fm_px[view_index, 1],
                        )
                    ).astype(np.float32)
                    model_valid = np.concatenate(
                        (
                            np.asarray(
                                [bool(v170_measurement.valid_fm[view_index].all())]
                            ),
                            np.asarray(
                                v170_measurement.valid_fm[view_index], dtype=bool
                            ),
                        )
                    )
                    model_valid &= self.planning_geometry.content_mask(model_anchors)
                    if v17_pixel_candidate:
                        corrected[view][-1] = model_anchors
                        valid[view][-1] = model_valid
                        grounding.diagnostics[view].update(
                            {
                                "current_tip_authority": v17_pixel_source,
                                "v17_visual_control_influence": True,
                            }
                        )
                        if not bool(model_valid.all()):
                            grounding.blocked_reasons = tuple(
                                dict.fromkeys(
                                    (
                                        *grounding.blocked_reasons,
                                        f"v171_{view}_measurement_out_of_content",
                                    )
                                )
                            )
                    probability = (
                        v170_measurement.projection_invalid_probability
                        if v170_measurement.projection_invalid_probability is not None
                        else v170_measurement.occlusion_probability
                    )
                    grounding.diagnostics[view]["v170"] = {
                        "anchor_authority": v17_pixel_candidate,
                        "geometry_anchor_authority": not v17_pixel_candidate,
                        "geometry_authority_source": (
                            None
                            if v17_pixel_candidate
                            else "calibrated_fk_projection"
                        ),
                        "measurement_source": (
                            v17_pixel_source
                            if v17_pixel_candidate
                            else "v17_0_query_transformer_diagnostic"
                        ),
                        "visual_control_influence": v17_pixel_candidate,
                        "mode": v170.mode,
                        "model_family": v170.fcpg.model_family,
                        "endpoint": v170_measurement.endpoint,
                        "ordered_fm_px": v170_measurement.fm_px[view_index].copy(),
                        "midpoint_px": v170_measurement.midpoint_px[view_index].copy(),
                        "valid_fm": v170_measurement.valid_fm[view_index].copy(),
                        "model_anchors_in_content": bool(model_valid.all()),
                        "probability_semantics": (
                            v170_measurement.probability_semantics
                        ),
                        "probability": probability[view_index].copy(),
                    }
                if v17_pixel_candidate:
                    grounding.blocked_reasons = tuple(
                        reason
                        for reason in grounding.blocked_reasons
                        if reason != "column_live_anchor_out_of_support"
                    )
                self._last_v170_shadow = {
                    "endpoint": v170_measurement.endpoint,
                    "fm_px": v170_measurement.fm_px.tolist(),
                    "midpoint_px": v170_measurement.midpoint_px.tolist(),
                    "valid_fm": v170_measurement.valid_fm.tolist(),
                    "probability_semantics": v170_measurement.probability_semantics,
                    "probability": probability.tolist(),
                    "transition_consumed": v170_measurement.transition_consumed,
                    "window_completed": v170_measurement.window_completed,
                    "commands_sent": 0,
                    "decision_influence": bool(v170.decision_influence),
                    "fcpg_anchor_authority": v17_pixel_candidate,
                    "geometry_anchor_authority": not v17_pixel_candidate,
                }
                self._last_v170_shadow_error = None
            except Exception as exc:
                self._last_v170_shadow = None
                self._last_v170_shadow_error = f"{type(exc).__name__}: {exc}"
                if v170.mode == "candidate":
                    grounding.blocked_reasons = tuple(
                        dict.fromkeys(
                            (*grounding.blocked_reasons, "v170_measurement_failed")
                        )
                    )
        # Settle the previous round's expectation against what arrived. Done
        # here, before anything is planned, so the residual is measured on the
        # observation and not on anything derived from it.
        self._last_grounding = grounding
        self._settle_world_model_expectation(raw_contexts)
        blocked = list(grounding.blocked_reasons)
        measurement_reliable = not grounding.blocked_reasons and all(
            bool(np.all(valid[view][-1])) for view in VIEWS
        )
        if self.config.control_mode == "anchor_trajectory":
            return self._plan_anchor_trajectory(
                prepared_frames,
                state_history,
                grounding,
                past_raw_actions=past_raw_actions,
                causal_memory=causal_memory,
                measurement_reliable=measurement_reliable,
            )

        column_pickup = self.ground_object(
            "pickup",
            "column",
            prepared_frames["column"][-1],
            gripper_center_px=corrected["column"][-1, 0],
        )
        self._hand_pickup_description_to_wrist(
            column_pickup, prepared_frames["column"][-1]
        )
        wrist_pickup = self.ground_object(
            "pickup",
            "wrist",
            prepared_frames["wrist"][-1],
            gripper_center_px=corrected["wrist"][-1, 0],
        )
        drop_zone = self.ground_object(
            "drop_zone",
            "column",
            prepared_frames["column"][-1],
            gripper_center_px=corrected["column"][-1, 0],
        )
        drop_zone = annotate_drop_phase_gate(
            drop_zone,
            gripper_center_px=corrected["column"][-1, 0],
            min_margin=self.config.single_view_min_margin,
        )
        if shadow_enabled(self):
            drop_centroid = None
            if drop_zone.get("visible") and drop_zone.get("centroid_px") is not None:
                drop_centroid = np.asarray(drop_zone["centroid_px"], dtype=np.float64)
            observer_diag = getattr(self, "_observer_last_diagnostics", {}) or {}
            observer_ready = bool(
                getattr(
                    getattr(self, "_recursive_observer", None),
                    "state",
                    None,
                )
                and getattr(self._recursive_observer.state, "updates", 0) >= 3
                and bool(observer_diag.get("accepted"))
            )
            drop_reproj = maybe_register_or_reproject_drop(
                self,
                centroid_px=drop_centroid if observer_ready else None,
                plane_z_base=float(self.contact_height_m),
                round_id=str(getattr(self, "_last_round_id", "") or ""),
                measurement_source=str(drop_zone.get("source") or ""),
                observer_accepted=bool(observer_diag.get("accepted")),
            )
            if drop_reproj is not None:
                drop_zone = {
                    **drop_zone,
                    "observer_xyz_base": drop_reproj["xyz_base"],
                    "observer_reprojected_px": drop_reproj["centroid_px"],
                    "observer_reprojected_visible": drop_reproj["visible"],
                    "observer_registered_round_id": drop_reproj.get(
                        "registered_round_id"
                    ),
                }
                # FrozenViT and the persistent 2-D identity stay in charge.
                # Converged reprojection may fill only when that measurement
                # is gone. An unconverged T_cb must not move the drop.
                if (
                    drop_reproj["visible"]
                    and observer_ready
                    and str(drop_zone.get("source"))
                    not in {"frozen_vit", "open_vocab", "predicted"}
                ):
                    drop_zone = {
                        **drop_zone,
                        "visible": True,
                        "configured": True,
                        "centroid_px": drop_reproj["centroid_px"],
                        "source": "reprojection",
                        "reason": "observer_reproject",
                    }
        peg_descriptor = _template_tracking_removed("peg", "column")
        peg_registration_current: bool | None = None
        peg_registration_error: str | None = None
        if peg_descriptor.get("configured", False):
            try:
                self._validate_context_registration()
                peg_registration_current = True
            except ContextRegistrationError as exc:
                peg_registration_current = False
                peg_registration_error = str(exc)
                blocked.append("peg_registration_contract_stale")
        peg_descriptor = {
            **peg_descriptor,
            "registration": self.context_registration,
            "clearance_m": self.config.peg_clearance_m,
            "registration_current": peg_registration_current,
            "registration_error": peg_registration_error,
        }

        gripper_column = corrected["column"][-1, 0]
        current_gripper_xyz = np.asarray(
            self.kinematics.forward(current_state), dtype=np.float32
        )
        pickup_distance = (
            float(
                np.linalg.norm(
                    gripper_column
                    - np.asarray(column_pickup["centroid_px"], dtype=np.float32)
                )
            )
            if column_pickup.get("visible", False)
            else None
        )
        column_transform = self._column_applied_transform_px()
        stored_pulse = (getattr(self, "live_anchor_transforms_px", None) or {}).get(
            "column"
        )
        if (
            getattr(self.config, "require_live_column_calibration", False)
            and column_transform is None
            and stored_pulse is None
        ):
            blocked.append("column_live_calibration_required")
        pickup_raw_backprojection_xyz: np.ndarray | None = None
        drop_raw_backprojection_xyz: np.ndarray | None = None
        pickup_relative_target_xyz: np.ndarray | None = None
        drop_relative_target_xyz: np.ndarray | None = None
        pickup_target_xyz: np.ndarray | None = None
        drop_target_xyz: np.ndarray | None = None
        pickup_planar_error_m: float | None = None
        drop_planar_error_m: float | None = None
        peg_base_xyz: np.ndarray | None = None
        if column_pickup.get("visible", False):
            try:
                pickup_raw_backprojection_xyz = column_live_pixel_to_base_xyz(
                    self.fits["column"],
                    np.asarray(column_pickup["centroid_px"], dtype=np.float32),
                    z_base=self.contact_height_m,
                    live_transform_px=column_transform,
                )
                pickup_px = np.asarray(
                    column_pickup["centroid_px"], dtype=np.float32
                )
                held_pickup_goal_xyz = self._held_pickup_goal_xyz(
                    pickup_px, current_gripper_xyz
                )
                pickup_standoff = self._descent_standoff.get("pickup")
                pickup_hysteresis = float(
                    getattr(self.config, "column_descent_hysteresis_band", 0.0)
                )
                (
                    pickup_target_xyz,
                    pickup_relative_target_xyz,
                    pickup_planar_error_m,
                ) = bounded_column_servo_target_xyz(
                    self.fits["column"],
                    gripper_column,
                    pickup_px,
                    current_gripper_xyz,
                    target_z_base=self.contact_height_m,
                    max_planar_step_m=self.config.max_column_servo_step_m,
                    descent_radius_m=self.config.column_descent_radius_m,
                    standoff_z_base=self.transport_height_m,
                    live_transform_px=column_transform,
                    max_z_step_m=self.config.max_column_z_step_m,
                    held_relative_target_xyz=held_pickup_goal_xyz,
                    servo_gain=self._live_column_servo_gain(),
                    previously_standoff=pickup_standoff,
                    hysteresis_band=pickup_hysteresis,
                )
                self._descent_standoff["pickup"] = descent_uses_standoff(
                    pickup_planar_error_m,
                    self.config.column_descent_radius_m,
                    previously_standoff=pickup_standoff,
                    hysteresis_band=pickup_hysteresis,
                )
                self._commit_pickup_goal(
                    pickup_px,
                    pickup_relative_target_xyz,
                    current_gripper_xyz,
                    held=held_pickup_goal_xyz is not None,
                )
            except ValueError:
                blocked.append("pickup_backprojection_invalid")
        wrist_pickup = self._session_ground_wrist_jaw(
            wrist_pickup,
            column_pickup,
            wrist_frame=prepared_frames["wrist"][-1],
            gripper_center_px=corrected["wrist"][-1, 0],
            wrist_valid=bool(np.all(valid["wrist"][-1])),
            planar_error_m=pickup_planar_error_m,
        )
        # Eye-in-hand final-approach correction: the column view aims, the
        # wrist view corrects. Gated on a margin-confident wrist template
        # match (continuity-supported sub-margin matches were measured up to
        # 47px wrong) and the previous round's phase; every gate failure
        # yields a bit-identical plan.
        wrist_correction_applied = False
        # Observability only. The wrist source reaching this gate has already
        # been rewritten to ``frozen_vit`` by the object-belief layer, which
        # makes the margin branch below unconditional, so nothing here records
        # why a round did or did not correct. Read the same values the gate
        # reads, without touching its short-circuit order.
        wrist_correction_math: dict[str, Any] = {}
        wrist_correction_gates = {
            "has_pickup_target": pickup_target_xyz is not None,
            "phase_approach": self.phase_tracker.phase == ManipulationPhase.APPROACH,
            "wrist_visible": bool(wrist_pickup.get("visible", False)),
            "source": wrist_pickup.get("source"),
            "source_admitted": wrist_pickup.get("source")
            in {"frozen_vit", "appearance_template"},
            "margin": (
                float(wrist_pickup["margin"])
                if wrist_pickup.get("margin") is not None
                else None
            ),
            "margin_checked": wrist_pickup.get("source") != "frozen_vit",
            "score": (
                float(wrist_pickup["score"])
                if wrist_pickup.get("score") is not None
                else None
            ),
            "has_relative_centroid": wrist_pickup.get("relative_centroid_px")
            is not None,
            "wrist_anchors_valid": bool(np.all(valid["wrist"][-1])),
        }
        if (
            self._wrist_correction_allowed()
            and pickup_target_xyz is not None
            and self.phase_tracker.phase == ManipulationPhase.APPROACH
            and wrist_pickup.get("visible", False)
            and wrist_pickup.get("source") in {"frozen_vit", "appearance_template"}
            and (
                wrist_pickup.get("source") == "frozen_vit"
                or float(wrist_pickup.get("margin") or 0.0)
                >= self.config.wrist_correction_min_margin
            )
            and wrist_pickup.get("relative_centroid_px") is not None
            and bool(np.all(valid["wrist"][-1]))
        ):
            correction = wrist_tip_similarity_correction(
                corrected["wrist"][-1],
                self.kinematics.forward_anchors(current_state),
                np.asarray(
                    wrist_pickup["relative_centroid_px"], dtype=np.float32
                ),
                gain=self.config.wrist_correction_gain,
                deadband_px=self.config.wrist_correction_deadband_px,
                max_correction_m=self.config.max_wrist_correction_m,
                diagnostics=wrist_correction_math,
            )
            if correction is not None:
                pickup_target_xyz = pickup_target_xyz.copy()
                pickup_target_xyz[:2] += correction
                pickup_relative_target_xyz = (
                    pickup_relative_target_xyz.copy()
                )
                pickup_relative_target_xyz[:2] += correction
                wrist_correction_applied = True
        wrist_correction_record = {
            "applied": bool(wrist_correction_applied),
            "enabled": self._wrist_correction_allowed(),
            "gate_passed": bool(wrist_correction_math),
            "correction_xy_m": (
                np.asarray(correction, dtype=np.float32).tolist()
                if wrist_correction_applied
                else None
            ),
            "deadband_px": float(self.config.wrist_correction_deadband_px),
            "max_correction_m": float(self.config.max_wrist_correction_m),
            "gain": float(self.config.wrist_correction_gain),
            **wrist_correction_gates,
            **wrist_correction_math,
        }
        column_transform_applied = bool(
            getattr(self, "_live_anchor_transform_applied", {}).get("column", False)
        )
        servo_scale = self._measure_servo_scale(
            gripper_anchor_px=gripper_column,
            current_gripper_xyz=current_gripper_xyz,
            planning_target_xyz=pickup_target_xyz,
            live_transform_applied=column_transform_applied,
        )
        if drop_zone.get("visible", False):
            try:
                drop_px = np.asarray(drop_zone["centroid_px"], dtype=np.float32)
                drop_raw_backprojection_xyz = column_live_pixel_to_base_xyz(
                    self.fits["column"],
                    drop_px,
                    z_base=self.release_height_m,
                    live_transform_px=column_transform,
                )
                held_drop_goal_xyz = self._held_drop_goal_xyz(
                    drop_px, current_gripper_xyz
                )
                (
                    drop_target_xyz,
                    drop_relative_target_xyz,
                    drop_planar_error_m,
                ) = bounded_column_servo_target_xyz(
                    self.fits["column"],
                    gripper_column,
                    drop_px,
                    current_gripper_xyz,
                    target_z_base=self.release_height_m,
                    max_planar_step_m=self.config.max_column_servo_step_m,
                    descent_radius_m=self.config.column_descent_radius_m,
                    standoff_z_base=None,
                    live_transform_px=column_transform,
                    max_z_step_m=self.config.max_column_z_step_m,
                    held_relative_target_xyz=held_drop_goal_xyz,
                    servo_gain=self._live_column_servo_gain(),
                )
                self._commit_drop_goal(
                    drop_px,
                    drop_relative_target_xyz,
                    current_gripper_xyz,
                    held=held_drop_goal_xyz is not None,
                )
            except ValueError:
                blocked.append("drop_zone_backprojection_invalid")
        if peg_descriptor.get("configured", False):
            if peg_descriptor.get("registration_current") is False:
                pass
            elif not peg_descriptor.get("visible", False):
                if self.config.experiment_method == "full":
                    blocked.append("peg_not_visible")
            else:
                try:
                    peg_base_xyz = column_live_pixel_to_base_xyz(
                        self.fits["column"],
                        np.asarray(
                            peg_descriptor["centroid_px"], dtype=np.float32
                        ),
                        z_base=self.contact_height_m,
                        live_transform_px=column_transform,
                    )
                    peg_descriptor = {
                        **peg_descriptor,
                        "base_xyz": peg_base_xyz.copy(),
                        "task_plane_z_m": self.contact_height_m,
                    }
                except ValueError:
                    if self.config.experiment_method == "full":
                        blocked.append("peg_backprojection_invalid")
        pickup_distance_m = (
            float(np.linalg.norm(current_gripper_xyz - pickup_relative_target_xyz))
            if pickup_relative_target_xyz is not None
            else None
        )
        drop_distance_m = (
            float(np.linalg.norm(current_gripper_xyz - drop_relative_target_xyz))
            if drop_relative_target_xyz is not None
            else None
        )
        drop_distance = (
            float(
                np.linalg.norm(
                    gripper_column
                    - np.asarray(drop_zone["centroid_px"], dtype=np.float32)
                )
            )
            if drop_zone.get("visible", False)
            else None
        )
        wrist_rel = wrist_pickup.get("relative_centroid_px")
        wrist_distance = (
            float(np.linalg.norm(np.asarray(wrist_rel, dtype=np.float32)))
            if wrist_pickup.get("visible", False) and wrist_rel is not None
            else None
        )
        pickup_inside_drop = False
        if column_pickup.get("visible", False) and drop_zone.get("visible", False):
            bbox = drop_zone.get("bbox_xyxy")
            if bbox is None:
                width = float(drop_zone.get("width_px") or drop_zone.get("length_px") or 16.0)
                cx, cy = map(float, drop_zone["centroid_px"])
                half = 0.5 * max(width, 8.0)
                bbox = (cx - half, cy - half, cx + half, cy + half)
            pickup_inside_drop = gripper_inside_bbox(
                column_pickup.get("centroid_px"), bbox
            )
        # Single-view target visibility logic: planning requires a reliable
        # column target (for 3D backprojection), but wrist is optional (used
        # for attachment confirmation and eye-in-hand correction). Reliability
        # is assessed by match quality (margin, source).
        visibility_result = compute_pickup_visible_with_fallback(
            column_pickup,
            wrist_pickup,
            min_margin=self.config.single_view_min_margin,
            allow_held=self.config.single_view_allow_held,
        )
        pickup_visible = visibility_result["pickup_visible"]
        # Attachment is the FCPG three-anchor closing corridor, not a
        # gripper-joint stall or column 28px/18mm gate. Wrist proximity is
        # still recorded; it does not decide held vs empty-bite.
        attachment_distance = wrist_distance
        three_point = self._three_point_closure(
            corrected,
            valid,
            column_pickup,
            wrist_pickup,
            gripper_position=float(current_state[-1]),
        )
        observer_diag = getattr(self, "_observer_last_diagnostics", {}) or {}
        legacy_sigma_applies = (
            current_fingertips is None and self.v165_runtime_adapter is None
        )
        phase = self.phase_tracker.update(
            PhaseEvidence(
                gripper_position=float(current_state[-1]),
                gripper_to_pickup_px=pickup_distance,
                gripper_to_drop_px=drop_distance,
                gripper_to_pickup_m=pickup_distance_m,
                gripper_to_drop_m=drop_distance_m,
                wrist_pickup_distance_px=attachment_distance,
                pickup_visible=pickup_visible,
                drop_zone_visible=bool(drop_zone.get("visible", False)),
                pickup_inside_drop_zone=pickup_inside_drop,
                three_point_verdict=str(three_point["verdict"]),
                drop_phase_gate_reliable=bool(
                    drop_zone.get("phase_gate_reliable", False)
                ),
                along_close_sigma_m=(
                    observer_diag.get("along_close_sigma_m")
                    if legacy_sigma_applies
                    else None
                ),
                max_sigma_m=(close_sigma_limit(self) if legacy_sigma_applies else None),
            )
        )
        if phase != ManipulationPhase.TRANSPORT:
            self._transport_squeeze_chunks = 0
        if phase in {ManipulationPhase.TRANSPORT, ManipulationPhase.DONE}:
            # Success clears the grasp-retry budget; APPROACH does not, so
            # regression retries stay counted across re-aim cycles.
            self._grasp_empty_close_events = 0
            self._grasp_recovery_engaged = False
        if phase in {ManipulationPhase.DONE, ManipulationPhase.FAULT}:
            blocked.append(f"phase_{phase.value}")
        # Approaching with closed jaws cannot end in a grasp (the gripper
        # calibration deliberately parks the jaws closed, and homing is not
        # guaranteed to have run). Instead of blocking on it, the APPROACH
        # candidate branch swaps in arm-frozen open-gripper prefixes until
        # the jaws are task-ready, so a single auto-run recovers by itself.
        approach_needs_open = exclusive_phase_gripper_open(
            phase,
            current_gripper=float(current_state[5]),
            gripper_empty_closed=float(self.thresholds.gripper_empty_closed),
            empty_bite_reapproach=self.phase_tracker.empty_bite_reapproach,
            three_point_verdict=str(three_point["verdict"]),
        )

        target_kind = target_role_for_phase(phase)
        target_descriptor: dict[str, Any] | None = None
        primary_view: str = "neither"
        if target_kind == "pickup":
            # Single-view target gate: prefer column, allow column-only when
            # wrist is occluded (common during grasp). Wrist-only not yet
            # supported (requires wrist-based xyz computation).
            target_descriptor, primary_view = select_primary_target_view(
                column_pickup, wrist_pickup, target_kind="pickup"
            )
            target_xyz = pickup_target_xyz
            relative_target_xyz = pickup_relative_target_xyz
            raw_backprojection_xyz = pickup_raw_backprojection_xyz
            target_planar_error_m = pickup_planar_error_m
        elif target_kind == "drop_zone":
            target_descriptor, primary_view = select_primary_target_view(
                drop_zone, {}, target_kind="drop_zone"
            )
            target_xyz = drop_target_xyz
            relative_target_xyz = drop_relative_target_xyz
            raw_backprojection_xyz = drop_raw_backprojection_xyz
            target_planar_error_m = drop_planar_error_m
        else:
            target_xyz = None
            relative_target_xyz = None
            raw_backprojection_xyz = None
            target_planar_error_m = None

        # Apply the single-view visibility gate
        should_block, block_reason = should_block_on_target_visibility(
            target_descriptor, target_xyz, target_kind=target_kind, primary_view=primary_view
        )
        if should_block:
            blocked.append(block_reason)
            target_px = np.asarray([np.nan, np.nan], dtype=np.float32)
        else:
            target_px = np.asarray(target_descriptor["centroid_px"], dtype=np.float32)


        candidate_reports: list[dict[str, Any]] = []
        shadow_candidate_chunks: np.ndarray | None = None
        selected_index: int | None = None
        selected_chunk: np.ndarray | None = None
        selected_prefix: np.ndarray | None = None
        selected_prediction: np.ndarray | None = None
        selected_commanded: np.ndarray | None = None
        selected_prediction_px: np.ndarray | None = None
        selected_command_px: np.ndarray | None = None
        selected_prediction_anchor_px: np.ndarray | None = None
        selected_command_anchor_px: np.ndarray | None = None
        selected_anchor_paths_px: dict[str, dict[str, np.ndarray]] = {}
        pending_expectation: tuple[Any, ...] | None = None
        rollout_variant = "not_executed"
        memory_token_count = 0
        rank_pixel_view = "none"
        rank_pixel_weight = 0.0
        # No target or a terminal phase skips the whole ranking block below,
        # and the telemetry is still assembled afterwards.
        closure: dict[str, Any] = {}

        if (
            np.all(np.isfinite(target_px))
            and target_xyz is not None
            and np.all(np.isfinite(target_xyz))
            and phase not in {
                ManipulationPhase.DONE,
                ManipulationPhase.FAULT,
            }
        ):
            bank_frames = getattr(getattr(self.primary, "data", None), "frames", None)
            if (
                wrist_pickup.get("visible", False)
                and bank_frames is not None
                and getattr(self, "bank_wrist_gripper_px", None) is not None
            ):
                bank_object_distance = demonstration_pickup_phase_distances(
                    wrist_pickup,
                    np.asarray(bank_frames[self.bank_rows]),
                    gripper_centers_px=self.bank_wrist_gripper_px,
                )
                extra_weight = self.config.object_phase_weight
            else:
                bank_object_distance = np.zeros(len(self.bank_rows), dtype=np.float32)
                extra_weight = 0.0
            retrieval = retrieve_diverse_candidates(
                np.concatenate((contexts["wrist"][-1], contexts["column"][-1])),
                current_state,
                np.concatenate((self.primary_bank, self.secondary_bank), axis=1),
                self.bank_states,
                self.bank_rows,
                self.bank_episodes,
                state_scale=self.state_scale,
                per_episode=self.config.candidates_per_episode,
                minimum_row_gap=self.config.candidate_min_gap,
                state_weight=self.config.state_retrieval_weight,
                extra_distance=bank_object_distance,
                extra_weight=extra_weight,
            )
            bank_index = {int(row): index for index, row in enumerate(self.bank_rows)}
            source_indices = np.asarray(
                [bank_index[int(row)] for row in retrieval.rows], dtype=np.int64
            )
            retrieval_prefixes = rebase_action_prefixes(
                self.bank_prefixes[source_indices],
                self.bank_states[source_indices],
                current_state,
            )
            if phase in {
                ManipulationPhase.APPROACH,
                ManipulationPhase.TRANSPORT,
            }:
                # Demonstration prefixes carry the demo's gripper column; a
                # demo may open and fully re-close inside one prefix, which
                # the endpoint-delta phase mask cannot see (this is how the
                # arm arrived at the object with closed jaws). Neither phase's
                # job includes moving the gripper — hold the live aperture.
                retrieval_prefixes = retrieval_prefixes.copy()
                retrieval_prefixes[:, :, 5] = current_state[5]
            retrieval_cost = np.asarray(
                retrieval.combined_distance, dtype=np.float32
            )
            selected_phase_distance = (
                bank_object_distance[source_indices]
                if phase in {ManipulationPhase.APPROACH, ManipulationPhase.GRASP}
                else np.zeros(len(retrieval_prefixes), dtype=np.float32)
            )
            candidate_metadata: list[dict[str, Any]] = [
                {
                    "source": "retrieved_demonstration",
                    "row": int(row),
                    "episode": int(retrieval.episodes[index]),
                    "frame": int(self.primary.data.frame_indices[int(row)]),
                }
                for index, row in enumerate(retrieval.rows)
            ]
            prefixes = retrieval_prefixes
            candidate_ik_safe = np.ones(len(prefixes), dtype=bool)
            role_prefixes: np.ndarray | None = None
            role_metadata: list[dict[str, Any]] = []
            role_ik_safe: np.ndarray | None = None
            # Same near-box gate as TRANSPORT→RELEASE: open jaws only when the
            # drop track is reliable and the gripper is over the mouth.
            near_box_release = (
                bool(drop_zone.get("visible", False))
                and bool(drop_zone.get("phase_gate_reliable", False))
                and drop_distance is not None
                and drop_distance <= self.thresholds.drop_near_px
                and drop_distance_m is not None
                and drop_distance_m <= self.thresholds.drop_near_m
            )
            release_keep_carry = (
                phase == ManipulationPhase.RELEASE and not near_box_release
            )
            if phase == ManipulationPhase.APPROACH and approach_needs_open:
                # Open in place before approaching: arm frozen, jaws ramp to
                # the task-ready aperture (clamped at the calibrated stop).
                # Empty-bite reapproach uses the same open targets so the
                # closed calibration stop cannot be re-commanded.
                open_source = (
                    "empty_bite_reapproach"
                    if self.phase_tracker.empty_bite_reapproach
                    else "approach_open"
                )
                role_prefixes, role_metadata = generate_gripper_phase_prefixes(
                    current_state,
                    empty_bite_reapproach_open_targets(
                        float(current_state[-1]),
                        gripper_open=float(self.thresholds.gripper_open),
                        calibration_open=float(self.gripper_calibration_open),
                    ),
                    position_lower=SO101_POSITION_LOWER,
                    position_upper=SO101_POSITION_UPPER,
                    horizon_raw=self.horizon_raw,
                    source=open_source,
                )
                role_ik_safe = np.ones(len(role_prefixes), dtype=bool)
            elif phase in {ManipulationPhase.APPROACH, ManipulationPhase.GRASP}:
                servo_prefixes, servo_metadata = generate_fcpg_goal_prefixes(
                    current_state,
                    gripper_column,
                    target_px,
                    self.fits["column"],
                    self.kinematics,
                    contact_height_m=self.contact_height_m,
                    position_lower=SO101_POSITION_LOWER,
                    position_upper=SO101_POSITION_UPPER,
                    horizon_raw=self.horizon_raw,
                    near_target_px=self.thresholds.pickup_near_px,
                    variants=self._live_image_servo_variants(
                        DEFAULT_APPROACH_VARIANTS
                    ),
                    source="pickup",
                    transit_height_m=self.transport_height_m,
                    live_transform_px=column_transform,
                    previously_transit=self._fcpg_transit.get("pickup"),
                    hysteresis_band=float(
                        getattr(self.config, "column_descent_hysteresis_band", 0.0)
                    ),
                )
                if servo_metadata:
                    self._fcpg_transit["pickup"] = bool(
                        servo_metadata[0].get("use_transit")
                    )
                ik_candidates = generate_ik_candidate_prefixes(
                    self.kinematics,
                    current_state,
                    target_xyz,
                    frameskip=self.frameskip,
                    horizon=self.horizon,
                    arrival_raw_steps=self.config.ik_arrival_raw_steps,
                    # 0.7 leaves headroom for allowed live state drift below
                    # the execution-time first-jump guard.
                    max_arrival_step=0.7 * self.config.max_prefix_step_jump,
                    path_specs=realtime_path_specs_for_phase(
                        phase.value,
                        three_point_verdict=str(three_point["verdict"]),
                    ),
                )
                role_prefixes = np.concatenate(
                    (servo_prefixes, ik_candidates.raw_prefixes), axis=0
                )
                role_ik_safe = np.concatenate(
                    (
                        np.ones(len(servo_prefixes), dtype=bool),
                        ik_candidates.converged.copy(),
                    )
                )
                ik_metadata = [
                    {
                        "source": "calibrated_cartesian_ik",
                        "goal_role": "pickup",
                        "variant": name,
                        "target_surface_xyz": ik_candidates.target_surface_xyz,
                        "target_approach_xyz": ik_candidates.target_approach_xyz[index],
                        "terminal_ik_error_m": float(
                            ik_candidates.terminal_ik_error_m[index]
                        ),
                        "ik_converged": bool(ik_candidates.converged[index]),
                    }
                    for index, name in enumerate(ik_candidates.names)
                ]
                role_metadata = [*servo_metadata, *ik_metadata]
                if self.phase_tracker.empty_bite_reapproach:
                    open_prefixes, open_metadata = generate_gripper_phase_prefixes(
                        current_state,
                        empty_bite_reapproach_open_targets(
                            float(current_state[-1]),
                            gripper_open=float(self.thresholds.gripper_open),
                            calibration_open=float(
                                self.gripper_calibration_open
                            ),
                        ),
                        position_lower=SO101_POSITION_LOWER,
                        position_upper=SO101_POSITION_UPPER,
                        horizon_raw=self.horizon_raw,
                        source="empty_bite_reapproach",
                    )
                    role_prefixes = np.concatenate(
                        (open_prefixes, role_prefixes), axis=0
                    )
                    role_ik_safe = np.concatenate(
                        (
                            np.ones(len(open_prefixes), dtype=bool),
                            role_ik_safe,
                        )
                    )
                    role_metadata = [*open_metadata, *role_metadata]
            elif phase == ManipulationPhase.TRANSPORT or release_keep_carry:
                servo_prefixes, servo_metadata = generate_fcpg_goal_prefixes(
                    current_state,
                    gripper_column,
                    target_px,
                    self.fits["column"],
                    self.kinematics,
                    # Carry at the release height: the planner has no obstacle
                    # model, so the carry must clear container walls by
                    # construction and drop the object in from above.
                    contact_height_m=self.release_height_m,
                    position_lower=SO101_POSITION_LOWER,
                    position_upper=SO101_POSITION_UPPER,
                    horizon_raw=self.horizon_raw,
                    near_target_px=self.thresholds.drop_near_px,
                    variants=self._live_image_servo_variants(
                        DEFAULT_TRANSPORT_VARIANTS
                    ),
                    source="drop_zone",
                    transit_height_m=self.release_height_m,
                    live_transform_px=column_transform,
                    previously_transit=self._fcpg_transit.get("drop_zone"),
                    hysteresis_band=float(
                        getattr(self.config, "column_descent_hysteresis_band", 0.0)
                    ),
                )
                if servo_metadata:
                    self._fcpg_transit["drop_zone"] = bool(
                        servo_metadata[0].get("use_transit")
                    )
                ik_candidates = generate_ik_candidate_prefixes(
                    self.kinematics,
                    current_state,
                    target_xyz,
                    frameskip=self.frameskip,
                    horizon=self.horizon,
                    arrival_raw_steps=self.config.ik_arrival_raw_steps,
                    # 0.7 leaves headroom for allowed live state drift below
                    # the execution-time first-jump guard.
                    max_arrival_step=0.7 * self.config.max_prefix_step_jump,
                    path_specs=realtime_path_specs_for_phase(
                        phase.value,
                        three_point_verdict=str(three_point["verdict"]),
                    ),
                )
                role_prefixes = np.concatenate(
                    (servo_prefixes, ik_candidates.raw_prefixes), axis=0
                )
                role_ik_safe = np.concatenate(
                    (
                        np.ones(len(servo_prefixes), dtype=bool),
                        ik_candidates.converged.copy(),
                    )
                )
                ik_metadata = [
                    {
                        "source": "calibrated_cartesian_ik",
                        "goal_role": "drop_zone",
                        "variant": name,
                        "target_surface_xyz": ik_candidates.target_surface_xyz,
                        "target_approach_xyz": ik_candidates.target_approach_xyz[index],
                        "terminal_ik_error_m": float(
                            ik_candidates.terminal_ik_error_m[index]
                        ),
                        "ik_converged": bool(ik_candidates.converged[index]),
                    }
                    for index, name in enumerate(ik_candidates.names)
                ]
                role_metadata = [*servo_metadata, *ik_metadata]
            elif phase == ManipulationPhase.RELEASE:
                # Open only when near_box_release is true (see gate above).
                # Open targets clamp at the calibrated stop, mirroring the
                # GRASP close clamp: without it the release ratchets +2 per
                # round past the physical stop (observed 32.98 vs calibrated
                # open 31.6) in a sustained stall no watchdog can see.
                opened = min(
                    max(
                        self.thresholds.gripper_open,
                        float(current_state[-1]) + 2.0,
                    ),
                    self.gripper_calibration_open + 1.0,
                )
                role_prefixes, role_metadata = generate_gripper_phase_prefixes(
                    current_state,
                    (
                        opened,
                        min(
                            opened + 4.0,
                            self.gripper_calibration_open + 2.0,
                        ),
                        min(
                            float(SO101_POSITION_UPPER[-1] - 2.0),
                            self.gripper_calibration_open + 2.0,
                        ),
                    ),
                    position_lower=SO101_POSITION_LOWER,
                    position_upper=SO101_POSITION_UPPER,
                    horizon_raw=self.horizon_raw,
                    source="release",
                )
                role_ik_safe = np.ones(len(role_prefixes), dtype=bool)

            if role_prefixes is not None:
                if role_ik_safe is None or role_ik_safe.shape != (len(role_prefixes),):
                    raise RuntimeError("role candidate IK safety mask is invalid")
                prefixes = np.concatenate((role_prefixes, prefixes), axis=0)
                candidate_ik_safe = np.concatenate(
                    (role_ik_safe, candidate_ik_safe)
                )
                retrieval_cost = np.concatenate(
                    (
                        np.zeros(len(role_prefixes), dtype=np.float32),
                        retrieval_cost,
                    )
                )
                selected_phase_distance = np.concatenate(
                    (
                        np.zeros(len(role_prefixes), dtype=np.float32),
                        selected_phase_distance,
                    )
                )
                candidate_metadata = [*role_metadata, *candidate_metadata]
            if phase == ManipulationPhase.TRANSPORT or release_keep_carry:
                # Maintain squeeze while carrying: position-hold alone loses
                # grip as the object shifts under motion (observed live: a
                # stalled grasp slipped out during the lift). Every carry
                # candidate ramps the gripper gently below the current
                # aperture — bounded per step, floored above the calibrated
                # stop, and capped so the transport phase mask (|delta|<=5)
                # still holds. Harmless when the jaws are empty.
                # The floor must clear the plan-time joint-limit guard band
                # (limit_guard 3.0 above the hard lower bound): a ramp that
                # enters the band synthesizes candidates validate_command_
                # chunk must reject, and a sole rejected candidate deadlocks
                # the phase (observed live at aperture 2.37).
                squeeze_floor = max(
                    self.gripper_calibration_closed + 1.0,
                    float(SO101_POSITION_LOWER[-1]) + 3.0 + 0.1,
                )
                aperture = float(current_state[5])
                if aperture > squeeze_floor + 0.3:
                    steps = prefixes.shape[1]
                    ramp = np.maximum(
                        aperture - 0.3 * np.arange(1, steps + 1),
                        max(squeeze_floor, aperture - 4.5),
                    ).astype(prefixes.dtype)
                    if self._transport_squeeze_chunks < 2:
                        # Load the grip BEFORE moving: lifting with an
                        # unloaded position-hold is how the object slipped
                        # out. The first transport chunks squeeze in place
                        # with the arm frozen; motion begins once the grip
                        # carries sustained position error.
                        hold = np.repeat(
                            current_state[None, :].astype(prefixes.dtype),
                            steps,
                            axis=0,
                        )[None]
                        hold[0, :, 5] = ramp
                        prefixes = hold
                        candidate_ik_safe = np.ones(1, dtype=bool)
                        retrieval_cost = np.zeros(1, dtype=np.float32)
                        selected_phase_distance = np.zeros(
                            1, dtype=np.float32
                        )
                        candidate_metadata = [
                            {
                                "source": "transport_squeeze",
                                "variant": "hold_and_load",
                            }
                        ]
                    else:
                        prefixes = prefixes.copy()
                        prefixes[:, :, 5] = ramp[None, :]
            (
                prefixes,
                candidate_ik_safe,
                retrieval_cost,
                selected_phase_distance,
                candidate_metadata,
            ) = self._attach_extra_actor_candidates(
                prefixes,
                candidate_ik_safe,
                retrieval_cost,
                selected_phase_distance,
                candidate_metadata,
                current_state=current_state,
                relative_goal_xyz=relative_target_xyz,
                phase=phase,
            )
            if phase is ManipulationPhase.GRASP:
                # The jaw is a dimension of the action, not a script laid over
                # it. The retired ramp overwrote column 5 of *every* candidate,
                # including the V11 actor's own gripper prediction, and freezing
                # the arm to move the jaw made closing cost cartesian progress
                # that a real grasp never has to give up. Instead each actuator
                # arm trajectory is crossed with a few jaw rates and the closure
                # objective picks; the originals survive unchanged, so V11 keeps
                # proposing its own aperture.
                jaw_rates = self._live_grasp_jaw_rates()
                aperture_ramps = grasp_aperture_candidates(
                    float(current_state[5]),
                    steps=int(prefixes.shape[1]),
                    closed_floor=grasp_closed_floor(
                        held=str(three_point.get("verdict")) == "held",
                        empty_closed=max(
                            float(self.gripper_calibration_closed),
                            float(SO101_POSITION_LOWER[-1]),
                        ),
                    ),
                    open_ceiling=min(
                        float(self.gripper_calibration_open),
                        float(SO101_POSITION_UPPER[-1]),
                    ),
                    rates=jaw_rates,
                )
                arm_rows = [
                    index
                    for index, metadata in enumerate(candidate_metadata)
                    if not str(metadata.get("source", "")).startswith("retrieved")
                    and metadata.get("source") != V168_ACTOR_SOURCE
                    and metadata.get("source") != "v17_0_full_history_actor"
                ]
                if aperture_ramps is not None and arm_rows:
                    rows = np.asarray(arm_rows, dtype=int)
                    blocks = []
                    jaw_metadata: list[dict[str, Any]] = []
                    for rate, ramp in zip(jaw_rates, aperture_ramps):
                        block = prefixes[rows].copy()
                        block[:, :, 5] = ramp[None, :].astype(prefixes.dtype)
                        blocks.append(block)
                        jaw_metadata.extend(
                            {
                                **candidate_metadata[index],
                                "variant": (
                                    f"{candidate_metadata[index].get('variant', '')}"
                                    f"+jaw{rate:+.2f}"
                                ),
                                "jaw_rate": float(rate),
                            }
                            for index in arm_rows
                        )
                    prefixes = np.concatenate((prefixes, *blocks), axis=0)
                    tile = len(jaw_rates)
                    candidate_ik_safe = np.concatenate(
                        (candidate_ik_safe, np.tile(candidate_ik_safe[rows], tile))
                    )
                    retrieval_cost = np.concatenate(
                        (retrieval_cost, np.tile(retrieval_cost[rows], tile))
                    )
                    selected_phase_distance = np.concatenate(
                        (
                            selected_phase_distance,
                            np.tile(selected_phase_distance[rows], tile),
                        )
                    )
                    candidate_metadata = [*candidate_metadata, *jaw_metadata]
            candidate_chunks = chunk_and_normalize_actions(
                prefixes,
                frameskip=self.frameskip,
                center=self.action_center,
                scale=self.action_scale,
            )
            past_chunks = self._past_action_chunks(current_state, past_raw_actions)
            self._pending_v11_rank_prefixes = prefixes
            v165 = self.v165_runtime_adapter
            v170 = self.v170_runtime_adapter
            state_fk_runtime = (
                v170
                if v170 is not None and v170.mode == "candidate"
                else v165
            )
            if state_fk_runtime is None:
                (
                    primary_prediction,
                    state_prediction,
                    rollout_variant,
                    memory_token_count,
                ) = self._predict_candidates(
                    # Raw projected CLS, not the FCPG-corrected context: the frozen
                    # dynamics head was trained and externally validated on this
                    # exact representation.
                    raw_contexts["wrist"],
                    normalized_states,
                    past_chunks,
                    candidate_chunks,
                    causal_memory,
                    measurement_reliable=measurement_reliable,
                    column_context=raw_contexts["column"],
                )
                del primary_prediction
                predicted_states = (
                    state_prediction * self.state_scale[None, None]
                    + self.state_center[None, None]
                )
            else:
                # The State/FK model was trained on exact five-command chunks.
                # Its output replaces only the prediction/scoring interface;
                # the commands still traverse every common gate below.
                state_fk_candidates = prefixes[:, : self.config.command_chunk]
                state_fk_contract_safe = (
                    state_fk_runtime.action_contract_fixed_point_mask(
                    current_state=current_state,
                    five_command_candidates=state_fk_candidates,
                    )
                )
                candidate_ik_safe &= state_fk_contract_safe
                # Invalid rows remain in the diagnostic bank, but LeWM must
                # never silently project them and then score a trajectory that
                # differs from the command. Predict hold for those rows and
                # keep the original rows fail-closed via candidate_ik_safe.
                state_fk_model_candidates = state_fk_candidates.copy()
                state_fk_model_candidates[~state_fk_contract_safe] = np.broadcast_to(
                    current_state,
                    (
                        int((~state_fk_contract_safe).sum()),
                        self.config.command_chunk,
                        6,
                    ),
                )
                predicted_states = (
                    state_fk_runtime.predict_candidate_states(
                        current_state=current_state,
                        five_command_candidates=state_fk_model_candidates,
                    )
                )
                rollout_variant = (
                    "v17_0_state_fk_five_command"
                    if state_fk_runtime is v170
                    else "v16_5_state_fk_five_command"
                )
                memory_token_count = 0
            lewm_goal_h, lewm_final_steps = self._lewm_goal_horizon(
                predicted_states.shape[1]
            )
            lewm_goal_states = predicted_states[:, :lewm_goal_h]
            predicted_anchor_pixels = project_future_anchor_pixels(
                predicted_states,
                self.fits["column"],
                self.kinematics,
                current_correction_px=(
                    corrected["column"][-1] - nominal["column"][-1]
                ),
                live_transform_px=column_transform,
            )
            commanded_states = prefixes[:, self.frameskip - 1 :: self.frameskip]
            model_reference_states = (
                commanded_states[:, : predicted_states.shape[1]]
                if state_fk_runtime is None
                else v165_command_reference_states(
                    prefixes,
                    predicted_steps=predicted_states.shape[1],
                )
            )
            commanded_anchor_pixels = project_future_anchor_pixels(
                commanded_states,
                self.fits["column"],
                self.kinematics,
                current_correction_px=(
                    corrected["column"][-1] - nominal["column"][-1]
                ),
                live_transform_px=column_transform,
            )
            predicted_anchor_pixels = self._apply_column_hard_match_residual(
                predicted_anchor_pixels
            )
            commanded_anchor_pixels = self._apply_column_hard_match_residual(
                commanded_anchor_pixels
            )
            predicted_pixels = predicted_anchor_pixels[:, :, 0]
            commanded_pixels = commanded_anchor_pixels[:, :, 0]
            closure_predicted_states = (
                predicted_states
                if state_fk_runtime is None
                else predicted_states[:, -1:]
            )
            closure_predicted_anchor_pixels = (
                predicted_anchor_pixels
                if state_fk_runtime is None
                else predicted_anchor_pixels[:, -1:]
            )
            # The grasp is judged in the wrist, so it should be scored there.
            # The column carries the whole reach, but at grasp range its jaws
            # subtend around a dozen pixels and the fish is a smaller offset
            # than the detector's own noise -- the corridor it yields cannot
            # tell closing on the fish from closing beside it. The wrist
            # resolves the jaws several times wider, and it is where FCPG
            # grounds them. It also loses the pickup under the tips at the last
            # centimetre, so the column stays as the fallback rather than the
            # objective going silent.
            closure = self._closure_objective(
                phase,
                predicted_states=closure_predicted_states,
                commanded_states=commanded_states,
                current_state=current_state,
                nominal=nominal,
                corrected=corrected,
                valid=valid,
                pickups={"wrist": wrist_pickup, "column": column_pickup},
                column_predicted_anchor_px=closure_predicted_anchor_pixels,
                column_commanded_anchor_px=commanded_anchor_pixels,
                three_point=three_point,
                gripper_to_pickup_px=pickup_distance,
                gripper_to_pickup_m=pickup_distance_m,
            )
            closure_cost = closure["closure_cost"]
            commanded_closure_cost = closure["commanded_closure_cost"]
            if phase is ManipulationPhase.GRASP and grasp_should_close_jaws(
                three_point,
                closure,
                gripper_to_pickup_px=pickup_distance,
                gripper_to_pickup_m=pickup_distance_m,
                pickup_near_px=self.thresholds.pickup_near_px,
                pickup_near_m=self.thresholds.pickup_near_m,
            ):
                commanded_closure_cost = prefer_closing_jaw_costs(
                    commanded_closure_cost, candidate_metadata
                )
            closure_weight = closure["closure_weight"]
            # V11.6 ranks on wrist+column. B2 backup stays wrist-only.
            rank_view_for_pixels = pixel_rank_view_for_rollout(rollout_variant)
            allowed_rank_views = rank_target_views(rank_view_for_pixels)
            rank_target_px = (
                wrist_target_px(wrist_pickup)
                if "wrist" in allowed_rank_views
                else None
            )
            n_cand = int(predicted_states.shape[0])
            lewm_pixel_terms: list[np.ndarray] = []
            command_pixel_terms: list[np.ndarray] = []
            used_rank_views: list[str] = []
            rank_pixel_view = "none"
            rank_pixel_weight = ranking_pixel_goal_weight(
                rank_view=rank_view_for_pixels,
                pixel_view=None,
                configured_weight=self.config.pixel_goal_weight,
            )
            if rank_target_px is not None:
                try:
                    wrist_transform = self.live_anchor_transforms_px.get("wrist")
                    predicted_rank_pixels = project_future_anchor_pixels(
                        lewm_goal_states,
                        self.fits["wrist"],
                        self.kinematics,
                        current_correction_px=(
                            corrected["wrist"][-1] - nominal["wrist"][-1]
                        ),
                        live_transform_px=wrist_transform,
                    )[:, :, 0]
                    commanded_rank_pixels = project_future_anchor_pixels(
                        commanded_states,
                        self.fits["wrist"],
                        self.kinematics,
                        current_correction_px=(
                            corrected["wrist"][-1] - nominal["wrist"][-1]
                        ),
                        live_transform_px=wrist_transform,
                    )[:, :, 0]
                    lewm_pixel_terms.append(
                        current_pixel_goal_cost(
                            predicted_rank_pixels,
                            rank_target_px,
                            final_steps=lewm_final_steps,
                            image_size=(224, 224),
                        )
                    )
                    command_pixel_terms.append(
                        current_pixel_goal_cost(
                            commanded_rank_pixels,
                            rank_target_px,
                            final_steps=self.config.goal_final_steps,
                            image_size=(224, 224),
                        )
                    )
                    used_rank_views.append("wrist")
                except ValueError:
                    pass
            if "column" in allowed_rank_views and np.all(np.isfinite(target_px)):
                lewm_pixel_terms.append(
                    current_pixel_goal_cost(
                        predicted_pixels[:, :lewm_goal_h],
                        target_px,
                        final_steps=lewm_final_steps,
                        image_size=(224, 224),
                    )
                )
                command_pixel_terms.append(
                    current_pixel_goal_cost(
                        commanded_pixels,
                        target_px,
                        final_steps=self.config.goal_final_steps,
                        image_size=(224, 224),
                    )
                )
                used_rank_views.append("column")
            if lewm_pixel_terms:
                lewm_pixel_goal_cost = np.mean(np.stack(lewm_pixel_terms, axis=0), axis=0)
                command_pixel_goal_cost = np.mean(
                    np.stack(command_pixel_terms, axis=0), axis=0
                )
                rank_pixel_view = "+".join(used_rank_views)
                rank_pixel_weight = ranking_pixel_goal_weight(
                    rank_view=rank_view_for_pixels,
                    pixel_view=used_rank_views[0],
                    configured_weight=self.config.pixel_goal_weight,
                )
            else:
                lewm_pixel_goal_cost = np.zeros(n_cand, dtype=np.float32)
                command_pixel_goal_cost = np.zeros(n_cand, dtype=np.float32)
            predicted_xyz = self.kinematics.forward(predicted_states)
            commanded_xyz = self.kinematics.forward(commanded_states)
            model_reference_xyz = self.kinematics.forward(model_reference_states)
            initial_target_error_m = float(
                np.linalg.norm(current_gripper_xyz - target_xyz)
            )
            first_chunk_target_error_m = np.linalg.norm(
                commanded_xyz[:, 0] - target_xyz[None], axis=-1
            )
            terminal_target_error_m = np.linalg.norm(
                commanded_xyz[:, -1] - target_xyz[None], axis=-1
            )
            first_chunk_target_progress = np.full(
                len(candidate_metadata), np.nan, dtype=np.float64
            )
            terminal_target_progress = np.full(
                len(candidate_metadata), np.nan, dtype=np.float64
            )
            first_chunk_command_cart_all = cartesian_terminal_goal_cost(
                commanded_xyz[:, :1, :],
                target_xyz,
                final_steps=1,
                scale_m=self.config.cartesian_goal_scale_m,
            )
            command_first_chunk_cartesian_goal_cost = np.asarray(
                first_chunk_command_cart_all, dtype=np.float64
            )
            for index, metadata in enumerate(candidate_metadata):
                if metadata.get("source") != "calibrated_cartesian_ik":
                    continue
                first_progress_m = float(
                    initial_target_error_m - first_chunk_target_error_m[index]
                )
                terminal_progress_m = float(
                    initial_target_error_m - terminal_target_error_m[index]
                )
                receding_horizon_qualified = bool(
                    first_progress_m >= IK_FIRST_CHUNK_PROGRESS_MIN_M
                    and terminal_progress_m > 0.0
                )
                first_chunk_target_progress[index] = first_progress_m
                terminal_target_progress[index] = terminal_progress_m
                metadata.update(
                    {
                        "ik_receding_horizon_qualified": receding_horizon_qualified,
                        "first_chunk_target_error_m": float(
                            first_chunk_target_error_m[index]
                        ),
                        "terminal_command_target_error_m": float(
                            terminal_target_error_m[index]
                        ),
                        "first_chunk_target_progress_m": first_progress_m,
                        "terminal_target_progress_m": terminal_progress_m,
                    }
                )
            # Phase B0 item 4: as with the pixel goal cost, keep the
            # LeWM-derived variant computed for shadow logging while only the
            # commanded variant is allowed to rank candidates under
            # mpc_only_collection.
            lewm_cartesian_goal_cost = cartesian_terminal_goal_cost(
                predicted_xyz[:, :lewm_goal_h],
                target_xyz,
                final_steps=lewm_final_steps,
                scale_m=self.config.cartesian_goal_scale_m,
            )
            command_cartesian_goal_cost = cartesian_terminal_goal_cost(
                commanded_xyz,
                target_xyz,
                final_steps=self.config.goal_final_steps,
                scale_m=self.config.cartesian_goal_scale_m,
            )
            # Shadow only: the same terminal score taken at the end of the chunk
            # that is actually transmitted, so a proposer offering a short chunk
            # and a proposer filling the horizon can be compared afterwards on
            # equal ground. Never ranked on.
            executed_steps = max(1, min(self.config.command_chunk, commanded_xyz.shape[1]))
            prefix_cartesian_goal_cost = cartesian_terminal_goal_cost(
                commanded_xyz[:, :executed_steps, :],
                target_xyz,
                final_steps=min(self.config.goal_final_steps, executed_steps),
                scale_m=self.config.cartesian_goal_scale_m,
            )
            model_command_cost = np.mean(
                np.square(
                    (
                        predicted_states[:, :lewm_goal_h]
                        - model_reference_states[:, :lewm_goal_h]
                    )
                    / self.state_scale[None, None]
                ),
                axis=(1, 2),
            )
            safety = candidate_safety_checks(
                prefixes[:, 0],
                current_state,
                max_joint_jump=self.config.max_first_command_jump,
                max_prefix_step_jump=self.config.max_prefix_step_jump,
                raw_prefixes=prefixes,
                predicted_states=predicted_states,
                position_lower=SO101_POSITION_LOWER,
                position_upper=SO101_POSITION_UPPER,
                prediction_bound_margin=self.config.prediction_bound_margin,
            )
            grasp_recovery_aperture = (
                self.gripper_calibration_closed + 2.0
                if phase == ManipulationPhase.GRASP
                else None
            )
            observer_diag = getattr(self, "_observer_last_diagnostics", {}) or {}
            allow_close = close_allowed_for_pickup(
                three_point_verdict=str(three_point["verdict"]),
                gripper_to_pickup_px=pickup_distance,
                gripper_to_pickup_m=pickup_distance_m,
                pickup_near_px=self.thresholds.pickup_near_px,
                pickup_near_m=self.thresholds.pickup_near_m,
                along_close_sigma_m=(
                    observer_diag.get("along_close_sigma_m")
                    if legacy_sigma_applies
                    else None
                ),
                max_sigma_m=(close_sigma_limit(self) if legacy_sigma_applies else None),
            )
            approach_receding = (
                first_chunk_target_error_m
                > initial_target_error_m + IK_FIRST_CHUNK_PROGRESS_MIN_M
            )
            phase_safe = keep_v11_candidates_phase_safe(
                candidate_phase_mask(
                    prefixes,
                    phase,
                    self.thresholds,
                    current_gripper=float(current_state[5]),
                    recovery_below=grasp_recovery_aperture,
                    empty_bite_reapproach=self.phase_tracker.empty_bite_reapproach,
                    allow_close=allow_close,
                    receding=approach_receding,
                ),
                candidate_metadata,
                prefixes=prefixes,
                allow_close=allow_close,
                receding=approach_receding,
                current_gripper=float(current_state[5]),
                empty_bite_reapproach=self.phase_tracker.empty_bite_reapproach,
            )
            phase_prediction_safe = phase_prediction_bounds_mask(
                predicted_states,
                phase,
                margin=self.config.prediction_bound_margin,
            )
            if phase in {ManipulationPhase.APPROACH, ManipulationPhase.GRASP}:
                phase_distance_safe = (
                    selected_phase_distance <= self.config.max_object_phase_distance
                )
            else:
                selected_phase_distance = np.zeros(len(prefixes), dtype=np.float32)
                phase_distance_safe = np.ones(len(prefixes), dtype=bool)
            prediction_lower_excess = np.max(
                SO101_POSITION_LOWER[None, None]
                - self.config.prediction_bound_margin
                - predicted_states,
                axis=(1, 2),
            )
            prediction_upper_excess = np.max(
                predicted_states
                - SO101_POSITION_UPPER[None, None]
                - self.config.prediction_bound_margin,
                axis=(1, 2),
            )
            prediction_bound_excess = np.maximum(
                0.0,
                np.maximum(prediction_lower_excess, prediction_upper_excess),
            )
            prediction_bound_dimension = np.argmax(
                np.maximum(
                    SO101_POSITION_LOWER[None, None]
                    - self.config.prediction_bound_margin
                    - predicted_states,
                    predicted_states
                    - SO101_POSITION_UPPER[None, None]
                    - self.config.prediction_bound_margin,
                ).max(axis=1),
                axis=1,
            )
            # The training-FK box must cover the operator-selected target: a
            # pickup at the box edge otherwise rejects every candidate that
            # descends to it — and every grasp-phase hold at it. Growth is
            # directional (only toward the target) and hard-capped.
            growth_cap = 0.06
            effective_workspace_lower = np.maximum(
                np.minimum(self.workspace_lower, target_xyz - 0.01),
                self.workspace_lower - growth_cap,
            )
            effective_workspace_upper = np.minimum(
                np.maximum(self.workspace_upper, target_xyz + 0.01),
                self.workspace_upper + growth_cap,
            )
            # Phase B0 item 4: the workspace veto is split into its two
            # pre-existing conjuncts so LeWM influence can be removed without
            # removing workspace safety.  The commanded conjunct bounds the
            # actual servo targets and is enforced unconditionally; the
            # predicted conjunct is the LeWM-derived term and is dropped under
            # mpc_only_collection.  Both use identical bounds, so dropping the
            # predicted term relaxes nothing that commands are not still
            # checked against.
            lewm_predicted_workspace_safe = np.all(
                (predicted_xyz >= effective_workspace_lower[None, None])
                & (predicted_xyz <= effective_workspace_upper[None, None]),
                axis=(1, 2),
            )
            command_workspace_safe = np.all(
                (commanded_xyz >= effective_workspace_lower[None, None])
                & (commanded_xyz <= effective_workspace_upper[None, None]),
                axis=(1, 2),
            )
            workspace_safe = command_workspace_safe & lewm_predicted_workspace_safe
            predicted_workspace_excess = np.maximum(
                0.0,
                np.maximum(
                    effective_workspace_lower[None, None] - predicted_xyz,
                    predicted_xyz - effective_workspace_upper[None, None],
                ),
            ).max(axis=(1, 2))
            commanded_workspace_excess = np.maximum(
                0.0,
                np.maximum(
                    effective_workspace_lower[None, None] - commanded_xyz,
                    commanded_xyz - effective_workspace_upper[None, None],
                ),
            ).max(axis=(1, 2))
            model_command_xyz_gap = np.max(
                np.linalg.norm(
                    predicted_xyz
                    - model_reference_xyz[:, : predicted_xyz.shape[1]],
                    axis=-1,
                ),
                axis=1,
            )
            # Uniform commanded tool-travel bound, family-independent. The
            # per-round caps applied to the servo TARGET do not bound IK
            # clearance/arc offsets or rebased demonstration trajectories;
            # enforce the operator's caps on every candidate's actual
            # commanded travel. Step 0 is the executed chunk's terminal
            # displacement from the current tool position. The 3mm allowance
            # covers the IK terminal tolerance.
            tool_step_xyz = np.diff(
                np.concatenate(
                    (
                        np.broadcast_to(
                            np.asarray(
                                current_gripper_xyz, dtype=commanded_xyz.dtype
                            )[None, None],
                            (len(prefixes), 1, 3),
                        ),
                        commanded_xyz,
                    ),
                    axis=1,
                ),
                axis=1,
            )
            tool_planar_step_m = np.linalg.norm(
                tool_step_xyz[:, :, :2], axis=-1
            )
            tool_z_step_m = np.abs(tool_step_xyz[:, :, 2])
            tool_travel_safe = np.all(
                (
                    tool_planar_step_m
                    <= self.config.max_column_servo_step_m + 3.0e-3
                )
                & (tool_z_step_m <= self.config.max_column_z_step_m + 3.0e-3),
                axis=1,
            )
            chunk_safe = np.zeros(len(prefixes), dtype=bool)
            chunk_reasons: list[tuple[str, ...]] = []
            for index in range(len(prefixes)):
                check = validate_command_chunk(
                    prefixes[index, : self.config.command_chunk],
                    current_state,
                    expected_commands=self.config.command_chunk,
                    max_first_jump=self.config.max_first_command_jump,
                    max_step_jump=self.config.max_prefix_step_jump,
                )
                chunk_safe[index] = check.safe
                chunk_reasons.append(check.reasons)
            if bool(getattr(self.config, "v165_no_contact_approach", False)):
                no_contact_joint_sweeps = conservative_joint_sweeps(
                    current_state,
                    prefixes[:, : self.config.command_chunk],
                )
                no_contact_xyz = tuple(
                    self.kinematics.forward_anchors(path)
                    for path in no_contact_joint_sweeps
                )
                no_contact_safe, no_contact_reasons = no_contact_approach_mask(
                    phase=phase,
                    current_state=current_state,
                    prefixes=prefixes[:, : self.config.command_chunk],
                    swept_xyz=no_contact_xyz,
                    contact_height_m=float(self.contact_height_m),
                )
                chunk_safe &= no_contact_safe
                chunk_reasons = [
                    tuple((*base, *extra))
                    for base, extra in zip(
                        chunk_reasons, no_contact_reasons, strict=True
                    )
                ]
            peg_configured = bool(peg_descriptor.get("configured", False))
            peg_minimum_distance = np.full(
                len(prefixes), np.nan, dtype=np.float32
            )
            peg_per_anchor_distance = np.full(
                (len(prefixes), 3), np.nan, dtype=np.float32
            )
            if not peg_configured:
                peg_safe = np.ones(len(prefixes), dtype=bool)
            elif peg_base_xyz is None:
                peg_safe = np.zeros(len(prefixes), dtype=bool)
            else:
                peg_clearance = self.context_clearance(
                    current_state,
                    prefixes,
                    peg_base_xyz[:2],
                )
                peg_minimum_distance = np.asarray(
                    peg_clearance["minimum_distance_m"], dtype=np.float32
                )
                peg_per_anchor_distance = np.asarray(
                    peg_clearance["per_anchor_minimum_distance_m"],
                    dtype=np.float32,
                )
                peg_safe = np.asarray(peg_clearance["safe"], dtype=bool)
            # Hoisted above the composition call: the same string must reach the
            # decision reasons and the per-candidate report, so it is computed
            # once rather than re-derived per candidate.
            peg_reason = (
                "peg_clearance"
                if peg_base_xyz is not None
                else "peg_context_unavailable"
            )
            # Phase B0 item 5: the decision composition lives in the module-level
            # pure function `compose_candidate_decision` so that the
            # LeWM-independence test exercises the same code this loop runs,
            # rather than a re-implementation of it.
            decision = compose_candidate_decision(
                hard_command_safe=safety.hard_command_safe,
                candidate_ik_safe=candidate_ik_safe,
                phase_safe=phase_safe,
                phase_distance_safe=phase_distance_safe,
                command_workspace_safe=command_workspace_safe,
                tool_travel_safe=tool_travel_safe,
                chunk_safe=chunk_safe,
                chunk_reasons=tuple(chunk_reasons),
                peg_safe=peg_safe,
                peg_reason=peg_reason,
                lewm_predicted_workspace_safe=lewm_predicted_workspace_safe,
                phase_prediction_safe=phase_prediction_safe,
                command_cartesian_goal_cost=command_cartesian_goal_cost,
                lewm_cartesian_goal_cost=lewm_cartesian_goal_cost,
                command_pixel_goal_cost=command_pixel_goal_cost,
                lewm_pixel_goal_cost=lewm_pixel_goal_cost,
                retrieval_cost=retrieval_cost,
                model_command_cost=model_command_cost,
                pixel_goal_weight=rank_pixel_weight,
                retrieval_cost_weight=self.config.retrieval_cost_weight,
                experiment_method=self.config.experiment_method,
                mpc_only_collection=self._lewm_eligibility_disabled(),
                lewm_ranking=not self._lewm_ranking_disabled(),
                closure_cost=commanded_closure_cost,
                closure_cost_weight=closure_weight,
                first_chunk_target_progress_m=first_chunk_target_progress,
                terminal_target_progress_m=terminal_target_progress,
                command_first_chunk_cartesian_goal_cost=(
                    command_first_chunk_cartesian_goal_cost
                ),
                model_command_max_xyz_gap_m=model_command_xyz_gap,
            )
            base_safe = decision.base_safe
            safe = decision.eligible
            common_supervisor_precheck_safe = decision.supervisor_precheck_safe
            current_column_px = np.asarray(
                corrected["column"][-1], dtype=np.float32
            )
            residual = getattr(self, "_column_hard_match_residual_px", None)
            if residual is not None and not getattr(
                self, "_column_tips_are_metal", False
            ):
                current_column_px = apply_column_frame_offset_px(
                    current_column_px, residual
                )
            costs = prefer_sane_column_pred_costs(
                decision.costs,
                safe=safe,
                metadata=candidate_metadata,
                predicted_column_px=predicted_anchor_pixels,
                commanded_column_px=commanded_anchor_pixels,
                target_px=target_px if np.all(np.isfinite(target_px)) else None,
                current_px=current_column_px,
            )
            cartesian_goal_cost = decision.cartesian_goal_cost
            pixel_goal_cost = decision.pixel_goal_cost
            # Recorded, never acted on. Under mpc_only_collection the ranking
            # above already is the world-model-free one, and putting a
            # model-derived number into the report there would break the
            # independence the collection mode exists to guarantee.
            mpc_only_costs = (
                None
                if self._lewm_ranking_disabled()
                else mpc_only_ranking_cost(
                    command_cartesian_goal_cost=command_cartesian_goal_cost,
                    command_pixel_goal_cost=command_pixel_goal_cost,
                    retrieval_cost=retrieval_cost,
                    pixel_goal_weight=rank_pixel_weight,
                    retrieval_cost_weight=self.config.retrieval_cost_weight,
                    closure_cost=commanded_closure_cost,
                    closure_cost_weight=closure_weight,
                )
            )
            prefix_costs = executed_prefix_ranking_cost(
                prefix_cartesian_goal_cost=prefix_cartesian_goal_cost,
                command_pixel_goal_cost=command_pixel_goal_cost,
                retrieval_cost=retrieval_cost,
                pixel_goal_weight=rank_pixel_weight,
                retrieval_cost_weight=self.config.retrieval_cost_weight,
                closure_cost=commanded_closure_cost,
                closure_cost_weight=closure_weight,
            )
            for index, metadata in enumerate(candidate_metadata):
                reasons = list(chunk_reasons[index])
                supervisor_reasons = list(chunk_reasons[index])
                if not safety.hard_command_safe[index]:
                    reasons.append("raw_prefix_safety")
                    supervisor_reasons.append("raw_prefix_safety")
                if not candidate_ik_safe[index]:
                    reasons.append("ik_not_converged")
                    supervisor_reasons.append("ik_not_converged")
                first_progress_value = float(first_chunk_target_progress[index])
                if (
                    np.isfinite(first_progress_value)
                    and first_progress_value < IK_FIRST_CHUNK_PROGRESS_MIN_M
                ):
                    reasons.append("ik_receding_horizon_unqualified")
                    supervisor_reasons.append("ik_receding_horizon_unqualified")
                # Phase 2.5 MPC_ONLY_COLLECTION: skip LeWM rejection in decision reasons
                if not self._lewm_eligibility_disabled():
                    if not phase_prediction_safe[index]:
                        reasons.append("lewm_prediction_bounds")
                        supervisor_reasons.append("lewm_prediction_bounds")
                if not phase_safe[index]:
                    reasons.append("gripper_phase")
                    supervisor_reasons.append("gripper_phase")
                if not phase_distance_safe[index]:
                    reasons.append("object_phase_distance")
                    supervisor_reasons.append("object_phase_distance")
                # Phase B0 item 4: the commanded and predicted workspace vetoes
                # are reported separately so a rejection can be attributed.
                # The supervisor keeps both unconditionally -- it is meant to
                # see what the model objected to even when the model is not
                # permitted to act on it.
                if not command_workspace_safe[index]:
                    reasons.append("command_workspace")
                    supervisor_reasons.append("command_workspace")
                if not lewm_predicted_workspace_safe[index]:
                    supervisor_reasons.append("lewm_prediction_workspace")
                    if not self._lewm_eligibility_disabled():
                        reasons.append("lewm_prediction_workspace")
                if not tool_travel_safe[index]:
                    reasons.append("tool_travel_exceeded")
                    supervisor_reasons.append("tool_travel_exceeded")
                if not peg_safe[index]:
                    supervisor_reasons.append(peg_reason)
                    if self.config.experiment_method == "full":
                        reasons.append(peg_reason)
                candidate_reports.append(
                    {
                        **metadata,
                        "index": index,
                        "cost": float(costs[index]),
                        "goal_cost": float(cartesian_goal_cost[index]),
                        "cartesian_goal_cost": float(cartesian_goal_cost[index]),
                        "pixel_goal_cost": float(pixel_goal_cost[index]),
                        "retrieval_cost": float(retrieval_cost[index]),
                        "model_command_cost": float(model_command_cost[index]),
                        "mpc_only_cost": (
                            None
                            if mpc_only_costs is None
                            else float(mpc_only_costs[index])
                        ),
                        "prefix_cost": float(prefix_costs[index]),
                        "prefix_cartesian_goal_cost": float(
                            prefix_cartesian_goal_cost[index]
                        ),
                        "closure_cost": float(closure_cost[index]),
                        "commanded_closure_cost": float(
                            commanded_closure_cost[index]
                        ),
                        "model_command_max_xyz_gap_m": float(
                            model_command_xyz_gap[index]
                        ),
                        "predicted_workspace_excess_m": float(
                            predicted_workspace_excess[index]
                        ),
                        "commanded_workspace_excess_m": float(
                            commanded_workspace_excess[index]
                        ),
                        "first_chunk_joint_delta": (
                            prefixes[index, self.config.command_chunk - 1]
                            - current_state
                        ),
                        "terminal_joint_state": prefixes[index, -1].copy(),
                        "raw_prefix_max_joint_step": float(
                            np.max(
                                np.abs(
                                    np.diff(
                                        np.concatenate(
                                            (current_state[None], prefixes[index]), axis=0
                                        ),
                                        axis=0,
                                    )
                                )
                            )
                        ),
                        "prediction_bound_excess": float(
                            prediction_bound_excess[index]
                        ),
                        "prediction_bound_dimension": int(
                            prediction_bound_dimension[index]
                        ),
                        "phase_distance": float(selected_phase_distance[index]),
                        "peg_configured": peg_configured,
                        "peg_min_anchor_distance_m": (
                            float(peg_minimum_distance[index])
                            if np.isfinite(peg_minimum_distance[index])
                            else None
                        ),
                        "peg_per_anchor_minimum_distance_m": (
                            peg_per_anchor_distance[index].copy()
                            if np.all(
                                np.isfinite(peg_per_anchor_distance[index])
                            )
                            else None
                        ),
                        "peg_required_clearance_m": self.config.peg_clearance_m,
                        "experiment_method": self.config.experiment_method,
                        "peg_visible_to_candidate_selection": (
                            self.config.experiment_method == "full"
                        ),
                        "method_selection_safe": bool(safe[index]),
                        "common_supervisor_precheck_safe": bool(
                            common_supervisor_precheck_safe[index]
                        ),
                        "common_supervisor_reasons": sorted(
                            set(supervisor_reasons)
                        ),
                        "safe": bool(safe[index]),
                        "reasons": sorted(set(reasons)),
                        "lewm_scored": True,
                        "predicted_terminal_lift_deg": float(
                            predicted_states[index, -1, 1]
                        ),
                        "commanded_terminal_lift_deg": float(
                            commanded_states[index, -1, 1]
                        ),
                        # The jaw is the one dimension the grasp turns on. If the
                        # world model's terminal aperture does not move with the
                        # commanded one, its rollout cannot score a closure and
                        # everything about the grasp is being decided by FK.
                        "predicted_terminal_gripper_deg": float(
                            predicted_states[index, -1, 5]
                        ),
                        "commanded_terminal_gripper_deg": float(
                            commanded_states[index, -1, 5]
                        ),
                    }
                )
            shadow_candidate_chunks = prefixes[
                :, : self.config.command_chunk
            ].copy()
            if np.any(safe):
                selected_index = int(np.argmin(np.where(safe, costs, np.inf)))
                selected_prefix = prefixes[selected_index].copy()
                selected_chunk = selected_prefix[: self.config.command_chunk].copy()
                selected_prediction = predicted_states[selected_index].copy()
                selected_commanded = commanded_states[selected_index].copy()
                selected_prediction_px = predicted_pixels[selected_index].copy()
                selected_command_px = commanded_pixels[selected_index].copy()
                selected_prediction_anchor_px = predicted_anchor_pixels[
                    selected_index
                ].copy()
                selected_command_anchor_px = commanded_anchor_pixels[
                    selected_index
                ].copy()
                selected_anchor_paths_px = self._anchor_paths_px(
                    predicted=selected_prediction,
                    commanded=selected_commanded,
                    nominal=nominal,
                    corrected=corrected,
                )
                if state_fk_runtime is None:
                    pending_expectation = (
                        raw_contexts,
                        normalized_states,
                        past_chunks,
                        candidate_chunks[selected_index],
                    )
            else:
                blocked.append("no_safe_phase_compatible_candidate")

        if pending_expectation is not None:
            self._hold_world_model_expectation(*pending_expectation, round_id=round_id)
        safe_target = (
            target_px
            if np.all(np.isfinite(target_px))
            else np.asarray([-1.0, -1.0], dtype=np.float32)
        )
        guard = PlanningRoundGuard(
            round_id=round_id,
            created_monotonic_s=time.monotonic(),
            state=current_state.copy(),
            target_px=safe_target.copy(),
            target_kind=target_kind or "none",
        )
        plan_model_contract = {
            **self.model_contract,
            **self.live_contract_fields(),
            "rollout_variant": rollout_variant,
            "active_memory_tokens": int(memory_token_count),
            "measurement_reliable": bool(measurement_reliable),
            "current_fingertip_pixel_authority": (
                (
                    V17_PIXEL_FM_SOURCES[
                        self.v170_runtime_adapter.fcpg.model_family
                    ]
                    if (
                        self.v170_runtime_adapter.mode == "candidate"
                        and self.v170_runtime_adapter.fcpg.model_family
                        in V17_PIXEL_FM_SOURCES
                    )
                    else "calibrated_fk_projection"
                )
                if self.v170_runtime_adapter is not None
                else (
                    self.v165_runtime_adapter.anchor_authority
                    if self.v165_runtime_adapter is not None
                    else (
                        "legacy_v12_grounding"
                        if current_fingertips is None
                        else current_fingertips.source
                    )
                )
            ),
            "v12_pixel_anchor_role": (
                "frozen_contextual_token_carrier_for_v17_current_image_reacquisition"
                if self.v170_runtime_adapter is not None
                else (
                    "shared_vit_carrier_for_v16_8_original_fcpg_foundation"
                    if self.v168_runtime_adapter is not None
                    else (
                        "shared_vit_carrier_for_v16_5_session_authority"
                        if self.v165_runtime_adapter is not None
                        else (
                            "legacy_control_fallback_before_prod_v13_h0"
                            if current_fingertips is None
                            else "not_executed_current_geometry_owned_by_prod_v13"
                        )
                    )
                )
            ),
            "v170_runtime": (
                None
                if self.v170_runtime_adapter is None
                else self.v170_runtime_adapter.status()
            ),
            "legacy_observer_sigma_controls_close": bool(
                current_fingertips is None
                and self.v165_runtime_adapter is None
            ),
            **rank_view_contract_fields(
                rank_pixel_view=rank_pixel_view,
                rank_pixel_weight=rank_pixel_weight,
                wrist_fish_grounded=bool(wrist_pickup.get("visible", False)),
            ),
        }
        return RealtimePlan(
            guard=guard,
            phase=phase,
            phase_status=self.phase_tracker.status(),
            current_state=current_state.copy(),
            frames={view: prepared_frames[view].copy() for view in VIEWS},
            nominal=nominal,
            correction_seed=correction_seed,
            corrected=corrected,
            valid=valid,
            grounding_diagnostics=grounding.diagnostics,
            wrist_correction=wrist_correction_record,
            target_descriptors={
                "pickup_column": column_pickup,
                "pickup_wrist": wrist_pickup,
                "drop_zone_column": drop_zone,
                "peg_column": peg_descriptor,
            },
            target_profile=self._target_profile_status(),
            target={
                "kind": target_kind,
                "visible": bool(target_descriptor and target_descriptor.get("visible")),
                "centroid_px": safe_target,
                "base_xyz": (
                    np.asarray(target_xyz, dtype=np.float32)
                    if target_xyz is not None
                    else None
                ),
                "relative_goal_xyz": (
                    np.asarray(relative_target_xyz, dtype=np.float32)
                    if relative_target_xyz is not None
                    else None
                ),
                "raw_backprojection_xyz": (
                    np.asarray(raw_backprojection_xyz, dtype=np.float32)
                    if raw_backprojection_xyz is not None
                    else None
                ),
                "planar_error_m": target_planar_error_m,
                "servo_goal_held": bool(getattr(self, "_pickup_goal_reused", False)),
                "bounded_column_servo": bool(
                    target_planar_error_m is not None
                    and target_planar_error_m > self.config.max_column_servo_step_m
                ),
                "gripper_distance_m": (
                    float(np.linalg.norm(current_gripper_xyz - relative_target_xyz))
                    if relative_target_xyz is not None
                    else None
                ),
                "servo_scale": servo_scale,
                "wrist_pickup_visible": bool(wrist_pickup.get("visible", False)),
                "wrist_pickup_distance_px": wrist_distance,
                "wrist_relative_centroid_px": (
                    np.asarray(
                        wrist_pickup["relative_centroid_px"], dtype=np.float32
                    )
                    if wrist_pickup.get("relative_centroid_px") is not None
                    else None
                ),
                "wrist_fish_grounded": bool(wrist_pickup.get("visible", False)),
                "three_point": {
                    "verdict": three_point.get("verdict"),
                    "target_between_tips": bool(
                        three_point.get("target_between_tips")
                    ),
                    "jaws_collapsed": bool(three_point.get("jaws_collapsed")),
                    "tip_separation_px": three_point.get("tip_separation_px"),
                    "closing_progress": three_point.get("closing_progress"),
                },
                "closure": {
                    "view": closure.get("closure_view"),
                    "reachable": closure.get("closure_reachable"),
                    "weight": float(closure.get("closure_weight", 0.0)),
                    "projected_span_px": closure.get("projected_span_px"),
                    "detected_span_px": closure.get("detected_span_px"),
                    "target_along_px": closure.get("target_along_px"),
                    "target_across_px": closure.get("target_across_px"),
                },
            },
            selected_index=selected_index,
            selected_chunk=selected_chunk,
            selected_prefix=selected_prefix,
            predicted_states=selected_prediction,
            commanded_states=selected_commanded,
            predicted_gripper_px=selected_prediction_px,
            commanded_gripper_px=selected_command_px,
            predicted_anchor_px=selected_prediction_anchor_px,
            commanded_anchor_px=selected_command_anchor_px,
            anchor_paths_px=selected_anchor_paths_px,
            candidate_rows=candidate_reports,
            executable=selected_index is not None and not blocked,
            blocked_reasons=tuple(sorted(set(blocked))),
            model_contract=plan_model_contract,
            world_model_residual=getattr(self, "last_world_model_residual", None),
            candidate_chunks=shadow_candidate_chunks,
        )


__all__ = [
    "LIVE_CONTRACT_HEIGHT_FIELDS",
    "CausalPlanningMemory",
    "GroundingObservation",
    "LiveFingertipMeasurement",
    "MPC_ONLY_INDEPENDENCE_EVIDENCE",
    "MPC_ONLY_INDEPENDENCE_SCOPE",
    "PreparedPlannerConfig",
    "PreparedSO101RealtimePlanner",
    "RealtimePlan",
    "cartesian_terminal_goal_cost",
    "lewm_decision_independence_fields",
    "planning_geometry_contract_fields",
    "bounded_column_servo_target_xyz",
    "column_live_pixel_to_base_xyz",
    "command_prefix_context_clearance",
    "experiment_candidate_selection_mask",
    "live_anchor_transform_applied",
    "descent_uses_standoff",
    "merge_extra_actor_prefixes",
    "plan_json",
    "RANK_VIEW",
    "recursive_anchor_seed",
    "live_fingertip_control_anchors",
]
