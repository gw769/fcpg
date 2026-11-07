"""SO-101 wiring between the live planner and the generic FCPG protocol.

This module is the ONLY place where SO-101 specifics (view names, role names,
RealtimePlan layout, planner method signatures) meet the generic grounding
protocol.  interfaces/, perception/, planning/ and task/ stay free of them, so
retargeting to another robot means writing another bridge, not editing the
protocol.

OWNERSHIP
---------
LeWM remains the sole motion planner.  This bridge may only:

    verify grounding          GroundingVerifier.review
    request an observation    by reporting a verdict, never by moving anything
    register a reference      AutonomousRegistrationPolicy.acquire
    halt                      by returning a verdict the runtime turns into a stop

It contains no joint targets, no trajectories, and no calls into any execution
path.  The structural guarantee is in interfaces/grounding.ObservationRequest:
it carries a sensing goal and no command fields, so a verdict cannot express
motion even if a caller wanted it to.

WHY THE DISTANCE GATE SURVIVES
------------------------------
The simulation study found geometric distance uninformative about whether
grounding is adequate (precision at the base rate, lift ~1.0).  That result
retires distance as a decision rule, not as a feasibility precondition: the
near-field camera must actually be close enough to resolve the target before an
observation can be attempted at all.  So the two conditions are kept separate
and both are required, which is what the deployed code conflated:

    feasibility (distance, aperture)   can an observation be taken now?
    necessity   (uncertainty vs tol)   is an observation needed at all?

MEASUREMENT PROVENANCE
----------------------
    aperture_m              MEASURED   0.05498 m jaw opening
    baseline_sigma_m        MEASURED   0.00882 m far-view planar error, total
                                       against ground truth, so it already
                                       contains the projection fit's residual
    object_width_m          ASSUMED    not yet measured with calipers
    near_sensor_sigma_m     ASSUMED    no near-view accuracy study

The two ASSUMED values are recorded in every interaction log record, so any
result carries its own assumptions rather than relying on this docstring.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from interfaces.grounding import (
    GroundingFeedback,
    GroundingVerdict,
    ProposedAction,
)
from interfaces.hardware import Frame
from interfaces.perception import (
    AppearanceDescriptor,
    TargetRegistrationInterface,
)
from perception.autonomous_registration import (
    AutonomousRegistrationPolicy,
    RegistrationOutcome,
    RegistrationPolicyConfig,
    RegistrationResult,
)
from perception.grounding_verifier import (
    GroundingVerifier,
    SensorCapability,
    VerifierConfig,
    grasp_tolerance_m,
)
from perception.uncertainty import EvidenceUncertaintyEstimator

# SO-101 role and view vocabulary.  Confined to this module by design.
PICKUP_ROLE = "pickup"
NEAR_VIEW = "wrist"
FAR_VIEW = "column"
NEAR_SENSOR_ROLE = "near_field"


@dataclass(frozen=True)
class GroundingBridgeConfig:
    """Physical and modelling inputs.  See module docstring for provenance."""

    # ── measured ────────────────────────────────────────────────────────────
    aperture_m: float = 0.05498
    baseline_sigma_m: float = 0.00882

    # ── assumed; recorded in every log record ───────────────────────────────
    object_width_m: float = 0.020
    near_sensor_sigma_m: float = 0.002

    pad_margin_m: float = 0.002
    safety_factor: float = 1.5
    max_observation_requests: int = 1
    max_correction_m: float | None = None

    # Component weights in metres, applied to dimensionless evidence values.
    #
    # candidate_ambiguity is the only channel with a genuine per-frame signal on
    # this robot: the template tracker emits best and runner-up scores every
    # frame, so it is what makes sigma vary per scene rather than sitting at a
    # population constant.
    #
    # projection_residual is deliberately ABSENT.  baseline_sigma_m was measured
    # as total planar grounding error against ground truth, which already
    # contains whatever the projection fit contributes; adding a residual term
    # would count the same error source twice.  It is also constant per fit, so
    # it could not contribute per-scene variation even if it were free to add.
    component_weights: dict[str, float] = field(
        default_factory=lambda: {
            "candidate_ambiguity": 0.010,
            "appearance_inconsistency": 0.006,
        }
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.object_width_m < self.aperture_m:
            raise ValueError(
                "object_width_m must be positive and narrower than the aperture"
            )
        if self.baseline_sigma_m <= 0:
            raise ValueError("baseline_sigma_m must be positive")

    @property
    def assumptions(self) -> dict[str, Any]:
        """Unmeasured inputs, travelling with every record they influenced."""
        return {
            "object_width_m": {"value": self.object_width_m, "status": "ASSUMED"},
            "near_sensor_sigma_m": {
                "value": self.near_sensor_sigma_m,
                "status": "ASSUMED",
            },
        }

    def grasp_tolerance_m(self) -> float | None:
        """Positional tolerance from jaw geometry; None when no slack exists."""
        try:
            return grasp_tolerance_m(
                aperture_m=self.aperture_m,
                object_width_m=self.object_width_m,
                pad_margin_m=self.pad_margin_m,
            )
        except ValueError:
            return None


class PlannerTargetRegistration(TargetRegistrationInterface):
    """Adapt the live planner's reference store to the generic interface.

    Calls ``planner.set_target_reference`` directly rather than the controller
    method of the same name: the controller method acquires the operation lock,
    and the bridge runs inside ``execute()`` which already holds it.
    """

    def __init__(self, planner: Any) -> None:
        self._planner = planner

    def register(
        self,
        frame: Frame,
        centroid_px: np.ndarray,
        role: str,
        view: str,
        *,
        radius_px: float,
    ) -> str:
        self._planner.set_target_reference(
            role,
            view,
            np.asarray(frame.data, dtype=np.uint8),
            np.asarray(centroid_px, dtype=np.float32),
            radius_px=float(radius_px),
        )
        return f"{role}:{view}"

    def clear(self, target_id: str) -> None:
        role, _, view = target_id.partition(":")
        self._planner.clear_target_reference(role, view or None)

    def is_registered(self, role: str, view: str) -> bool:
        status = self._planner.target_status()
        if not isinstance(status, dict):
            return False
        if view == NEAR_VIEW and role == PICKUP_ROLE:
            return bool(status.get("wrist_pickup_ready"))
        if view == FAR_VIEW and role == PICKUP_ROLE:
            return bool(status.get("pickup_ready"))
        return False

    def list_registered(self) -> dict[str, dict[str, Any]]:
        status = self._planner.target_status()
        return {"planner_target_status": dict(status)} if isinstance(status, dict) else {}


def frame_from_planning_array(
    data: np.ndarray, role: str, *, timestamp_s: float | None = None
) -> Frame:
    """Wrap a planning-frame array as a generic Frame."""
    array = np.asarray(data, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("planning frame must be (H, W, 3)")
    return Frame(
        data=array,
        width=int(array.shape[1]),
        height=int(array.shape[0]),
        timestamp_s=float(time.time() if timestamp_s is None else timestamp_s),
        camera_id=role,
        role=role,
        encoding="bgr",
    )


def extract_evidence(plan: Any, config: GroundingBridgeConfig) -> dict[str, Any]:
    """Collect per-frame grounding evidence from a live plan.

    Only signals the stack actually produces are included.  A missing channel is
    omitted rather than defaulted, because the estimator combines in quadrature
    and a defaulted zero would silently lower the reported sigma.
    """
    evidence: dict[str, Any] = {}
    descriptors = getattr(plan, "target_descriptors", None)
    if not isinstance(descriptors, dict):
        return evidence

    near = descriptors.get(f"{PICKUP_ROLE}_{NEAR_VIEW}")
    far = descriptors.get(f"{PICKUP_ROLE}_{FAR_VIEW}")
    # Near-field wrist evidence wins when the object is already in the jaws.
    # Column-only scores used to treat a visible wrist fish as unseen.
    chosen = None
    if isinstance(near, dict) and near.get("visible"):
        chosen = near
        evidence["wrist_visible"] = True
        relative = near.get("relative_centroid_px")
        if relative is not None:
            offset = np.asarray(relative, dtype=np.float64).reshape(-1)
            if offset.shape == (2,) and np.all(np.isfinite(offset)):
                evidence["wrist_pickup_distance_px"] = float(np.linalg.norm(offset))
    elif isinstance(far, dict) and far.get("visible"):
        chosen = far
        evidence["wrist_visible"] = False
    if isinstance(chosen, dict):
        best = chosen.get("score")
        second = chosen.get("second_score")
        if best is not None and np.isfinite(float(best)):
            evidence["best_score"] = float(best)
            # A negative sentinel means "no second candidate found", which is
            # unambiguous, not maximally ambiguous.
            if second is not None and np.isfinite(float(second)):
                evidence["runner_up_score"] = max(float(second), 0.0)

    # No projection_residual key: see GroundingBridgeConfig.component_weights.
    # The measured baseline already contains the projection fit's contribution,
    # and the residual is constant per fit, so reporting it as evidence would
    # both double-count and add no per-scene information.
    return evidence


def proposed_action_from_plan(
    plan: Any, config: GroundingBridgeConfig
) -> ProposedAction | None:
    """Build the planner's proposal in protocol terms.

    Uses ``raw_backprojection_xyz``: ``base_xyz`` carries an approach standoff in
    Z that contaminates the planar comparison the verifier performs.
    """
    target = getattr(plan, "target", None)
    if not isinstance(target, dict):
        return None
    xyz = target.get("raw_backprojection_xyz")
    if xyz is None:
        xyz = target.get("base_xyz")
    if xyz is None:
        return None
    world = np.asarray(xyz, dtype=np.float64).reshape(-1)
    if world.shape != (3,) or not np.all(np.isfinite(world)):
        return None
    tolerance = config.grasp_tolerance_m()
    if tolerance is None:
        return None
    phase = getattr(plan, "phase", None)
    return ProposedAction(
        target_world_xyz=world,
        tolerance_m=float(tolerance),
        phase=getattr(phase, "value", None) if phase is not None else None,
        metadata={
            "action_type": "grasp",
            "round_id": getattr(getattr(plan, "guard", None), "round_id", None),
            "planar_error_m": target.get("planar_error_m"),
        },
    )


@dataclass
class InteractionRecord:
    """One planner proposal and the grounding response to it.

    This is the closed-loop trace: what the planner wanted, what grounding
    thought of it, whether an observation was requested and what it produced,
    and what the runtime did as a result.
    """

    round_id: str | None
    phase: str | None
    planner_action: dict[str, Any]
    grounding_verdict: str
    uncertainty_m: float
    uncertainty_calibrated: bool
    tolerance_m: float | None
    reason: str | None
    checks: dict[str, Any]
    uncertainty_components: dict[str, float] = field(default_factory=dict)
    """Per-channel evidence values behind the sigma.  Recorded so a reader can
    see WHICH signal drove a verdict, not merely that one was reached."""
    observation_request: dict[str, Any] | None = None
    registration: dict[str, Any] | None = None
    final_action: str = "pending"
    assumptions: dict[str, Any] = field(default_factory=dict)
    timestamp_s: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp_s": round(self.timestamp_s, 6),
            "round_id": self.round_id,
            "phase": self.phase,
            "planner_action": self.planner_action,
            "grounding_verdict": self.grounding_verdict,
            "uncertainty_m": round(self.uncertainty_m, 6),
            "uncertainty_calibrated": self.uncertainty_calibrated,
            "uncertainty_components": {
                k: round(float(v), 6)
                for k, v in sorted(self.uncertainty_components.items())
            },
            "tolerance_m": (
                None if self.tolerance_m is None else round(self.tolerance_m, 6)
            ),
            "reason": self.reason,
            "checks": self.checks,
            "observation_request": self.observation_request,
            "registration": self.registration,
            "final_action": self.final_action,
            "assumptions": self.assumptions,
        }


class GroundingBridge:
    """Run one grounding cycle for a live plan, without commanding motion."""

    def __init__(
        self,
        planner: Any,
        detector: Any,
        config: GroundingBridgeConfig,
        registration_policy_config: RegistrationPolicyConfig,
        *,
        log_path: Path | None = None,
        observe_only: bool = False,
    ) -> None:
        self._planner = planner
        self._config = config
        # observe_only runs detection and every pre-registration check, reports
        # the result, and then declines to commit the reference.  It exists so
        # the feasibility boundary (at what distance can the near camera
        # actually resolve the target?) can be MEASURED across poses without
        # each measurement mutating the reference store and changing what the
        # next measurement sees.  Default False: normal operation registers.
        self._observe_only = bool(observe_only)
        self._registration = PlannerTargetRegistration(planner)
        self._policy = AutonomousRegistrationPolicy(
            detector=detector,
            registration=self._registration,
            config=registration_policy_config,
        )
        self._verifier = GroundingVerifier(
            estimator=EvidenceUncertaintyEstimator(
                baseline_sigma_m=config.baseline_sigma_m,
                component_weights=dict(config.component_weights),
            ),
            config=VerifierConfig(
                safety_factor=config.safety_factor,
                max_observation_requests=config.max_observation_requests,
                max_correction_m=config.max_correction_m,
            ),
            sensors={
                NEAR_SENSOR_ROLE: SensorCapability(
                    role=NEAR_SENSOR_ROLE,
                    achievable_sigma_m=config.near_sensor_sigma_m,
                )
            },
        )
        self._log_path = log_path
        self._records: list[InteractionRecord] = []

    # ── accessors ────────────────────────────────────────────────────────────

    @property
    def records(self) -> tuple[InteractionRecord, ...]:
        return tuple(self._records)

    def reset(self) -> None:
        """Begin a new trial: clear observation-request budget and history."""
        self._verifier.reset()
        self._records.clear()

    # ── the cycle ────────────────────────────────────────────────────────────

    def review(self, plan: Any) -> tuple[GroundingFeedback | None, InteractionRecord | None]:
        """Verify the planner's proposal.  No side effects on the robot."""
        action = proposed_action_from_plan(plan, self._config)
        if action is None:
            return None, None
        evidence = extract_evidence(plan, self._config)
        feedback = self._verifier.review(action, evidence)
        record = InteractionRecord(
            round_id=getattr(getattr(plan, "guard", None), "round_id", None),
            phase=action.phase,
            planner_action={
                "action_type": action.metadata.get("action_type"),
                "target_world_xyz": [round(v, 6) for v in action.target_world_xyz],
                "tolerance_m": round(action.tolerance_m, 6),
                "planar_error_m": action.metadata.get("planar_error_m"),
            },
            grounding_verdict=feedback.verdict.value,
            uncertainty_m=feedback.uncertainty.position_sigma_m,
            uncertainty_calibrated=feedback.uncertainty.calibration.is_calibrated,
            tolerance_m=action.tolerance_m,
            reason=feedback.reason,
            checks=dict(feedback.checks),
            uncertainty_components=dict(feedback.uncertainty.components),
            observation_request=(
                {
                    "sensor_role": feedback.observation_request.sensor_role,
                    "reason": feedback.observation_request.reason,
                }
                if feedback.observation_request is not None
                else None
            ),
            assumptions=self._config.assumptions,
        )
        self._records.append(record)
        return feedback, record

    def observe_and_register(
        self,
        plan: Any,
        record: InteractionRecord,
        *,
        reference_descriptor: AppearanceDescriptor | None = None,
    ) -> RegistrationResult:
        """Attempt autonomous near-field registration from the current frame.

        No repositioning is performed.  The runtime only reaches this path once
        the near-field view is already within acquisition range, so the frame in
        hand is the observation.  If the target is not resolvable in it, the
        result is a fail-closed outcome and the caller must halt; there is no
        fallback to operator registration.
        """
        frames = getattr(plan, "frames", None)
        if not isinstance(frames, dict) or NEAR_VIEW not in frames:
            result = RegistrationResult(
                outcome=RegistrationOutcome.NO_CANDIDATE,
                reason="no near-field planning frame available",
            )
            record.registration = _registration_dict(result)
            return result

        frame = frame_from_planning_array(frames[NEAR_VIEW][-1], NEAR_VIEW)

        if self._observe_only:
            result = self._inspect_without_registering(frame)
            record.registration = _registration_dict(result)
            return result

        result = self._policy.acquire(
            frame,
            PICKUP_ROLE,
            NEAR_VIEW,
            reference_descriptor=reference_descriptor,
        )
        record.registration = _registration_dict(result)
        return result

    def _inspect_without_registering(self, frame: Frame) -> RegistrationResult:
        """Run detection and every pre-registration check, then stop short.

        Returns a result whose outcome describes what registration WOULD have
        done, with the candidate statistics in ``checks`` so a sweep over poses
        can record them.  Never calls register(), so the reference store is
        untouched and successive measurements are independent.
        """
        candidates = self._policy._detector.detect(frame)
        checks: dict[str, Any] = {
            "observe_only": True,
            "n_candidates": len(candidates),
        }
        if not candidates:
            return RegistrationResult(
                outcome=RegistrationOutcome.NO_CANDIDATE,
                reason="observe-only: detector returned no candidate",
                checks=checks,
            )

        best = candidates[0]
        checks["best_score"] = float(best.score)
        checks["best_area_px"] = int(best.area_px)
        checks["best_solidity"] = round(float(best.solidity), 4)
        checks["best_centroid_px"] = [round(float(v), 2) for v in best.centroid_px]

        if len(candidates) > 1:
            runner_up = candidates[1]
            ratio = runner_up.score / best.score if best.score > 0 else 1.0
            checks["runner_up_score"] = float(runner_up.score)
            checks["score_ratio"] = round(float(ratio), 4)
            if ratio >= self._policy._config.ambiguity_score_ratio:
                return RegistrationResult(
                    outcome=RegistrationOutcome.AMBIGUOUS_CANDIDATES,
                    reason=(
                        f"observe-only: {len(candidates)} candidates, "
                        f"runner-up ratio {ratio:.3f}"
                    ),
                    checks=checks,
                )

        passed, reason, sanity = self._policy._sanity_check(best, frame, None)
        checks.update(sanity)
        if not passed:
            return RegistrationResult(
                outcome=RegistrationOutcome.SANITY_CHECK_FAILED,
                reason=f"observe-only: {reason}",
                checks=checks,
            )

        # Would have registered.  Report that without doing it, and keep the
        # outcome distinguishable from a real registration so no caller can
        # mistake a measurement for a committed reference.
        return RegistrationResult(
            outcome=RegistrationOutcome.WITHHELD,
            reason="observe-only: all checks passed; registration withheld",
            centroid_px=np.asarray(best.centroid_px, dtype=np.float64).copy(),
            checks={**checks, "would_register": True},
        )

    # ── logging ──────────────────────────────────────────────────────────────

    def finalise(self, record: InteractionRecord, final_action: str) -> None:
        """Record what the runtime did, then persist."""
        record.final_action = final_action
        self.flush()

    def flush(self) -> None:
        if self._log_path is None:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {"interactions": [r.as_dict() for r in self._records]},
                handle,
                indent=2,
            )


def build_bridge_from_config(
    planner: Any,
    config_path: str | Path,
    *,
    log_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
    observe_only: bool = False,
) -> GroundingBridge:
    """Construct a bridge from a perception YAML config.

    The detector adapter is named in the config as a dotted path and imported by
    name, so swapping detectors is a config edit.  Nothing about the target's
    appearance, name, or task reaches this function.
    """
    import importlib

    import yaml

    path = Path(config_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a mapping")

    detector_cfg = dict(document.get("detector") or {})
    policy_cfg = dict(document.get("policy") or {})
    if not detector_cfg:
        raise ValueError(f"{path} is missing a 'detector' section")

    dotted = detector_cfg.pop("adapter_class", None)
    if not isinstance(dotted, str) or "." not in dotted:
        raise ValueError("detector.adapter_class must be a dotted import path")
    module_name, _, class_name = dotted.rpartition(".")
    detector_class = getattr(importlib.import_module(module_name), class_name)
    detector = detector_class(**detector_cfg)

    bridge_cfg_fields = set(GroundingBridgeConfig.__dataclass_fields__)
    bridge_kwargs = {
        key: value
        for key, value in (document.get("grounding") or {}).items()
        if key in bridge_cfg_fields
    }
    if overrides:
        bridge_kwargs.update(
            {k: v for k, v in overrides.items() if k in bridge_cfg_fields}
        )

    policy_fields = set(RegistrationPolicyConfig.__dataclass_fields__)
    unknown = set(policy_cfg) - policy_fields
    if unknown:
        raise ValueError(
            f"unknown policy keys in {path}: {', '.join(sorted(unknown))}"
        )

    return GroundingBridge(
        planner=planner,
        detector=detector,
        config=GroundingBridgeConfig(**bridge_kwargs),
        registration_policy_config=RegistrationPolicyConfig(**policy_cfg),
        log_path=log_path,
        observe_only=observe_only,
    )


def _registration_dict(result: RegistrationResult) -> dict[str, Any]:
    return {
        "outcome": result.outcome.value,
        "ok": result.ok,
        "target_id": result.target_id,
        "centroid_px": (
            None
            if result.centroid_px is None
            else [round(float(v), 2) for v in result.centroid_px]
        ),
        "reason": result.reason,
        "checks": dict(result.checks),
    }
